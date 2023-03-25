import torch
from transformers import AutoModel, AutoTokenizer
import structlog
from .utils.modules.base_model import WikiSQLBase
from .utils.modules.select import WikiSQLSelect
from .utils.modules.aggregation import WikiSQLSAgg
from .utils.modules.conditions import WikiSQLConditions

logger = structlog.get_logger('__name__')


class WikiSQLModel(WikiSQLBase):
    def __init__(self, base_model_type, N_lat=None, attention_type='cross', col_drop=False,
                 local_model_type=None, max_conds=10):
        super().__init__(base_model_type, N_lat=N_lat, attention_type=attention_type,
                         col_drop=col_drop, local_model_type=local_model_type)
        self.n_heads = 3
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_type)
        self.model = AutoModel.from_pretrained(self.base_model_type)
        self.criterion = torch.nn.CrossEntropyLoss()
        if not self.hidden_dim:
            self.hidden_dim = self.model.config.hidden_size
        self.seq_len = self.model.config.max_position_embeddings
        self.vocab_size = self.model.config.vocab_size
        self.sel_layer = WikiSQLSelect(self.hidden_dim, attention_type)
        self.agg_layer = WikiSQLSAgg(self.hidden_dim, 6, attention_type)
        self.cond_layer = WikiSQLConditions(self.tokenizer, self.hidden_dim, self.seq_len,
                                            self.vocab_size, attention_type, max_conds)

    def forward(self, data):
        text_tokenized, columns_tokenized = data
        with torch.no_grad():
            text_outputs = self.model(**text_tokenized)
            columns_outputs = self.model(**columns_tokenized)
        text_last_hs = text_outputs.last_hidden_state
        columns_last_hs = columns_outputs.last_hidden_state
        text_last_hs = text_last_hs[:, 1:, :]
        columns_last_hs = columns_last_hs[:, 1:, :]
        layer_input = (text_last_hs, columns_last_hs)
        sel_out = self.sel_layer(layer_input)
        if self.col_drop or self.attention_type == 'cross':
            sel_out = self.compose_outputs(columns_tokenized, sel_out)
        agg_out = self.agg_layer(layer_input)
        cond_out = self.cond_layer(layer_input)
        if self.col_drop or self.attention_type == 'cross':
            cond_num_out, cond_column_out, cond_op_out, cond_text_out = cond_out
            cond_column_out = self.compose_outputs(columns_tokenized, cond_column_out, multi=True)
            cond_out = (cond_num_out, cond_column_out, cond_op_out, cond_text_out)
        return sel_out, agg_out, cond_out

    def tokenize(self, data):
        text_imp, columns_imp = data
        if isinstance(text_imp, str) and isinstance(columns_imp, str):
            text_imp, columns_imp = [text_imp], [columns_imp]
        if self.col_drop:
            columns_imp = self.reduce_col_name(columns_imp)
        text_imp = [text + ' ' + columns for text, columns in zip(text_imp, columns_imp)]
        if self.base_model == 'gpt':
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            columns_imp = [self.tokenizer.eos_token + ' ' + col_txt for col_txt in columns_imp]
            text_imp = [self.tokenizer.eos_token + ' ' + col_txt for col_txt in text_imp]
        text_tokenized = self.tokenizer(text_imp, padding='max_length', return_tensors='pt')
        columns_tokenized = self.tokenizer(columns_imp, padding='max_length', return_tensors='pt')
        return text_tokenized, columns_tokenized

    def loss(self, outputs, targets):
        losses = [self.criterion(output, target) for output, target in zip(outputs, targets)]
        loss = torch.stack(losses, dim=0).sum(dim=0)
        return loss

    # loss(torch.transpose(logits, 1, 2), label['input_ids'])
