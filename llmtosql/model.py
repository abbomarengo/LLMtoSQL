from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.nn.parameter import Parameter
import structlog
from .utils.modules.select import WikiSQLSelect
from .utils.modules.aggregation import WikiSQLSAgg

logger = structlog.get_logger('__name__')


class WikiSQLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_type)
        self.model = AutoModel.from_pretrained(self.base_model_type)
        self.seq_length = self.model.config.max_position_embeddings
        self.sel_layer = WikiSQLSelect()
        self.agg_layer = WikiSQLSAgg()

    def forward(self, data):
        sel_out = self.sel_layer(data)
        agg_out = self.agg_layer(data)
        return sel_out, agg_out

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
