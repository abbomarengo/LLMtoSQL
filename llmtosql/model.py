from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
import structlog

logger = structlog.get_logger('__name__')


class WikiSQLModel(nn.Module):
    def __init__(self, base_model_type, N_lat=None, attention_type='cross'):
        super().__init__()
        self.attention_type = attention_type
        logger.info(f'Using {attention_type} attention mechanism')
        if not base_model_type:
            logger.error(f'{type(base_model_type)}  not valid')
            raise TypeError(f'{type(base_model_type)}  not valid')
        if 'gpt' in base_model_type:
            self.base_model = 'gpt'
        elif 'bert' in base_model_type:
            self.base_model = 'bert'
        else:
            logger.error(f'Model type not valid - {base_model_type}')
            raise TypeError(f'Model type not valid - {base_model_type}')
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_type)
        self.model = AutoModel.from_pretrained(base_model_type)
        self.seq_length = self.model.config.max_position_embeddings
        if not N_lat:
            self.hidden_dim = self.model.config.hidden_size
        if self.attention_type == 'cross':
            self.cross_att = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
            self.batch_norm = nn.BatchNorm1d(self.seq_length)
            self.out = nn.Linear(self.hidden_dim, 1)
        elif self.attention_type == 'sqlnet':
            # FROM SQLNET
            self.sel_att = nn.Linear(self.hidden_dim, 1)
            self.sel_out_K = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.sel_out_col = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(self.hidden_dim, 1))
            self.softmax = nn.Softmax(dim=-1)
        else:
            logger.error(f'{type(attention_type)}  not valid')
            raise TypeError(f'{type(attention_type)}  not valid')

    def tokenize(self, data):
        text_imp, columns_imp = data
        if (type(text_imp) != list) and (type(columns_imp) != list):
            text_imp, columns_imp = list(text_imp), list(columns_imp)
        if self.base_model == 'gpt':
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            columns_imp = [self.tokenizer.eos_token + ' ' + col_txt for col_txt in columns_imp]
            text_imp = [self.tokenizer.eos_token + ' ' + col_txt for col_txt in text_imp]
        text_tokenized = self.tokenizer(text_imp, padding='max_length', return_tensors='pt')
        columns_tokenized = self.tokenizer(columns_imp, padding='max_length', return_tensors='pt')
        return (text_tokenized, columns_tokenized)

    def forward(self, data):
        text_tokenized, columns_tokenized = data
        with torch.no_grad():
            text_outputs = self.model(**text_tokenized)
            columns_outputs = self.model(**columns_tokenized)
        text_last_hs = text_outputs.last_hidden_state
        columns_last_hs = columns_outputs.last_hidden_state
        if self.attention_type == 'cross':
            attn_output, _ = self.cross_att(columns_last_hs, text_last_hs, text_last_hs)
            cross_attention_norm = self.batch_norm(attn_output)
            cross_layer_out = torch.add(columns_last_hs, cross_attention_norm)
            ret = self.out(cross_layer_out).squeeze()
            return self.compose_outputs(columns_tokenized, ret)
        elif self.attention_type == 'sqlnet':
            # FROM SQLNET
            att_val = self.sel_att(text_last_hs).squeeze()
            att = self.softmax(att_val)
            K_sel = (text_last_hs * att.unsqueeze(2).expand_as(text_last_hs)).sum(1)
            K_sel_expand = K_sel.unsqueeze(1)
            sel_score = self.sel_out(self.sel_out_K(K_sel_expand) +
                                     self.sel_out_col(columns_last_hs)).squeeze()
            return self.compose_outputs(columns_tokenized, sel_score)

    def compose_outputs(self, col_vector, final_vector):
        comma = self.tokenizer.convert_tokens_to_ids(',')
        comma_idx = (col_vector['input_ids'] == comma).nonzero(as_tuple=True)
        dim_1 = torch.max(torch.unique(comma_idx[0], return_counts=True)[1]).item() + 1
        dim_0 = len(col_vector['input_ids'])
        out = torch.zeros([dim_0, dim_1], dtype=torch.float32)
        slice_start = 0
        current_idx = 0
        y = 0
        for idx in range(comma_idx[0].shape[0]):
            slice_end = comma_idx[1][idx].item()
            x = comma_idx[0][idx].item()
            if x != current_idx:
                out[x-1][y] = final_vector[x-1][slice_start:].sum()
                y = 0
                slice_start = 0
                current_idx = x
            out[x][y] = final_vector[x][slice_start:slice_end].sum()
            slice_start = slice_end + 1
            y += 1
        out[x][y] = final_vector[x][slice_start:].sum()
        return out
