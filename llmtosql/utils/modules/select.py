import torch
from torch import nn
from .base_model import WikiSQLBase
import structlog

logger = structlog.get_logger('__name__')


class WikiSQLSelect(WikiSQLBase):
    def __init__(self):
        super().__init__()
        if not self.N_lat:
            self.hidden_dim = self.model.config.hidden_size
        if self.attention_type == 'cross':
            self.cross_att = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
            self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
            self.out = nn.Linear(self.hidden_dim, 1)
        elif self.attention_type == 'sqlnet':
            # FROM SQLNET
            self.sel_att = nn.Linear(self.hidden_dim, 1)
            self.sel_out_K = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.sel_out_col = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(self.hidden_dim, 1))
            self.softmax = nn.Softmax(dim=-1)
        else:
            logger.error(f'Was not able to load SELECT module -  {type(self.attention_type)}  not valid')
            raise TypeError(f'Was not able to load SELECT module -  {type(self.attention_type)}  not valid')

    def forward(self, data):
        text_tokenized, columns_tokenized = data
        with torch.no_grad():
            text_outputs = self.model(**text_tokenized)
            columns_outputs = self.model(**columns_tokenized)
        text_last_hs = text_outputs.last_hidden_state
        columns_last_hs = columns_outputs.last_hidden_state
        text_last_hs = text_last_hs[:, 1:, :]
        columns_last_hs = columns_last_hs[:, 1:, :]
        if self.attention_type == 'cross':
            attn_output, _ = self.cross_att(columns_last_hs, text_last_hs, text_last_hs)
            # cross_attention_norm = self.batch_norm(torch.transpose(attn_output, 1, 2))
            # cross_layer_out = torch.add(columns_last_hs, torch.transpose(cross_attention_norm, 1, 2))
            cross_attention_add = torch.add(columns_last_hs, attn_output)
            cross_attention_norm = self.batch_norm(torch.transpose(cross_attention_add, 1, 2))
            ret = self.out(torch.transpose(cross_attention_norm, 1, 2)).squeeze()
            return self.compose_outputs(columns_tokenized, ret)
        elif self.attention_type == 'sqlnet':
            # FROM SQLNET
            att_val = self.sel_att(text_last_hs).squeeze()
            att = self.softmax(att_val)
            K_sel = (text_last_hs * att.unsqueeze(2).expand_as(text_last_hs)).sum(1)
            K_sel_expand = K_sel.unsqueeze(1)
            sel_score = self.sel_out(self.sel_out_K(K_sel_expand) +
                                     self.sel_out_col(columns_last_hs)).squeeze()
            if self.col_drop:
                return sel_score
            else:
                return self.compose_outputs(columns_tokenized, sel_score)
