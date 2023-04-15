import torch
from torch import nn
import structlog

from .base_module import WikiSQLBaseModule

logger = structlog.get_logger('__name__')


class WikiSQLSelect(nn.Module):
    def __init__(self, hidden_dim, attention_type):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        if self.attention_type == 'cross':
            self.select_layer = WikiSQLBaseModule(self.hidden_dim, self.hidden_dim, attention_type)
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
        if self.attention_type == 'cross':
            sel_out = self.select_layer(data)
            ret = self.out(sel_out).squeeze()
            return ret
        elif self.attention_type == 'sqlnet':
            text_last_hs, columns_last_hs = data
            # FROM SQLNET
            att_val = self.sel_att(text_last_hs).squeeze()
            att = self.softmax(att_val)
            K_sel = (text_last_hs * att.unsqueeze(2).expand_as(text_last_hs)).sum(1)
            K_sel_expand = K_sel.unsqueeze(1)
            sel_score = self.sel_out(self.sel_out_K(K_sel_expand) +
                                     self.sel_out_col(columns_last_hs)).squeeze()
            return sel_score
