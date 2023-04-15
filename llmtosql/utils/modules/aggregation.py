import torch
from torch import nn
import structlog

from .base_module import WikiSQLBaseModule

logger = structlog.get_logger('__name__')


class WikiSQLSAgg(nn.Module):
    def __init__(self, hidden_dim, dim_out, attention_type):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.attention_type = attention_type
        if self.attention_type == 'cross':
            self.agg_layer = WikiSQLBaseModule(self.hidden_dim, self.hidden_dim, attention_type)
            self.out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.Tanh(), nn.Linear(self.hidden_dim, self.dim_out))
        elif self.attention_type == 'sqlnet':
            # FROM SQLNET
            self.agg_att = nn.Linear(self.hidden_dim, 1)
            self.agg_out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                         nn.Tanh(), nn.Linear(self.hidden_dim, self.dim_out))
            self.softmax = nn.Softmax(dim=-1)
        else:
            logger.error(f'Was not able to load AGGREGATION module -  {type(self.attention_type)}  not valid')
            raise TypeError(f'Was not able to load AGGREGATION module -  {type(self.attention_type)}  not valid')

    def forward(self, data):
        if self.attention_type == 'cross':
            agg_out = self.agg_layer(data)
            ret = self.out(agg_out.mean(dim=1))
            return ret
        elif self.attention_type == 'sqlnet':
            text_last_hs, columns_last_hs = data
            # FROM SQLNET
            att_val = self.agg_att(text_last_hs).squeeze()
            att = self.softmax(att_val)
            K_agg = (text_last_hs * att.unsqueeze(2).expand_as(text_last_hs)).sum(1)
            agg_score = self.agg_out(K_agg)
            return agg_score
