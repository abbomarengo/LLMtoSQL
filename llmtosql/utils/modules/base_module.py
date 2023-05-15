import torch
from torch import nn
import structlog

logger = structlog.get_logger('__name__')


class WikiSQLBaseModule(nn.Module):
    def __init__(self, hidden_dim, dim_out, attention_type):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.attention_type = attention_type
        if self.attention_type == 'cross':
            self.cross_att = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
            self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
            self.out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.Tanh(), nn.Linear(self.hidden_dim, self.dim_out))
        elif self.attention_type == 'sqlnet':
            logger.error(f'SQLNET not implemented for the Condition Head')
            raise TypeError(f'SQLNET not implemented for the Condition Head')
        else:
            logger.error(f'Was not able to load SELECT module -  {type(self.attention_type)}  not valid')
            raise TypeError(f'Was not able to load SELECT module -  {type(self.attention_type)}  not valid')

    def forward(self, data):
        text_last_hs, columns_last_hs = data
        attn_output, _ = self.cross_att(columns_last_hs, text_last_hs, text_last_hs)
        cross_attention_add = torch.add(columns_last_hs, attn_output)
        cross_attention_norm = self.batch_norm(torch.transpose(cross_attention_add, 1, 2))
        return self.out(torch.transpose(cross_attention_norm, 1, 2))


class WikiSQLBaseModuleSA(nn.Module):
    def __init__(self, hidden_dim, dim_out):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.cross_att = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
        self.out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.Tanh(), nn.Linear(self.hidden_dim, self.dim_out))

    def forward(self, data):
        attn_output, _ = self.cross_att(data, data, data)
        self_attention_add = torch.add(data, attn_output)
        self_attention_norm = self.batch_norm(torch.transpose(self_attention_add, 1, 2))
        return self.out(torch.transpose(self_attention_norm, 1, 2))
