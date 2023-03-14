import torch
from torch import nn
import structlog

logger = structlog.get_logger('__name__')


class WikiSQLSAgg(nn.Module):
    def __init__(self, hidden_dim, attention_type):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        if self.attention_type == 'cross':
            self.cross_att = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
            self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
            self.out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                         nn.Tanh(), nn.Linear(self.hidden_dim, 6))
        elif self.attention_type == 'sqlnet':
            # FROM SQLNET
            self.agg_att = nn.Linear(self.hidden_dim, 1)
            self.agg_out = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                         nn.Tanh(), nn.Linear(self.hidden_dim, 6))
            self.softmax = nn.Softmax(dim=-1)
        else:
            logger.error(f'Was not able to load AGGREGATION module -  {type(self.attention_type)}  not valid')
            raise TypeError(f'Was not able to load AGGREGATION module -  {type(self.attention_type)}  not valid')

    def forward(self, data):
        text_last_hs, columns_last_hs = data
        if self.attention_type == 'cross':
            attn_output, _ = self.cross_att(columns_last_hs, text_last_hs, text_last_hs)
            cross_attention_add = torch.add(columns_last_hs, attn_output)
            cross_attention_norm = self.batch_norm(torch.transpose(cross_attention_add, 1, 2))
            cross_attention_norm_transpose = torch.transpose(cross_attention_norm, 1, 2)
            ret = self.out(cross_attention_norm_transpose.mean(dim=1))
            return ret
        elif self.attention_type == 'sqlnet':
            # FROM SQLNET
            att_val = self.agg_att(text_last_hs).squeeze()
            att = self.softmax(att_val)
            K_agg = (text_last_hs * att.unsqueeze(2).expand_as(text_last_hs)).sum(1)
            agg_score = self.agg_out(K_agg)
            return agg_score
