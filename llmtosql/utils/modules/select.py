import torch
from torch import nn
import structlog

logger = structlog.get_logger('__name__')


class WikiSQLSelect(nn.Module):
    def __init__(self, hidden_dim, attention_type, col_drop):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        self.col_drop = col_drop
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

    def forward(self, data, columns_tokenized):
        text_last_hs, columns_last_hs = data
        if self.attention_type == 'cross':
            attn_output, _ = self.cross_att(columns_last_hs, text_last_hs, text_last_hs)
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
