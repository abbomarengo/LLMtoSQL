import torch
from torch import nn
from torch.nn import Softmax
import structlog

from .base_module import WikiSQLBaseModule
from .aggregation import WikiSQLSAgg

logger = structlog.get_logger('__name__')


class WikiSQLConditions(nn.Module):
    def __init__(self, tokenizer, hidden_dim, cond_op_out, cond_text_out, attention_type, max_conds=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.cond_text_out = cond_text_out
        self.attention_type = attention_type
        self.max_conds = max_conds
        self.pred_function = Softmax(dim=-1)
        if self.attention_type == 'cross':
            # Step 1 - condition number
            self.cond_num_layer = WikiSQLSAgg(self.hidden_dim, max_conds + 1, attention_type)
            # Step 2 - condition columns
            self.cond_column = WikiSQLBaseModule(self.hidden_dim, self.hidden_dim, attention_type)
            self.ff2 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.Tanh(), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.column_out = nn.Linear(self.hidden_dim, 1)
            # Step 3 - condition operation
            self.cond_op = WikiSQLBaseModule(self.hidden_dim, self.hidden_dim, attention_type)
            self.ff3 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.Tanh(), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.op_out = nn.Linear(self.hidden_dim, cond_op_out)
            # Step 4 - Condition text
            self.cond_text_layer = WikiSQLBaseModule(self.hidden_dim, self.hidden_dim, attention_type)
            self.ff4 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.Tanh(), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.text_out = nn.Linear(self.hidden_dim, self.cond_text_out)
        elif self.attention_type == 'sqlnet':
            logger.error(f'SQLNET not implemented for the Condition Head')
            raise TypeError(f'SQLNET not implemented for the Condition Head')
        else:
            logger.error(f'Was not able to load SELECT module -  {type(self.attention_type)}  not valid')
            raise TypeError(f'Was not able to load SELECT module -  {type(self.attention_type)}  not valid')

    def forward(self, data):
        # Step 1 - condition number
        cond_num_out = self.cond_num_layer(data)
        num_conditions = torch.argmax(self.pred_function(cond_num_out), dim=-1)
        # Prep for next steps
        dim_0 = data[0].shape[0]
        dim_1 = data[0].shape[1]
        dim_2 = self.max_conds
        if dim_2 == 0:
            return cond_num_out
        # Step 2 - condition columns
        cond_column = self.cond_column(data)
        # cross_transpose_c = torch.transpose(cond_column, 1, 2)
        concat_c = torch.cat([cond_column] * dim_2, dim=-1)
        reshaped_in_c = concat_c.view(dim_0, dim_1, dim_2, self.hidden_dim)
        feed_forward_c = self.ff2(reshaped_in_c)
        last_layer_c = self.column_out(feed_forward_c)
        cond_column_out = last_layer_c.squeeze(-1)
        # Step 3 - condition operation
        cond_op = self.cond_op(data)
        # cross_transpose_o = torch.transpose(cond_op, 1, 2)
        concat_o = torch.cat([cond_op] * dim_2, dim=-1)
        reshaped_in_o = concat_o.view(dim_0, dim_1, dim_2, self.hidden_dim)
        feed_forward_o = self.ff3(reshaped_in_o)
        cond_op_out = self.op_out(feed_forward_o.mean(dim=1))
        # Step 4 - Condition text
        cond_text = self.cond_text_layer(data)
        # cross_transpose_t = torch.transpose(cond_text, 1, 2)
        concat_t = torch.cat([cond_text]*dim_2, dim=-1)
        reshaped_in_t = concat_t.view(dim_0, dim_1, dim_2, self.hidden_dim)
        feed_forward_t = self.ff4(reshaped_in_t)
        cond_text_out = self.text_out(feed_forward_t)
        cond_text_out = torch.transpose(cond_text_out, 1, 3) # NEW
        return cond_num_out, cond_column_out, cond_op_out, cond_text_out
