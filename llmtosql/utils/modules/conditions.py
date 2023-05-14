import torch
from torch import nn
from torch.nn import Softmax
import structlog

from .base_module import WikiSQLBaseModule, WikiSQLBaseModuleSA
from .aggregation import WikiSQLSAgg

logger = structlog.get_logger('__name__')


class WikiSQLConditions(nn.Module):
    def __init__(self, hidden_dim, cond_op_out, cond_text_out, attention_type, max_conds=4):
        super().__init__()
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
            self.cond_column_sa = WikiSQLBaseModuleSA(self.hidden_dim, self.hidden_dim)
            self.ff2 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.Tanh(), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.column_out = nn.Linear(self.hidden_dim, 1)
            # Step 3 - condition operation
            self.cond_op = WikiSQLBaseModule(self.hidden_dim, self.hidden_dim, attention_type)
            self.cond_op_sa = WikiSQLBaseModuleSA(self.hidden_dim, self.hidden_dim)
            self.ff3 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.Tanh(), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.op_out = nn.Linear(self.hidden_dim, cond_op_out)
            # Step 4 - Condition text
            self.cond_text_layer = WikiSQLBaseModule(self.hidden_dim, self.hidden_dim, attention_type)
            self.cond_text_layer_sa = WikiSQLBaseModuleSA(self.hidden_dim, self.hidden_dim)
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
        cond_column_out = self.cond_column(data)
        cond_column_out = torch.cat([cond_column_out] * dim_2, dim=1)
        cond_column_out = self.cond_column_sa(cond_column_out)
        cond_column_out = cond_column_out.view(dim_0, dim_1, dim_2, self.hidden_dim)
        cond_column_out = self.ff2(cond_column_out)
        cond_column_out = self.column_out(cond_column_out)
        cond_column_out = cond_column_out.squeeze(-1)
        # Step 3 - condition operation
        cond_op_out = self.cond_op(data)
        cond_op_out = torch.cat([cond_op_out] * dim_2, dim=1)
        cond_op_out = self.cond_op_sa(cond_op_out)
        cond_op_out = cond_op_out.view(dim_0, dim_1, dim_2, self.hidden_dim)
        cond_op_out = self.ff3(cond_op_out)
        cond_op_out = self.op_out(cond_op_out.mean(dim=1))
        # Step 4 - Condition text
        cond_text_out = self.cond_text_layer(data)
        cond_text_out = torch.cat([cond_text_out]*dim_2, dim=1)
        cond_text_out = self.cond_text_layer_sa(cond_text_out)
        cond_text_out = cond_text_out.view(dim_0, dim_1, dim_2, self.hidden_dim)
        cond_text_out = self.ff4(cond_text_out)
        cond_text_out = self.text_out(cond_text_out)
        cond_text_out = torch.transpose(cond_text_out, 1, 3)
        return cond_num_out, cond_column_out, cond_op_out, cond_text_out
