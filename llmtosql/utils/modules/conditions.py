import torch
from torch import nn
from torch.nn import Softmax
from torch.nn.parameter import Parameter
import structlog

from .select import WikiSQLSelect
from .aggregation import WikiSQLSAgg

logger = structlog.get_logger('__name__')


class WikiSQLConditions(nn.Module):
    def __init__(self, tokenizer, hidden_dim, seq_len, vocab_size, attention_type):
        super().__init__()
        # self.parameter = Parameter(torch.empty((1, 1)))
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.attention_type = attention_type
        self.pred_function = Softmax(dim=-1)
        if self.attention_type == 'cross':
            # Step 1 - condition number
            self.cond_num_layer = WikiSQLSAgg(self.hidden_dim, 10, attention_type)
            # Step 4 - Condition text
            # self.q = nn.Linear(1, self.hidden_dim)
            self.cross_att = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True)
            self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
            self.ff = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                         nn.Tanh(), nn.Linear(self.hidden_dim, self.hidden_dim))
            self.out = nn.Linear(self.hidden_dim, self.vocab_size)
        elif self.attention_type == 'sqlnet':
            logger.error(f'SQLNET not implemented for the Condition Head')
            raise TypeError(f'SQLNET not implemented for the Condition Head')
        else:
            logger.error(f'Was not able to load SELECT module -  {type(self.attention_type)}  not valid')
            raise TypeError(f'Was not able to load SELECT module -  {type(self.attention_type)}  not valid')

    def forward(self, data):
        text_last_hs, columns_last_hs = data
        # Step 1 - condition number
        cond_num_out = self.cond_num_layer(data)
        num_conditions = torch.argmax(self.pred_function(cond_num_out), dim=-1)

        # Step 4 - Condition text
        dim_0 = text_last_hs.shape[0]
        dim_1 = self.seq_len
        dim_2 = num_conditions
        # device_n = self.parameter.get_device()
        # if self.parameter.is_cuda:
        #     device = 'cuda:' + str(device_n) if device_n != -1 else 'cpu'
        # else:
        #     device = 'mps:' + str(device_n) if device_n != -1 else 'cpu'
        # mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        # dec_inp = torch.ones([dim_0, dim_1], dtype=torch.float32, device=device)*mask_id
        # query = self.q(dec_inp.unsqueeze(dim=-1))
        attn_output, _ = self.cross_att(columns_last_hs, text_last_hs, text_last_hs)
        cross_attention_add = torch.add(columns_last_hs, attn_output)
        cross_attention_norm = self.batch_norm(torch.transpose(cross_attention_add, 1, 2))
        cross_transpose = torch.transpose(cross_attention_norm, 1, 2)
        concat = torch.cat([cross_transpose]*dim_2, dim=-1)
        reshaped_in = concat.view(dim_0, dim_1, dim_2, self.hidden_dim)
        feed_forward = self.ff(reshaped_in)
        cond_text_out = self.out(feed_forward)

        return cond_text_out

