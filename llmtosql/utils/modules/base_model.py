import torch
from torch import nn
from torch.nn.parameter import Parameter
import structlog
import os

logger = structlog.get_logger('__name__')


class WikiSQLBase(nn.Module):
    def __init__(self, base_model_type, N_lat=None, attention_type='cross', col_drop=False,
                 local_model_type=None, heads=(True, True, True)):
        super().__init__()
        self.parameter = Parameter(torch.empty((1, 1)))
        self.base_model_type = base_model_type
        self.hidden_dim = N_lat
        self.attention_type = attention_type
        self.col_drop = col_drop
        self.sel_head, self.agg_head, self.cond_head = heads
        logger.info(f'Using {attention_type} attention mechanism')
        if not base_model_type:
            logger.error(f'{type(base_model_type)}  not valid')
            raise TypeError(f'{type(base_model_type)}  not valid')
        if 'gpt' in base_model_type:
            self.base_model = 'gpt'
        elif 'bert' in base_model_type:
            self.base_model = 'bert'
        else:
            if os.path.isdir(base_model_type):
                if 'gpt' in local_model_type:
                    self.base_model = 'gpt'
                elif 'bert' in local_model_type:
                    self.base_model = 'bert'
                else:
                    logger.error('A local directory was passed. Need to identify the model - pass local_model_type')
                    raise TypeError('A local directory was passed. Need to identify the model - pass local_model_type')
            else:
                logger.error(f'Model type not valid - {base_model_type}')
                raise TypeError(f'Model type not valid - {base_model_type}')
        if self.attention_type == 'cross':
            pass
        elif self.attention_type == 'sqlnet':
            # FROM SQLNET
            pass
        else:
            logger.error(f'{type(self.attention_type)}  not valid')
            raise TypeError(f'{type(self.attention_type)}  not valid')

    def forward(self, data):
        pass

    def unpack(self, data, device):
        sel_labels, agg_labels, conds_labels = None, None, None
        inputs = (data['tokenized_inputs']['question'].to(device),
                  data['tokenized_inputs']['columns'].to(device))
        if self.sel_head:
            sel_labels = data['labels']['sel'].to(device)
        if self.agg_head:
            agg_labels = data['labels']['agg'].to(device)
        if self.cond_head:
            conds_labels = []
            for cond in data['labels']['conds']:
                if isinstance(cond, torch.Tensor):
                    conds_labels.append(cond.to(device))
                else:
                    inner_cond = []
                    for tensor in cond:
                        inner_cond.append(tensor.to(device))
                    conds_labels.append(inner_cond)
        return inputs, (sel_labels, agg_labels, conds_labels)

    def compose_outputs(self, col_vector, final_vector, multi=False):
        comma = self.tokenizer.convert_tokens_to_ids(',')
        if (len(final_vector.shape) == 1) or (multi and (len(final_vector.shape) == 2)):
            final_vector = final_vector.unsqueeze(0)
        if len(col_vector['input_ids'].shape) == 1:
            col_indices = col_vector['input_ids'].unsqueeze(0)
        else:
            col_indices = col_vector['input_ids']
        comma_idx = (col_indices == comma).nonzero(as_tuple=True)
        comma_idx = (
            comma_idx[0],
            comma_idx[1] - 1
        )
        dim_1 = torch.max(torch.unique(comma_idx[0], return_counts=True)[1]).item() + 1
        dim_0 = col_indices.shape[0]
        if multi:
            dim_2 = final_vector.shape[2]
        device_n = self.parameter.get_device()
        if self.parameter.is_cuda:
            device = 'cuda:' + str(device_n) if device_n != -1 else 'cpu'
        else:
            device = 'mps:' + str(device_n) if device_n != -1 else 'cpu'
        if not multi:
            out = torch.zeros([dim_0, dim_1], dtype=torch.float32, device=device)
        else:
            out = torch.zeros([dim_0, dim_1, dim_2], dtype=torch.float32, device=device)
        slice_start = 0
        current_idx = 0
        y = 0
        for idx in range(comma_idx[0].shape[0]):
            slice_end = comma_idx[1][idx].item()
            x = comma_idx[0][idx].item()
            if x != current_idx:
                if not multi:
                    out[x-1][y] = final_vector[x-1][slice_start:].mean()
                else:
                    for z in range(dim_2):
                        out[x - 1][y][z] = final_vector[x - 1, slice_start:, z].mean()
                y = 0
                slice_start = 0
                current_idx = x
            if not multi:
                out[x][y] = final_vector[x][slice_start:slice_end].mean()
            else:
                for z in range(dim_2):
                    out[x][y][z] = final_vector[x, slice_start:slice_end, z].mean()
            slice_start = slice_end + 1
            y += 1
        if not multi:
            out[x][y] = final_vector[x][slice_start:].mean()
        else:
            for z in range(dim_2):
                out[x][y][z] = final_vector[x, slice_start:, z].mean()
        return out

    def reduce_col_name(self, columns):
        if isinstance(columns, list):
            new_columns = []
            for cols in columns:
                col_text = cols.split(',')
                new_col = ''
                for column in col_text:
                    new_col += column.split()[0] + ' '
                new_columns.append(new_col.strip())
        else:
            new_columns = ''
            for column in columns.split(','):
                new_columns += column.split()[0] + ' '
            new_columns = new_columns.strip()
        return new_columns
