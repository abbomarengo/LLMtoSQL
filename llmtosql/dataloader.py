import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding
import os
import structlog
from tqdm import tqdm

logger = structlog.get_logger('__name__')


class WikiSQLDataset(Dataset):
    def __init__(self, type='dev', data_folder_path='../WikiSQL/data', model=None):
        self.model = model
        data_path = os.path.join(data_folder_path, type+'.jsonl')
        table_path = os.path.join(data_folder_path, type+'.tables.jsonl')
        with open(data_path) as f:
            self.data = [eval(line) for line in f]
            self.maxcondsLength = max(len(x['sql']['conds']) for x in self.data)
        with open(table_path) as f:
            table_data = [eval(line) for line in f]
        self.collate_table_data = {row['id']: row['header'] for row in table_data}
        if self.model:
            logger.info('Tokenizing dataset.')
            for idx, row in enumerate(tqdm(self.data)):
                table = row['table_id']
                text = row['question']
                conds = row['sql']['conds']
                cond_num = len(conds)
                columns = ', '.join(self.collate_table_data[table])
                self.data[idx]['columns'] = columns
                self.data[idx]['tokenized_inputs'] = model.tokenize((text, columns))
                self.data[idx]['tokenized_cond_3'] = [model.tokenize((str(cond[2]), str(cond[2]))) for cond in conds]
                self.data[idx]['cond_num'] = cond_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        table = self.data[item]['table_id']
        input = self.data[item]['question']
        sel = self.data[item]['sql']['sel']
        agg = self.data[item]['sql']['agg']
        cond_1 = [int(cond[0]) for cond in self.data[item]['sql']['conds']]
        cond_2 = [int(cond[1]) for cond in self.data[item]['sql']['conds']]
        if self.model:
            if len(cond_1) != self.maxcondsLength:
                list_extension = [self.model.seq_len-1] * (self.maxcondsLength - len(cond_1))
                cond_1.extend(list_extension)
            if len(cond_2) != self.maxcondsLength:
                list_extension = [self.model.cond_op_out-1] * (self.maxcondsLength - len(cond_2))
                cond_2.extend(list_extension)
            if self.model.max_conds != self.maxcondsLength:
                raise AttributeError(f'Model max condition out does not much max condition out un labels. '
                                     f'Found {self.maxcondsLength} mal condition in labels.')
            cond_0 = self.data[item]['cond_num']
            seq_len = self.model.seq_len
            empty_tensor = torch.zeros(seq_len, dtype=torch.int8)
            columns = self.data[item]['columns']
            tokenized_inputs = self.data[item]['tokenized_inputs']
            tokenized_cond_3 = self.data[item]['tokenized_cond_3']
            try:
                cond_3 = [BatchEncoding({k: v.squeeze() for k,v in cond[0].items()}) for cond in tokenized_cond_3]
            except:
                cond_3 = [{'input_ids': empty_tensor, 'token_type_ids': empty_tensor, 'attention_mask': empty_tensor}]
            if len(cond_3) != self.maxcondsLength:
                list_extension = [{'input_ids': empty_tensor,
                                   'token_type_ids': empty_tensor,
                                   'attention_mask': empty_tensor}]*(self.maxcondsLength-len(cond_3))
                cond_3.extend(list_extension)
            return {
                'table_id': str(table),
                'columns': columns,
                'input': (str(input), str(columns)),
                'tokenized_inputs': {
                    'question': BatchEncoding({k: v.squeeze() for k,v in tokenized_inputs[0].items()}),
                    'columns': BatchEncoding({k: v.squeeze() for k,v in tokenized_inputs[1].items()})
                },
                'labels': {
                    'sel': int(sel),
                    'agg': int(agg),
                    'conds': (cond_0, cond_1, cond_2, cond_3)
                }
            }
        else:
            if len(cond_1) != self.maxcondsLength:
                list_extension = [-100] * (self.maxcondsLength - len(cond_1))
                cond_1.extend(list_extension)
            if len(cond_2) != self.maxcondsLength:
                list_extension = [-100] * (self.maxcondsLength - len(cond_2))
                cond_2.extend(list_extension)
            cond_3 = [str(cond[2]) for cond in self.data[item]['sql']['conds']]
            if len(cond_3) != self.maxcondsLength:
                list_extension = ['']*(self.maxcondsLength-len(cond_3))
            cond_3.extend(list_extension)
            columns = ', '.join(self.collate_table_data[table])
            return {
                'table_id': str(table),
                'input': (str(input), str(columns)),
                'labels': {
                    'sel': int(sel),
                    'agg': int(agg),
                    'conds': (cond_1, cond_2, cond_3)
                }
            }
