from torch.utils.data import Dataset
import numpy as np
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
                columns = ', '.join(self.collate_table_data[table])
                self.data[idx]['columns'] = columns
                self.data[idx]['tokenized_inputs'] = model.tokenize((text, columns))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        table = self.data[item]['table_id']
        input = self.data[item]['question']
        sel = self.data[item]['sql']['sel']
        agg = self.data[item]['sql']['agg']
        conds = [{'cond_1': int(cond[0]), 'cond_2': int(cond[1]), 'cond_3': str(cond[2])}
                 for cond in self.data[item]['sql']['conds']]
        if len(conds) != self.maxcondsLength:
            list_extension = [{'cond_1': np.NaN, 'cond_2': np.NaN, 'cond_3': ''}]*(self.maxcondsLength-len(conds))
            conds.extend(list_extension)
        if self.model:
            columns = self.data[item]['columns']
            tokenized_inputs = self.data[item]['tokenized_inputs']
            return {
                'table_id': str(table),
                'columns': columns,
                'input': (str(input), str(columns)),
                # 'tokenized_inputs': tokenized_inputs,
                'tokenized_inputs': {
                    'question': tokenized_inputs[0],
                    'columns': tokenized_inputs[1]
                },
                'labels': {
                    'sel': int(sel),
                    'agg': int(agg),
                    'conds': conds
                }
            }
        else:
            columns = ', '.join(self.collate_table_data[table])
            return {
                'table_id': str(table),
                'input': (str(input), str(columns)),
                'labels': {
                    'sel': int(sel),
                    'agg': int(agg),
                    'conds': conds
                }
            }

