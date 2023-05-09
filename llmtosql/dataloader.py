import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding
import os
import re
import structlog
from tqdm import tqdm
from collections import defaultdict

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
                self.data[idx]['cond_3'] = [self._generate_mapping(text.split(),
                              [fr'(?i)\b{token.lower()}\b' for token in
                               self._clean_text(str(cond[2])).split()],
                               self._clean_text(str(cond[2])).split())
                               for cond in conds]
                self.data[idx]['cond_num'] = cond_num

                cond_check = [cond[2] for cond in conds]
                self.data[idx]['cond_check'] = cond_check
                cleaned_q = list(self._generate_cond3(text.split()))
                self.data[idx]['check'] = [self._digitize(' '.join(cleaned_q[cond_range[0]:cond_range[0] + cond_range[1]]))
                                           for cond_range in self.data[idx]['cond_3']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        table = self.data[item]['table_id']
        input = self.data[item]['question']
        sel = self.data[item]['sql']['sel']
        agg = self.data[item]['sql']['agg']
        cond_1 = [int(cond[0]) for cond in self.data[item]['sql']['conds']]
        cond_2 = [int(cond[1]) + 1 for cond in self.data[item]['sql']['conds']]
        cond_3 = self.data[item]['cond_3']
        if self.model:
            if self.model.max_conds != self.maxcondsLength:
                raise AttributeError(f'Model max condition out does not much max condition out in labels. '
                                     f'Found {self.maxcondsLength} max condition in labels.')
            cond_0 = self.data[item]['cond_num']
            if len(cond_1) != self.maxcondsLength:
                list_extension = [0] * (self.maxcondsLength - len(cond_1))
                cond_1.extend(list_extension)
            if len(cond_2) != self.maxcondsLength:
                list_extension = [0] * (self.maxcondsLength - len(cond_2))
                cond_2.extend(list_extension)
            if len(cond_3) != self.maxcondsLength:
                list_extension = [[0, 0]] * (self.maxcondsLength - len(cond_3))
                cond_3.extend(list_extension)
            columns = self.data[item]['columns']
            tokenized_inputs = self.data[item]['tokenized_inputs']

            check = self.data[item]['check']
            conds__3 = self.data[item]['cond_check']
            return {
                'table_id': str(table),
                'columns': columns,
                'input': (str(input), str(columns)),
                'tokenized_inputs': {
                    'question': BatchEncoding({k: v.squeeze() for k, v in tokenized_inputs[0].items()}),
                    'columns': BatchEncoding({k: v.squeeze() for k, v in tokenized_inputs[1].items()})
                },
                'labels': {
                    'sel': int(sel),
                    'agg': int(agg),
                    'conds': (cond_0, cond_1, cond_2, cond_3)
                },
                'CHECK': (conds__3, check)
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

    def _generate_mapping(self, question_token_list, pattern_list, gt):
        token_dict = defaultdict(list)
        for pattern, key in zip(pattern_list, gt):
            for idx, token in enumerate(question_token_list):
                if (len(key) == 1):
                    if key.lower() == self._clean_text(token):
                        token_dict[key].append(idx)
                else:
                    if (re.findall(pattern, self._clean_text(token))) or \
                            (key.lower() == self._clean_text(token)) or \
                            ((re.findall(r'^[-+]?(?:[0-9]+)*[0-9]+(?:\.[0-9]+)?$', key)) and
                             (re.findall(r'^[-+]?(?:[0-9]+)*[0-9]+(?:\.[0-9]+)?$', self._clean_text(token))) and
                             (float(self._clean_text(token)) == float(self._clean_text(key)))):
                        token_dict[key].append(idx)
        first_tokens = set(token_dict[gt[0]])
        end_tokens = set(token_dict[gt[-1]])
        index_list = None
        for end in end_tokens:
            for start in first_tokens:
                if (end - start + 1) == len(gt):
                    index_list = [start, end]
                else:
                    removed = sum([it in question_token_list for it in '?+$[](){}*#.%"`'])
                    if (end - start + 1) == len(gt) + removed:
                        index_list = [start, end]
        if index_list:
            return [index_list[0], index_list[-1] - index_list[0] + 1]
        else:
            return [0, 0]

    def _clean_text(self, text):
        text = text.lower()
        char_list = '?+$[](){}*#.%"`|'
        for char in char_list:
            text = text.replace(char, '')
        if re.findall(r'\b\'s', text):
            text = text.replace("'s", '')
        text = text.replace("'", '')
        text = re.sub(r'\b,\s', ' ', text)
        text = re.sub(r'\b;\s', ' ', text)
        if len(text) > 1:
            text = text.rstrip(";")
            text = text.rstrip(",")
        return text

    @classmethod
    def _generate_cond3(cls, lst):
        for text in lst:
            char_list = '"?><`'
            for char in char_list:
                if char == '>' and len(text) == 1:
                    continue
                if char == '"' and re.findall(r'^(?!$|.*\'[^\x22]+$)(?:([0-9]+)\')?(?:([0-9]+)\x22?)?$', text):
                    continue
                text = text.replace(char, '')
            yield text.lower()

    @classmethod
    def _digitize(cls, text):
        text = text.strip(",")
        text = text.rstrip(";")
        try:
            if ('jr' not in text.split()[-1].lower()) and (text.count('.') == 1):
                text = text.rstrip(".")
        except:
            pass
        if re.findall((r'^[-+]?(?:[0-9]+)*[0-9]+(?:\.[0-9]+)?\.$'), text):
            text = text.rstrip(".")
        text = text.strip("'")
        if text.endswith("'s"):
            text = re.sub(r"'s", '', text)
        if text.endswith("´s"):
            text = re.sub(r"´s", '', text)
        if len(text.split()) == 1:
            text = text.replace('#', '')
        if (('(' in text) and not (')' in text)) or ((')' in text) and not ('(' in text)):
            text = text.strip("(")
            text = text.strip(")")
        if (' ( ' in text) and not (' )' in text):
            text = text + ' )'
        if (' [ ' in text) and not (' ]' in text):
            text = text + ' ]'
        return text
