import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer
import structlog
from .utils.modules.base_model import WikiSQLBase
from .utils.modules.select import WikiSQLSelect
from .utils.modules.aggregation import WikiSQLSAgg
from .utils.modules.conditions import WikiSQLConditions

logger = structlog.get_logger('__name__')


class WikiSQLModel(WikiSQLBase):
    def __init__(self, base_model_type, N_lat=None, attention_type='cross', col_drop=False,
                 local_model_type=None, op_out=6, text_out=2, max_conds=4, heads=(True, True, True)):
        super().__init__(base_model_type, N_lat=N_lat, attention_type=attention_type,
                         col_drop=col_drop, local_model_type=local_model_type, heads=heads)
        self.n_heads = sum(heads)
        self.head_names = [head for head, check in zip(['SELECT', 'AGG', 'CONDS'], heads) if check == True]
        logger.info(f'{self.n_heads} heads model -- {self.head_names}')
        self.cond_op_out = op_out + 1
        self.max_conds = max_conds
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_type)
        self.model = AutoModel.from_pretrained(self.base_model_type)
        self.criterion = torch.nn.CrossEntropyLoss()
        if not self.hidden_dim:
            self.hidden_dim = self.model.config.hidden_size
        self.text_out = text_out
        self.sel_layer = WikiSQLSelect(self.hidden_dim, attention_type)
        self.agg_layer = WikiSQLSAgg(self.hidden_dim, op_out, attention_type)
        self.cond_layer = WikiSQLConditions(self.hidden_dim, self.cond_op_out,
                                            self.text_out, attention_type, max_conds=self.max_conds)
        self.soft1 = torch.nn.Softmax(dim=-1)
        self.soft2 = torch.nn.Softmax(dim=1)
        self.logsoft1 = torch.nn.LogSoftmax(dim=-1)
        self.logsoft2 = torch.nn.LogSoftmax(dim=1)

    def forward(self, data):
        sel_out, agg_out, cond_out = None, None, None
        text_tokenized, columns_tokenized = data
        with torch.no_grad():
            text_outputs = self.model(**text_tokenized)
            columns_outputs = self.model(**columns_tokenized)
        text_last_hs = text_outputs.last_hidden_state
        columns_last_hs = columns_outputs.last_hidden_state
        text_last_hs = text_last_hs[:, 1:, :]
        columns_last_hs = columns_last_hs[:, 1:, :]
        layer_input = (text_last_hs, columns_last_hs)
        if self.sel_head:
            sel_out = self.sel_layer(layer_input)
            if self.col_drop or self.attention_type == 'cross':
                sel_out = self.compose_outputs(columns_tokenized, sel_out)
        if self.agg_head:
            agg_out = self.agg_layer(layer_input)
        if self.cond_head:
            cond_out = self.cond_layer(layer_input)
            if self.col_drop or self.attention_type == 'cross':
                if len(cond_out) == 1:
                    cond_out = None
                else:
                    cond_num_out, cond_column_out, cond_op_out, cond_text_out = cond_out
                    if len(cond_column_out.shape) == 2:
                        cond_column_out = self.compose_outputs(columns_tokenized, cond_column_out, multi=False)
                    else:
                        cond_column_out = self.compose_outputs(columns_tokenized, cond_column_out, multi=True)
                    cond_out = (cond_num_out, cond_column_out, cond_op_out, cond_text_out)
        return sel_out, agg_out, cond_out

    def tokenize(self, data):
        text_imp, columns_imp = data
        if isinstance(text_imp, str) and isinstance(columns_imp, str):
            text_imp, columns_imp = [text_imp], [columns_imp]
        if self.col_drop:
            columns_imp = self.reduce_col_name(columns_imp)
        text_imp = [text + ' ' + columns for text, columns in zip(text_imp, columns_imp)]
        if self.base_model == 'gpt':
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            columns_imp = [self.tokenizer.eos_token + ' ' + col_txt for col_txt in columns_imp]
            text_imp = [self.tokenizer.eos_token + ' ' + col_txt for col_txt in text_imp]
        text_tokenized = self.tokenizer(text_imp, padding='max_length', return_tensors='pt')
        columns_tokenized = self.tokenizer(columns_imp, padding='max_length', return_tensors='pt')
        return text_tokenized, columns_tokenized

    def predict(self, logits, function_type='softmax'):
        if (None in logits):
            logits = tuple([output for output in logits if output is not None])
        ret = []
        for output in logits:
            if isinstance(output, torch.Tensor):
                if function_type == 'softmax':
                    ret.append(torch.argmax(self.soft1(output), dim=-1))
                else:
                    ret.append(torch.argmax(self.logsoft1(output), dim=-1))
            elif isinstance(output, tuple):
                cond_ret = []
                for idx, cond_out in enumerate(output):
                    if idx != 1:
                        if function_type == 'softmax':
                            cond_ret.append(torch.argmax(self.soft1(cond_out), dim=-1))
                        else:
                            cond_ret.append(torch.argmax(self.logsoft1(cond_out), dim=-1))
                    else:
                        if function_type == 'softmax':
                            cond_ret.append(torch.argmax(self.soft2(cond_out), dim=1))
                        else:
                            cond_ret.append(torch.argmax(self.logsoft2(cond_out), dim=1))
                ret.append(tuple(cond_ret))
            else:
                pass
        return tuple(ret)

    def loss(self, outputs, targets):
        if (None in outputs) and (None in targets):
            outputs = tuple([output for output in outputs if output is not None])
            targets = tuple([target for target in targets if target is not None])
        loss = .0
        for output, target in zip(outputs, targets):
            if isinstance(output, torch.Tensor):
                loss += self.criterion(output, target)
            else:
                for idx, (cond_out, cond_lab) in enumerate(zip(output, target)):
                    if idx == 0:
                        loss += self.criterion(cond_out, cond_lab)
                    elif idx == 1:
                        loss += self.criterion(cond_out, torch.transpose(torch.stack(cond_lab), 0, 1))
                    elif idx == 2:
                        loss += self.criterion(torch.transpose(cond_out, 1, 2),
                                               torch.transpose(torch.stack(cond_lab), 0, 1))
                    else:
                        for idx, cond_lab_text in enumerate(cond_lab):
                            loss += self.criterion(torch.transpose(cond_out[:, :, idx, :], 1, 2),
                                                   torch.transpose(torch.stack(cond_lab_text), 0, 1))
        # losses = [self.criterion(output, target) for output, target in zip(outputs, targets)]
        # loss = torch.stack(losses, dim=0).sum(dim=0)
        return loss

    def calculate_accuracy(self, predictions, targets):
        if None in targets:
            targets = tuple([target for target in targets if target is not None])
        accuracy = []
        for prediction, target in zip(predictions, targets):
            if isinstance(prediction, torch.Tensor):
                accuracy.append(
                    accuracy_score(target.cpu().detach().numpy(), prediction.cpu().detach().numpy())
                )
            elif isinstance(prediction, tuple):
                acc_cond = []
                for idx, (cond_pred, cond_target) in enumerate(zip(prediction, target)):
                    if idx == 0:
                        acc_cond.append(
                            accuracy_score(cond_target.cpu().detach().numpy(), cond_pred.cpu().detach().numpy())
                        )
                    elif (idx == 1) or (idx == 2):
                        acc_cond.append(
                            (cond_pred.cpu().detach().numpy() == torch.transpose(torch.stack(cond_target), 0, 1)
                             .cpu().detach().numpy()).all(axis=(0)).mean()
                        )
                    else:
                        t = torch.transpose(torch.stack([torch.stack(lab) for lab in cond_target]), 0, 2)
                        acc_cond.append(
                            (cond_pred.cpu().detach().numpy() == t.cpu().detach().numpy()).all(axis=(0, 1)).mean()
                        )
                accuracy.append(np.average(acc_cond))
            else:
                pass
        return tuple(accuracy)
