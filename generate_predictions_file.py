from tqdm import tqdm
import json
import torch

from torch.utils.data import DataLoader
from llmtosql.model import WikiSQLModel
from llmtosql.dataloader import WikiSQLDataset
from llmtosql.utils.utils import load_model

model_path = 'model_output/model.pth'
base_model_type = 'bert-base-uncased'
path_file_output = 'model_output/test_results.jsonl'
set_type = 'test'
BATCH_SIZE = 2

# TODO: check function for bug [null, null, null]
def generate():
    model = WikiSQLModel(base_model_type=base_model_type, attention_type='cross')
    model = load_model(model, model_path)
    test_set = WikiSQLDataset(type=set_type, model=model)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    sel = []
    agg = []
    conds = []
    with tqdm(test_loader, unit='batch') as tepoch:
        for data in tepoch:
            questions = data['input'][0]
            inputs, _ = model.unpack(data, device)
            outputs = model(inputs)
            predictions = model.predict(outputs)
            sel.extend(predictions[0].tolist())
            agg.extend(predictions[1].tolist())
            for idx, cond in enumerate(predictions[2]):
                if len(cond.shape) == 1:
                    cond = cond.unsqueeze(1)
                if idx == 0:
                    max_num_conditions = torch.max(cond).item()
                    if max_num_conditions == 0:
                        cond_1,  cond_2, cond_3 = [[None]], [[None]], [[None]]
                        break
                elif idx == 1:
                    cond_1 = cond.T.tolist()
                    cond_1 = cond_1[:max_num_conditions]
                elif idx == 2:
                    cond = cond - 1
                    cond_2 = cond.T.tolist()
                    cond_2 = cond_2[:max_num_conditions]
                elif idx == 3:
                    outer_list = []
                    for condition in torch.transpose(predictions[2][3].T, 1, 2).tolist():
                        batch_list = [
                            WikiSQLDataset._digitize(' '.join((q.split())[cond_range[0]:cond_range[0] + cond_range[1]]))
                            for cond_range, q in zip(condition, questions)]
                        outer_list.append(batch_list)
                    cond_3 = outer_list
                    cond_3 = cond_3[:max_num_conditions]
            all_conds = []
            for c1, c2, c3 in zip(cond_1, cond_2, cond_3):
                inner_all_conds = []
                for b1, b2, b3 in zip(c1, c2, c3):
                    if b2 == -1:
                        b1, b2, b3 = None, None, None
                    inner_all_conds.append((b1, b2, b3))
                all_conds.append(inner_all_conds)
            conds.extend([list(x) for x in zip(*all_conds)])
    final = []
    for s, a, c in zip(sel, agg, conds):
        solution = {
            "query": {
                "sel": s,
                "agg": a
            }
        }
        if all([all([x is None for x in cond]) for cond in c]):
            c = None
        if c is not None:
            solution["query"]["conds"] = [list(x) for x in c if x != (None, None, None)]
        final.append(solution)
    with open(path_file_output, 'w+') as f:
        for line in final:
            json.dump(line, f)
            f.write('\n')


if __name__ == "__main__":
    generate()
