from transformers import AutoModel, AutoTokenizer
model_path = 'models/transformers/' # will be created automatically if not exists

#%% download and save the model to local directory
model_name = "bert-base-uncased"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
