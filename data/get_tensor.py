from transformers import BertTokenizer, BertModel, AutoModel
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
import sklearn.metrics as sm
import pickle
intent=[1,2,3]
print(intent.index(3))
device=torch.device('cuda:0')
tokenizer_1=BertTokenizer.from_pretrained('../tokenizers/bert-base-uncased')
model_1=AutoModel.from_pretrained('../model/pretrained_models/bert-base-uncased')
model_1.to(device)
model_1.eval()
def get_text_embedding(text):
    with torch.no_grad():
        text=tokenizer_1(text,add_special_tokens=True,padding="max_length",truncation=True,return_tensors="pt",max_length=128)
        for key,value in text.items():
            text[key]=value.to(device)
        output=model_1(**text).last_hidden_state[:,0,:]
    return output

with open("atis/test_lack.json", "r", encoding="utf-8") as f:
    data=json.load(f)
    all_test_embedding=None
    for  i in tqdm.tqdm(data):
        t=get_text_embedding(" ".join(i['text']))
        if all_test_embedding is None:
            all_test_embedding=t
        else:
            all_test_embedding=torch.cat((all_test_embedding,t),dim=0)
    with open("atis/test_lack.pkl", "wb") as f:
        pickle.dump(all_test_embedding,f)
with open("atis/test.json", "r", encoding="utf-8") as f:
    data=json.load(f)
    all_test_embedding=None
    for  i in tqdm.tqdm(data):
        t=get_text_embedding(" ".join(i['text']))
        if all_test_embedding is None:
            all_test_embedding=t
        else:
            all_test_embedding=torch.cat((all_test_embedding,t),dim=0)
    with open("atis/test.pkl", "wb") as f:
        pickle.dump(all_test_embedding,f)