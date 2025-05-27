import argparse

from transformers import AutoTokenizer, AutoModel
import json
import torch
import tqdm
import pickle
parser = argparse.ArgumentParser("This is the file to get the embedding of the previous utterances")
parser.add_argument("--model_path", default='../pretrain_model/bert-base-uncased', help="The path of the pretrained model",type=str)
parser.add_argument("--input_file", default='sim/train.json', help="The input file to get tensor",type=str)
parser.add_argument("--max_length", default=128, help="The max length of the tokens",type=int)
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_1=AutoTokenizer.from_pretrained(args.model_path)
model_1=AutoModel.from_pretrained(args.model_path)
model_1.to(device)
model_1.eval()
print("model path:",args.model_path)
print("input file:",args.input_file)
output_file=args.input_file.replace(".json",".pkl")
print("output file:",output_file)
def get_text_embedding(text,max_length):
    with torch.no_grad():
        text=tokenizer_1(text,add_special_tokens=True,padding="max_length",truncation=True,return_tensors="pt",max_length=max_length)
        for key,value in text.items():
            text[key]=value.to(device)
        output=model_1(**text).last_hidden_state[:,0,:]
    return output

with open(args.input_file, "r", encoding="utf-8") as f:
    data=json.load(f)
    all_test_embedding=None
    for  i in tqdm.tqdm(data):
        t=get_text_embedding(" ".join(i['text']),args.max_length)
        if all_test_embedding is None:
            all_test_embedding=t
        else:
            all_test_embedding=torch.cat((all_test_embedding,t),dim=0)
    with open(output_file, "wb") as f:
        pickle.dump(all_test_embedding,f)

