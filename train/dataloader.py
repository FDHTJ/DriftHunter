from torch.utils.data import  Dataset
import torch
import json
import pickle
import os
import tqdm
class MyDataset(Dataset):
    def __init__(self,tokenizer,max_len,data_source,is_train,is_lack,is_original=True):
        super(MyDataset,self).__init__()
        data_file="train" if is_train else "test"
        data_file+=("_original" if is_original else "")
        data_file+=("_lack" if is_lack else "")
        self.data=None
        if os.path.exists(os.path.join(data_source,data_file+f"_{max_len}"+".plk")):
            self.data=pickle.load(open(os.path.join(data_source,data_file+f"_{max_len}"+".plk"),"rb"))
        else:
            self.process_data(tokenizer,max_len,data_source,data_file)
    def get_input(self,tokenizer,text,B_slot_number):
        tokens=tokenizer.encode(text,add_special_tokens=False)
        if B_slot_number!=0:
            if B_slot_number%2==0:
                slots=[B_slot_number]*len(tokens)
            else:
                slots=[B_slot_number+1]*len(tokens)
                slots[0]=B_slot_number
        else:
            slots=[0]*len(tokens)
        return tokens,slots
    def get_true(self,tokenizer,text,slots_labels):
        true_slots_labels=[]
        true_text=[]
        for i in range(len(text)):
            (t,s)=self.get_input(tokenizer,text[i],slots_labels[i])
            true_text+=t
            true_slots_labels+=s
        return true_text,true_slots_labels
    def get_attention_mask(self,text,slots_labels,max_len,tokenizer):
        attention_masks=[1]*len(text)
        pad_len=max_len-2-len(text)
        if pad_len<=0:
            attention_masks=attention_masks[:max_len-2]
            text=text[:max_len-2]
            slots_labels=slots_labels[:max_len-2]
        text=tokenizer.encode("[CLS]",add_special_tokens=False)+text+tokenizer.encode("[SEP]",add_special_tokens=False)
        slots_labels=[0]+slots_labels+[0]
        attention_masks=[1]+attention_masks+[1]
        if pad_len>0:
            text=text+[0]*pad_len
            slots_labels=slots_labels+[0]*pad_len
            attention_masks=attention_masks+[0]*pad_len
        return text,slots_labels,attention_masks
    def process_data(self,tokenizer,max_len,data_source,data_file):
        with open(os.path.join(data_source,data_file+".json"),"r",encoding="utf-8") as f:
            data=json.load(f)
            intent=json.load(open(os.path.join(data_source,"intent.json"),"r",encoding="utf-8"))
            slots=json.load(open(os.path.join(data_source,"slots.json"),"r",encoding="utf-8"))
            all_process_data=[]
            for d in tqdm.tqdm(data):
                if d["speaker"]=='system':
                    continue
                d["intent"]=intent.index(d["intent"])
                d["slots_labels"]=[slots.index(i) for i in d["slots_labels"]]
                (d['text'],d["slots_labels"])=self.get_true(tokenizer,d["text"],d["slots_labels"])
                (d['text'],d["slots_labels"],d["attention_masks"])=self.get_attention_mask(d["text"],d["slots_labels"],max_len,tokenizer)

                d.pop("speaker")
                for k,v in d.items():
                    d[k]=torch.tensor(v)
                all_process_data.append(d)
            self.data=all_process_data
            with open(os.path.join(data_source,data_file+f"_{max_len}"+".plk"),"wb") as f:
                pickle.dump(all_process_data,f)
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)
# if __name__=="__main__":
#     from transformers import AutoTokenizer
#     tokenizer=AutoTokenizer.from_pretrained("../tokenizers/bert-base-uncased")
#     data_iter=DataLoader(dataset=MyDataset(tokenizer,128,"atis",True,True),shuffle=False,batch_size=32)
#     data_iter1=DataLoader(dataset=MyDataset(tokenizer,128,"atis",False,True),shuffle=False,batch_size=32)
#     # data_iter2 = DataLoader(dataset=MyDataset(tokenizer, 128, "multiwoz", True, True), shuffle=False, batch_size=32)
#     # data_iter3 = DataLoader(dataset=MyDataset(tokenizer, 128, "multiwoz", False, True), shuffle=False, batch_size=32)
#     # for i in data_iter:
#     #     print(i)
#     #     break