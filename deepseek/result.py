import json
import sklearn.metrics as sm
def get_result(respond:str):
    index=respond.find("</think>")
    if index != -1:
        respond=respond[index+8:]
    respond=respond.lower()
    if respond.find("answer") !=-1:
        respond=respond[respond.find("answer")+6:]
    elif respond.find("decision") !=-1:
        respond=respond[respond.find("decision")+8:]
    elif respond.find("conclusion") !=-1:
        respond=respond[respond.find("conclusion")+10:]
    else:
        pass
    if respond.find("yes") !=-1:
        return 1
    elif respond.find("no") !=-1:
        return 0
    else:return -1
def get_pred():
    with open("../multiwoz/test_original.json","r",encoding="utf-8") as f:
        data=json.load(f)
    with open("deepseek_r1_8B_multiwoz_test.jsonl", 'r',encoding='utf-8') as f:
        all_lines = f.readlines()
        all_processed_data = []
        index=0
        for line in all_lines:
            d = json.loads(line)
            if d['speaker']=="system":
                all_processed_data.append(d)
                index+=1
                continue
            else:
                d["pred_intent_drift"]=get_result(d["res_deepseek"])
                d["start"]=data[index]["start"]
                all_processed_data.append(d)
            index+=1
        with open("deepseek_r1_8B_multiwoz_test_orignal.json", 'w',encoding='utf-8') as f:
            json.dump(all_processed_data,f,ensure_ascii=False)
get_pred()
with open("deepseek_r1_8B_multiwoz_test_orignal.json", 'r',encoding='utf-8') as f:
    data = json.load(f)
    pred=[]
    true=[]
    for d  in data:
        if d["speaker"]=="system":continue
        if d["start"]:continue
        true.append(d["intent_shift"])
        pred.append(d["pred_intent_drift"] if d["pred_intent_drift"] !=-1 else 0)
    print("Acc",sm.accuracy_score(true,pred))
    print("Prec",sm.precision_score(true,pred))
    print("Rec",sm.recall_score(true,pred))
    print("F1",sm.f1_score(true,pred))
