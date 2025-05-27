import json
import os


def get_slots_labels(text,slots_value):
    labels=['O']*len(text)
    i=0
    while i<len(text):
        for key,value in slots_value.items():
            j=0
            while j<len(value) and i+j<len(text) and text[i+j] == value[j]:
                j+=1
            if j>=len(value):
                labels[i]="B_"+key
                i+=1
                for k in range(1,j):
                    labels[i]="I_"+key
                    i+=1
                break
        i+=1
    return labels
def split_word(word):
    i=len(word)-1
    res=[]
    while i>=0 and not word[i].isalnum():
        res.append(word[i])
        i-=1
    res.append(word[:i+1])
    res.reverse()
    return res
def split_text(text):
    res=text.split()
    final_res=[]
    for t in res:
        if not t[-1].isalnum():
            final_res+=split_word(t)
        else:
            final_res.append(t)
    return final_res
def process_data_sim(folder,input_file, output_file):
    if os.path.exists(os.path.join(folder,"intent.json")):
        with open(os.path.join(folder,"intent.json"), 'r',encoding='utf-8') as f:
            intents = json.load(f)
        with open(os.path.join(folder,"slots.json"), 'r',encoding='utf-8') as f:
            slots = json.load(f)
    else:
        intents = ['UNK']
        slots = ["O"]
    with open(os.path.join(folder,input_file), 'r',encoding='utf-8') as f:
        data = json.load(f)
        all_processed_data=[]
        for d in data:
            intent_temp="UNK"
            for t in d["turns"]:
                data_temp={}
                if "user_intents" in t.keys():
                    data_temp["intent"]=t["user_intents"][0]
                    intent_temp=data_temp["intent"]
                    if data_temp["intent"]  not in intents:
                        intents.append(data_temp["intent"])
                else:
                    data_temp["intent"]=intent_temp
                if "system_utterance" in t.keys():
                    all_processed_data.append({
                        "text": t["system_utterance"]["tokens"],
                        "speaker":"system",
                    })
                data_temp["text"]=t["user_utterance"]["tokens"]
                slots_value={}
                for s in t["user_utterance"]["slots"]:
                    slots_value[s["slot"]]=data_temp["text"][s["start"]:s["exclusive_end"]]
                    if "B_"+s["slot"] not in slots:
                        slots.append("B_"+s["slot"])
                        slots.append("I_"+ s["slot"])
                data_temp["slots_labels"]=get_slots_labels(data_temp["text"],slots_value)
                data_temp["speaker"]="user"
                all_processed_data.append(data_temp)
    with open(os.path.join(folder,output_file), 'w',encoding='utf-8') as f:
        json.dump(all_processed_data,f,indent=4)
    if not os.path.exists(os.path.join(folder,"intent.json")):
        with open(os.path.join(folder,"intent.json"), 'w',encoding='utf-8') as f:
            json.dump(intents,f,indent=4,ensure_ascii=False)
        with open(os.path.join(folder,"slots.json"), 'w',encoding='utf-8') as f:
            json.dump(slots,f,indent=4,ensure_ascii=False)
def process_data_atis(folder,input_file, output_file):
    if os.path.exists(os.path.join(folder,"intent.json")):
        with open(os.path.join(folder,"intent.json"), 'r',encoding='utf-8') as f:
            intents = json.load(f)
        with open(os.path.join(folder,"slots.json"), 'r',encoding='utf-8') as f:
            slots = json.load(f)
    else:
        intents = ['UNK']
        slots = ["O"]
    with open(os.path.join(folder,input_file), 'r',encoding='utf-8') as f:
        data = json.load(f)
        data=data["rasa_nlu_data"]["common_examples"]
        all_processed_data=[]
        for d in data:
            data_temp={}
            data_temp["intent"]=d["intent"]
            if data_temp["intent"] not in intents:
                continue
                intents.append(data_temp["intent"])

            data_temp["text"]=split_text(d["text"])
            slots_value={}
            for s in d["entities"]:
                if "B_"+s["entity"] not in slots:
                    continue
                    slots.append("B_"+s["entity"])
                    slots.append("I_"+s["entity"])
                slots_value[s["entity"]] = split_text(s["value"])
            data_temp["slots_labels"] = get_slots_labels(data_temp["text"], slots_value)
            data_temp["speaker"] = "user"
            all_processed_data.append(data_temp)
    with open(os.path.join(folder,output_file), 'w',encoding='utf-8') as f:
        json.dump(all_processed_data,f,indent=4,ensure_ascii=False)
    if not os.path.exists(os.path.join(folder,"intent.json")):
        with open(os.path.join(folder,"intent.json"), 'w',encoding='utf-8') as f:
            json.dump(intents,f,indent=4,ensure_ascii=False)
        with open(os.path.join(folder,"slots.json"), 'w',encoding='utf-8') as f:
            json.dump(slots,f,indent=4,ensure_ascii=False)
def process_data_multiwoz(folder,input_file, output_file):
    if os.path.exists(os.path.join(folder,"intent.json")):
        with open(os.path.join(folder,"intent.json"), 'r',encoding='utf-8') as f:
            intents = json.load(f)
        with open(os.path.join(folder,"slots.json"), 'r',encoding='utf-8') as f:
            slots = json.load(f)
    else:
        intents = ['UNK']
        slots = ["O"]
    with open(os.path.join(folder,input_file), 'r',encoding='utf-8') as f:
        data = json.load(f)
        all_processed_data=[]
        for d in data:
            start=False
            for t in d["turns"]:
                if not start:
                    start=True
                    is_start=1
                else:
                    is_start=0
                if t["speaker"]=="SYSTEM":
                    all_processed_data.append({"text":split_text(t["utterance"]),
                                               "speaker":"system",})
                    continue
                data_temp={}
                data_temp["text"]=split_text(t["utterance"])
                for f in t["frames"]:
                    if f["state"]["active_intent"] !="NONE":
                        data_temp["intent"] = f["state"]["active_intent"]
                        slots_value={}
                        for s in f["slots"]:
                            if "copy_from" in s.keys():
                                continue
                            slots_value[s["slot"]] = split_text(s["value"])
                            if "B_" + s["slot"] not in slots:
                                slots.append("B_" + s["slot"])
                                slots.append("I_" + s["slot"])
                        data_temp["slots_labels"] = get_slots_labels(data_temp["text"], slots_value)
                        break
                if "intent" not in data_temp.keys():
                    data_temp["intent"] = "UNK"
                    data_temp["slots_labels"] = ['O']*len(data_temp["text"])
                if data_temp["intent"] not in intents:
                    intents.append(data_temp["intent"])
                data_temp["speaker"] = "user"
                data_temp["start"]=is_start
                all_processed_data.append(data_temp)
    with open(os.path.join(folder,output_file), 'w',encoding='utf-8') as f:
        json.dump(all_processed_data,f,indent=4,ensure_ascii=False)
    if not os.path.exists(os.path.join(folder,"intent.json")):
        with open(os.path.join(folder,"intent.json"), 'w',encoding='utf-8') as f:
            json.dump(intents,f,indent=4,ensure_ascii=False)
        with open(os.path.join(folder,"slots.json"), 'w',encoding='utf-8') as f:
            json.dump(slots,f,indent=4,ensure_ascii=False)
def get_intent_shift_label_and_position_number(input_file, output_file):
    with open(input_file, 'r',encoding='utf-8') as f:
        data=json.load(f)
        pre_intent="UNK"
        index=0
        for d in data:
            if d["speaker"] =="system":
                index+=1
                continue
            if pre_intent==d["intent"]:
                d["intent_shift"]=0
            else:
                d["intent_shift"]=1
                pre_intent=d["intent"]
            d["index"]=index
            index+=1
        with open(output_file, 'w',encoding='utf-8') as f:
            json.dump(data,f,indent=4,ensure_ascii=False)
if __name__ == '__main__':
    #Here is a example processing the sim dataset
    #The "sim" is the folder of your data and
    #the "test/train_raw.json" is the all the dialogues you have merged from the original dataset.
    #You can see the "test/train_raw.json" file to further understand it.
    #Run the code below, and you can obtain all the intents (intent.json) and all the slots (slots.json)
    process_data_sim("sim","test_raw.json","test.json")
    process_data_sim("sim","train_raw.json","train.json")
    get_intent_shift_label_and_position_number("sim/test.json","sim/test.json")
    get_intent_shift_label_and_position_number("sim/train.json","sim/train.json")
