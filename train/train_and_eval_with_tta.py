import json
import os.path
import random

import numpy as np
import torch
import tqdm
import pickle
# import utills_without_g as utills
# import utills
import utills_for_parameter as utills
# import utills_for_alpha as utills
def set_seed(seed=42):
    # Python 随机数种子
    random.seed(seed)

    # NumPy 随机数种子
    np.random.seed(seed)

    # PyTorch 随机数种子（CPU和GPU）
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况

    # 固定 CuDNN 行为（可能牺牲性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量（可选）
    # os.environ['PYTHONHASHSEED'] = str(seed)


# 使用示例
set_seed(42)
def get_sentence_embeddings_per_batch(sentence_embeddings, index, p_u):
    res=None
    for i in index:
        t = sentence_embeddings[i:i+p_u].unsqueeze(dim=0)
        if res == None:
            res = t
        else:
            res = torch.cat((res, t), dim=0)
    return res


def train_epoch(model, train_loader, loss_intent_shift,loss_intent,loss_slots, optimizer, p_u,data_source,is_lack,h):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_true_intent, all_pred_intent = [], []
    all_true_slots, all_pred_slots = [], []
    all_true_intent_shift=[]
    all_pred_intent_shift=[]
    with open(os.path.join(data_source,"train_lack.pkl" if is_lack else "train.pkl"), "rb") as f:
        all_sentence_embeddings = pickle.load(f)
    if p_u!=0:
        prex = torch.zeros((p_u, 768), device=device)
        all_sentence_embeddings = torch.cat((prex, all_sentence_embeddings), dim=0)
    loss_sum = 0
    for d in tqdm.tqdm(train_loader):
        for k,v in d.items():
            d[k]=v.to(device)
        if p_u!=0:
            sentence_embeddings = get_sentence_embeddings_per_batch(all_sentence_embeddings,d["index"].flatten().tolist(),p_u)
        else:
            sentence_embeddings=None
        (intent_shift_logits,intent_logits,slots_logits) = model(d, sentence_embeddings)
        intent_shift_true_label=d["intent_shift"]
        intent_true_label=d["intent"]
        slots_true_label=d["slots_labels"]
        all_logits = []
        all_labels = []

        number = d['attention_masks'].sum(dim=1).tolist()

        for i in range(len(slots_logits)):
            for j in range(0,number[i]):
                all_logits.append(slots_logits[i][j])  # shape: [slot_dim]
                all_labels.append(slots_true_label[i][j])  # scalar

        # 转成 tensor
        all_logits = torch.stack(all_logits, dim=0).to(slots_logits.device)  # shape: [N, slot_dim]
        all_labels = torch.tensor(all_labels).to(slots_logits.device)
        l = (
            # h[0]*loss_intent_shift(intent_shift_logits.flatten(), intent_shift_true_label.float().flatten())+
            h[1]*loss_intent(intent_logits.view((-1, intent_logits.shape[-1])), intent_true_label.long().flatten())+
            h[2]*loss_slots(all_logits.view((-1,all_logits.shape[-1])), all_labels.long().flatten()))
        loss_sum += l.item()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        intent_shift_true_label=intent_shift_true_label.flatten().tolist()
        intent_true_label = intent_true_label.detach().flatten().tolist()
        slots_true_label = slots_true_label.detach().tolist()
        intent_shift_pred = (intent_shift_logits.sigmoid() > 0.5).detach().flatten().to(int).tolist()
        slots_pred=slots_logits.softmax(dim=-1).argmax(dim=-1).detach().to(int).tolist()
        intent_pred=intent_logits.softmax(dim=-1).argmax(dim=-1).detach().flatten().to(int).tolist()

        for i in range(len(intent_shift_pred)):
            all_true_intent_shift.append(intent_shift_true_label[i])
            all_true_intent.append(intent_true_label[i])
            all_pred_intent_shift.append(intent_shift_pred[i])
            all_pred_intent.append(intent_pred[i])
        number=d['attention_masks'].sum(dim=1).tolist()
        for i in range(len(slots_pred)):
            for j in range(number[i]):
                all_true_slots.append(slots_true_label[i][j])
                all_pred_slots.append(slots_pred[i][j])
                
    return loss_sum / len(train_loader), all_true_intent_shift,all_pred_intent_shift,all_true_intent, all_pred_intent ,all_true_slots, all_pred_slots


def eval(model, train_loader,loss_intent_shift, loss_intent, loss_slots, p_u, data_source, is_lack,h):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_true_intent, all_pred_intent = [], []
    all_true_slots, all_pred_slots = [], []
    all_true_intent_shift = []
    all_pred_intent_shift = []
    with open(os.path.join(data_source, "test_lack.pkl" if is_lack else "test.pkl"), "rb") as f:
        all_sentence_embeddings = pickle.load(f)
    if p_u != 0:
        prex = torch.zeros((p_u, 768), device=device)
        all_sentence_embeddings = torch.cat((prex, all_sentence_embeddings), dim=0)
    loss_sum = 0
    for d in tqdm.tqdm(train_loader):
        for k, v in d.items():
            d[k] = v.to(device)
        if p_u != 0:
            sentence_embeddings = get_sentence_embeddings_per_batch(all_sentence_embeddings,
                                                                    d["index"].flatten().tolist(), p_u)
        else:
            sentence_embeddings = None
        (intent_shift_logits,intent_logits,slots_logits) = model(d, sentence_embeddings)
        intent_shift_true_label = d["intent_shift"]
        intent_true_label = d["intent"]
        slots_true_label = d["slots_labels"]
        all_logits = []
        all_labels = []
        number = d['attention_masks'].sum(dim=1).tolist()

        for i in range(len(slots_logits)):
            for j in range(0, number[i]):
                all_logits.append(slots_logits[i][j])  # shape: [slot_dim]
                all_labels.append(slots_true_label[i][j])  # scalar

        # 转成 tensor
        all_logits = torch.stack(all_logits, dim=0).to(slots_logits.device)  # shape: [N, slot_dim]
        all_labels = torch.tensor(all_labels).to(slots_logits.device)
        l = (
                h[0]*loss_intent_shift(intent_shift_logits.flatten(), intent_shift_true_label.float().flatten()) +
                h[1]*loss_intent(intent_logits.view((-1, intent_logits.shape[-1])), intent_true_label.long().flatten()) +
                h[2]*loss_slots(all_logits.view((-1, all_logits.shape[-1])), all_labels.long().flatten()))
        loss_sum += l.item()
        intent_shift_true_label = intent_shift_true_label.flatten().tolist()
        intent_true_label = intent_true_label.detach().flatten().tolist()
        slots_true_label = slots_true_label.detach().tolist()
        intent_shift_pred = (intent_shift_logits.sigmoid() > 0.5).detach().flatten().to(int).tolist()
        slots_pred = slots_logits.softmax(dim=-1).argmax(dim=-1).detach().to(int).tolist()
        intent_pred = intent_logits.softmax(dim=-1).argmax(dim=-1).detach().flatten().to(int).tolist()

        for i in range(len(intent_shift_pred)):
            all_true_intent_shift.append(intent_shift_true_label[i])
            all_true_intent.append(intent_true_label[i])
            all_pred_intent_shift.append(intent_shift_pred[i])
            all_pred_intent.append(intent_pred[i])
        number = d['attention_masks'].sum(dim=1).tolist()
        for i in range(len(slots_pred)):
            for j in range(number[i]):
                all_true_slots.append(slots_true_label[i][j])
                all_pred_slots.append(slots_pred[i][j])

    return loss_sum / len(
        train_loader), all_true_intent_shift, all_pred_intent_shift, all_true_intent, all_pred_intent, all_true_slots, all_pred_slots


def train(model, train_loader, test_loader, optimizer, epoch, loss_intent_shift,loss_intent,loss_slots,p_u,data_source,is_lack,file,h,l):
    for i in range(epoch):
        l_train, tis_train,pis_train,ti_train, pi_train,ts_train,ps_train = train_epoch(model, train_loader,loss_intent_shift,loss_intent,loss_slots, optimizer,p_u,data_source,is_lack,h)
        # before=model.state_dict().copy()
        # before_metrix=model.tta.transition_matrix.cpu().tolist().copy()
        # before_last_result=model.tta.last_result.cpu().tolist().copy()
        l_test, tis_test,pis_test,ti_test, pi_test,ts_test, ps_test = eval(model, test_loader, loss_intent_shift,loss_intent,loss_slots,p_u,data_source,is_lack,h)
        print("epoch:", i + 1)
        print("train loss:", l_train)
        # utills.get_metrics_intent_drift(tis_train,pis_train,True,file)
        # utills.get_metrics_intent(ti_train, pi_train,True,file)
        # utills.get_metrics_slots(ts_train,ps_train,True,file)
        print("test loss:", l_test)
        if utills.get_metrics_intent_drift(tis_test,pis_test,False,file,i,l):
            pass
        #     pass
            # after_metrix = model.tta.transition_matrix.cpu().tolist().copy()
            # after_last_result = model.tta.last_result.cpu().tolist().copy()
            # after=model.state_dict()
            # torch.save(before, "model_state_before_distance.pth")
            # with open("before_information_distance.json",'w',encoding='utf-8') as f:
            #     json.dump({"metrix":before_metrix,"last_result":before_last_result},f)
            # torch.save(after, "model_state_after_distance.pth")
            # with open("after_information_distance.json",'w',encoding='utf-8') as f:
            #     json.dump({"metrix":after_metrix,"last_result":after_last_result},f)
        utills.get_metrics_intent(ti_test, pi_test, False,file,i,l)
        utills.get_metrics_slots(ts_test, ps_test, False,file,i,l)
def get_statistic_information(data_source):
    train_data=json.load(open(os.path.join(data_source, "train.json"), "r",encoding="utf-8"))
    intents=json.load(open(os.path.join(data_source, "intent.json"),'r',encoding='utf-8'))
    slots=json.load(open(os.path.join(data_source, "slots.json"),'r',encoding='utf-8'))
    l_intent=len(intents)
    l_slots=len(slots)
    statistic_result=torch.zeros((l_intent+l_slots,l_intent+l_slots))
    pre_intent="UNK"
    for d in train_data:
        if d["speaker"]=="system":
            continue
        statistic_result[intents.index(d["intent"])][intents.index(pre_intent)]+=1
        statistic_result[intents.index(pre_intent)][intents.index(d["intent"])] += 1
        curr_intent_index=intents.index(d["intent"])
        statistic_result[curr_intent_index][curr_intent_index]+=1
        pre_intent=d["intent"]
        slots_indexes=[]
        for s in d["slots_labels"]:
            if slots.index(s) not in slots_indexes:
                slots_indexes.append(slots.index(s))
        for s_index in slots_indexes:
            statistic_result[s_index+l_intent][curr_intent_index]+=1
            statistic_result[curr_intent_index][s_index+l_intent]+=1
        for i in range(len(slots_indexes)):
            for j in range(i,len(slots_indexes)):
                statistic_result[slots_indexes[i]+l_intent][slots_indexes[j]+l_intent]+=1
                statistic_result[slots_indexes[j]+l_intent][slots_indexes[i]+l_intent]+=1
    # print(statistic_result)
    return statistic_result
if __name__ == '__main__':
    model_name="bert-base-uncased"
    with_TTA = True
    without_G=False
    is_lack=False
    max_len=128
    intent_shift_size=1
    p_u = 32  ##0
    for data_source in ["multiwoz"]:#[,,]
        intent_size=len(json.load(open(os.path.join(data_source, "intent.json"),'r',encoding='utf-8')))
        slots_size=len(json.load(open(os.path.join(data_source, "slots.json"),'r',encoding='utf-8')))
        for method in ["AAAI"]:#[]
            for l in np.arange(0.4, 1.1, 0.1):#[0.2]:
                print("当前数据集为：",data_source)
                print("当前方法为：",method)
                if method=="BERT":
                    from bert import BertWithTTA
                    model = BertWithTTA(intent_shift_size, intent_size, slots_size, max_len, model_name,p_u)
                    h=(0.05,0.15,0.8)
                elif method=="AAAI":
                    from model_complex import AdaptiveGlobalLocalContextFusionModelWithTTA
                    model = AdaptiveGlobalLocalContextFusionModelWithTTA(intent_shift_size, intent_size, slots_size, max_len,p_u,l)
                    h=(0.05,0.25,0.7)
                    print("当前配比为：",h)
                else:
                    intent_size =  len(json.load(open(os.path.join(data_source, "intent.json"),'r',encoding='utf-8')))
                    slots_size = len(json.load(open(os.path.join(data_source, "slots.json"), 'r', encoding='utf-8')))
                    from DanceWithLabels import DanceWithLabelsWithTTA
                    model=DanceWithLabelsWithTTA(intent_shift_size,intent_size, slots_size, max_len, p_u)
                    statistic_result = get_statistic_information(data_source).softmax(dim=-1)
                    model.DH_LGIL_intent.transition_matrix=statistic_result
                    model.DH_LGIL_slots.transition_matrix=statistic_result
                    h=(0.05,0.35,0.6)
                print("current_method:",method)
                from torch.optim import AdamW
                optimizer=AdamW(model.parameters(), lr=1e-5)
                from torch.utils.data import DataLoader
                from dataloader import  MyDataset
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained(f"../tokenizers/{model_name}")
                train_iter=DataLoader(dataset=MyDataset(tokenizer,max_len,data_source,True,is_lack,False),batch_size=32,shuffle=False)
                test_iter=DataLoader(dataset=MyDataset(tokenizer,max_len,data_source,False,is_lack,False),batch_size=32,shuffle=False)
                loss_intent_shift=torch.nn.BCEWithLogitsLoss()
                loss_intent=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
                loss_slots=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                file = os.path.join("results", method, f"{data_source}")
                train(model,train_iter,test_iter,optimizer,50,loss_intent_shift,loss_intent, loss_slots, p_u, data_source, is_lack,file,h,l)

            #
            # from bert import BertWithTTA
            # method="BERT"
            # with_TTA=True
            # model=BertWithTTA(intent_shift_size,intent_shift_size, slots_size, max_len, model_name,p_u)

            # from model_complex import AdaptiveGlobalLocalContextFusionModel
            # model=AdaptiveGlobalLocalContextFusionModel(intent_shift_size,intent_size, slots_size, max_len, p_u)

            # from DanceWithLabels import DanceWithLabels
            # model=DanceWithLabels(intent_shift_size,intent_size, slots_size, max_len, p_u)
            # statistic_result = get_statistic_information(data_source).softmax(dim=-1)
            # model.DH_LGIL_intent.transition_matrix=statistic_result
            # model.DH_LGIL_slots.transition_matrix=statistic_result

            # from DanceWithLabels import DanceWithLabelsWithTTA
            # model=DanceWithLabelsWithTTA(intent_shift_size,intent_size, slots_size, max_len, p_u)