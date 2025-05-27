import sklearn.metrics as sm
import json
import os

def save_result(file,current_result,epoch):
    import os
    if not os.path.exists(file) or epoch==0:
        with open(file,'w',encoding='utf-8') as f:
            json.dump(current_result,f,indent=4,ensure_ascii=False)
        return
    with open(file,'r',encoding='utf-8') as f:
        pre=json.load(f)
    if pre["F1"]<current_result["F1"]:
        with open(file,'w',encoding='utf-8') as f:
            json.dump(current_result,f,indent=4,ensure_ascii=False)
def get_metrics_intent_drift(true_label,pred,is_train,folder,epoch):
    if is_train:
        print("intent drift"
              "train set：")
    else:
        print("intent drift"
              "test set：")
    metrics_dict = {
        "Accuracy": sm.accuracy_score(true_label, pred),
        "Precision": sm.precision_score(true_label, pred),
        "Recall": sm.recall_score(true_label, pred),
        "F1": sm.f1_score(true_label, pred)
    }
    print("Accuracy :", sm.accuracy_score(true_label, pred))
    print("Precision :", sm.precision_score(true_label, pred))
    print("Recall :", sm.recall_score(true_label, pred))
    print("F1 :", sm.f1_score(true_label, pred))
    if not is_train:
        file=os.path.join(folder,"intent_drift_with_tta.json")
        save_result(file,metrics_dict,epoch)
def get_metrics_slots(true_label,pred,is_train,folder,epoch):
    if is_train:
        print("slots"
              "train set：")
    else:
        print("slots"
              "test set：")
    metrics_dict = {
        "Accuracy": sm.accuracy_score(true_label, pred),
        "Precision": sm.precision_score(true_label, pred,average='macro',zero_division=0),
        "Recall": sm.recall_score(true_label, pred,average='macro',zero_division=0),
        "F1": sm.f1_score(true_label, pred,average='macro',zero_division=0)
    }
    print("Accuracy :", sm.accuracy_score(true_label, pred))
    print("Precision :", sm.precision_score(true_label, pred,average='macro',zero_division=0))
    print("Recall :", sm.recall_score(true_label, pred,average='macro',zero_division=0))
    print("F1 :", sm.f1_score(true_label, pred,average='macro',zero_division=0))
    if not is_train:
        file=os.path.join(folder,"slots_with_tta.json")
        save_result(file,metrics_dict,epoch)
def get_metrics_intent(true_label,pred,is_train,folder,epoch):
    if is_train:
        print("intent"
              "train set：")
    else:
        print("intent"
              "test set：")
    metrics_dict = {
        "Accuracy": sm.accuracy_score(true_label, pred),
        "Precision": sm.precision_score(true_label, pred, average='macro',zero_division=0),
        "Recall": sm.recall_score(true_label, pred, average='macro',zero_division=0),
        "F1": sm.f1_score(true_label, pred, average='macro',zero_division=0)
    }
    print("Accuracy :", sm.accuracy_score(true_label, pred))
    print("Precision :", sm.precision_score(true_label, pred,average='macro',zero_division=0))
    print("Recall :", sm.recall_score(true_label, pred,average='macro',zero_division=0))
    print("F1 :", sm.f1_score(true_label, pred,average='macro',zero_division=0))
    if not is_train:
        file=os.path.join(folder,"intent_with_tta.json")
        save_result(file,metrics_dict,epoch)
