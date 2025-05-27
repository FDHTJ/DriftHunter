import gc
import json
import tqdm
import os

from modelscope import snapshot_download
#Download the model
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-8B',cache_dir="deepseek-r1")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 



from transformers import pipeline, AutoTokenizer
import torch
model_id = "deepseek-r1/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    framework="pt",
)

def get_result(prompt: str) -> str:
    out = generator(
        prompt,
        max_new_tokens=10240,
        return_full_text=False,
    )
    res=out[0]["generated_text"]
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return res
prompt_template='''
Previous utterance:
<previous utterance>

Current utterance:
<current utterance>

Instruction
You are given a previous utterance and a current utterance from the user in a dialogue. Your task is to determine whether the intent of the current utterance is different from the previous one.

Response format:

Analysis: Provide a step-by-step reasoning comparing the two utterances, focusing on intent differences.
Decision: Output one of the following:
[Yes, they have different intents./No, they have the same intent.]

'''
for data_source in ["sim"]:
    if not os.path.exists(f"deepseek_r1_8B_{data_source}_test.jsonl"):
        with open(f"deepseek_r1_8B_{data_source}_test.jsonl", 'w', encoding="utf-8") as f:
            pass
    with open(f"deepseek_r1_8B_{data_source}_test.jsonl",'r',encoding="utf-8") as f:
        l=len(f.readlines())
    with open(f"../{data_source}/test.json",'r',encoding='utf-8') as f:
        with open(f"deepseek_r1_8B_{data_source}_test.jsonl",'a',encoding="utf-8") as w:
            data=json.load(f)[l:]
            pre_utterance=""
            for d in tqdm.tqdm(data):
                if d["speaker"]=="system":
                    w.write(json.dumps(d) + '\n')
                    continue
                current_utterance=" ".join(d["text"])
                prompt=prompt_template.replace("<previous utterance>",pre_utterance)
                prompt=prompt.replace("<current utterance>",current_utterance)
                res=get_result(prompt)
                d["res_deepseek"]=res
                pre_utterance=current_utterance
                w.write(json.dumps(d) + '\n')


