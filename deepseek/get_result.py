import json

import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda:2")



model = AutoModelForCausalLM.from_pretrained(
    'deepseek-r1/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained('deepseek-r1/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')

def get_result(prompt):
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=131072,  # 调整生成长度
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.2
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        r = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return r
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
for data_source in ["atis","sim","multiwoz"]:
    with open(f"../{data_source}/test.json",'r',encoding='utf-8') as f:
        with open(f"deepseek_r1_1.5B_{data_source}_test.jsonl",'a',encoding="utf-8") as w:
            data=json.load(f)
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


