import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'
torch.set_num_threads(8)
work=Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step14_9b_review')
ref=json.loads((work.parent/'step13_model_correctness/reference_9B_hf_cpu.json').read_text())
path='/home/u2601112013/models/Qwen3.5-9B'
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').eval()
out=[]
for idx,n in ((1,9),(2,4)):
    prompt=tok(ref[idx]['prompt'],return_tensors='pt')
    inputs=torch.cat([prompt['input_ids'],torch.tensor([ref[idx]['new_token_ids'][:n]])],dim=1)
    with torch.inference_mode():
        logits=model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits[0,-1].float()
        scores=torch.log_softmax(logits,dim=-1)
        values,indices=torch.topk(scores,8)
    record={'case':idx+1,'prefix_ids':ref[idx]['new_token_ids'][:n],'top':[[int(i),tok.decode([int(i)]),float(v)] for v,i in zip(values,indices)]}
    out.append(record)
    print(json.dumps(record,ensure_ascii=False),flush=True)
    (work/'cpu_fixed_prefix_logprobs.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
