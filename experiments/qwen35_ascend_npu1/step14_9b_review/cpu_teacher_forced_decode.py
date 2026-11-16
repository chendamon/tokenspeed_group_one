import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3'
torch.set_num_threads(8)
work=Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step14_9b_review')
records=json.loads((work/'engine_tp1_decode_logprobs.json').read_text())
path='/home/u2601112013/models/Qwen3.5-9B'
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').eval()
all_results=[]
for rec in records:
    prompt=torch.tensor([rec['input_ids']],dtype=torch.long)
    ids=rec['result']['output_ids']
    steps=[]
    with torch.inference_mode():
        out=model(input_ids=prompt,attention_mask=torch.ones_like(prompt),use_cache=True)
        for step,token_id in enumerate(ids):
            scores=torch.log_softmax(out.logits[0,-1].float(),dim=-1)
            val,index=torch.max(scores,dim=-1)
            steps.append({'step':step+1,'forced_id':token_id,'forced_logprob':float(scores[token_id]),'cpu_argmax_id':int(index),'cpu_argmax_logprob':float(val)})
            if step+1<len(ids):
                out=model(input_ids=torch.tensor([[token_id]]),past_key_values=out.past_key_values,use_cache=True)
    result={'case':rec['case'],'steps':steps}; all_results.append(result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    (work/'cpu_teacher_forced_decode.json').write_text(json.dumps(all_results,indent=2,ensure_ascii=False)+'\n')
