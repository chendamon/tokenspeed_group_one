import gc
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'
torch.set_num_threads(8)
work=Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step13_model_correctness')
for size in ('9B','27B'):
    path=f'/home/u2601112013/models/Qwen3.5-{size}'
    ref=work/('reference_27B_hf_cpu_extended.json' if size=='27B' else 'reference_9B_hf_cpu.json')
    rec=json.loads(ref.read_text())[2]
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True,attn_implementation='eager').eval()
    base=tok(rec['prompt'],return_tensors='pt')
    with torch.inference_mode():
        out=model(**base,use_cache=True)
        cache=out.past_key_values
        for token_id in rec['new_token_ids'][:4]:
            out=model(input_ids=torch.tensor([[token_id]]),past_key_values=cache,use_cache=True)
            cache=out.past_key_values
        logits=out.logits[0,-1].float()
        vals, ids=torch.topk(logits,8)
    result=[{'token_id':int(i),'token':tok.decode([int(i)]),'logit':float(v)} for i,v in zip(ids,vals)]
    print(json.dumps({'model':size,'incremental_top8':result},ensure_ascii=False),flush=True)
    del model, out, cache, logits
    gc.collect()
