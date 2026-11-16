import gc
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'
torch.set_num_threads(8)
work = Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step13_model_correctness')
for size in ('9B', '27B'):
    path = f'/home/u2601112013/models/Qwen3.5-{size}'
    ref_path = work / ('reference_27B_hf_cpu_extended.json' if size == '27B' else 'reference_9B_hf_cpu.json')
    rec = json.loads(ref_path.read_text())[2]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True, attn_implementation='eager').eval()
    inputs = tokenizer(rec['prompt'], return_tensors='pt')
    prefix = torch.tensor([rec['new_token_ids'][:4]], dtype=torch.long)
    inputs['input_ids'] = torch.cat([inputs['input_ids'], prefix], dim=-1)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
        vals, ids = torch.topk(logits, 8)
    result = [{'token_id': int(i), 'token': tokenizer.decode([int(i)]), 'logit': float(v)} for i,v in zip(ids, vals)]
    print(json.dumps({'model': size, 'top8': result, 'tp2_text': rec['tp2_text'], 'cpu_text': rec['text']}, ensure_ascii=False), flush=True)
    del model, inputs, logits
    gc.collect()
