import json
import os
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '2,3'
torch.set_num_threads(8)
model_path = '/home/u2601112013/models/Qwen3.5-27B'
work = Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step13_model_correctness')
requests = json.loads((work / 'requests_27B_tp2.json').read_text())
start = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True, attn_implementation='eager').eval()
print(f'model_loaded_seconds={time.monotonic()-start:.1f}', flush=True)
records = []
for i, request in enumerate(requests):
    inputs = tokenizer(request['prompt'], return_tensors='pt')
    with torch.inference_mode():
        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=16, use_cache=True)
    new_ids = tokens[0, inputs['input_ids'].shape[-1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    record = {'prompt': request['prompt'], 'input_ids': inputs['input_ids'][0].tolist(), 'new_token_ids': new_ids, 'text': text, 'tp2_text': request['text'], 'match': text == request['text']}
    records.append(record)
    print(json.dumps({'request': i + 1, 'match': record['match'], 'cpu': text, 'tp2': request['text']}, ensure_ascii=False), flush=True)
    (work / 'reference_27B_hf_cpu_extended.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
