import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

model_size = sys.argv[1]
assert model_size in ('0.8B', '4B', '9B', '27B')
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '2,3'
torch.set_num_threads(8)
model_path = f'/home/u2601112013/models/Qwen3.5-{model_size}'
work = Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step13_model_correctness')
requests = json.loads((work / f'requests_{model_size}_tp2.json').read_text())
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True, attn_implementation='eager').eval()
print('model_loaded', flush=True)
records = []
for i, request in enumerate(requests):
    inputs = tokenizer(request['prompt'], return_tensors='pt')
    with torch.inference_mode():
        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=16, use_cache=True)
    new_ids = tokens[0, inputs['input_ids'].shape[-1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    record = {'prompt': request['prompt'], 'new_token_ids': new_ids, 'text': text, 'tp2_text': request['text'], 'match': text == request['text']}
    records.append(record)
    print(json.dumps({'request': i + 1, 'match': record['match'], 'cpu': text, 'tp2': request['text']}, ensure_ascii=False), flush=True)
    (work / f'reference_{model_size}_hf_cpu.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
