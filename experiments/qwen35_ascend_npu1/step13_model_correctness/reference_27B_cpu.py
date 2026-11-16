import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '2,3'
torch.set_num_threads(8)
model_path = '/home/u2601112013/models/Qwen3.5-27B'
prompt = 'The capital of France is'
start = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True, attn_implementation='eager').eval()
print(f'model_loaded_seconds={time.monotonic()-start:.1f}', flush=True)
inputs = tokenizer(prompt, return_tensors='pt')
with torch.inference_mode():
    tokens = model.generate(**inputs, do_sample=False, max_new_tokens=4, use_cache=True)
new_tokens = tokens[0, inputs['input_ids'].shape[-1]:].tolist()
result = {'prompt': prompt, 'input_ids': inputs['input_ids'][0].tolist(), 'new_token_ids': new_tokens, 'text': tokenizer.decode(new_tokens, skip_special_tokens=False), 'dtype': 'bfloat16', 'device': 'cpu', 'transformers_reference': 'Qwen3_5ForConditionalGeneration'}
Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step13_model_correctness/reference_27B_hf_cpu.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
