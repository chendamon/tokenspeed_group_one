import json
import sys
import urllib.request
from pathlib import Path

model, tp, out_path = sys.argv[1:]
long_prompt = 'The following numbers are listed in order: ' + ', '.join(str(i) for i in range(80)) + '. The next number is'
prompts = [
    'The capital of France is',
    'Compute 2 + 3 =',
    long_prompt,
    'The capital of France is',
]
records = []
for prompt in prompts:
    body = {'model': f'qwen35-{model}', 'prompt': prompt, 'max_tokens': 16, 'temperature': 0, 'seed': 42}
    request = urllib.request.Request('http://127.0.0.1:38172/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    choice = result['choices'][0]
    record = {'prompt': prompt, 'text': choice['text'], 'finish_reason': choice['finish_reason'], 'usage': result['usage']}
    assert result['usage']['completion_tokens'] > 0, record
    records.append(record)
    print(json.dumps({'model': model, 'tp': int(tp), 'request': len(records), 'text': choice['text'], 'usage': result['usage']}, ensure_ascii=False), flush=True)
Path(out_path).write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
if records[0]['text'] != records[-1]['text']:
    raise AssertionError('Repeated prompt changed after intervening requests')
