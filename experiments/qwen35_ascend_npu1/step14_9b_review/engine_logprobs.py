import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer
from tokenspeed.runtime.entrypoints.engine import Engine

def main():
    work = Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step14_9b_review')
    previous = work.parent / 'step13_model_correctness'
    tp = int(sys.argv[1])
    assert tp in (1, 2)
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == ('2' if tp == 1 else '2,3')
    model_path = '/home/u2601112013/models/Qwen3.5-9B'
    ref = json.loads((previous / 'reference_9B_hf_cpu.json').read_text())
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    engine = None
    try:
        engine = Engine(model=model_path, device='npu', dtype='bfloat16', mamba_ssm_dtype='bfloat16', language_model_only=True, attention_backend='mha', sampling_backend='greedy', enforce_eager=True, disable_prefill_graph=True, disable_pdl=True, disable_autotune=True, disable_overlap_schedule=True, enable_prefix_caching=False, disable_kvstore=True, enable_output_logprobs=True, gpu_memory_utilization=0.8, retraction_snapshot_ratio=0, max_model_len=512, max_num_seqs=1, max_total_tokens=4096, chunked_prefill_size=512, attn_tp_size=tp, port=30600, log_level='info')
        output = []
        for idx, n in ((1, 9), (2, 4)):
            row = ref[idx]
            prompt_ids = tok(row['prompt'], add_special_tokens=True)['input_ids']
            prefix = row['new_token_ids'][:n]
            result = engine.generate(input_ids=prompt_ids + prefix, sampling_params={'max_new_tokens': 1, 'temperature': 0, 'ignore_eos': True}, return_logprob=True, top_logprobs_num=8, return_text_in_logprobs=False)
            record = {'case': idx + 1, 'teacher_forced_prefix_ids': prefix, 'prompt_length': len(prompt_ids), 'result': result}
            output.append(record)
            print(json.dumps(record, ensure_ascii=False, default=str), flush=True)
            (work / f'engine_tp{tp}_logprobs.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str) + '\n')
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
