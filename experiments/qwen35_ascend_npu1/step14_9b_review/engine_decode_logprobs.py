import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer
from tokenspeed.runtime.entrypoints.engine import Engine

def main():
    work=Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step14_9b_review')
    prev=work.parent/'step13_model_correctness'
    tp=int(sys.argv[1]); assert tp in (1,2)
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES']==('2' if tp==1 else '2,3')
    path='/home/u2601112013/models/Qwen3.5-9B'
    rows=json.loads((prev/'requests_9B_tp2.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    engine=None
    try:
        engine=Engine(model=path,device='npu',dtype='bfloat16',mamba_ssm_dtype='bfloat16',language_model_only=True,attention_backend='mha',sampling_backend='greedy',enforce_eager=True,disable_prefill_graph=True,disable_pdl=True,disable_autotune=True,disable_overlap_schedule=True,enable_prefix_caching=False,disable_kvstore=True,enable_output_logprobs=True,gpu_memory_utilization=0.8,retraction_snapshot_ratio=0,max_model_len=512,max_num_seqs=1,max_total_tokens=4096,chunked_prefill_size=512,attn_tp_size=tp,port=30600,log_level='info')
        out=[]
        for idx in (1,2):
            ids=tokenizer(rows[idx]['prompt'],add_special_tokens=True)['input_ids']
            result=engine.generate(input_ids=ids,sampling_params={'max_new_tokens':16,'temperature':0,'ignore_eos':True},return_logprob=True,top_logprobs_num=0,return_text_in_logprobs=False)
            rec={'case':idx+1,'input_ids':ids,'result':result}
            out.append(rec)
            (work/f'engine_tp{tp}_decode_logprobs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str)+'\n')
            print(json.dumps(rec,ensure_ascii=False,default=str),flush=True)
    finally:
        if engine is not None: engine.shutdown()

if __name__=='__main__': main()
