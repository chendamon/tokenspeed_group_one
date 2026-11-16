import json
import os
from pathlib import Path
import torch
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5LinearDecoderLayer, Qwen3_5AttentionDecoderLayer, Qwen3_5GatedDeltaNet
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.layers.attention.backends.state.mamba import MambaAttnBackend
from tokenspeed.runtime.entrypoints.engine import Engine
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor

ROOT=Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step15_tp_layout')
ACTIVE=None

def shape(x):
    if isinstance(x, torch.Tensor): return list(x.shape)
    if isinstance(x, (tuple,list)): return [shape(v) for v in x]
    return None

def log(record):
    if ACTIVE is None: return
    rank,layer=ACTIVE
    if layer not in (-2,-1,0,3): return
    row={'rank':rank,'layer':layer,'pid':os.getpid(),**record}
    with (ROOT/f'rank{rank}.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')

def layer_wrap(cls):
    original=cls.forward
    def wrapped(self,*args,**kwargs):
        global ACTIVE
        prev=ACTIVE
        ACTIVE=(self.mapping.attn.tp_rank,self.layer_id)
        try:
            if self.layer_id in (0,3):
                attn=self.linear_attn if hasattr(self,'linear_attn') else self
                weights={'gate_up':shape(self.mlp.gate_up_proj.weight),'down':shape(self.mlp.down_proj.weight)}
                if hasattr(self,'linear_attn'):
                    weights.update({'in_proj_qkvzba':shape(attn.in_proj_qkvzba.weight),'conv':shape(attn.conv1d.weight),'out_proj':shape(attn.out_proj.weight),'A_log':shape(attn.A_log)})
                else:
                    weights.update({'qkv_proj':shape(self.qkv_proj.weight),'out_proj':shape(self.o_proj.weight)})
                log({'event':'layer','input':shape(kwargs.get('hidden_states') if 'hidden_states' in kwargs else args[0 if hasattr(self,'linear_attn') else 1]),'weights':weights,'attn_tp':self.mapping.attn.tp_size,'dense_tp':self.mapping.dense.tp_size})
            return original(self,*args,**kwargs)
        finally:
            ACTIVE=prev
    cls.forward=wrapped

layer_wrap(Qwen3_5LinearDecoderLayer)
layer_wrap(Qwen3_5AttentionDecoderLayer)

def wrap_method(cls,name):
    original=getattr(cls,name)
    def wrapped(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        log({'event':name,'input':shape(args[0]),'output':shape(result)})
        return result
    setattr(cls,name,wrapped)

wrap_method(Qwen3_5GatedDeltaNet,'_forward_input_proj')
wrap_method(Qwen3_5AttentionDecoderLayer,'_project_qkv')
wrap_method(MambaAttnBackend,'_state_components')

def wrap_collective(name):
    original=getattr(HcclBackend,name)
    def wrapped(self,*args,**kwargs):
        if ACTIVE is not None and ACTIVE[1] in (-2,-1,0,3):
            tensors=[{'shape':shape(x),'dtype':str(x.dtype),'bytes':x.numel()*x.element_size()} for x in args if isinstance(x,torch.Tensor)]
            log({'event':'hccl_'+name,'tensors':tensors})
        return original(self,*args,**kwargs)
    setattr(HcclBackend,name,wrapped)
for name in ('all_reduce','all_gather_single','reduce_scatter'): wrap_collective(name)

original_embedding_forward=VocabParallelEmbedding.forward
def embedding_forward(self,*args,**kwargs):
    global ACTIVE
    prev=ACTIVE
    ACTIVE=(self.tp_rank,-1)
    try:
        log({'event':'embedding','input':shape(args[0]),'weight':shape(self.weight),'tp_size':self.tp_size})
        return original_embedding_forward(self,*args,**kwargs)
    finally:
        ACTIVE=prev
VocabParallelEmbedding.forward=embedding_forward

original_get_logits=LogitsProcessor._get_logits
def get_logits(self,*args,**kwargs):
    global ACTIVE
    prev=ACTIVE
    ACTIVE=(self.tp_rank,-2)
    try:
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        lm_head=kwargs.get('lm_head',args[1] if len(args)>1 else None)
        log({'event':'logits','input':shape(hidden),'weight':shape(lm_head.weight),'tp_size':self.tp_size})
        return original_get_logits(self,*args,**kwargs)
    finally:
        ACTIVE=prev
LogitsProcessor._get_logits=get_logits

def main():
    assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3' and os.environ['ASCEND_VISIBLE_DEVICES']=='2,3'
    ROOT.mkdir(parents=True,exist_ok=True)
    engine=None
    try:
        engine=Engine(model='/home/u2601112013/models/Qwen3.5-9B',device='npu',dtype='bfloat16',mamba_ssm_dtype='bfloat16',language_model_only=True,attention_backend='mha',sampling_backend='greedy',enforce_eager=True,disable_prefill_graph=True,disable_pdl=True,disable_autotune=True,disable_overlap_schedule=True,enable_prefix_caching=False,disable_kvstore=True,gpu_memory_utilization=0.8,retraction_snapshot_ratio=0,max_model_len=512,max_num_seqs=1,max_total_tokens=4096,chunked_prefill_size=512,attn_tp_size=2,port=30600,log_level='info')
        result=engine.generate(prompt='The capital of France is',sampling_params={'max_new_tokens':3,'temperature':0,'ignore_eos':True})
        (ROOT/'output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
        print('TRACE_REQUEST_OK',flush=True)
    finally:
        if engine is not None: engine.shutdown()
if __name__=='__main__': main()
