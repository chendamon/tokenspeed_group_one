import concurrent.futures
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

root=Path('/home/u2601112013/tokenspeed_group_one')
work=root/'experiments/qwen35_ascend_npu1/step16_tp_performance'
model,tp=sys.argv[1],int(sys.argv[2])
assert (model,tp) in [('4B',1),('4B',2),('27B',2)]
model_id=f'qwen35-{model}'
name=f'{model}_tp{tp}'
status={'phase':'startup'}
stop=threading.Event()
samples=[]

def snapshot(server_pid):
    raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=12)
    geometry=re.findall(r'\|\s*([01])\s+([23])\s*\|[^|]*\|\s*\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)',raw)
    devices={int(chip):{'physical_id':int(phy),'total_used_mb':int(used),'total_mb':int(total)} for chip,phy,used,total in geometry}
    if set(devices)!={0,1}: raise RuntimeError(f'Cannot resolve NPU 1 chip mapping: {geometry}')
    parents={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
        try: pid,ppid=map(int,line.split()); parents[pid]=ppid
        except ValueError: pass
    descendants={server_pid}
    while True:
        expanded=descendants | {pid for pid,ppid in parents.items() if ppid in descendants}
        if len(expanded)==len(descendants): break
        descendants=expanded
    processes=re.findall(r'\|\s*1\s+([01])\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|',raw)
    for chip in devices:
        owned=[{'pid':int(pid),'name':proc.strip(),'memory_mb':int(mem)} for c,pid,proc,mem in processes if int(c)==chip and int(pid) in descendants]
        devices[chip]['experiment_processes']=owned
        devices[chip]['experiment_process_memory_mb']=sum(x['memory_mb'] for x in owned)
    return {'wall_time':time.time(),'monotonic':time.monotonic(),'phase':status['phase'],'devices':devices}

def sample_loop(server_pid):
    while not stop.is_set():
        try: samples.append(snapshot(server_pid))
        except Exception as e: samples.append({'wall_time':time.time(),'phase':status['phase'],'error':str(e)})
        stop.wait(1.5)

def call(prompt):
    body={'model':model_id,'prompt':prompt,'max_tokens':32,'temperature':0,'seed':42}
    req=urllib.request.Request('http://127.0.0.1:38172/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    started=time.monotonic()
    try:
        with urllib.request.urlopen(req,timeout=240) as res: data=json.load(res)
        elapsed=time.monotonic()-started
        return {'ok':True,'elapsed_s':elapsed,'usage':data['usage'],'finish_reason':data['choices'][0]['finish_reason']}
    except Exception as e: return {'ok':False,'elapsed_s':time.monotonic()-started,'error':repr(e)}

def run_batch(prompt,concurrency,count):
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures=[pool.submit(call,prompt) for _ in range(count)]
        rows=[x.result() for x in futures]
    return rows,time.monotonic()-start

def percentile95(v):
    v=sorted(v)
    return v[max(0,math.ceil(0.95*len(v))-1)]

# CPU tokenization is used only to make exact, reproducible prompt lengths.
tok=AutoTokenizer.from_pretrained(f'/home/u2601112013/models/Qwen3.5-{model}',local_files_only=True)
ids=tok.encode(' hello'*600,add_special_tokens=False)
prompts={n:tok.decode(ids[:n],skip_special_tokens=False) for n in (128,512)}
for n,s in prompts.items(): assert len(tok.encode(s,add_special_tokens=False))==n
baseline=snapshot(-1)
required=[0] if tp==1 else [0,1]
minimum=37000 if model=='27B' else 12000
for chip in required:
    dev=baseline['devices'][chip]
    free=dev['total_mb']-dev['total_used_mb']
    if free<minimum: raise RuntimeError(f'NPU 1 chip {chip} free HBM {free} MB below guard {minimum} MB')
try:
    with urllib.request.urlopen('http://127.0.0.1:38172/health',timeout=2): raise RuntimeError('Target port already in use')
except (urllib.error.URLError,TimeoutError): pass
(work/f'baseline_{name}.json').write_text(json.dumps(baseline,indent=2)+'\n')
log=(work/f'server_{name}.log').open('w')
proc=subprocess.Popen(['bash',str(work/'run_server.sh'),model,str(tp)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
sampler=threading.Thread(target=sample_loop,args=(proc.pid,),daemon=True)
sampler.start()
results=[]
try:
    ready=False
    for _ in range(360):
        if proc.poll() is not None: break
        try:
            with urllib.request.urlopen('http://127.0.0.1:38172/v1/models',timeout=3) as res:
                registered=json.load(res).get('data',[])
            if any(x.get('id')==model_id for x in registered): ready=True; break
        except (urllib.error.URLError,TimeoutError): pass
        time.sleep(1)
    if not ready: raise RuntimeError(f'Server not ready; exit={proc.poll()}')
    status['phase']='loaded'
    time.sleep(3)
    print('READY',name,flush=True)
    for length in (128,512):
      for concurrency in ((1,4) if model == '4B' else (1,)):
        key=f'L{length}_C{concurrency}'
        status['phase']=f'{key}_warmup'
        warm,_=run_batch(prompts[length],concurrency,concurrency)
        if any(not x['ok'] for x in warm): raise RuntimeError(f'Warmup failed {key}: {warm}')
        for repeat in (1,2):
          status['phase']=f'{key}_r{repeat}'
          batch,elapsed=run_batch(prompts[length],concurrency,8)
          good=[x for x in batch if x['ok']]
          if len(good)!=8: raise RuntimeError(f'Failed requests {key}: {batch}')
          if any(x['usage']['prompt_tokens']!=length for x in good): raise RuntimeError(f'Prompt length mismatch: {batch}')
          latency=[x['elapsed_s'] for x in good]
          completion_tokens=sum(x['usage']['completion_tokens'] for x in good)
          result={'model':model,'tp':tp,'input_tokens':length,'concurrency':concurrency,'repeat':repeat,'successful_requests':len(good),'window_s':elapsed,'output_tokens':completion_tokens,'output_tokens_per_s':completion_tokens/elapsed,'latency_p50_s':statistics.median(latency),'latency_p95_s':percentile95(latency),'requests':batch}
          results.append(result)
          (work/f'results_{name}.json').write_text(json.dumps(results,indent=2)+'\n')
          print('DONE',name,key,'repeat',repeat,'output_tps',round(result['output_tokens_per_s'],2),'p50_s',round(result['latency_p50_s'],3),flush=True)
    print('CASE_COMPLETE',name,flush=True)
finally:
    status['phase']='shutdown'
    stop.set()
    sampler.join(timeout=20)
    (work/f'samples_{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples))
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=15)
    log.close()
