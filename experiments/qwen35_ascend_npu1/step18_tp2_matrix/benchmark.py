import concurrent.futures as cf
import datetime, hashlib, json, math, os, re, signal, socket, statistics, subprocess, threading, time, urllib.request, urllib.error
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path('/home/u2601112013/tokenspeed_group_one')
BASE=ROOT/'experiments/qwen35_ascend_npu1/step18_tp2_matrix'
OUT=BASE/('run_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir(parents=True,exist_ok=False)
PORT=38274
phase='startup'
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def snap(pid):
    raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
    dev={}
    for c,p,b,a,u,t in re.findall(r'\|\s*([01])\s+([23])\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)',raw):
        dev[int(c)]={'physical_id':int(p),'aicore':int(a),'used_mb':int(u),'total_mb':int(t)}
    assert set(dev)=={0,1}, dev
    ps=subprocess.check_output(['ps','-eo','pid=,ppid=,uid=,args='],text=True)
    records={}
    for line in ps.splitlines():
        x=line.strip().split(None,3)
        if len(x)==4: records[int(x[0])]={'ppid':int(x[1]),'uid':int(x[2]),'args':x[3]}
    own={pid}
    while True:
        new=own|{p for p,x in records.items() if x['ppid'] in own and x['uid']==os.getuid()}
        if new==own: break
        own=new
    for c in dev:
        entries=[{'pid':int(p),'name':n.strip(),'memory_mb':int(m),'owned':int(p) in own,**records.get(int(p),{})} for cc,p,n,m in re.findall(r'\|\s*1\s+([01])\s*\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|',raw) if int(cc)==c]
        dev[c]['processes']=entries
        dev[c]['own_mb']=sum(x['memory_mb'] for x in entries if x['owned'])
    return {'time':time.time(),'phase':phase,'devices':dev}
def percentile(v,q):
    return sorted(v)[max(0,math.ceil(q*len(v))-1)]
def request(model,prompt,barrier):
    prompt,expected_length=prompt
    barrier.wait()
    start=time.monotonic()
    body={'model':'qwen35-'+model,'prompt':prompt,'temperature':0,'seed':42,'max_tokens':64,'min_tokens':64,'min_new_tokens':64,'ignore_eos':True,'skip_special_tokens':False,'stream':True,'stream_options':{'include_usage':True},'return_token_ids':True}
    req=urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    events=[]; times=[]; text=''; usage=None; finish=None; done=False
    with urllib.request.urlopen(req,timeout=300) as res:
        for line in res:
            line=line.decode().strip()
            if not line.startswith('data:'): continue
            data=line[5:].strip()
            if data=='[DONE]': done=True; break
            obj=json.loads(data); now=time.monotonic()
            events.append({'offset_s':now-start,'data':obj})
            if obj.get('usage'): usage=obj['usage']
            for ch in obj.get('choices',[]):
                s=ch.get('text') or ''
                if s: times.append(now); text+=s
                if ch.get('finish_reason'): finish=ch['finish_reason']
    end=time.monotonic()
    assert usage and usage['completion_tokens']==64,(usage,events[-2:])
    assert done and times,(done,len(times))
    assert usage['prompt_tokens']==expected_length,(usage,expected_length)
    return {'start':start,'end':end,'first':times[0],'last':times[-1],'e2e_ms':(end-start)*1000,'ttft_ms':(times[0]-start)*1000,'tpot_ms':(times[-1]-times[0])*1000/63,'usage':usage,'finish':finish,'content_chunks':len(times),'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'events':events}
def batch(model,prompt,c):
    barrier=threading.Barrier(c)
    begin=time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=c) as pool:
        try: rows=list(pool.map(lambda _:request(model,prompt,barrier),range(c)))
        except urllib.error.HTTPError as e:
            print('HTTP_ERROR_BODY',e.read().decode(),flush=True)
            raise
    elapsed=time.monotonic()-begin
    overlap=c==1 or max(r['first'] for r in rows)<min(r['last'] for r in rows)
    return rows,elapsed,overlap
all_cells=[]
save(OUT/'protocol.json',{'models':['0.8B','4B','9B','27B'],'tp':2,'physical_ids':[2,3],'lengths':[128,512],'concurrency':[1,2],'output':64,'warmup_rounds':2,'measurement_rounds':5,'port':PORT,'shared_device':True,'timing':'HTTP SSE first/last nonempty content arrival; TPOT denominator 63; raw events retained','root':str(ROOT)})
(BASE/'latest_run.txt').write_text(str(OUT)+'\n')
for model in ['0.8B','4B','9B','27B']:
    proc=None; samples=[]; stop=threading.Event(); phase=model+'_startup'
    try:
        baseline=snap(-1); save(OUT/(model+'_baseline.json'),baseline)
        need=37000 if model=='27B' else 14000 if model=='9B' else 12000
        assert all(d['total_mb']-d['used_mb']>=need for d in baseline['devices'].values()),baseline
        with socket.socket() as s: assert s.connect_ex(('127.0.0.1',PORT))!=0,'Port occupied'
        launcher=(ROOT/'experiments/qwen35_ascend_npu1/step16_tp_performance/run_server.sh').read_text().replace('--port 38172',f'--port {PORT}').replace('--max-num-seqs 4','--max-num-seqs 2')
        (OUT/(model+'_launcher.sh')).write_text(launcher)
        log=(OUT/(model+'_server.log')).open('w')
        proc=subprocess.Popen(['bash','-c',launcher,'bench',model,'2'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        def sample_loop():
            while not stop.is_set():
                try: samples.append(snap(proc.pid))
                except Exception as e: samples.append({'phase':phase,'error':repr(e)})
                stop.wait(1.5)
        sampler=threading.Thread(target=sample_loop,daemon=True); sampler.start()
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            assert proc.poll() is None,'Server exited'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models',timeout=2) as r:
                    if any(x['id']=='qwen35-'+model for x in json.load(r)['data']): break
            except (urllib.error.URLError,TimeoutError): pass
            time.sleep(1)
        else: raise RuntimeError('Readiness timeout')
        print('READY',model,flush=True)
        tok=AutoTokenizer.from_pretrained('/home/u2601112013/models/Qwen3.5-'+model,local_files_only=True)
        ids=tok.encode(' hello'*600,add_special_tokens=False)
        for length in [128,512]:
            prompt=(tok.decode(ids[:length],skip_special_tokens=False),length)
            assert len(tok.encode(prompt[0],add_special_tokens=False))==length
            for c in [1,2]:
                measured=[]; windows=[]; overlaps=[]
                for rep in range(7):
                    phase=f'{model}_L{length}_C{c}_'+('warmup' if rep<2 else 'measure')+str(rep)
                    rows,wall,overlap=batch(model,prompt,c)
                    record={'phase':phase,'model':model,'length':length,'concurrency':c,'repeat':rep,'warmup':rep<2,'wall_s':wall,'token_stream_overlap':overlap,'start_skew_ms':(max(r['start'] for r in rows)-min(r['start'] for r in rows))*1000,'requests':rows}
                    with (OUT/'rounds.jsonl').open('a') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if rep>=2:
                        measured+=rows; windows.append(wall); overlaps.append(overlap)
                    print('ROUND',phase,'seconds',round(wall,2),'overlap',overlap,flush=True)
                cell={'model':model,'tp':2,'input_tokens':length,'concurrency':c,'requests':len(measured),'output_tokens':sum(r['usage']['completion_tokens'] for r in measured),'throughput_tps':sum(r['usage']['completion_tokens'] for r in measured)/sum(windows),'overlap_all_rounds':all(overlaps),'output_variants':len(set(r['text_sha256'] for r in measured)),'content_chunks_min':min(r['content_chunks'] for r in measured),'content_chunks_max':max(r['content_chunks'] for r in measured)}
                for metric in ['e2e_ms','ttft_ms','tpot_ms']:
                    vals=[r[metric] for r in measured]
                    cell[metric+'_p50']=statistics.median(vals); cell[metric+'_p95']=percentile(vals,.95)
                all_cells.append(cell); save(OUT/'performance_matrix.json',all_cells)
        print('MODEL_COMPLETE',model,flush=True)
    except Exception as e:
        save(OUT/(model+'_error.json'),{'error':repr(e),'phase':phase})
        print('MODEL_FAILED',model,repr(e),flush=True)
        raise
    finally:
        phase=model+'_shutdown'; stop.set()
        if proc:
            sampler.join(timeout=20)
            if proc.poll() is None:
                assert os.stat('/proc/'+str(proc.pid)).st_uid==os.getuid()
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=15)
            log.close()
        (OUT/(model+'_samples.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in samples))
        save(OUT/(model+'_shutdown.json'),{'returncode':proc.returncode if proc else None,'after':snap(-1)})
save(OUT/'complete.json',{'cells':len(all_cells),'formal_requests':sum(x['requests'] for x in all_cells),'success':len(all_cells)==16})
print('ALL_COMPLETE',str(OUT),flush=True)
