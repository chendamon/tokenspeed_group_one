import csv,json,re,statistics,os
from pathlib import Path
BASE=Path('/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step18_tp2_matrix')
out=Path((BASE/'latest_run.txt').read_text().strip())
assert json.loads((out/'complete.json').read_text())['success']
cells=json.loads((out/'performance_matrix.json').read_text())
rounds=[json.loads(l) for l in (out/'rounds.jsonl').read_text().splitlines()]
hbm={}; validation={}
for m in ['0.8B','4B','9B','27B']:
    samples=[json.loads(l) for l in (out/(m+'_samples.jsonl')).read_text().splitlines()]
    valid=[s for s in samples if 'devices' in s and ('measure' in s['phase'] or 'warmup' in s['phase'])]
    text=(out/(m+'_server.log')).read_text()
    peaks=[max(s['devices'][str(c)]['own_mb'] for s in valid) for c in [0,1]]
    total=[max(s['devices'][str(c)]['used_mb'] for s in valid) for c in [0,1]]
    ranks={str(c):sorted({p['pid'] for s in valid for p in s['devices'][str(c)]['processes'] if p['owned'] and 'scheduler_tp'+str(c) in p.get('args','')}) for c in [0,1]}
    own={p['pid'] for s in samples if 'devices' in s for d in s['devices'].values() for p in d['processes'] if p['owned']}
    after=json.loads((out/(m+'_shutdown.json')).read_text())
    remaining=[p for d in after['after']['devices'].values() for p in d['processes'] if p['pid'] in own]
    hbm[m]={'own_peak_mb':peaks,'device_total_peak_mb':total,'samples':len(valid),'rank_pids':ranks}
    validation[m]={'both_rank_logs':all('TP RANK '+str(c) in text for c in [0,1]),'rank_pids':ranks,'cache_tokens_values':sorted(set(map(int,re.findall(r'cache_tokens=(\d+)',text)))),'own_npu_processes_after_shutdown':remaining,'sample_errors':sum('error' in s for s in samples),'concurrency2_engine_seen':'#running-req: 2' in text}
    for cell in cells:
        if cell['model']!=m: continue
        group=[r for r in rounds if not r['warmup'] and r['model']==m and r['length']==cell['input_tokens'] and r['concurrency']==cell['concurrency']]
        cell['round_tps']=[sum(x['usage']['completion_tokens'] for x in r['requests'])/r['wall_s'] for r in group]
        cell['round_tps_stdev']=statistics.stdev(cell['round_tps'])
        cell['start_skew_ms_max']=max(r['start_skew_ms'] for r in group)
        cell_samples=[s for s in valid if s['phase'].startswith(m+'_L'+str(cell['input_tokens'])+'_C'+str(cell['concurrency'])+'_')]
        cell['chip0_own_peak_mb']=max(s['devices']['0']['own_mb'] for s in cell_samples)
        cell['chip1_own_peak_mb']=max(s['devices']['1']['own_mb'] for s in cell_samples)
requests=[{**{k:r[k] for k in ['model','length','concurrency','repeat','warmup']},**{k:x[k] for k in ['e2e_ms','ttft_ms','tpot_ms','content_chunks','text_sha256']}} for r in rounds for x in r['requests']]
validation['overall']={'cells':len(cells),'formal_requests':sum(not r['warmup'] for r in requests),'warmup_requests':sum(r['warmup'] for r in requests),'all_64_output_tokens':all(x['usage']['completion_tokens']==64 for r in rounds for x in r['requests']),'all_64_content_chunks':all(x['content_chunks']==64 for r in rounds for x in r['requests']),'all_concurrent_streams_overlap':all(r['token_stream_overlap'] for r in rounds),'all_5_measurement_rounds':all(sum(not r['warmup'] and r['model']==c['model'] and r['length']==c['input_tokens'] and r['concurrency']==c['concurrency'] for r in rounds)==5 for c in cells)}
for name,obj in [('performance_matrix.json',cells),('hbm_summary.json',hbm),('final_validation.json',validation)]:
    (out/name).write_text(json.dumps(obj,indent=2)+'\n')
for name,data in [('performance_matrix.csv',cells),('request_metrics.csv',requests)]:
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
report=['# NPU1 四模型双 chip 性能实验报告','', '日期：2026-11-04；TP=2；NPU1 chip0/chip1，物理 ID 2/3。','',
'## 测量方法','', '四模型依次驻留，BF16、greedy、eager，关闭 prefix cache 和 overlap schedule。输入精确为 128/512 token；并发 1/2；固定输出 64 token；每格 2 轮预热、5 轮正式测量。服务 max-num-seqs=2，max-total-tokens=16384，max-model-len=1024，chunked-prefill-size=1024。启动器继承本项目 step16，仅修改 HTTP 端口和最大并发。','',
'吞吐=正式轮实际输出 token 总和/各轮 wall time 总和；E2E 从发请求至 SSE 流结束；TTFT 从发请求至首个非空内容块；TPOT=(末内容块到达-首内容块到达)/63。所有请求均核验 64 个内容块及服务端 completion_tokens=64；原始 SSE 数据及单调时间保留在 rounds.jsonl。p50/p95 为每格 5 或 10 请求的经验分位数，p95 样本较少。','',
'## 性能矩阵','',
'|模型|输入|并发|吞吐 token/s|E2E p50/p95 ms|TTFT p50/p95 ms|TPOT p50/p95 ms|',
'|---|---:|---:|---:|---:|---:|---:|']
for c in cells:
    report.append('|'+ '|'.join([c['model'],str(c['input_tokens']),str(c['concurrency']),f"{c['throughput_tps']:.3f}"]+[f"{c[k+'_p50']:.1f} / {c[k+'_p95']:.1f}" for k in ['e2e_ms','ttft_ms','tpot_ms']])+'|')
report+=['','## HBM 与双 rank','', '以下为每模型预热与测量期间采样峰值，单位为 npu-smi 报告的 MB；采样间隔约 1.5 秒加查询开销，不能保证捕获瞬时峰值。实验进程值包含自有辅助上下文；设备总值包含其他用户。','',
'|模型|本实验 chip0/chip1 MB|设备总量 chip0/chip1 MB|有效采样数|rank0/rank1 PID|','|---|---:|---:|---:|---|']
for m,x in hbm.items():
    report.append(f"|{m}|{x['own_peak_mb'][0]} / {x['own_peak_mb'][1]}|{x['device_total_peak_mb'][0]} / {x['device_total_peak_mb'][1]}|{x['samples']}|{x['rank_pids']['0']} / {x['rank_pids']['1']}|")
report+=['','## 验证与解释','',json.dumps(validation['overall'],ensure_ascii=False),'',
'各模型双 rank 日志、同一卡物理设备映射、实验进程树和 HBM 样本保留；每模型结束后核验自有 NPU PID 未残留。全部并发 2 轮次均核验 token 流重叠，并检查引擎 running-req=2。输出序列差异通过 output_variants 和 text_sha256 单独记录；完成工作量不等于模型正确性已通过。','',
'本轮为共享设备测量，后台 u2601213438 服务常驻，结果不能作为独占卡基准。四模型固定启动顺序，五轮采样仍不足以消除共享干扰；模型尺寸不保证吞吐严格单调。该轮仅 TP=2，无相同新配置 TP=1 对照，因此不从本轮宣称 TP 加速比。与历史 32-token、并发 1/4、非流式结果的工作量不同，应分开报告。','',
'## 原始证据','',str(out),'','performance_matrix.json/csv、request_metrics.csv、rounds.jsonl、hbm_summary.json、final_validation.json、各模型 baseline/server/launcher/samples/shutdown 文件和 complete.json。早期环境和输入格式失败记录保留在 step18 目录及较早 run 子目录，均未纳入本轮矩阵。']
report+=['','## 结果分析','',
'并发从 1 增至 2 时，128-token 输入吞吐分别增加约 66.9%、70.7%、80.2%、66.5%（0.8B、4B、9B、27B）；512-token 输入分别增加约 58.6%、38.1%、36.6%、43.5%。吞吐提升同时伴随 TTFT 和端到端延迟增加，并非 TP 加速比。','',
'四模型在并发 2 的正式轮次中，两个流的实际重叠时长分别约为 3.77-3.92、5.32-5.82、5.11-5.28、9.82-10.88 秒。0.8B 的两档输入各有一轮较高 TTFT，128/C2 的轮吞吐约 16.12，而其他轮约 30.51-31.25 token/s；512/C2 类似。原始异常轮未剔除，聚合吞吐包含其耗时；现有数据不足以确定其因果。','',
'增加输入长度对 0.8B 的吞吐影响较小，对 4B/9B/27B 的 TTFT 影响更明显。各模型均关闭前缀缓存，cache_tokens=0，长输入实际工作量已由服务端用量验证。4B 与 9B 的吞吐排序接近，不能仅按参数规模推断差异原因。','',
'本实验显存随模型增大明显增加；27B 两个 chip 自有进程采样峰值为 28691/28355 MB，设备总峰值为 44342/43563 MB，分别低于单 chip 65536 MB 容量。性能矩阵 CSV 中另存逐配置采样峰值；报告 HBM 表为每模型全部四格的峰值。']
(out/'REPORT.md').write_text('\n'.join(report)+'\n')
print(json.dumps({'cells':cells,'hbm':hbm,'validation':validation},ensure_ascii=False,indent=2))
