import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path('/home/u2601112013/tokenspeed_group_one')
work = root / 'experiments/qwen35_ascend_npu1/step13_model_correctness'
python = root / '.venv/bin/python'
cases = [('0.8B', 2), ('0.8B', 1), ('9B', 2), ('9B', 1), ('27B', 2)]
for model, tp in cases:
    tag = f'{model}_tp{tp}'
    snapshot = subprocess.check_output(['npu-smi', 'info'], text=True)
    (work / f'npu_before_{tag}.log').write_text(snapshot)
    matches = re.findall(r'\|\s*[01]\s+([23])\s*\|[^|]*\|\s*\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)', snapshot)
    free = {int(phy): int(total) - int(used) for phy, used, total in matches}
    threshold = 37000 if model == '27B' else (25000 if model == '9B' and tp == 1 else 12000)
    required = [2, 3] if tp == 2 else [2]
    if any(free.get(p, 0) < threshold for p in required):
        print(f'SKIP {tag}: insufficient or unverified free HBM {free}; threshold {threshold} MB', flush=True)
        continue
    try:
        with urllib.request.urlopen('http://127.0.0.1:38172/health', timeout=1) as resp:
            print(f'STOP {tag}: port already serving; refusing to interfere', flush=True)
            break
    except (urllib.error.URLError, TimeoutError):
        pass
    print(f'START {tag} free_hbm_mb={free}', flush=True)
    server_log = (work / f'server_{tag}.log').open('w')
    proc = subprocess.Popen(['bash', str(work / 'run_server.sh'), model, str(tp)], cwd=root, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        ready = False
        for _ in range(300):
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen('http://127.0.0.1:38172/health', timeout=2) as resp:
                    healthy = resp.status == 200
                if healthy:
                    with urllib.request.urlopen('http://127.0.0.1:38172/v1/models', timeout=2) as resp:
                        models = json.load(resp).get('data', [])
                    ready = any(item.get('id') == f'qwen35-{model}' for item in models)
                if ready:
                    break
                time.sleep(1)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        if not ready:
            print(f'FAIL {tag} startup rc={proc.poll()}', flush=True)
            continue
        (work / f'npu_{tag}.log').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))
        result = subprocess.run([str(python), str(work / 'requests.py'), model, str(tp), str(work / f'requests_{tag}.json')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=400)
        (work / f'client_{tag}.log').write_text(result.stdout)
        print(f'{"PASS" if result.returncode == 0 else "FAIL"} {tag} request_rc={result.returncode}', flush=True)
    except (subprocess.TimeoutExpired, Exception) as exc:
        print(f'FAIL {tag}: {type(exc).__name__}: {exc}', flush=True)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
        server_log.close()
        time.sleep(3)
print('CASES_FINISHED', flush=True)
