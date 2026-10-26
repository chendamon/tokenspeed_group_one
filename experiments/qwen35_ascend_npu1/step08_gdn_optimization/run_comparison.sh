#!/usr/bin/env bash

set -euo pipefail

mode="${1:?usage: run_comparison.sh MODE PORT}"
port="${2:?usage: run_comparison.sh MODE PORT}"
here=/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step08_gdn_optimization
python_bin=/home/u2601112013/tokenspeed/.venv/bin/python
server_log="$here/${mode}_server.log"
client_json="$here/${mode}_client_results.json"
npu_jsonl="$here/${mode}_npu_samples.jsonl"

server_pid=
sampler_pid=
cleanup() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ "$mode" == *fallback* ]]; then
  export TOKENSPEED_ASCEND_GDN_NATIVE=0
else
  export TOKENSPEED_ASCEND_GDN_NATIVE=1
fi

bash "$here/run_server.sh" "$port" >"$server_log" 2>&1 &
server_pid=$!
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 100 "$server_log"
    exit 1
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${port}/health" >/dev/null

"$python_bin" "$here/sample_npu.py" --output "$npu_jsonl" --samples 2400 &
sampler_pid=$!
"$python_bin" "$here/benchmark.py" \
  --port "$port" \
  --repeats 3 \
  --output "$client_json" | tee "$here/${mode}_benchmark.log"

kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true
sampler_pid=

echo "mode=$mode server_pid=$server_pid"
