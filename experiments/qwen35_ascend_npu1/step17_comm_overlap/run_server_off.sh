#!/usr/bin/env bash
set -euo pipefail
model=${1:?model size required}
tp=${2:?TP size required}
case "$model" in 0.8B|4B|9B|27B) ;; *) exit 2;; esac
case "$tp" in 1) visible=2;; 2) visible=2,3;; *) exit 2;; esac
root=/home/u2601112013/tokenspeed_group_one
cd "$root"
set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; set -u
export VIRTUAL_ENV="$root/.venv" PATH="$root/.venv/bin:$PATH"
unset TORCH_DEVICE_BACKEND_AUTOLOAD
export ASCEND_RT_VISIBLE_DEVICES="$visible" ASCEND_VISIBLE_DEVICES="$visible"
export PYTHONPATH="$root/python:$root/tokenspeed-kernel/python:$root/tokenspeed-kernel-npu/python"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$VIRTUAL_ENV/bin/ts" serve --model "/home/u2601112013/models/Qwen3.5-$model" \
  --host 127.0.0.1 --port 38172 --language-model-only --served-model-name "qwen35-$model" \
  --device npu --tensor-parallel-size "$tp" --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype auto --attention-backend mha --sampling-backend greedy \
  --gpu-memory-utilization 0.8 --enforce-eager --disable-prefill-graph --disable-pdl \
  --disable-autotune --disable-overlap-schedule --disable-prefix-caching --disable-kvstore \
  --enable-log-request-stats --no-enable-log-requests --decode-log-interval 16 \
  --retraction-snapshot-ratio 0 --max-model-len 1024 --max-num-seqs 4 --max-total-tokens 16384 --chunked-prefill-size 1024 --prefix-granularity 64
