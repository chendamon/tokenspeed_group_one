#!/usr/bin/env bash
set -euo pipefail
root=/home/u2601112013/tokenspeed_group_one
cd "$root"
set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; set -u
export VIRTUAL_ENV="$root/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export ASCEND_RT_VISIBLE_DEVICES=2 ASCEND_VISIBLE_DEVICES=2
export PYTHONPATH="$root/python:$root/tokenspeed-kernel/python:$root/tokenspeed-kernel-npu/python"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$VIRTUAL_ENV/bin/ts" serve --model /home/u2601112013/models/Qwen3.5-0.8B \
  --host 127.0.0.1 --port 38172 --language-model-only --served-model-name qwen35-0.8b \
  --device npu --tensor-parallel-size 1 --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype auto --attention-backend mha --sampling-backend greedy \
  --gpu-memory-utilization 0.2 --enforce-eager --disable-prefill-graph --disable-pdl \
  --disable-autotune --disable-overlap-schedule --disable-prefix-caching --disable-kvstore \
  --enable-log-request-stats --no-enable-log-requests --decode-log-interval 1 \
  --retraction-snapshot-ratio 0 --max-model-len 512 --max-num-seqs 1 --max-total-tokens 4096 --chunked-prefill-size 512 --prefix-granularity 64
