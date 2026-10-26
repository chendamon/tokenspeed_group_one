#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -euo pipefail

repo_root=/home/u2601112013/tokenspeed_group_one
ts_bin=/home/u2601112013/tokenspeed/.venv/bin/ts
model_path=/home/u2601112013/models/Qwen3.5-0.8B
port="${1:?usage: run_server.sh PORT}"

cd "$repo_root"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

# NPU ID 1 / Chip ID 0 maps to physical device 2. It is exposed as logical npu:0.
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_VISIBLE_DEVICES=2
export PYTHONPATH="$repo_root/python:$repo_root/tokenspeed-kernel/python:$repo_root/tokenspeed-kernel-npu/python"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec "$ts_bin" serve --model "$model_path" \
  --host 127.0.0.1 \
  --port "$port" \
  --language-model-only \
  --served-model-name qwen35-0.8b \
  --device npu \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --kv-cache-dtype auto \
  --attention-backend mha \
  --sampling-backend greedy \
  --enforce-eager \
  --disable-prefill-graph \
  --disable-pdl \
  --disable-autotune \
  --disable-overlap-schedule \
  --disable-prefix-caching \
  --disable-kvstore \
  --enable-log-request-stats \
  --no-enable-log-requests \
  --decode-log-interval 1 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-total-tokens 16384 \
  --chunked-prefill-size 8192 \
  --prefix-granularity 64
