# 第9步：Qwen3.5-0.8B Ascend 910C 单 Chip 最终验证与实验报告

## 1. 结论摘要

本组已在 Ascend 910C 的单个目标 chip 上完成 Qwen3.5-0.8B 的 TokenSpeed
适配、连续推理、分层正确性验证和性能测量。

- 目标设备固定为 `NPU ID 1 / Chip ID 0 / Phy-ID 2`；通过
  `ASCEND_RT_VISIBLE_DEVICES=2`、`ASCEND_VISIBLE_DEVICES=2` 暴露为 PyTorch
  逻辑设备 `npu:0`。
- 模型权重、主要计算、KV Cache 和 Gated DeltaNet 状态均位于 NPU；CPU 负责
  HTTP 路由、调度、tokenization 和少量元数据处理。
- 模型以 BF16 权重、BF16 GDN 状态、MHA、greedy、eager 模式成功启动，当前
  OpenAI 兼容服务监听 `127.0.0.1:30317`。
- Prefill、Decode、连续 32-token 推理、重复请求状态隔离和多输入长度服务请求均通过。
- 核心优化复用了 `torch_npu.npu_recurrent_gated_delta_rule`，没有重新实现已有
  Ascend 算子。相对第7步正确性优先基线，32-token Prefill 降低约 70%，512-token
  Prefill 降低约 88.5%；Decode 仍约 53--54 ms/token，是下一阶段主要瓶颈。

## 2. 范围、代码边界与设备约束

- 项目目录：`/home/u2601112013/tokenspeed_group_one`
- 模型目录：`/home/u2601112013/models/Qwen3.5-0.8B`
- 所有代码改动、实验脚本、日志和报告均位于项目目录内。
- 所有 NPU 实验仅使用 `NPU ID 1 / Chip ID 0 / Phy-ID 2`。
- 未参考服务器上的历史参考实现；实现基于当前仓库、模型配置、可信张量递推参考和
  已安装 Ascend/PyTorch API。
- 复用 TokenSpeed 原有模型结构、权重加载器、调度器、状态池、KV Cache、采样器、
  HTTP/SMG 服务和算子分发框架，只修改不兼容或缺失的 Ascend 路径。

## 3. 环境与启动方式

### 3.1 软件环境

| 项目 | 版本/值 |
|---|---|
| OS | openEuler 24.03 LTS-SP3, aarch64 |
| Kernel | 5.10.0-216.0.0.115.oe2203sp4.aarch64 |
| NPU 管理工具 | npu-smi 25.5.2 |
| CANN | 9.0.0 |
| Python | 3.11.16 |
| PyTorch | 2.9.0+cpu |
| torch-npu | 2.9.0 |
| Transformers | 5.12.0 |
| Triton / Triton-Ascend | 3.5.0 / 3.2.1 |
| TokenSpeed | 0.1.1 |
| tokenspeed-kernel-npu | 0.1.0 |
| tokenspeed-scheduler | 0.1.40 |

基础环境先使用已配备的 nanoGPT 在目标逻辑设备执行 BF16 forward，输出 logits
位于 `npu:0`，验证 torch-npu、模型权重和基础算子可用。

### 3.2 当前服务启动

```bash
cd /home/u2601112013/tokenspeed_group_one
export TOKENSPEED_ASCEND_GDN_NATIVE=1
bash experiments/qwen35_ascend_npu1/step08_gdn_optimization/run_server.sh 30317
```

脚本固定物理设备 2，配置 BF16 权重和 BF16 GDN 状态，并设置
`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。为保证结果
可解释，关闭 Prefill graph、PDL、autotune、overlap schedule、prefix cache 和
KVStore。

健康检查与请求示例：

```bash
curl -f http://127.0.0.1:30317/health
curl -sS http://127.0.0.1:30317/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-0.8b","prompt":"The capital of France is","max_tokens":32,"temperature":0,"ignore_eos":true}'
```

## 4. 模型结构和算子支持矩阵

模型文本配置为：hidden size 1024、24 层、词表 248320；每 4 层一个 full
attention，因此共有 6 个 MHA 层和 18 个 Gated DeltaNet/linear-attention 层。
MHA 使用 8 个 query heads、2 个 KV heads、head dim 256；GDN 使用 16 个 value
heads，key/value head dim 均为 128。

| 模块 | 主要输入输出形状（单请求，T 为本轮 token 数） | 数据类型 | 实现来源 | Ascend 状态 | 验证方法 |
|---|---|---|---|---|---|
| Token embedding / LM head | `[T] -> [T,1024]`; `[T,1024] -> [T,248320]` | BF16 | 原 TokenSpeed/PyTorch | 直接复用 | 端到端生成、HF token 对照 |
| Gemma RMSNorm | `[T,1024] -> [T,1024]` | BF16，FP32 累加 | 原实现的 NPU 兼容分支 | 已适配 | NPU 对 FP32 参考，`2e-2` 判据 |
| Gated RMSNorm | `[T,16,128]` + gate -> 同形状 | BF16，FP32 归一化 | 原模块 + NPU 张量实现 | 已适配 | 典型 17-token 实际形状对照 |
| Causal Conv1d | packed token branch与卷积状态 -> 同 branch 形状 | BF16 | 仓库已有 causal-conv/Ascend 路径 | 直接复用 | import/isolated probe、模型 Prefill/Decode |
| MHA | Q `[T,8,256]`, K/V `[T,2,256]` -> `[T,8,256]` | BF16 | TokenSpeed MHA + Ascend backend | 已适配/复用 | MHA 专项与 backend 综合回归 |
| GDN Prefill | Q/K/V `[1,T,16,128]`; state `[S,16,128,128]` -> output `[1,T,16,128]` + final state | BF16 原生；FP32 回退 | `torch_npu.npu_recurrent_gated_delta_rule` | 新增 Ascend 分发与封装 | 长度 1/2/8/9/17、变长、padding、跨 chunk 对照 |
| GDN Decode | Q/K/V `[B,1,16,128]`; paged state `[P,16,128,128]` -> output `[B,1,16,128]` | BF16 原生；FP32 回退 | 同上 | 新增读写页安全更新 | 分离读写页、padding、多步与请求隔离 |
| Sampling offset | cache frontier + input lengths -> sample position | INT32 | 原调度路径 | 修复 torch-npu `index_copy` 源张量变更 | 持久 frontier 不变测试 |
| Greedy sampling | logits -> token id | BF16/FP32 logits | 原 TokenSpeed | 直接复用 | 固定配置多次端到端生成 |

## 5. 关键计算路径：Gated DeltaNet

### 5.1 调用路径

```text
Qwen3_5 decoder layer
  -> linear-attention block
  -> projection + causal convolution
  -> Q/K normalization and gate/beta construction
  -> tokenspeed_kernel GDN dispatcher
  -> Ascend backend registration
  -> tokenspeed_kernel_npu.ops.gdn
  -> torch_npu.npu_recurrent_gated_delta_rule
  -> paged GDN state pool update
```

### 5.2 布局转换和 Prefill 状态更新

TokenSpeed 上层保持 packed `[1,T,H,D]` 布局和 K-last 状态池
`[state_page,H,Dv,Dk]`。Ascend 原生算子只接受 BF16，且单次序列长度不超过 8，
因此 Prefill 按最多 8 token 分段。每段使用同一状态页索引，上一段写回的状态直接成为
下一段输入；长度 9 和 17 的测试专门覆盖跨段边界。`cu_seqlens` 仅用于取得请求边界；
主要矩阵计算和状态更新在 NPU 原生算子中执行。

### 5.3 Decode 状态页更新与隔离

Decode 允许读状态页和写状态页不同。实现先根据 read indices gather 所有源状态，再
通过 `index_copy_` 写入目标页，避免调度器复用页面时一个请求提前覆盖另一个请求的
输入状态；随后融合算子在目标页原位执行 decay、prediction、delta update 和输出矩阵
乘。padding/无效页返回零输出且不写状态。每个请求使用独立调度状态页，结束后由原
TokenSpeed 生命周期管理器回收。

### 5.4 设备同步

正常推理不插入逐 token 主机同步。数值测试和微基准只在计时边界调用 NPU synchronize。
服务端 RequestStats 由调度器时间戳记录；NPU 利用率由独立 `npu-smi` 采样器采集。

## 6. 适配内容

### 6.1 直接复用

- Qwen3.5 模型定义、参数命名和 safetensors 权重加载。
- TokenSpeed Engine、scheduler、KV/state cache arena、请求生命周期和 greedy sampling。
- MHA 的总体接口和已存在的 Ascend attention 能力。
- causal convolution、线性层、embedding、LM head 和 HTTP/SMG 服务框架。
- `torch_npu` 的原生 recurrent gated delta rule 算子。

### 6.2 修改或新增

- `python/tokenspeed/runtime/models/qwen3_5.py`：避免在 NPU 路径构造
  `torch.cuda.Stream`。
- `python/tokenspeed/runtime/layers/layernorm.py` 与
  `linear/layernorm_gated.py`：补齐 Gemma RMSNorm、gated RMSNorm 的 NPU 行为。
- `python/tokenspeed/runtime/execution/model_executor.py`：显式 clone 后执行
  `index_copy_`，防止 torch-npu 2.9 改写持久 cache frontier。
- `tokenspeed-kernel-npu/.../ops/mha.py`：补齐项目 MHA 接口所需的 Ascend 路径。
- `tokenspeed-kernel/.../gdn/ascend.py` 与 dispatcher 注册：让 NPU 选择 Ascend GDN。
- `tokenspeed-kernel-npu/.../ops/gdn.py`：BF16 原生 Prefill/Decode；FP32 和未验证
  形状保留正确性优先回退；支持 `TOKENSPEED_ASCEND_GDN_NATIVE=0/1` A/B 开关。
- 对应测试覆盖真实 Qwen 形状、边界形状、读写状态页和 backend 配置。

## 7. 分层正确性验证

### 7.1 参考实现和误差判据

GDN 可信参考按 token 使用 FP32 张量运算：state decay、`k @ state` prediction、
beta correction、外积状态更新和 `q @ state` 输出。输入和持久状态使用 BF16，与原生
接口一致。BF16 不是逐位确定，综合单步舍入、持久状态和跨 8-token 分段误差，采用：

- 输出：`rtol=2e-2, atol=2e-2`
- 状态：`rtol=2e-2, atol=3e-2`

实际最大输出绝对误差为 `1.37e-4`，最大状态绝对误差为 `2.959e-3`，明显低于判据。

| 序列长度 | 输出最大绝对误差 | 状态最大绝对误差 |
|---:|---:|---:|
| 1 | 0.000060 | 0.001919 |
| 2 | 0.000060 | 0.001918 |
| 8 | 0.000120 | 0.001898 |
| 9 | 0.000117 | 0.002018 |
| 17 | 0.000137 | 0.002959 |

### 7.2 自动化测试

- GDN 专项：7/7 通过。
- GDN + MHA + NPU attention backend 综合回归：22/22 通过。
- Python `py_compile` 通过，`git diff --check` 通过。
- 环境未安装 `ruff`，未将缺失的静态工具误报为测试通过。

### 7.3 端到端 token 对照

固定输入 token `[760,6511,314,9338,369]`，Hugging Face CPU 参考前 8 个新 token
为 `[11751,13,198,760,6511,314,9338,369]`。TokenSpeed NPU 第一次和第二次请求均
逐 token 完全一致；连续 32-token 请求成功，`finish_reason=length`。

### 7.4 状态更新与请求隔离

- 算子层：覆盖 Prefill 多段连续状态、Decode 分离读写页、无效页和 padding。
- Engine 层：相同请求执行两次，前 8 token 均与 CPU 参考完全一致。
- HTTP 层：A 请求生成 32 token，中间插入不同的 B 请求，再次执行 A；两次 A 文本
  完全一致，B 文本不同，且三次 `cached_tokens=0`。证明状态未在请求间泄漏。

## 8. 性能测量方法

- 单请求、batch size 1、`temperature=0`、`ignore_eos=true`。
- 输入长度 32、512 tokens；生成长度 16、64 tokens，共 4 组。
- 每组 1 次预热、5 次正式测量，取中位数。
- 关闭 prefix cache，服务日志确认实际 prompt tokens 且 `cache_tokens=0`。
- Prefill、TTFT、total 来自原生 RequestStats。
- Decode TPOT = `(total_ms - ttft_ms) / (output_tokens - 1)`。
- HBM 和利用率由 `npu-smi` 在相同请求时间窗口采样。

## 9. 最终服务级结果

| 输入 | 输出 | Prefill (ms) | TTFT (ms) | Decode TPOT (ms/token) | Decode token/s | Total (ms) | 进程 HBM峰值 (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 80.89 | 81.32 | 51.254 | 19.511 | 850.53 | 2783 |
| 32 | 64 | 84.27 | 84.67 | 54.205 | 18.448 | 3500.47 | 2783 |
| 512 | 16 | 411.98 | 412.44 | 53.897 | 18.554 | 1218.29 | 2784 |
| 512 | 64 | 409.34 | 409.75 | 53.255 | 18.777 | 3764.81 | 2784 |

客户端与服务端差值约 2.4--2.8 ms，HTTP 路由不是主要瓶颈。HBM 峰值与第7步
基线相同，说明优化没有通过增加显著常驻显存换取速度。

## 10. 优化前后对比

| 输入×输出 | Prefill 基线→优化 (ms) | Prefill 降幅/加速 | Decode TPOT 基线→优化 | Total 降幅/加速 |
|---|---:|---:|---:|---:|
| 32×16 | 280.30 → 80.89 | 71.14% / 3.465x | 52.998 → 51.254 | 21.05% / 1.267x |
| 32×64 | 283.02 → 84.27 | 70.23% / 3.358x | 54.806 → 54.205 | 6.26% / 1.067x |
| 512×16 | 3578.27 → 411.98 | 88.49% / 8.686x | 53.763 → 53.897 | 72.22% / 3.600x |
| 512×64 | 3576.03 → 409.34 | 88.55% / 8.736x | 54.989 → 53.255 | 46.86% / 1.882x |

单层隔离微基准显示 32/512-token GDN Prefill 分别获得 17.48x/21.34x；端到端
加速较小是因为还包含投影、卷积、norm、6 层 MHA、调度和采样。长输入 AICore 峰值由
基线约 25% 提升到 54--63%，支持“逐 token 主机发射瓶颈已显著缓解”的判断。

## 11. 瓶颈分析与下一优化点

### 11.1 已解决的主要瓶颈

原 Ascend GDN Prefill 在 Python 中逐 token 执行多个 NPU 张量算子，并把
`cu_seqlens` 同步到 CPU，造成大量小 kernel、主机发射和同步。改为最多 8 token 的
原生 recurrent 算子后，长输入 Prefill 降低约 88.5%，证明判断正确。

### 11.2 当前主要瓶颈

Decode TPOT 在优化前后都约 53--55 ms/token，输出越长，Total 越由 Decode 累积主导。
原因包括 18 个 GDN 层逐 token 的状态页 gather/copy、Q/K normalization、卷积状态、
层间小算子和主机调度。单层 Decode 原生路径仅 1.18x，加上模型其他部分后端到端
改善只有约 -0.25% 到 3.29%。

因此最值得继续优化的是：

1. 合并 Decode 的状态页 gather/copy 与 recurrent update，减少中间张量；
2. 融合 Q/K L2 normalization、gate/beta 计算与 GDN；
3. 在保持状态隔离测试通过后评估 NPU graph 或 overlap schedule；
4. 使用设备 profiler 对 18 个 GDN 层、causal conv 和 norm 的 kernel launch 占比做
   精细归因。

## 12. 最大适配障碍及处理

1. CUDA 假设泄漏到模型代码：NPU 路径创建 `torch.cuda.Stream`。
   通过平台感知的 device stream/禁用 CUDA 专属分支修复。
2. GDN 缺少 Ascend backend：增加 dispatcher 和 NPU 实现，同时保留原有状态布局。
3. 原生 GDN 只接受 BF16 且单段 `L<=8`：显式选择 BF16 state，Prefill 分段并跨段
   复用状态；FP32/非目标形状安全回退。
4. torch-npu 2.9 的 nominally out-of-place `index_copy` 改写源 tensor：先 clone 再
   `index_copy_`，并增加持久 frontier 不变测试。
5. 历史服务进程形成大量 PPID 1 僵尸，旧 `pids.max=40000` 导致 Gloo/SMG 无法创建
   线程。管理员将上限恢复到 2000000 后，HTTP 服务、warmup 和完整基准均成功。该问题
   属于容器进程回收，不是 NPU 算子失败。

## 13. 实验步骤回顾

1. 环境、nanoGPT 和初始 TokenSpeed 复现。
2. Qwen3.5 计算路径、算子矩阵和 Ascend 注册能力梳理。
3. causal conv、RMSNorm、MHA 等基础模块探测。
4. CUDA 假设与 NPU norm 的最小兼容修改，打通真实权重 Prefill。
5. GDN Ascend dispatcher、正确性回退和初始连续推理。
6. Decode 状态更新、采样 frontier、CPU 参考和请求隔离修复。
7. 2×2 服务级性能基线及瓶颈定位。
8. 接入 torch-npu 原生 GDN、分层回归和单层微基准。
9. 优化后服务级复测、HTTP 状态隔离和最终报告归档。

## 14. 第9步产物

目录：`experiments/qwen35_ascend_npu1/step09_final_validation/`

- `STEP09_REPORT.md`：本报告。
- `client_results.json`：20 次正式请求和客户端时间。
- `server.log`：服务配置、内存摘要和 RequestStats。
- `npu_samples.jsonl`：目标 NPU 的资源采样。
- `summary.json`：优化后服务级汇总。
- `comparison.json`：第7步与第9步逐项比较。
- `benchmark.log`、`summary.log`、`sampler.log`：执行日志。
- `repeat_a.json`、`different.json`、`repeat_b.json`、`e2e_validation.log`：HTTP
  请求隔离证据。

## 15. 最终验收判断

- 单 chip 启动 Qwen3.5-0.8B：通过。
- Prefill、Decode、完整连续推理：通过。
- 关键算子/模块数值测试：通过。
- 多输入长度与多生成长度端到端测试：通过。
- 多步状态更新和请求间隔离：通过。
- 首 Token、Decode TPOT、HBM 基线：已测得并可复现。
- 主要计算在 NPU：满足；CPU 未替代主要模型计算。
- 当前最值得优化的计算环节：Decode GDN 状态搬运、norm/gate 和小算子调度。

因此，本阶段总体目标“在 Ascend 910C 的单个 chip 上正确运行 Qwen3.5-0.8B”已完成。
