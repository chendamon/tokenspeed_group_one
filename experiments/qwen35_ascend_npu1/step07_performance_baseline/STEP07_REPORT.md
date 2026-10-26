# 第7步：Qwen3.5-0.8B 单 Chip 性能基线

## 1. 目标和设备约束

- 目标设备：NPU ID 1 / Chip ID 0 / Phy-ID 2。
- 运行时可见设备：`ASCEND_RT_VISIBLE_DEVICES=2`、`ASCEND_VISIBLE_DEVICES=2`。
- PyTorch 逻辑设备：`npu:0`。
- 所有新增脚本和实验结果均位于
  `/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/step07_performance_baseline`。

## 2. 测量配置与方法

- 模型：Qwen3.5-0.8B。
- 权重精度：BF16；KV Cache：auto。
- Attention backend：MHA；Sampling backend：greedy。
- 单请求、batch size 1、`temperature=0`、`ignore_eos=true`。
- 使用 eager 执行，关闭 Prefill graph、overlap schedule、prefix caching 和 KVStore。
  关闭缓存是为了保证每次输入均执行完整 Prefill，避免重复请求命中前缀缓存。
- 输入长度：32、512 tokens；生成长度：16、64 tokens，组成 2×2 四组配置。
- 输入由 Qwen3.5 tokenizer 中稳定的一 token 文本片段构造；服务端日志确认实际
  `prompt_tokens` 分别为 32 和 512，且 `cache_tokens=0`。
- 每组先预热 1 次，再正式测量 5 次；结果取中位数，预热数据不计入基线。
- Prefill、TTFT、总耗时来自项目原生 `RequestStats`；客户端同时通过流式 API 测量。
- Decode TPOT 定义为 `(total_ms - ttft_ms) / (output_tokens - 1)`。
- HBM、AICore、AIVector、HBM 带宽和 NPU 利用率由
  `npu-smi info -t proc-mem/usages -i 1 -c 0` 周期采样。

## 3. 正式结果

| 输入 tokens | 输出 tokens | Prefill 中位数 (ms) | TTFT 中位数 (ms) | Decode TPOT (ms/token) | Decode (token/s) | 总耗时中位数 (ms) | Prefill 占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 280.30 | 280.75 | 53.00 | 18.87 | 1077.26 | 26.0% |
| 32 | 64 | 283.02 | 283.48 | 54.81 | 18.25 | 3734.20 | 7.6% |
| 512 | 16 | 3578.27 | 3578.71 | 53.76 | 18.60 | 4386.13 | 81.6% |
| 512 | 64 | 3576.03 | 3576.56 | 54.99 | 18.19 | 7084.96 | 50.5% |

队列等待中位数为 0.46–0.56 ms。客户端 TTFT 和总耗时仅比引擎侧高约
2.5–3.7 ms，因此 HTTP/gateway 不是当前主要开销。

每组第一次预热总耗时分别为 7559、3687、8242 和 6949 ms，包含新形状首次执行、
算子/JIT 冷启动等一次性成本，故不计入稳态基线。

## 4. HBM 与利用率

| 输入×输出 | 进程 HBM 峰值 (MB) | 卡级 HBM 使用率峰值 | AICore 平均/峰值 | AIVector 平均/峰值 | HBM 带宽平均/峰值 | NPU 利用率平均/峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 32×16 | 2783 | 8% | 9.3% / 11% | 6.0% / 9% | 2.0% / 2% | 23.0% / 31% |
| 32×64 | 2783 | 8% | 8.3% / 11% | 5.2% / 9% | 2.0% / 2% | 20.4% / 31% |
| 512×16 | 2784 | 8% | 9.9% / 25% | 9.0% / 24% | 0.5% / 2% | 23.0% / 31% |
| 512×64 | 2784 | 8% | 9.2% / 25% | 7.0% / 24% | 1.1% / 2% | 25.4% / 48% |

服务启动日志中的 driver view 占用为 2.65 GB，与 `npu-smi` 进程峰值约 2.72 GiB
一致。卡级 8% 还包括驱动/设备侧基线，不作为模型独占内存。由于 `npu-smi` 单次查询
本身耗时较高，各组窗口内分别取得 4、14、17、27 个样本；利用率数据用于瓶颈定性，
不作为细粒度 kernel 时间占比。

## 5. 瓶颈判断

1. 输入从 32 增加到 512 tokens（16 倍）时，Prefill 从约 280 ms 增加到约
   3578 ms（12.8 倍）；长输入、短输出场景中 Prefill 占总耗时 81.6%。
2. Decode TPOT 在四组中稳定为 53–55 ms/token，几乎不随 prompt 长度变化。
   输出从 16 增至 64 tokens 后，增加的耗时基本由连续 Decode 累积造成。
3. AICore 平均仅 8–10%，HBM 带宽峰值仅 2%，说明当前不是算力或 HBM 带宽饱和，
   而是大量小算子、主机发射、设备同步和数据布局/状态更新造成的执行碎片化。
4. 模型共 24 层，其中 18 层是 Gated DeltaNet/linear attention，只有 6 层是
   full attention。当前 Ascend `gdn_chunk_prefill` 是正确性优先实现：先将
   `cu_seqlens` 拷回 CPU，再在 Python 中逐 token 循环，每一步调用多个 NPU tensor
   算子；Decode 状态更新也由多个独立 tensor 算子组成。这与低硬件利用率、Prefill
   近线性增长以及约 54 ms/token 的 Decode 延迟一致。

因此当前最值得优化的是 Ascend GDN 路径，优先级为：

1. 将 Prefill 的逐 token Python scan 替换为设备侧 chunk/fused scan，消除
   `cu_seqlens.cpu()` 同步及逐 token 主机调度；
2. 融合 Decode 的 decay、prediction、state update 和 output matmul，减少每层 kernel
   launch 和中间张量；
3. 在上述正确性保持后，再评估 CUDA/NPU graph 或 overlap schedule，而不是先用不完整
   计算掩盖瓶颈。

## 6. 结果文件

- `summary.json`：机器可读的最终汇总。
- `client_results.json`：20 次正式请求的客户端流式计时和请求 ID。
- `server.log`：服务配置、原生 RequestStats 和设备内存摘要。
- `npu_samples.jsonl`：NPU 1 / Chip 0 的周期资源采样。
- `benchmark.log`、`summary.log`：实验过程和汇总日志。
- `run_server.sh`、`benchmark.py`、`sample_npu.py`、`analyze_results.py`：可复现实验脚本。
