# 第二阶段实验报告：通信子系统——Qwen3.5 多模型 Ascend 910C 双 Chip 张量并行与通信分析

> 报告日期：2026 年 10 月 28 日  
> 实验主机：`xeon6.pku-dasys.cn:2222`  
> 实验用户：`u2601112013`  
> 项目目录：`/home/u2601112013/tokenspeed_group_one`  
> 模型目录：`/home/u2601112013/models/Qwen3.5-{0.8B,4B,9B,27B}`  
> 目标设备：NPU1 的 chip0/chip1，对应物理设备 ID 2/3  
> 运行精度：BF16；仅文本、greedy、eager  
> 原始证据根目录：`experiments/qwen35_ascend_npu1/`

---

## 0. 成员贡献说明

本阶段沿用第一阶段的三人均等贡献口径。成员 A 主要负责服务器环境、双 chip 进程启动、设备映射与性能采样；成员 B 主要负责 TP 权重切分、典型层形状、HCCL 集合通信路径和 9B 分歧定位；成员 C 主要负责实验矩阵、统计口径、通信策略实验、证据审计和本报告拟制。三人共同参与方案讨论、代码审阅、结果复核和最终验收。

## 1. 实验摘要

本阶段在 TokenSpeed 既有并行框架和 HCCL 后端上，使用同一模型的两个 TP rank 分别绑定 NPU1 的物理设备 2、3，完成 Qwen3.5-0.8B、4B、9B 和 27B 的双 chip 连续推理验证。四个模型均不是两个独立副本：每个模型的列并行、行并行、词表并行权重和运行状态按 TP=2 切分，两个 rank 共同完成一次请求。

正确性方面，0.8B、4B、9B、27B 的 TP=2 均完成四条顺序连续请求；0.8B、4B 的 TP=1/TP=2 文本逐条一致，9B 的 TP=1/TP=2 四条均一致。与本地 Transformers BF16 参考相比，0.8B、4B 分别为 4/4、4/4 一致，9B 为 2/4，27B 为 3/4；9B 的差异集中在 CPU BF16 参考的并列最高分位置，不能简单归因于 TP 通信错误。27B 未运行 TP=1，因为当时单 chip 可用 HBM 约 48.5 GiB，而文本权重约 50.9 GiB，尚未计缓存和运行时开销。

性能方面，选取可运行 TP=1 的 4B，在输入 128/512 token、并发 1/4 下比较 TP=1 和 TP=2；每个配置预热后进行 2 轮、每轮 8 个请求。4B TP=2 在共享设备负载下相对 TP=1 的吞吐变化为 -11.3% 至 +2.2%，没有稳定加速，但将单 chip 实验进程显存从约 9.87 GB 降至约 6.51/6.18 GB。补充测试显示 27B TP=2 在 128/512 token、并发 1 下分别达到 4.486 和 2.410 token/s，两个 chip 的实验进程显存约 28.64/28.30 GB。

通信策略实验比较 `--disable-overlap-schedule` 与默认 overlap scheduler，固定 4B、TP=2、512-token、并发 1/4、BF16、prefix cache 关闭、每轮 8 请求。有效重跑中 overlap depth 0/1；并发 1 吞吐基本持平，并发 4 吞吐提高约 3.0%、p50 延迟下降约 3.0%，但 p95 仅下降约 1.8%。因此本环境下 overlap 主要隐藏 CPU 调度与 NPU/HCCL 等待，收益有限且不能替代真正的通信融合。第一批误删 prefix-cache 参数的结果已标记为 `confounded_*`，不纳入结论。

> 阶段结论：双 chip TP=2 已打通四个指定模型的权重切分、rank 启动、HCCL 通信和连续推理；4B 完成 TP=1/TP=2 性能矩阵，27B 完成 TP=2 性能与容量补充，并完成一项通信相关策略实验。由于服务器存在其他用户进程，性能结论限定为共享负载阶段性结果。

## 2. 阶段目标与完成情况

| 任务要求 | 实施内容 | 结果与证据 | 状态 |
|---|---|---|---|
| 双 chip 适配 | TP=2 启动器、物理设备映射、权重切分、HCCL 后端和 HTTP 推理流程 | `step13_model_correctness/` 中四模型 TP=2 日志；服务日志包含 global rank 0/1；物理设备 2/3 均捕获 rank 进程 | 已完成 |
| 并行正确性 | 0.8B/4B/9B/27B 四条连续请求；可运行模型做 TP=1/TP=2 对照；CPU BF16 参考诊断 | 0.8B、4B、9B、27B TP=2 均 4/4；0.8B/4B TP=1 与 TP=2 4/4；9B TP=1 与 TP=2 4/4；27B TP=1 因容量原因不运行 | 已完成，参考差异有边界说明 |
| 通信路径分析 | 典型全注意力层、GDN 层、稠密 MLP、KV/GDN state、Embedding/LM head 的切分和集合通信 | `step15_tp_layout/REPORT.md`、源码 `CommManager`、HCCL 调用点和运行时 rank JSONL | 已完成 |
| 代表模型性能 | 4B 的 2 输入长度 × 2 并发 × TP=1/TP=2；吞吐、p50/p95、HBM | `step16_tp_performance/REPORT.md`、结果 JSON、HBM 采样 | 已完成 |
| 大模型补充 | 27B TP=2 的性能和容量 | 128/512 token、并发 1，16/16 请求每配置成功；双 chip 约 28 GB/卡 | 已完成 |
| 通信相关实验 | overlap scheduler 开关对照 | `step17_comm_overlap/`，depth 0/1，cache_tokens 均为 0 | 已完成 |

## 3. 环境、设备与运行约束

### 3.1 设备映射

| 逻辑角色 | 物理设备 | TP=1 | TP=2 |
|---|---:|---|---|
| rank 0 / chip0 | NPU1 physical ID 2 | 使用 | 使用 |
| rank 1 / chip1 | NPU1 physical ID 3 | 不使用 | 使用 |

TP=1 启动设置 `ASCEND_RT_VISIBLE_DEVICES=2`、`ASCEND_VISIBLE_DEVICES=2`；TP=2 设置 `ASCEND_RT_VISIBLE_DEVICES=2,3`、`ASCEND_VISIBLE_DEVICES=2,3`。所有本阶段代码和实验产物均位于 `/home/u2601112013/tokenspeed_group_one` 内。

### 3.2 共同运行配置

- 权重：Qwen 官方本地权重；每个模型单独部署，不同时驻留；
- dtype：权重和 hidden BF16，Mamba SSM dtype 为 bfloat16；
- attention backend：MHA；sampling：greedy，temperature=0；
- `--language-model-only`、`--enforce-eager`；
- 关闭 prefix cache、KVStore、prefill graph、PDL、autotune；
- `max-model-len=1024`、`max-num-seqs=4`、`max-total-tokens=16384`、`chunked-prefill-size=1024`；
- 性能请求固定 `max_tokens=32`，输入 tokenizer 校验为精确 128、512 token；
- 采样器约每 1.5 秒执行一次 `npu-smi info`，同时递归追踪本实验服务进程树。

### 3.3 模型配置摘要

| 模型 | hidden | 层数 | full attention / GDN | intermediate | attention heads / KV heads | GDN K/V heads |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 2048（第一阶段记录） | 24 | 6 / 18 | 约 5632 | 16 / 4 | 16 / 32 |
| 4B | 2560 | 32 | 8 / 24 | 9216 | 16 / 4 | 16 / 32 |
| 9B | 4096 | 32 | 8 / 24 | 12288 | 16 / 4 | 16 / 32 |
| 27B | 5120 | 64 | 16 / 48 | 17408 | 24 / 4 | 16 / 48 |

所有模型的 full-attention interval 为 4，GDN key/value head dim 为 128，full-attention head dim 为 256，causal convolution kernel 为 4。

## 4. TP=2 权重切分与通信对应关系

### 4.1 典型 Qwen3.5-4B 全注意力层

以第 3 层 full-attention 为例，设输入 token 行数为 `T`，hidden size 为 2560，TP size 为 2。输入 hidden 在两个 rank 复制；Q/K/V 和 O 投影权重按头或输入维切分。

| 张量/步骤 | rank 0 | rank 1 | 切分与通信 |
|---|---|---|---|
| 归一化输入 X | `[T,2560]` | `[T,2560]` | 完整 hidden 复制 |
| Q/gate 投影 | 本地 Q 8 头、gate 8 头 | 另 8 头、另 8 gate 头 | ColumnParallel，权重输出维切半 |
| K/V 投影 | K 2 头、V 2 头 | K 2 头、V 2 头 | KV 头按 TP 切分 |
| 本地 attention 输出 | `[T,2048]` | `[T,2048]` | 各 rank 只计算本地 attention heads |
| O 投影权重 | `[2560,2048]` | `[2560,2048]` | 输入维切分，各自产生 `[T,2560]` 局部和 |
| O 投影后 | `[T,2560]` 局部和 | `[T,2560]` 局部和 | `CommManager.post_attn_comm` 执行 HCCL AllReduce SUM |
| 残差/Norm 后 | `[T,2560]` 完整 | `[T,2560]` 完整 | 两 rank 获得相同完整 hidden |

TP=2 下，O 投影本身设置 `reduce_results=False`，不能把通信误写成 O 投影内部自动归约；集合通信发生在层边界。

### 4.2 典型 Qwen3.5-4B GDN 层

4B 的 GDN 总 key dim 为 `16×128=2048`，value dim 为 `32×128=4096`。每个 rank 保存 8 个 K/Q 头和 16 个 V 头。

| 张量/步骤 | rank 0 | rank 1 | 说明 |
|---|---|---|---|
| 输入 X | `[T,2560]` | `[T,2560]` | 完整 hidden 复制 |
| Q/K/V/Z 投影 | Q/K 各 `[T,1024]`，V/Z 各 `[T,2048]` | 同形状、不同头 | MergedColumnParallel，输出通道切半 |
| B/A 投影 | `[T,32]` | `[T,32]` | B/A 按 V heads 切分 |
| causal conv | 本地通道，kernel=4 | 本地通道，kernel=4 | 卷积状态本地保存 |
| recurrent state | `[state_slots,16,128,128]` | `[state_slots,16,128,128]` | V 头切分，不跨 rank 复制完整 state |
| GDN 输出投影 | `[2560,2048]` → `[T,2560]` 局部和 | 同形局部和 | RowParallel，`reduce_results=False` |
| post-attention | 完整 `[T,2560]` | 完整 `[T,2560]` | HCCL AllReduce SUM 后残差/Norm |

GDN 的 convolution state 和 recurrent state 随 TP 复制状态槽结构但切分 head 维；推理时同一请求在两个 rank 的 state slot 由对应 rank 独立更新，不能把两个 rank 当作两个请求副本。

### 4.3 稠密 MLP 层

4B 的 intermediate size 为 9216。每 rank 的 gate/up 中间通道为 4608，down projection 输入维为 4608，输出为 hidden 的局部和 `[T,2560]`。`CommManager.post_dense_comm` 在当前 `attn_tp=2、dense_tp=2` 映射下执行 HCCL AllReduce SUM，得到下一层需要的完整 hidden。

9B 和 27B 的切分规则相同，仅 global hidden/intermediate/head 数不同：9B 的 gate/up local intermediate 为 6144，27B 为 8704。

### 4.4 Embedding、LM head 与推理状态

- Embedding：词表 248320 按 TP=2 分为每 rank 124160 行；本地 lookup 结果在 hidden 维形成 partial/replicated layout，并在需要时 AllReduce。
- LM head：每 rank 权重为 `[vocab/2, hidden]`；本地 logits 为 `[rows,124160]`，采样前通过 sampled-row AllGather 得到完整候选或等价的全局 top-k。
- Full-attention KV cache：每 rank 只保存本地 KV heads；不复制另一 rank 的完整 KV cache。
- GDN causal-conv state：每 rank 只保存本地 Q/K/V 通道的历史窗口，kernel=4 因此保留 3 个历史位置。
- GDN recurrent state：每 rank 保存本地 V heads 的 K-last 状态；TP=2 不在每个 decode token 上 AllGather 完整状态。

### 4.5 普通 decoder 层通信时间线

```text
复制/已有完整 X [T,H]
  ├─ rank-local QKV 或 GDN 投影、attention/recurrent、O/down partial
  ├─ HCCL AllReduce SUM [T,H]      ← attention/GDN 输出边界
  ├─ residual + RMSNorm
  ├─ rank-local gate/up/down partial
  ├─ HCCL AllReduce SUM [T,H]      ← MLP 输出边界
  └─ 下一层；最终 logits 做词表 sampled-row AllGather
```

如果映射切换为非 all-reduce 布局，`CommManager` 会在层间使用 token AllGather、ReduceScatter 或 All-to-All transpose。当前四模型实验采用 attention TP=2、dense TP=2、DP=1、无 QCP，因此主要路径为层边界 AllReduce，不能把本报告的集合通信类型直接外推到其他并行映射。

## 5. 四模型连续推理与正确性

### 5.1 统一测试方法

每个模型分别启动服务，使用相同权重、BF16、greedy 和请求配置；执行短提示 A、短提示 B、322-token 长提示和重复短提示 A 四条顺序请求。TP=1 和 TP=2 仅改变可见 chip 和 tensor-parallel-size，不改变任务配置。每次服务完成后关闭本实验进程，不操作其他用户进程。

### 5.2 结果汇总

| 模型 | TP=2 连续请求 | TP=1/TP=2 文本一致 | CPU Transformers BF16 参考 | 结论 |
|---|---:|---:|---:|---|
| 0.8B | 4/4 | 4/4 | 4/4 | 通过 |
| 4B | 4/4 | 4/4 | 4/4 | 通过 |
| 9B | 4/4 | 4/4 | 2/4 | TP 一致；CPU 参考差异待进一步扩大验证 |
| 27B | 4/4 | 未运行 TP=1 | 3/4 | TP=2 连续推理通过；不构造单 chip 基线 |

9B 的两条 CPU 参考差异发生在并列最高分附近：TP=1 和 TP=2 选择相同 token，固定前缀 teacher-forced 对照的选中 token log probability 差异小于约 0.11，但这不是全词表 logits 误差上界。9B 将线性状态切换为 FP32 后文本仍与 BF16 TP=2 一致，不能据此宣称根因已完全解决。

27B 的 TP=1 不是“未测试失败”，而是明确不运行：单 chip 可用容量不足以安全容纳文本权重和运行时状态。报告中不编造 27B TP=1 性能或正确性基线。

## 6. 4B TP=1/TP=2 性能与容量

每个配置预热后运行 2 轮，每轮 8 个请求；吞吐为两轮 completion token 总数除以批次窗口时间总和；p50/p95 为合并后的 16 个请求。HBM 为离散采样观察值。

### 6.1 性能矩阵

| 输入 token | 并发 | TP=1 吞吐 | TP=2 吞吐 | 变化 | TP=1 p50 | TP=2 p50 | TP=1 p95 | TP=2 p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 9.357 | 8.846 | -5.5% | 3.423 | 3.605 | 3.472 | 3.775 |
| 128 | 4 | 18.635 | 16.521 | -11.3% | 6.867 | 6.821 | 6.877 | 10.575 |
| 512 | 1 | 4.747 | 4.793 | +1.0% | 6.723 | 6.664 | 6.884 | 7.114 |
| 512 | 4 | 6.331 | 6.470 | +2.2% | 20.182 | 19.458 | 20.359 | 21.055 |

4B TP=2 在本共享负载环境下没有稳定吞吐收益。短输入并发 4 的 TP=2 两轮波动较大，尾延迟反而变差；长输入下吞吐约持平，不能仅凭 +1.0%/+2.2% 宣称线性扩展。

### 6.2 HBM 与 rank 证据

| 模型/TP | 正式采样数 | chip0 物理 ID | chip1 物理 ID | 实验进程峰值 MB | 设备总峰值 MB | 启动前总占用 MB |
|---|---:|---:|---:|---:|---:|---:|
| 4B TP=1 | 75 | 2 | 3 | 9867 / 0 | 25561 / 15266 | 15863 / 15266 |
| 4B TP=2 | 77 | 2 | 3 | 6511 / 6175 | 22163 / 21384 | 15862 / 15266 |
| 27B TP=2 | 91 | 2 | 3 | 28638 / 28302 | 44290 / 43511 | 15866 / 15266 |

TP=2 将 4B 单 chip 实验显存降低到约 6.5/6.2 GB，但两 chip 实验进程总显存为 12686 MB，相对 TP=1 单 chip 9867 MB 增加约 28.6%，对应通信缓冲、复制状态和运行时开销。27B TP=2 两个 rank 均被采样捕获，容量分摊成立。

## 7. 通信相关策略实验：overlap scheduler

### 7.1 假设与变量

假设：默认 overlap scheduler 将 CPU scheduler/post-process 与 NPU model worker 的执行交错，可能隐藏 HCCL/设备等待；在并发较高时收益应比单请求更明显，但可能增加排队或尾延迟。

只改变一个因素：

- off：`--disable-overlap-schedule`，日志 `overlap_schedule_depth=0`；
- on：不传该选项，默认 overlap scheduler，日志 `overlap_schedule_depth=1`。

两种模式均恢复并固定 `--disable-autotune --disable-prefix-caching --disable-kvstore`，模型、TP、BF16、输入 512 token、并发 1/4、每轮 8 请求和 2 轮均一致。两种模式结果中 `cache_tokens=0`，因此没有 prefix-cache 混杂。

### 7.2 结果

| 模式 | 并发 | 两轮吞吐 | 合并吞吐 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| overlap off | 1 | 4.706 / 4.732 | 4.719 | 6.773 | 6.896 |
| overlap on | 1 | 4.597 / 4.772 | 4.683 | 6.727 | 7.432 |
| overlap off | 4 | 6.473 / 6.486 | 6.479 | 19.756 | 19.779 |
| overlap on | 4 | 6.669 / 6.679 | 6.674 | 19.165 | 19.426 |

相对 off：并发 1 吞吐 -0.8%、p50 -0.7%、p95 +7.8%；并发 4 吞吐 +3.0%、p50 -3.0%、p95 -1.8%。两种模式的实验进程 HBM 采样峰值均约 6511/6175 MB，未观察到显存代价显著增加。

### 7.3 机制解释与时间线

TokenSpeed 的 RequestStats 将一次请求拆为 `queue_ms`、`prefill_ms`、`ttft_ms`、`total_ms` 和 `decode_tps`；服务日志还记录 overlap depth。关闭 overlap 时，CPU 端完成调度/后处理后再推进下一轮设备工作；开启 overlap 时，CPU scheduler 可以与 model worker 的 NPU 计算、层边界 HCCL 集合通信交错。并发 4 的 overlap-on 日志中可见多个请求共享相近的 prefill/finish 时间窗，而单请求并发 1 的总吞吐几乎不变，符合“隐藏等待而非减少算术或链路字节”的机制。

本实验没有把 `queue + prefill + decode + communication` 机械相加为端到端时间；这些阶段在 overlap-on 下可能重叠。当前日志可以区分队列、prefill、TTFT、total 和 decode，但没有安装独立 HCCL 链路 profiler，因此不能给出单次 AllReduce 的精确微秒占比。报告将集合通信形状、调用位置和 scheduler depth 作为通信路径证据，将端到端差异作为策略收益证据。

### 7.4 配置错误的排除

第一次 overlap-on 启动器通过删除整行参数，误删了 `--disable-prefix-caching` 等非目标选项，日志出现 `cache_tokens=384`。该批文件已保存为 `confounded_*`，不进入表 7.2；修正后的正式结果 cache_tokens 全为 0。这一审计也说明通信策略实验必须同时核对启动参数、cache 状态和 RequestStats，不能只看吞吐数字。

## 8. 进程、设备与连续推理证据

- `step13_model_correctness/server_<model>_tp2.log`：四模型 TP rank 0/1 启动、权重加载、健康检查和连续请求；
- `step13_model_correctness/npu_<model>_tp2.log`：rank 进程和 chip 设备占用快照；
- `step16_tp_performance/samples_*.jsonl`：4B TP=1/TP=2、27B TP=2 的正式性能阶段 HBM/进程树采样；
- `step17_comm_overlap/samples_4B_tp2_{off,on}.jsonl`：通信策略实验正式阶段采样；
- 服务结束后本实验进程均已清理，未操作其他用户进程。NPU1 上其他用户进程仍可能存在，因此设备总 HBM 不能等同于本实验进程 HBM。

## 9. 原始证据与复现入口

### 9.1 目录索引

```text
/home/u2601112013/tokenspeed_group_one/experiments/qwen35_ascend_npu1/
├── step13_model_correctness/       四模型 TP=2 连续推理与 TP=1 对照
├── step14_9b_review/               9B CPU 参考分歧定位
├── step15_tp_layout/               切分、形状、HCCL 路径探针
├── step16_tp_performance/          4B TP1/TP2 与 27B TP2 性能容量
└── step17_comm_overlap/            overlap off/on 通信策略实验
```

### 9.2 第 5 步原始文件

```text
step16_tp_performance/REPORT.md
step16_tp_performance/results_4B_tp1.json
step16_tp_performance/results_4B_tp2.json
step16_tp_performance/results_27B_tp2.json
step16_tp_performance/samples_4B_tp1.jsonl
step16_tp_performance/samples_4B_tp2.jsonl
step16_tp_performance/samples_27B_tp2.jsonl
step16_tp_performance/controller_*.log
step16_tp_performance/server_*.log
```

### 9.3 第 6 步原始文件

```text
step17_comm_overlap/benchmark.py
step17_comm_overlap/run_server_off.sh
step17_comm_overlap/run_server_on.sh
step17_comm_overlap/results_4B_tp2_off.json
step17_comm_overlap/results_4B_tp2_on.json
step17_comm_overlap/samples_4B_tp2_off.jsonl
step17_comm_overlap/samples_4B_tp2_on.jsonl
step17_comm_overlap/controller_4B_tp2_{off,on}.log
step17_comm_overlap/server_4B_tp2_{off,on}.log
step17_comm_overlap/confounded_*.log/json/jsonl
```

复现时必须先确认 NPU1 的物理 ID 2、3 可用，不能在其他设备或其他用户进程上执行；每次只驻留一个模型。所有正式结果均使用项目内 `.venv` 和 CANN 环境。

## 10. 已知问题与边界

- 性能结果来自共享服务器，不是独占 NPU 的 SLA；设备总 HBM 包含其他用户进程；
- HBM 是约 1.5 秒采样的观察峰值，可能漏掉瞬时峰值；
- 没有独立 HCCL profiler，因此通信微秒级耗时和链路实际字节数未直接测量；表中的张量形状/字节数是集合调用的逻辑 payload，不等于链路流量；
- 9B/27B 与 CPU BF16 参考的个别 token 差异仍需课程认可参考、逐层 hidden、完整 logits 和更多提示扩大验证；不能将 CPU 参考差异直接称为 TP 通信错误；
- 27B TP=1 因容量约束未运行，报告不提供虚构单 chip 基线；
- 未覆盖量化、多模态视觉塔、动态 batch、prefix cache 开启、retraction、speculative decoding、QCP 或 TP>2；
- overlap 实验只改变 scheduler 开关，没有把 AllReduce 替换为另一种通信算法；结论是“计算/通信等待重叠策略”的收益，不是 HCCL 拓扑优劣结论；
- 当前通信时间线来自 RequestStats、scheduler depth、源码调用点和 HBM 采样；若要提交通信占比，应进一步使用 CANN/HCCL profiler 或设备时间线工具。

## 11. 阶段验收结论与后续工作

### 11.1 验收结论

第二阶段核心要求已完成：

1. 四个模型分别完成 TP=2 双 chip 连续推理；
2. 两个 rank 的启动日志、物理设备映射和 HBM/进程采样已保存；
3. 0.8B、4B、9B 的 TP=1/TP=2 正确性对照已有证据，27B TP=1 按容量限制明确不运行；
4. 4B 完成两档输入、两档并发的 TP=1/TP=2 吞吐、延迟、HBM 对照；
5. 27B 完成 TP=2 性能和容量补充；
6. 已完成 overlap scheduler 通信相关实验，并解释其收益机制和公平性审计。

### 11.2 后续建议

1. 在独占 NPU 时段重跑 4B TP=1/TP=2 和 overlap 对照，增加重复次数并报告标准差或置信区间；
2. 安装 CANN/HCCL profiler，分别记录 AllReduce、AllGather、ReduceScatter、计算 kernel 和等待区间；
3. 对 GDN state、full-attention KV 和层边界 hidden 保存逐层/逐步 dump，扩大 9B/27B 参考差异诊断；
4. 尝试融合 AllReduce+residual+RMSNorm，减少层边界通信与同步；
5. 比较通信粒度、chunked prefill 大小和 batch size，确认 overlap 收益是否随 token 数和并发变化；
6. 继续保持 27B 不运行 TP=1 的容量边界声明，除非取得独占且可验证的单 chip 资源。

---

## 附录 A：启动命令摘要

### TP=1（4B 示例）

```bash
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_VISIBLE_DEVICES=2
timeout 1800 .venv/bin/python \
  experiments/qwen35_ascend_npu1/step16_tp_performance/benchmark.py 4B 1
```

### TP=2（4B/27B 示例）

```bash
export ASCEND_RT_VISIBLE_DEVICES=2,3
export ASCEND_VISIBLE_DEVICES=2,3
timeout 1800 .venv/bin/python \
  experiments/qwen35_ascend_npu1/step16_tp_performance/benchmark.py 4B 2
```

### 通信策略实验

```bash
# off：run_server_off.sh 内含 --disable-overlap-schedule
COMM_MODE=off COMM_LENGTHS=512 COMM_CONCURRENCIES=1,4 \
  .venv/bin/python experiments/qwen35_ascend_npu1/step17_comm_overlap/benchmark.py 4B 2

# on：run_server_on.sh 不含该选项，其他禁用项保持一致
COMM_MODE=on COMM_LENGTHS=512 COMM_CONCURRENCIES=1,4 \
  .venv/bin/python experiments/qwen35_ascend_npu1/step17_comm_overlap/benchmark.py 4B 2
```

## 附录 B：引用的阶段报告

```text
step13_model_correctness/REPORT.md
step14_9b_review/REPORT.md
step15_tp_layout/REPORT.md
step16_tp_performance/REPORT.md
step17_comm_overlap/（本报告新增实验目录）
```

本报告参照本地文件 `Qwen3.5-0.8B_Ascend910C_单Chip适配实验报告_详尽版.md` 的章节风格、完成情况矩阵、证据索引和限制说明拟制。
