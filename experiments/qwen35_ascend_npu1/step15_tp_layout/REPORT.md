# 第四步：Qwen3.5-9B TP=2 切分与通信对应关系

日期：2026-10-28；项目版本 `4c3993e4` 加本工作区未提交的 Ascend TP 适配。只讨论已运行的纯文本、BF16、attention TP=2、dense TP=2、DP=1、无 QCP、无推测解码、eager 配置。Qwen3.5-9B 是稠密模型，不走 MoE expert 并行，也不包含视觉塔。

## 1. 依据与证据边界

- 模型配置：hidden=4096，intermediate=12288，vocab=248320，32 层，其中 24 层 GDN、8 层全注意力；全注意力 Q 头数 16、KV 头数 4、每头 256；GDN K 头 16、V 头 32、每头 128，conv kernel=4。
- NPU 1 的 Chip 0/1 对应物理 ID 2/3；本次 `ASCEND_RT_VISIBLE_DEVICES=2,3`，rank 0/1 对应进程内 npu:0/1，前次 `npu_9B_tp2.log` 中已记录设备进程占用。
- `trace_layout.py` 只在独立 Engine 实验进程中临时包裹函数，记录形状、dtype、调用与 PID，不改模型或框架代码；不是性能探针，也不记录网络传输时间。短请求 prompt 为 5 token，生成 3 token；`trace_embedding_logits.log` 返回 `TRACE_REQUEST_OK`，`output.json` 保存输出。
- `rank0.jsonl` 与 `rank1.jsonl` 保存各 rank 的投影权重、投影输出、GDN state 和 HCCL 调用。文件内含多次探针运行及初始化 warmup；可用 PID 和输入形状区分。最新 embedding/输出头运行 PID 810515/810516；状态池的独立运行 PID 809429/809430。
- 下面表格的 `T` 指本次前向的 token 行数：本例 prefill T=5，单请求 decode T=1。模型算子权重形状均按 PyTorch 的 `[输出维, 输入维]` 表示；局部张量相同形状不意味着权重内容被复制。

## 2. 全注意力层：第 3 层（从 0 开始）

| 张量或步骤 | rank 0 | rank 1 | 说明 |
|---|---|---|---|
| 归一化后输入 X | `[T,4096]` | `[T,4096]` | 相同 token 行和完整 hidden 维，复制 |
| fused Q/gate/K/V 投影权重 | `[5120,4096]` | `[5120,4096]` | 输出维按头分片：本地 Q 8 头、gate 8 头、K/V 各 2 头 |
| Q、gate | `[T,8,256]` 各一份 | `[T,8,256]` 各一份 | 每 rank 持不同 Q/gate 头 |
| K、V | `[T,512]` 各一份 | `[T,512]` 各一份 | 每 rank 2 个 KV 头，等价 `[T,2,256]` |
| 本地 attention/gate 输出 | `[T,2048]` | `[T,2048]` | 各自计算本地头 |
| O 投影权重 | `[4096,2048]` | `[4096,2048]` | 输入维分片，各算局部贡献 |
| O 投影后 | `[T,4096]` 部分和 | `[T,4096]` 部分和 | 两者不完整；`CommManager.post_attn_comm` 对部分和做 **HCCL AllReduce SUM**，然后执行残差与归一化 |

实测 `_project_qkv` 在 T=5 时输出 `[5,8,256]`、`[5,512]`、`[5,512]`、`[5,8,256]`；`rank*.jsonl` 中第 3 层观察到两次 `[5,4096]` BF16 AllReduce，分别对应 attention 后和 MLP 后。每次每 rank 输入张量为 40,960 B；T=1 时为 8,192 B。这里的字节数是调用张量大小，不是 HCCS/SIO 链路的实际流量。

## 3. Gated DeltaNet 层：第 0 层

K 维总计 16×128=2048，V 维总计 32×128=4096。每 rank 分别处理 8 个 K/Q 头及 16 个 V 头。

| 张量或步骤 | rank 0 | rank 1 | 说明 |
|---|---|---|---|
| 输入 X | `[T,4096]` | `[T,4096]` | 完整 hidden，复制 |
| 合并输入投影权重 | `[6176,4096]` | `[6176,4096]` | Q/K/V/Z/B/A 各自按 head 分片后打包；实测为该权重形状 |
| Q/K/V/Z 输出 | `[T,6144]` | `[T,6144]` | 本地 Q、K 各 1024，V、Z 各 2048 |
| B/A 输出 | `[T,32]` | `[T,32]` | B、A 各 16 |
| Q/K/V 混合与 causal conv | `[T,4096]`，conv 权重 `[4096,1,4]` | 同形、不同分片 | Q/K/V 本地宽度 1024+1024+2048；`A_log` 和 `dt_bias` 各 `[16]`，按 V 头切分 |
| GDN 递推、门控及归一化 | `[T,2048]` | `[T,2048]` | 每 rank 的本地 V 头；不在此处跨 rank 合并状态 |
| 输出投影权重与局部和 | `[4096,2048]` → `[T,4096]` | `[4096,2048]` → `[T,4096]` | 输出投影不自行 reduce；在 post-attention 边界通过 **HCCL AllReduce SUM** 合并，再残差与归一化 |

运行时探针核对：T=5 的输入投影结果为 `[5,6144]` 与 `[5,32]`；两 rank 均记录两次 `[5,4096]` BF16 AllReduce。代码中投影设置 `reduce_results=False`，所以不能把通信位置误写为 `out_proj` 内部自动归约。

## 4. 共同的稠密 MLP

两种层都使用同样的 gated MLP。每 rank 的 gate/up 合并投影权重 `[12288,4096]`，其中 gate 和 up 各持有全局 12288 中间通道的一半：各 `[6144,4096]`；对应输出各 `[T,6144]`。本地计算 `SiLU(gate) * up`，结果 `[T,6144]`；down 投影权重 `[4096,6144]`，各自生成 `[T,4096]` 部分和。`CommManager.post_dense_comm` 对这些部分和做 **HCCL AllReduce SUM**，形成下一层所需的完整 hidden。

`gate_up`、`down` 权重形状在两 rank 的 `layer` 事件中实测；中间 `[T,6144]` 是根据投影切分和权重形状推导，探针未单独记录 gate/up 激活。归一化和残差在本配置中以完整 hidden 表示；norm 参数复制。

## 5. 推理状态和词表

| 结构 | rank 0 / rank 1 的处理 | 数据来源 |
|---|---|---|
| 全注意力 KV cache | 各 rank 保留各自 2 个 KV 头；每个 K 或 V 的逻辑页为 `[128,2,256]`。跨 rank 不复制整份 KV；实际设备池还带页/分配维度 | `kv_cache/recipes/qwen35.py` 的 `_kv_shape`；页粒度 128 |
| GDN causal conv 状态 | 每个状态槽 `[4096,3]`，按 Q/K/V 通道分片；kernel=4 因此保留 3 个历史位置 | 配方及实测 `_state_components` |
| GDN recurrent 状态 | 每个状态槽 `[16,128,128]`，按 V 头切分，K-last；本次配置状态为 BF16 | `linear_attn.py` 的 `temporal_state_shape` 及实测 |
| 状态池实际形状 | 两 rank 分别为 conv `[9,4096,3]`、recurrent `[9,16,128,128]` | `rank*.jsonl`；9 是该运行的池容量，并非同时有 9 个请求 |
| embedding | vocab 248320 按词表分两片，每 rank 权重 `[124160,4096]`；每 rank 计算本地 lookup 的部分输出，对 `[T,4096]` 做 **AllReduce SUM** | 探针 `embedding` 和 `hccl_all_reduce`；`vocab_parallel_embedding.py` |
| LM head 和采样 | 每 rank 权重 `[124160,4096]`，本地 sampled-row logits `[1,124160]`；**AllGather** 为 `[2,124160]` 后重排成完整 `[1,248320]`，再 greedy 采样 | 探针 `logits` 和 `hccl_all_gather_single`；`logits_processor.py` |

本次 sampled-row AllGather 每 rank 输入为 248,320 B、输出缓冲为 496,640 B；这同样是张量大小而非链路实际字节数。上述 cache 形状针对该模型与 TP=2，不能直接推到 0.8B/4B/27B 的头数或其他 TP 设置。

## 6. 端到端路径（一个普通 decoder 层）

```text
每 rank 复制 X [T,4096]
        │
        ├─ 按 rank 做全注意力或 GDN 分片计算及本地输出投影
        │          rank 0: [T,4096] 局部和
        │          rank 1: [T,4096] 局部和
        ├─ HCCL AllReduce SUM → 两端完整 [T,4096] → 残差/Norm
        ├─ 分片 gate/up → 本地 [T,6144] → 本地 down → [T,4096] 局部和
        └─ HCCL AllReduce SUM → 两端完整 [T,4096] → 下一层
```

层前还有词表分片 embedding AllReduce；模型出口是词表 logits AllGather。GDN conv/递推状态和 full-attention KV 不在每层做 AllGather；它们跟各自 head 分片保存在本地。此说明只适用于当前 `attn_tp=2`、`dense_tp=2`、非 query-sharded 且未启用融合的路径；不同映射可以走 ReduceScatter/AllGather 或融合通信，不能照搬。

## 7. 源码与尚未完成的分析

- `python/tokenspeed/runtime/models/qwen3_5.py:200,223,526,608,752,893`：层边界、GDN 与全注意力投影。
- `python/tokenspeed/runtime/models/qwen3_5_moe.py:109`：稠密 gate/up/down 的 TP。
- `python/tokenspeed/runtime/distributed/comm_manager.py:428,513,592,785`：当前映射下的层边界通信。
- `python/tokenspeed/runtime/layers/attention/configs/linear_attn.py:93` 和 `kv_cache/recipes/qwen35.py:174`：GDN 状态及 KV 页几何。
- `python/tokenspeed/runtime/layers/vocab_parallel_embedding.py:216,587` 和 `logits_processor.py:1135`：embedding、输出词表通信。
- `python/tokenspeed/runtime/distributed/comm_backend/hccl.py:32,80`：HCCL 后端。

本步完成了切分与通信路径的代码核查及代表性运行时形状证据；尚未测量通信耗时、计算耗时、等待、重叠或跨配置性能，属于后续步骤。9B 对 CPU 参考在并列最高分位置出现的输出差异也仍需按第三步的报告继续验证。
