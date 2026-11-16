# 第二、三步实验记录：四模型 TP=2 与正确性对照

日期：2026-10-28。服务器工作区 `/home/u2601112013/tokenspeed_group_one`。

## 实验范围

- 四模型分别加载，均为 Qwen 官方权重、BF16、仅文本、greedy、输出上限 16 token。NPU 1 Chip 0/1 对应物理 ID 2/3，TP=2 设置 `ASCEND_RT_VISIBLE_DEVICES=2,3`。TP=1 只使用物理 ID 2。
- 一组 4 条顺序请求：短提示 A、短提示 B、322 token 长提示、重复短提示 A；成功记录了连续推理与重复请求的文本结果。该重复测试并不能穷尽并发状态隔离。
- 两个 chip 另有用户服务，结果只用于功能和正确性，不作为性能基准。每次使用单请求并在完成后关闭本实验服务。

## 结果

| 模型 | TP=2 四条连续请求 | TP=1/TP=2 逐条文本一致 | CPU Transformers BF16 参考与 TP=2 一致 | rank 0 / rank 1 进程显存快照 MB |
|---|---|---|---|---|
| 0.8B | 4/4 | 4/4 | 4/4 | 2433 / 2433 |
| 4B | 4/4 | 4/4 | 4/4 | 5957 / 5957 |
| 9B | 4/4 | 4/4 | 2/4 | 10527 / 10527 |
| 27B | 4/4 | 未运行：可用 HBM 约 48.5 GiB，文本权重约 50.9 GiB，尚未计运行时缓存 | 3/4 | 27903 / 27903 |

进程显存为 `npu-smi` 快照，非峰值；同一 rank 还可能在另一 chip 出现轻量上下文，具体见各 `npu_*_tp2.log`。各服务日志均记录了 global rank 0/1、两个权重加载和缓存初始化。0.8B、4B、9B TP=1 与 TP=2 权重、BF16 精度、生成参数和请求一致；TP 配置不同。Transformers 为本地 5.12.0 的 `Qwen3_5ForConditionalGeneration`，CPU BF16、eager、greedy；是否为课程认可参考有待确认。

## 差异与诊断

- 9B：第 2 条短请求和第 3 条长请求与 CPU 参考文本不同，但 TokenSpeed TP=1 与 TP=2 的四条均一致。长请求重新编码后的首次分歧发生在第 5 个生成 token；短请求在第 10 个。重新编码文本所得 token 序列不等同于直接读取服务生成 token IDs，分歧位置仅作诊断。
- 27B：第 3 条长请求与 CPU 参考不一致，首 4 个生成 token 一致，第 5 个发生分歧（同样根据输出重新编码定位）。短请求及重复短请求一致。
- 使用 `mamba-ssm-dtype=float32` 重测 9B TP=2，四条文本与 BF16 状态结果一致，参考差异仍存在。
- 用同一前缀对 Transformers CPU 的 9B/27B 进行逐 token decode 和整体重 prefill：9B 增量 top logits ` What` 与 `\n\n` 同为 18.375，而整体 prefill 分别为 18.5、18.25；27B 增量 ` What`、`\n\n` 和 `\n` 均为 18.0，整体 prefill 前两者为 18.0。此结果表明参考内部该位置对执行路径和相近 logits 敏感，但不能证明差异仅由数值造成，也未证明 TP=2 中间张量正确。
- 服务启用输出 logprobs 后，OpenAI completions 响应未包含候选概率；不能据此量化 TokenSpeed 与 CPU 的 logits 差异。相关诊断保留在 `27B_tp2_long_logprobs.json`。
- 9B TP=1 首次因模型路由注册时序返回 404；确认 `/v1/models` 后重试通过。控制器已增加路由检查；参见 `server_9B_tp1_retry.log` 和 `requests_9B_tp1.json`。

## 证据与待办

`run_server.sh`、`requests.py`、`server_<model>_tp2.log`、`npu_<model>_tp2.log`、`requests_<model>_tp1.json`、`requests_<model>_tp2.json`、`reference_<model>_hf_cpu.json`（27B 使用 `reference_27B_hf_cpu_extended.json`）、`reference_divergence_logits.log` 和 `reference_incremental_logits.log` 均在本目录。

第二步已覆盖四模型指定请求下的双 rank 连续推理；尚需按课程验收要求核查权重和状态切分的详细证据。第三步尚不能判定完全通过：需确认 CPU 参考是否被课程认可，获取可比的逐层/逐步 logits 数值，解释或修复 9B 与 27B 参考差异，扩大上下文、请求和状态隔离覆盖。27B 若需 TP=1 应等待单 chip 空闲容量足够，并先核实设备授权与实际占用。按用户要求，后续通信分析与性能实验待审核后开展。
