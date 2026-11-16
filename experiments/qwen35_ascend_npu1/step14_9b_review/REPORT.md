# 9B 并行正确性复核与分歧定位

日期：2026-10-28。仅使用 NPU 1 Chip 0/1（物理 ID 2/3）；27B 没有运行 TP=1。

## 方法

- 比较相同 Qwen3.5-9B 权重、BF16、greedy、相同输入和生成配置。TokenSpeed TP=1/TP=2 实验串行运行，分别显式设置 `ASCEND_RT_VISIBLE_DEVICES=2`、`2,3`。
- 针对此前和 Transformers CPU 参考不同的两条输入，使用 TokenSpeed 离线 Engine 直接取得 16 个实际生成 token ID 及每步被选 token 的 log probability，不依赖 HTTP 网关的文本重新编码。
- CPU 使用 Transformers 5.12.0 `Qwen3_5ForConditionalGeneration`，BF16、eager、增量 decode；通过 teacher forcing 固定使用 TokenSpeed 生成的相同 token 前缀，再计算每一步的完整词表概率和 CPU argmax。
- 两种提示：case 2 为 `Compute 2 + 3 =`，case 3 为 322-token 数列提示。还单独将两种模型中一致的 token 前缀作为完整 prefill，比较各自下一个 token 的 top-k log probability。

## 结果

- TP=1、TP=2 在 case 2 和 case 3 均生成完全相同的 16 个实际 token ID；此前四条连续请求的文本也全部一致。
- 同一实际 token 序列下，逐步选中 token 的 log probability 最大绝对差：TP=1 对 TP=2，case 2 为 0.047335，case 3 为 0.077187；CPU 对 TP=1 为 0.073371、0.098597；CPU 对 TP=2 为 0.066286、0.107431。这只是 16 个被选 token 的采样值，不是全词表 logits 最大误差。
- case 2 第 10 个输出 token：CPU argmax 为 token 13（句点），TokenSpeed TP=1 和 TP=2 为 token 271（双换行）。CPU 增量路径中两个 token 的 log probability 完全相同：-0.732730925；TP=1 对选中 token 为 -0.676906，TP=2 为 -0.675634。
- case 3 第 5 个输出 token：CPU argmax 为 token 271（双换行），TokenSpeed TP=1 和 TP=2 为 token 3437（` What`）。CPU 增量路径中两个 token 的 log probability 完全相同：-1.724748969；TP=1 对选中 token 为 -1.626152，TP=2 为 -1.617318。
- 完整 prefill 固定前缀时：case 2 的 CPU token 13 和 271 并列（-0.740245），TP=1 分别为 -0.800946、-0.675946，TP=2 分别为 -0.867454、-0.617454；case 3 的三种路径均选择 token 3437，CPU、TP=1、TP=2 的 log probability 分别为 -1.618771、-1.609154、-1.608852。
- 9B 将线性注意力状态由 BF16 改成 FP32 后，固定的四条生成文本与原 TP=2 结果一致，CPU 参考分歧仍存在；该单一实验不足以排除其他状态或精度问题。

## 结论与限制

这两条输入的首次文本分歧都位于 CPU 参考 BF16 最高分并列位置，且两个 TokenSpeed 并行度生成相同 token ID。证据支持分歧对数值和并列分数的选择敏感；由于硬件、前向计算与规约路径不同，不能证明根因已完全定位，也不能仅据此排除模型层误差。不能把两处固定 prefill 的统计外推为全部 decode 或全词表的误差上界。

还需更多输入、较长 decode、逐层隐藏状态和完整 logits 的对照，以及课程认可参考与误差阈值。NPU 1 当时有其他用户服务，因此不将本次结果用作性能基准。所有本次临时引擎服务退出；其他用户的进程未操作。

## 记录

- `engine_tp1_decode_logprobs.json`、`engine_tp2_decode_logprobs.json`：实际 token 和被选 token 概率。
- `cpu_teacher_forced_decode.json`、`teacher_forced_comparison.json`：同 token 前缀的 CPU 参考及汇总。
- `engine_tp1_logprobs.json`、`engine_tp2_logprobs.json`、`cpu_fixed_prefix_logprobs.json`：固定前缀 prefill top-k。
- `engine_decode_logprobs.py`、`engine_logprobs.py`、`cpu_teacher_forced_decode.py`、`cpu_fixed_prefix_logprobs.py`：复现实验脚本。
