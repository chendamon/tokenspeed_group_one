# 第8步：Ascend GDN 融合优化与回归

## 1. 目标与约束

- 目标：针对第7步识别出的 Gated DeltaNet（GDN）逐 Token Python/NPU 调度瓶颈，复用现有 Ascend 运行库完成一次正确性优先的优化尝试。
- 代码与实验文件全部位于 `/home/u2601112013/tokenspeed_group_one`。
- 所有 NPU 测试均设置 `ASCEND_RT_VISIBLE_DEVICES=2`、`ASCEND_VISIBLE_DEVICES=2`，对应 `NPU ID 1 / Chip ID 0 / Phy-ID 2`；PyTorch 内部显示为逻辑 `npu:0`。
- 充分复用原项目 GDN dispatcher、状态布局、模型调用路径与测试框架，仅在 Ascend GDN 实现层接入运行库算子。

## 2. 可复用能力与实现选择

当前 `torch_npu` 暴露：

- `npu_recurrent_gated_delta_rule`
- `npu_recurrent_gated_delta_rule_functional`

其状态布局为 `[BlockNum, Nv, Dv, Dk]`，与 TokenSpeed 当前 K-last 状态池一致；输入/状态要求 BF16，单序列长度要求 `L <= 8`。因此没有重新实现 WY/Triton 算法，而是：

1. BF16 状态：使用 `torch_npu.npu_recurrent_gated_delta_rule`。
2. FP32 状态或未验证形状：保留原有 FP32 tensor recurrence 回退。
3. Prefill：按最多 8 Token 分段调用原生算子，跨段复用同一状态页。
4. Decode：先从读页 gather，再写入目标页，然后让融合算子在目标页原位更新，以保持调度器的分离读/写页语义。
5. 仅对已验证的 Qwen3.5 形状族启用融合路径：`Nk == Nv`，且 `Dk/Dv` 为 128 的倍数；其他形状安全回退。
6. 提供模块加载期 A/B 开关 `TOKENSPEED_ASCEND_GDN_NATIVE=0/1`，默认开启；该开关用于实验，不增加每 Token 环境变量读取。

## 3. 修改内容

- `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/gdn.py`
  - 增加原生 GDN 能力判定。
  - 增加 BF16 Prefill 分段融合路径。
  - 增加 BF16 Decode 分离状态页融合路径。
  - 保留 FP32/非目标形状的原有回退。
- `tokenspeed-kernel-npu/test/test_gdn.py`
  - 覆盖 BF16/FP32、变长序列、跨 8 Token 边界、padding、分离读写页。
  - 新增 Qwen3.5 实际形状（16 heads、128 维）Prefill/Decode 数值测试。
- `experiments/qwen35_ascend_npu1/step08_gdn_optimization/`
  - 复用第7步基准脚本并加入 BF16 状态和可复现线程设置。
  - 增加算子探测、数值探测、微基准和离线 Engine 连续推理脚本/日志。

## 4. 数值验证

### 4.1 原生算子对可信递推参考

参考实现按 Token 以 FP32 执行：衰减状态、`k @ state` 预测、beta 校正、外积状态更新、`q @ state` 输出。输入和持久状态使用 BF16，以匹配原生算子接口。

Qwen3.5 实际形状 `Nk=Nv=16, Dk=Dv=128` 的结果：

| 序列长度 | 输出最大绝对误差 | 输出平均绝对误差 | 状态最大绝对误差 | 状态平均绝对误差 |
|---:|---:|---:|---:|---:|
| 1 | 0.000060 | 0.000003 | 0.001919 | 0.000052 |
| 2 | 0.000060 | 0.000004 | 0.001918 | 0.000052 |
| 8 | 0.000120 | 0.000005 | 0.001898 | 0.000067 |
| 9 | 0.000117 | 0.000006 | 0.002018 | 0.000075 |
| 17 | 0.000137 | 0.000006 | 0.002959 | 0.000084 |

误差远低于测试采用的 BF16 判据（输出 `rtol/atol=2e-2/2e-2`，状态 `2e-2/3e-2`）。状态误差主要来自 BF16 持久状态和 8 Token 分段边界舍入，不要求跨硬件逐位相同。

### 4.2 自动化回归

- GDN 专项：7/7 通过。
- GDN + MHA + NPU attention backend 综合回归：22/22 通过。
- `git diff --check` 通过。
- 环境未安装 `ruff`，因此未运行 ruff；Python `py_compile` 已通过。

## 5. 端到端验证

采用 TokenSpeed `Engine`，配置：Qwen3.5-0.8B、BF16 权重、BF16 GDN 状态、MHA、greedy、eager、关闭 overlap/prefix cache/KVStore，单 chip。

固定输入 Token：`[760, 6511, 314, 9338, 369]`。Hugging Face CPU 参考的前 8 个新 Token 为：

`[11751, 13, 198, 760, 6511, 314, 9338, 369]`

结果：

- 第一次 8 Token 请求：逐 Token 完全一致，1.656 s（含首次算子/形状冷启动）。
- 第二次相同 8 Token 请求：逐 Token 完全一致，0.463 s，证明请求间状态隔离。
- 连续 32 Token 请求：成功完成，`finish_reason=length`，1.909 s；前 8 Token 与参考完全一致。
- 启动时 driver 视角设备占用约 2.48 GB；该配置的 token capacity 与第7步不同，不将其作为严格 HBM 前后对比。

## 6. 性能结果

为了隔离 HTTP/路由/调度噪声，使用同一进程、同一输入、同一 BF16 状态分别开关融合路径；每次计时前后进行 NPU 同步。

| 模块/形状 | 原逐步实现中位数 | 原生融合中位数 | 加速比 |
|---|---:|---:|---:|
| Prefill，32 Token，单层 | 21.239 ms | 1.215 ms | 17.48x |
| Prefill，512 Token，单层 | 198.561 ms | 9.303 ms | 21.34x |
| Decode，batch=1，单层/Token | 1.060 ms | 0.897 ms | 1.18x |

结论：Prefill 的主要瓶颈确为逐 Token kernel launch/主机调度，8 Token 融合后收益显著；Decode 的单步融合收益较小，因为状态 gather/copy、QK normalization、模型其他层和调度开销仍占主导。

## 7. 遇到的问题与处理

1. 原生算子仅支持 BF16 状态，而第7步默认基线使用 FP32 状态。
   - 处理：使用项目已有 `--mamba-ssm-dtype bfloat16`，并为 FP32 保留原回退；没有隐式改变默认 FP32 路径。
2. 原生 Prefill 单序列长度上限为 8。
   - 处理：在 Ascend 实现层分段，保持状态页连续更新，不侵入模型层。
3. 首次完整测试出现进程中止。
   - 诊断后确认主机 OpenBLAS 无法继续创建 64 个线程；设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1` 后全部测试通过。Ascend 日志没有对应算子错误。
4. HTTP 服务 A/B 重测被主机 PID 表耗尽阻断。
   - 当前账号约有 873 个 `PPID=1, STAT=Z` 的僵尸进程，主要为历史 `sshd/sftp-server/bash/python/tokenspeed::sch/ts-serve`。它们不执行、不占 NPU/HBM，但占 PID 表项，导致 SMG 日志线程创建失败。
   - 处理：使用持久 SSH 会话减少新增僵尸；采用单进程微基准和不启动 HTTP/SMG 的 TokenSpeed Engine 完成性能与端到端验证。僵尸进程需管理员让 PID 1 回收或重启容器，普通用户 `kill` 无效。

## 8. 当前结论与后续优化方向

- 已完成并保留的适配：BF16 Qwen3.5 GDN Prefill/Decode 原生融合路径；FP32 和非目标形状安全回退。
- 最大适配障碍：原生接口的 BF16 状态约束及 `L<=8` 分段约束；环境侧另有容器 PID 1 不回收僵尸的问题。
- 状态正确性证据：跨分段数值对照、Decode 分离读写页测试、重复请求逐 Token 一致、连续 32 Token 推理。
- 当前最值得继续优化：Decode 的状态页 gather/copy 与 QK normalization 融合，以及减少每层 Python/算子调度。证据是 Prefill 已获得 17-21x，而 Decode 单层仅 1.18x。
- 在管理员清理僵尸进程后，应重新运行第7步四组 HTTP 基线，获得服务级 TTFT/TPOT/HBM 的严格优化前后对比。

最终资源状态：`NPU ID 1 / Chip ID 0` 无进程。
