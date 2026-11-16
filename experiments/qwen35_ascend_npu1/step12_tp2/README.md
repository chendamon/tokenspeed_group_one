# Qwen3.5 TP=2 初步验证（2026-10-28）

## 范围与环境

- 工作区：`/home/u2601112013/tokenspeed_group_one`，版本 `4c3993e4`；不覆盖已有未提交改动。
- NPU 1 Chip 0/1 对应物理 ID 2/3；实验设置 `ASCEND_RT_VISIBLE_DEVICES=2,3` 和 `ASCEND_VISIBLE_DEVICES=2,3`。
- 两个 chip 同时有用户 u2601213438 的工作负载，因此本次仅为功能 smoke，不作为性能基准。

## 实验与证据

1. `hccl_smoke.py`：两 rank 的 AllReduce 求和均为 3，见 `hccl_smoke.log`。
2. `hccl_backend_integration.py`：使用 HcclBackend 实测 AllGather 输出 `[1, 2]`，ReduceScatter 两 rank 分别得到 `[3]`、`[5]`，见 `hccl_backend_integration.log`。测试中仅替换进程组解析为已初始化的 WORLD 组。
3. 0.8B TP=2：`run_tp2_08b.sh` 指定 BF16、greedy、eager、512 最大模型长度、单并发和 4096 token 缓存。两个 rank 加载权重并运行，两条连续请求各生成 12 token，见 `server_tp2_08b_retry4.log`、`requests_tp2.jsonl` 和 `npu_tp2_running.log`。设备快照中 rank 0/1 所在进程分别在物理 chip 2/3，均报告约 2439 MB 进程显存。
4. TP=1：相同权重、精度、任务配置和请求，见 `run_tp1_08b.sh`、`requests_tp1.jsonl`。两条输出文本与 TP=2 一致；这是小样本验证，不代表完整数值正确性。
5. `test/runtime/distributed/test_hccl_backend.py`：18 passed，见 `hccl_unit_test_retry.log`。
6. `test/runtime/test_logits_processor.py`：33 passed、1 skipped、3 failed；失败项在当前环境触发 CUDA 图捕获接口而不可用，见 `logits_unit_test.log`，未证明与本次修改是否有关。

## 适配变更

- NPU 的 TP logits 路径绕过 CUDA 专用的采样与 multicast 优化，使用 HCCL AllGather 和常规采样。
- HCCL 后端使用 PyTorch 的 `all_gather_into_tensor` 和 `reduce_scatter_tensor`，替换当前环境中不存在的接口；同步修改相关单元测试。

## 尚未完成

- 4B、9B、27B 的双 chip 服务和连续推理；更多输入、状态隔离、逐层 logits 与参考比较。
- 两档输入长度、两档并发下的 TP=1/2 性能比较、大模型容量、通信时间线与通信策略实验。
- 需要在设备资源允许且负载受控时进行性能测试，避免与现有服务争用。
