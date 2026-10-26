# 第10步：最终验收与可复现性交付审计

## 1. 验收结论

第10步通过。Qwen3.5-0.8B 当前在 Ascend 910C 的单个目标 chip 上保持运行，
健康检查和最终短推理均成功；源文件语法、补丁格式、报告及机器可读结果已归档并生成
SHA-256 校验和。

## 2. 约束核对

- 项目及全部新增产物：`/home/u2601112013/tokenspeed_group_one`
- 目标设备：`NPU ID 1 / Chip ID 0 / Phy-ID 2`
- 设备可见性：`ASCEND_RT_VISIBLE_DEVICES=2`、`ASCEND_VISIBLE_DEVICES=2`
- PyTorch 逻辑设备：`npu:0`
- 未修改项目目录外的服务器代码。
- 最终验收期间未启动其他 NPU 任务，未占用其他 NPU。

## 3. 运行状态

- 服务地址：`http://127.0.0.1:30317`
- 主服务 PID：`3673263`
- NPU 进程：
  - `tokenspeedsch` PID `3673391`，约 2560 MB
  - `ts-serve` PID `3673263`，约 112 MB
  - Python/SMG PID `3673342`，约 112 MB
- 合计进程 HBM：约 2784 MB
- `/health`：通过
- 最终生成请求：通过，4 个输入 token、8 个输出 token、`cached_tokens=0`

## 4. 源码与格式审计

- `git diff --check`：通过。
- 关键修改 Python 文件 `py_compile`：通过。
- tracked diff：9 个文件，163 行新增、8 行删除。
- 另有新增 Ascend GDN 实现、dispatcher、测试和实验目录，均位于项目目录内。
- 未执行 git commit、push、reset 或覆盖用户已有修改。

主要源码范围：

```text
python/tokenspeed/runtime/execution/model_executor.py
python/tokenspeed/runtime/layers/attention/backends/state/mamba.py
python/tokenspeed/runtime/layers/attention/linear/layernorm_gated.py
python/tokenspeed/runtime/layers/layernorm.py
python/tokenspeed/runtime/models/qwen3_5.py
tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/mha.py
tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/gdn.py
tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/ascend.py
tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/__init__.py
tokenspeed-kernel-npu/test/test_gdn.py
test/runtime/layers/test_npu_attention_backend.py
```

## 5. 正确性与性能证据索引

- 环境和基础验证：`experiments/qwen35_ascend_npu1/step01_baseline/`
- 计算路径和算子矩阵：`step02_path_matrix/`
- 基础算子验证：`step03_base_ops/`
- 最小兼容适配：`step04_minimal_adaptation/`
- GDN 初始适配：`step05_gdn_adaptation/`
- Decode、参考对照和状态隔离：`step06_decode_debug/`
- 原始性能基线：`step07_performance_baseline/STEP07_REPORT.md`
- GDN 原生算子优化：`step08_gdn_optimization/STEP08_REPORT.md`
- 优化后复测和总报告：`step09_final_validation/STEP09_REPORT.md`

最终服务级结果位于：

```text
experiments/qwen35_ascend_npu1/step09_final_validation/summary.json
experiments/qwen35_ascend_npu1/step09_final_validation/comparison.json
experiments/qwen35_ascend_npu1/step09_final_validation/client_results.json
experiments/qwen35_ascend_npu1/step09_final_validation/npu_samples.jsonl
```

## 6. 可复现命令

启动服务：

```bash
cd /home/u2601112013/tokenspeed_group_one
export TOKENSPEED_ASCEND_GDN_NATIVE=1
nohup bash experiments/qwen35_ascend_npu1/step08_gdn_optimization/run_server.sh 30317 \
  > experiments/qwen35_ascend_npu1/step08_gdn_optimization/live_server.log 2>&1 < /dev/null &
```

健康检查和推理：

```bash
curl -f http://127.0.0.1:30317/health
curl -sS http://127.0.0.1:30317/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-0.8b","prompt":"The capital of France is","max_tokens":32,"temperature":0,"ignore_eos":true}'
```

校验交付文件：

```bash
cd /home/u2601112013/tokenspeed_group_one
sha256sum -c experiments/qwen35_ascend_npu1/step10_delivery_audit/SHA256SUMS
```

## 7. 本环节问题与解决方案

第一次通过 PowerShell、SSH 和远端 shell 三层传递内联 JSON 时，转义字符被保留，服务
按预期返回 `invalid_json`。这不是模型或服务故障。处理方式是将请求保存为项目目录内的
`smoke_request.json`，再使用 `curl --data-binary @file` 发送；重测成功并覆盖原失败
响应记录。

## 8. 交付产物

目录：`experiments/qwen35_ascend_npu1/step10_delivery_audit/`

- `FINAL_ACCEPTANCE.md`：本验收说明。
- `SHA256SUMS`：16 个关键源码、测试、报告和结果文件的校验和。
- `current_status.txt`：工作区状态快照。
- `tracked_diff_stat.txt`：tracked diff 统计。
- `py_compile.log`：关键 Python 文件语法检查记录；空文件表示无错误输出。
- `smoke_request.json`、`smoke_response.json`：最终短推理输入输出。
- `npu1_processes.log`：目标 NPU 进程快照。

## 9. 未解决但不阻塞验收的事项

- Decode 仍约 53--54 ms/token，是当前性能瓶颈。
- 容器 PID 1 历史上未及时回收僵尸进程；管理员已将 `pids.max` 提升至 2000000，
  当前服务可正常运行，但长期仍应使用具备 reaper 能力的 init（如 tini）。
- 当前工作区未提交。是否整理 commit/patch 由用户决定，避免擅自改变版本历史。

综上，本阶段适配、验证、性能基线、优化复测和交付审计均已完成。
