import json
import math
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu


ROOT = Path(__file__).resolve().parent
TOTAL_BYTES = 4 * 1024 * 1024
CASES = [
    ("1x4MiB", 4 * 1024 * 1024, 1),
    ("4x1MiB", 1 * 1024 * 1024, 4),
    ("16x256KiB", 256 * 1024, 16),
    ("64x64KiB", 64 * 1024, 64),
    ("512x8KiB", 8 * 1024, 512),
]
WARMUP = 3
REPEATS = 30


def percentile_nearest_rank(values, q):
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "2,3"
    assert os.environ.get("ASCEND_VISIBLE_DEVICES") == "2,3"
    assert int(os.environ["WORLD_SIZE"]) == 2
    assert local_rank in (0, 1)

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", timeout=timedelta(seconds=300))
    results = []
    try:
        for name, bytes_per_call, calls_per_repeat in CASES:
            assert bytes_per_call * calls_per_repeat == TOTAL_BYTES
            assert bytes_per_call % 2 == 0
            tensor = torch.zeros(
                bytes_per_call // 2,
                dtype=torch.bfloat16,
                device=f"npu:{local_rank}",
            )

            dist.barrier()
            for _ in range(WARMUP):
                for _ in range(calls_per_repeat):
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.npu.synchronize()

            elapsed_ms = []
            for _ in range(REPEATS):
                dist.barrier()
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(calls_per_repeat):
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                torch.npu.synchronize()
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)

            p50_ms = statistics.median(elapsed_ms)
            p95_ms = percentile_nearest_rank(elapsed_ms, 0.95)
            row = {
                "case": name,
                "rank": rank,
                "local_rank": local_rank,
                "physical_id": (2, 3)[local_rank],
                "bytes_per_call": bytes_per_call,
                "calls_per_repeat": calls_per_repeat,
                "total_bytes_per_repeat": TOTAL_BYTES,
                "warmup": WARMUP,
                "repeats": REPEATS,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "p50_us_per_call": p50_ms * 1000.0 / calls_per_repeat,
                "effective_payload_gbps": (TOTAL_BYTES / 1e9) / (p50_ms / 1000.0),
                "elapsed_ms": elapsed_ms,
            }
            results.append(row)
            print(json.dumps(row), flush=True)

        payload = {
            "hostname": socket.gethostname(),
            "rank": rank,
            "world_size": dist.get_world_size(),
            "visible_physical_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
            "dtype": "bfloat16",
            "results": results,
        }
        (ROOT / f"allreduce_rank{rank}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
