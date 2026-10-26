# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


STATS_PATTERN = re.compile(
    r"Req: (?P<request_id>\S+) Finish! RequestStats\("
    r"status='(?P<status>[^']+)', reason='(?P<reason>[^']+)', "
    r"prompt_tokens=(?P<prompt_tokens>\d+), cache_tokens=(?P<cache_tokens>\d+), "
    r"output_tokens=(?P<output_tokens>\d+), cache_hit_rate=(?P<cache_hit_rate>[\d.]+), "
    r"queue_ms=(?P<queue_ms>[\d.]+), prefill_ms=(?P<prefill_ms>[\d.]+), "
    r"ttft_ms=(?P<ttft_ms>[\d.]+), total_ms=(?P<total_ms>[\d.]+), "
    r"preempt_ms=(?P<preempt_ms>[\d.]+), preempt_count=(?P<preempt_count>\d+), "
    r"decode_tps=(?P<decode_tps>[\d.]+), .*?"
    r"recv_ts=(?P<recv_ts>[\d.]+), commit_ts=(?P<commit_ts>[\d.]+), "
    r"finish_ts=(?P<finish_ts>[\d.]+)\)"
)


def rounded_median(values: list[float]) -> float:
    return round(statistics.median(values), 3)


def rounded_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--npu-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = json.loads(args.client.read_text(encoding="utf-8"))
    requested_ids = {record["request_id"] for record in client["records"]}
    server_stats = {}
    for match in STATS_PATTERN.finditer(args.server_log.read_text(encoding="utf-8")):
        values = match.groupdict()
        request_id = values.pop("request_id")
        if request_id not in requested_ids:
            continue
        values["request_id"] = request_id
        for key in ("prompt_tokens", "cache_tokens", "output_tokens", "preempt_count"):
            values[key] = int(values[key])
        for key in (
            "cache_hit_rate",
            "queue_ms",
            "prefill_ms",
            "ttft_ms",
            "total_ms",
            "preempt_ms",
            "decode_tps",
            "recv_ts",
            "commit_ts",
            "finish_ts",
        ):
            values[key] = float(values[key])
        values["decode_ms_per_token"] = (
            (values["total_ms"] - values["ttft_ms"])
            / (values["output_tokens"] - 1)
        )
        server_stats[request_id] = values

    missing = requested_ids - server_stats.keys()
    if missing:
        raise RuntimeError(f"missing server stats for request ids: {sorted(missing)}")

    samples = [
        json.loads(line)
        for line in args.npu_samples.read_text(encoding="utf-8").splitlines()
        if line
    ]
    groups = []
    for prompt_tokens, output_tokens in ((32, 16), (32, 64), (512, 16), (512, 64)):
        records = [
            record
            for record in client["records"]
            if record["prompt_tokens"] == prompt_tokens
            and record["requested_output_tokens"] == output_tokens
        ]
        stats = [server_stats[record["request_id"]] for record in records]
        windows = [(record["recv_ts"], record["finish_ts"]) for record in stats]
        group_samples = [
            sample
            for sample in samples
            if any(start <= sample["timestamp"] <= finish for start, finish in windows)
        ]
        if not group_samples:
            raise RuntimeError(f"no NPU samples for {prompt_tokens=} {output_tokens=}")
        decode_ms = [record["decode_ms_per_token"] for record in stats]
        prefill_ms = [record["prefill_ms"] for record in stats]
        total_ms = [record["total_ms"] for record in stats]
        groups.append(
            {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "repeats": len(records),
                "queue_ms_median": rounded_median([record["queue_ms"] for record in stats]),
                "prefill_ms_median": rounded_median(prefill_ms),
                "ttft_ms_median": rounded_median([record["ttft_ms"] for record in stats]),
                "decode_ms_per_token_median": rounded_median(decode_ms),
                "decode_tps_median": round(1000.0 / statistics.median(decode_ms), 3),
                "total_ms_median": rounded_median(total_ms),
                "prefill_fraction": round(statistics.median(prefill_ms) / statistics.median(total_ms), 4),
                "client_ttft_ms_median": rounded_median([record["client_ttft_ms"] for record in records]),
                "client_total_ms_median": rounded_median([record["client_total_ms"] for record in records]),
                "hbm_process_memory_mb_peak": max(sample["process_memory_mb"] for sample in group_samples),
                "hbm_usage_pct_peak": max(sample["hbm_usage_pct"] for sample in group_samples),
                "aicore_pct_mean": rounded_mean([sample["aicore_pct"] for sample in group_samples]),
                "aicore_pct_peak": max(sample["aicore_pct"] for sample in group_samples),
                "aivector_pct_mean": rounded_mean([sample["aivector_pct"] for sample in group_samples]),
                "aivector_pct_peak": max(sample["aivector_pct"] for sample in group_samples),
                "hbm_bandwidth_pct_mean": rounded_mean([sample["hbm_bandwidth_pct"] for sample in group_samples]),
                "hbm_bandwidth_pct_peak": max(sample["hbm_bandwidth_pct"] for sample in group_samples),
                "npu_utilization_pct_mean": rounded_mean([sample["npu_utilization_pct"] for sample in group_samples]),
                "npu_utilization_pct_peak": max(sample["npu_utilization_pct"] for sample in group_samples),
                "npu_sample_count": len(group_samples),
            }
        )

    result = {
        "device": {
            "npu_id": 1,
            "chip_id": 0,
            "physical_id": 2,
            "logical_torch_device": "npu:0",
        },
        "method": {
            "dtype": "bfloat16",
            "attention_backend": "mha",
            "sampling_backend": "greedy",
            "eager": True,
            "prefix_caching": False,
            "kvstore": False,
            "overlap_schedule": False,
            "warmups_per_case": 1,
            "measured_repeats_per_case": 5,
        },
        "results": groups,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
