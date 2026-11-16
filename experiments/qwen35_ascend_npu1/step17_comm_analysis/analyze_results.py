import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def nearest_rank(values, q):
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def phase_for(record):
    info = record.get("input")
    if info and info.get("shape"):
        tokens = info["shape"][0]
    else:
        tensors = record.get("tensors") or []
        shape = tensors[0].get("shape") if tensors else None
        tokens = shape[0] if shape else None
    if tokens == 128:
        return "prefill"
    if tokens == 1:
        return "decode"
    return f"other_{tokens}"


def analyze_profile():
    summary = {"ranks": {}}
    for rank in (0, 1):
        rows = [json.loads(line) for line in (ROOT / f"profile_rank{rank}.jsonl").read_text().splitlines() if line]
        rank_summary = {}
        for phase in sorted({phase_for(row) for row in rows}):
            phase_rows = [row for row in rows if phase_for(row) == phase]
            components = [row for row in phase_rows if row["event"] == "component_total"]
            collectives = [row for row in phase_rows if row["event"].startswith("hccl_")]
            by_op = defaultdict(list)
            for row in collectives:
                by_op[row["event"]].append(row["duration_ms"])
            component_ms = sum(row["duration_ms"] for row in components)
            comm_ms = sum(row["duration_ms"] for row in collectives)
            rank_summary[phase] = {
                "component_count": len(components),
                "component_ms": component_ms,
                "collective_count": len(collectives),
                "collective_ms": comm_ms,
                "collective_share_of_component_percent": 100.0 * comm_ms / component_ms,
                "compute_dispatch_proxy_ms": component_ms - comm_ms,
                "operations": {op: {"count": len(values), "sum_ms": sum(values), "p50_ms": statistics.median(values), "p95_ms": nearest_rank(values, 0.95)} for op, values in sorted(by_op.items())},
            }
        summary["ranks"][str(rank)] = rank_summary
    output = json.loads((ROOT / "profile_output.json").read_text())
    summary["request_wall_time_s"] = output["wall_time_s"]
    return summary


def analyze_granularity():
    by_rank = {}
    for rank in (0, 1):
        payload = json.loads((ROOT / f"allreduce_rank{rank}.json").read_text())
        by_rank[str(rank)] = payload["results"]
    combined = []
    cases = [row["case"] for row in by_rank["0"]]
    for case in cases:
        rows = [next(row for row in by_rank[str(rank)] if row["case"] == case) for rank in (0, 1)]
        combined.append({
            "case": case, "bytes_per_call": rows[0]["bytes_per_call"],
            "calls_per_repeat": rows[0]["calls_per_repeat"],
            "critical_p50_ms": max(row["p50_ms"] for row in rows),
            "critical_p95_ms": max(row["p95_ms"] for row in rows),
            "critical_effective_payload_gbps": min(row["effective_payload_gbps"] for row in rows),
            "rank_p50_ms": [row["p50_ms"] for row in rows],
        })
    fused = combined[0]["critical_p50_ms"]
    for row in combined:
        row["slowdown_vs_fused"] = row["critical_p50_ms"] / fused
    return {"by_rank": by_rank, "combined_critical_path": combined}


def main():
    summary = {
        "profile": analyze_profile(),
        "allreduce_granularity": analyze_granularity(),
        "method_note": "Model profile inserts a device synchronize before and after each measured collective and component. It exposes non-overlapped service and wait time, but perturbs the original timeline and must not be added to the uninstrumented end-to-end benchmark.",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
