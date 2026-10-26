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
import subprocess
import time
from pathlib import Path


RATE_NAMES = {
    "HBM Usage Rate": "hbm_usage_pct",
    "Aicore Usage Rate": "aicore_pct",
    "Aivector Usage Rate": "aivector_pct",
    "Aicpu Usage Rate": "aicpu_pct",
    "HBM Bandwidth Usage Rate": "hbm_bandwidth_pct",
    "NPU Utilization": "npu_utilization_pct",
    "Aicube Usage Rate": "aicube_pct",
}


def query(*args: str) -> str:
    result = subprocess.run(
        ["npu-smi", "info", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def parse_sample() -> dict:
    usages = query("-t", "usages", "-i", "1", "-c", "0")
    proc_mem = query("-t", "proc-mem", "-i", "1", "-c", "0")
    sample = {"timestamp": time.time(), "process_memory_mb": 0}
    for label, key in RATE_NAMES.items():
        match = re.search(rf"{re.escape(label)}\(%\)\s*:\s*(\d+)", usages)
        sample[key] = int(match.group(1)) if match else None
    memory_values = [
        int(value)
        for value in re.findall(r"Process memory\(MB\)\s*:\s*(\d+)", proc_mem)
    ]
    sample["process_memory_mb"] = sum(memory_values)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=2400)
    args = parser.parse_args()

    with args.output.open("w", encoding="utf-8") as output:
        for _ in range(args.samples):
            output.write(json.dumps(parse_sample()) + "\n")
            output.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
