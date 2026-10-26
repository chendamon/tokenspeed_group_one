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
import statistics
import time
from pathlib import Path

import requests


PROMPT_VARIANTS = (" hello", " token", " world", " model", " data")
CASES = ((32, 16), (32, 64), (512, 16), (512, 64))


def exact_prompt(length: int, variant: int) -> str:
    # Each fragment is one token in the Qwen3.5 tokenizer. The final variant
    # prevents accidental cross-request identity while preserving exact length.
    return " hello" * (length - 1) + PROMPT_VARIANTS[variant % len(PROMPT_VARIANTS)]


def stream_completion(
    url: str, prompt_tokens: int, output_tokens: int, variant: int
) -> dict:
    payload = {
        "model": "qwen35-0.8b",
        "prompt": exact_prompt(prompt_tokens, variant),
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    start = time.perf_counter()
    first_text = None
    response_id = None
    output_text = ""
    with requests.post(url, json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line or not raw_line.startswith(b"data: "):
                continue
            data = raw_line[6:]
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            response_id = response_id or chunk.get("id")
            choices = chunk.get("choices") or []
            text = choices[0].get("text", "") if choices else ""
            if text and first_text is None:
                first_text = time.perf_counter()
            output_text += text
    finish = time.perf_counter()
    if first_text is None:
        raise RuntimeError("stream finished without a non-empty token chunk")
    return {
        "request_id": response_id,
        "prompt_tokens": prompt_tokens,
        "requested_output_tokens": output_tokens,
        "client_ttft_ms": (first_text - start) * 1e3,
        "client_total_ms": (finish - start) * 1e3,
        "output_text": output_text,
    }


def median(values: list[float]) -> float:
    return round(statistics.median(values), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/completions"
    records = []
    variant = 0
    for prompt_tokens, output_tokens in CASES:
        warmup = stream_completion(url, prompt_tokens, output_tokens, variant)
        variant += 1
        print(
            f"warmup input={prompt_tokens} output={output_tokens} "
            f"total_ms={warmup['client_total_ms']:.3f}",
            flush=True,
        )
        for repeat in range(args.repeats):
            record = stream_completion(url, prompt_tokens, output_tokens, variant)
            variant += 1
            record["repeat"] = repeat
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = []
    for prompt_tokens, output_tokens in CASES:
        group = [
            record
            for record in records
            if record["prompt_tokens"] == prompt_tokens
            and record["requested_output_tokens"] == output_tokens
        ]
        summary.append(
            {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "repeats": len(group),
                "client_ttft_ms_median": median(
                    [record["client_ttft_ms"] for record in group]
                ),
                "client_total_ms_median": median(
                    [record["client_total_ms"] for record in group]
                ),
            }
        )

    args.output.write_text(
        json.dumps(
            {"records": records, "client_summary": summary},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
