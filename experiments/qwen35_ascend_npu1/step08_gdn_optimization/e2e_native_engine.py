from __future__ import annotations

import json
import time

from tokenspeed.runtime.entrypoints.engine import Engine


def main() -> None:
    engine = None
    try:
        started = time.perf_counter()
        engine = Engine(
            model="/home/u2601112013/models/Qwen3.5-0.8B",
            device="npu",
            dtype="bfloat16",
            mamba_ssm_dtype="bfloat16",
            language_model_only=True,
            attention_backend="mha",
            sampling_backend="greedy",
            enforce_eager=True,
            disable_prefill_graph=True,
            disable_pdl=True,
            disable_autotune=True,
            disable_overlap_schedule=True,
            enable_prefix_caching=False,
            disable_kvstore=True,
            max_model_len=4096,
            max_num_seqs=2,
            max_total_tokens=8192,
            chunked_prefill_size=8192,
            weight_loader_prefetch_num_threads=1,
            port=30600,
            log_level="info",
        )
        ready = time.perf_counter()
        prompt_ids = [760, 6511, 314, 9338, 369]
        reference_ids = [11751, 13, 198, 760, 6511, 314, 9338, 369]
        requests = []
        for output_tokens in (8, 8, 32):
            request_started = time.perf_counter()
            result = engine.generate(
                input_ids=prompt_ids,
                sampling_params={
                    "max_new_tokens": output_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            )
            request_finished = time.perf_counter()
            output_ids = result.get("output_ids")
            requests.append(
                {
                    "requested_output_tokens": output_tokens,
                    "elapsed_s": request_finished - request_started,
                    "output_ids": output_ids,
                    "reference_prefix_exact": output_ids[:8] == reference_ids,
                    "text": result.get("text"),
                    "meta_info": {
                        key: value
                        for key, value in result.get("meta_info", {}).items()
                        if key != "logprobs"
                    },
                }
            )
        print(
            json.dumps(
                {
                    "startup_s": ready - started,
                    "prompt_ids": prompt_ids,
                    "reference_ids": reference_ids,
                    "requests": requests,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
