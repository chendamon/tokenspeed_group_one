import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.entrypoints.engine import Engine
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5AttentionDecoderLayer, Qwen3_5LinearDecoderLayer


ROOT = Path(__file__).resolve().parent
MARKER = ROOT / "profile_enabled"
ACTIVE = None


def tensor_info(value):
    if not isinstance(value, torch.Tensor):
        return None
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.numel() * value.element_size()}


def append_record(record):
    if ACTIVE is None or not MARKER.exists():
        return
    rank, layer, component = ACTIVE
    row = {"rank": rank, "layer": layer, "component": component, "pid": os.getpid(), "wall_time": time.time(), **record}
    with (ROOT / f"profile_rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def synchronize():
    torch.npu.synchronize()


def wrap_layer(cls):
    original = cls.forward

    def wrapped(self, *args, **kwargs):
        global ACTIVE
        previous = ACTIVE
        rank = self.mapping.attn.tp_rank
        layer = self.layer_id
        component = type(self).__name__
        ACTIVE = (rank, layer, component)
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            hidden = args[0 if hasattr(self, "linear_attn") else 1]
        enabled = MARKER.exists()
        try:
            if enabled:
                synchronize()
                start = time.perf_counter()
            result = original(self, *args, **kwargs)
            if enabled:
                synchronize()
                append_record({"event": "component_total", "input": tensor_info(hidden), "duration_ms": (time.perf_counter() - start) * 1000.0})
            return result
        finally:
            ACTIVE = previous

    cls.forward = wrapped


wrap_layer(Qwen3_5LinearDecoderLayer)
wrap_layer(Qwen3_5AttentionDecoderLayer)


def wrap_collective(name):
    original = getattr(HcclBackend, name)

    def wrapped(self, *args, **kwargs):
        enabled = ACTIVE is not None and MARKER.exists()
        if enabled:
            synchronize()
            start = time.perf_counter()
        result = original(self, *args, **kwargs)
        if enabled:
            synchronize()
            tensors = [tensor_info(value) for value in args if isinstance(value, torch.Tensor)]
            append_record({"event": f"hccl_{name}", "tensors": tensors, "duration_ms": (time.perf_counter() - start) * 1000.0})
        return result

    setattr(HcclBackend, name, wrapped)


for collective_name in ("all_reduce", "all_gather_single", "reduce_scatter"):
    wrap_collective(collective_name)


def wrap_component(cls, method_name, layer_id, component_name, rank_attr):
    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        global ACTIVE
        previous = ACTIVE
        rank = getattr(self, rank_attr)
        ACTIVE = (rank, layer_id, component_name)
        input_value = kwargs.get("hidden_states", args[0] if args else None)
        enabled = MARKER.exists()
        try:
            if enabled:
                synchronize()
                start = time.perf_counter()
            result = original(self, *args, **kwargs)
            if enabled:
                synchronize()
                append_record({"event": "component_total", "input": tensor_info(input_value), "duration_ms": (time.perf_counter() - start) * 1000.0})
            return result
        finally:
            ACTIVE = previous

    setattr(cls, method_name, wrapped)


wrap_component(VocabParallelEmbedding, "forward", -1, "embedding", "tp_rank")
wrap_component(LogitsProcessor, "_get_logits", -2, "logits", "tp_rank")


def exact_prompt(tokenizer, token_count):
    ids = tokenizer.encode(" hello" * (token_count * 3), add_special_tokens=False)
    prompt = tokenizer.decode(ids[:token_count], skip_special_tokens=False)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    assert actual == token_count, (token_count, actual)
    return prompt


def main():
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "2,3"
    assert os.environ.get("ASCEND_VISIBLE_DEVICES") == "2,3"
    for path in ROOT.glob("profile_rank*.jsonl"):
        path.unlink()
    MARKER.unlink(missing_ok=True)

    model_path = "/home/u2601112013/models/Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompt = exact_prompt(tokenizer, 128)
    engine = None
    try:
        engine = Engine(
            model=model_path, device="npu", dtype="bfloat16", mamba_ssm_dtype="bfloat16",
            language_model_only=True, attention_backend="mha", sampling_backend="greedy",
            enforce_eager=True, disable_prefill_graph=True, disable_pdl=True, disable_autotune=True,
            disable_overlap_schedule=True, enable_prefix_caching=False, disable_kvstore=True,
            gpu_memory_utilization=0.8, retraction_snapshot_ratio=0, max_model_len=256,
            max_num_seqs=1, max_total_tokens=4096, chunked_prefill_size=256,
            attn_tp_size=2, port=30600, log_level="info",
        )
        MARKER.write_text("enabled\n", encoding="utf-8")
        start = time.perf_counter()
        result = engine.generate(prompt=prompt, sampling_params={"max_new_tokens": 4, "temperature": 0, "ignore_eos": True})
        elapsed_s = time.perf_counter() - start
        MARKER.unlink(missing_ok=True)
        output = {"model": "Qwen3.5-4B", "tp": 2, "physical_devices": [2, 3], "input_tokens": 128, "max_new_tokens": 4, "wall_time_s": elapsed_s, "result": result}
        (ROOT / "profile_output.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print("PROFILE_REQUEST_OK", json.dumps({"wall_time_s": elapsed_s}), flush=True)
    finally:
        MARKER.unlink(missing_ok=True)
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
