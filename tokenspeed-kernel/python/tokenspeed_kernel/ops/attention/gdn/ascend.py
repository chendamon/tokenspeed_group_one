# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ascend Gated DeltaNet kernel registrations."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.gdn import gdn_chunk_prefill as _gdn_chunk_prefill
    from tokenspeed_kernel_npu.ops.gdn import gdn_decode_step as _gdn_decode_step

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _DTYPES = {torch.float16, torch.bfloat16}

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="ascend_gdn_chunk_prefill",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PORTABLE,
        traits={
            "output_h": frozenset({False}),
            "qk_l2norm": frozenset({False, True}),
        },
        features={"portability"},
    )
    def gdn_chunk_prefill(**kwargs):
        return _gdn_chunk_prefill(**kwargs)

    @register_kernel(
        "attention",
        "gdn_decode_step",
        name="ascend_gdn_decode_step",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PORTABLE,
        traits={"strided_qkv": frozenset({False, True})},
        features={"portability"},
    )
    def gdn_decode_step(**kwargs):
        return _gdn_decode_step(**kwargs)


__all__ = ["gdn_chunk_prefill", "gdn_decode_step"]
