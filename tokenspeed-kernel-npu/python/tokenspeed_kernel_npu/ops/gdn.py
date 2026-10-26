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

"""Ascend implementations of Gated DeltaNet scans.

The public dispatcher and state layout remain in ``tokenspeed-kernel``. This
module supplies the device math that the portable Triton kernels cannot
currently compile with Triton-Ascend. BF16 state uses torch_npu's fused
recurrent operator; FP32 state keeps the correctness-first tensor fallback.
Inputs, outputs, and the K-last state pool retain their original layout.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torch_npu


_NATIVE_MAX_SEQUENCE_LENGTH = 8
_NATIVE_ENABLED = os.getenv("TOKENSPEED_ASCEND_GDN_NATIVE", "1") != "0"


def _expand_qk_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    num_q_heads = x.shape[-2]
    if num_v_heads % num_q_heads:
        raise ValueError("num_v_heads must be divisible by num_q_heads")
    return x.repeat_interleave(num_v_heads // num_q_heads, dim=-2)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + 1e-6)


def _native_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
) -> bool:
    """Whether torch_npu's fused recurrent GDN accepts these tensors."""
    return (
        _NATIVE_ENABLED
        and q.dtype == k.dtype == v.dtype == state.dtype == torch.bfloat16
        and q.device.type == k.device.type == v.device.type == state.device.type == "npu"
        and q.shape[-2] <= 256
        and v.shape[-2] <= 256
        and q.shape[-1] <= 256
        and v.shape[-1] <= 256
        # The installed runtime aborts on some documented GQA/small-dimension
        # combinations. Keep the fused path to the Qwen3.5 shape family that
        # is covered by numerical tests; the tensor recurrence handles others.
        and q.shape[-2] == v.shape[-2]
        and q.shape[-1] % 128 == 0
        and v.shape[-1] % 128 == 0
    )


def _native_qk(x: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return x
    return _l2norm(x).to(x.dtype)


def _delta_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    scale: float,
    qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one vectorized token step to FLA-layout ``[B, Hv, K, V]`` state."""
    num_v_heads = v.shape[-2]
    q = _expand_qk_heads(q, num_v_heads).float()
    k = _expand_qk_heads(k, num_v_heads).float()
    if qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    v = v.float()
    state = state * decay.float().unsqueeze(-1).unsqueeze(-1)
    prediction = torch.matmul(k.unsqueeze(-2), state).squeeze(-2)
    delta = beta.float().unsqueeze(-1) * (v - prediction)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    output = torch.matmul(q.unsqueeze(-2), state).squeeze(-2) * scale
    return output, state


def gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Run one GDN decode token and update the K-last state pool on NPU."""
    if q.shape[1] != 1:
        raise ValueError("Ascend GDN decode requires a single token")
    if scale is None:
        scale = q.shape[-1] ** -0.5

    read_indices = initial_state_indices.to(torch.int64)
    write_indices = (
        read_indices
        if output_state_indices is None
        else output_state_indices.to(torch.int64)
    )
    valid_read = read_indices >= 0
    valid_write = write_indices >= 0

    if _native_supported(q, k, v, initial_state):
        valid_positions = torch.nonzero(
            valid_read & valid_write, as_tuple=False
        ).flatten()
        output = v.new_zeros((v.shape[0], v.shape[-2], v.shape[-1]))
        if valid_positions.numel() == 0:
            return output.unsqueeze(1)

        native_reads = read_indices.index_select(0, valid_positions)
        native_writes = write_indices.index_select(0, valid_positions)
        # Gather before writing so separate output pages cannot clobber another
        # request's input page when a scheduler reuses state storage.
        source_state = initial_state.index_select(0, native_reads)
        initial_state.index_copy_(0, native_writes, source_state)

        native_q = _native_qk(
            q[:, 0].index_select(0, valid_positions), use_qk_l2norm
        )
        native_k = _native_qk(
            k[:, 0].index_select(0, valid_positions), use_qk_l2norm
        )
        native_v = v[:, 0].index_select(0, valid_positions)
        native_a = a[:, 0].index_select(0, valid_positions)
        native_b = b[:, 0].index_select(0, valid_positions)
        native_gate = -torch.exp(A_log.float()).reshape(1, -1) * F.softplus(
            native_a.float() + dt_bias.float().reshape(1, -1)
        )
        native_output = torch_npu.npu_recurrent_gated_delta_rule(
            native_q,
            native_k,
            native_v,
            initial_state,
            beta=torch.sigmoid(native_b.float()).to(q.dtype),
            scale=float(scale),
            actual_seq_lengths=torch.ones(
                valid_positions.numel(), dtype=torch.int32, device=q.device
            ),
            ssm_state_indices=native_writes.to(torch.int32),
            g=native_gate,
        )
        output.index_copy_(0, valid_positions, native_output)
        return output.unsqueeze(1)

    safe_read = read_indices.clamp_min(0)
    state = initial_state.index_select(0, safe_read).transpose(-2, -1).float()
    state = torch.where(
        valid_read.reshape(-1, 1, 1, 1),
        state,
        torch.zeros_like(state),
    )

    gate = -torch.exp(A_log.float()).reshape(1, -1) * F.softplus(
        a[:, 0].float() + dt_bias.float().reshape(1, -1)
    )
    output, state = _delta_step(
        q[:, 0],
        k[:, 0],
        v[:, 0],
        torch.exp(gate),
        torch.sigmoid(b[:, 0].float()),
        state,
        scale=float(scale),
        qk_l2norm=use_qk_l2norm,
    )
    output = torch.where(
        valid_read.reshape(-1, 1, 1), output, torch.zeros_like(output)
    )

    valid_positions = torch.nonzero(write_indices >= 0, as_tuple=False).flatten()
    initial_state.index_copy_(
        0,
        write_indices.index_select(0, valid_positions),
        state.transpose(-2, -1)
        .to(initial_state.dtype)
        .index_select(0, valid_positions),
    )
    return output.unsqueeze(1).to(q.dtype)


def gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
):
    """Run a variable-length GDN prefill scan using Ascend tensor ops.

    BF16 state uses torch_npu's fused recurrent operator in chunks of at most
    eight tokens, its documented sequence-length limit. Other state dtypes use
    the existing token recurrence. No model or cache-management logic is
    duplicated here.
    """
    if output_h:
        raise NotImplementedError("Ascend GDN prefill does not expose checkpoints")
    if q.shape[0] != 1:
        raise ValueError("Ascend GDN prefill expects packed batch dimension 1")
    if scale is None:
        scale = q.shape[-1] ** -0.5

    from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult

    bounds = cu_seqlens.detach().cpu().to(torch.int64).tolist()
    if _native_supported(q, k, v, initial_state):
        native_q = _native_qk(q[0], qk_l2norm)
        native_k = _native_qk(k[0], qk_l2norm)
        native_state = initial_state.clone()
        native_outputs: list[torch.Tensor] = []
        for seq_idx, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            for chunk_start in range(start, end, _NATIVE_MAX_SEQUENCE_LENGTH):
                chunk_end = min(chunk_start + _NATIVE_MAX_SEQUENCE_LENGTH, end)
                chunk_length = chunk_end - chunk_start
                native_outputs.append(
                    torch_npu.npu_recurrent_gated_delta_rule(
                        native_q[chunk_start:chunk_end],
                        native_k[chunk_start:chunk_end],
                        v[0, chunk_start:chunk_end],
                        native_state,
                        beta=beta[0, chunk_start:chunk_end].to(q.dtype),
                        scale=float(scale),
                        actual_seq_lengths=torch.tensor(
                            [chunk_length], dtype=torch.int32, device=q.device
                        ),
                        ssm_state_indices=torch.full(
                            (chunk_length,),
                            seq_idx,
                            dtype=torch.int32,
                            device=q.device,
                        ),
                        g=g[0, chunk_start:chunk_end].float(),
                    )
                )
        if native_outputs:
            out = torch.cat(native_outputs, dim=0).unsqueeze(0)
        else:
            out = q.new_empty((1, 0, v.shape[-2], v.shape[-1]))
        final_state = native_state if output_final_state else None
        return GdnChunkPrefillResult(out=out, final_state=final_state)

    outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for seq_idx, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = initial_state[seq_idx].transpose(-2, -1).float().unsqueeze(0)
        for token_idx in range(start, end):
            output, state = _delta_step(
                q[:, token_idx],
                k[:, token_idx],
                v[:, token_idx],
                torch.exp(g[:, token_idx]),
                beta[:, token_idx],
                state,
                scale=float(scale),
                qk_l2norm=qk_l2norm,
            )
            outputs.append(output.squeeze(0).to(q.dtype))
        final_states.append(
            state.squeeze(0).transpose(-2, -1).to(initial_state.dtype)
        )

    if outputs:
        out = torch.stack(outputs, dim=0).unsqueeze(0)
    else:
        out = q.new_empty((1, 0, v.shape[-2], v.shape[-1]))
    final_state = torch.stack(final_states, dim=0) if output_final_state else None
    return GdnChunkPrefillResult(out=out, final_state=final_state)


__all__ = ["gdn_chunk_prefill", "gdn_decode_step"]
