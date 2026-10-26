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

import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend GDN tests require an NPU"
)


def _step(q, k, v, decay, beta, state, scale):
    value_heads = v.shape[-2]
    q = q.float().repeat_interleave(value_heads // q.shape[-2], dim=-2)
    k = k.float().repeat_interleave(value_heads // k.shape[-2], dim=-2)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    state = state.float() * decay.float()[..., None, None]
    prediction = torch.matmul(k.unsqueeze(-2), state).squeeze(-2)
    delta = beta.float()[..., None] * (v.float() - prediction)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
    output = torch.matmul(q.unsqueeze(-2), state).squeeze(-2) * scale
    return output, state


def test_gdn_decode_separate_pages_and_padding() -> None:
    torch.manual_seed(11)
    batch, heads, value_heads, key_dim, value_dim = 3, 2, 4, 32, 24
    dtype, device = torch.bfloat16, "npu"
    q = torch.randn(batch, 1, heads, key_dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(batch, 1, value_heads, value_dim, dtype=dtype, device=device)
    a_log = torch.randn(value_heads, device=device)
    a = torch.randn(batch, 1, value_heads, dtype=dtype, device=device)
    dt_bias = torch.randn(value_heads, device=device)
    b = torch.randn(batch, 1, value_heads, dtype=dtype, device=device)
    state = torch.randn(7, value_heads, value_dim, key_dim, dtype=dtype, device=device)
    state_before = state.clone()
    reads = torch.tensor([1, 3, -1], device=device)
    writes = torch.tensor([2, 5, -1], device=device)
    scale = key_dim**-0.5

    gate = -torch.exp(a_log.cpu())[None] * F.softplus(
        a[:2, 0].cpu().float() + dt_bias.cpu()[None]
    )
    expected_out, expected_state = _step(
        q[:2, 0].cpu(),
        k[:2, 0].cpu(),
        v[:2, 0].cpu(),
        torch.exp(gate),
        torch.sigmoid(b[:2, 0].cpu().float()),
        state_before[reads[:2]].cpu().transpose(-2, -1),
        scale,
    )
    output = gdn_decode_step(
        q,
        k,
        v,
        A_log=a_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=state,
        initial_state_indices=reads,
        output_state_indices=writes,
        solution="torch_npu",
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output[:2, 0].cpu().float(), expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        state[writes[:2]].cpu().float(),
        expected_state.transpose(-2, -1),
        rtol=2e-2,
        atol=3e-2,
    )
    assert torch.count_nonzero(output[2]).item() == 0
    untouched = torch.tensor([0, 1, 3, 4, 6], device=device)
    assert torch.equal(state[untouched], state_before[untouched])


def test_gdn_qwen_shape_decode_native_separate_pages() -> None:
    torch.manual_seed(12)
    batch, heads, dim = 2, 16, 128
    dtype, device = torch.bfloat16, "npu"
    q = torch.randn(batch, 1, heads, dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a_log = torch.randn(heads, device=device)
    a = torch.randn(batch, 1, heads, dtype=dtype, device=device)
    dt_bias = torch.randn(heads, device=device)
    b = torch.randn(batch, 1, heads, dtype=dtype, device=device)
    state = (
        torch.randn(6, heads, dim, dim, dtype=dtype, device=device) * 0.01
    )
    state_before = state.clone()
    reads = torch.tensor([1, 3], dtype=torch.int32, device=device)
    writes = torch.tensor([2, 5], dtype=torch.int32, device=device)
    scale = dim**-0.5

    gate = -torch.exp(a_log.cpu())[None] * F.softplus(
        a[:, 0].cpu().float() + dt_bias.cpu()[None]
    )
    expected_out, expected_state = _step(
        q[:, 0].cpu(),
        k[:, 0].cpu(),
        v[:, 0].cpu(),
        torch.exp(gate),
        torch.sigmoid(b[:, 0].cpu().float()),
        state_before[reads].cpu().transpose(-2, -1),
        scale,
    )
    output = gdn_decode_step(
        q,
        k,
        v,
        A_log=a_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=state,
        initial_state_indices=reads,
        output_state_indices=writes,
        solution="torch_npu",
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        output[:, 0].cpu().float(), expected_out, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        state[writes].cpu().float(),
        expected_state.transpose(-2, -1),
        rtol=2e-2,
        atol=3e-2,
    )
    untouched = torch.tensor([0, 1, 3, 4], device=device)
    assert torch.equal(state[untouched], state_before[untouched])


@pytest.mark.parametrize("lengths", ([2, 4], [9, 4]))
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float32))
def test_gdn_variable_length_prefill(lengths, state_dtype) -> None:
    torch.manual_seed(13)
    total, heads, value_heads, dim = sum(lengths), 2, 4, 32
    dtype, device = torch.bfloat16, "npu"
    q = torch.randn(1, total, heads, dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, total, value_heads, dim, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(1, total, value_heads, device=device))
    beta = torch.sigmoid(torch.randn(1, total, value_heads, dtype=dtype, device=device))
    initial = (
        torch.randn(2, value_heads, dim, dim, dtype=state_dtype, device=device)
        * 0.1
    )
    cu = torch.tensor(
        [0, lengths[0], total], dtype=torch.int32, device=device
    )
    scale = dim**-0.5

    expected_out, expected_final = [], []
    for sequence, (start, end) in enumerate(
        zip([0, lengths[0]], [lengths[0], total])
    ):
        state = initial[sequence : sequence + 1].cpu().transpose(-2, -1)
        for token in range(start, end):
            output, state = _step(
                q[:, token].cpu(),
                k[:, token].cpu(),
                v[:, token].cpu(),
                torch.exp(g[:, token].cpu()),
                beta[:, token].cpu(),
                state,
                scale,
            )
            expected_out.append(output.squeeze(0))
        expected_final.append(state.squeeze(0).transpose(-2, -1))

    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial,
        cu_seqlens=cu,
        qk_l2norm=True,
        output_final_state=True,
        solution="torch_npu",
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        result.out.cpu().float(),
        torch.stack(expected_out).unsqueeze(0),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        result.final_state.cpu().float(),
        torch.stack(expected_final),
        rtol=2e-2,
        atol=3e-2,
    )


def test_gdn_qwen_shape_prefill_crosses_native_chunk_boundary() -> None:
    torch.manual_seed(17)
    length, heads, dim = 9, 16, 128
    dtype, device = torch.bfloat16, "npu"
    q = torch.randn(1, length, heads, dim, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = F.logsigmoid(torch.randn(1, length, heads, device=device))
    beta = torch.sigmoid(
        torch.randn(1, length, heads, dtype=dtype, device=device)
    )
    initial = torch.randn(
        1, heads, dim, dim, dtype=dtype, device=device
    ) * 0.01
    cu = torch.tensor([0, length], dtype=torch.int32, device=device)
    scale = dim**-0.5

    expected_out = []
    expected_state = initial.cpu().transpose(-2, -1)
    for token in range(length):
        output, expected_state = _step(
            q[:, token].cpu(),
            k[:, token].cpu(),
            v[:, token].cpu(),
            torch.exp(g[:, token].cpu()),
            beta[:, token].cpu(),
            expected_state,
            scale,
        )
        expected_out.append(output.squeeze(0))

    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial,
        cu_seqlens=cu,
        qk_l2norm=True,
        output_final_state=True,
        solution="torch_npu",
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        result.out.cpu().float(),
        torch.stack(expected_out).unsqueeze(0),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        result.final_state.cpu().float().transpose(-2, -1),
        expected_state,
        rtol=2e-2,
        atol=3e-2,
    )
