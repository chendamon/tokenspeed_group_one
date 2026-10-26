import math

import torch
import torch_npu


def reference(q, k, v, g, beta, state, scale):
    q = q.float()
    k = k.float()
    v = v.float()
    state = state.float()
    outputs = []
    for token in range(q.shape[0]):
        state = state * torch.exp(g[token]).unsqueeze(-1).unsqueeze(-1)
        prediction = torch.matmul(k[token].unsqueeze(-2), state).squeeze(-2)
        delta = beta[token].float().unsqueeze(-1) * (v[token] - prediction)
        state = state + k[token].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.matmul(q[token].unsqueeze(-2), state).squeeze(-2) * scale)
    return torch.stack(outputs), state


def native_chunked(q, k, v, g, beta, state, scale, chunk_size=8):
    state = state.clone()
    outputs = []
    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])
        length = end - start
        output = torch_npu.npu_recurrent_gated_delta_rule(
            q[start:end],
            k[start:end],
            v[start:end],
            state,
            beta=beta[start:end],
            scale=scale,
            actual_seq_lengths=torch.tensor([length], dtype=torch.int32, device="npu"),
            ssm_state_indices=torch.zeros(length, dtype=torch.int32, device="npu"),
            g=g[start:end],
        )
        outputs.append(output)
    return torch.cat(outputs), state


def main():
    torch.manual_seed(20261022)
    nk = nv = 16
    dk = dv = 128
    scale = dk**-0.5
    for length in (1, 2, 8, 9, 17):
        q_cpu = torch.randn(length, nk, dk, dtype=torch.bfloat16)
        k_cpu = torch.randn(length, nk, dk, dtype=torch.bfloat16)
        q_cpu = torch.nn.functional.normalize(q_cpu.float(), dim=-1).to(torch.bfloat16)
        k_cpu = torch.nn.functional.normalize(k_cpu.float(), dim=-1).to(torch.bfloat16)
        v_cpu = torch.randn(length, nv, dv, dtype=torch.bfloat16)
        g_cpu = -torch.rand(length, nv, dtype=torch.float32)
        beta_cpu = torch.rand(length, nv, dtype=torch.bfloat16)
        state_cpu = (torch.randn(1, nv, dv, dk) * 0.01).to(torch.bfloat16)
        expected_out, expected_state = reference(
            q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu, state_cpu[0].transpose(-2, -1), scale
        )
        got_out, got_state = native_chunked(
            q_cpu.npu(),
            k_cpu.npu(),
            v_cpu.npu(),
            g_cpu.npu(),
            beta_cpu.npu(),
            state_cpu.npu(),
            scale,
        )
        torch.npu.synchronize()
        out_diff = (got_out.cpu().float() - expected_out).abs()
        state_diff = (
            got_state.cpu()[0].float().transpose(-2, -1) - expected_state
        ).abs()
        print(
            f"length={length} "
            f"out_max={out_diff.max().item():.6f} out_mean={out_diff.mean().item():.6f} "
            f"state_max={state_diff.max().item():.6f} state_mean={state_diff.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
