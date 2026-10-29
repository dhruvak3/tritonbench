import argparse
from functools import partial

import torch
from torch.nn.functional import scaled_dot_product_attention
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_metric,
    register_x_val,
)
from tritonbench.utils.triton_utils import has_tlx_op


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["tflops"]

    def __init__(self, tb_args, extra_args=None):
        super().__init__(tb_args, extra_args)
        parser = argparse.ArgumentParser()
        parser.add_argument("--tlx-space", choices=["heuristic"], default=None)
        self.tlx_space = parser.parse_args(self.extra_args).tlx_space

    def get_input_iter(self):
        raise ValueError("Use --suite sm90_all")

    @register_benchmark(baseline=True, fwd_only=True)
    def aten_sdpa(self, q, k, v, cu_q, cu_k, lens_q, lens_k, causal):
        batch, max_q, max_k = len(lens_q), max(lens_q), max(lens_k)

        def pad(tensor, lengths, longest):
            padded = tensor.new_zeros((batch, longest, *tensor.shape[1:]))
            offset = 0
            for b, length in enumerate(lengths):
                padded[b, :length] = tensor[offset : offset + length]
                offset += length
            return padded.transpose(1, 2)

        qp, kp, vp = pad(q, lens_q, max_q), pad(k, lens_k, max_k), pad(v, lens_k, max_k)
        q_lengths = torch.tensor(lens_q, device=q.device)[:, None, None]
        k_lengths = torch.tensor(lens_k, device=q.device)[:, None, None]
        rows = torch.arange(max_q, device=q.device)[None, :, None]
        cols = torch.arange(max_k, device=q.device)[None, None, :]
        mask = (rows < q_lengths) & (cols < k_lengths)
        if causal:
            mask &= cols <= rows + k_lengths - q_lengths
        mask = mask[:, None]
        valid_q = rows[:, :, 0] < q_lengths[:, :, 0]
        self._varlen_unpad = lambda output: output.transpose(1, 2)[valid_q]
        backend = torch.ops.aten._fused_sdp_choice(qp, kp, vp, mask)
        if self._tlx_l2:
            self._tlx_evidence.append(
                {
                    "kind": "baseline_backend",
                    "input_id": self._cur_input_id,
                    "backend": int(backend),
                }
            )

        def run():
            return scaled_dot_product_attention(qp, kp, vp, attn_mask=mask)

        return run

    @register_benchmark(
        enabled=has_tlx_op("flash_attn_varlen"), fwd_only=True, tags=["tlx"]
    )
    def tlx_ops_flash_attn_varlen(self, q, k, v, cu_q, cu_k, lens_q, lens_k, causal):
        from triton.tlx.ops import flash_attn_varlen

        return partial(
            flash_attn_varlen,
            q,
            k,
            v,
            cu_q,
            cu_k,
            max(lens_q),
            max(lens_k),
            causal=causal,
            space=self.tlx_space,
        )

    @register_x_val(label="(B, Q lengths, KV lengths, H, D, dtype, causal)")
    def get_x_val(self, inputs):
        q, k, v, cu_q, cu_k, lens_q, lens_k, causal = inputs
        return str(
            (len(lens_q), lens_q, lens_k, q.shape[1], q.shape[2], str(q.dtype), causal)
        )

    @register_metric()
    def flops(self, fn_name, inputs, metrics):
        q, k, v, cu_q, cu_k, lens_q, lens_k, causal = inputs
        return sum(
            4 * n * m * q.shape[1] * q.shape[2] for n, m in zip(lens_q, lens_k)
        ) * (0.5 if causal else 1)
