"""Profile the worker-compute and expert-compute functions on one GPU.

Times one layer's kernels with the model's shapes on the target GPU, as vLLM
runs them: the QKV and output projections, QK normalization, rotary embedding,
and paged FlashAttention for batches of requests with ``P`` reused-prefix and
``N`` fresh tokens (the attention time), the router gate (the gate time), and
vLLM's fused MoE kernel for a given number of token-expert assignments on one
GPU (the expert time). It then fits the coefficients of
:class:`shunt.compute_model.ComputeCoefficients` by least squares and writes
them, with every measured sample, to a JSON file that the planner and the
proxy load through ``compute_profile``.

Example::

    python -m shunt.profiling.compute --model /models/Qwen3-235B-A22B \\
        --ep 16 --out profiles/compute.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..compute_model import attention_pairs
from ..config import ModelSpec


def _timeit(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e-3)
    return float(np.median(times))


class LayerKernels:
    """One layer's attention, gate, and expert kernels with random weights."""

    def __init__(self, m: ModelSpec, ep: int, dtype=torch.bfloat16,
                 block_size: int = 16, max_tokens: int = 65536):
        from vllm import _custom_ops as ops
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        self.m, self.ops, self.dtype = m, ops, dtype
        self.fa_version = get_flash_attn_version()
        dev = "cuda"
        s = 0.02
        H, D = m.hidden, m.head_dim
        self.w_qkv = torch.randn(m.q_dim + 2 * m.kv_dim, H, device=dev, dtype=dtype) * s
        self.w_o = torch.randn(H, m.q_dim, device=dev, dtype=dtype) * s
        self.w_gate = torch.randn(m.num_experts, H, device=dev, dtype=dtype) * s
        self.norm_w = torch.ones(D, device=dev, dtype=dtype)
        rot = torch.randn(max_tokens, D, device=dev, dtype=dtype)
        self.cos_sin = rot
        self.block_size = block_size
        n_blocks = math.ceil(max_tokens / block_size) + 8
        self.k_cache = torch.randn(n_blocks, block_size, m.num_kv_heads, D, device=dev,
                                   dtype=dtype)
        self.v_cache = torch.randn_like(self.k_cache)
        self.local_experts = max(1, m.num_experts // ep)
        E, I = self.local_experts, m.moe_intermediate
        self.w1 = torch.randn(E, 2 * I, H, device=dev, dtype=dtype) * s
        self.w2 = torch.randn(E, H, I, device=dev, dtype=dtype) * s

    def attention(self, reqs: list[tuple[int, int]]):
        """A callable running one layer's attention for requests ``(P, N)``."""
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        m, dev, bs = self.m, "cuda", self.block_size
        T = sum(n for _, n in reqs)
        x = torch.randn(T, m.hidden, device=dev, dtype=self.dtype)
        q_lens = [n for _, n in reqs]
        seq_lens = [p + n for p, n in reqs]
        cu_q = torch.tensor([0] + list(itertools.accumulate(q_lens)), device=dev,
                            dtype=torch.int32)
        seqused = torch.tensor(seq_lens, device=dev, dtype=torch.int32)
        blocks_per = [math.ceil(s / bs) for s in seq_lens]
        width = max(blocks_per)
        table = torch.zeros(len(reqs), width, device=dev, dtype=torch.int32)
        nxt = 0
        for i, b in enumerate(blocks_per):
            table[i, :b] = torch.arange(nxt, nxt + b, device=dev, dtype=torch.int32)
            nxt += b
        positions = torch.cat([torch.arange(p, p + n, device=dev) for p, n in reqs])
        out = torch.empty(T, m.num_q_heads, m.head_dim, device=dev, dtype=self.dtype)
        max_q, max_k = max(q_lens), max(seq_lens)

        def run():
            qkv = F.linear(x, self.w_qkv)
            q, k, v = qkv.split([m.q_dim, m.kv_dim, m.kv_dim], dim=-1)
            qn = torch.empty_like(q)
            kn = torch.empty_like(k)
            self.ops.rms_norm(qn.view(-1, m.head_dim), q.reshape(-1, m.head_dim),
                              self.norm_w, 1e-6)
            self.ops.rms_norm(kn.view(-1, m.head_dim), k.reshape(-1, m.head_dim),
                              self.norm_w, 1e-6)
            self.ops.rotary_embedding(positions, qn, kn, m.head_dim, self.cos_sin, True)
            flash_attn_varlen_func(
                q=qn.view(T, m.num_q_heads, m.head_dim), k=self.k_cache, v=self.v_cache,
                out=out, cu_seqlens_q=cu_q, max_seqlen_q=max_q, seqused_k=seqused,
                max_seqlen_k=max_k, softmax_scale=m.head_dim ** -0.5, causal=True,
                block_table=table, fa_version=self.fa_version)
            return F.linear(out.view(T, m.q_dim), self.w_o)
        return run

    def gate(self, T: int):
        x = torch.randn(T, self.m.hidden, device="cuda", dtype=self.dtype)
        k = self.m.top_k

        def run():
            logits = F.linear(x, self.w_gate)
            return torch.topk(torch.softmax(logits.float(), dim=-1), k, dim=-1)
        return run

    def experts(self, assignments: int):
        """``assignments`` token-expert pairs spread evenly over the local experts."""
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        T = max(1, assignments)
        x = torch.randn(T, self.m.hidden, device="cuda", dtype=self.dtype)
        ids = (torch.arange(T, device="cuda") % self.local_experts).to(torch.int32)
        ids = ids.view(T, 1)
        w = torch.ones(T, 1, device="cuda", dtype=torch.float32)

        def run():
            return fused_experts(x, self.w1, self.w2, w, ids)
        return run


def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.maximum(coef, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="HF model dir or config.json")
    ap.add_argument("--ep", type=int, default=16, help="EP width of the prefill")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fresh", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--prefix", type=int, nargs="+", default=[0, 1024, 4096, 8192])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--assignments", type=int, nargs="+",
                    default=[1024, 4096, 16384, 32768, 65536, 131072])
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()

    m = ModelSpec.from_hf_config(a.model)
    max_tokens = max(a.batch) * (max(a.prefix) + max(a.fresh)) + 1024
    ker = LayerKernels(m, a.ep, max_tokens=max_tokens)
    samples = {"attention": [], "gate": [], "experts": []}
    for b, p, n in itertools.product(a.batch, a.prefix, a.fresh):
        reqs = [(p, n)] * b
        t = _timeit(ker.attention(reqs), iters=a.iters)
        samples["attention"].append({"batch": b, "prefix": p, "fresh": n, "s": t})
        print(f"attention batch={b} P={p} N={n}: {t * 1e3:.3f} ms", flush=True)
    for n in sorted({b * n for b in a.batch for n in a.fresh}):
        t = _timeit(ker.gate(n), iters=a.iters)
        samples["gate"].append({"tokens": n, "s": t})
    for x in a.assignments:
        t = _timeit(ker.experts(x), iters=a.iters)
        samples["experts"].append({"assignments": x, "s": t})
        print(f"experts assignments={x}: {t * 1e3:.3f} ms", flush=True)

    att = samples["attention"]
    X = np.array([[s["batch"] * s["fresh"],
                   s["batch"] * attention_pairs(s["prefix"], s["fresh"])] for s in att])
    c_tok, c_pair = _fit(X, np.array([s["s"] for s in att]))
    g = samples["gate"]
    (c_gate,) = _fit(np.array([[s["tokens"]] for s in g]), np.array([s["s"] for s in g]))
    e = samples["experts"]
    c_e0, c_e = _fit(np.array([[1.0, s["assignments"]] for s in e]),
                     np.array([s["s"] for s in e]))
    out = {
        "gpu": torch.cuda.get_device_name(), "model": m.name, "ep": a.ep,
        "coefficients": {"c_tok": float(c_tok), "c_pair": float(c_pair),
                         "c_gate": float(c_gate), "c_expert": float(c_e),
                         "c_expert0": float(c_e0)},
        "samples": samples,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}: {out['coefficients']}")


if __name__ == "__main__":
    main()
