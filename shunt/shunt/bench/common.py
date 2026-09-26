"""Shared pieces of the single-node attention benchmarks.

One layer of a GQA attention block with random weights and the model's shapes,
and per-rank batches of requests with a paged KV cache holding their reused
prefixes, laid out as vLLM lays them out. The elastic-attention benchmark
calls the same compute functions as the serving runtime
(``shunt.runtime.elastic``) on these objects.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist

from ..runtime.elastic import _RecvMeta

BLOCK = 16


def dist_init() -> tuple[int, int]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda",
                                                           torch.cuda.current_device()))
    return rank, world


def attention_module(m, dtype=torch.bfloat16, seed: int = 0):
    """Stand-in for vLLM's Qwen3-MoE attention module with random weights."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    s = 0.02
    w_qkv = torch.randn(m.q_dim + 2 * m.kv_dim, m.hidden, device=dev, dtype=dtype,
                        generator=g) * s
    w_o = torch.randn(m.hidden, m.q_dim, device=dev, dtype=dtype, generator=g) * s
    norm_w = torch.ones(m.head_dim, device=dev, dtype=dtype)
    cos_sin = torch.randn(1 << 17, m.head_dim, device=dev, dtype=dtype, generator=g)

    def rms(x):
        y = torch.empty_like(x)
        ops.rms_norm(y.view(-1, m.head_dim), x.reshape(-1, m.head_dim), norm_w, 1e-6)
        return y

    def rope(positions, q, k):
        ops.rotary_embedding(positions, q, k, m.head_dim, cos_sin, True)
        return q, k

    attn = SimpleNamespace(
        impl=SimpleNamespace(vllm_flash_attn_version=get_flash_attn_version()),
        kv_cache_dtype="auto",
        _k_scale=torch.ones((), device=dev), _v_scale=torch.ones((), device=dev))
    return SimpleNamespace(qkv_proj=SimpleNamespace(weight=w_qkv, bias=None),
                           o_proj=SimpleNamespace(weight=w_o), q_norm=rms, k_norm=rms,
                           rotary_emb=rope, scaling=m.head_dim ** -0.5, attn=attn)


@dataclass
class RankBatch:
    """One rank's requests for one layer: inputs, metadata, and paged KV."""

    reqs: list[tuple[int, int]]
    T: int
    P: int
    x: torch.Tensor
    positions: torch.Tensor
    md: SimpleNamespace
    kv_cache: torch.Tensor
    slot_mapping: torch.Tensor
    prefix_slots: torch.Tensor | None

    @classmethod
    def build(cls, m, reqs: list[tuple[int, int]], dtype=torch.bfloat16) -> "RankBatch":
        dev = "cuda"
        q_lens = [n for _, n in reqs]
        seq_lens = [p + n for p, n in reqs]
        T, P = sum(q_lens), sum(p for p, _ in reqs)
        blocks = [math.ceil(s / BLOCK) for s in seq_lens] or [0]
        nb = sum(blocks) + 1
        kv = torch.randn(2, nb, BLOCK, m.num_kv_heads, m.head_dim, device=dev, dtype=dtype)
        width = max(blocks) if blocks else 1
        table = torch.zeros(max(1, len(reqs)), max(1, width), device=dev, dtype=torch.int32)
        slots, pslots = [], []
        nxt = 0
        for i, ((p, n), b) in enumerate(zip(reqs, blocks)):
            ids = torch.arange(nxt, nxt + b, device=dev, dtype=torch.int32)
            table[i, :b] = ids
            pos = torch.arange(p + n, device=dev)
            sl = ids.long()[pos // BLOCK] * BLOCK + pos % BLOCK
            pslots.append(sl[:p])
            slots.append(sl[p:])
            nxt += b
        cu_q = torch.zeros(len(reqs) + 1, device=dev, dtype=torch.int32)
        if reqs:
            cu_q[1:] = torch.cumsum(torch.tensor(q_lens, device=dev), 0)
        md = SimpleNamespace(
            query_start_loc=cu_q, max_query_len=max(q_lens, default=0),
            seq_lens=torch.tensor(seq_lens, device=dev, dtype=torch.int32),
            max_seq_len=max(seq_lens, default=0), block_table=table,
            num_actual_tokens=T)
        positions = (torch.cat([torch.arange(p, p + n, device=dev) for p, n in reqs])
                     if reqs else torch.zeros(0, dtype=torch.long, device=dev))
        return cls(reqs=reqs, T=T, P=P,
                   x=torch.randn(T, m.hidden, device=dev, dtype=dtype),
                   positions=positions, md=md, kv_cache=kv,
                   slot_mapping=torch.cat(slots) if slots else torch.zeros(
                       0, dtype=torch.long, device=dev),
                   prefix_slots=torch.cat(pslots) if P else None)

    def recv_meta(self) -> _RecvMeta:
        """The metadata a recipient of this batch's heads derives."""
        dev = "cuda"
        q = torch.tensor([n for _, n in self.reqs], device=dev, dtype=torch.int64)
        ctx = torch.tensor([p for p, _ in self.reqs], device=dev, dtype=torch.int64)
        R = len(self.reqs)
        seg = ctx + q
        cu_q = torch.zeros(R + 1, dtype=torch.int32, device=dev)
        cu_q[1:] = torch.cumsum(q, 0)
        cu_k = torch.zeros(R + 1, dtype=torch.int32, device=dev)
        cu_k[1:] = torch.cumsum(seg, 0)
        r_new = torch.repeat_interleave(torch.arange(R, device=dev), q)
        off = torch.arange(self.T, device=dev) - cu_q[:-1].long()[r_new]
        idx_new = cu_k[:-1].long()[r_new] + ctx[r_new] + off
        idx_prefix = None
        if self.P:
            r_pre = torch.repeat_interleave(torch.arange(R, device=dev), ctx)
            starts = torch.cumsum(ctx, 0) - ctx
            idx_prefix = cu_k[:-1].long()[r_pre] + torch.arange(self.P, device=dev) \
                - starts[r_pre]
        return _RecvMeta(T=self.T, P=self.P, cu_q=cu_q, cu_k=cu_k,
                         max_q=int(q.max()) if R else 0,
                         max_k=int(seg.max()) if R else 0,
                         positions=ctx[r_new] + off, idx_new=idx_new,
                         idx_prefix=idx_prefix)


def time_region(fn, reps: int, warmup: int = 2) -> float:
    """Median over ``reps`` of the slowest rank's time for ``fn`` (seconds);
    every repetition starts after a barrier."""
    import numpy as np

    times = []
    for i in range(warmup + reps):
        dist.barrier()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        t = torch.tensor([s.elapsed_time(e) * 1e-3], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if i >= warmup:
            times.append(float(t))
    return float(np.median(times))
