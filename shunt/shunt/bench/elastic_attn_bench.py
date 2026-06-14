"""Fig 18: elastic-attention NVLink overhead, 8-GPU single node (§4.4).

Measures the only true runtime cost of elastic attention -- the two NVLink
transfers around the attention compute -- as a fraction of the (transfer +
attention) phase, across straggler severity and helper count.

The straggler (rank 0) offloads a fraction of its query heads to ``helpers``
on-node GPUs over NVLink:

- pre-attention: broadcast the layer's hidden state and the offloaded heads'
  prefix-KV to the helpers. This overlaps the straggler's own attention, so only
  its *exposed* part counts (near zero).
- attention: every GPU runs scaled-dot-product attention for its heads.
- post-attention: the helpers reduce their head outputs back into the straggler
  and return the new-KV they produced. This waits for attention to finish, so it
  is exposed.

Run on one node::

    torchrun --nproc_per_node=8 -m shunt.bench.elastic_attn_bench \
        --seq 4096 --out elastic_results.jsonl

Sweeps helper counts {1,3,5,7} and severities {3,5,7,9,11}; writes one JSON line
per (severity, helpers) with the exposed pre / attention / post fractions.
GQA-aware: query heads split across helpers, each helper holds the shared KV head
for the query heads it takes.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

HIDDEN = 4096
Q_HEADS = 64
KV_HEADS = 4
HEAD_DIM = 128
Q_PER_KV = Q_HEADS // KV_HEADS


def _evt():
    return torch.cuda.Event(enable_timing=True)


def _attn(q, k, v):
    # q: [h, S, d]; expand KV heads to match (GQA)
    return F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0),
                                          v.unsqueeze(0), is_causal=True).squeeze(0)


def run_case(rank, world, helpers, severity, seq, iters=20):
    """Return exposed pre / attn / post times (ms) summed over iters on rank 0."""
    dev = torch.device(f"cuda:{rank}")
    # severity sets how many of the straggler's heads must move: a heavier
    # straggler (higher max/mean) sheds more heads to the helpers.
    moved = min(Q_HEADS - Q_HEADS // (helpers + 1),
                int(Q_HEADS * (severity - 1) / severity))
    per_helper = max(1, moved // helpers) if helpers else 0
    my_heads = (Q_HEADS - moved) if rank == 0 else (per_helper if rank <= helpers else 0)
    if my_heads == 0:
        # idle GPUs still join the collectives below with empty payloads
        my_heads = 0

    pre = attn = post = 0.0
    for _ in range(iters):
        hidden = torch.randn(seq, HIDDEN, device=dev, dtype=torch.bfloat16)
        e0, e1, e2, e3 = _evt(), _evt(), _evt(), _evt()
        torch.cuda.synchronize()

        # pre: straggler broadcasts hidden + offloaded KV; overlaps own compute
        e0.record()
        dist.broadcast(hidden, src=0)
        e1.record()

        # attention for my heads
        if my_heads > 0:
            q = torch.randn(my_heads, seq, HEAD_DIM, device=dev, dtype=torch.bfloat16)
            kvh = max(1, my_heads // Q_PER_KV)
            k = torch.randn(my_heads, seq, HEAD_DIM, device=dev, dtype=torch.bfloat16)
            v = torch.randn(my_heads, seq, HEAD_DIM, device=dev, dtype=torch.bfloat16)
            out = _attn(q, k, v)
        else:
            out = torch.zeros(1, device=dev, dtype=torch.bfloat16)
        e2.record()

        # post: helpers reduce head outputs back to straggler + return KV
        red = torch.zeros(Q_HEADS, seq, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        if my_heads > 0 and out.dim() == 3:
            red[:my_heads] = out
        dist.reduce(red, dst=0)
        e3.record()
        torch.cuda.synchronize()

        if rank == 0:
            t_pre, t_attn, t_post = e0.elapsed_time(e1), e1.elapsed_time(e2), e2.elapsed_time(e3)
            # exposed pre = broadcast not hidden behind compute
            pre += max(0.0, t_pre - t_attn)
            attn += t_attn
            post += t_post
    return pre, attn, post


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig 18 elastic-attention overhead")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--out", default="elastic_results.jsonl")
    ap.add_argument("--severities", default="3,5,7,9,11")
    ap.add_argument("--helpers", default="1,3,5,7")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    rows = []
    for helpers in [int(x) for x in a.helpers.split(",")]:
        for sev in [int(x) for x in a.severities.split(",")]:
            pre, attn, post = run_case(rank, world, helpers, sev, a.seq)
            if rank == 0:
                tot = pre + attn + post
                rows.append({"severity": sev, "helpers": helpers,
                             "pre_frac": pre / tot, "attn_frac": attn / tot,
                             "post_frac": post / tot})
                print(f"sev={sev} helpers={helpers}: pre={pre/tot:.1%} "
                      f"attn={attn/tot:.1%} post={post/tot:.1%}")
            dist.barrier()
    if rank == 0:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {a.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
