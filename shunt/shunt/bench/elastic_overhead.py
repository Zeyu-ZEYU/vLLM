"""Exposed transfer time of elastic attention on one node (Fig. S3).

Replays iterations picked from a measured run (``shunt.bench.pick_iterations``)
on one node: rank 0 holds the straggler's batch; ranks 1..h hold the batches
of the node's h lightest ranks and act as helpers. The head split comes from
Algorithm S1 on the estimated compute of these ranks. One attention layer is
timed in three variants, each started after a barrier and measured on the
slowest rank:

1. compute only: every rank computes with its inputs already in place;
2. with the pre-attention transfer: the straggler sends the layer input and
   the prefix KV of the donated heads' KV heads before computing;
3. complete: additionally the helpers return their output-projection partial
   sums and the new KV of fully donated KV heads, and the straggler adds them.

Shares of the complete time: pre-attention transfer (2 - 1), attention
compute (1), post-attention transfer (3 - 2). The compute functions are those
of the serving runtime.

Example (8 GPUs)::

    torchrun --nproc-per-node 8 -m shunt.bench.elastic_overhead \\
        --model /models/Qwen3-235B-A22B --iterations bench/iterations.json \\
        --helpers 1 3 5 7 --profile profiles/compute.json --out bench/eap_overhead.json
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.distributed as dist

from .. import algorithms
from ..compute_model import ComputeModel
from ..config import DeploySpec, ModelSpec
from ..runtime.elastic import (ElasticAttention, _gather_prefix, _Weights, _write_kv,
                               derive_assignment)
from .common import RankBatch, attention_module, dist_init, time_region


def plan_heads(m: ModelSpec, cm: ComputeModel, batches: list[list], h: int):
    """Algorithm S1 over the straggler (index 0) and the first ``h`` helpers."""
    t = [sum(cm.worker_time(p, n) for p, n in b) for b in batches]
    a = [sum(cm.attention_time(p, n) for p, n in b) for b in batches]
    mean = sum(t) / len(t)
    moves, _ = algorithms.balance_heads(t[:h + 1], a[:h + 1], mean, m.num_q_heads, 0.0)
    return moves


def run_config(rank: int, m: ModelSpec, ea: ElasticAttention, mod, batches, h: int,
               moves, reps: int) -> dict:
    """Time the three variants for one iteration and helper count. Every rank
    follows its own part of the plan, as in serving (a rank may both lend and
    borrow heads)."""
    W = _Weights(mod, ea)
    D = m.head_dim
    asg = derive_assignment(moves, rank, m.num_q_heads, m.q_per_kv, m.num_kv_heads)
    part = rank <= h
    mine = RankBatch.build(m, [tuple(r) for r in batches[rank]]) if part else None
    donors = {s: RankBatch.build(m, [tuple(r) for r in batches[s]])
              for s, _, _ in (asg.inn if asg else [])}
    metas = {s: b.recv_meta() for s, b in donors.items()}
    staged = {}
    if asg is not None:
        for s, a, b in asg.inn:
            g0, g1 = ea._kv_range(a, b)
            kp = vp = None
            if donors[s].P:
                kp = torch.randn(donors[s].P, g1 - g0, D, device="cuda",
                                 dtype=torch.bfloat16)
                vp = torch.randn_like(kp)
            staged[s] = (donors[s].x.clone(), kp, vp)
    keep = asg.keep if asg is not None else m.num_q_heads

    def own():
        if mine is None or mine.T == 0:
            return None
        return ea._kept_attention(mod, W, mine.positions, mine.x, mine.md, mine.kv_cache,
                                  mine.slot_mapping, mod.attn, mine.T, keep)

    def donated(inputs):
        return [(s, a, b, ea._donated_heads(mod, W, metas[s], *inputs[s], a, b))
                for s, a, b in asg.inn]

    def pre_ops():
        ops, recv = [], {}
        for d, a, b in asg.out:
            ops.append(dist.P2POp(dist.isend, mine.x, d))
            if mine.P:
                g0, g1 = ea._kv_range(a, b)
                kp, vp = _gather_prefix(mine.kv_cache, mine.prefix_slots, g0, g1)
                ops += [dist.P2POp(dist.isend, kp, d), dist.P2POp(dist.isend, vp, d)]
        for s, a, b in asg.inn:
            g0, g1 = ea._kv_range(a, b)
            xs = torch.empty_like(donors[s].x)
            ops.append(dist.P2POp(dist.irecv, xs, s))
            kp = vp = None
            if donors[s].P:
                kp = torch.empty(donors[s].P, g1 - g0, D, device="cuda",
                                 dtype=torch.bfloat16)
                vp = torch.empty_like(kp)
                ops += [dist.P2POp(dist.irecv, kp, s), dist.P2POp(dist.irecv, vp, s)]
            recv[s] = (xs, kp, vp)
        return (dist.batch_isend_irecv(ops) if ops else []), recv

    def compute_only():
        if asg is not None:
            own()
            donated(staged)

    def with_pre():
        if asg is None:
            return
        works, recv = pre_ops()
        own()
        for w in works:
            w.wait()
        donated(recv)

    def complete():
        if asg is None:
            return
        works, recv = pre_ops()
        out = own()
        for w in works:
            w.wait()
        post = []
        for s, a, b, (y, k_new, v_new) in donated(recv):
            post.append(dist.P2POp(dist.isend, y, s))
            g0, _ = ea._kv_range(a, b)
            for g in asg.returns_to.get(s, []):
                post += [dist.P2POp(dist.isend, k_new[:, g - g0].contiguous(), s),
                         dist.P2POp(dist.isend, v_new[:, g - g0].contiguous(), s)]
        parts, back = [], []
        for d, _, _ in asg.out:
            y = torch.empty(mine.T, m.hidden, device="cuda", dtype=torch.bfloat16)
            post.append(dist.P2POp(dist.irecv, y, d))
            parts.append(y)
            for g in asg.returned_by.get(d, []):
                k = torch.empty(mine.T, D, device="cuda", dtype=torch.bfloat16)
                v = torch.empty_like(k)
                post += [dist.P2POp(dist.irecv, k, d), dist.P2POp(dist.irecv, v, d)]
                back.append((g, k, v))
        for w in (dist.batch_isend_irecv(post) if post else []):
            w.wait()
        if asg.out:
            for y in parts:
                out[:mine.T] += y
            kc, vc = mine.kv_cache.unbind(0)
            for g, k, v in back:
                _write_kv(mod.attn, k.view(-1, 1, D), v.view(-1, 1, D),
                          kc[:, :, g:g + 1], vc[:, :, g:g + 1], mine.slot_mapping)

    t_c = time_region(compute_only, reps)
    t_pc = time_region(with_pre, reps)
    t_full = time_region(complete, reps)
    return {"compute_s": t_c, "with_pre_s": t_pc, "complete_s": t_full,
            "pre_share": max(0.0, t_pc - t_c) / t_full,
            "attention_share": min(t_c, t_full) / t_full,
            "post_share": max(0.0, t_full - t_pc) / t_full}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--iterations", required=True)
    ap.add_argument("--helpers", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--profile", default=None, help="compute profile for the split")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rank, world = dist_init()
    m = ModelSpec.from_hf_config(a.model)
    cm = ComputeModel.from_profile(a.profile, m, DeploySpec())
    ea = ElasticAttention(None, 256, model=m)
    mod = attention_module(m)
    with open(a.iterations) as f:
        iters = json.load(f)["iterations"]
    results = []
    for it in iters:
        batches = it["ranks"]
        for h in a.helpers:
            if h + 1 > min(world, len(batches)):
                continue
            moves = plan_heads(m, cm, batches, h)
            if not moves:
                if rank == 0:
                    print(f"imbalance {it['imbalance']:.2f}, {h} helpers: no heads move")
                continue
            r = run_config(rank, m, ea, mod, batches, h, moves, a.reps)
            r |= {"imbalance": it["imbalance"], "target": it.get("target"),
                  "helpers": h, "heads_moved": len(moves)}
            results.append(r)
            if rank == 0:
                print(f"imbalance {it['imbalance']:.2f}, {h} helpers: pre "
                      f"{100 * r['pre_share']:.2f}% attention "
                      f"{100 * r['attention_share']:.2f}% post "
                      f"{100 * r['post_share']:.2f}%", flush=True)
    if rank == 0:
        with open(a.out, "w") as f:
            json.dump({"model": m.name, "gpu": torch.cuda.get_device_name(),
                       "world": world, "results": results}, f, indent=1)
        print(f"wrote {a.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
