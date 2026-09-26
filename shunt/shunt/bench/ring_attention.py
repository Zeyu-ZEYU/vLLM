"""Striped ring attention on the elastic-attention iterations (Table S6).

The straggler's batch (rank 0) is striped over the ranks that elastic
attention would use, with the same work shares as the head split: every
request's context (reused prefix and new tokens) is cut into 16-token units,
and units are dealt to the ranks by a weighted round-robin pattern. Ranks with
a zero share stay out of the ring. The straggler first sends every ring member
the layer input of its new tokens and the prefix KV of its units; each helper
runs its own batch (one attention layer over all heads) and then joins the
ring. In a ring of r ranks, every rank computes its queries against the units
it holds, passes those units on with NCCL point-to-point, and repeats r
times, merging partial results by their log-sum-exp; a query sees every
earlier unit of its request and, causally, its own unit.

Reported per configuration: the makespan (until the straggler's attention is
complete on every rank), the idle share of the straggler and the helpers
(time waiting for the next units over the time in the ring), and the mean
length of one wait. Compare the makespan with the complete variant of
``shunt.bench.elastic_overhead`` on the same configuration.

Example (8 GPUs)::

    torchrun --nproc-per-node 8 -m shunt.bench.ring_attention \\
        --model /models/Qwen3-235B-A22B --iterations bench/iterations.json \\
        --helpers 1 3 7 --profile profiles/h20.json --out bench/ring.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.distributed as dist

from ..compute_model import ComputeModel
from ..config import DeploySpec, ModelSpec
from ..runtime.elastic import ElasticAttention, _Weights, head_ranges
from .common import BLOCK, RankBatch, attention_module, dist_init
from .elastic_overhead import plan_heads


def shares_from_moves(moves, H: int, h: int) -> list[float]:
    """Each rank's share of the straggler's attention work under the head split."""
    ranges = head_ranges(moves, H)
    keep = ranges[0][0][1] if 0 in ranges else H
    share = [keep / H] + [0.0] * h
    for d, a, b in ranges.get(0, []):
        share[d] += (b - a) / H
    return share


def unit_pattern(share: list[float], length: int = 64) -> list[int]:
    """Weighted round-robin over the ranks with a nonzero share."""
    slots = []
    for r, s in enumerate(share):
        c = round(s * length)
        slots += [((k + 0.5) / c, r) for k in range(c)]
    return [r for _, r in sorted(slots)] or [0]


def units_of(reqs, pattern, rank: int) -> list[tuple]:
    """(request, unit index, start, length, first query position) per unit."""
    out = []
    for i, (p, n) in enumerate(reqs):
        ctx = p + n
        for u in range(math.ceil(ctx / BLOCK)):
            if pattern[u % len(pattern)] == rank:
                s = u * BLOCK
                out.append((i, u, s, min(BLOCK, ctx - s), max(s, p)))
    return out


def merge(o1, l1, o2, l2):
    """Merge partial attention outputs [T, H, D] with log-sum-exp [H, T]."""
    a, b = l1.transpose(0, 1), l2.transpose(0, 1)
    m = torch.maximum(a, b)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w1, w2 = torch.exp(a - m), torch.exp(b - m)
    s = w1 + w2
    o = (o1.float() * w1.unsqueeze(-1) + o2.float() * w2.unsqueeze(-1)) \
        / s.clamp_min(1e-30).unsqueeze(-1)
    lse = torch.where(s > 0, m + torch.log(s.clamp_min(1e-30)),
                      torch.full_like(s, -float("inf")))
    return o.to(o1.dtype), lse.transpose(0, 1).contiguous()


def plan_step(q_units, k_units, dev="cuda") -> list[tuple]:
    """Metadata for this rank's queries against one holder's units, grouped by
    whether the query unit's own unit is among them (causal) or not."""
    by_req: dict[int, list[int]] = {}
    for j, (ri, u, *_rest) in enumerate(k_units):
        by_req.setdefault(ri, []).append(j)
    groups = {True: [], False: []}
    off = 0
    for ri, u, s, length, q0 in q_units:
        nq = s + length - q0
        if nq <= 0:
            continue
        earlier = [j for j in by_req.get(ri, []) if k_units[j][1] < u]
        same = [j for j in by_req.get(ri, []) if k_units[j][1] == u]
        vis = earlier + same
        if vis:
            groups[bool(same)].append((off, nq, vis, sum(k_units[j][3] for j in vis)))
        off += nq
    out = []
    for causal, items in groups.items():
        if not items:
            continue
        idx = torch.cat([torch.arange(o, o + n, device=dev) for o, n, _, _ in items])
        cu = torch.zeros(len(items) + 1, dtype=torch.int32, device=dev)
        cu[1:] = torch.cumsum(torch.tensor([n for _, n, _, _ in items], device=dev), 0)
        table = torch.zeros(len(items), max(len(v) for _, _, v, _ in items),
                            dtype=torch.int32, device=dev)
        for i, (_, _, v, _) in enumerate(items):
            table[i, :len(v)] = torch.tensor(v, dtype=torch.int32, device=dev)
        used = torch.tensor([x for *_, x in items], dtype=torch.int32, device=dev)
        out.append((causal, idx, cu, table, used, int(used.max())))
    return out


def step_attention(q, plan, blocks, fa_version, scale, H):
    """Run one ring step from its precomputed plan."""
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    out = torch.zeros_like(q)
    lse = torch.full((H, q.shape[0]), -float("inf"), device=q.device)
    for causal, idx, cu, table, used, max_k in plan:
        o, l = flash_attn_varlen_func(
            q=q[idx].contiguous(), k=blocks[0], v=blocks[1], cu_seqlens_q=cu,
            max_seqlen_q=BLOCK, seqused_k=used, max_seqlen_k=max_k,
            softmax_scale=scale, causal=causal, block_table=table,
            fa_version=fa_version, return_softmax_lse=True)
        out[idx] = o
        lse[:, idx] = l
    return out, lse


def split_global(strag, units, ring, m, bf):
    """Rank 0: random layer input of every new token and prefix KV of every
    reused-prefix token of the straggler's batch, cut into each ring member's
    pieces (in its unit order). Keeps the global tensors for the check."""
    global _GLOBAL
    Hkv, D = m.num_kv_heads, m.head_dim
    xs = [torch.randn(n, m.hidden, **bf) for _, n in strag]
    pre = [torch.randn(2, p, Hkv, D, **bf) for p, _ in strag]
    _GLOBAL = (xs, pre)
    out = {}
    for r in ring:
        xr, kr = [], []
        for i, u, s, l, q0 in units[r]:
            p = strag[i][0]
            if s < q0:
                kr.append(pre[i][:, s:min(s + l, q0)])
            if s + l > q0:
                xr.append(xs[i][q0 - p:s + l - p])
        out[r] = (torch.cat(xr) if xr else torch.zeros(0, m.hidden, **bf),
                  torch.cat(kr, dim=1) if kr else torch.zeros(2, 0, Hkv, D, **bf))
    return out


_GLOBAL = None


def _check(rank, ring, units, strag, acc, W, m, mod, fa):
    """Compare the ring's outputs with one attention over the whole batch."""
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    H, Hkv, D = m.num_q_heads, m.num_kv_heads, m.head_dim
    world = dist.get_world_size()
    n_rows = torch.tensor([acc.shape[0] if acc is not None else 0], device="cuda")
    counts = [torch.zeros_like(n_rows) for _ in range(world)]
    dist.all_gather(counts, n_rows)
    if rank == 0:
        got = {0: acc}
        for r in ring:
            if r:
                buf = torch.empty(int(counts[r]), H, D, device="cuda", dtype=acc.dtype)
                dist.recv(buf, r)
                got[r] = buf
        xs, pre = _GLOBAL
        x = torch.cat(xs)
        pos = torch.cat([torch.arange(p, p + n, device="cuda") for p, n in strag])
        q, k, v = W.project(x, 0, H, 0, Hkv)
        q, k = W.norm_rope(pos, q, k, H, Hkv)
        q, k, v = q.view(-1, H, D), k.view(-1, Hkv, D), v.view(-1, Hkv, D)
        ks, vs, off = [], [], 0
        for (p, n), kv in zip(strag, pre):
            ks += [kv[0], k[off:off + n]]
            vs += [kv[1], v[off:off + n]]
            off += n
        cu_q = torch.tensor([0] + list(np.cumsum([n for _, n in strag])),
                            dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0] + list(np.cumsum([p + n for p, n in strag])),
                            dtype=torch.int32, device="cuda")
        ref = flash_attn_varlen_func(
            q=q, k=torch.cat(ks), v=torch.cat(vs), cu_seqlens_q=cu_q,
            max_seqlen_q=max(n for _, n in strag), cu_seqlens_k=cu_k,
            max_seqlen_k=max(p + n for p, n in strag), softmax_scale=mod.scaling,
            causal=True, fa_version=fa)
        starts = [0] + list(np.cumsum([n for _, n in strag]))
        err = 0.0
        for r in ring:
            row = 0
            for i, u, s, l, q0 in units[r]:
                n = s + l - q0
                if n <= 0:
                    continue
                p = strag[i][0]
                a = starts[i] + (q0 - p)
                d = (got[r][row:row + n].float() - ref[a:a + n].float()).abs().max()
                err = max(err, float(d))
                row += n
        print(f"check: max |ring - reference| = {err:.4g}", flush=True)
    elif rank in ring:
        dist.send(acc.contiguous(), 0)


def run_config(rank, m, mod, batches, share, reps, check=False):
    D, H, Hkv = m.head_dim, m.num_q_heads, m.num_kv_heads
    ring = [r for r, s in enumerate(share) if s > 0]
    ea = ElasticAttention(None, 256, model=m)
    W = _Weights(mod, ea)
    fa = mod.attn.impl.vllm_flash_attn_version
    strag = [tuple(r) for r in batches[0]]
    pattern = unit_pattern(share)
    units = {r: units_of(strag, pattern, r) for r in ring}
    own = RankBatch.build(m, [tuple(r) for r in batches[rank]]) \
        if (rank in ring and rank > 0) else None
    bf = dict(device="cuda", dtype=torch.bfloat16)

    def sizes(r):
        us = units[r]
        n_new = sum(s + l - q0 for _, _, s, l, q0 in us if s + l > q0)
        n_pre = sum(max(0, min(s + l, q0) - s) for _, _, s, l, q0 in us)
        return n_new, n_pre

    pieces = split_global(strag, units, ring, m, bf) if rank == 0 else None
    plans = {}
    if rank in ring:
        i = ring.index(rank)
        for step in range(len(ring)):
            src = ring[(i - step) % len(ring)]
            plans[step] = plan_step(units[rank], units[src])
    res = {"makespan": [], "idle": [], "waits": []}
    for it in range(reps + 2):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        waits, ring_start = [], None
        if rank in ring:
            # scatter: the straggler sends each member its new-token inputs and
            # the prefix KV of its units
            ops, mine = [], None
            for r in ring:
                n_new, n_pre = sizes(r)
                if rank == 0:
                    x, kvp = pieces[r]
                    if r == 0:
                        mine = (x, kvp)
                    else:
                        ops += [dist.P2POp(dist.isend, x, r),
                                dist.P2POp(dist.isend, kvp, r)]
                elif r == rank:
                    mine = (torch.empty(n_new, m.hidden, **bf),
                            torch.empty(2, n_pre, Hkv, D, **bf))
                    ops += [dist.P2POp(dist.irecv, mine[0], 0),
                            dist.P2POp(dist.irecv, mine[1], 0)]
            works = dist.batch_isend_irecv(ops) if ops else []
            if own is not None and own.T:
                ea._kept_attention(mod, W, own.positions, own.x, own.md, own.kv_cache,
                                   own.slot_mapping, mod.attn, own.T, H)
            ring_start = torch.cuda.Event(enable_timing=True)
            ring_start.record()
            for w in works:
                w.wait()
            x, kvp = mine
            mu = units[rank]
            pos = torch.cat([torch.arange(q0, s + l, device="cuda")
                             for _, _, s, l, q0 in mu if s + l > q0]) if x.shape[0] \
                else torch.zeros(0, dtype=torch.long, device="cuda")
            q, k, v = W.project(x, 0, H, 0, Hkv)
            q, k = W.norm_rope(pos, q, k, H, Hkv)
            q = q.view(-1, H, D)
            k, v = k.view(-1, Hkv, D), v.view(-1, Hkv, D)
            blocks = torch.zeros(2, max(1, len(mu)), BLOCK, Hkv, D, **bf)
            on = op = 0
            for j, (_, _, s, l, q0) in enumerate(mu):
                npre = max(0, min(s + l, q0) - s)
                if npre:
                    blocks[:, j, :npre] = kvp[:, op:op + npre]
                    op += npre
                nnew = l - npre
                if nnew:
                    blocks[0, j, npre:l] = k[on:on + nnew]
                    blocks[1, j, npre:l] = v[on:on + nnew]
                    on += nnew
            acc = torch.zeros_like(q)
            acc_lse = torch.full((H, q.shape[0]), -float("inf"), device="cuda")
            hold = blocks
            i = ring.index(rank)
            nxt, prv = ring[(i + 1) % len(ring)], ring[i - 1]
            for step in range(len(ring)):
                works = []
                if step < len(ring) - 1:
                    src = ring[(i - step - 1) % len(ring)]
                    recv = torch.empty(2, max(1, len(units[src])), BLOCK, Hkv, D, **bf)
                    works = dist.batch_isend_irecv([
                        dist.P2POp(dist.isend, hold.contiguous(), nxt),
                        dist.P2POp(dist.irecv, recv, prv)])
                o, l_ = step_attention(q, plans[step], hold, fa, mod.scaling, H)
                acc, acc_lse = merge(acc, acc_lse, o, l_)
                if works:
                    e1 = torch.cuda.Event(enable_timing=True)
                    e2 = torch.cuda.Event(enable_timing=True)
                    e1.record()
                    for w in works:
                        w.wait()
                    e2.record()
                    waits.append((e1, e2))
                    hold = recv
        t1.record()
        t1.synchronize()
        if check and it == reps + 1:
            _check(rank, ring, units, strag, acc if rank in ring else None, W, m,
                   mod, fa)
        total = torch.tensor([t0.elapsed_time(t1)], device="cuda")
        dist.all_reduce(total, op=dist.ReduceOp.MAX)
        if it >= 2:
            w = [a.elapsed_time(b) for a, b in waits]
            span = ring_start.elapsed_time(t1) if ring_start is not None else 0.0
            res["makespan"].append(float(total) * 1e-3)
            res["idle"].append(sum(w) / span if span > 0 else 0.0)
            res["waits"] += [x for x in w if x > 0.01]
    mine = torch.tensor([float(np.median(res["idle"])),
                         float(np.mean(res["waits"])) if res["waits"] else 0.0,
                         float(rank in ring)], device="cuda")
    allst = [torch.zeros_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(allst, mine)
    helpers = [s.tolist() for r, s in enumerate(allst) if r in ring and r != 0]
    waits_all = [s[1].item() for s in allst if s[2].item() > 0 and s[1].item() > 0]
    return {"makespan_s": float(np.median(res["makespan"])), "ring_size": len(ring),
            "straggler_idle": float(allst[0][0]),
            "helper_idle_mean": float(np.mean([h[0] for h in helpers])) if helpers else 0.0,
            "helper_idle_max": float(np.max([h[0] for h in helpers])) if helpers else 0.0,
            "mean_wait_ms": float(np.mean(waits_all)) if waits_all else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--iterations", required=True)
    ap.add_argument("--helpers", type=int, nargs="+", default=[1, 3, 7])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--check", action="store_true",
                    help="compare the ring's output with one full attention")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rank, world = dist_init()
    m = ModelSpec.from_hf_config(a.model)
    cm = ComputeModel.from_profile(a.profile, m, DeploySpec())
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
            share = shares_from_moves(moves, m.num_q_heads, h)
            r = run_config(rank, m, mod, batches, share, a.reps, a.check)
            r |= {"imbalance": it["imbalance"], "helpers": h, "shares": share}
            results.append(r)
            if rank == 0:
                print(f"imbalance {it['imbalance']:.2f}, {h} helpers: ring of "
                      f"{r['ring_size']}, makespan {r['makespan_s'] * 1e3:.2f} ms, "
                      f"straggler idle {100 * r['straggler_idle']:.1f}%", flush=True)
    if rank == 0:
        with open(a.out, "w") as f:
            json.dump({"model": m.name, "gpu": torch.cuda.get_device_name(),
                       "world": world, "results": results}, f, indent=1)
        print(f"wrote {a.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
