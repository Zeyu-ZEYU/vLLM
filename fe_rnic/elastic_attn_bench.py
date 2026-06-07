#!/usr/bin/env python3
"""Multi-layer Qwen3-235B-A22B prefill forward microbenchmark on DP16/EP16
(16 ranks, 8/node x2). Measures per-layer execution time for an imbalanced batch
WITHOUT vs WITH straggler-aware elastic attention parallelism (Sec 3.3), to
quantify how much elastic shrinks the per-layer time / prefill makespan / TTFT.

- Attention: REAL (vLLM flash-attn, GQA). prefix=0 (pure-prefill self-attn) for
  now; Stage 3c will add cached prefixes from the trace.
- MoE all-to-all + expert FFN: REAL ops, representative volumes (synthetic
  weights, balanced dispatch). Elastic does not touch these.
- Per-layer time = critical path = max-worker attention (+ NVLink for elastic) +
  dispatch + expert + combine.

Elastic (one straggler, here rank 0 on node 0): rank 0 broadcasts its input to
the 7 same-node helpers over NVLink; the 64 q-heads of the straggler's tokens
are split across the 8 node-0 GPUs (Alg 2 greedy balance); each helper also does
its OWN attention; the straggler-head partial O's are reduced back to rank 0.
Node 1 (no straggler) runs plain attention.
"""
import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.vllm_flash_attn import flash_attn_varlen_func

HID, NQ, NKV, HD = 4096, 64, 4, 128
NEXP, TOPK, MOE_INT = 128, 8, 1536
SCALE = 0.02


def build_weights(dev, n_local_exp):
    g = lambda *s: torch.randn(*s, device=dev, dtype=torch.bfloat16) * SCALE
    return dict(
        wqkv=g(HID, (NQ + 2 * NKV) * HD),
        wo=g(NQ * HD, HID),
        wgate=g(HID, NEXP),
        e_gate_up=g(n_local_exp, HID, 2 * MOE_INT),
        e_down=g(n_local_exp, MOE_INT, HID),
    )


def attn_partial(h, W, lo, hi):
    """Attention for q-heads [lo,hi) of input h; returns partial O (N,HID).
    KV is gathered per q-head (MHA form) so any head range satisfies flash-attn's
    head-divisibility; compute still equals GQA (one attention per q-head)."""
    N = h.shape[0]
    nq = hi - lo
    q = (h @ W["wqkv"][:, lo * HD : hi * HD]).view(N, nq, HD)
    koff, voff = NQ * HD, (NQ + NKV) * HD
    k_all = (h @ W["wqkv"][:, koff : koff + NKV * HD]).view(N, NKV, HD)
    v_all = (h @ W["wqkv"][:, voff : voff + NKV * HD]).view(N, NKV, HD)
    idx = torch.arange(lo, hi, device=h.device) // (NQ // NKV)  # kv head per q head
    k = k_all.index_select(1, idx).contiguous()
    v = v_all.index_select(1, idx).contiguous()
    cu = torch.tensor([0, N], device=h.device, dtype=torch.int32)
    o = flash_attn_varlen_func(q, k, v, max_seqlen_q=N, cu_seqlens_q=cu,
                               max_seqlen_k=N, cu_seqlens_k=cu, causal=True)
    if isinstance(o, tuple):
        o = o[0]
    return o.reshape(N, nq * HD) @ W["wo"][lo * HD : hi * HD, :]


def distribute_heads(t_help, e_s, n_help, H):
    """Greedy Alg-2 balance: assign the straggler's H heads across [straggler]+
    n_help helpers so all finish together. straggler own-load=0, helper=t_help,
    each assigned head of the straggler adds e_s. Returns [keep, help1..helpN]."""
    times = [0.0] + [t_help] * n_help
    assign = [0] * (1 + n_help)
    for _ in range(H):
        g = min(range(len(times)), key=lambda i: times[i])
        assign[g] += 1
        times[g] += e_s
    return assign


def moe(h, W, total_fresh, world):
    dev = h.device
    n_local = W["e_gate_up"].shape[0]
    _ = (h @ W["wgate"]).float().softmax(-1).topk(TOPK, -1)
    per = (total_fresh * TOPK) // world
    per = max(world * n_local, (per // (world * n_local)) * (world * n_local))
    send = torch.randn(per, HID, device=dev, dtype=torch.bfloat16)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    t = per // n_local
    x = recv[: t * n_local].view(n_local, t, HID)
    gu = torch.bmm(x, W["e_gate_up"])
    y = torch.bmm(F.silu(gu[..., :MOE_INT]) * gu[..., MOE_INT:], W["e_down"])
    out = recv.clone()
    out[: t * n_local] = y.reshape(t * n_local, HID)
    recv2 = torch.empty_like(out)
    dist.all_to_all_single(recv2, out)


def run_pass(mode, h, W, total_fresh, world, rank, local, args, node0_grp, assign, dev):
    """One layer forward. mode: 'baseline' or 'elastic'. Returns nothing (timed by caller)."""
    if mode == "baseline" or rank >= 8:
        _ = attn_partial(h, W, 0, NQ)            # plain full attention
    else:
        # node-0 elastic. broadcast straggler input over NVLink.
        sbuf = h if rank == 0 else torch.empty(args.n_strag, HID, device=dev, dtype=torch.bfloat16)
        dist.broadcast(sbuf, src=0, group=node0_grp)
        lo, hi = assign[local]
        if rank != 0:
            _ = attn_partial(h, W, 0, NQ)        # helper's own attention
        part = attn_partial(sbuf, W, lo, hi) if hi > lo else \
            torch.zeros(args.n_strag, HID, device=dev, dtype=torch.bfloat16)
        dist.reduce(part, dst=0, group=node0_grp, op=dist.ReduceOp.SUM)  # NVLink
    moe(h, W, total_fresh, world)


def timed(mode, h, W, total_fresh, world, rank, local, args, node0_grp, assign, dev, ns):
    attn_ms, layer_ms = [], []
    for it in range(args.warmup + args.layers):
        dist.barrier(); torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0 = time.perf_counter(); ev0.record()
        if mode == "baseline" or rank >= 8:
            _ = attn_partial(h, W, 0, NQ)
        else:
            sbuf = h if rank == 0 else torch.empty(ns, HID, device=dev, dtype=torch.bfloat16)
            dist.broadcast(sbuf, src=0, group=node0_grp)
            lo, hi = assign[local]
            if rank != 0:
                _ = attn_partial(h, W, 0, NQ)
            part = attn_partial(sbuf, W, lo, hi) if hi > lo else \
                torch.zeros(ns, HID, device=dev, dtype=torch.bfloat16)
            dist.reduce(part, dst=0, group=node0_grp, op=dist.ReduceOp.SUM)
        ev1.record()
        moe(h, W, total_fresh, world)
        torch.cuda.synchronize(); dist.barrier()
        t1 = time.perf_counter()
        if it >= args.warmup:
            attn_ms.append(ev0.elapsed_time(ev1))
            layer_ms.append((t1 - t0) * 1000)
    return sum(attn_ms) / len(attn_ms), sum(layer_ms) / len(layer_ms)


def run_config(ns, W, world, rank, local, args, node0_grp, dev):
    N = ns if rank == 0 else args.n_base
    h = torch.randn(N, HID, device=dev, dtype=torch.bfloat16)
    nt = torch.tensor([N], device=dev); dist.all_reduce(nt); total_fresh = int(nt.item())
    b_attn, b_layer = timed("baseline", h, W, total_fresh, world, rank, local, args, node0_grp, None, dev, ns)
    g = [torch.zeros(1, device=dev) for _ in range(world)]
    dist.all_gather(g, torch.tensor([b_attn], device=dev))
    at = [x.item() for x in g]
    t_strag, t_help = at[0], sum(at[1:8]) / 7
    counts = distribute_heads(t_help, t_strag / NQ, 7, NQ)
    bounds, c = [], 0
    for cnt in counts:
        bounds.append((c, c + cnt)); c += cnt
    e_attn, e_layer = timed("elastic", h, W, total_fresh, world, rank, local, args, node0_grp, bounds, dev, ns)
    mxmean = max(at) / (sum(at) / world)
    return dict(ns=ns, total=total_fresh, mxmean=mxmean, b_layer=b_layer, e_layer=e_layer, counts=counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=2048)
    ap.add_argument("--n-strag", type=int, default=8192)
    ap.add_argument("--sweep", type=str, default="", help="comma-sep n_strag values")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()

    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    dev = torch.device("cuda", local)
    node0_grp = dist.new_group(list(range(8)))
    _ = dist.new_group(list(range(8, 16)))
    W = build_weights(dev, NEXP // world)

    sweep = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.n_strag]
    res = [run_config(ns, W, world, rank, local, args, node0_grp, dev) for ns in sweep]

    if rank == 0:
        print(f"\n[SWEEP] n_base={args.n_base} layers={args.layers}  (per-layer ms, makespan = x94 layers)", flush=True)
        print(f"{'n_strag':>8}{'total':>8}{'max/mean':>9}{'base_ms':>9}{'elas_ms':>9}{'save%':>7}{'mks_b(s)':>9}{'mks_e(s)':>9}", flush=True)
        for r in res:
            sv = (r["b_layer"] - r["e_layer"]) / r["b_layer"] * 100
            print(f"{r['ns']:>8}{r['total']:>8}{r['mxmean']:>9.2f}{r['b_layer']:>9.2f}"
                  f"{r['e_layer']:>9.2f}{sv:>7.1f}{r['b_layer']*94/1000:>9.2f}{r['e_layer']*94/1000:>9.2f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
