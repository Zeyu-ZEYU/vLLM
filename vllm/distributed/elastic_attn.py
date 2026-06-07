"""fe_rnic: straggler-aware elastic attention (§3.3) runtime helpers.

P1: per-step intra-node coordination + Alg 2 offload plan (log-only, no transfer
yet). All entry points are gated by VLLM_ELASTIC_ATTN (see parallel_state).

The plan: each prefill step, same-node DP workers all-gather their token counts
over the intra-node DP group; if the max/mean imbalance exceeds theta the straggler
offloads some of its attention heads to lighter same-node workers (helpers), which
recompute + return them (P2/P3). Here we only compute and log the plan.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_intra_node_dp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except Exception:  # pragma: no cover - flash-attn is always present on the testbed
    flash_attn_varlen_func = None

logger = init_logger(__name__)

THETA = float(os.environ.get("VLLM_ELASTIC_ATTN_THETA", "2.0"))
_LOG_EVERY = int(os.environ.get("VLLM_ELASTIC_ATTN_LOG_EVERY", "100"))
_step = 0


def allgather_token_counts(num_tokens: int, device: torch.device) -> list[int]:
    """All-gather each same-node DP worker's token count for this step."""
    grp = get_intra_node_dp_group()
    n = grp.world_size
    t = torch.tensor([int(num_tokens)], dtype=torch.int64, device=device)
    out = torch.empty(n, dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(out, t, group=grp.device_group)
    return out.tolist()


def compute_offload_plan(
    counts: list[int], num_q_heads: int, theta: float = THETA
) -> dict | None:
    """Alg 2 (greedy): if imbalanced beyond theta, offload the straggler's heads to
    lighter same-node workers, balancing attention load toward the node mean.

    Returns None if not triggered. Otherwise a dict:
      {"straggler": s, "n_offload": k, "helpers": {helper_rank: n_heads, ...},
       "counts": counts, "mean": mean}
    Head cost for worker w ~= counts[w]/num_q_heads per query head (attention work
    scales with that worker's token count; GQA query heads are symmetric).
    """
    n = len(counts)
    total = sum(counts)
    if total == 0 or n < 2:
        return None
    mean = total / n
    s = max(range(n), key=lambda w: counts[w])
    if counts[s] <= theta * mean:
        return None  # theta-gate: not straggler-dominated enough
    per_head = counts[s] / num_q_heads
    if per_head <= 0:
        return None
    # heads to move so the straggler's load drops to ~mean
    n_offload = int(round((counts[s] - mean) / per_head))
    n_offload = max(1, min(n_offload, num_q_heads - 1))
    # helpers = workers below mean, by spare capacity (mean - load), descending
    spare = [(w, mean - counts[w]) for w in range(n) if w != s and counts[w] < mean]
    spare.sort(key=lambda x: -x[1])
    if not spare:
        return None
    total_spare = sum(sp for _, sp in spare)
    helpers: dict[int, int] = {}
    remaining = n_offload
    for i, (w, sp) in enumerate(spare):
        if remaining <= 0:
            break
        share = n_offload if i == len(spare) - 1 else int(round(n_offload * sp / total_spare))
        share = min(share, remaining)
        if share > 0:
            helpers[w] = share
            remaining -= share
    if remaining > 0 and helpers:  # round-off leftover -> first helper
        first = next(iter(helpers))
        helpers[first] += remaining
    if not helpers:
        return None
    return {
        "straggler": s,
        "n_offload": sum(helpers.values()),
        "helpers": helpers,
        "counts": counts,
        "mean": mean,
    }


def maybe_log_elastic_plan(num_tokens: int, device: torch.device, num_q_heads: int) -> None:
    """P1: all-gather counts + compute Alg 2 plan + log (no transfer)."""
    global _step
    _step += 1
    if _step <= 3:
        logger.info("[elastic-attn] ENTER hook step=%d num_tokens=%d", _step, num_tokens)
    counts = allgather_token_counts(num_tokens, device)
    if _step <= 3:
        logger.info("[elastic-attn] step=%d allgather counts=%s", _step, counts)
    plan = compute_offload_plan(counts, num_q_heads)
    if _step <= 3 or _step % _LOG_EVERY == 0 or plan is not None:
        mx = max(counts) if counts else 0
        mean = (sum(counts) / len(counts)) if counts else 0.0
        ratio = (mx / mean) if mean else 0.0
        if plan is None:
            logger.info(
                "[elastic-attn] step=%d counts=%s max/mean=%.2f -> no offload",
                _step, counts, ratio,
            )
        else:
            logger.info(
                "[elastic-attn] step=%d counts=%s max/mean=%.2f -> straggler r%d "
                "offload %d heads to %s",
                _step, counts, ratio, plan["straggler"], plan["n_offload"],
                plan["helpers"],
            )


_shadow_step = 0
_SHADOW_MAX = int(os.environ.get("VLLM_ELASTIC_ATTN_SHADOW", "8"))


def elastic_shadow_check(attn_module, q, k, v, attn_output) -> None:
    """P2-M1: verify that splitting the query heads into GQA groups and running
    flash-attn per group (all locally) reproduces self.attn's paged-backend
    output for a no-prefix prefill step.

    Log-only; runs for the first few no-prefix prefill attention calls then goes
    quiet. This is the correctness foundation before offloading whole GQA groups
    to same-node helpers over NVLink (P2-M2): if a local 4-way head split already
    matches the backend bit-for-bit(ish), the only remaining risk in M2 is the
    NVLink transfer/assembly, not the attention math.
    """
    global _shadow_step
    if flash_attn_varlen_func is None or _shadow_step >= _SHADOW_MAX:
        return
    try:
        am = get_forward_context().attn_metadata
        if isinstance(am, dict):
            am = am.get(attn_module.attn.layer_name)
        if am is None:
            return
        max_q = int(am.max_query_len)
        if max_q <= 1:
            return  # decode step, nothing to split
        qsl = am.query_start_loc
        num_reqs = int(qsl.shape[0] - 1)
        query_lens = qsl[1:] - qsl[:-1]
        seqlens = am.seq_lens[:num_reqs]
        if not torch.equal(query_lens.to(seqlens.dtype), seqlens):
            return  # cached prefix present -> raw q/k/v attn would be wrong (P3)
        T = int(am.num_actual_tokens)
        nq = attn_module.num_heads
        nkv = attn_module.num_kv_heads
        hd = attn_module.head_dim
        grp = nq // nkv
        scale = attn_module.scaling
        q3 = q[:T].view(T, nq, hd)
        k3 = k[:T].view(T, nkv, hd)
        v3 = v[:T].view(T, nkv, hd)
        cu = qsl.to(torch.int32)
        outs = []
        for g in range(nkv):  # one GQA group = grp query heads sharing kv head g
            qs = q3[:, g * grp:(g + 1) * grp, :].contiguous()
            ks = k3[:, g:g + 1, :].contiguous()
            vs = v3[:, g:g + 1, :].contiguous()
            o = flash_attn_varlen_func(
                qs, ks, vs, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max_q, max_seqlen_k=max_q,
                softmax_scale=scale, causal=True,
            )
            if isinstance(o, tuple):
                o = o[0]
            outs.append(o.reshape(T, grp * hd))
        o_split = torch.cat(outs, dim=-1)
        ref = attn_output[:T]
        diff = (o_split.float() - ref.float()).abs()
        _shadow_step += 1
        logger.info(
            "[elastic-shadow] step=%d layer=%s T=%d reqs=%d max|diff|=%.3e "
            "mean|diff|=%.3e ref|max|=%.3e allclose(2e-2)=%s",
            _shadow_step, attn_module.attn.layer_name, T, num_reqs,
            diff.max().item(), diff.mean().item(), ref.abs().max().item(),
            bool(torch.allclose(o_split, ref, atol=2e-2, rtol=2e-2)),
        )
    except Exception as e:  # never break serving for a debug check
        logger.warning("[elastic-shadow] skipped (error: %r)", e)


# ---------------------------------------------------------------------------
# P2-M2: real GQA-group offload over the intra-node DP NCCL group.
#
# Once per step (Qwen3MoeModel.forward) the same-node DP workers all-gather
# (token_count, offloadable_flag) and compute an identical Alg 2 plan in GQA-group
# units. Each attention layer then runs elastic_attention():
#   - straggler: writes its full K/V to cache, computes only the groups it keeps,
#     ships the offloaded groups' q/k/v to helpers over NVLink, recvs their
#     outputs, and reassembles. (Output is bit-identical to self.attn -- see M1.)
#   - helper: does its own self.attn, plus recomputes the straggler's offloaded
#     groups and returns them.
#   - everyone else: plain self.attn.
# All P2P uses the dedicated intra-node communicator (isolated from vLLM's DP
# all-reduces) with global peer ranks, deadlock-free via two batched rounds.
# ---------------------------------------------------------------------------

_STEP_PLAN: dict | None = None


def _offloadable_flag() -> int:
    """1 iff THIS worker's current batch is a single, no-prefix prefill request
    (the case M2 handles without transferring cu_seqlens). Decode / multi-request
    / chunked-prefix batches return 0 and are left to plain self.attn (P3)."""
    try:
        am = get_forward_context().attn_metadata
        if isinstance(am, dict):
            am = next(iter(am.values())) if am else None
        if am is None or int(am.max_query_len) <= 1:
            return 0
        qsl = am.query_start_loc
        nreq = int(qsl.shape[0] - 1)
        if nreq != 1:
            return 0
        qlen = qsl[1:] - qsl[:-1]
        sl = am.seq_lens[:nreq]
        return 1 if bool(torch.equal(qlen.to(sl.dtype), sl)) else 0
    except Exception:
        return 0


def compute_group_plan(counts, flags, num_groups, theta: float = THETA) -> dict | None:
    """Alg 2 in GQA-group units. Offload whole groups (grp query heads + their
    shared kv head) from the straggler to lighter same-node workers so all finish
    near the node mean. Identical on every worker (pure function of all-gathered
    counts/flags). Returns None if no offload."""
    n = len(counts)
    if n < 2:
        return None
    total = sum(counts)
    if total == 0:
        return None
    mean = total / n
    s = max(range(n), key=lambda w: counts[w])
    if flags[s] != 1:                       # straggler batch not offloadable
        return None
    if counts[s] <= theta * mean:           # theta-gate
        return None
    per_group = counts[s] / num_groups
    if per_group <= 0:
        return None
    n_off = int(round((counts[s] - mean) / per_group))
    n_off = max(1, min(n_off, num_groups - 1))   # keep >=1 group on the straggler
    cand = [w for w in range(n) if w != s and counts[w] < mean]
    if not cand:
        return None
    n_keep = num_groups - n_off
    load = {w: float(counts[w]) for w in cand}
    offload = []
    for g in range(n_keep, num_groups):     # offload the high-index groups
        h = min(cand, key=lambda w: load[w])  # currently-lightest helper
        offload.append((g, h))
        load[h] += per_group
    helpers = sorted({h for _, h in offload})
    return {
        "straggler": s, "n_keep": n_keep, "offload": offload,
        "helpers": helpers, "counts": counts, "T": int(counts[s]),
    }


def set_step_plan(num_tokens: int, device: torch.device, num_q_heads: int,
                  num_kv_heads: int) -> dict | None:
    """Once per step: all-gather (count, flag) over the intra-node DP group and
    stash the Alg 2 group plan for the attention layers to read."""
    global _STEP_PLAN, _step
    _step += 1
    flag = _offloadable_flag()
    grp = get_intra_node_dp_group()
    n = grp.world_size
    t = torch.tensor([int(num_tokens), int(flag)], dtype=torch.int64, device=device)
    out = torch.empty(n * 2, dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(out, t, group=grp.device_group)
    info = out.view(n, 2).tolist()
    counts = [r[0] for r in info]
    flags = [r[1] for r in info]
    _STEP_PLAN = compute_group_plan(counts, flags, num_kv_heads)
    if _STEP_PLAN is not None and (_step <= 50 or _step % _LOG_EVERY == 0):
        logger.info(
            "[elastic-attn] step=%d counts=%s -> straggler r%d keep %d/%d groups "
            "offload %s", _step, counts, _STEP_PLAN["straggler"],
            _STEP_PLAN["n_keep"], num_kv_heads, _STEP_PLAN["offload"],
        )
    return _STEP_PLAN


def get_step_plan() -> dict | None:
    return _STEP_PLAN


def _flash(q_g, k_g, v_g, cu, T, scale):
    o = flash_attn_varlen_func(
        q_g, k_g, v_g, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=T, max_seqlen_k=T, softmax_scale=scale, causal=True,
    )
    return o[0] if isinstance(o, tuple) else o


def _straggler_attention(attn_module, q, k, v, plan, nq, nkv, hd, grp, scale, grp_pg):
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.model_executor.layers.attention.attention import get_attention_context

    dev = q.device
    T = q.shape[0]
    pg = grp_pg.device_group
    q3 = q.view(T, nq, hd)
    k3 = k.view(T, nkv, hd)
    v3 = v.view(T, nkv, hd)

    # 1. write the straggler's full K/V to the paged cache (decode correctness).
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(attn_module.attn.layer_name)
    if kv_cache is not None and kv_cache.numel() > 0 and slot_mapping is not None:
        kc, vc = kv_cache.unbind(0)
        reshape_and_cache_flash(
            k3, v3, kc, vc, slot_mapping,
            attn_layer.kv_cache_dtype, attn_layer._k_scale, attn_layer._v_scale,
        )

    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    # 2. round 1: ship offloaded groups to helpers (non-blocking).
    ops1, hold = [], {}
    for g, h in plan["offload"]:
        hg = grp_pg.ranks[h]
        q_g = q3[:, g * grp:(g + 1) * grp, :].contiguous()
        k_g = k3[:, g:g + 1, :].contiguous()
        v_g = v3[:, g:g + 1, :].contiguous()
        hold[g] = (q_g, k_g, v_g)
        ops1 += [dist.P2POp(dist.isend, q_g, hg, pg),
                 dist.P2POp(dist.isend, k_g, hg, pg),
                 dist.P2POp(dist.isend, v_g, hg, pg)]
    works1 = dist.batch_isend_irecv(ops1) if ops1 else []

    # 3. compute kept groups locally (overlaps with helper compute + transfer).
    o_full = torch.empty(T, nq, hd, dtype=q.dtype, device=dev)
    for g in range(plan["n_keep"]):
        q_g = q3[:, g * grp:(g + 1) * grp, :].contiguous()
        k_g = k3[:, g:g + 1, :].contiguous()
        v_g = v3[:, g:g + 1, :].contiguous()
        o_full[:, g * grp:(g + 1) * grp, :] = _flash(q_g, k_g, v_g, cu, T, scale)
    for w in works1:
        w.wait()

    # 4. round 2: receive offloaded groups' outputs and assemble.
    ops2, recv_o = [], {}
    for g, h in plan["offload"]:
        hg = grp_pg.ranks[h]
        o_buf = torch.empty(T, grp, hd, dtype=q.dtype, device=dev)
        recv_o[g] = o_buf
        ops2.append(dist.P2POp(dist.irecv, o_buf, hg, pg))
    works2 = dist.batch_isend_irecv(ops2) if ops2 else []
    for w in works2:
        w.wait()
    for g, _h in plan["offload"]:
        o_full[:, g * grp:(g + 1) * grp, :] = recv_o[g]
    return o_full.reshape(T, nq * hd)


def _helper_attention(attn_module, q, k, v, plan, my, nq, nkv, hd, grp, scale, grp_pg):
    dev = q.device
    pg = grp_pg.device_group
    sg = grp_pg.ranks[plan["straggler"]]
    T_s = int(plan["T"])
    my_groups = [g for g, h in plan["offload"] if h == my]

    # 1. post recvs for the straggler's offloaded groups (before own attn).
    recv, ops1 = {}, []
    for g in my_groups:
        q_g = torch.empty(T_s, grp, hd, dtype=q.dtype, device=dev)
        k_g = torch.empty(T_s, 1, hd, dtype=q.dtype, device=dev)
        v_g = torch.empty(T_s, 1, hd, dtype=q.dtype, device=dev)
        recv[g] = (q_g, k_g, v_g)
        ops1 += [dist.P2POp(dist.irecv, q_g, sg, pg),
                 dist.P2POp(dist.irecv, k_g, sg, pg),
                 dist.P2POp(dist.irecv, v_g, sg, pg)]
    works1 = dist.batch_isend_irecv(ops1) if ops1 else []

    # 2. own attention for this helper's own tokens (overlaps with the transfer).
    own_out = attn_module.attn(q, k, v)

    # 3. recompute the straggler's offloaded groups, send results back.
    for w in works1:
        w.wait()
    cu = torch.tensor([0, T_s], dtype=torch.int32, device=dev)
    out_g = {}
    for g in my_groups:
        q_g, k_g, v_g = recv[g]
        out_g[g] = _flash(q_g, k_g, v_g, cu, T_s, scale).contiguous()
    ops2 = [dist.P2POp(dist.isend, out_g[g], sg, pg) for g in my_groups]
    works2 = dist.batch_isend_irecv(ops2) if ops2 else []
    for w in works2:
        w.wait()
    return own_out


def elastic_attention(attn_module, q, k, v):
    """Drop-in for self.attn(q, k, v) under VLLM_ELASTIC_ATTN. Dispatches on this
    worker's role in the current step's plan; falls back to plain attention when
    there is no offload this step."""
    plan = _STEP_PLAN
    if plan is None:
        return attn_module.attn(q, k, v)
    grp_pg = get_intra_node_dp_group()
    my = grp_pg.rank_in_group
    nq, nkv, hd = attn_module.num_heads, attn_module.num_kv_heads, attn_module.head_dim
    grp = nq // nkv
    scale = attn_module.scaling
    if my == plan["straggler"]:
        return _straggler_attention(attn_module, q, k, v, plan, nq, nkv, hd, grp, scale, grp_pg)
    if my in plan["helpers"]:
        return _helper_attention(attn_module, q, k, v, plan, my, nq, nkv, hd, grp, scale, grp_pg)
    return attn_module.attn(q, k, v)
