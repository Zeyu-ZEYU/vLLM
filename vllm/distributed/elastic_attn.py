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
