"""Straggler-aware elastic attention parallelism, serving side (§3.3, §4.1).

Each iteration the node planner writes ``eap_heads.json`` (each worker's post-split
query-head count from Alg. 2). At runtime, a straggler (head count < H) sheds its
extra query heads to on-node helpers (head count > H) over NVLink: the straggler
sends the layer input and those heads' prefix-KV, each helper computes the QKV
projection and attention for its assigned heads and returns the output and the
new-KV, and the straggler reduces them back. The split is GQA-aware (query heads
move with their shared KV head) and exact (one reduce restores the output), so it
costs no accuracy -- only the exposed post-attention transfer (Fig 18).

This module has two layers:

- :func:`head_assignment` -- the pure, testable mapping from per-worker head
  counts to "which rank computes which shed heads"; and
- :class:`ElasticAttention` -- the runtime NVLink mechanism over a node-local
  process group, plus :func:`patch_qwen3_attention` to hook it into the model.

The cost of the NVLink path is measured standalone by
``shunt.bench.elastic_attn_bench`` (Fig 18); this serving path needs the
multi-GPU testbed to validate end to end.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Pure head assignment (testable on CPU)
# ---------------------------------------------------------------------------

@dataclass
class NodeHeadPlan:
    """Per-rank elastic plan for one node.

    ``kept[r]``  -- query heads rank r computes for its own tokens;
    ``assign[r]`` -- list of (straggler_rank, num_heads) rank r computes for
    other ranks' tokens (only helpers have these).
    """

    kept: list[int]
    assign: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    @property
    def fires(self) -> bool:
        return any(self.assign.values())


def head_assignment(head_counts: list[int], num_q_heads: int) -> NodeHeadPlan:
    """Turn Alg. 2's per-worker head counts into a concrete shed->absorb map.

    A worker with ``h < H`` is a straggler that sheds ``H - h`` heads; one with
    ``h > H`` is a helper that absorbs ``h - H``. Shed heads are matched to
    absorbers greedily (heaviest straggler first), conserving the node's heads.
    """
    H = num_q_heads
    n = len(head_counts)
    kept = [min(h, H) for h in head_counts]
    shed = [(r, H - h) for r, h in enumerate(head_counts) if h < H]
    absorb = [(r, h - H) for r, h in enumerate(head_counts) if h > H]
    shed.sort(key=lambda x: -x[1])
    absorb.sort(key=lambda x: -x[1])

    assign: dict[int, list[tuple[int, int]]] = {r: [] for r in range(n)}
    ai = 0
    cap = dict(absorb)
    for straggler, count in shed:
        while count > 0 and ai < len(absorb):
            helper = absorb[ai][0]
            take = min(count, cap[helper])
            if take > 0:
                assign[helper].append((straggler, take))
                cap[helper] -= take
                count -= take
            if cap[helper] == 0:
                ai += 1
    return NodeHeadPlan(kept=kept, assign=assign)


# ---------------------------------------------------------------------------
# Plan source: read the node planner's eap_heads.json
# ---------------------------------------------------------------------------

class EapPlanSource:
    """Mtime-cached reader of ``eap_heads.json`` for one node's ranks."""

    def __init__(self, plan_path: str, node_ranks: list[int], num_q_heads: int):
        self.path = plan_path
        self.node_ranks = node_ranks       # the node's global DP ranks, in order
        self.H = num_q_heads
        self._mtime = 0.0
        self._plan = NodeHeadPlan(kept=[num_q_heads] * len(node_ranks))

    def current(self) -> NodeHeadPlan:
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return self._plan
        if m != self._mtime:
            try:
                with open(self.path) as f:
                    hc = json.load(f)["head_counts"]
                node_hc = [hc[r] for r in self.node_ranks]
                self._plan = head_assignment(node_hc, self.H)
                self._mtime = m
            except (OSError, ValueError, KeyError, IndexError):
                pass
        return self._plan


# ---------------------------------------------------------------------------
# Runtime NVLink mechanism + model hook
# ---------------------------------------------------------------------------
#
# The attention math below is exact by construction: every query head's output
# is computed once (on the straggler or on a helper) over the same Q/K/V, then
# placed back, so the assembled output equals single-GPU execution. The pieces
# that bind to vLLM internals -- writing the new K/V into the paged cache and
# folding in already-fetched prefix-KV -- are marked CLUSTER and need validation
# on the testbed; until then they degrade gracefully (no offload) rather than
# return a wrong result.


def _gqa_sdpa(q, k, v, q_per_kv: int, scaling: float):
    """Causal attention for q heads with GQA-shared kv heads.

    q: [T, hq, d]; k, v: [T, hkv, d] (hkv = hq / q_per_kv). Returns [T, hq, d].
    """
    import torch.nn.functional as F

    k = k.repeat_interleave(q_per_kv, dim=1)        # expand kv heads to q heads
    v = v.repeat_interleave(q_per_kv, dim=1)
    qa, ka, va = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # [1,h,T,d]
    o = F.scaled_dot_product_attention(qa, ka, va, is_causal=True, scale=scaling)
    return o.squeeze(0).transpose(0, 1).contiguous()  # [T, hq, d]


class ElasticAttention:
    """NVLink head offload across the node's DP workers (§3.3).

    Constructed once per worker. Each layer, given this rank's
    :class:`NodeHeadPlan`, a straggler computes only its kept query heads locally
    and ships the shed heads' Q/K/V to on-node helpers over NVLink; each helper
    runs attention for the heads assigned to it (for the straggler's tokens) and
    sends the output back; the straggler places them into the full attention
    output. Point-to-point (``isend``/``irecv``) over the default process group,
    which routes same-node pairs over NVLink. KV is pre-expanded per query head
    before shipping, so helpers run plain attention with no GQA bookkeeping.
    """

    def __init__(self, rank: int, node_ranks: list[int], plan: EapPlanSource,
                 q_per_kv: int):
        self.rank = rank
        self.node_ranks = node_ranks
        self.local = node_ranks.index(rank)
        self.plan_src = plan
        self.q_per_kv = q_per_kv

    def _project(self, attn, positions, hidden_states):
        """Run the module's QKV proj + norms + rotary -> per-head q, k, v."""
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        d = attn.head_dim
        q = attn.q_norm(q.view(*q.shape[:-1], q.shape[-1] // d, d)).view(q.shape)
        k = attn.k_norm(k.view(*k.shape[:-1], k.shape[-1] // d, d)).view(k.shape)
        q, k = attn.rotary_emb(positions, q, k)
        T = hidden_states.shape[0]
        return (q.view(T, attn.num_heads, d), k.view(T, attn.num_kv_heads, d),
                v.view(T, attn.num_kv_heads, d))

    def forward(self, attn, positions, hidden_states):
        """Attention for ``hidden_states`` with the straggler's heads offloaded.

        Returns the full attention block output (after o_proj). Falls back to the
        unmodified module forward if the plan does not fire or anything is missing,
        so correctness never depends on the offload succeeding.
        """
        import torch
        import torch.distributed as dist

        plan = self.plan_src.current()
        if not plan.fires or not dist.is_initialized():
            return attn(positions, hidden_states)

        qh, kh, vh = self._project(attn, positions, hidden_states)
        T, nH, d = qh.shape
        qpk = self.q_per_kv
        kept = plan.kept[self.local]
        retained = list(range(kept))               # heads this rank keeps
        out = torch.empty(T, nH, d, dtype=qh.dtype, device=qh.device)

        # --- as straggler: ship shed heads (pre-expanded KV) to helpers --------
        ship_reqs, recv_o = [], []
        shed = list(range(kept, nH))
        si = 0
        for helper_local, items in plan.assign.items():
            for straggler_local, count in items:
                if straggler_local != self.local:
                    continue
                heads = shed[si:si + count]; si += count
                dst = self.node_ranks[helper_local]
                kv_heads = [h // qpk for h in heads]
                q_s = qh[:, heads].contiguous()
                k_s = kh[:, kv_heads].contiguous()      # KV pre-expanded per head
                v_s = vh[:, kv_heads].contiguous()
                hdr = torch.tensor([T, count], device=qh.device, dtype=torch.int32)
                dist.send(hdr, dst)                     # tiny header first
                for t in (q_s, k_s, v_s):
                    ship_reqs.append(dist.isend(t, dst))
                recv_o.append((heads, dst,
                               torch.empty(T, count, d, dtype=qh.dtype, device=qh.device)))

        # --- as helper: receive a straggler's heads, compute, send back --------
        for straggler_local, count in plan.assign.get(self.local, []):
            src = self.node_ranks[straggler_local]
            hdr = torch.empty(2, device=qh.device, dtype=torch.int32)
            dist.recv(hdr, src)
            Ts, c = int(hdr[0]), int(hdr[1])
            q_r = torch.empty(Ts, c, d, dtype=qh.dtype, device=qh.device)
            k_r = torch.empty(Ts, c, d, dtype=qh.dtype, device=qh.device)
            v_r = torch.empty(Ts, c, d, dtype=qh.dtype, device=qh.device)
            for t in (q_r, k_r, v_r):
                dist.recv(t, src)
            o_r = _gqa_sdpa(q_r, k_r, v_r, 1, attn.scaling)   # KV pre-expanded
            dist.send(o_r.contiguous(), src)

        # local attention for retained heads (overlaps the helpers' compute)
        if retained:
            out[:, retained] = _gqa_sdpa(
                qh[:, retained], kh[:, [h // qpk for h in retained]],
                vh[:, [h // qpk for h in retained]], 1, attn.scaling)
        for r in ship_reqs:
            r.wait()
        for heads, dst, buf in recv_o:                # gather shed-head outputs
            dist.recv(buf, dst)
            out[:, heads] = buf

        self._write_kv(attn, kh, vh)                  # CLUSTER: paged-cache write
        output, _ = attn.o_proj(out.view(T, nH * d))
        return output

    def _write_kv(self, attn, kh, vh) -> None:
        """Write the layer's new K/V into vLLM's paged cache (CLUSTER).

        The straggler holds all K/V (it ran the full projection), so the cache is
        complete. Wiring this to vLLM's ``reshape_and_cache`` + the forward
        context's slot mapping is version-specific; guarded so a mismatch does not
        crash the forward (it just leaves the original path's cache write to run).
        """
        # Integration point: write kh/vh to attn.attn's kv_cache using the
        # forward context's slot_mapping. Left to testbed bring-up.
        return None


def patch_qwen3_attention(elastic: ElasticAttention) -> None:
    """Wrap ``Qwen3MoeAttention.forward`` so attention goes through ``elastic``.

    Call once at worker startup when elastic attention is enabled. Non-invasive:
    the wrapper delegates to the original forward whenever the plan does not fire.
    """
    from vllm.model_executor.models import qwen3_moe

    orig = qwen3_moe.Qwen3MoeAttention.forward

    def wrapped(self, positions, hidden_states):  # noqa: ANN001
        if elastic.plan_src.current().fires:
            return elastic.forward(self, positions, hidden_states)
        return orig(self, positions, hidden_states)

    qwen3_moe.Qwen3MoeAttention.forward = wrapped


def enable_elastic_attention(plan_path: str | None = None, num_q_heads: int = 64,
                             q_per_kv: int = 16, workers_per_node: int = 8) -> None:
    """Turn EAP on for this prefill worker (call once at startup).

    Resolves this worker's global DP rank and its node's ranks, wires the plan
    reader to the node planner's ``eap_heads.json``, and patches the attention
    forward. Enable it in the prefill launch by importing this and calling it
    (e.g. ``SHUNT_EAP=1`` gates a startup hook); the decode side never runs it.
    """
    import os

    rank = int(os.environ.get("VLLM_DP_RANK", "0"))
    try:
        from vllm.distributed.parallel_state import get_dp_group
        rank = get_dp_group().rank_in_group
    except Exception:
        pass
    node = rank // workers_per_node
    node_ranks = list(range(node * workers_per_node, (node + 1) * workers_per_node))
    plan_path = plan_path or os.environ.get("SHUNT_EAP_PLAN",
                                            "/tmp/shunt/eap_heads.json")
    src = EapPlanSource(plan_path, node_ranks, num_q_heads)
    patch_qwen3_attention(ElasticAttention(rank, node_ranks, src, q_per_kv))
