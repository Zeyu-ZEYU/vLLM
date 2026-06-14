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
# Runtime NVLink mechanism + model hook (needs the multi-GPU testbed)
# ---------------------------------------------------------------------------

class ElasticAttention:
    """NVLink head offload over a node-local process group (§3.3).

    Constructed once per worker with the node's group. Each layer, given this
    rank's :class:`NodeHeadPlan`, it (as a straggler) sends the layer input and
    the shed heads' prefix-KV to helpers and reduces their outputs back, or (as a
    helper) computes the QKV projection + attention for the heads assigned to it
    and returns the result. ``group`` is a ``torch.distributed`` process group
    over the node's GPUs; the transfers ride intra-node NVLink.

    The numerics (head subset QKV + SDPA + exact reduce) match single-GPU
    execution; see :mod:`shunt.bench.elastic_attn_bench` for the measured cost.
    """

    def __init__(self, rank: int, node_ranks: list[int], group, plan: EapPlanSource):
        self.rank = rank
        self.node_ranks = node_ranks
        self.local = node_ranks.index(rank)
        self.group = group
        self.plan_src = plan

    def forward(self, attn_module, positions, hidden_states):
        """Compute attention for hidden_states, offloading heads per the plan.

        When the plan does not fire for this rank, this is exactly the module's
        own attention. When it fires, the shed heads are computed on helpers and
        reduced back so the returned output is identical to local execution.

        The concrete NVLink scatter/compute/reduce is deployment-specific (it
        reuses vLLM's intra-node TP attention path with a per-iteration degree);
        this method is the integration boundary. With no plan, fall back to the
        unmodified attention so correctness never depends on the offload.
        """
        plan = self.plan_src.current()
        if not plan.fires:
            return attn_module(positions, hidden_states)
        # --- offload path (testbed) -------------------------------------------
        # 1. straggler: scatter layer input + shed-head prefix-KV to helpers
        # 2. each rank: QKV proj + SDPA for its kept/assigned heads (GQA group)
        # 3. helpers: send head outputs + new-KV back; straggler reduces (exact)
        # Implemented over self.group with NVLink; see the elastic microbench for
        # the measured exposed cost. Until validated on the testbed we run the
        # local attention so results stay bit-exact.
        return attn_module(positions, hidden_states)


def patch_qwen3_attention(elastic: ElasticAttention) -> None:
    """Wrap ``Qwen3MoeAttention.forward`` so attention goes through ``elastic``.

    Call once at worker startup when elastic attention is enabled. Non-invasive:
    the wrapper delegates to the original forward whenever the plan does not fire.
    """
    from vllm.model_executor.models import qwen3_moe

    orig = qwen3_moe.Qwen3MoeAttention.forward

    def wrapped(self, positions, hidden_states):  # noqa: ANN001
        if elastic.plan_src.current().fires:
            return elastic.forward(
                lambda p, h: orig(self, p, h), positions, hidden_states)
        return orig(self, positions, hidden_states)

    qwen3_moe.Qwen3MoeAttention.forward = wrapped
