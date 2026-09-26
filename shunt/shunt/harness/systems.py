"""The system configurations of the evaluation, by name.

A system fixes the proxy's placement, the per-iteration plan options, whether
KV rides a lower-priority traffic class than the A2A, and the engine settings.
``python -m shunt.harness.run --list-systems`` prints them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class System:
    name: str
    placement: str = "rr"          # rr | lpt | oracle | kva | kva_lb
    eap: bool = False
    theta: float = 1.5
    kvlb_budget: bool = False      # per-port budget and overflow paths
    kvlb_borrow: bool = True
    kvlb_frontend: bool = True
    kv_priority: bool = False      # KV in a lower traffic class than the A2A
    kv_mode: str = "pool"          # pool: Mooncake store on the decode nodes
                                   # local: prefill-local CPU memory only
    kv_router: bool = False        # Shunt's per-port KV routing backend (all
                                   # Shunt variants); otherwise stock LMCache
    a2a: str = "nccl"              # nccl | deepep (vLLM all2all backend)
    dbo: bool = False
    prefix_cache: bool = True      # vLLM prefix cache and LMCache pool reuse
    tp: int = 1
    chunked_prefill_tokens: int | None = None
    notes: str = ""

    @property
    def runtime_needed(self) -> bool:
        """Whether prefill ranks run the Shunt runtime (planner, elastic
        attention, KV routing, timing)."""
        return self.kv_router or self.eap or self.kvlb_budget

    def options(self) -> dict:
        return {"eap": self.eap, "theta": self.theta,
                "kvlb_budget": self.kvlb_budget, "kvlb_borrow": self.kvlb_borrow,
                "kvlb_frontend": self.kvlb_frontend, "dbo": self.dbo}


_SHUNT = System("shunt", placement="lpt", eap=True, kvlb_budget=True,
                kv_priority=True, kv_router=True,
                notes="LPT placement, elastic attention, KVLB")

_BASE: dict[str, System] = {
    "baseline": System("baseline", notes="round-robin dispatch, all KV on the backend"),
    "ors": System("ors", placement="oracle",
                  notes="Baseline with the offline compute-balanced oracle placement"),
    "combo": System("combo", placement="lpt", kv_priority=True,
                    notes="LPT placement and low-priority KV, no elastic attention "
                          "or budgets"),
    "shunt": _SHUNT,
    "sh-ors": replace(_SHUNT, name="sh-ors", placement="oracle",
                      notes="Shunt with the oracle placement"),
    # leave-one-out ablation
    "no-rs": replace(_SHUNT, name="no-rs", placement="rr", notes="Shunt without LPT"),
    "no-eap": replace(_SHUNT, name="no-eap", eap=False,
                      notes="Shunt without elastic attention"),
    "no-kvlb": replace(_SHUNT, name="no-kvlb", kvlb_budget=False, kv_priority=False,
                       notes="Shunt without KVLB: no priority, budget, or offload"),
    # inside KVLB
    "no-prio": replace(_SHUNT, name="no-prio", kv_priority=False,
                       notes="budget and both paths, KV in the A2A's class"),
    "no-budget": replace(_SHUNT, name="no-budget", kvlb_budget=False,
                         notes="priority only; KV never leaves its port"),
    "no-borrow": replace(_SHUNT, name="no-borrow", kvlb_borrow=False,
                         notes="overflow to the frontend only"),
    "no-frontend": replace(_SHUNT, name="no-frontend", kvlb_frontend=False,
                           notes="overflow to other backend ports only"),
    # KV-cache-aware routing, alone and inside Shunt
    "kva": System("kva", placement="kva", notes="KV-cache-aware routing alone"),
    "kva-lb": System("kva-lb", placement="kva_lb",
                     notes="KV-cache-aware routing with load balancing alone"),
    "lpt": System("lpt", placement="lpt", notes="LPT placement alone"),
    "shunt-kva": replace(_SHUNT, name="shunt-kva", placement="kva",
                         notes="Shunt with KVA in place of LPT"),
    "shunt-kva-lb": replace(_SHUNT, name="shunt-kva-lb", placement="kva_lb",
                            notes="Shunt with KVA-LB in place of LPT"),
    # contention measurement
    "no-contention": System("no-contention", kv_mode="local",
                            notes="Baseline with prefill KV in local CPU memory only"),
}


def get(name: str) -> System:
    """A system by name. Suffixes compose:

    - ``+deepep``: DeepEP high-throughput A2A kernels;
    - ``+dbo``: DeepEP with dual-batch overlap (implies ``+deepep``);
    - ``+nocache``: prefix cache off;
    - ``+tpN``: TP=N (DP = GPUs / N);
    - ``+chunkN``: chunked prefill with N-token chunks;
    - ``+thetaX``: compute-straggler threshold X.
    """
    base, *mods = name.split("+")
    if base not in _BASE:
        raise KeyError(f"unknown system {base!r}; known: {', '.join(sorted(_BASE))}")
    s = _BASE[base]
    for m in mods:
        if m == "deepep":
            s = replace(s, a2a="deepep")
        elif m == "dbo":
            s = replace(s, a2a="deepep", dbo=True)
        elif m == "nocache":
            s = replace(s, prefix_cache=False)
        elif m.startswith("tp"):
            s = replace(s, tp=int(m[2:]))
        elif m.startswith("chunk"):
            s = replace(s, chunked_prefill_tokens=int(m[5:]))
        elif m.startswith("theta"):
            s = replace(s, theta=float(m[5:]))
        else:
            raise KeyError(f"unknown modifier {m!r} in {name!r}")
    return replace(s, name=name)


def names() -> list[str]:
    return sorted(_BASE)


def describe() -> str:
    lines = [f"{n:14s} {s.notes}" for n, s in sorted(_BASE.items())]
    lines.append("modifiers: +deepep +dbo +nocache +tpN +chunkN +thetaX "
                 "(e.g. shunt+dbo, baseline+tp4, shunt+theta2.5)")
    return "\n".join(lines)
