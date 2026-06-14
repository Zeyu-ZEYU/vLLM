"""Plain data carriers shared by the planner, proxy, and node planner."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    """One admitted request, as the scheduler sees it (§3.2)."""

    req_id: str
    prefix_tokens: int   # reused-prefix length P (known at admission)
    fresh_tokens: int    # freshly prefilled tokens N (known at admission)
    compute_time: float = 0.0  # worker-compute time it adds (filled by the model)


@dataclass
class DirectionPlan:
    """KVLB allocation for one worker, one direction (inbound or outbound).

    Bytes are per layer. ``backend`` maps a backend-port index to bytes this
    worker sends there (its own port plus any it borrows); ``frontend`` is bytes
    on the shared frontend RNIC. Overflow past every budget is folded into these
    once spread, so the maps are the complete placement.
    """

    owner: int = 0
    backend: dict[int, float] = field(default_factory=dict)
    frontend: float = 0.0

    @property
    def own(self) -> float:
        """Bytes kept on the worker's own backend port."""
        return self.backend.get(self.owner, 0.0)

    @property
    def borrowed(self) -> dict[int, float]:
        """Bytes lent to other workers' backend ports, by port index."""
        return {p: b for p, b in self.backend.items() if p != self.owner and b > 0}

    @property
    def total(self) -> float:
        return sum(self.backend.values()) + self.frontend


@dataclass
class IterationPlan:
    """The full per-iteration plan (§3.1): one object, computed before any layer.

    Indices are EP-group worker ranks (0..ep_group_workers-1). ``assignment`` maps
    a request id to the worker RS placed it on; ``head_counts`` is EAP's per-worker
    query-head count; ``t_cmp`` is the compute-window length after the head split;
    ``inbound``/``outbound`` are the KVLB allocations per worker.
    """

    assignment: dict[str, int] = field(default_factory=dict)
    head_counts: list[int] = field(default_factory=list)
    worker_compute: list[float] = field(default_factory=list)  # post-split tau^wk
    attention_time: list[float] = field(default_factory=list)
    t_cmp: float = 0.0
    t_a2a: float = 0.0
    inbound: list[DirectionPlan] = field(default_factory=list)
    outbound: list[DirectionPlan] = field(default_factory=list)
