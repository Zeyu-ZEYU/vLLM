"""The three Shunt algorithms (§3.2-§3.4) plus the offline oracle baseline.

These are the reference implementations: short, pure-Python, and algorithmically
identical to the C++ in ``csrc/`` (a parity test pins them together). The online
proxy and node planner call the C++ for the microsecond-scale decision cost the
paper reports; the offline analyses and tests call these.
"""
from __future__ import annotations

from .types import DirectionPlan


# ---------------------------------------------------------------------------
# Algorithm 1: compute-aware request scheduling (RS), §3.2
# ---------------------------------------------------------------------------

def lpt_schedule(compute_times: list[float], num_workers: int) -> list[int]:
    """Longest-processing-time assignment of requests to workers (Alg. 1).

    Take requests heaviest-first and put each on the worker lightest so far; this
    makes the largest worker load (the straggler's compute time) as small as a
    fast online rule can. Returns ``worker_of[i]`` for each input request ``i``.
    """
    load = [0.0] * num_workers
    worker_of = [0] * len(compute_times)
    order = sorted(range(len(compute_times)), key=lambda i: compute_times[i], reverse=True)
    for i in order:
        w = min(range(num_workers), key=lambda w: load[w])
        worker_of[i] = w
        load[w] += compute_times[i]
    return worker_of


def optimal_oracle(compute_times: list[float], num_workers: int,
                   iters: int = 4000) -> list[int]:
    """Offline min-makespan assignment used as the ORS baseline (§2.3, §4.2).

    LPT seed then local moves and swaps toward the lower bound
    ``max(total/W, largest job)``. Far too slow to run online (it is the
    "oracle"); RS approximates it. Returns ``worker_of[i]``.
    """
    W = num_workers
    n = len(compute_times)
    c = compute_times
    order = sorted(range(n), key=lambda i: c[i], reverse=True)
    load = [0.0] * W
    groups: list[list[int]] = [[] for _ in range(W)]
    for i in order:
        w = min(range(W), key=lambda w: load[w])
        groups[w].append(i)
        load[w] += c[i]

    for _ in range(iters):
        hi = max(range(W), key=lambda w: load[w])
        order_lo = sorted(range(W), key=lambda w: load[w])
        moved = False
        if len(groups[hi]) > 1:
            for i in sorted(groups[hi], key=lambda x: c[x]):
                for lo in order_lo:
                    if lo != hi and load[lo] + c[i] < load[hi] - 1e-12:
                        groups[hi].remove(i); groups[lo].append(i)
                        load[hi] -= c[i]; load[lo] += c[i]; moved = True
                        break
                if moved:
                    break
        if not moved:
            for lo in order_lo:
                if lo == hi:
                    continue
                for i in sorted(groups[hi], key=lambda x: -c[x]):
                    for j in sorted(groups[lo], key=lambda x: c[x]):
                        if c[i] > c[j] and load[hi] - c[i] + c[j] < load[hi] - 1e-12 \
                           and load[lo] - c[j] + c[i] < load[hi] - 1e-12:
                            groups[hi].remove(i); groups[lo].remove(j)
                            groups[hi].append(j); groups[lo].append(i)
                            load[hi] += c[j] - c[i]; load[lo] += c[i] - c[j]
                            moved = True; break
                    if moved:
                        break
                if moved:
                    break
        if not moved:
            break

    worker_of = [0] * n
    for w, g in enumerate(groups):
        for i in g:
            worker_of[i] = w
    return worker_of


# ---------------------------------------------------------------------------
# Algorithm 2: straggler-aware elastic attention parallelism (EAP), §3.3
# ---------------------------------------------------------------------------

def balance_heads(worker_compute: list[float], attention_time: list[float],
                  group_mean: float, num_q_heads: int, theta: float = 1.5
                  ) -> tuple[list[int], list[float]]:
    """Per-node attention-head balancing (Alg. 2).

    Acts only when the node's worst worker-compute time exceeds ``theta`` times
    the *group* mean; then it hands one query head at a time from the busiest
    worker to the idlest until every GPU is within one head of the node mean.
    Returns ``(head_counts, post_split_worker_compute)`` for this node's workers.

    ``worker_compute`` and ``attention_time`` are this node's workers only;
    ``group_mean`` is the mean over the whole EP group (the straggler is judged
    against the group, not the node).
    """
    W = len(worker_compute)
    H = num_q_heads
    if W == 0 or max(worker_compute) <= theta * group_mean:
        return [H] * W, list(worker_compute)

    t = list(worker_compute)
    e = [a / H for a in attention_time]   # per-head attention time, per worker
    h = [H] * W
    while True:
        s = max(range(W), key=lambda w: t[w])
        d = min(range(W), key=lambda w: t[w])
        if t[s] - t[d] <= e[s] or h[s] <= 0:
            break
        t[s] -= e[s]; t[d] += e[s]      # hand one of s's heads to d
        h[s] -= 1;   h[d] += 1
    return h, t


# ---------------------------------------------------------------------------
# Algorithm 3: contention- and urgency-aware KV traffic load balancing, §3.4
# ---------------------------------------------------------------------------

def allocate_offload(volumes: list[float], budget_be: float, budget_fe: float,
                     bw_port: float = 1.0, bw_fe: float = 1.0,
                     pcie_distance: list[list[float]] | None = None
                     ) -> list[DirectionPlan]:
    """Per-iteration KV offload allocation for one direction (Alg. 3).

    Each backend port carries ``budget_be`` (= BW_port x T_cmp, from the post-split
    window); the shared frontend carries ``budget_fe``. A worker's KV first fills
    its own port; overflow borrows spare budget on the nearest backend ports
    (PCIe distance), then the idle frontend; anything past every budget is spread
    across all links to balance finish times (``bw_port``/``bw_fe`` set the
    relative drain rates). ``volumes[w]`` is worker w's KV bytes (one direction,
    one layer). Returns one :class:`DirectionPlan` per worker.
    """
    W = len(volumes)
    if pcie_distance is None:
        pcie_distance = [[abs(i - j) for j in range(W)] for i in range(W)]

    plans = [DirectionPlan(owner=w) for w in range(W)]
    spare = [0.0] * W           # spare backend budget left on each port
    residual = [0.0] * W        # bytes still unplaced for each worker
    for w in range(W):
        own = min(volumes[w], budget_be)
        plans[w].backend[w] = own
        spare[w] = budget_be - own
        residual[w] = max(0.0, volumes[w] - budget_be)

    fe_left = budget_fe
    for w in sorted(range(W), key=lambda x: residual[x], reverse=True):
        if residual[w] <= 0:
            continue
        # 1) borrow nearest backend ports with spare budget (PCIe distance)
        for lender in sorted(range(W), key=lambda j: (pcie_distance[w][j], j)):
            if residual[w] <= 1e-12:
                break
            if lender == w or spare[lender] <= 0:
                continue
            take = min(residual[w], spare[lender])
            plans[w].backend[lender] = plans[w].backend.get(lender, 0.0) + take
            spare[lender] -= take
            residual[w] -= take
        # 2) then the idle frontend, up to its remaining budget
        if residual[w] > 1e-12 and fe_left > 0:
            take = min(residual[w], fe_left)
            plans[w].frontend += take
            fe_left -= take
            residual[w] -= take

    # 3) anything past every budget spills past the window: spread it across all
    #    links to equalize finish times (minimize the transfer straggler). Each
    #    link's committed load and bandwidth set its current finish time; water-
    #    fill the leftover so the slowest link finishes as early as possible.
    leftover = [(w, residual[w]) for w in range(W) if residual[w] > 1e-12]
    if leftover:
        committed = [budget_be - spare[p] for p in range(W)] + [budget_fe - fe_left]
        bw = [bw_port] * W + [bw_fe]
        total_extra = sum(r for _, r in leftover)
        extra = _waterfill(committed, bw, total_extra)   # bytes to add per link
        # assign the extra on each link to the still-residual workers in order
        donors = [w for w, _ in sorted(leftover, key=lambda x: -x[1])]
        di = 0
        for link in range(W + 1):
            give = extra[link]
            while give > 1e-12 and di < len(donors):
                w = donors[di]
                take = min(give, residual[w])
                if link < W:
                    plans[w].backend[link] = plans[w].backend.get(link, 0.0) + take
                else:
                    plans[w].frontend += take
                residual[w] -= take
                give -= take
                if residual[w] <= 1e-12:
                    di += 1
    return plans


def _waterfill(load: list[float], bw: list[float], extra: float) -> list[float]:
    """Add ``extra`` total bytes across links to minimize the max finish time.

    Returns per-link added bytes. Finish time of a link is (load+added)/bw; we
    raise all links to a common finish level T with sum_l max(0, T*bw-load)=extra.
    """
    n = len(load)
    if extra <= 0:
        return [0.0] * n
    # binary search the common finish time T
    lo, hi = 0.0, max(load[i] / bw[i] for i in range(n)) + extra / sum(bw)
    for _ in range(100):
        T = 0.5 * (lo + hi)
        need = sum(max(0.0, T * bw[i] - load[i]) for i in range(n))
        if need < extra:
            lo = T
        else:
            hi = T
    T = hi
    return [max(0.0, T * bw[i] - load[i]) for i in range(n)]
