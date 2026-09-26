"""Reference implementations of Shunt's per-iteration algorithms.

- :func:`lpt_schedule`: LPT placement of requests onto DP workers (RS).
- :func:`optimal_oracle`: offline min-makespan placement (the ORS baseline).
- :func:`balance_heads`: per-node attention-head balancing (Algorithm S1).
- :func:`allocate_offload`: per-node KV offload allocation (Algorithm 1).

The C++ versions in ``csrc/include/shunt_core.hpp`` implement the same rules
with the same tie-breaking; ``tests/test_native_parity.py`` checks that both
produce identical results. The engine runtime and the proxy use the C++ code.
"""
from __future__ import annotations

import math

EPS = 1e-9


# ---------------------------------------------------------------------------
# Request placement
# ---------------------------------------------------------------------------

def lpt_schedule(compute_times: list[float], num_workers: int,
                 init_load: list[float] | None = None) -> list[int]:
    """Longest-processing-time placement.

    Requests are taken heaviest first (ties: lower index first) and each goes
    to the currently least-loaded worker (ties: lower worker index).
    ``init_load`` is the work already placed on each worker (default: none).
    Returns ``worker_of[i]`` for every request ``i``.
    """
    n = len(compute_times)
    order = sorted(range(n), key=lambda i: (-compute_times[i], i))
    load = [float(x) for x in init_load] if init_load is not None \
        else [0.0] * num_workers
    worker_of = [0] * n
    for i in order:
        w = min(range(num_workers), key=lambda x: (load[x], x))
        worker_of[i] = w
        load[w] += compute_times[i]
    return worker_of


def makespan_lower_bound(compute_times: list[float], num_workers: int) -> float:
    """No placement can beat the largest request or the mean load."""
    if not compute_times:
        return 0.0
    return max(max(compute_times), sum(compute_times) / num_workers)


def optimal_oracle(compute_times: list[float], num_workers: int,
                   max_rounds: int = 20000) -> list[int]:
    """Offline min-makespan placement used by the ORS baseline.

    Starts from LPT and applies improving moves and swaps between the most
    loaded worker and the others until no move lowers the makespan or the
    makespan meets :func:`makespan_lower_bound`, at which point it is optimal.
    """
    c = compute_times
    n, W = len(c), num_workers
    worker_of = lpt_schedule(c, W)
    groups: list[list[int]] = [[] for _ in range(W)]
    load = [0.0] * W
    for i, w in enumerate(worker_of):
        groups[w].append(i)
        load[w] += c[i]
    bound = makespan_lower_bound(c, W)

    for _ in range(max_rounds):
        hi = max(range(W), key=lambda w: (load[w], -w))
        if load[hi] <= bound * (1 + 1e-12):
            break
        improved = False
        others = sorted((w for w in range(W) if w != hi), key=lambda w: (load[w], w))
        # single move: send a job from hi to a worker where it lowers the max
        for i in sorted(groups[hi], key=lambda x: (c[x], x)):
            for lo in others:
                if load[lo] + c[i] < load[hi] - EPS:
                    groups[hi].remove(i)
                    groups[lo].append(i)
                    load[hi] -= c[i]
                    load[lo] += c[i]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # swap a larger job on hi with a smaller one elsewhere
            for lo in others:
                for i in sorted(groups[hi], key=lambda x: (-c[x], x)):
                    for j in sorted(groups[lo], key=lambda x: (c[x], x)):
                        delta = c[i] - c[j]
                        if delta > EPS and load[lo] + delta < load[hi] - EPS:
                            groups[hi].remove(i)
                            groups[lo].remove(j)
                            groups[hi].append(j)
                            groups[lo].append(i)
                            load[hi] -= delta
                            load[lo] += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break

    out = [0] * n
    for w, g in enumerate(groups):
        for i in g:
            out[i] = w
    return out


# ---------------------------------------------------------------------------
# Algorithm S1: attention-head balancing on one prefill node
# ---------------------------------------------------------------------------

def balance_heads(worker_compute: list[float], attention_time: list[float],
                  group_mean: float, num_q_heads: int, theta: float
                  ) -> tuple[list[tuple[int, int]], list[float]]:
    """Hand query heads from the busiest worker to the idlest, one at a time.

    ``worker_compute`` and ``attention_time`` hold the node's workers only;
    ``group_mean`` is the mean worker-compute time over the whole EP group.
    Nothing moves unless a worker exceeds ``theta * group_mean``. Each move
    takes one of the donor's own heads, carrying ``attention_time[s] / H``.
    The loop stops when the busiest worker has none of its own heads left or
    when the move would bring the recipient up to the donor's time.

    Returns the moves ``[(donor, recipient), ...]`` in order (node-local
    indices, one per head) and the post-split worker-compute times.
    """
    W, H = len(worker_compute), num_q_heads
    t = [float(x) for x in worker_compute]
    if W == 0 or H <= 0 or max(t) <= theta * group_mean:
        return [], t
    e = [float(a) / H for a in attention_time]
    own = [H] * W
    moves: list[tuple[int, int]] = []
    while True:
        s = max(range(W), key=lambda w: (t[w], -w))
        d = min(range(W), key=lambda w: (t[w], w))
        if own[s] == 0 or e[s] <= 0.0 or t[d] + e[s] >= t[s]:
            break
        t[s] -= e[s]
        t[d] += e[s]
        own[s] -= 1
        moves.append((s, d))
    return moves, t


# ---------------------------------------------------------------------------
# Algorithm 1: KV offload allocation on one prefill node, one direction
# ---------------------------------------------------------------------------

def waterfill(load: list[float], bw: list[float], extra: float) -> list[float]:
    """Add ``extra`` bytes over links so the latest finish time is minimal.

    A link's finish time is ``load / bw``. Returns the bytes added per link.
    Exact: raises the lowest links to a common finish level.
    """
    n = len(load)
    added = [0.0] * n
    if extra <= EPS or n == 0:
        return added
    order = sorted(range(n), key=lambda i: (load[i] / bw[i], i))
    cum_b = cum_l = 0.0
    level = 0.0
    k = 0
    for k, i in enumerate(order):
        cum_b += bw[i]
        cum_l += load[i]
        level = (extra + cum_l) / cum_b
        nxt = load[order[k + 1]] / bw[order[k + 1]] if k + 1 < n else math.inf
        if level <= nxt:
            break
    for i in order[:k + 1]:
        added[i] = max(0.0, level * bw[i] - load[i])
    return added


def allocate_offload(volumes: list[float], budget_be: float, budget_fe: float,
                     bw_port: float, bw_fe: float,
                     pcie_distance: list[list[float]] | None = None,
                     allow_borrow: bool = True, allow_frontend: bool = True
                     ) -> list[dict]:
    """KV allocation for one node and one direction (Algorithm 1).

    ``volumes[w]`` is worker ``w``'s KV bytes per layer. Each worker first
    fills its own port up to ``budget_be``. Over-budget workers, largest
    overflow first, borrow spare budget on other ports (nearest in
    ``pcie_distance`` first), then use the frontend up to ``budget_fe``. What
    no budget absorbs is spread over the links the worker may use, balancing
    their finish times. Returns one dict per worker with ``own`` bytes,
    ``borrow`` (port -> bytes), and ``frontend`` bytes.
    """
    W = len(volumes)

    def dist(i: int, j: int) -> float:
        return pcie_distance[i][j] if pcie_distance is not None else float(abs(i - j))

    own = [min(v, budget_be) for v in volumes]
    spare = [budget_be - o for o in own]
    residual = [max(0.0, v - budget_be) for v in volumes]
    borrow = [[0.0] * W for _ in range(W)]
    fe = [0.0] * W
    fe_left = budget_fe if allow_frontend else 0.0

    order = sorted(range(W), key=lambda x: (-residual[x], x))
    for w in order:
        if residual[w] <= EPS:
            continue
        if allow_borrow:
            for p in sorted(range(W), key=lambda j: (dist(w, j), j)):
                if residual[w] <= EPS:
                    break
                if p == w or spare[p] <= EPS:
                    continue
                take = min(residual[w], spare[p])
                borrow[w][p] += take
                spare[p] -= take
                residual[w] -= take
        if allow_frontend and residual[w] > EPS and fe_left > EPS:
            take = min(residual[w], fe_left)
            fe[w] += take
            fe_left -= take
            residual[w] -= take

    # Beyond every budget: spread over the allowed links, balancing finish times.
    port_load = [own[p] + sum(borrow[w][p] for w in range(W)) for p in range(W)]
    fe_load = sum(fe)
    for w in sorted(range(W), key=lambda x: (-residual[x], x)):
        if residual[w] <= EPS:
            continue
        links = [p for p in range(W) if p == w or allow_borrow]
        loads = [port_load[p] for p in links]
        bws = [bw_port] * len(links)
        if allow_frontend:
            loads.append(fe_load)
            bws.append(bw_fe)
        added = waterfill(loads, bws, residual[w])
        for k, p in enumerate(links):
            if added[k] <= 0.0:
                continue
            if p == w:
                own[w] += added[k]
            else:
                borrow[w][p] += added[k]
            port_load[p] += added[k]
        if allow_frontend and added[-1] > 0.0:
            fe[w] += added[-1]
            fe_load += added[-1]
        residual[w] = 0.0

    return [{"own": own[w],
             "borrow": {p: b for p, b in enumerate(borrow[w]) if b > 0.0},
             "frontend": fe[w]} for w in range(W)]
