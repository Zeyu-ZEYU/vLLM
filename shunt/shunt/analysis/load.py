"""Read one run directory written by ``shunt.harness.run``.

Client-side metrics come from ``requests.jsonl`` (the trace driver): the time
to first token (TTFT) and completions. Per-request decode metrics come from
``proxy.jsonl``. Engine-side metrics come from the per-rank runtime logs under
``shunt/<host>/``: ``plan`` (the group plan, written by rank 0), ``step``
(per-layer phase times), ``reqs`` (the requests of every rank per iteration),
``route`` and ``kvio`` (KV bytes per port).
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(x, q) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, q)) if x.size else float("nan")


class Run:
    def __init__(self, path: str | Path, warmup_s: float = 0.0, cooldown_s: float = 0.0):
        self.path = Path(path)
        self.warmup_s = warmup_s
        self.cooldown_s = cooldown_s
        meta = self.path / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}

    @property
    def name(self) -> str:
        return self.path.name

    # --- client side ----------------------------------------------------------------

    @cached_property
    def requests(self) -> list[dict]:
        rows = read_jsonl(self.path / "requests.jsonl")
        ok = [r for r in rows if "error" not in r and "t_first" in r]
        if not ok:
            return []
        t0 = min(r["t_send"] for r in ok) + self.warmup_s
        t1 = max(r["t_last"] for r in ok) - self.cooldown_s
        return [r for r in ok if t0 <= r["t_send"] and r["t_last"] <= t1]

    def errors(self) -> int:
        return sum("error" in r for r in read_jsonl(self.path / "requests.jsonl"))

    def ttft(self, from_schedule: bool | None = None) -> np.ndarray:
        """TTFT in seconds. Open-loop runs measure from the scheduled arrival
        (queueing included) unless ``from_schedule`` says otherwise."""
        if from_schedule is None:
            from_schedule = bool(self.meta.get("load", "").startswith("open"))
        key = "t_sched" if from_schedule else "t_send"
        return np.array([r["t_first"] - r[key] for r in self.requests])

    def completion_rate(self) -> float:
        """Completed requests per second over the measured window."""
        rows = self.requests
        if len(rows) < 2:
            return float("nan")
        t0 = min(r["t_send"] for r in rows)
        t1 = max(r["t_last"] for r in rows)
        return len(rows) / (t1 - t0)

    @cached_property
    def proxy(self) -> dict[str, dict]:
        return {r["id"]: r for r in read_jsonl(self.path / "proxy.jsonl")}

    def tpot(self) -> np.ndarray:
        """Decode time per output token (s), from the proxy's timestamps."""
        out = []
        for r in self.proxy.values():
            n = r.get("n_out", 0)
            if "t_decode_first" in r and n >= 2 and "t_last" in r:
                out.append((r["t_last"] - r["t_decode_first"]) / (n - 1))
        return np.array(out)

    # --- engine side -----------------------------------------------------------------

    def _rank_logs(self, kind: str) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        for p in sorted((self.path / "shunt").rglob(f"{kind}.rank*.jsonl")):
            rank = int(p.stem.split("rank")[-1])
            out[rank] = read_jsonl(p)
        return out

    @cached_property
    def plans(self) -> list[dict]:
        logs = self._rank_logs("plan")
        return logs.get(0, [])

    @cached_property
    def steps(self) -> dict[int, dict[int, dict]]:
        """``steps[seq][rank]`` = that rank's per-layer phase times."""
        by: dict[int, dict[int, dict]] = defaultdict(dict)
        for rank, rows in self._rank_logs("step").items():
            for r in rows:
                by[r["seq"]][rank] = r
        return dict(by)

    @cached_property
    def rank_reqs(self) -> dict[int, dict[int, list]]:
        """``rank_reqs[seq][rank]`` = [(id, prefix, fresh, kv_in, kv_out), ...]."""
        by: dict[int, dict[int, list]] = defaultdict(dict)
        for rank, rows in self._rank_logs("reqs").items():
            for r in rows:
                by[r["seq"]][rank] = r["reqs"]
        return dict(by)

    @cached_property
    def routes(self) -> dict[int, dict[int, dict]]:
        by: dict[int, dict[int, dict]] = defaultdict(dict)
        for rank, rows in self._rank_logs("route").items():
            for r in rows:
                by[r["seq"]][rank] = r
        return dict(by)

    @property
    def num_ranks(self) -> int:
        ranks = set()
        for d in self.steps.values():
            ranks |= set(d)
        return len(ranks)

    def busy_steps(self, min_ranks_with_work: int = 1) -> list[int]:
        """Iterations in which at least ``min_ranks_with_work`` ranks had requests."""
        out = []
        for seq, d in sorted(self.steps.items()):
            if sum(1 for r in d.values() if r.get("nreq", 0) > 0) >= min_ranks_with_work:
                out.append(seq)
        return out


# --- per-iteration quantities ----------------------------------------------------------

COMPUTE_PHASES = ("attn", "gate", "route")


def phase_sum(step: dict, names) -> float:
    """Sum over layers of the given phases (ms)."""
    ph = step.get("phases_ms", {})
    return float(sum(sum(ph.get(n, [])) for n in names))


def a2a_ms(step: dict) -> float:
    ph = step.get("phases_ms", {})
    if "mk_prepare" in ph:
        return float(sum(ph["mk_prepare"]) + sum(ph.get("mk_finalize", [])))
    return float(sum(ph.get("dispatch", [])) + sum(ph.get("combine", [])))


def experts_ms(step: dict) -> float:
    ph = step.get("phases_ms", {})
    if "mk_experts" in ph:
        return float(sum(ph["mk_experts"]))
    return float(sum(ph.get("moe", [])) - sum(ph.get("route", [])))


def imbalance(values) -> float:
    v = np.asarray(values, dtype=float)
    m = v.mean() if v.size else 0.0
    return float(v.max() / m) if m > 0 else float("nan")


def worker_compute_imbalance(run: Run, num_ranks: int | None = None) -> np.ndarray:
    """Per-iteration max/mean of measured worker-compute time over the DP ranks."""
    G = num_ranks or run.num_ranks
    out = []
    for seq in run.busy_steps():
        d = run.steps[seq]
        if len(d) < G:
            continue
        out.append(imbalance([phase_sum(d[r], COMPUTE_PHASES) for r in range(G)]))
    return np.array([x for x in out if np.isfinite(x)])


def kv_imbalance(run: Run, num_ranks: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-iteration max/mean of inbound and outbound KV tokens over the ranks."""
    G = num_ranks or run.num_ranks
    kin, kout = [], []
    for seq in run.busy_steps():
        d = run.rank_reqs.get(seq, {})
        vin = [sum(q[3] for q in d.get(r, [])) for r in range(G)]
        vout = [sum(q[4] for q in d.get(r, [])) for r in range(G)]
        if sum(vin) > 0:
            kin.append(imbalance(vin))
        if sum(vout) > 0:
            kout.append(imbalance(vout))
    return np.array(kin), np.array(kout)
