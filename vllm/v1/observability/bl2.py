"""BL2 (vision-text disaggregation) per-event embedding-transfer recorder.

Companion to bl1.py. While bl1 captures per-request, per-phase durations and
NVML utilization, bl2 captures the cross-node leg only: each time a vision
embedding (keyed by mm_hash) crosses the wire, both producer and consumer
emit one event into a sidecar JSONL.

Sidecar schema (one JSON object per line):

  Producer:
    {"mm_hash": "<hex>", "req_ids": [...], "role": "producer",
     "t_event": <float host-time>, "n_bytes": <int>}

  Consumer:
    {"mm_hash": "<hex>", "req_ids": [...], "role": "consumer",
     "t_event": <float host-time>,
     "d_wait":  <float seconds>,    # single-clock perf_counter
     "d_pull":  <float seconds>,    # single-clock perf_counter
     "n_bytes": <int>}

Cross-node merge (mk_scripts/merge_metrics.py) computes three per-request
metrics from these events:
  - d_vemb_total: max over mm_hashes of (consumer.t_event - producer.t_event).
    Cross-node, depends on NTP sync. Reflects user-perceived end-to-end
    latency for the slowest image.
  - d_vemb_wait:  sum over mm_hashes of consumer.d_wait. Single-clock; the
    consumer side processes mm_hashes serially in start_load_caches, so
    summing gives the total wall-clock the request was blocked on PUSH-arrival
    waits.
  - d_vemb_pull:  sum over mm_hashes of consumer.d_pull. Single-clock; same
    serial reasoning, gives total wall-clock spent in NIXL READ + scratch
    copy-out for this request.

The d_wait / d_pull split exists because the cross-node `time.time()` diff
that defines d_vemb_total is bounded below by NTP precision (~ms on a LAN,
worse during clock corrections) and so cannot reliably resolve the pure
data-plane RDMA transfer time (~ms for a typical 66 MiB encoded tensor).
The two single-clock perf_counter metrics decompose the consumer-visible
latency into "waiting for producer's PUSH notif" and "executing NIXL READ
+ buffer copy-out" without any cross-node drift.

Activation: env var MONO_KERNEL_BL2_VEMB_PATH must be set to a writable path.
When unset, the singleton stays None and every public method is a no-op.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional


class Bl2VembRecorder:
    """Append-only sidecar writer for vision-embedding transfer events."""

    def __init__(self, sidecar_path: str, role: str):
        if role not in ("producer", "consumer"):
            raise ValueError(f"role must be 'producer' or 'consumer', got {role!r}")
        self._sidecar_path = sidecar_path
        self._role = role
        os.makedirs(os.path.dirname(sidecar_path) or ".", exist_ok=True)
        self._sidecar = open(sidecar_path, "a", buffering=1, encoding="utf-8")
        self._write_lock = threading.Lock()

    def stop(self) -> None:
        try:
            with self._write_lock:
                self._sidecar.flush()
                self._sidecar.close()
        except Exception:
            pass

    # ---- event emission ----------------------------------------------------

    def record_send_done(
        self, mm_hash: str, req_ids, n_bytes: int
    ) -> None:
        """Producer side: NIXL push for this mm_hash has finished (DMA fence
        reached + PUSH notif emitted). Captures host wall-clock at the moment
        of return — used by the merger as the cross-node anchor for
        d_vemb_total.
        """
        rec = {
            "mm_hash": str(mm_hash),
            "req_ids": list(req_ids) if req_ids else [],
            "role": self._role,
            "t_event": time.time(),
            "n_bytes": int(n_bytes),
        }
        self._write(rec)

    def record_recv_done(
        self, mm_hash: str, req_ids, n_bytes: int,
        d_wait: float, d_pull: float,
    ) -> None:
        """Consumer side: NIXL pull for this mm_hash has finished and the
        encoder cache slot has been populated locally. Records:

        - t_event (host wall-clock): cross-node anchor for d_vemb_total.
        - d_wait (seconds, perf_counter): time blocked in `wait_for_notif`
          for this mm_hash's PUSH to arrive (single-clock, no NTP).
        - d_pull (seconds, perf_counter): time spent inside `consumer_pull`
          (NIXL READ + scratch -> tensor copy) for this mm_hash (single-clock).
        """
        rec = {
            "mm_hash": str(mm_hash),
            "req_ids": list(req_ids) if req_ids else [],
            "role": self._role,
            "t_event": time.time(),
            "d_wait": float(d_wait),
            "d_pull": float(d_pull),
            "n_bytes": int(n_bytes),
        }
        self._write(rec)

    def _write(self, rec: dict) -> None:
        try:
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            with self._write_lock:
                self._sidecar.write(line)
        except Exception:
            # Never raise out of a worker hook on disk error.
            pass


# ---- module-level singleton -------------------------------------------------

_RECORDER: Optional[Bl2VembRecorder] = None
_RECORDER_LOCK = threading.Lock()


def create_recorder(role: str) -> Optional[Bl2VembRecorder]:
    """Create the singleton if MONO_KERNEL_BL2_VEMB_PATH is set, else None.

    `role` is 'producer' on the vision instance, 'consumer' on the text
    instance; the orchestrator decides which by setting the env var on each
    side.
    """
    sidecar_path = os.environ.get("MONO_KERNEL_BL2_VEMB_PATH")
    if not sidecar_path:
        return None
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = Bl2VembRecorder(sidecar_path, role)
        return _RECORDER


def get_recorder() -> Optional[Bl2VembRecorder]:
    return _RECORDER


def stop_recorder() -> None:
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is not None:
            _RECORDER.stop()
            _RECORDER = None
