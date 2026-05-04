"""BL2 (vision-text disaggregation) per-event embedding-transfer recorder.

Companion to bl1.py. While bl1 captures per-request, per-phase durations and
NVML utilization, bl2 captures the cross-node leg only: each time a vision
embedding (keyed by mm_hash) crosses the wire, both producer and consumer
emit one event into a sidecar JSONL.

Sidecar schema (one JSON object per line):
    {"mm_hash": "<hex>", "req_ids": ["<vllm_req_id>", ...],
     "role": "producer" | "consumer",
     "t_event": <float-host-time>, "n_bytes": <int>}

Cross-node merge (in mk_scripts/merge_metrics.py) reconciles producer's
t_event with consumer's t_event for the same mm_hash to compute
d_vemb_transfer = t_event_consumer - t_event_producer. Cross-node clocks are
assumed to be NTP-synced; sub-ms drift is acceptable for ms-scale transfers.

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
        reached). Captures host wall-clock at the moment of return."""
        self._emit(mm_hash, req_ids, n_bytes)

    def record_recv_done(
        self, mm_hash: str, req_ids, n_bytes: int
    ) -> None:
        """Consumer side: NIXL pull for this mm_hash has finished and the
        encoder cache slot has been populated locally."""
        self._emit(mm_hash, req_ids, n_bytes)

    def _emit(self, mm_hash: str, req_ids, n_bytes: int) -> None:
        rec = {
            "mm_hash": str(mm_hash),
            "req_ids": list(req_ids) if req_ids else [],
            "role": self._role,
            "t_event": time.time(),
            "n_bytes": int(n_bytes),
        }
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
