"""BL1 SM-level metrics recorder via DCGM (separate pass).

Same phase-tracking architecture as ``bl1.py`` but the sampler reads
``DCGM_FI_PROF_SM_ACTIVE`` and ``DCGM_FI_PROF_SM_OCCUPANCY`` from a
running ``nv-hostengine``.

Per-request, per-phase computation (relaxed Definition A — DCGM
aggregates across SMs, see project README):
  ``smu_phase = mean(SM_ACTIVE within phase) * 100``       (%)
  ``nsm_phase = total_SM_count * mean(SM_ACTIVE within phase)``
                                                         (equiv. SM count)

Phase boundaries map to host wall-clock via the same CUDA-event anchor
mechanism as ``bl1.py``: bounds reflect device-actual GPU execution
time, not host enqueue time.

The bracketing-sample fallback is identical to ``bl1.py`` (single
sample at first ``t >= t_end`` when window is empty).

Env vars:
  ``MONO_KERNEL_BL1_SM_METRICS_PATH``   sidecar path; unset → no-op.
  ``MONO_KERNEL_BL1_DCGM_HOST``         default ``127.0.0.1:5556``.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
import json
from dataclasses import dataclass, field
from typing import Optional

import torch

# DCGM Python bindings — installed at
# /usr/share/datacenter-gpu-manager-4/bindings/python3/ inside the
# mono_kernel container, then copied into the mamba env's site-packages.
try:
    import pydcgm  # type: ignore
    import dcgm_structs  # type: ignore  # noqa: F401
    import dcgm_fields  # type: ignore
    _DCGM_AVAILABLE = True
except Exception:
    pydcgm = None  # type: ignore
    dcgm_fields = None  # type: ignore
    _DCGM_AVAILABLE = False


PHASE_VISION = "vision"
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
_PHASES = (PHASE_VISION, PHASE_PREFILL, PHASE_DECODE)

# Sampler tick: how often to drain DCGM's buffer. The driver emits
# profiling samples on its own cadence (typically ~100 ms internally);
# faster ticks just guarantee we do not lose buffer entries.
_SAMPLER_TICK_S = 0.05
# Bound each in-memory deque (~10 minutes at the driver cadence).
_BUFFER_MAX = 60000


@dataclass
class _PhaseBounds:
    first_start_event: torch.cuda.Event
    last_end_event: torch.cuda.Event


@dataclass
class StepCtx:
    """Returned by step_begin / vision_begin; consumed by step_end /
    vision_end. Mirrors the StepCtx in bl1.py."""
    start_event: torch.cuda.Event
    req_phases: dict[str, str] = field(default_factory=dict)


class Bl1SmRecorder:
    def __init__(self, sidecar_path: str, device_index: int,
                 dcgm_host: str = "127.0.0.1:5556"):
        self._sidecar_path = sidecar_path
        self._device_index = device_index
        self._dcgm_host = dcgm_host

        # Phase state — main-thread access only.
        self._phase_state: dict[str, dict[str, _PhaseBounds]] = {}

        # Sample buffers. Producer = sampler thread, consumer =
        # finalize_request on the main thread.
        self._sm_active_samples: collections.deque[tuple[float, float]] = (
            collections.deque(maxlen=_BUFFER_MAX))
        self._sm_occ_samples: collections.deque[tuple[float, float]] = (
            collections.deque(maxlen=_BUFFER_MAX))
        self._lock = threading.Lock()

        # Sampler thread.
        self._sampler_stop = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None

        # DCGM handles (set on start_sampler).
        self._dcgm_handle = None
        self._dcgm_group = None
        self._dcgm_fg = None
        self._dcgm_dfvc = None  # accumulator for incremental drains

        # Driver-µs to host wall-clock offset (set on first non-empty
        # drain). Same pattern as bl1.py for NVML.
        self._dcgm_to_host_offset_s: Optional[float] = None

        # CUDA-event anchor for device-time mapping (mirror bl1.py).
        self._anchor_event: Optional[torch.cuda.Event] = None
        self._anchor_host_t: Optional[float] = None
        self._anchor_lock = threading.Lock()

        # Total SMs on this GPU — the multiplier for nsm.
        self._total_sm: int = 0

        # Sidecar (line-buffered append, single writer).
        os.makedirs(os.path.dirname(self._sidecar_path) or ".",
                    exist_ok=True)
        self._sidecar = open(self._sidecar_path, "a", buffering=1,
                             encoding="utf-8")

    # ---- lifecycle ----
    def start_sampler(self) -> None:
        if not _DCGM_AVAILABLE:
            print("[bl1_sm] pydcgm not importable; SM metrics disabled",
                  file=sys.stderr)
            return
        if self._sampler_thread is not None:
            return

        try:
            self._total_sm = torch.cuda.get_device_properties(
                self._device_index).multi_processor_count
        except Exception:
            self._total_sm = 0

        try:
            self._dcgm_handle = pydcgm.DcgmHandle(ipAddress=self._dcgm_host)
            system = pydcgm.DcgmSystem(self._dcgm_handle)
            self._dcgm_group = system.GetDefaultGroup()
            field_ids = [
                dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
                dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
            ]
            self._dcgm_fg = pydcgm.DcgmFieldGroup(
                self._dcgm_handle,
                name=f"bl1_sm_{os.getpid()}",
                fieldIds=field_ids,
            )
            self._dcgm_group.samples.WatchFields(
                self._dcgm_fg,
                updateFreq=100_000,   # 100 ms; driver decides actual emit rate
                maxKeepAge=600.0,
                maxKeepSamples=0,
            )
        except Exception as e:
            print(f"[bl1_sm] DCGM setup failed: {e}", file=sys.stderr)
            self._dcgm_handle = None
            return

        self._sampler_thread = threading.Thread(
            target=self._sampler_loop, name="bl1-sm-sampler", daemon=True)
        self._sampler_thread.start()

    def stop(self) -> None:
        self._sampler_stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
            self._sampler_thread = None
        try:
            self._sidecar.flush()
            self._sidecar.close()
        except Exception:
            pass

    # ---- sampler thread ----
    def _sampler_loop(self) -> None:
        while not self._sampler_stop.is_set():
            self._drain()
            self._sampler_stop.wait(_SAMPLER_TICK_S)

    def _drain(self) -> None:
        if self._dcgm_handle is None:
            return
        try:
            self._dcgm_dfvc = self._dcgm_group.samples.GetAllSinceLastCall(
                self._dcgm_dfvc, self._dcgm_fg)
        except Exception:
            return

        per_gpu = self._dcgm_dfvc.values.get(self._device_index, {})
        sm_active = per_gpu.get(dcgm_fields.DCGM_FI_PROF_SM_ACTIVE, [])
        sm_occ = per_gpu.get(dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY, [])
        if not sm_active and not sm_occ:
            return

        host_now = time.time()
        if self._dcgm_to_host_offset_s is None:
            latest_us = max([s.ts for s in sm_active] + [s.ts for s in sm_occ])
            if latest_us > 0:
                self._dcgm_to_host_offset_s = host_now - latest_us / 1e6
        offset = self._dcgm_to_host_offset_s
        if offset is None:
            return

        with self._lock:
            for s in sm_active:
                ts_host = offset + s.ts / 1e6
                self._sm_active_samples.append((ts_host, float(s.value)))
            for s in sm_occ:
                ts_host = offset + s.ts / 1e6
                self._sm_occ_samples.append((ts_host, float(s.value)))

        # Clear so the next GetAllSinceLastCall returns a fresh delta.
        self._dcgm_dfvc.values.clear()

    # ---- anchor + phase tracking (mirror bl1.py) ----
    def _ensure_anchor(self) -> None:
        if self._anchor_event is not None:
            return
        with self._anchor_lock:
            if self._anchor_event is not None:
                return
            try:
                torch.cuda.synchronize()
                anchor = torch.cuda.Event(enable_timing=True)
                anchor.record()
                anchor.synchronize()
                self._anchor_host_t = time.time()
                self._anchor_event = anchor
            except Exception:
                pass

    def _device_host_time(self, event: torch.cuda.Event) -> Optional[float]:
        if self._anchor_event is None or self._anchor_host_t is None:
            return None
        try:
            delta_ms = self._anchor_event.elapsed_time(event)
        except Exception:
            return None
        return self._anchor_host_t + delta_ms / 1000.0

    def step_begin(self, scheduler_output) -> StepCtx:
        self._ensure_anchor()
        req_phases: dict[str, str] = {}
        for nr in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_phases[nr.req_id] = PHASE_PREFILL
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            try:
                ids = list(cached.req_ids)
                num_output_tokens = list(cached.num_output_tokens)
                for rid, n_out in zip(ids, num_output_tokens):
                    req_phases[rid] = (PHASE_PREFILL if n_out == 0
                                        else PHASE_DECODE)
            except Exception:
                pass
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return StepCtx(start_event=start_event, req_phases=req_phases)

    def step_end(self, ctx: StepCtx) -> None:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        for rid, phase in ctx.req_phases.items():
            self._update_phase(rid, phase, ctx.start_event, end_event)

    def vision_begin(self, req_ids) -> StepCtx:
        self._ensure_anchor()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return StepCtx(
            start_event=start_event,
            req_phases={rid: PHASE_VISION for rid in req_ids})

    def vision_end(self, ctx: StepCtx) -> None:
        self.step_end(ctx)

    def _update_phase(self, req_id, phase, start_event, end_event):
        per_req = self._phase_state.setdefault(req_id, {})
        bounds = per_req.get(phase)
        if bounds is None:
            per_req[phase] = _PhaseBounds(
                first_start_event=start_event, last_end_event=end_event)
        else:
            bounds.last_end_event = end_event

    # ---- finalize ----
    def finalize_request(self, req_id: str) -> None:
        phase_map = self._phase_state.pop(req_id, None)
        if not phase_map:
            return

        for bounds in phase_map.values():
            try:
                bounds.last_end_event.synchronize()
            except Exception:
                pass

        record: dict[str, object] = {"vllm_req_id": req_id}
        for phase in _PHASES:
            bounds = phase_map.get(phase)
            if bounds is None:
                record[f"nsm_{phase}"] = None
                record[f"smu_{phase}"] = None
                record[f"sm_occ_{phase}"] = None
                continue

            t_start = self._device_host_time(bounds.first_start_event)
            t_end = self._device_host_time(bounds.last_end_event)
            sm_active = sm_occ = None
            if t_start is not None and t_end is not None:
                sm_active = self._window_value(
                    self._sm_active_samples, t_start, t_end)
                sm_occ = self._window_value(
                    self._sm_occ_samples, t_start, t_end)

            if sm_active is not None:
                record[f"smu_{phase}"] = sm_active * 100.0
                record[f"nsm_{phase}"] = (
                    self._total_sm * sm_active if self._total_sm else None)
            else:
                record[f"smu_{phase}"] = None
                record[f"nsm_{phase}"] = None
            record[f"sm_occ_{phase}"] = (
                sm_occ * 100.0 if sm_occ is not None else None)

        try:
            self._sidecar.write(json.dumps(record) + "\n")
        except Exception:
            pass

    @staticmethod
    def _bracketing_value(samples, t_end: float) -> list[float]:
        """First sample emitted at or after t_end, or [] if none."""
        nxt = next((v for t, v in samples if t >= t_end), None)
        return [nxt] if nxt is not None else []

    def _window_value(self, samples, t_start: float, t_end: float
                      ) -> Optional[float]:
        if t_end <= t_start:
            return None
        with self._lock:
            in_window = [v for t, v in samples if t_start <= t <= t_end]
            if not in_window:
                in_window = self._bracketing_value(samples, t_end)
        if not in_window:
            return None
        return sum(in_window) / len(in_window)


# Module-level singleton (mirror bl1.py pattern).
_RECORDER: Optional[Bl1SmRecorder] = None
_RECORDER_LOCK = threading.Lock()


def create_sm_recorder(device_index: int) -> Optional[Bl1SmRecorder]:
    sidecar_path = os.environ.get("MONO_KERNEL_BL1_SM_METRICS_PATH")
    if not sidecar_path:
        return None
    dcgm_host = os.environ.get("MONO_KERNEL_BL1_DCGM_HOST",
                               "127.0.0.1:5556")
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = Bl1SmRecorder(sidecar_path, device_index, dcgm_host)
        return _RECORDER


def get_sm_recorder() -> Optional[Bl1SmRecorder]:
    return _RECORDER


def stop_sm_recorder() -> None:
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is not None:
            _RECORDER.stop()
            _RECORDER = None
