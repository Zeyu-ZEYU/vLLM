"""BL1 SM-level metrics recorder via DCGM (separate pass).

Same per-execution-forward phase tracking as ``bl1.py`` but the sampler
reads ``DCGM_FI_PROF_SM_ACTIVE`` and ``DCGM_FI_PROF_SM_OCCUPANCY`` from a
running ``nv-hostengine``.

Per-request, per-phase outputs (fractions in [0, 1] for smu / sm_occ;
counts in [0, total_sm] for nsm; ko + smu == 1 by construction):
  ``smu_phase``  = mean of per-execution-pooled SM_ACTIVE samples.
  ``ko_phase``   = 1 - smu_phase.
  ``nsm_phase``  = total_sm * smu_phase.
  ``sm_occ_phase`` = mean of per-execution-pooled SM_OCCUPANCY samples.

Phase boundaries map to host wall-clock via the same CUDA-event anchor
mechanism as ``bl1.py``: bounds reflect device-actual GPU execution
time, not host enqueue time.

The bracketing fallback is per-execution (single sample at first
``t > t_end`` when an execution's window is empty), and the phase value
is the pooled mean across all of R's execution forwards in that phase.

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

# Sampler tick: drain DCGM's buffer every 10 ms. Coupled with
# updateFreq=10_000 µs (10 ms emit cadence) so we observe each new
# sample within one tick of its emission.
_SAMPLER_TICK_S = 0.01
# DCGM updateFreq (µs). 10 ms; PerfWorks honors this down to 10 ms on H20.
_DCGM_UPDATE_FREQ_US = 10_000
# Bound each in-memory deque (~10 minutes at 100 Hz cadence).
_BUFFER_MAX = 60000


@dataclass
class _ExecutionBounds:
    """One execution forward's CUDA-event pair for a (req_id, phase)."""
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


@dataclass
class StepCtx:
    """Returned by step_begin / vision_begin; consumed by step_end /
    vision_end."""
    start_event: torch.cuda.Event
    req_phases: dict[str, str] = field(default_factory=dict)


class Bl1SmRecorder:
    def __init__(self, sidecar_path: str, device_index: int,
                 dcgm_host: str = "127.0.0.1:5556"):
        self._sidecar_path = sidecar_path
        self._device_index = device_index
        self._dcgm_host = dcgm_host

        # Per-request, per-phase list of execution-forward bounds.
        self._phase_state: dict[
            str, dict[str, list[_ExecutionBounds]]
        ] = {}

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
                updateFreq=_DCGM_UPDATE_FREQ_US,
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
        # Defense in depth: swallow any per-tick exception so the daemon
        # thread can never die mid-experiment and never spam the server
        # log with tracebacks (which the run_bl1.sh monitor would
        # mistakenly flag as fatal).
        while not self._sampler_stop.is_set():
            try:
                self._drain()
            except Exception:
                pass
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
                if s.value is None:
                    continue  # DCGM emits None for stale/uninitialized
                try:
                    val = float(s.value)
                except (TypeError, ValueError):
                    continue
                # DCGM blank/error sentinels for FP64 fields are
                # ~1.4e14 (DCGM_FP64_BLANK) and similarly for
                # NOT_FOUND/NOT_SUPPORTED/NOT_PERMISSIONED. Real
                # SM_ACTIVE / SM_OCCUPANCY values are in [0, 1].
                # Reject anything outside [0, 1] as a sentinel leak.
                if not (0.0 <= val <= 1.0):
                    continue
                ts_host = offset + s.ts / 1e6
                self._sm_active_samples.append((ts_host, val))
            for s in sm_occ:
                if s.value is None:
                    continue
                try:
                    val = float(s.value)
                except (TypeError, ValueError):
                    continue
                if not (0.0 <= val <= 1.0):
                    continue
                ts_host = offset + s.ts / 1e6
                self._sm_occ_samples.append((ts_host, val))

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
            self._append_execution(rid, phase, ctx.start_event, end_event)

    def vision_begin(self, req_ids) -> StepCtx:
        self._ensure_anchor()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return StepCtx(
            start_event=start_event,
            req_phases={rid: PHASE_VISION for rid in req_ids})

    def vision_end(self, ctx: StepCtx) -> None:
        self.step_end(ctx)

    def _append_execution(self, req_id, phase, start_event, end_event):
        per_req = self._phase_state.setdefault(req_id, {})
        per_req.setdefault(phase, []).append(
            _ExecutionBounds(start_event=start_event, end_event=end_event)
        )

    # ---- finalize ----
    def finalize_request(self, req_id: str) -> None:
        phase_map = self._phase_state.pop(req_id, None)
        if not phase_map:
            return

        # Synchronize the last end_event of each phase. CUDA stream order
        # guarantees earlier events for this request are complete.
        for executions in phase_map.values():
            if not executions:
                continue
            try:
                executions[-1].end_event.synchronize()
            except Exception:
                pass

        record: dict[str, object] = {"vllm_req_id": req_id}
        for phase in _PHASES:
            executions = phase_map.get(phase) or []
            n_exec = len(executions)
            record[f"n_executions_{phase}"] = n_exec
            if n_exec == 0:
                record[f"d_{phase}"] = None
                record[f"d_{phase}_span"] = None
                record[f"smu_{phase}"] = None
                record[f"sm_occ_{phase}"] = None
                record[f"nsm_{phase}"] = None
                record[f"ko_{phase}"] = None
                continue

            # d_phase = sum of execution durations; d_phase_span = first
            # start to last end.
            d_sum_ms = 0.0
            d_span_ms = None
            try:
                for ex in executions:
                    d_sum_ms += ex.start_event.elapsed_time(ex.end_event)
                d_span_ms = executions[0].start_event.elapsed_time(
                    executions[-1].end_event)
            except Exception:
                d_sum_ms = None
                d_span_ms = None
            record[f"d_{phase}"] = (
                d_sum_ms / 1000.0 if d_sum_ms is not None else None)
            record[f"d_{phase}_span"] = (
                d_span_ms / 1000.0 if d_span_ms is not None else None)

            # Build per-execution host-time windows for sample pooling.
            windows: list[tuple[float, float]] = []
            for ex in executions:
                ts = self._device_host_time(ex.start_event)
                te = self._device_host_time(ex.end_event)
                if ts is not None and te is not None and te >= ts:
                    windows.append((ts, te))

            smu = self._aggregate_phase(self._sm_active_samples, windows)
            sm_occ = self._aggregate_phase(self._sm_occ_samples, windows)

            record[f"smu_{phase}"] = smu  # fraction in [0, 1] or None
            record[f"sm_occ_{phase}"] = sm_occ
            if smu is not None:
                record[f"nsm_{phase}"] = (
                    self._total_sm * smu if self._total_sm else None)
                # ko = 1 - smu; smu + ko == 1 by construction.
                record[f"ko_{phase}"] = 1.0 - smu
            else:
                record[f"nsm_{phase}"] = None
                record[f"ko_{phase}"] = None

        try:
            self._sidecar.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _aggregate_phase(
        self,
        ring: collections.deque,
        windows: list[tuple[float, float]],
    ) -> Optional[float]:
        """Pool samples across all execution windows for one phase.

        Per execution window:
          - Collect samples whose timestamp lies in [t_start, t_end].
          - If empty: post-end fallback — first sample with t > t_end.
          - If still empty: this execution contributes nothing.

        Phase value = arithmetic mean of pooled samples (fraction in
        [0, 1]) or None if pool is empty.
        """
        if not windows:
            return None
        with self._lock:
            snapshot = list(ring)
        if not snapshot:
            return None
        pool: list[float] = []
        for t_start, t_end in windows:
            in_window = [v for t, v in snapshot if t_start <= t <= t_end]
            if not in_window:
                post = next((v for t, v in snapshot if t > t_end), None)
                if post is not None:
                    in_window = [post]
            if in_window:
                pool.extend(in_window)
        if not pool:
            return None
        return sum(pool) / len(pool)


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
