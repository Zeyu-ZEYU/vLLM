"""BL1 (single-GPU origin baseline) per-request, per-phase metrics recorder.

The recorder is a process-singleton inside the GPU worker. When the env var
MONO_KERNEL_BL1_METRICS_PATH is set, it records, for every request and every
phase (vision / prefill / decode), the CUDA-event-bracketed span of EACH
individual execution forward (one entry per call to _execute_mm_encoder for
vision; one entry per _model_forward for prefill/decode). A background NVML
sampler thread captures GPU utilization (driver-side ring buffer) and GPU
memory usage (point query). On request finalize, per-phase metrics are
aggregated:

  d_{phase}            = sum of each execution forward's elapsed_time
  d_{phase}_span       = first execution start to last execution end
  n_executions_{phase} = number of execution forwards
  gu_{phase}, gmu_{phase} = pooled mean of all in-window samples (with
                            per-execution post-end-bracketing fallback)
                            across every execution of the phase. Empty pool
                            → null. Units are fractions in [0, 1].

When the env var is unset, every public method returns immediately so the
hook sites in the model runner / worker pay zero overhead.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

# Optional pynvml import: tolerate absence in test environments.
try:
    import pynvml  # type: ignore
    _PYNVML_AVAILABLE = True
except Exception:
    pynvml = None  # type: ignore
    _PYNVML_AVAILABLE = False

# Phase names.
PHASE_VISION = "vision"
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"

_PHASES = (PHASE_VISION, PHASE_PREFILL, PHASE_DECODE)

# Sampler tick: how often we run the inner sampler loop body. Memory is
# point-queried every tick (10 ms); GPU utilization is drained every
# `_UTIL_DRAIN_EVERY` ticks (effective ~30 ms; matches NVML driver
# capability — H20 emits util samples at fixed 5 Hz / 200 ms cadence, so
# 30 ms drain rate guarantees no buffered samples are missed).
_SAMPLER_TICK_S = 0.01
_UTIL_DRAIN_EVERY = 3
# Bracketing-sample fallback: when an execution forward is shorter than
# the underlying driver sample period (e.g., NVML util dt = 200 ms on H20;
# memory point query interval 10 ms here), no sample timestamp falls
# strictly inside [t_start, t_end] for that execution. Fallback rule (per
# spec): pick a SINGLE sample, the one emitted at or after t_end. Its
# implicit averaging interval [t - dt, t] brackets the short execution
# end-to-end, so the value is the best available single-sample estimate
# of GPU activity during the execution. If no such sample exists yet,
# this execution contributes nothing to the pool.
_NVML_BUFFER_MAX = 60000


@dataclass
class _ExecutionBounds:
    """One execution forward's CUDA-event pair for a (req_id, phase).

    A single physical (start_event, end_event) pair is shared by reference
    across all requests classified into this execution forward — that's
    correct (the device-side timestamps describe the same kernel range).
    """
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


@dataclass
class StepCtx:
    """Returned by step_begin / vision_begin; consumed by step_end / vision_end."""
    start_event: torch.cuda.Event
    # Map req_id -> phase name for the requests in this step.
    req_phases: dict[str, str] = field(default_factory=dict)


class Bl1Recorder:
    def __init__(self, sidecar_path: str, device_index: int):
        self._sidecar_path = sidecar_path
        self._device_index = device_index

        # Per-request, per-phase list of execution-forward bounds.
        # Main-thread access only.
        self._phase_state: dict[
            str, dict[str, list[_ExecutionBounds]]
        ] = {}

        # NVML sample ring buffers (separate for util and memory) + lock.
        # Producer = sampler thread, consumer = finalize_request on main
        # thread. Util samples come from nvmlDeviceGetSamples (driver's
        # internal buffer, ~10-100 ms native cadence). Memory % is a point
        # value sampled once per tick.
        self._gu_samples: collections.deque[tuple[float, float]] = (
            collections.deque(maxlen=_NVML_BUFFER_MAX)
        )
        self._gmu_samples: collections.deque[tuple[float, float]] = (
            collections.deque(maxlen=_NVML_BUFFER_MAX)
        )
        self._nvml_lock = threading.Lock()
        # Offset to convert driver-side sample microseconds to host wall-
        # clock. Initialized on first non-empty drain.
        self._nvml_to_host_offset_s: Optional[float] = None
        # Last seen NVML sample timestamp (microseconds), advanced each tick.
        self._gu_last_seen_us: int = 0

        # Sampler thread.
        self._sampler_stop = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._nvml_handle = None

        # Anchor for converting CUDA-event device time to host wall-clock.
        # Recorded + synchronized once on first phase begin so that
        # elapsed_time(anchor, event) gives a relative ms from a known host
        # wall-time. Used to align phase windows to NVML sample timestamps.
        self._anchor_event: Optional[torch.cuda.Event] = None
        self._anchor_host_t: Optional[float] = None
        self._anchor_lock = threading.Lock()

        # Sidecar file (line-buffered append). One writer, no lock needed.
        os.makedirs(os.path.dirname(self._sidecar_path) or ".", exist_ok=True)
        self._sidecar = open(self._sidecar_path, "a", buffering=1, encoding="utf-8")

    # ---- lifecycle ----------------------------------------------------------

    def start_sampler(self) -> None:
        if not _PYNVML_AVAILABLE:
            return
        if self._sampler_thread is not None:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except Exception:
            self._nvml_handle = None
            return
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop, name="bl1-nvml-sampler", daemon=True,
        )
        self._sampler_thread.start()

    def stop(self) -> None:
        self._sampler_stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
            self._sampler_thread = None
        if _PYNVML_AVAILABLE and self._nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None
        try:
            self._sidecar.flush()
            self._sidecar.close()
        except Exception:
            pass

    def _sampler_loop(self) -> None:
        i = 0
        while not self._sampler_stop.is_set():
            self._poll_gmu_point()
            if i % _UTIL_DRAIN_EVERY == 0:
                self._drain_gu_samples()
            i += 1
            self._sampler_stop.wait(_SAMPLER_TICK_S)

    def _drain_gu_samples(self) -> None:
        """Pull all new GPU utilization samples from the driver's internal
        buffer since the last drain, convert their microsecond timestamps
        to host wall-clock, and push into the gu deque (as fractions in
        [0, 1])."""
        if self._nvml_handle is None:
            return
        try:
            _, samples = pynvml.nvmlDeviceGetSamples(
                self._nvml_handle,
                pynvml.NVML_GPU_UTILIZATION_SAMPLES,
                self._gu_last_seen_us,
            )
        except Exception:
            return
        if not samples:
            return
        host_now = time.time()
        # On first drain, anchor driver-µs to host wall-clock using the
        # latest sample (smallest extrapolation error).
        if self._nvml_to_host_offset_s is None:
            latest_us = int(samples[-1].timeStamp)
            self._nvml_to_host_offset_s = host_now - latest_us / 1e6
        offset = self._nvml_to_host_offset_s
        max_us = self._gu_last_seen_us
        with self._nvml_lock:
            for s in samples:
                ts_us = int(s.timeStamp)
                try:
                    # NVML returns 0-100 percent; convert to fraction.
                    val = float(s.sampleValue.uiVal) / 100.0
                except Exception:
                    continue
                ts_host = offset + ts_us / 1e6
                self._gu_samples.append((ts_host, val))
                if ts_us > max_us:
                    max_us = ts_us
        self._gu_last_seen_us = max_us

    def _poll_gmu_point(self) -> None:
        """Single point query of GPU memory used / total (fraction in [0, 1])."""
        if self._nvml_handle is None:
            return
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        except Exception:
            return
        try:
            gmu = (float(mem.used) / float(mem.total)) if mem.total else 0.0
        except Exception:
            return
        with self._nvml_lock:
            self._gmu_samples.append((time.time(), gmu))

    # ---- step-level hooks ---------------------------------------------------

    def _ensure_anchor(self) -> None:
        """Establish a one-time CUDA event whose device time is aligned to
        host wall-clock. Subsequent phase events compare to this anchor via
        ``cuda.Event.elapsed_time`` to get device-side timestamps usable for
        NVML sample windowing."""
        if self._anchor_event is not None:
            return
        with self._anchor_lock:
            if self._anchor_event is not None:
                return
            try:
                # Drain any pending work so the anchor record-time on the
                # device closely matches host wall-clock.
                torch.cuda.synchronize()
                anchor = torch.cuda.Event(enable_timing=True)
                anchor.record()
                anchor.synchronize()
                self._anchor_host_t = time.time()
                self._anchor_event = anchor
            except Exception:
                # If anchor init fails (e.g. CUDA not ready), leave None;
                # finalize_request will then skip NVML windowing for this run.
                pass

    def _device_host_time(self, event: torch.cuda.Event) -> Optional[float]:
        """Host-wall-clock equivalent of when ``event`` was actually recorded
        on the device. Caller must have already synchronized ``event``."""
        if self._anchor_event is None or self._anchor_host_t is None:
            return None
        try:
            delta_ms = self._anchor_event.elapsed_time(event)
        except Exception:
            return None
        return self._anchor_host_t + delta_ms / 1000.0

    def step_begin(self, scheduler_output) -> StepCtx:
        """Classify each request in the step into prefill or decode, and
        record a step-start CUDA event."""
        self._ensure_anchor()
        req_phases: dict[str, str] = {}

        # New requests (this scheduling step is their first appearance) are
        # always in prefill.
        for nr in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_phases[nr.req_id] = PHASE_PREFILL

        # Cached (running) requests: classify via num_output_tokens.
        # is_context_phase() returns True iff num_output_tokens == 0 → prefill.
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            try:
                ids = list(cached.req_ids)
                num_output_tokens = list(cached.num_output_tokens)
                for rid, n_out in zip(ids, num_output_tokens):
                    req_phases[rid] = (
                        PHASE_PREFILL if n_out == 0 else PHASE_DECODE
                    )
            except Exception:
                # Fall back: don't classify if scheduler shape is unexpected.
                pass

        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return StepCtx(
            start_event=start_event,
            req_phases=req_phases,
        )

    def step_end(self, ctx: StepCtx) -> None:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        for req_id, phase in ctx.req_phases.items():
            self._append_execution(req_id, phase, ctx.start_event, end_event)

    # ---- vision-encoder hooks ----------------------------------------------

    def vision_begin(self, req_ids) -> StepCtx:
        self._ensure_anchor()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return StepCtx(
            start_event=start_event,
            req_phases={rid: PHASE_VISION for rid in req_ids},
        )

    def vision_end(self, ctx: StepCtx) -> None:
        # Same as step_end — vision is just another phase.
        self.step_end(ctx)

    # ---- per-request finalize ----------------------------------------------

    def finalize_request(self, req_id: str) -> None:
        phase_map = self._phase_state.pop(req_id, None)
        if not phase_map:
            return

        record: dict[str, object] = {"vllm_req_id": req_id}

        # Synchronize on the very last end_event of each phase once. This
        # guarantees all earlier events for this request have completed
        # (CUDA stream order), so subsequent elapsed_time() and
        # _device_host_time() calls are safe.
        for executions in phase_map.values():
            if not executions:
                continue
            try:
                executions[-1].end_event.synchronize()
            except Exception:
                pass

        for phase in _PHASES:
            executions = phase_map.get(phase) or []
            n_exec = len(executions)
            record[f"n_executions_{phase}"] = n_exec
            if n_exec == 0:
                record[f"d_{phase}"] = None
                record[f"d_{phase}_span"] = None
                record[f"gu_{phase}"] = None
                record[f"gmu_{phase}"] = None
                continue

            # d_phase = sum of each execution's elapsed_time.
            # d_phase_span = elapsed_time(first.start, last.end).
            d_sum_ms = 0.0
            d_span_ms = None
            try:
                for ex in executions:
                    d_sum_ms += ex.start_event.elapsed_time(ex.end_event)
                d_span_ms = executions[0].start_event.elapsed_time(
                    executions[-1].end_event
                )
            except Exception:
                d_sum_ms = None
                d_span_ms = None

            record[f"d_{phase}"] = (d_sum_ms / 1000.0) if d_sum_ms is not None else None
            record[f"d_{phase}_span"] = (
                d_span_ms / 1000.0 if d_span_ms is not None else None
            )

            # Build per-execution host time windows for sample pooling.
            windows: list[tuple[float, float]] = []
            for ex in executions:
                ts = self._device_host_time(ex.start_event)
                te = self._device_host_time(ex.end_event)
                if ts is not None and te is not None and te >= ts:
                    windows.append((ts, te))

            gu = self._aggregate_phase(self._gu_samples, windows)
            gmu = self._aggregate_phase(self._gmu_samples, windows)
            record[f"gu_{phase}"] = gu
            record[f"gmu_{phase}"] = gmu

        try:
            self._sidecar.write(json.dumps(record) + "\n")
        except Exception:
            # Last-resort: never raise out of a worker hook on disk error.
            pass

    # ---- internals ---------------------------------------------------------

    def _append_execution(
        self, req_id: str, phase: str,
        start_event: torch.cuda.Event, end_event: torch.cuda.Event,
    ) -> None:
        per_req = self._phase_state.setdefault(req_id, {})
        per_req.setdefault(phase, []).append(
            _ExecutionBounds(start_event=start_event, end_event=end_event)
        )

    def _aggregate_phase(
        self,
        ring: collections.deque,
        windows: list[tuple[float, float]],
    ) -> Optional[float]:
        """Pool samples across all execution windows for one phase.

        For each (t_start, t_end) window:
          - Collect samples whose timestamp t lies in [t_start, t_end].
          - If empty: try post-end fallback — the first sample with t > t_end
            (its averaging interval [t - dt, t] brackets the short execution).
          - If still empty: this execution contributes nothing.

        Phase value = arithmetic mean of the pooled samples, or None if empty.
        """
        if not windows:
            return None
        pool: list[float] = []
        with self._nvml_lock:
            # Snapshot to avoid holding lock through Python aggregation.
            snapshot = list(ring)
        if not snapshot:
            return None
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


# Module-level singleton slot (set by Worker.init_device, cleared by shutdown).
_RECORDER: Optional[Bl1Recorder] = None
_RECORDER_LOCK = threading.Lock()


def create_recorder(device_index: int) -> Optional[Bl1Recorder]:
    """Create the singleton recorder if MONO_KERNEL_BL1_METRICS_PATH is set."""
    sidecar_path = os.environ.get("MONO_KERNEL_BL1_METRICS_PATH")
    if not sidecar_path:
        return None
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = Bl1Recorder(sidecar_path, device_index)
        return _RECORDER


def get_recorder() -> Optional[Bl1Recorder]:
    return _RECORDER


def stop_recorder() -> None:
    global _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is not None:
            _RECORDER.stop()
            _RECORDER = None
