"""BL1 (single-GPU origin baseline) per-request, per-phase metrics recorder.

The recorder is a process-singleton inside the GPU worker. When the env var
MONO_KERNEL_BL1_METRICS_PATH is set, it records per-request CUDA-event spans
across the three phases (vision / prefill / decode) plus an in-process NVML
sampler thread that captures GPU utilization and memory %. On request finish
it computes per-phase durations from the CUDA events, windows the NVML
samples to each phase, and appends one JSON line to the sidecar file.

When the env var is unset, every public method returns immediately so the
hook sites in the model runner / worker pay zero overhead.

Phase semantics (decided with stakeholder):
  - d_phase = elapsed time from the START of the FIRST forward step where the
    request is in this phase to the END of the LAST forward step where it is
    in this phase. Wall-clock span. No per-step apportionment when batches
    mix phases. d_decode includes sampling time only if the wrap site is
    placed after _sample; in BL1 we wrap before _sample so d_decode is
    forward-only.
  - gu_phase = mean of NVML utilization samples whose host timestamp falls in
    the (first_start_host, last_end_host) window. ko_phase = 100 - gu_phase.
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

# Sampler tick: how often we drain the driver's GPU-utilization sample
# buffer and read a memory-usage point. The driver itself produces
# utilization samples at ~10–100 ms cadence depending on the card; we
# tick faster so we never miss buffer entries and so gmu point samples
# land inside short phase windows.
_SAMPLER_TICK_S = 0.03
# Bracketing-sample fallback: when a phase is shorter than the driver's
# sample period (H20: dt = 200 ms fixed 5 Hz cadence), no sample
# timestamp falls strictly inside [t_start, t_end]. Instead of widening
# the window by some magic dt-bound, we pick a SINGLE sample: the next
# sample emitted at or after t_end. Each NVML utilization sample at
# timestamp t represents the GPU's averaged busy-fraction over
# [t - dt, t]; when t ≥ t_end and the phase is shorter than dt, that
# interval brackets the phase end-to-end, so the value is the best
# available estimate of GPU activity during the phase. No magic width
# constant needed — the rule is dt-agnostic.
# Bound on each in-memory deque (~10 minutes of headroom at ~33 Hz).
_NVML_BUFFER_MAX = 60000


@dataclass
class _PhaseBounds:
    first_start_event: torch.cuda.Event
    last_end_event: torch.cuda.Event


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

        # Per-request, per-phase bounds. Main-thread access only.
        self._phase_state: dict[str, dict[str, _PhaseBounds]] = {}

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
        while not self._sampler_stop.is_set():
            self._drain_gu_samples()
            self._poll_gmu_point()
            self._sampler_stop.wait(_SAMPLER_TICK_S)

    def _drain_gu_samples(self) -> None:
        """Pull all new GPU utilization samples from the driver's internal
        buffer since the last drain, convert their microsecond timestamps
        to host wall-clock, and push into the gu deque."""
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
                    val = float(s.sampleValue.uiVal)
                except Exception:
                    continue
                ts_host = offset + ts_us / 1e6
                self._gu_samples.append((ts_host, val))
                if ts_us > max_us:
                    max_us = ts_us
        self._gu_last_seen_us = max_us

    def _poll_gmu_point(self) -> None:
        if self._nvml_handle is None:
            return
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        except Exception:
            return
        try:
            gmu_pct = (float(mem.used) / float(mem.total)) * 100.0 if mem.total else 0.0
        except Exception:
            return
        with self._nvml_lock:
            self._gmu_samples.append((time.time(), gmu_pct))

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
            self._update_phase(req_id, phase, ctx.start_event, end_event)

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

        # Synchronize once over all events for this request.
        for bounds in phase_map.values():
            try:
                bounds.last_end_event.synchronize()
            except Exception:
                pass

        for phase in _PHASES:
            bounds = phase_map.get(phase)
            if bounds is None:
                record[f"d_{phase}"] = None
                record[f"gu_{phase}"] = None
                record[f"gmu_{phase}"] = None
                record[f"ko_{phase}"] = None
                continue
            try:
                d_ms = bounds.first_start_event.elapsed_time(bounds.last_end_event)
            except Exception:
                d_ms = None
            d_s = (d_ms / 1000.0) if d_ms is not None else None
            t_start = self._device_host_time(bounds.first_start_event)
            t_end = self._device_host_time(bounds.last_end_event)
            if t_start is not None and t_end is not None:
                gu, gmu = self._window_mean(t_start, t_end)
            else:
                gu, gmu = None, None
            ko = (100.0 - gu) if gu is not None else None
            record[f"d_{phase}"] = d_s
            record[f"gu_{phase}"] = gu
            record[f"gmu_{phase}"] = gmu
            record[f"ko_{phase}"] = ko

        try:
            self._sidecar.write(json.dumps(record) + "\n")
        except Exception:
            # Last-resort: never raise out of a worker hook on disk error.
            pass

    # ---- internals ---------------------------------------------------------

    def _update_phase(
        self, req_id: str, phase: str,
        start_event: torch.cuda.Event, end_event: torch.cuda.Event,
    ) -> None:
        per_req = self._phase_state.setdefault(req_id, {})
        bounds = per_req.get(phase)
        if bounds is None:
            per_req[phase] = _PhaseBounds(
                first_start_event=start_event,
                last_end_event=end_event,
            )
        else:
            bounds.last_end_event = end_event

    @staticmethod
    def _bracketing_value(samples, t_end: float) -> list[float]:
        """Return a single-element list with the value of the first
        sample emitted at or after ``t_end``. Its averaging interval
        [t-dt, t] brackets the phase when dt > phase length and is the
        best single-sample estimate. If no such sample exists yet (phase
        finalized before driver could emit), return [] — caller treats
        that as None."""
        nxt = next((v for t, v in samples if t >= t_end), None)
        return [nxt] if nxt is not None else []

    def _bracketing_diag(self, samples, t_end: float, label: str) -> None:
        """Diagnostic dump when bracketing fails: deque size, first/last
        sample timestamps, and how t_end compares. Writes one line to
        $MONO_KERNEL_BL1_DIAG_PATH if set, else silently returns."""
        path = os.environ.get("MONO_KERNEL_BL1_DIAG_PATH")
        if not path:
            return
        try:
            with self._nvml_lock:
                n = len(samples)
                first_t = samples[0][0] if n else None
                last_t = samples[-1][0] if n else None
            with open(path, "a", buffering=1, encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": label,
                    "t_end": t_end,
                    "deque_n": n,
                    "first_t": first_t,
                    "last_t": last_t,
                    "t_end_minus_last_t": (t_end - last_t) if last_t is not None else None,
                    "anchor_host_t": self._anchor_host_t,
                    "nvml_offset_s": self._nvml_to_host_offset_s,
                }) + "\n")
        except Exception:
            pass

    def _window_mean(self, t_start: float, t_end: float
                     ) -> tuple[Optional[float], Optional[float]]:
        """Estimate phase-window GPU utilization and memory %.

        Rule (per spec):
          - 0 samples in [t_start, t_end]: use the first sample emitted
            at or after t_end (its averaging interval brackets the phase
            when phase < driver dt).
          - 1 sample in window: use that value.
          - >=2 samples in window: arithmetic mean.
          - No bracketing sample available either: None.
        """
        if t_end <= t_start:
            return None, None
        with self._nvml_lock:
            gu_vals = [v for t, v in self._gu_samples if t_start <= t <= t_end]
            gmu_vals = [v for t, v in self._gmu_samples if t_start <= t <= t_end]
            if not gu_vals:
                gu_vals = self._bracketing_value(self._gu_samples, t_end)
            if not gmu_vals:
                gmu_vals = self._bracketing_value(self._gmu_samples, t_end)
        if not gu_vals:
            self._bracketing_diag(self._gu_samples, t_end, "gu_miss")
        if not gmu_vals:
            self._bracketing_diag(self._gmu_samples, t_end, "gmu_miss")
        gu = (sum(gu_vals) / len(gu_vals)) if gu_vals else None
        gmu = (sum(gmu_vals) / len(gmu_vals)) if gmu_vals else None
        return gu, gmu


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
