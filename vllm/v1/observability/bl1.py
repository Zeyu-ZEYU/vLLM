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

# NVML sampling rate.
_NVML_HZ = 50.0
_NVML_PERIOD_S = 1.0 / _NVML_HZ
# Bound the in-memory sample buffer (~10 minutes at 50 Hz).
_NVML_BUFFER_MAX = 30000


@dataclass
class _PhaseBounds:
    first_start_event: torch.cuda.Event
    first_start_host: float
    last_end_event: torch.cuda.Event
    last_end_host: float


@dataclass
class StepCtx:
    """Returned by step_begin / vision_begin; consumed by step_end / vision_end."""
    start_event: torch.cuda.Event
    start_host: float
    # Map req_id -> phase name for the requests in this step.
    req_phases: dict[str, str] = field(default_factory=dict)


class Bl1Recorder:
    def __init__(self, sidecar_path: str, device_index: int):
        self._sidecar_path = sidecar_path
        self._device_index = device_index

        # Per-request, per-phase bounds. Main-thread access only.
        self._phase_state: dict[str, dict[str, _PhaseBounds]] = {}

        # NVML sample ring buffer + lock. Producer = sampler thread,
        # consumer = finalize_request on main thread.
        self._nvml_samples: collections.deque[tuple[float, float, float]] = (
            collections.deque(maxlen=_NVML_BUFFER_MAX)
        )
        self._nvml_lock = threading.Lock()

        # Sampler thread.
        self._sampler_stop = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._nvml_handle = None

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
            t = time.time()
            gu, gmu = self._read_nvml()
            if gu is not None and gmu is not None:
                self._nvml_samples.append((t, gu, gmu))
            # Sleep until next tick; if we lag, just continue immediately.
            self._sampler_stop.wait(_NVML_PERIOD_S)

    def _read_nvml(self) -> tuple[Optional[float], Optional[float]]:
        if self._nvml_handle is None:
            return None, None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            gu_pct = float(util.gpu)
            gmu_pct = (float(mem.used) / float(mem.total)) * 100.0 if mem.total else 0.0
            return gu_pct, gmu_pct
        except Exception:
            return None, None

    # ---- step-level hooks ---------------------------------------------------

    def step_begin(self, scheduler_output) -> StepCtx:
        """Classify each request in the step into prefill or decode, and
        record a step-start CUDA event + host timestamp."""
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
            start_host=time.time(),
            req_phases=req_phases,
        )

    def step_end(self, ctx: StepCtx) -> None:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_host = time.time()
        for req_id, phase in ctx.req_phases.items():
            self._update_phase(req_id, phase, ctx.start_event, ctx.start_host,
                               end_event, end_host)

    # ---- vision-encoder hooks ----------------------------------------------

    def vision_begin(self, req_ids) -> StepCtx:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return StepCtx(
            start_event=start_event,
            start_host=time.time(),
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
            gu, gmu = self._window_mean(
                bounds.first_start_host, bounds.last_end_host
            )
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
        start_event: torch.cuda.Event, start_host: float,
        end_event: torch.cuda.Event, end_host: float,
    ) -> None:
        per_req = self._phase_state.setdefault(req_id, {})
        bounds = per_req.get(phase)
        if bounds is None:
            per_req[phase] = _PhaseBounds(
                first_start_event=start_event, first_start_host=start_host,
                last_end_event=end_event, last_end_host=end_host,
            )
        else:
            bounds.last_end_event = end_event
            bounds.last_end_host = end_host

    def _window_mean(self, t_start: float, t_end: float
                     ) -> tuple[Optional[float], Optional[float]]:
        if t_end <= t_start:
            return None, None
        gu_sum = 0.0
        gmu_sum = 0.0
        n = 0
        with self._nvml_lock:
            # Snapshot the relevant samples (deque is short — O(N) is fine).
            for t, gu, gmu in self._nvml_samples:
                if t_start <= t <= t_end:
                    gu_sum += gu
                    gmu_sum += gmu
                    n += 1
        if n == 0:
            return None, None
        return gu_sum / n, gmu_sum / n


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
