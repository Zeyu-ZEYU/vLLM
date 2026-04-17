# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight per-iteration logger that writes JSONL files with request
phase information (vision encoder / prefill / decode) for post-hoc
profiling analysis.

Enabled by setting ``VLLM_LOG_ITERATIONS=1``. Output directory is
controlled by ``VLLM_ITERATION_LOG_DIR`` (defaults to CWD).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from vllm.v1.core.sched.output import SchedulerOutput


class _NvmlSampler:
    """Background thread that samples GPU utilization + memory via pynvml.

    Samples are (ts_mono, gpu_util_pct, mem_util_pct, mem_used_bytes).
    The sampler runs for the life of the process; shutdown stops the
    thread cleanly.
    """

    def __init__(self, interval_s: float = 0.01):
        self._interval_s = interval_s
        self._samples: list[tuple[float, float, float, int]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._nvml = None

    def start(self):
        try:
            import pynvml
        except ImportError:
            return False
        try:
            pynvml.nvmlInit()
        except Exception:
            return False
        # CUDA_VISIBLE_DEVICES masks physical GPUs; the process sees
        # device index 0 as the FIRST visible one. But pynvml uses the
        # physical NVIDIA index — we need to map from CUDA's view back
        # to NVML's via the CUDA_VISIBLE_DEVICES env var.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # Take first entry (our process only uses one GPU).
        first = cvd.split(",")[0].strip()
        try:
            nvml_idx = int(first) if first else 0
        except ValueError:
            nvml_idx = 0
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
        except Exception:
            return False
        self._nvml = pynvml
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self):
        while not self._stop.is_set():
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                ts = time.monotonic()
                with self._lock:
                    self._samples.append(
                        (ts, float(util.gpu), float(util.memory), int(mem.used))
                    )
            except Exception:
                pass
            self._stop.wait(self._interval_s)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def aggregate_window(
        self, start_ts: float, end_ts: float
    ) -> dict | None:
        """Average samples in [start_ts, end_ts]. Returns None if no samples."""
        with self._lock:
            snap = list(self._samples)
        in_range = [s for s in snap if start_ts <= s[0] <= end_ts]
        if not in_range:
            return None
        n = len(in_range)
        return {
            "nvml_samples": n,
            "nvml_gpu_util_pct_mean": round(sum(s[1] for s in in_range) / n, 2),
            "nvml_gpu_util_pct_max": round(max(s[1] for s in in_range), 2),
            "nvml_mem_util_pct_mean": round(sum(s[2] for s in in_range) / n, 2),
            "nvml_mem_used_MiB_mean": round(
                sum(s[3] for s in in_range) / n / (1024**2), 2
            ),
            "nvml_mem_used_MiB_max": round(
                max(s[3] for s in in_range) / (1024**2), 2
            ),
        }


class IterationLogger:
    """Writes per-iteration metadata to ``iterations.jsonl`` and
    per-request phase mapping to ``requests.jsonl`` on shutdown.
    """

    def __init__(self, log_dir: str):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._iter_file = open(self._log_dir / "iterations.jsonl", "w")
        self._iteration_index = 0
        self._prev_ts_mono: float | None = None

        # Start NVML sampler for GPU util/memory.
        self._nvml_sampler = _NvmlSampler(interval_s=0.01)
        if self._nvml_sampler.start():
            import logging

            logging.getLogger(__name__).info(
                "[IterationLogger] NVML sampler started"
            )
        else:
            self._nvml_sampler = None

        # request_id -> {"encoder_iters": [], "prefill_iters": [],
        #                 "decode_iters": [], "first_iter": N}
        self._request_map: dict[str, dict] = defaultdict(
            lambda: {
                "encoder_iters": [],
                "prefill_iters": [],
                "decode_iters": [],
                "first_iter": None,
                "last_iter": None,
            }
        )

    def close(self):
        """Write requests.jsonl and close file handles."""
        if self._nvml_sampler is not None:
            self._nvml_sampler.stop()
        self._iter_file.close()
        self._write_requests()

    def _write_requests(self):
        req_file = self._log_dir / "requests.jsonl"
        with open(req_file, "w") as f:
            for internal_id, info in self._request_map.items():
                external_id = internal_id.rsplit("-", 1)[0]
                record = {
                    "internal_id": internal_id,
                    "external_id": external_id,
                    **info,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _mem_snapshot_before(self) -> tuple[float | None, float | None]:
        """Synchronize GPU, then capture memory baseline and reset peak."""
        try:
            import torch

            device = torch.device("cuda:0")
            # Synchronize so that any pending GPU memory ops from the
            # previous iteration are reflected before we read/reset stats.
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
            return round(allocated / (1024**2), 2), None
        except Exception:
            return None, None

    def _mem_snapshot_after(self) -> tuple[float | None, float | None]:
        """Synchronize GPU, then capture memory and read peak."""
        try:
            import torch

            device = torch.device("cuda:0")
            # Synchronize so that all GPU memory allocations/frees from
            # this iteration are finalized before we read stats.
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            peak = torch.cuda.max_memory_allocated(device)
            return (
                round(allocated / (1024**2), 2),
                round(peak / (1024**2), 2),
            )
        except Exception:
            return None, None

    @contextmanager
    def log_iteration(self, scheduler_output: SchedulerOutput):
        """Context manager that records one iteration's metadata.

        Yields control for the actual model execution, then writes the
        record with elapsed time.
        """
        idx = self._iteration_index

        # --- Classify requests ---
        # Which requests have vision encoder inputs this iteration.
        encoder_req_ids = list(
            scheduler_output.scheduled_encoder_inputs.keys()
        )

        # Prefill vs decode: same logic as compute_iteration_details().
        new_req_ids = {
            r.req_id for r in scheduler_output.scheduled_new_reqs
        }
        prefill_req_ids: list[str] = []
        decode_req_ids: list[str] = []
        num_prefill_tokens = 0
        num_decode_tokens = 0

        for req_id, num_tokens in (
            scheduler_output.num_scheduled_tokens.items()
        ):
            is_prefill = (
                req_id in new_req_ids
                or scheduler_output.scheduled_cached_reqs.is_context_phase(
                    req_id
                )
            )
            if is_prefill:
                prefill_req_ids.append(req_id)
                num_prefill_tokens += num_tokens
            else:
                decode_req_ids.append(req_id)
                num_decode_tokens += num_tokens

        ts_mono = time.monotonic()
        ts_wall = time.time()

        # Measure GPU memory before/after model execution.
        mem_before, mem_peak = self._mem_snapshot_before()

        yield  # model execution happens here

        mem_after, mem_peak_after = self._mem_snapshot_after()

        # In async scheduling mode, the GPU work is submitted before
        # this context manager, so yield-based timing would only measure
        # the future.result() wait (microseconds, not real execution).
        # Instead, compute step latency from inter-iteration intervals.
        if self._prev_ts_mono is not None:
            step_latency_ms = (ts_mono - self._prev_ts_mono) * 1000.0
        else:
            step_latency_ms = 0.0  # first iteration, no prior reference
        self._prev_ts_mono = ts_mono

        step_latency_s = step_latency_ms / 1000.0
        num_reqs = len(
            scheduler_output.num_scheduled_tokens
        )
        rps = (
            round(num_reqs / step_latency_s, 3)
            if step_latency_s > 0
            else 0.0
        )

        # --- Write iteration record ---
        record = {
            "iter": idx,
            "ts_mono": round(ts_mono, 6),
            "ts_wall": round(ts_wall, 6),
            "step_latency_ms": round(step_latency_ms, 3),
            "num_reqs": num_reqs,
            "step_rps": rps,
            "has_encoder": len(encoder_req_ids) > 0,
            "encoder_req_ids": encoder_req_ids,
            "prefill_req_ids": prefill_req_ids,
            "decode_req_ids": decode_req_ids,
            "num_prefill_reqs": len(prefill_req_ids),
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_reqs": len(decode_req_ids),
            "num_decode_tokens": num_decode_tokens,
            "total_tokens": (
                scheduler_output.total_num_scheduled_tokens
            ),
            "gpu_mem_allocated_MiB": mem_before,
            "gpu_mem_peak_MiB": mem_peak_after,
            "gpu_mem_delta_MiB": (
                round(mem_peak_after - mem_before, 2)
                if mem_before is not None and mem_peak_after is not None
                else None
            ),
        }

        # Aggregate NVML samples that fell within this iteration window.
        # This gives per-iteration GPU utilization % and memory %.
        if self._nvml_sampler is not None and self._prev_ts_mono is not None:
            # ts_mono is the END of this iteration (inter-iteration delta).
            # The iteration ran from (ts_mono - step_latency_s) to ts_mono.
            window_start = ts_mono - step_latency_s
            nvml_agg = self._nvml_sampler.aggregate_window(
                window_start, ts_mono
            )
            if nvml_agg:
                record.update(nvml_agg)
        self._iter_file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        self._iter_file.flush()

        # --- Update request map ---
        for req_id in encoder_req_ids:
            entry = self._request_map[req_id]
            entry["encoder_iters"].append(idx)
            if entry["first_iter"] is None:
                entry["first_iter"] = idx
            entry["last_iter"] = idx

        for req_id in prefill_req_ids:
            entry = self._request_map[req_id]
            entry["prefill_iters"].append(idx)
            if entry["first_iter"] is None:
                entry["first_iter"] = idx
            entry["last_iter"] = idx

        for req_id in decode_req_ids:
            entry = self._request_map[req_id]
            entry["decode_iters"].append(idx)
            if entry["first_iter"] is None:
                entry["first_iter"] = idx
            entry["last_iter"] = idx

        self._iteration_index += 1


# --- Module-level factory (used by EngineCore) ---

_logger_instance: IterationLogger | None = None


def get_iteration_logger() -> IterationLogger | None:
    """Return the global IterationLogger if enabled, else None."""
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    from vllm import envs

    if not envs.VLLM_LOG_ITERATIONS:
        return None

    log_dir = envs.VLLM_ITERATION_LOG_DIR
    _logger_instance = IterationLogger(log_dir)
    return _logger_instance


def shutdown_iteration_logger():
    """Flush and close the logger if it was initialized."""
    global _logger_instance
    if _logger_instance is not None:
        _logger_instance.close()
        _logger_instance = None
