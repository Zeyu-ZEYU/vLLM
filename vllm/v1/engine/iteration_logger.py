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
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from vllm.v1.core.sched.output import SchedulerOutput


class IterationLogger:
    """Writes per-iteration metadata to ``iterations.jsonl`` and
    per-request phase mapping to ``requests.jsonl`` on shutdown.
    """

    def __init__(self, log_dir: str):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._iter_file = open(self._log_dir / "iterations.jsonl", "w")
        self._iteration_index = 0

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

        yield  # model execution happens here

        elapsed_ms = (time.monotonic() - ts_mono) * 1000.0
        elapsed_s = elapsed_ms / 1000.0
        num_reqs = len(
            scheduler_output.num_scheduled_tokens
        )
        rps = round(num_reqs / elapsed_s, 3) if elapsed_s > 0 else 0.0

        # --- Write iteration record ---
        record = {
            "iter": idx,
            "ts_mono": round(ts_mono, 6),
            "ts_wall": round(ts_wall, 6),
            "elapsed_ms": round(elapsed_ms, 3),
            "step_latency_ms": round(elapsed_ms, 3),
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
        }
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
