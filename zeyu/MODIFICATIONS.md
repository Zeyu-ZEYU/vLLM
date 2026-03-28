# vLLM Source Modifications for Per-Request Latency and Per-Iteration Profiling

This document describes all modifications made to vLLM source files.

## Overview

Two categories of modifications:

1. **Vision encoder timing always-on** — Remove the observability config gate so the existing encoder timing infrastructure runs by default.
2. **Per-iteration logging** — Add a lightweight JSONL logger that records which requests are in each iteration and their phase (vision encoder / prefill / decode), for correlation with nsys/ncu GPU profiling data.

## Modified Files

### 1. `vllm/v1/worker/gpu_model_runner.py`

**Change A — Encoder timing always-on**: In `_execute_mm_encoder()` (around line 2725), removed the observability config gate:

```python
# BEFORE:
should_time = bool(
    self.observability_config
    and self.observability_config.enable_mm_processor_stats
    and scheduler_output.scheduled_encoder_inputs
)

# AFTER:
should_time = bool(scheduler_output.scheduled_encoder_inputs)
```

**Change B — NVTX marker for vision encoder**: Added `record_function_or_nullcontext("gpu_model_runner: vision_encoder")` around the encoder execution loop (lines 2792-2878). This creates a distinct NVTX range in nsys so vision encoder kernels can be separated from other preprocessing. Zero overhead when `VLLM_NVTX_SCOPES_FOR_PROFILING` is not set.

### 2. `vllm/v1/engine/core.py`

**Change — Hook iteration logger**: Added `_log_iteration_data()` context manager in `EngineCore.step()` alongside existing `log_error_detail` and `log_iteration_details`. Also calls `shutdown_iteration_logger()` in `shutdown()`.

### 3. `vllm/envs.py`

**Change — Register env vars**:
- `VLLM_LOG_ITERATIONS` (bool, default False): Enable per-iteration JSONL logging.
- `VLLM_ITERATION_LOG_DIR` (str, default "."): Output directory for iteration/request JSONL files.

## New Files

### `vllm/v1/engine/iteration_logger.py`

`IterationLogger` class that:
- Writes `iterations.jsonl` with per-iteration metadata (request IDs, phase classification, token counts, elapsed time)
- Accumulates request-to-iteration mapping and writes `requests.jsonl` on shutdown
- Uses same prefill/decode classification logic as `compute_iteration_details()` from `vllm/v1/utils.py`

## Profiling Pipeline

```
1. Run with VLLM_LOG_ITERATIONS=1 + nsys
   -> iterations.jsonl, requests.jsonl (request/phase data)
   -> nsys_report.nsys-rep (GPU kernel timeline)
   -> Export to CSV: nsys_kernels.csv, nsys_nvtx.csv

2. (Optional) Run with ncu for SM metrics
   -> ncu_metrics.csv (per-kernel SM throughput, warp occupancy)

3. Post-process: python zeyu/analyze_profile.py <output_dir>
   -> consolidated_iterations.jsonl (iterations + GPU util + SM metrics)
   -> consolidated_requests.jsonl (requests + per-phase GPU/SM averages)
```

## Impact Assessment

- **Iteration logger overhead**: One JSON line per iteration (~200 bytes). Negligible.
- **NVTX marker overhead**: Zero when `VLLM_NVTX_SCOPES_FOR_PROFILING` is not set (returns `nullcontext`).
- **Encoder timing overhead**: ~1 ms per vision encoder batch from `torch.accelerator.synchronize()`.
- **Backward compatibility**: All changes are gated behind env vars or no-op when not enabled.
