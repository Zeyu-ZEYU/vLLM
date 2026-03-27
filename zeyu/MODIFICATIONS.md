# vLLM Source Modifications for Per-Request Latency Measurement

This document describes all modifications made to vLLM source files to support per-request vision encoder latency tracking.

## Overview

vLLM already tracks **prefill time**, **decode time**, and **TPOT** (Time Per Output Token) per request via `RequestStateStats` (accessible through `RequestOutput.metrics` when `disable_log_stats=False`).

vLLM also has a built-in **encoder timing infrastructure** (`timed_encoder_operation()`, `encoder_timing_registry`, `get_encoder_timing_stats()`) in the V1 model runner, but it is gated behind `observability_config.enable_mm_processor_stats`. The only modification needed is to remove this gate so encoder timing is always recorded.

## Modified File

### `vllm/v1/worker/gpu_model_runner.py`

**Purpose**: Make vision encoder timing always-on instead of requiring an observability config flag.

**Change**: In `_execute_mm_encoder()` method (around line 2725), removed the observability config gate:

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

**Rationale**: The existing `timed_encoder_operation()` context manager already uses `torch.accelerator.synchronize()` barriers and `time.perf_counter()` for accurate GPU timing. It records per-request encoder forward pass time in `self.encoder_timing_registry`. The only thing preventing it from running was the `enable_mm_processor_stats` flag check. By removing this gate, the timing is always recorded when there are multimodal encoder inputs to process.

**Retrieval path**: The accumulated timing is retrieved via `collective_rpc("get_encoder_timing_stats")`, which works in both single-process and multi-process engine modes:

```python
encoder_stats = llm.llm_engine.collective_rpc("get_encoder_timing_stats")
# Returns: list[dict[internal_req_id, {"encoder_forward_secs": float, "num_encoder_calls": int}]]
```

## How the Measurement Pipeline Works

```
LLM.generate()
  |-- EngineCore schedules requests
  |     |-- scheduler_output.scheduled_encoder_inputs = {req_id: [input_ids]}
  |
  |-- GPUWorker.execute_model(scheduler_output)
  |     |-- GPUModelRunner._execute_mm_encoder(scheduler_output)
  |           |-- should_time = bool(scheduled_encoder_inputs)  <-- MODIFIED
  |           |-- for each modality group:
  |           |     |-- with timed_encoder_operation(...):
  |           |           |-- torch.accelerator.synchronize()
  |           |           |-- time.perf_counter() -> t_start
  |           |           |-- model.embed_multimodal(**kwargs)
  |           |           |-- torch.accelerator.synchronize()
  |           |           |-- time.perf_counter() -> t_end
  |           |           |-- encoder_timing_registry[req_id] += elapsed
  |           |-- Cache encoder outputs in encoder_cache
  |
  |-- After inference completes:
        |-- Script calls: llm.llm_engine.collective_rpc("get_encoder_timing_stats")
        |     |-- Returns encoder timing dict from each worker
        |-- Script reads: output.metrics (RequestStateStats) for prefill/decode
        |-- Script combines all metrics and writes JSON output
```

## Impact Assessment

- **Performance impact**: The `torch.accelerator.synchronize()` calls add a small amount of latency (typically < 1 ms) per vision encoder batch. This overhead is negligible for profiling purposes.
- **Backward compatibility**: This change is fully backward-compatible. The timing was already implemented; this change only removes the gate that prevented it from running by default.
- **No new dependencies**: No new imports, classes, or fields were added. The change reuses 100% of the existing infrastructure.
