# Single-GPU Vision-Encoder ↔ Decode Pipeline (`--mm-pipeline`)

This doc describes the `--mm-pipeline` feature: a **single-GPU** mode that
overlaps the vision encoder (ViT, compute-bound) with text decode (memory-
bandwidth-bound) so both run concurrently on the same device. It is
independent from the cross-node PD-disaggregation mode documented in
[README_DISAGG.md](./README_DISAGG.md); the two are orthogonal.

---

## Contents

- [Motivation](#motivation)
- [How it works](#how-it-works)
- [Usage](#usage)
- [What changed in the code](#what-changed-in-the-code)
- [Verifying correctness](#verifying-correctness)
- [Measuring speedup](#measuring-speedup)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

---

## Motivation

In vanilla vLLM V1, a scheduler step that admits a new multimodal request
runs like this on a single GPU:

```
default stream: [ VE ] → [ gather ] → [ text forward (decodes of A,C,D + prefill of B) ]
                 ~50 ms      tiny         ~200 ms
```

The ViT work for request B blocks the decode work for reqs A, C, D. For
a workload where new multimodal requests arrive during a steady stream
of decodes, the GPU sits compute-bound during VE while its memory
bandwidth is idle — and then vice versa during decode.

With `--mm-pipeline on`:

```
encoder stream: [ VE for req B ]
default stream: [ text forward (decodes of A,C,D) ]
                 ──────────── concurrent ────────────

(next iter)
default stream: [ text forward (decodes of A,C,D + prefill of B) ]
```

VE is pulled forward by one scheduler iteration onto a dedicated side
stream, so the default stream keeps decoding other reqs without stalling
on the ViT. The new req's first token arrives one iteration later than
it would have without pipeline — but the total wall-clock for the
workload drops by roughly `min(T_VE, T_decode_iter)` per multimodal
arrival.

Chunked prefill already gives you this kind of interleaving between
text-prefill and text-decode via variable-length attention inside a
single forward pass. Pipeline mode is the analogous trick for the
vision encoder, which is outside that forward pass.

---

## How it works

Three coordinated changes across the engine:

### 1. Scheduler pre-schedules the VE of `waiting` reqs

In `Scheduler.schedule()` (V1), a new pass `_schedule_pipeline_prefetch`
runs **after** the running-reqs loop and **before** the waiting-admission
loop. For each req in `self.waiting` that:

- has multimodal inputs not yet in the encoder cache, and
- fits into the current iteration's `encoder_compute_budget`,

the scheduler:

- reserves encoder-cache slots via `EncoderCacheManager.allocate`,
- records the encoder input indices in `SchedulerOutput.scheduled_encoder_inputs[req_id]`,
- sets `SchedulerOutput.num_scheduled_tokens[req_id] = 0` (no text forward this iter),
- emits a minimal `NewRequestData` in `SchedulerOutput.scheduled_new_reqs`
  with **empty** `block_ids` so the worker has the req's `mm_features`
  available but no KV allocation happens.

The admission loop then skips these reqs (routes them into
`skipped_waiting`), deferring their text-prefill to a later iter. When
the scheduler eventually admits the req for real,
`_try_schedule_encoder_inputs` sees the encoder cache hit and skips
re-scheduling the ViT; the req runs text-prefill only.

### 2. Worker launches VE kernels on a side stream

`GPUModelRunner._execute_mm_encoder` wraps its kernel-launching body with

```python
self.encoder_stream.wait_stream(torch.cuda.current_stream())
torch.cuda.set_stream(self.encoder_stream)
# ... all ViT kernels ...
self.encoder_done_event = torch.cuda.Event()
self.encoder_done_event.record(self.encoder_stream)
torch.cuda.set_stream(default_stream)
```

All ViT kernels run on `self.encoder_stream`, created in `__init__`
only when `multimodal_config.mm_pipeline == "on"`. After the last VE
kernel is recorded on the side stream, the default stream issues

```python
torch.cuda.current_stream().wait_event(self.encoder_done_event)
```

right before `_gather_mm_embeddings` reads the encoder cache. This
preserves correctness: the text forward never sees partial ViT output.

Because the ViT kernels landed on a different stream, the CUDA caching
allocator tracks the output tensor's stream; the `wait_event` above is
what lets the default stream read those tensors safely.

### 3. Worker accepts "prefetch-only" new reqs

`GPUModelRunner._update_states` detects pipeline-prefetch-only new reqs
via the combination

```python
(num_scheduled_tokens.get(req_id, 0) == 0
 and bool(scheduled_encoder_inputs.get(req_id)))
```

For these, it creates a `CachedRequestState` and inserts it into
`self.requests` (so `_batch_mm_inputs_from_scheduler` can pull
`mm_features`) but does **not** add it to the `input_batch` (no tokens
to run). In the subsequent iter that admits the req, the scheduler
re-announces it via `scheduled_new_reqs` with real `block_ids`, and
`_update_streaming_request` adds it to `input_batch` normally.

---

## Usage

### Enable via `run_qwen35_vision_offline.py`

```bash
# Pipeline off (current default, matches upstream behavior)
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3-VL-8B-Instruct \
    --num-prompts 20 --max-tokens 64 \
    --mm-pipeline off \
    --output-dir zeyu/outputs/baseline

# Pipeline on
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3-VL-8B-Instruct \
    --num-prompts 20 --max-tokens 64 \
    --mm-pipeline on \
    --output-dir zeyu/outputs/pipeline
```

The flag is **single-GPU only**. If you pass it together with
`--role prefill` or `--role decode` (PD-disagg), the script prints a
warning and forces it off.

### Or pass directly to `LLM(...)`

```python
from vllm import LLM
llm = LLM(
    model="/path/to/Qwen3-VL-8B-Instruct",
    mm_pipeline="on",
    enforce_eager=True,
    ...
)
```

Or via `--mm-pipeline` on any vLLM-based CLI entry point that builds
its config from `EngineArgs`.

---

## What changed in the code

Every change is gated by `multimodal_config.mm_pipeline == "on"`; when
off, the worker path is identical to upstream.

| File | Change |
|---|---|
| `vllm/config/multimodal.py` | New `MMPipelineMode` type alias; new `mm_pipeline: Literal["off","on"] = "off"` field on `MultiModalConfig`. |
| `vllm/config/model.py` | New `mm_pipeline` `InitVar`; passed into `MultiModalConfig` kwargs. |
| `vllm/engine/arg_utils.py` | New `mm_pipeline` arg field; new `--mm-pipeline` CLI flag; forwarded into `ModelConfig(...)` via `create_*`. |
| `vllm/v1/core/sched/scheduler.py` | `_mm_pipeline_on` attribute; `_schedule_pipeline_prefetch()` method; call site between running-pass and waiting-admission-loop; skip prefetched reqs in admission; merge prefetch `NewRequestData` into final `new_reqs_data`. |
| `vllm/v1/worker/gpu_model_runner.py` | `self.encoder_stream`, `self.encoder_done_event`; `_execute_mm_encoder` wraps body with stream swap + event record; `execute_model` inserts `wait_event` before encoder-cache consumption; `_update_states` handles pipeline-prefetch-only new reqs (keeps in `self.requests`, skips `input_batch`). |
| `zeyu/run_qwen35_vision_offline.py` | `--mm-pipeline {off,on}` flag; forced off for `--role` in `{prefill, decode}`; forwarded through `_common_llm_kwargs`. |

---

## Verifying correctness

The pipeline must produce **identical token output** compared to
`--mm-pipeline off` on the same seed (deterministic greedy sampling):

```bash
# Baseline
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3-VL-8B-Instruct \
    --num-prompts 4 --max-tokens 32 \
    --mm-pipeline off \
    --output-dir zeyu/outputs/pipeline_off

# Pipeline on, same seed
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3-VL-8B-Instruct \
    --num-prompts 4 --max-tokens 32 \
    --mm-pipeline on \
    --output-dir zeyu/outputs/pipeline_on

# Compare generated_text — must match
diff <(jq -S '.requests[] | .generated_text' \
         zeyu/outputs/pipeline_off/latency.json) \
     <(jq -S '.requests[] | .generated_text' \
         zeyu/outputs/pipeline_on/latency.json)
```

If the diff is non-empty, something in the side-stream ordering is
wrong. Most likely culprit: missing `wait_event` before a consumer of
`encoder_cache` on the default stream. Check
`vllm/v1/worker/gpu_model_runner.py:execute_model` and confirm each
reader of `encoder_cache` is preceded by a `wait_event`.

> **Important caveat for token-match testing.** Compare modes only
> under **batch-submit** workloads (all requests enqueued at once via
> `llm.generate(...)`). Staggered arrivals (`--delay N`, or
> `llm.enqueue(...)` in a loop) introduce timing-dependent batch
> composition that changes the order of floating-point reductions,
> and *two back-to-back runs of the same mode* can produce a few
> different tokens. This is not a pipeline bug — it is an inherent
> property of async scheduling with chunked prefill. Measured on
> this fork (Qwen3-VL-8B-Instruct, 20 reqs, 100 ms inter-arrival):
> two back-to-back `--mm-pipeline off` runs diverged on 2 of 20
> requests, the same order as the `on` vs `off` diff. Bit-exact
> comparison is only meaningful when submission is deterministic.

---

## Measuring speedup

Two data sources:

1. **`iterations.jsonl`** (always on via `VLLM_LOG_ITERATIONS=1`) —
   one record per scheduler iteration with sampled GPU util / memory
   from pynvml and step-level timing.
2. **`consolidated_iterations.jsonl`** (written by
   `analyze_profile.py` when the run was wrapped in `nsys profile`, i.e.
   `--nsys` was passed to the launcher) — same records with kernel-level
   fields added from the nsys CSV exports.

nsys is required for the fine-grained per-phase kernel metrics
(kernel-busy % during the VE sub-range vs the text-forward sub-range,
kernel-launch gap, etc.). pynvml alone gives 100 Hz sampled GPU util
for the whole iteration, not broken down by phase.

Compare between modes:

| Metric | Source | Expected change with pipeline on |
|---|---|---|
| `step_latency_ms` | `iterations.jsonl` | Lower on iters with new-MM arrivals |
| `vision_encoder_kernel_time_ns` | `consolidated_iterations.jsonl` | Unchanged (same kernels) |
| `text_forward_kernel_time_ns` | `consolidated_iterations.jsonl` | Unchanged |
| **`nvtx_overlap_ns`** (per-iter) | `consolidated_iterations.jsonl` | **0 in off mode; > 0 on overlap iters in on mode** |
| **`sum_nvtx_overlap_ns`** (summary) | `disagg_summary.json` per-phase block | Total pipeline wall-clock savings |
| `rps_end_to_end` | summary | Higher when workload has new MM arrivals during ongoing decode |
| `nvml_gpu_util_pct_mean` | `iterations.jsonl` | Higher on overlap iters |

### About `nvtx_overlap_ns`

Computed in `analyze_profile.py` as the wall-clock duration during
which the GPU was executing **both** vision-encoder kernels AND
text-forward kernels at the same time, inside a single scheduler
iteration. Measured from the `cuda_gpu_trace` (actual kernel start +
end per stream), not from host-side NVTX endpoints — the host-side
scopes for the encoder are just microseconds (the time to launch
async kernels on the side stream), so they would always read ~0 and
miss the real GPU-side concurrency.

Implementation:
1. Filter kernels whose start falls in each iter's VE NVTX window →
   `ve_kernels`.
2. Same for text forward → `fwd_kernels`.
3. Merge each set's intervals → `ve_busy`, `fwd_busy`.
4. Sum the length of their intersection.

With `--mm-pipeline off`: both sets run on the default stream and are
serialized by the GPU → intersection = 0 by construction.
With `--mm-pipeline on`: VE is on `encoder_stream`, forward on default,
and the GPU can execute both (compute-bound + memory-bound) → positive
intersection whenever the iter has a prefetched VE concurrent with an
ongoing decode / prefill.

Aggregate: `disagg_summary.json["summary"]["decode" / "prefill" /
"vision_encoder"]` blocks will include `avg_nvtx_overlap_pct` (per-iter
fraction of the iter window that was overlapped) and `sum_nvtx_overlap_ns`
(total wall-clock covered by overlap across all iters of that phase).

---

## Limitations

- **Single-GPU only.** The flag is forced off when `--role` is
  `prefill`/`decode` (PD-disagg). A future iteration can combine the two
  by running the pipeline on the disagg prefill node (where the same
  GPU does VE + prefill for different reqs).
- **TP = 1 is the tested configuration.** Each TP rank will get its
  own `encoder_stream` and will receive the same `scheduled_encoder_inputs`
  via `SchedulerOutput`, so in principle TP > 1 should work, but it has
  not been benchmarked.
- **Encoder CUDA graph** (`cudagraph_mm_encoder=True`) replay on a
  non-capture stream has historically had edge cases. If you enable
  both `--mm-pipeline on` and encoder CUDA graphs and see crashes,
  disable one or re-capture on the `encoder_stream` during warmup.
- **No first-iter benefit.** If the VERY first request of the engine
  session is multimodal, there is no prior iter to run its VE in; the
  pipeline falls back to the synchronous path for that one request.
- **`is_encoder_decoder=True` models** (classic encoder-decoder
  architectures) are excluded from the prefetch pass — they have a
  different encoder-semantic that the V1 scheduler already specializes
  for.

---

## Troubleshooting

**Tokens differ between `--mm-pipeline on` and `off`**
Most likely a missing `wait_event` on the default stream before a
reader of `encoder_cache`. Grep for `encoder_done_event` in
`gpu_model_runner.py`; ensure every consumer waits before reading.

**Engine crashes at `_batch_mm_inputs_from_scheduler` with KeyError**
The scheduler emitted `scheduled_encoder_inputs[req_id]` without a
corresponding `NewRequestData` entry. Verify the prefetch pass in
`Scheduler._schedule_pipeline_prefetch` appends to
`pipeline_prefetch_new_reqs_data` and that the caller in
`Scheduler.schedule` merges it into `new_reqs_data`.

**`step_latency_ms` unchanged or higher with `on`**
Overlap requires that the default stream has work to do while VE runs.
If your workload is a single stream of multimodal reqs with no ongoing
decodes (e.g. 1-req-at-a-time), there's nothing to overlap with, so
pipeline is a no-op or slight regression due to event overhead.

**Warning "forcing off for --role {prefill,decode}"**
Expected. Pipeline is single-GPU only; the PD-disagg path manages its
own VE/decode split across nodes.
