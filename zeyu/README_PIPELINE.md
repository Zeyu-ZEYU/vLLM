# Single-GPU Vision-Encoder ↔ Decode Pipeline (`--mm-pipeline`)

A single-GPU mode for multimodal LLMs that overlaps the vision encoder
(ViT, compute-bound) with text decode (memory-bandwidth-bound) by
scheduling the ViT kernels on a dedicated CUDA side stream one
iteration ahead of when the new request is admitted.

**In one sentence**: when a new multimodal request arrives while other
requests are already decoding, its vision encoder runs in parallel with
those ongoing decodes instead of stalling them.

This feature is **orthogonal to PD-disaggregation**
([README_DISAGG.md](./README_DISAGG.md)). It runs on a single GPU; it
does not use cross-node KV transfer.

---

## Contents

1. [When this helps vs when it doesn't](#when-this-helps-vs-when-it-doesnt)
2. [Prerequisites](#prerequisites)
3. [Checkout + build](#checkout--build)
4. [Quick start (offline, single GPU)](#quick-start-offline-single-gpu)
5. [Verify correctness](#verify-correctness)
6. [Profile with nsys (optional but recommended)](#profile-with-nsys-optional-but-recommended)
7. [Observe the overlap empirically: server + async client](#observe-the-overlap-empirically-server--async-client)
8. [Metrics reference](#metrics-reference)
9. [Limitations](#limitations)
10. [Troubleshooting](#troubleshooting)
11. [Appendix: how it works inside](#appendix-how-it-works-inside)

---

## When this helps vs when it doesn't

Pipeline mode **helps** when the workload is a steady stream of
multimodal requests with ongoing decodes — e.g. an online API server
with concurrent user traffic. Every new multimodal arrival that hits
the scheduler while at least one other request is decoding gets its
ViT pre-computed for free on the side stream.

Pipeline mode **does not help** when you submit all requests at once
and wait (`llm.generate([...])` on a batch). In that case the scheduler
prefetches everyone's ViT in iter 0 (greedy) and iters 1+ are
pure-decode with nothing left to overlap. You will see correct output
and the feature is a no-op. See the [server recipe
below](#observe-the-overlap-empirically-server--async-client) for the
workload where overlap actually fires.

---

## Prerequisites

Hardware:
- 1 NVIDIA GPU with enough memory for the target MM model (Qwen3-VL-8B
  needs ≈18 GiB for weights + working set at `max-model-len=4096`).

Software:
- CUDA runtime + driver that match the PyTorch build (any version that
  already works with upstream vLLM V1 works here).
- Python ≥3.10.
- `nsys` (Nsight Systems) CLI on PATH — optional, only needed if you
  want kernel-level overlap metrics. Any recent version works; just
  ensure `nsys --version` runs in the same environment where you
  launch vLLM.

Model:
- Any multimodal model vLLM V1 already supports. Examples in this doc
  use `Qwen/Qwen3-VL-8B-Instruct`; point to your local checkpoint or
  the HF ID.

---

## Checkout + build

```bash
# Clone the fork and check out the development branch.
git clone <this-fork-url> vllm
cd vllm
git switch mono_kernel_dev
git log --oneline | head -5     # HEAD should include the mm-pipeline
                                 # commits: "feat(mm): single-GPU …" and
                                 # "fix(mm-pipeline): correct wait_event …"

# Editable install (same as upstream vLLM).
pip install -e .
```

That is the only build step — `--mm-pipeline` is a config flag, not a
build-time switch.

---

## Quick start (offline, single GPU)

Smoke-test with the built-in image assets. `zeyu/run_qwen35_vision_offline.py`
is an offline runner that accepts `--mm-pipeline {off,on}`:

```bash
# Pipeline OFF — baseline, matches upstream behavior.
python zeyu/run_qwen35_vision_offline.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --num-prompts 4 --max-tokens 32 \
    --mm-pipeline off \
    --output-dir ./out_off

# Pipeline ON — side-stream VE.
python zeyu/run_qwen35_vision_offline.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --num-prompts 4 --max-tokens 32 \
    --mm-pipeline on \
    --output-dir ./out_on
```

Both runs should finish successfully. In this offline batch workload
you should NOT expect a wall-clock speedup — see the next section on
why the offline script is only meaningful for correctness testing.

If you have multiple GPUs, pin one with `--gpu N` (sets
`CUDA_VISIBLE_DEVICES`).

---

## Verify correctness

Run the same prompts twice — once with `off`, once with `on` — and
diff the generated text. With greedy sampling (the default here,
`--temperature 0.0`) the two runs must produce **bit-exact identical
tokens**.

```bash
python - <<'PY'
import json, glob, sys
off = json.load(open(sorted(glob.glob("out_off/latency_*.json"))[-1]))
on  = json.load(open(sorted(glob.glob("out_on/latency_*.json"))[-1]))
a = [r["generated_text"] for r in off["requests"]]
b = [r["generated_text"] for r in on["requests"]]
diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
print(f"{len(a)} requests, {len(diffs)} mismatches")
if diffs:
    for i, x, y in diffs[:3]:
        print(f"--- req {i} ---\nOFF: {x!r:.120}\nON : {y!r:.120}")
    sys.exit(1)
PY
```

0 mismatches is the pass criterion. If you get any diffs, open an
issue — it indicates a missed `wait_event` on the side-stream encoder
output.

> **Caveat.** Bit-exact comparison only works for batch-submit (a
> single `llm.generate([...])` call). Staggered submission
> (`--delay N` or an async client in a loop) produces timing-dependent
> batch composition, which changes the order of floating-point
> reductions — and *two back-to-back runs of the same mode* can differ
> in a few tokens. This is an inherent property of async scheduling
> with chunked prefill, not a pipeline bug.

---

## Profile with nsys (optional but recommended)

If you want kernel-level metrics (GPU-util, VE kernel time, text-
forward kernel time, and — when the workload exposes it —
`nvtx_overlap_ns`), wrap the run in `nsys profile`.

Two settings are required for nsys to actually see per-kernel GPU
activity on typical CUDA/CUPTI combinations:

1. `--enforce-eager` on the script (disables CUDA graph capture; graph
   replay can hide kernels from `cuda_gpu_trace`).
2. `VLLM_ENABLE_V1_MULTIPROCESSING=0` (run the engine in the same
   process as nsys instead of a spawned subprocess).

The repo ships a helper that sets everything up:

```bash
bash zeyu/pipeline_run.sh \
    --mm-pipeline on \
    --num-prompts 4 --max-tokens 32 \
    --gpu 0 \
    --nsys
```

Output under `zeyu/outputs/pipeline_on_<UTC>/`:
- `nsys_report.nsys-rep` — raw profile, open in Nsight Systems GUI.
- `nsys_kernels_cuda_gpu_trace.csv` — every kernel's start / end on
  each CUDA stream.
- `nsys_nvtx_pushpop_nvtx_pushpop_trace.csv` — NVTX scope timings.
- `iterations.jsonl` — per-iter pynvml samples + scheduler state.
- `consolidated_iterations.jsonl` — `iterations.jsonl` enriched with
  per-phase kernel metrics from the CSVs, including `nvtx_overlap_ns`.

If `nsys` is not at `/usr/local/bin/nsys` or similar, pass
`--nsys-bin /path/to/nsys` to the helper.

You can also run the full chain by hand; see `zeyu/pipeline_run.sh` for
the exact commands.

---

## Observe the overlap empirically: server + async client

The offline `llm.generate(...)` path cannot show overlap because every
request is in the scheduler queue before the first iteration runs.
To see `nvtx_overlap_ns > 0` you need **staggered arrival**: a new
multimodal request must enter the waiting queue while another request
is already decoding.

The simplest setup is vLLM's OpenAI-compatible server + a small async
client that submits requests at a steady rate.

### Terminal 1 — start the server

```bash
VLLM_LOG_ITERATIONS=1 \
VLLM_ITERATION_LOG_DIR=./server_logs \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
CUDA_VISIBLE_DEVICES=0 \
nsys profile --trace=cuda,nvtx \
     --output=./server_logs/nsys_report --force-overwrite=true \
  python -m vllm.entrypoints.openai.api_server \
         --model Qwen/Qwen3-VL-8B-Instruct \
         --max-model-len 4096 \
         --max-num-seqs 8 \
         --enforce-eager \
         --mm-pipeline on \
         --limit-mm-per-prompt '{"image": 1}' \
         --host 127.0.0.1 --port 8000
```

(Drop the `nsys` wrapper if you just want pynvml metrics; keep
`VLLM_LOG_ITERATIONS=1` so `iterations.jsonl` gets written.)

### Terminal 2 — async streaming client

```python
# save as stream_client.py
import asyncio, base64, io, time
from openai import AsyncOpenAI
from PIL import Image

client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")

def b64_image(path):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

async def one(i, img_b64, prompt):
    t0 = time.monotonic()
    r = await client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=64,
        temperature=0.0,
    )
    print(f"req {i} done in {time.monotonic()-t0:.2f}s")
    return r.choices[0].message.content

async def main():
    # Use N DISTINCT images (different mm_hash → one VE per request).
    img_paths = ["img01.jpg", "img02.jpg", ..., "img20.jpg"]
    prompt = "Describe this image in one sentence."

    imgs = [b64_image(p) for p in img_paths]

    # Stagger: launch one request every 200 ms so later arrivals hit
    # the scheduler while earlier ones are still decoding.
    tasks = []
    for i, img_b64 in enumerate(imgs):
        tasks.append(asyncio.create_task(one(i, img_b64, prompt)))
        await asyncio.sleep(0.2)
    await asyncio.gather(*tasks)

asyncio.run(main())
```

```bash
python stream_client.py
```

### Why this shows overlap

With the server above and the staggered client:

- Iter K: req 3 is decoding. Scheduler sees req 4 in `waiting`, pre-
  schedules req 4's VE.
- Iter K, worker: req 4's VE kernels launch on `encoder_stream`; req
  3's decode kernels launch on the default stream. These two streams
  execute in parallel — **this is the overlap**.
- Iter K+1: req 4 admitted for real. Its ViT output is already in the
  encoder cache; only text prefill runs this iter.

Checking the results:

```bash
# Produce consolidated_iterations.jsonl from the nsys run.
python zeyu/analyze_profile.py ./server_logs

# Count overlap iters.
python - <<'PY'
import json
rows = [json.loads(l) for l in open("./server_logs/consolidated_iterations.jsonl") if l.strip()]
ov = [r for r in rows if r.get("nvtx_overlap_ns", 0) > 0]
print(f"iters: {len(rows)}, iters with overlap > 0: {len(ov)}")
total = sum(r.get("nvtx_overlap_ns", 0) for r in rows) / 1e6
print(f"total overlap: {total:.2f} ms")
PY
```

You should see a non-zero count on overlap iters and a non-zero total.
If `nvtx_overlap_ns` is 0 across the board, either (a) the client
isn't actually staggered (all images were sent in one HTTP batch), or
(b) all images have the same `mm_hash` (e.g. the same file cloned)
and the encoder cache is serving them all from one run. Fix by using
truly distinct images and a real `asyncio.sleep` between submissions.

### Without nsys

Same server + client, skip the nsys wrapper. You will not get
`nvtx_overlap_ns`, but `iterations.jsonl` will have per-iter
`step_latency_ms` and `nvml_gpu_util_pct_mean`. Compare between
`--mm-pipeline off` and `on`:

- In `off`: iters with a new MM arrival show a large
  `step_latency_ms` (VE + text forward serialized).
- In `on`: the same iters show `step_latency_ms` close to the
  no-new-MM iters (VE was pre-computed on the side stream).

---

## Metrics reference

Produced without nsys (just `VLLM_LOG_ITERATIONS=1`):

| Field | File | What it is |
|---|---|---|
| `step_latency_ms` | `iterations.jsonl` | Wall time of this scheduler iteration. |
| `nvml_gpu_util_pct_mean` | `iterations.jsonl` | Mean pynvml-sampled GPU utilization during the iter. |
| `nvml_gpu_util_pct_max` | `iterations.jsonl` | Peak sample during the iter. |
| `num_scheduled_tokens` | `iterations.jsonl` | Total text tokens scheduled this iter. |
| Request-level TTFT / JCT / decode times | `latency_*.json` | Per-request timing from `RequestOutput.metrics`. |

Added by nsys + `analyze_profile.py` (writes `consolidated_iterations.jsonl`):

| Field | What it is |
|---|---|
| `gpu_util_pct` | Fraction of iter window during which at least one GPU kernel was running. |
| `vision_encoder_kernel_time_ns` | Total kernel wall time inside the `vision_encoder` NVTX window (side stream when pipeline is on). |
| `text_forward_kernel_time_ns` | Same for the `forward` NVTX window (default stream). |
| `kernel_launch_gap_ns` | Sum of gaps between consecutive kernels — a large value means CPU-bound, small means GPU saturated. |
| `num_kernels` | Kernel count in the iter. |
| **`nvtx_overlap_ns`** | Wall-clock duration during which VE kernels and text-forward kernels were both executing simultaneously (kernel-level intersection between the two streams). **This is the headline metric.** |
| `nvtx_overlap_pct` | Same as above, normalized by iter window. |

With `--mm-pipeline off` all VE and forward kernels are on the default
stream so they're serialized by the GPU — `nvtx_overlap_ns = 0` by
construction.

With `--mm-pipeline on` in a streaming workload `nvtx_overlap_ns > 0`
on iters where a prefetched VE ran concurrent with an ongoing
decode / prefill.

---

## Limitations

- **Single-GPU only.** The flag is silently forced off when the offline
  runner is started with `--role prefill` or `--role decode`
  (PD-disagg path). Pipeline mode + PD-disagg is not wired up (yet) —
  would be a future change on the prefill node.
- **TP = 1 tested.** TP > 1 should work (each rank creates its own
  `encoder_stream` and receives the same `scheduled_encoder_inputs`
  from the scheduler), but has not been benchmarked.
- **Encoder CUDA graphs** (`cudagraph_mm_encoder=True`) + side stream:
  untested. If you enable both and see crashes, disable encoder CUDA
  graphs or re-capture on `encoder_stream` during warmup.
- **First request of the engine session.** If the very first request
  is multimodal, there is no prior iter on which to pre-run its VE,
  so that one request runs the synchronous path. Subsequent MM
  arrivals benefit normally.
- **Encoder-decoder architectures** (`is_encoder_decoder=True`) are
  excluded from the prefetch pass. Their VE is always synchronous.

---

## Troubleshooting

**Tokens differ between `off` and `on` under batch-submit (greedy,
same seed).** Bug. Most likely a missing `wait_event` on the default
stream before a reader of `encoder_cache`. Grep
`vllm/v1/worker/gpu_model_runner.py` for `encoder_done_event` and
check that every reader of `self.encoder_cache[mm_hash]` is preceded
by either the conditional `wait_event` block or the unconditional
wait (encoder-decoder path).

**`nvtx_overlap_ns` is 0 in every iter even with `--mm-pipeline on`.**
Either (a) the workload is batch-submit (see
[the server recipe](#observe-the-overlap-empirically-server--async-client)),
or (b) all requests share the same image → single mm_hash → one VE
total, nothing to overlap. Use distinct images.

**`cuda_gpu_trace` comes back empty from `nsys stats`.** Need
`--enforce-eager` + `VLLM_ENABLE_V1_MULTIPROCESSING=0`. CUDA graph
replay and the spawn subprocess both fool CUPTI.

**`RuntimeError: Engine core initialization failed. Failed core
proc(s): {}`.** This is the PD-disagg startup error covered in
[README_DISAGG.md](./README_DISAGG.md), not a pipeline issue. Make
sure you are not combining `--mm-pipeline` with `--role
{prefill,decode}`.

**Warning printed: "forcing --mm-pipeline off for --role
{prefill,decode}".** Expected. Pipeline is single-GPU; the disagg
path already splits VE from decode across nodes.

**`step_latency_ms` unchanged or slightly higher with `on`.** Pipeline
only helps when the default stream has work to do while VE runs. A
workload of "one request at a time to completion" has nothing to
overlap with; the side stream setup costs microseconds, but you'll
see no benefit.

---

## Appendix: how it works inside

If you want to understand or extend the implementation, here is a
brief map of the three coordinated changes. All are gated by
`multimodal_config.mm_pipeline == "on"`; with `off` the worker path
is identical to upstream.

### 1. Scheduler: pre-schedule the VE of waiting reqs

`vllm/v1/core/sched/scheduler.py::_schedule_pipeline_prefetch` is a new
pass that runs after the running-reqs loop and before the waiting-
admission loop. For each waiting req that has MM inputs not yet
cached and fits in `encoder_compute_budget`:

- Reserves encoder-cache slots via `EncoderCacheManager.allocate`.
- Adds entries to `SchedulerOutput.scheduled_encoder_inputs[req_id]`.
- Sets `num_scheduled_tokens[req_id] = 0` (no text forward).
- Emits a minimal `NewRequestData` with empty `block_ids` so the
  worker has the req's `mm_features` but no KV is allocated yet.

The admission loop skips these reqs, deferring their text prefill to
a later iter. When the scheduler eventually admits a prefetched req
for real, `_try_schedule_encoder_inputs` sees the cache hit and
schedules only text tokens.

### 2. Worker: VE kernels on a side stream

`vllm/v1/worker/gpu_model_runner.py::_execute_mm_encoder` wraps its
kernel-launching body:

```python
self.encoder_stream.wait_stream(default_stream)
torch.cuda.set_stream(self.encoder_stream)
# ... all ViT kernels ...
self.encoder_done_event = torch.cuda.Event()
self.encoder_done_event.record(self.encoder_stream)
self._pending_ve_hashes.update(mm_hashes)
torch.cuda.set_stream(default_stream)
```

`self._pending_ve_hashes` is the set of mm_hashes whose ViT kernels
have been issued on the side stream but whose completion has NOT yet
been observed by the default stream. It accumulates across iterations.

Before the default stream reads `encoder_cache`, `execute_model`
issues a conditional wait:

```python
if encoder_done_event and self._pending_ve_hashes:
    admitted_hashes = union of mm_hashes in this iter's input_batch
    if self._pending_ve_hashes & admitted_hashes:
        current_stream.wait_event(encoder_done_event)
        self._pending_ve_hashes.clear()
        encoder_done_event = None
```

This is what preserves correctness **and** enables overlap:
- If the iter only has decodes of earlier reqs (no reader of the
  just-prefetched hash), no wait fires, and decode runs in parallel
  with the side-stream ViT.
- If the iter admits a req that will read a pending hash, we wait
  first. Stream ordering on `encoder_stream` guarantees the wait
  covers *all* prior events, so we can safely clear the whole
  pending set.

### 3. Worker: accept "prefetch-only" new reqs

`_update_states` detects new reqs that have
`num_scheduled_tokens == 0` but non-empty
`scheduled_encoder_inputs`. It creates a `CachedRequestState` in
`self.requests` (so `_batch_mm_inputs_from_scheduler` can find
`mm_features`) but does NOT add it to `input_batch`. When the
admission iter eventually arrives, the scheduler re-announces the req
with real `block_ids` and `_update_streaming_request` adds it to
`input_batch` normally.

### Files touched

All edits are gated by `mm_pipeline=="on"`:

- `vllm/config/multimodal.py` — new `MMPipelineMode` type + field.
- `vllm/config/model.py` — pass `mm_pipeline` through `InitVar`.
- `vllm/engine/arg_utils.py` — `--mm-pipeline` CLI flag.
- `vllm/v1/core/sched/scheduler.py` — `_schedule_pipeline_prefetch`,
  call site, skip-in-admission, merge into `new_reqs_data`.
- `vllm/v1/worker/gpu_model_runner.py` — `encoder_stream`,
  `encoder_done_event`, `_pending_ve_hashes`, side-stream wrap,
  conditional wait, prefetch-only branch in `_update_states`.
- `zeyu/run_qwen35_vision_offline.py` — `--mm-pipeline` flag,
  forced off under `--role {prefill,decode}`.
- `zeyu/analyze_profile.py` — `nvtx_overlap_ns` / `nvtx_overlap_pct`
  per-iter fields.
- `zeyu/pipeline_run.sh` — one-shot helper (nsys wrap, CSV export,
  analyze, summary).
