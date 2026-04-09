# Disaggregated Prefill-Decode (PD) Mode

This document describes how to run Qwen3.5-9B with **prefill-decode disaggregation**: vision encoder + text prefill on GPU 0, text decode on GPU 1.

## How It Works

The script launches two separate vLLM instances via `multiprocessing.Process`:

| GPU | Role | What runs |
|-----|------|-----------|
| GPU 0 | `kv_producer` | Vision encoder + text prefill (generates 1 token) |
| GPU 1 | `kv_consumer` | Text decode (receives KV cache, generates remaining tokens) |

KV cache is transferred between GPUs using vLLM's built-in `P2pNcclConnector` (NCCL peer-to-peer).

```
GPU 0 (prefill)                     GPU 1 (decode)
+---------------------------+       +---------------------------+
| Load model                |       | Load model                |
| Vision encode (images)    |       | Wait for prefill signal   |
| Text prefill (1st token)  |       |                           |
| Transfer KV cache --------|------>| Receive KV cache          |
| Signal done               |       | Decode (remaining tokens) |
| Wait (keep NCCL alive)    |       | Report results            |
+---------------------------+       +---------------------------+
```

## Prerequisites

- **2 GPUs** visible to the process (same node)
- vLLM installed with the instrumentation changes (editable install)
- Model accessible locally or via HuggingFace

## Usage

### Basic Run

```bash
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3.5-9B \
    --disagg
```

### With JSONL Input

```bash
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3.5-9B \
    --disagg \
    --input zeyu/inputs/reqs/sample.jsonl
```

### With nsys Profiling

```bash
bash zeyu/profile_run.sh --nsys-only --model /path/to/Qwen3.5-9B --disagg
```

Then post-process:

```bash
python zeyu/analyze_profile.py zeyu/outputs/profile_<timestamp>/
```

`analyze_profile.py` automatically detects disagg mode when it finds `prefill/` and `decode/` subdirectories.

## Output Structure

### Latency Results

`zeyu/outputs/latency_<timestamp>.json` contains the merged results with `"mode": "disagg"`. Vision encoder and prefill metrics come from GPU 0; decode metrics come from GPU 1.

### Iteration Logs (when `VLLM_LOG_ITERATIONS=1`)

Each GPU writes to its own subdirectory to avoid overwriting:

```
zeyu/outputs/profile_<timestamp>/
+-- prefill/
|   +-- iterations.jsonl      # GPU 0 iterations (encoder + prefill steps)
|   +-- requests.jsonl        # GPU 0 request-to-iteration mapping
+-- decode/
|   +-- iterations.jsonl      # GPU 1 iterations (decode steps)
|   +-- requests.jsonl        # GPU 1 request-to-iteration mapping
+-- nsys_report.nsys-rep      # nsys traces both GPUs
+-- nsys_kernels_*.csv
+-- nsys_nvtx_*.csv
```

### Consolidated Output (after `analyze_profile.py`)

```
+-- consolidated_iterations.jsonl   # All iterations from both GPUs,
|                                   # sorted by wall-clock timestamp,
|                                   # each tagged with "gpu_role"
+-- consolidated_requests.jsonl     # Per-request data merged from both GPUs
```

Each iteration record includes a `gpu_role` field (`"prefill"` or `"decode"`) so you can filter by GPU.

## Metrics Available

All the same metrics as single-GPU mode are collected per iteration:

| Metric | Source | Available in disagg? |
|--------|--------|---------------------|
| `step_latency_ms` | Inter-iteration timestamp diff | Yes (per GPU) |
| `step_rps` | Requests / latency | Yes (per GPU) |
| `gpu_util_pct` | nsys kernel timeline | Yes (per GPU) |
| `vision_encoder_gpu_util_pct` | nsys vision_encoder NVTX range | Yes (GPU 0 only) |
| `text_forward_gpu_util_pct` | nsys forward NVTX range | Yes (per GPU) |
| `kernel_launch_gap_ns` | Step-level GPU idle time between kernels (ns) | Yes (per GPU) |
| `kernel_launch_gap_pct` | Step-level GPU idle fraction (100% - gpu_util%) | Yes (per GPU) |
| `vision_encoder_kernel_launch_gap_ns` | Vision encoder GPU idle gap (ns) | Yes (GPU 0 only) |
| `vision_encoder_kernel_launch_gap_pct` | Vision encoder GPU idle fraction | Yes (GPU 0 only) |
| `text_forward_kernel_launch_gap_ns` | Text forward GPU idle gap (ns) | Yes (per GPU) |
| `text_forward_kernel_launch_gap_pct` | Text forward GPU idle fraction | Yes (per GPU) |
| `gpu_mem_allocated_MiB` | torch.cuda.memory_allocated | Yes (per GPU) |
| `prefill_req_ids` / `decode_req_ids` | Scheduler phase classification | Yes (per GPU) |

In the consolidated request file, metrics from both GPUs are merged per request:
- `encoder_iters` and `prefill_iters` come from the prefill GPU
- `decode_iters` come from the decode GPU

## Comparison with Single-GPU Mode

| Aspect | Single GPU | Disagg (PD) |
|--------|-----------|-------------|
| GPUs needed | 1 | 2 |
| `--disagg` flag | No | Yes |
| Prefill and decode | Same GPU, interleaved | Separate GPUs |
| KV cache transfer | None | P2pNcclConnector (NCCL) |
| `enforce_eager` | Not required | Forced on (CUDA graphs not supported with KV transfer) |
| Iteration logs | Single `iterations.jsonl` | `prefill/iterations.jsonl` + `decode/iterations.jsonl` |

## Limitations

- Requires both GPUs on the **same node** (NCCL P2P requires direct GPU communication).
- `enforce_eager=True` is forced — CUDA graph capture is not compatible with KV transfer.
- The prefill process stays alive (sleeping) until decode finishes, because NCCL requires both endpoints to remain active.
- Delays (`--delay`) are not supported in disagg mode (all requests are submitted at once to the prefill instance).

## Troubleshooting

**"NCCL error"**: Ensure both GPUs can communicate. Check `nvidia-smi topo -m` for P2P connectivity.

**Model doesn't fit on one GPU**: Reduce `--gpu-memory-utilization` (default 0.9). In disagg mode, each GPU loads the full model independently.

**Decode output is empty**: The prefill process must generate exactly 1 token per request. If prefill fails, the decode process has no KV cache to consume. Check the prefill process logs.
