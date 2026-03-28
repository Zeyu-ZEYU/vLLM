# Qwen3.5 Vision Offline Inference with Latency Measurement

This directory contains scripts and configuration for running **Qwen3.5-9B** (or other Qwen-family vision-language models) in offline mode with detailed per-request latency measurement.

## Measured Metrics

For each request, the following metrics are recorded:

| Metric | Description |
|---|---|
| **Vision Encoder Time** | Time spent running the vision encoder (`embed_multimodal`) |
| **Prefill Time** | Time from request scheduling to first generated token |
| **Decode Time** | Time from first generated token to last generated token |
| **Decode Token Count** | Number of tokens generated in the decode phase |
| **TPOT** | Average Time Per Output Token = decode_time / (num_tokens - 1) |

## Prerequisites

1. **Python environment** (using `uv`):

   ```bash
   # From the vLLM repository root:
   uv venv --python 3.12
   source .venv/bin/activate
   ```

2. **Install vLLM** (with the latency instrumentation changes):

   ```bash
   VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
   ```

3. **GPU**: At least 1 GPU with sufficient memory for the model:
   - Qwen3.5-9B: ~20 GB VRAM (1 GPU)
   - Qwen3.5-27B: ~56 GB VRAM (1 GPU) or use TP=2
   - Larger variants require more GPUs with `--tensor-parallel-size`

4. **Model access**: Ensure you have access to the model on HuggingFace, or download it locally (e.g., via ModelScope).

## Usage

### Basic Run (built-in examples)

```bash
# From the vLLM repository root:
python zeyu/run_qwen35_vision_offline.py --model /path/to/Qwen3.5-9B
```

No special environment variables needed. Without `--input`, the script uses built-in example images and questions.

### Using a JSONL Input File

```bash
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3.5-9B \
    --input zeyu/inputs/reqs/sample.jsonl
```

The JSONL file contains one JSON object per line. Each line specifies a request:

```json
{"images": ["zeyu/inputs/imgs/cherry_blossom.jpg"], "text": "What is in this image?", "delay": 0}
{"images": ["zeyu/inputs/imgs/a.jpg", "zeyu/inputs/imgs/b.jpg"], "text": "Compare these two images."}
{"text": "What is the capital of France?"}
{"images": [], "text": "Hello world"}
```

| Field | Required | Description |
|---|---|---|
| `text` | Yes | The user question / prompt text |
| `images` | No | Single path string or list of path strings. Omitting, `null`, or `[]` means text-only. |
| `delay` | No | Milliseconds to wait before submitting this request (default: 0) |

All paths are relative to the current working directory (where you run `python`).

### Per-Request Delays

Each request in the JSONL file can have a `delay` field (in milliseconds). When any request has `delay > 0`, the script submits requests one by one with the specified wait time, simulating staggered arrival.

Override all per-request delays with `--delay`:

```bash
# All requests delayed by 500ms regardless of JSONL values
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3.5-9B \
    --input zeyu/inputs/reqs/sample.jsonl \
    --delay 500

# Force zero delay (batch all at once) even if JSONL has delays
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3.5-9B \
    --input zeyu/inputs/reqs/sample.jsonl \
    --delay 0
```

### Multi-Image Requests

Requests can include multiple images. The prompt template automatically adds one `<|image_pad|>` placeholder per image:

```json
{"images": ["img1.jpg", "img2.jpg", "img3.jpg"], "text": "Describe the differences between these images."}
```

The script sets `limit_mm_per_prompt` to the maximum number of images in any single request.

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | HuggingFace model identifier or local path |
| `--input` | (none) | Path to JSONL file with requests. Uses built-in examples if not provided. |
| `--delay` | (none) | Global delay override in ms. Overrides all per-request delay values. |
| `--max-model-len` | `4096` | Maximum context length |
| `--max-num-seqs` | `5` | Maximum batch size |
| `--max-tokens` | `128` | Max generated tokens per request |
| `--tensor-parallel-size` / `-tp` | `1` | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU memory to use |
| `--temperature` | `0.0` | Sampling temperature (0.0 = greedy) |
| `--dtype` | `auto` | Model data type |

## Output

Results are written to `zeyu/outputs/latency_<YYYYMMDD_HHMMSS>.json`.

A summary table is also printed to stdout after each run.

## Directory Structure

```
zeyu/
+-- inputs/
|   +-- imgs/                      # Input images for JSONL requests
|   |   +-- cherry_blossom.jpg     # Sample image (from vLLM assets)
|   |   +-- stop_sign.jpg          # Sample image (from vLLM assets)
|   +-- reqs/
|       +-- sample.jsonl           # Sample JSONL input file
+-- outputs/                       # Timestamped JSON output files
|   +-- .gitkeep
+-- run_qwen35_vision_offline.py   # Main inference script
+-- MODIFICATIONS.md               # Documents vLLM source changes
+-- README.md                      # This file
```

## Notes

- Vision encoder timing uses vLLM's built-in `timed_encoder_operation()` infrastructure with `torch.accelerator.synchronize()` barriers for accurate GPU timing.
- Prefill and decode times come from vLLM's `RequestStateStats`, which uses monotonic timestamps from the engine core.
- Vision encoder timing is retrieved via `collective_rpc("get_encoder_timing_stats")`, which works in both single-process and default multi-process engine modes.
- Only the first request to use a given image triggers the vision encoder; subsequent requests with the same image hit the encoder cache and show VE = 0.
- When delays are used, requests are submitted via `LLM.enqueue()` + `time.sleep()` + `LLM.wait_for_completion()`. When no delays are needed, the more efficient batch `LLM.generate()` is used.
