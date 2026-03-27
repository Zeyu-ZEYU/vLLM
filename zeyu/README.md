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

### Basic Run

```bash
# From the vLLM repository root:
python zeyu/run_qwen35_vision_offline.py --model /path/to/Qwen3.5-9B
```

No special environment variables needed. The script works with vLLM's default multiprocessing mode.

### Custom Configuration

```bash
python zeyu/run_qwen35_vision_offline.py \
    --model Qwen/Qwen3.5-9B \
    --max-model-len 4096 \
    --max-num-seqs 5 \
    --max-tokens 128 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --temperature 0.0 \
    --dtype auto
```

### Using a Different Model Size

```bash
# Qwen3.5-27B (requires more GPU memory)
python zeyu/run_qwen35_vision_offline.py \
    --model Qwen/Qwen3.5-27B \
    --tensor-parallel-size 2

# Qwen3-VL-8B (older but also supported)
python zeyu/run_qwen35_vision_offline.py \
    --model Qwen/Qwen3-VL-8B-Instruct

# Qwen2.5-VL-7B
python zeyu/run_qwen35_vision_offline.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct
```

### Custom Input Images

Place `.jpg` or `.png` files in `zeyu/data/`. The script automatically picks them up and includes them as additional requests alongside the built-in test images.

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3.5-9B` | HuggingFace model identifier or local path |
| `--max-model-len` | `4096` | Maximum context length |
| `--max-num-seqs` | `5` | Maximum batch size |
| `--max-tokens` | `128` | Max generated tokens per request |
| `--tensor-parallel-size` / `-tp` | `1` | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU memory to use |
| `--temperature` | `0.0` | Sampling temperature (0.0 = greedy) |
| `--dtype` | `auto` | Model data type |

## Output

Results are written to `zeyu/outputs/latency_<YYYYMMDD_HHMMSS>.json`.

### Example Output Structure

```json
{
  "model": "Qwen/Qwen3.5-9B",
  "timestamp": "2026-03-28T06:30:00+00:00",
  "config": {
    "max_model_len": 4096,
    "max_num_seqs": 5,
    "max_tokens": 128,
    "tensor_parallel_size": 1,
    "temperature": 0.0,
    "dtype": "auto"
  },
  "summary": {
    "num_requests": 4,
    "total_decode_tokens": 512,
    "avg_vision_encoder_time_ms": 15.2,
    "avg_prefill_time_ms": 48.7,
    "avg_decode_time_ms": 340.5,
    "avg_tpot_ms": 2.68
  },
  "requests": [
    {
      "request_id": "0",
      "image_source": "cherry_blossom (built-in)",
      "question": "What is the content of this image?",
      "generated_text": "...",
      "num_prompt_tokens": 256,
      "num_generation_tokens": 128,
      "vision_encoder_time_ms": 15.2,
      "prefill_time_ms": 48.7,
      "decode_time_ms": 340.5,
      "tpot_ms": 2.68
    }
  ]
}
```

A summary table is also printed to stdout after each run.

## Directory Structure

```
zeyu/
+-- data/                           # Put custom input images here
|   +-- .gitkeep
+-- outputs/                        # Timestamped JSON output files
|   +-- .gitkeep
+-- run_qwen35_vision_offline.py    # Main inference script
+-- MODIFICATIONS.md                # Documents vLLM source changes
+-- README.md                       # This file
```

## Notes

- Vision encoder timing uses vLLM's built-in `timed_encoder_operation()` infrastructure with `torch.accelerator.synchronize()` barriers for accurate GPU timing.
- Prefill and decode times come from vLLM's `RequestStateStats`, which uses monotonic timestamps from the engine core.
- Vision encoder timing is retrieved via `collective_rpc("get_encoder_timing_stats")`, which works in both single-process and default multi-process engine modes.
- Only the first request to use a given image triggers the vision encoder; subsequent requests with the same image hit the encoder cache and show VE = 0.
