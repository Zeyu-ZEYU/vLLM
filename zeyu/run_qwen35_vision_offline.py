#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Offline inference script for Qwen3.5-9B (vision-language) with per-request
latency measurement.

Measures:
  - Vision encoder latency   (time spent in embed_multimodal)
  - Prefill latency           (first_token_ts - scheduled_ts)
  - Decode latency            (last_token_ts - first_token_ts)
  - Decode token count and average Time-Per-Output-Token (TPOT)

Results are written to zeyu/outputs/latency_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Ensure repository root is importable so vllm resolves correctly.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.assets.image import ImageAsset  # noqa: E402
from vllm.multimodal.image import convert_image_mode  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Default model -- can be overridden with --model
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

# Qwen3.5 / Qwen3-VL / Qwen2.5-VL share the same prompt template.
PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ---------------------------------------------------------------------------
# Example inputs
# ---------------------------------------------------------------------------
def build_example_inputs() -> list[dict]:
    """Return a list of example request dicts, each with an image and prompt."""

    # Built-in vLLM test images.
    cherry = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    stop = convert_image_mode(ImageAsset("stop_sign").pil_image, "RGB")

    # Also load local images from zeyu/data/ if any jpg/png files exist.
    local_images: list[tuple[str, Image.Image]] = []
    if DATA_DIR.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in sorted(DATA_DIR.glob(ext)):
                local_images.append(
                    (p.name, convert_image_mode(Image.open(p), "RGB"))
                )

    examples = [
        {
            "image_source": "cherry_blossom (built-in)",
            "image": cherry,
            "question": "What is the content of this image?",
        },
        {
            "image_source": "cherry_blossom (built-in)",
            "image": cherry,
            "question": "Describe this image in detail.",
        },
        {
            "image_source": "stop_sign (built-in)",
            "image": stop,
            "question": "What sign is shown in this image and what does it mean?",
        },
        {
            "image_source": "stop_sign (built-in)",
            "image": stop,
            "question": "List all objects you can see in this image.",
        },
    ]

    # Append local images with a generic question.
    for fname, img in local_images:
        examples.append(
            {
                "image_source": fname,
                "image": img,
                "question": "Describe this image in detail.",
            }
        )

    return examples


# ---------------------------------------------------------------------------
# Retrieve vision encoder timings via collective_rpc
# ---------------------------------------------------------------------------
def get_vision_encoder_times(llm: LLM) -> dict[str, float]:
    """Retrieve per-request vision encoder timing from the model runner.

    Uses ``collective_rpc("get_encoder_timing_stats")`` which works in both
    single-process and multi-process engine modes (no special environment
    variables required).

    The encoder timing registry uses *internal* request IDs (which have a
    random suffix appended by ``assign_request_id``), while
    ``RequestOutput.request_id`` uses the *external* (user-provided) ID.
    This function maps internal -> external IDs so callers can look up
    timing by ``RequestOutput.request_id``.
    """
    try:
        # collective_rpc returns a list (one dict per worker).
        worker_stats_list = llm.llm_engine.collective_rpc(
            "get_encoder_timing_stats"
        )
    except Exception as exc:
        print(f"[WARN] Could not retrieve encoder timing stats: {exc}")
        return {}

    # Merge results from all workers, mapping internal -> external IDs.
    result: dict[str, float] = {}
    for worker_stats in worker_stats_list:
        if not worker_stats:
            continue
        for internal_id, stats_dict in worker_stats.items():
            # Internal ID format: "{external_id}-{8_hex_chars}"
            external_id = internal_id.rsplit("-", 1)[0]
            elapsed = stats_dict.get("encoder_forward_secs", 0.0)
            # Accumulate in case multiple workers processed the same request.
            result[external_id] = result.get(external_id, 0.0) + elapsed

    return result


# ---------------------------------------------------------------------------
# Compute per-request latency metrics from RequestOutput.metrics
# ---------------------------------------------------------------------------
def extract_request_metrics(output, vision_times: dict[str, float]) -> dict:
    """Extract latency data from a single RequestOutput."""
    stats = output.metrics  # RequestStateStats or None
    result: dict = {}

    if stats is not None:
        prefill_time = stats.first_token_ts - stats.scheduled_ts
        decode_time = stats.last_token_ts - stats.first_token_ts
        num_gen = stats.num_generation_tokens
        tpot = decode_time / (num_gen - 1) if num_gen > 1 else 0.0
        result["num_prompt_tokens"] = (
            len(output.prompt_token_ids) if output.prompt_token_ids else 0
        )
        result["num_generation_tokens"] = num_gen
        result["prefill_time_s"] = round(prefill_time, 6)
        result["decode_time_s"] = round(decode_time, 6)
        result["tpot_s"] = round(tpot, 6)
        result["prefill_time_ms"] = round(prefill_time * 1000, 3)
        result["decode_time_ms"] = round(decode_time * 1000, 3)
        result["tpot_ms"] = round(tpot * 1000, 3)
    else:
        result["num_prompt_tokens"] = (
            len(output.prompt_token_ids) if output.prompt_token_ids else 0
        )
        result["num_generation_tokens"] = (
            len(output.outputs[0].token_ids) if output.outputs else 0
        )

    # Vision encoder time (keyed by external request ID).
    ve_time = vision_times.get(output.request_id, 0.0)
    result["vision_encoder_time_s"] = round(ve_time, 6)
    result["vision_encoder_time_ms"] = round(ve_time * 1000, 3)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 Vision offline inference with latency measurement"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=5,
        help="Maximum number of sequences in a batch (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per request (default: 128)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype, e.g. auto, bfloat16, float16 (default: auto)",
    )
    args = parser.parse_args()

    # Ensure output directory exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build example inputs.
    examples = build_example_inputs()
    print(f"Prepared {len(examples)} example requests.")

    # Build prompts.
    prompts = [PROMPT_TEMPLATE.format(question=ex["question"]) for ex in examples]

    # Build vLLM inputs with multimodal data.
    inputs = [
        {"prompt": prompt, "multi_modal_data": {"image": ex["image"]}}
        for prompt, ex in zip(prompts, examples)
    ]

    # Initialize LLM with stats enabled.
    print(f"Loading model: {args.model} ...")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        disable_log_stats=False,  # Enable stats for latency tracking
        seed=42,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Run inference.
    print("Running inference ...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # Retrieve vision encoder timings via collective_rpc.
    vision_times = get_vision_encoder_times(llm)
    print(
        f"Retrieved vision encoder timings for {len(vision_times)} request(s)."
    )

    # Collect results.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    request_results = []

    for i, output in enumerate(outputs):
        metrics = extract_request_metrics(output, vision_times)
        generated_text = output.outputs[0].text if output.outputs else ""

        request_results.append(
            {
                "request_id": output.request_id,
                "image_source": examples[i]["image_source"],
                "question": examples[i]["question"],
                "generated_text": generated_text,
                **metrics,
            }
        )

    # Compute summary statistics.
    n = len(request_results)
    summary = {}
    if n > 0:
        # Only count actual encoder runs (exclude cache hits where VE = 0).
        ve_times_actual = [
            r["vision_encoder_time_ms"]
            for r in request_results
            if r["vision_encoder_time_ms"] > 0
        ]
        prefill_times = [r.get("prefill_time_ms", 0.0) for r in request_results]
        decode_times = [r.get("decode_time_ms", 0.0) for r in request_results]
        tpots = [r.get("tpot_ms", 0.0) for r in request_results]
        total_decode_tokens = sum(
            r.get("num_generation_tokens", 0) for r in request_results
        )

        num_ve_runs = len(ve_times_actual)
        summary = {
            "num_requests": n,
            "num_encoder_runs": num_ve_runs,
            "total_decode_tokens": total_decode_tokens,
            "avg_vision_encoder_time_ms": (
                round(sum(ve_times_actual) / num_ve_runs, 3)
                if num_ve_runs > 0
                else 0.0
            ),
            "avg_prefill_time_ms": round(sum(prefill_times) / n, 3),
            "avg_decode_time_ms": round(sum(decode_times) / n, 3),
            "avg_tpot_ms": round(sum(tpots) / n, 3),
        }

    # Build final output.
    final_output = {
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_tokens": args.max_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "temperature": args.temperature,
            "dtype": args.dtype,
        },
        "summary": summary,
        "requests": request_results,
    }

    # Write to file.
    output_file = OUTPUT_DIR / f"latency_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to: {output_file}")

    # Print summary table.
    print("\n" + "=" * 80)
    print(f"{'LATENCY MEASUREMENT SUMMARY':^80}")
    print(f"Model: {args.model}")
    print("=" * 80)
    header = (
        f"{'Req':>4} | {'Image Source':<30} | {'VE(ms)':>8} | "
        f"{'Prefill(ms)':>11} | {'Decode(ms)':>10} | "
        f"{'GenTok':>6} | {'TPOT(ms)':>8}"
    )
    print(header)
    print("-" * 80)
    for r in request_results:
        print(
            f"{r['request_id']:>4} | "
            f"{r['image_source']:<30} | "
            f"{r.get('vision_encoder_time_ms', 0):>8.2f} | "
            f"{r.get('prefill_time_ms', 0):>11.2f} | "
            f"{r.get('decode_time_ms', 0):>10.2f} | "
            f"{r.get('num_generation_tokens', 0):>6} | "
            f"{r.get('tpot_ms', 0):>8.3f}"
        )
    print("-" * 80)
    if summary:
        avg_gen_tokens = summary["total_decode_tokens"] / summary["num_requests"]
        ve_note = f"(n={summary['num_encoder_runs']}, excl. cache hits)"
        print(
            f"{'AVG':>4} | "
            f"{'':30} | "
            f"{summary['avg_vision_encoder_time_ms']:>8.2f} | "
            f"{summary['avg_prefill_time_ms']:>11.2f} | "
            f"{summary['avg_decode_time_ms']:>10.2f} | "
            f"{avg_gen_tokens:>6.0f} | "
            f"{summary['avg_tpot_ms']:>8.3f}"
        )
        print(
            f"{'SUM':>4} | "
            f"{'':30} | "
            f"{'':>8} | "
            f"{'':>11} | "
            f"{'':>10} | "
            f"{summary['total_decode_tokens']:>6} | "
            f"{'':>8}"
        )
        print(f"  * VE avg computed over actual encoder runs only {ve_note}")
    print("=" * 80)


if __name__ == "__main__":
    main()
