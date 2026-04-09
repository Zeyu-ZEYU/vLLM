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

Supports single-GPU and PD disaggregated mode (--disagg):
  GPU 0: vision encoder + text prefill
  GPU 1: text decode

Results are written to zeyu/outputs/latency_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Event, Process
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
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Default model -- can be overridden with --model
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(question: str, num_images: int) -> str:
    """Build a Qwen3.5 chat-template prompt.

    Args:
        question: The user question text.
        num_images: Number of images attached (0 = text-only).
    """
    vision_block = (
        "<|vision_start|><|image_pad|><|vision_end|>" * num_images
    )
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{vision_block}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Load requests from JSONL file
# ---------------------------------------------------------------------------
def load_requests_from_file(
    path: str, delay_override: int | None
) -> list[dict]:
    """Load requests from a JSONL file.

    Each line is a JSON object with:
      - ``text`` (required): the user question.
      - ``images`` (optional): a single path string or a list of path strings.
        An empty list ``[]`` is treated the same as omitting the field.
      - ``delay`` (optional, default 0): milliseconds to wait before
        submitting this request.

    All paths are relative to the current working directory.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    examples: list[dict] = []
    with open(filepath) as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {filepath}: {exc}"
                ) from exc

            # --- text (required) ---
            text = obj.get("text")
            if not text:
                raise ValueError(
                    f"Missing 'text' field on line {line_num} of {filepath}"
                )

            # --- images (optional) ---
            raw_images = obj.get("images")
            image_paths: list[str] = []
            if isinstance(raw_images, str):
                image_paths = [raw_images]
            elif isinstance(raw_images, list):
                image_paths = [p for p in raw_images if p]  # filter empties

            pil_images: list[Image.Image] = []
            for img_path in image_paths:
                p = Path(img_path)
                if not p.exists():
                    raise FileNotFoundError(
                        f"Image not found: {p} "
                        f"(line {line_num} of {filepath})"
                    )
                pil_images.append(convert_image_mode(Image.open(p), "RGB"))

            # --- delay ---
            delay = (
                delay_override
                if delay_override is not None
                else obj.get("delay", 0)
            )

            # --- Build prompt and multi_modal_data ---
            num_imgs = len(pil_images)
            prompt = build_prompt(text, num_imgs)

            mm_data: dict = {}
            if num_imgs == 1:
                mm_data["image"] = pil_images[0]
            elif num_imgs > 1:
                mm_data["image"] = pil_images

            image_source = (
                ", ".join(image_paths) if image_paths else "(text-only)"
            )

            examples.append(
                {
                    "image_source": image_source,
                    "question": text,
                    "delay": delay,
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "num_images": num_imgs,
                }
            )

    return examples


# ---------------------------------------------------------------------------
# Built-in example inputs (used when --input is not provided)
# ---------------------------------------------------------------------------
def build_example_inputs() -> list[dict]:
    """Return a list of example request dicts using vLLM built-in assets."""

    cherry = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    stop = convert_image_mode(ImageAsset("stop_sign").pil_image, "RGB")

    raw = [
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

    examples: list[dict] = []
    for r in raw:
        examples.append(
            {
                "image_source": r["image_source"],
                "question": r["question"],
                "delay": 0,
                "prompt": build_prompt(r["question"], num_images=1),
                "multi_modal_data": {"image": r["image"]},
                "num_images": 1,
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
        worker_stats_list = llm.llm_engine.collective_rpc(
            "get_encoder_timing_stats"
        )
    except Exception as exc:
        print(f"[WARN] Could not retrieve encoder timing stats: {exc}")
        return {}

    result: dict[str, float] = {}
    for worker_stats in worker_stats_list:
        if not worker_stats:
            continue
        for internal_id, stats_dict in worker_stats.items():
            external_id = internal_id.rsplit("-", 1)[0]
            elapsed = stats_dict.get("encoder_forward_secs", 0.0)
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
        "--input",
        type=str,
        default=None,
        help="Path to a JSONL file with requests. If not provided, "
        "built-in example inputs are used.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=None,
        help="Global delay override in ms. Overrides per-request delay "
        "values from the JSONL file.",
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
    parser.add_argument(
        "--disagg",
        action="store_true",
        default=False,
        help="Enable PD disaggregation: vision encoder + prefill on GPU 0, "
        "decode on GPU 1. Requires 2 GPUs.",
    )
    args = parser.parse_args()

    # Ensure output directory exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build examples -- either from file or built-in.
    if args.input:
        examples = load_requests_from_file(args.input, args.delay)
    else:
        examples = build_example_inputs()
        # Apply --delay override to built-in examples too.
        if args.delay is not None:
            for ex in examples:
                ex["delay"] = args.delay

    print(f"Prepared {len(examples)} requests.")

    if args.disagg:
        run_disaggregated(args, examples)
        return

    run_single_gpu(args, examples)


# ---------------------------------------------------------------------------
# Single-GPU inference
# ---------------------------------------------------------------------------
def run_single_gpu(args, examples: list[dict]):
    """Run all requests on a single GPU (default mode)."""
    max_images = max(
        (ex.get("num_images", 0) for ex in examples), default=1
    )
    max_images = max(max_images, 1)

    print(f"Loading model: {args.model} ...")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        limit_mm_per_prompt={"image": max_images},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        disable_log_stats=False,
        seed=42,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    vllm_inputs = _build_vllm_inputs(examples)
    has_delays = any(ex.get("delay", 0) > 0 for ex in examples)

    print("Running inference ...")
    if has_delays:
        for ex, inp in zip(examples, vllm_inputs):
            delay_ms = ex.get("delay", 0)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            llm.enqueue(inp, sampling_params=sampling_params)
        outputs = llm.wait_for_completion()
    else:
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

    vision_times = get_vision_encoder_times(llm)
    print(
        f"Retrieved vision encoder timings for {len(vision_times)} request(s)."
    )

    _report_results(args, examples, outputs, vision_times)


# ---------------------------------------------------------------------------
# Disaggregated PD mode
# ---------------------------------------------------------------------------
def _build_vllm_inputs(examples: list[dict]) -> list[dict]:
    """Convert example dicts to vLLM input dicts."""
    vllm_inputs: list[dict] = []
    for ex in examples:
        inp: dict = {"prompt": ex["prompt"]}
        if ex["multi_modal_data"]:
            inp["multi_modal_data"] = ex["multi_modal_data"]
        vllm_inputs.append(inp)
    return vllm_inputs


def _common_llm_kwargs(args, examples: list[dict]) -> dict:
    """Return LLM constructor kwargs shared by single-GPU and disagg modes."""
    max_images = max(
        (ex.get("num_images", 0) for ex in examples), default=1
    )
    max_images = max(max_images, 1)
    return dict(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        limit_mm_per_prompt={"image": max_images},
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        disable_log_stats=False,
        seed=42,
    )


def _run_prefill(args, examples: list[dict], prefill_done):
    """Prefill process: vision encoder + prefill on GPU 0."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Write iteration logs to a prefill-specific subdirectory so the
    # decode process does not overwrite them.
    iter_log_dir = os.environ.get("VLLM_ITERATION_LOG_DIR", ".")
    prefill_log_dir = os.path.join(iter_log_dir, "prefill")
    os.environ["VLLM_ITERATION_LOG_DIR"] = prefill_log_dir

    from vllm.config import KVTransferConfig

    llm_kwargs = _common_llm_kwargs(args, examples)
    llm_kwargs["enforce_eager"] = True
    llm_kwargs["kv_transfer_config"] = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
    )

    print("[Prefill/GPU0] Loading model ...")
    llm = LLM(**llm_kwargs)

    vllm_inputs = _build_vllm_inputs(examples)
    # Prefill generates only 1 token — KV cache is transferred to decode.
    prefill_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=1,
    )

    print("[Prefill/GPU0] Running prefill ...")
    outputs = llm.generate(vllm_inputs, sampling_params=prefill_params)

    # Save prompts + first token for decode process.
    new_prompts = []
    for output in outputs:
        first_tok = output.outputs[0].text if output.outputs else ""
        new_prompts.append(output.prompt + first_tok)

    tmp_file = OUTPUT_DIR / "_disagg_prompts.json"
    with open(tmp_file, "w") as f:
        json.dump(new_prompts, f)

    # Collect prefill-side metrics.
    vision_times = get_vision_encoder_times(llm)
    prefill_metrics = []
    for i, output in enumerate(outputs):
        m = extract_request_metrics(output, vision_times)
        prefill_metrics.append(m)

    metrics_file = OUTPUT_DIR / "_disagg_prefill_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(prefill_metrics, f)

    print("[Prefill/GPU0] Done. Signaling decode ...")
    prefill_done.set()

    # Stay alive until decode finishes (NCCL needs both endpoints).
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def _run_decode(args, examples: list[dict], prefill_done):
    """Decode process: text decode on GPU 1."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Write iteration logs to a decode-specific subdirectory.
    iter_log_dir = os.environ.get("VLLM_ITERATION_LOG_DIR", ".")
    decode_log_dir = os.path.join(iter_log_dir, "decode")
    os.environ["VLLM_ITERATION_LOG_DIR"] = decode_log_dir

    from vllm.config import KVTransferConfig

    llm_kwargs = _common_llm_kwargs(args, examples)
    llm_kwargs["enforce_eager"] = True
    llm_kwargs["kv_transfer_config"] = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
    )

    print("[Decode/GPU1] Loading model ...")
    llm = LLM(**llm_kwargs)

    print("[Decode/GPU1] Waiting for prefill to finish ...")
    prefill_done.wait()

    tmp_file = OUTPUT_DIR / "_disagg_prompts.json"
    with open(tmp_file) as f:
        prompts = json.load(f)

    # Build text-only inputs (no images — vision encoding already done).
    decode_inputs = [{"prompt": p} for p in prompts]

    decode_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("[Decode/GPU1] Running decode ...")
    outputs = llm.generate(decode_inputs, sampling_params=decode_params)

    vision_times = get_vision_encoder_times(llm)

    # Merge prefill metrics.
    metrics_file = OUTPUT_DIR / "_disagg_prefill_metrics.json"
    prefill_metrics = []
    if metrics_file.exists():
        with open(metrics_file) as f:
            prefill_metrics = json.load(f)

    _report_results(
        args,
        examples,
        outputs,
        vision_times,
        prefill_metrics=prefill_metrics,
        mode="disagg",
    )

    # Clean up temp files.
    tmp_file.unlink(missing_ok=True)
    metrics_file.unlink(missing_ok=True)


def run_disaggregated(args, examples: list[dict]):
    """Launch prefill and decode processes on separate GPUs."""
    print("=" * 60)
    print("PD Disaggregated Mode: GPU 0 = prefill, GPU 1 = decode")
    print("=" * 60)

    prefill_done = Event()
    prefill_proc = Process(
        target=_run_prefill, args=(args, examples, prefill_done)
    )
    decode_proc = Process(
        target=_run_decode, args=(args, examples, prefill_done)
    )

    prefill_proc.start()
    decode_proc.start()

    decode_proc.join()
    # Decode finished — terminate prefill (it's waiting in a sleep loop).
    prefill_proc.terminate()
    prefill_proc.join(timeout=5)
    print("\nDisaggregated inference complete.")


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------
def _report_results(
    args,
    examples: list[dict],
    outputs,
    vision_times: dict[str, float],
    *,
    prefill_metrics: list[dict] | None = None,
    mode: str = "single",
):
    """Collect metrics, write JSON, and print summary table."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    request_results = []

    for i, output in enumerate(outputs):
        metrics = extract_request_metrics(output, vision_times)

        # In disagg mode, merge prefill-side VE + prefill timing.
        if prefill_metrics and i < len(prefill_metrics):
            pm = prefill_metrics[i]
            if pm.get("vision_encoder_time_ms", 0) > 0:
                metrics["vision_encoder_time_ms"] = pm[
                    "vision_encoder_time_ms"
                ]
                metrics["vision_encoder_time_s"] = pm[
                    "vision_encoder_time_s"
                ]
            if pm.get("prefill_time_ms", 0) > 0:
                metrics["prefill_time_ms"] = pm["prefill_time_ms"]
                metrics["prefill_time_s"] = pm["prefill_time_s"]

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

    n = len(request_results)
    summary = {}
    if n > 0:
        ve_times_actual = [
            r["vision_encoder_time_ms"]
            for r in request_results
            if r["vision_encoder_time_ms"] > 0
        ]
        prefill_times = [
            r.get("prefill_time_ms", 0.0) for r in request_results
        ]
        decode_times = [
            r.get("decode_time_ms", 0.0) for r in request_results
        ]
        tpots = [r.get("tpot_ms", 0.0) for r in request_results]
        total_decode_tokens = sum(
            r.get("num_generation_tokens", 0) for r in request_results
        )

        num_ve_runs = len(ve_times_actual)
        summary = {
            "mode": mode,
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

    final_output = {
        "model": args.model,
        "mode": mode,
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

    output_file = OUTPUT_DIR / f"latency_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to: {output_file}")

    # Print summary table.
    mode_label = "DISAGG (prefill GPU0 + decode GPU1)" if mode == "disagg" else "SINGLE GPU"
    print("\n" + "=" * 80)
    print(f"{'LATENCY MEASUREMENT SUMMARY':^80}")
    print(f"Model: {args.model}  |  Mode: {mode_label}")
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
