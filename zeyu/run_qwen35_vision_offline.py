#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Offline inference script for a Qwen vision-language model with per-request
latency measurement.

Supported modes (via ``--role``):
  - ``none`` (default): single-GPU single-process inference.
  - ``prefill``: run only vision encoder + prefill on ONE GPU. Exposes
    a P2pNcclConnector kv_producer so a peer decode process (on the same
    or a different node) consumes the KV cache.
  - ``decode``: run only decode on ONE GPU. P2pNcclConnector kv_consumer
    receives KV cache from a peer prefill process.

Cross-node usage relies on the management NIC (e.g. ``eth0``). Set the
peer IP with ``--peer-ip`` (required for ``--role prefill`` and
``--role decode``). All paths are relative to the current working
directory.

Per-request metrics collected:
  * Vision encoder time, prefill time (prefill role)
  * Decode time, number of generation tokens, TPOT (decode role)
  * KV transfer time (prefill→decode handoff, estimated)
  * JCT (job completion time: overall prefill-start → decode-end)

Results are written to ``zeyu/outputs/latency_<timestamp>.json``.
When ``--role`` is set, the output directory is augmented with a role
subdirectory (``prefill/`` or ``decode/``) for iteration logs.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
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
DEFAULT_MODEL = "/home/zeyu/models/Qwen3-VL-8B-Instruct"

# Control-channel ZMQ port (separate from KV NCCL port).
# Used for cross-node coordination (ready signals, JCT sync).
DEFAULT_CTRL_PORT = 25500

# Default KV transfer NCCL port (used by P2pNcclConnector).
DEFAULT_KV_PORT = 25555


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(question: str, num_images: int) -> str:
    """Build a Qwen VL chat-template prompt."""
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
    """Load requests from a JSONL file."""
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

            text = obj.get("text")
            if not text:
                raise ValueError(
                    f"Missing 'text' field on line {line_num} of {filepath}"
                )

            raw_images = obj.get("images")
            image_paths: list[str] = []
            if isinstance(raw_images, str):
                image_paths = [raw_images]
            elif isinstance(raw_images, list):
                image_paths = [p for p in raw_images if p]

            pil_images: list[Image.Image] = []
            for img_path in image_paths:
                p = Path(img_path)
                if not p.exists():
                    raise FileNotFoundError(
                        f"Image not found: {p} "
                        f"(line {line_num} of {filepath})"
                    )
                pil_images.append(convert_image_mode(Image.open(p), "RGB"))

            delay = (
                delay_override
                if delay_override is not None
                else obj.get("delay", 0)
            )

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
def build_example_inputs(num: int = 4) -> list[dict]:
    """Return a list of example request dicts using vLLM built-in assets.

    If ``num > 4``, cycle through the 4 base questions to reach ``num``.
    """
    cherry = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    stop = convert_image_mode(ImageAsset("stop_sign").pil_image, "RGB")

    base = [
        ("cherry_blossom (built-in)", cherry, "What is the content of this image?"),
        ("cherry_blossom (built-in)", cherry, "Describe this image in detail."),
        ("stop_sign (built-in)", stop, "What sign is shown in this image and what does it mean?"),
        ("stop_sign (built-in)", stop, "List all objects you can see in this image."),
    ]

    examples: list[dict] = []
    for i in range(num):
        src, img, q = base[i % len(base)]
        examples.append(
            {
                "image_source": src,
                "question": q,
                "delay": 0,
                "prompt": build_prompt(q, num_images=1),
                "multi_modal_data": {"image": img},
                "num_images": 1,
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Retrieve vision encoder timings via collective_rpc
# ---------------------------------------------------------------------------
def get_vision_encoder_times(llm: LLM) -> dict[str, float]:
    """Retrieve per-request vision encoder timing from the model runner."""
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
    stats = output.metrics
    result: dict = {}

    if stats is not None:
        prefill_time = stats.first_token_ts - stats.scheduled_ts
        decode_time = stats.last_token_ts - stats.first_token_ts
        num_gen = stats.num_generation_tokens
        tpot = decode_time / (num_gen - 1) if num_gen > 1 else 0.0
        arrival_ts = getattr(stats, "arrival_time", 0.0) or 0.0
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
        # Expose raw timestamps for cross-phase JCT / KV transfer calc.
        result["arrival_ts"] = round(arrival_ts, 6)
        result["scheduled_ts"] = round(stats.scheduled_ts, 6)
        result["first_token_ts"] = round(stats.first_token_ts, 6)
        result["last_token_ts"] = round(stats.last_token_ts, 6)
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
        description="Qwen VL offline inference with latency measurement"
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
        "--num-prompts",
        type=int,
        default=4,
        help="Number of built-in example prompts to cycle through "
        "(ignored when --input is provided). Default: 4.",
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
        "--gpu",
        type=int,
        default=0,
        help="Index of the physical GPU to use in this process "
        "(sets CUDA_VISIBLE_DEVICES). Default: 0.",
    )

    # --- Disaggregation flags ---
    parser.add_argument(
        "--role",
        choices=["none", "prefill", "decode"],
        default="none",
        help="Role for this process: "
        "'none' = single-GPU (default); "
        "'prefill' = run vision encoder + prefill, kv_producer; "
        "'decode' = run decode only, kv_consumer.",
    )
    parser.add_argument(
        "--peer-ip",
        type=str,
        default=None,
        help="IP address of the peer node/process. "
        "For --role prefill: IP of the decode process. "
        "For --role decode: IP of the prefill process. "
        "For same-node test, use 127.0.0.1.",
    )
    parser.add_argument(
        "--kv-port",
        type=int,
        default=DEFAULT_KV_PORT,
        help=f"P2pNcclConnector KV transfer port (default: {DEFAULT_KV_PORT}).",
    )
    parser.add_argument(
        "--ctrl-port",
        type=int,
        default=DEFAULT_CTRL_PORT,
        help=f"Control-channel ZMQ port for coordination "
        f"(default: {DEFAULT_CTRL_PORT}).",
    )
    parser.add_argument(
        "--iface",
        type=str,
        default="eth0",
        help="Network interface for NCCL/GLOO (sets NCCL_SOCKET_IFNAME). "
        "Default: eth0.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: zeyu/outputs). "
        "Used by disagg launcher to keep role logs together.",
    )

    # Deprecated flag kept for backward compatibility.
    parser.add_argument(
        "--disagg",
        action="store_true",
        default=False,
        help="[Deprecated] Single-node 2-GPU disagg via multiprocessing. "
        "Prefer --role prefill / --role decode for cross-node.",
    )
    args = parser.parse_args()

    # Set up output dir.
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build examples -- either from file or built-in.
    if args.input:
        examples = load_requests_from_file(args.input, args.delay)
    else:
        examples = build_example_inputs(num=args.num_prompts)
        if args.delay is not None:
            for ex in examples:
                ex["delay"] = args.delay

    print(f"Prepared {len(examples)} requests.")

    # Dispatch based on role.
    if args.role == "prefill":
        run_prefill_role(args, examples)
    elif args.role == "decode":
        run_decode_role(args, examples)
    elif args.disagg:
        # Legacy same-node multiprocessing path.
        run_legacy_same_node_disagg(args, examples)
    else:
        run_single_gpu(args, examples)


# ---------------------------------------------------------------------------
# Single-GPU inference (unchanged)
# ---------------------------------------------------------------------------
def run_single_gpu(args, examples: list[dict]):
    """Run all requests on a single GPU (default mode)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Loading model: {args.model} ...")
    llm = LLM(**_common_llm_kwargs(args, examples))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    vllm_inputs = _build_vllm_inputs(examples)
    has_delays = any(ex.get("delay", 0) > 0 for ex in examples)

    print("Running inference ...")
    t_start = time.time()
    if has_delays:
        for ex, inp in zip(examples, vllm_inputs):
            delay_ms = ex.get("delay", 0)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            llm.enqueue(inp, sampling_params=sampling_params)
        outputs = llm.wait_for_completion()
    else:
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
    t_end = time.time()

    vision_times = get_vision_encoder_times(llm)
    print(
        f"Retrieved vision encoder timings for {len(vision_times)} request(s)."
    )

    _report_results(
        args,
        examples,
        outputs,
        vision_times,
        wall_time_s=t_end - t_start,
    )


# ---------------------------------------------------------------------------
# Shared helpers
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
    """Return LLM constructor kwargs shared by all modes."""
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


def _setup_network_env(iface: str):
    """Force NCCL/GLOO to use the specified network interface."""
    os.environ.setdefault("NCCL_SOCKET_IFNAME", iface)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
    # Disable InfiniBand — we're using TCP over the management NIC.
    os.environ.setdefault("NCCL_IB_DISABLE", "1")


def _local_ip_for(iface: str) -> str:
    """Get the IPv4 address of the given interface. Falls back to
    a UDP-based heuristic if ``ip`` command is not available."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", iface],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "inet" and i + 1 < len(parts):
                    return parts[i + 1].split("/")[0]
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Disaggregation: prefill role (cross-node capable)
# ---------------------------------------------------------------------------
def run_prefill_role(args, examples: list[dict]):
    """Run vision encoder + prefill as a kv_producer."""
    if not args.peer_ip:
        raise ValueError(
            "--peer-ip is required for --role prefill "
            "(the decode process's IP)."
        )

    # P2pNcclConnector relies on request_ids carrying both endpoints'
    # ZMQ addresses. Disable vLLM's random suffixing so our IDs are
    # identical on prefill and decode sides (required for the KV
    # tensor key to match on both sides).
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _setup_network_env(args.iface)

    # Route iteration logs into prefill/ subdir.
    log_dir = Path(os.environ.get("VLLM_ITERATION_LOG_DIR", str(OUTPUT_DIR)))
    prefill_log_dir = log_dir / "prefill"
    os.environ["VLLM_ITERATION_LOG_DIR"] = str(prefill_log_dir)

    local_ip = _local_ip_for(args.iface)
    prefill_kv_port = args.kv_port
    decode_kv_port = args.kv_port + 100  # decode uses a different port
    print(
        f"[Prefill] Role=kv_producer  local_ip={local_ip}  "
        f"peer_ip={args.peer_ip}  "
        f"prefill_kv_port={prefill_kv_port}  decode_kv_port={decode_kv_port}  "
        f"iface={args.iface}"
    )

    # Build LLM with P2pNcclConnector kv_producer.
    from vllm.config import KVTransferConfig

    llm_kwargs = _common_llm_kwargs(args, examples)
    llm_kwargs["enforce_eager"] = True
    llm_kwargs["kv_transfer_config"] = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_buffer_size=1e9,
        kv_ip=local_ip,
        kv_port=str(prefill_kv_port),
        kv_connector_extra_config={
            "send_type": "PUT_ASYNC",
            "nccl_num_channels": "8",
        },
    )

    print("[Prefill] Loading model ...")
    llm = LLM(**llm_kwargs)

    # Sync with decode peer: exchange addresses and request IDs.
    _sync_ready(
        args,
        role="prefill",
        local_ip=local_ip,
        local_kv_port=prefill_kv_port,
    )
    peer_info = args._peer_info
    decode_ip = peer_info["ip"]

    # Build request_ids that encode both endpoints' addresses.
    req_ids = _build_disagg_request_ids(
        num=len(examples),
        prefill_ip=local_ip,
        prefill_port=prefill_kv_port,
        decode_ip=decode_ip,
        decode_port=decode_kv_port,
    )
    # Send the generated request IDs to decode side so it uses the same IDs.
    args._ctrl_sock.send_json({"op": "req_ids", "ids": req_ids})

    vllm_inputs = _build_vllm_inputs(examples)
    prefill_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=1,
    )

    print(f"[Prefill] Running prefill for {len(vllm_inputs)} requests ...")
    wall_start = time.time()
    outputs = _generate_with_request_ids(
        llm, vllm_inputs, prefill_params, req_ids
    )
    wall_end = time.time()

    vision_times = get_vision_encoder_times(llm)

    prefill_metrics = []
    for i, output in enumerate(outputs):
        m = extract_request_metrics(output, vision_times)
        m["request_id"] = output.request_id
        m["request_index"] = i
        m["image_source"] = examples[i]["image_source"]
        m["question"] = examples[i]["question"]
        prefill_metrics.append(m)

    # Write prefill latency JSON.
    out_file = prefill_log_dir / "latency.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(
            {
                "role": "prefill",
                "model": args.model,
                "wall_time_s": round(wall_end - wall_start, 6),
                "num_requests": len(prefill_metrics),
                "requests": prefill_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Prefill] Wrote {out_file}")

    # Signal decode that prefill is done (and share timing).
    _signal_done(args, role="prefill", payload={
        "wall_end": wall_end,
        "num_requests": len(prefill_metrics),
    })

    # Keep the prefill process alive briefly so NCCL transfers finish.
    print("[Prefill] Holding NCCL endpoint open until decode signals exit ...")
    _wait_for_peer_exit(args, role="prefill")
    print("[Prefill] Done.")


# ---------------------------------------------------------------------------
# Disaggregation: decode role
# ---------------------------------------------------------------------------
def run_decode_role(args, examples: list[dict]):
    """Run decode as a kv_consumer."""
    if not args.peer_ip:
        raise ValueError(
            "--peer-ip is required for --role decode "
            "(the prefill process's IP)."
        )

    # See explanation in run_prefill_role.
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _setup_network_env(args.iface)

    log_dir = Path(os.environ.get("VLLM_ITERATION_LOG_DIR", str(OUTPUT_DIR)))
    decode_log_dir = log_dir / "decode"
    os.environ["VLLM_ITERATION_LOG_DIR"] = str(decode_log_dir)

    local_ip = _local_ip_for(args.iface)
    prefill_kv_port = args.kv_port
    decode_kv_port = args.kv_port + 100
    print(
        f"[Decode] Role=kv_consumer  local_ip={local_ip}  "
        f"peer_ip={args.peer_ip}  "
        f"prefill_kv_port={prefill_kv_port}  decode_kv_port={decode_kv_port}  "
        f"iface={args.iface}"
    )

    from vllm.config import KVTransferConfig

    llm_kwargs = _common_llm_kwargs(args, examples)
    llm_kwargs["enforce_eager"] = True
    llm_kwargs["kv_transfer_config"] = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_buffer_size=8e9,
        kv_ip=local_ip,
        kv_port=str(decode_kv_port),
        kv_connector_extra_config={
            "send_type": "PUT_ASYNC",
            "nccl_num_channels": "8",
        },
    )

    print("[Decode] Loading model ...")
    llm = LLM(**llm_kwargs)

    # Exchange addresses with prefill side.
    _sync_ready(
        args,
        role="decode",
        local_ip=local_ip,
        local_kv_port=decode_kv_port,
    )
    # Receive the request IDs chosen by prefill side.
    msg = args._ctrl_sock.recv_json()
    assert msg.get("op") == "req_ids", f"unexpected msg: {msg}"
    req_ids: list[str] = msg["ids"]

    print("[Decode] Waiting for prefill to complete ...")
    prefill_info = _wait_for_done(args, role="decode")
    prefill_done_ts = prefill_info.get("wall_end", time.time())

    # Run full inference (max_tokens=N). P2pNcclConnector on the consumer
    # side will skip prefill if the producer has already produced KV.
    vllm_inputs = _build_vllm_inputs(examples)
    decode_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"[Decode] Running decode for {len(vllm_inputs)} requests ...")
    wall_start = time.time()
    outputs = _generate_with_request_ids(
        llm, vllm_inputs, decode_params, req_ids
    )
    wall_end = time.time()

    vision_times = get_vision_encoder_times(llm)

    # KV transfer time estimate: scheduled_ts - prefill_done_ts on decode
    # side. If decode's scheduled_ts < prefill_done_ts (different clocks),
    # fall back to min(0, diff).
    decode_metrics = []
    for i, output in enumerate(outputs):
        m = extract_request_metrics(output, vision_times)
        m["request_id"] = output.request_id
        m["request_index"] = i
        m["image_source"] = examples[i]["image_source"]
        m["question"] = examples[i]["question"]
        m["generated_text"] = (
            output.outputs[0].text if output.outputs else ""
        )

        # Compute KV transfer wall-clock estimate from peer's completion
        # timestamp to our scheduled_ts. Clocks may differ; report both
        # raw and clamped.
        sched_ts = m.get("scheduled_ts", 0.0)
        if sched_ts > 0:
            raw_kv = sched_ts - prefill_done_ts
            m["kv_transfer_time_s"] = round(raw_kv, 6)
            m["kv_transfer_time_ms"] = round(raw_kv * 1000, 3)
        decode_metrics.append(m)

    # Write decode latency JSON.
    out_file = decode_log_dir / "latency.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(
            {
                "role": "decode",
                "model": args.model,
                "wall_time_s": round(wall_end - wall_start, 6),
                "wall_start": wall_start,
                "wall_end": wall_end,
                "num_requests": len(decode_metrics),
                "requests": decode_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Decode] Wrote {out_file}")

    # Signal exit to prefill so it tears down NCCL.
    _signal_exit(args, role="decode", payload={"wall_end": wall_end})
    print("[Decode] Done.")


# ---------------------------------------------------------------------------
# Cross-node coordination (ZMQ PAIR sockets over eth0)
# ---------------------------------------------------------------------------
def _sync_ready(args, role: str, local_ip: str, local_kv_port: int):
    """Prefill binds, decode connects. Both exchange addresses and
    ensure the peer is ready."""
    import zmq

    ctx = zmq.Context.instance()

    if role == "prefill":
        sock = ctx.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://0.0.0.0:{args.ctrl_port}")
        print(
            f"[Prefill] Ctrl bound on 0.0.0.0:{args.ctrl_port}, "
            f"waiting for decode READY ..."
        )
        msg = sock.recv_json()
        assert msg.get("op") == "ready", f"unexpected msg: {msg}"
        peer_info = {
            "ip": msg.get("ip"),
            "kv_port": msg.get("kv_port"),
        }
        sock.send_json({
            "op": "ack",
            "ts": time.time(),
            "ip": local_ip,
            "kv_port": local_kv_port,
        })
        args._ctrl_sock = sock
        args._peer_info = peer_info
    else:  # decode
        sock = ctx.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        url = f"tcp://{args.peer_ip}:{args.ctrl_port}"
        sock.connect(url)
        print(f"[Decode] Ctrl connected to {url}, sending READY ...")
        sock.send_json({
            "op": "ready",
            "ts": time.time(),
            "ip": local_ip,
            "kv_port": local_kv_port,
        })
        msg = sock.recv_json()
        assert msg.get("op") == "ack", f"unexpected msg: {msg}"
        args._ctrl_sock = sock
        args._peer_info = {
            "ip": msg.get("ip"),
            "kv_port": msg.get("kv_port"),
        }


# ---------------------------------------------------------------------------
# Request ID helpers for P2pNcclConnector
# ---------------------------------------------------------------------------
def _build_disagg_request_ids(
    num: int,
    prefill_ip: str,
    prefill_port: int,
    decode_ip: str,
    decode_port: int,
) -> list[str]:
    """Build request IDs that encode both endpoints' ZMQ addresses.

    P2pNcclConnector parses these to route KV tensors between prefill
    and decode. The format must match the regex in
    ``P2pNcclConnector.parse_request_id``:
      * prefill side:  ``___decode_addr_<ip>:<port>``
      * decode side:   ``___prefill_addr_<ip>:<port>___``

    We put both markers in a single shared ID so prefill and decode can
    use identical IDs (required for tensor key matching).
    """
    import uuid

    ids = []
    for i in range(num):
        uid = uuid.uuid4().hex[:12]
        rid = (
            f"req{i}"
            f"___prefill_addr_{prefill_ip}:{prefill_port}"
            f"___decode_addr_{decode_ip}:{decode_port}"
            f"_{uid}"
        )
        ids.append(rid)
    return ids


def _generate_with_request_ids(
    llm: LLM,
    vllm_inputs: list[dict],
    sampling_params: SamplingParams,
    request_ids: list[str],
):
    """Submit requests with caller-provided request IDs and return
    outputs in the same order as request_ids."""
    from vllm.outputs import RequestOutput

    assert len(vllm_inputs) == len(request_ids), (
        f"Got {len(vllm_inputs)} inputs and {len(request_ids)} IDs"
    )
    for inp, rid in zip(vllm_inputs, request_ids):
        llm.llm_engine.add_request(rid, inp, sampling_params)

    # Drive the engine until all requests finish.
    outputs: list[RequestOutput] = llm._run_engine(
        RequestOutput, use_tqdm=True
    )
    # _run_engine returns outputs sorted by request_id (string sort), so
    # re-sort by our injected order.
    by_id = {o.request_id: o for o in outputs}
    return [by_id[r] for r in request_ids if r in by_id]


def _signal_done(args, role: str, payload: dict):
    """Prefill → Decode: prefill has finished, send timing payload."""
    assert role == "prefill"
    args._ctrl_sock.send_json({"op": "prefill_done", **payload})


def _wait_for_done(args, role: str) -> dict:
    """Decode waits for prefill's done signal."""
    assert role == "decode"
    msg = args._ctrl_sock.recv_json()
    assert msg.get("op") == "prefill_done"
    return msg


def _signal_exit(args, role: str, payload: dict):
    """Decode → Prefill: decode finished, OK to exit."""
    assert role == "decode"
    args._ctrl_sock.send_json({"op": "decode_done", **payload})


def _wait_for_peer_exit(args, role: str):
    """Prefill waits for decode's exit signal."""
    assert role == "prefill"
    msg = args._ctrl_sock.recv_json()
    assert msg.get("op") == "decode_done"


# ---------------------------------------------------------------------------
# Legacy same-node disagg via multiprocessing (kept for backward compat)
# ---------------------------------------------------------------------------
def run_legacy_same_node_disagg(args, examples: list[dict]):
    """Deprecated: same-node 2-GPU via multiprocessing.

    NOTE: only works for non-hybrid models. For cross-node, use
    --role prefill / --role decode with disagg_run.sh.
    """
    print("=" * 60)
    print("[Legacy] Same-node 2-GPU disagg via multiprocessing.")
    print("         For cross-node, use disagg_run.sh instead.")
    print("=" * 60)

    prefill_done = Event()

    def _prefill_entry():
        args.gpu = 0
        args.role = "prefill"
        args.peer_ip = "127.0.0.1"
        run_prefill_role(args, examples)

    def _decode_entry():
        args.gpu = 1
        args.role = "decode"
        args.peer_ip = "127.0.0.1"
        run_decode_role(args, examples)

    p = Process(target=_prefill_entry)
    d = Process(target=_decode_entry)
    p.start()
    d.start()
    d.join()
    p.join(timeout=15)


# ---------------------------------------------------------------------------
# Shared reporting (single-GPU)
# ---------------------------------------------------------------------------
def _report_results(
    args,
    examples: list[dict],
    outputs,
    vision_times: dict[str, float],
    *,
    wall_time_s: float | None = None,
    mode: str = "single",
):
    """Collect metrics, write JSON, and print summary table."""
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
        if wall_time_s is not None and wall_time_s > 0:
            summary["wall_time_s"] = round(wall_time_s, 3)
            summary["rps"] = round(n / wall_time_s, 3)

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

    mode_label = "SINGLE GPU"
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
        avg_gen_tokens = (
            summary["total_decode_tokens"] / summary["num_requests"]
        )
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
        if "rps" in summary:
            print(
                f"  * wall_time = {summary['wall_time_s']:.3f}s  "
                f"RPS = {summary['rps']:.3f}"
            )
        print(f"  * VE avg computed over actual encoder runs only {ve_note}")
    print("=" * 80)


if __name__ == "__main__":
    main()
