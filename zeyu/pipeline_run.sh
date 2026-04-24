#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Single-GPU runner for the MM pipeline feature (--mm-pipeline {off,on})
# with optional nsys wrapping.
#
# What this script does vs just running run_qwen35_vision_offline.py:
#   * Sets VLLM_LOG_ITERATIONS=1 + VLLM_ITERATION_LOG_DIR for per-iter
#     JSONL logs.
#   * When --nsys is passed:
#       - forces --enforce-eager (no CUDA graphs) + sets
#         VLLM_ENABLE_V1_MULTIPROCESSING=0 (single-process engine);
#         both are required for nsys to see per-kernel GPU activity
#         on the current driver + CUPTI combo.
#       - wraps python in `nsys profile --trace=cuda,nvtx`.
#       - exports cuda_gpu_trace and nvtx_pushpop_trace CSVs.
#       - runs analyze_profile.py to emit consolidated_iterations.jsonl
#         including nvtx_overlap_ns.
#
# Usage:
#   bash zeyu/pipeline_run.sh \
#       --mm-pipeline on \
#       --num-prompts 4 --max-tokens 32 \
#       --gpu 7 \
#       --nsys
#
# Flags:
#   --mm-pipeline {off,on}    default off
#   --num-prompts N           default 4
#   --max-tokens N            default 32
#   --max-num-seqs N          default 5
#   --gpu IDX                 default 0 (CUDA_VISIBLE_DEVICES)
#   --model PATH              default /home/zeyu/models/Qwen3-VL-8B-Instruct
#   --output-dir DIR          default zeyu/outputs/pipeline_<UTC-timestamp>
#   --nsys                    wrap in nsys profile (+ force eager/no-mp)
#   --no-nsys                 explicit opt-out (default)
#   --nsys-bin PATH           default 'nsys' (auto-detects /usr/local/bin/nsys
#                             or /opt/nvidia/nsight-systems/*/bin/nsys)

set -euo pipefail

# --- defaults ---
MM_PIPELINE=off
NUM_PROMPTS=4
MAX_TOKENS=32
MAX_NUM_SEQS=5
GPU=0
MODEL=/home/zeyu/models/Qwen3-VL-8B-Instruct
OUTPUT_DIR=""
ENABLE_NSYS=false
NSYS_BIN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mm-pipeline) MM_PIPELINE="$2"; shift 2;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2;;
        --max-tokens) MAX_TOKENS="$2"; shift 2;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --nsys) ENABLE_NSYS=true; shift;;
        --no-nsys) ENABLE_NSYS=false; shift;;
        --nsys-bin) NSYS_BIN="$2"; shift 2;;
        -h|--help) sed -n '1,40p' "$0"; exit 0;;
        *) echo "Unknown flag: $1" >&2; exit 1;;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

if [[ -z "$OUTPUT_DIR" ]]; then
    TS=$(date -u +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$SCRIPT_DIR/outputs/pipeline_${MM_PIPELINE}_${TS}"
fi
mkdir -p "$OUTPUT_DIR"

# --- locate nsys when requested ---
if $ENABLE_NSYS; then
    if [[ -z "$NSYS_BIN" ]]; then
        for cand in nsys /usr/local/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys; do
            if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
                NSYS_BIN="$cand"
                break
            fi
        done
    fi
    if [[ -z "$NSYS_BIN" ]] || ! command -v "$NSYS_BIN" >/dev/null 2>&1 && [[ ! -x "$NSYS_BIN" ]]; then
        echo "ERROR: nsys not found. Install nsight-systems or pass --nsys-bin PATH." >&2
        exit 2
    fi
    echo "[pipeline_run] Using nsys: $NSYS_BIN"
    echo "[pipeline_run] NOTE: --nsys forces --enforce-eager + VLLM_ENABLE_V1_MULTIPROCESSING=0"
fi

echo "============================================================"
echo "  Pipeline-mode single-GPU run"
echo "============================================================"
echo "  --mm-pipeline : $MM_PIPELINE"
echo "  --num-prompts : $NUM_PROMPTS"
echo "  --max-tokens  : $MAX_TOKENS"
echo "  --gpu         : $GPU"
echo "  --output-dir  : $OUTPUT_DIR"
echo "  nsys enabled  : $ENABLE_NSYS"
echo "============================================================"

# --- env for iteration logging ---
export VLLM_LOG_ITERATIONS=1
export VLLM_ITERATION_LOG_DIR="$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

# --- base python command ---
PY_CMD=(
    python "$REPO_ROOT/zeyu/run_qwen35_vision_offline.py"
        --model "$MODEL"
        --num-prompts "$NUM_PROMPTS"
        --max-tokens "$MAX_TOKENS"
        --max-num-seqs "$MAX_NUM_SEQS"
        --mm-pipeline "$MM_PIPELINE"
        --output-dir "$OUTPUT_DIR"
)

if $ENABLE_NSYS; then
    # Required for nsys' cuda_gpu_trace to be populated:
    PY_CMD+=(--enforce-eager)
    export VLLM_NVTX_SCOPES_FOR_PROFILING=1
    export VLLM_ENABLE_V1_MULTIPROCESSING=0

    pushd "$OUTPUT_DIR" >/dev/null
    "$NSYS_BIN" profile \
        --trace=cuda,nvtx \
        --output=nsys_report \
        --force-overwrite=true \
        "${PY_CMD[@]}" 2>&1 | tee run.log
    RC=${PIPESTATUS[0]}

    echo "[pipeline_run] Exporting CSVs ..."
    "$NSYS_BIN" stats --force-export=true --format csv \
        -r cuda_gpu_trace --output nsys_kernels \
        nsys_report.nsys-rep 2>&1 | tail -4
    "$NSYS_BIN" stats --force-export=true --format csv \
        -r nvtx_pushpop_trace --output nsys_nvtx_pushpop \
        nsys_report.nsys-rep 2>&1 | tail -4
    popd >/dev/null

    echo "[pipeline_run] Running analyze_profile.py ..."
    python "$REPO_ROOT/zeyu/analyze_profile.py" "$OUTPUT_DIR" \
        2>&1 | tee -a "$OUTPUT_DIR/analyze.log"

    # Short summary of what to look at.
    if [[ -f "$OUTPUT_DIR/consolidated_iterations.jsonl" ]]; then
        python3 - <<PYSUM
import json
iters = [json.loads(l) for l in open("$OUTPUT_DIR/consolidated_iterations.jsonl") if l.strip()]
ov = sum(it.get("nvtx_overlap_ns", 0) for it in iters)
ov_pct = [it.get("nvtx_overlap_pct", 0) for it in iters if it.get("nvtx_overlap_pct",0) > 0]
ve_kt = sum(it.get("vision_encoder_kernel_time_ns", 0) for it in iters)
fw_kt = sum(it.get("text_forward_kernel_time_ns", 0) for it in iters)
print("==== nvtx_overlap_ns summary ====")
print(f"  iters                     : {len(iters)}")
print(f"  iters with overlap > 0    : {len(ov_pct)}")
print(f"  sum nvtx_overlap_ns       : {ov:,}  ({ov/1e6:.2f} ms)")
print(f"  sum vision_encoder_ns     : {ve_kt:,}  ({ve_kt/1e6:.2f} ms)")
print(f"  sum text_forward_ns       : {fw_kt:,}  ({fw_kt/1e6:.2f} ms)")
if ve_kt > 0:
    print(f"  overlap / ve_kt pct       : {ov/ve_kt*100:.2f}%")
PYSUM
    fi
    exit "$RC"
else
    "${PY_CMD[@]}"
fi
