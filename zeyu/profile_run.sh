#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Profile wrapper for Qwen3.5 offline inference with per-iteration tracking.
#
# Usage:
#   bash zeyu/profile_run.sh [--nsys-only|--ncu-only|--skip-profile] [script args...]
#
# Examples:
#   # Full profiling (nsys + ncu)
#   bash zeyu/profile_run.sh --model /path/to/Qwen3.5-9B
#
#   # nsys only (GPU utilization per iteration)
#   bash zeyu/profile_run.sh --nsys-only --model /path/to/Qwen3.5-9B
#
#   # Iteration JSONL only (no GPU profiling)
#   bash zeyu/profile_run.sh --skip-profile --model /path/to/Qwen3.5-9B
#
#   # With JSONL input and delays
#   bash zeyu/profile_run.sh --nsys-only --model /path/to/Qwen3.5-9B --input zeyu/inputs/reqs/sample.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse our flags (before passing remaining args to the Python script).
RUN_NSYS=true
RUN_NCU=true
SKIP_PROFILE=false

SCRIPT_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nsys-only)
            RUN_NCU=false
            shift
            ;;
        --ncu-only)
            RUN_NSYS=false
            shift
            ;;
        --skip-profile)
            SKIP_PROFILE=true
            RUN_NSYS=false
            RUN_NCU=false
            shift
            ;;
        *)
            SCRIPT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Create timestamped output directory.
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
OUT_DIR="$SCRIPT_DIR/outputs/profile_$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Profile output directory: $OUT_DIR"
echo "=========================================="

# Common env vars for iteration logging.
export VLLM_LOG_ITERATIONS=1
export VLLM_ITERATION_LOG_DIR="$OUT_DIR"

PYTHON_SCRIPT="$SCRIPT_DIR/run_qwen35_vision_offline.py"

# --- Run 1: nsys profiling ---
if $RUN_NSYS; then
    echo ""
    echo "[Run 1/2] nsys profiling ..."
    export VLLM_NVTX_SCOPES_FOR_PROFILING=1

    nsys profile \
        --trace=cuda,nvtx \
        --cuda-graph-trace=node \
        --output="$OUT_DIR/nsys_report" \
        --force-overwrite=true \
        python "$PYTHON_SCRIPT" "${SCRIPT_ARGS[@]}"

    echo ""
    echo "Exporting nsys CSV reports ..."

    # Kernel timing trace.
    nsys stats \
        -r cuda_gpu_trace \
        --format csv \
        --output "$OUT_DIR/nsys_kernels" \
        "$OUT_DIR/nsys_report.nsys-rep" 2>/dev/null || true

    # NVTX ranges with GPU projection.
    nsys stats \
        -r nvtx_gpu_proj_trace \
        --format csv \
        --output "$OUT_DIR/nsys_nvtx" \
        "$OUT_DIR/nsys_report.nsys-rep" 2>/dev/null || true

    # NVTX push/pop trace (CPU-side boundaries, same clock as kernels).
    nsys stats \
        -r nvtx_pushpop_trace \
        --format csv \
        --output "$OUT_DIR/nsys_nvtx_pushpop" \
        "$OUT_DIR/nsys_report.nsys-rep" 2>/dev/null || true

    echo "nsys CSV export done."

# --- No nsys, just iteration logging ---
elif ! $RUN_NCU; then
    echo ""
    echo "[Iteration logging only] ..."
    python "$PYTHON_SCRIPT" "${SCRIPT_ARGS[@]}"
fi

# --- Run 2: ncu profiling (optional, slow) ---
if $RUN_NCU; then
    echo ""
    echo "[Run 2/2] ncu profiling (this will be SLOW, 10-100x) ..."
    echo "Targeting: gemm, attention, matmul, embed, conv kernels"

    # Unset NVTX to reduce noise in ncu.
    unset VLLM_NVTX_SCOPES_FOR_PROFILING 2>/dev/null || true

    ncu \
        --set full \
        --kernel-name "regex:.*gemm.*|.*attention.*|.*matmul.*|.*embed.*|.*conv.*" \
        --launch-skip 50 \
        --launch-count 500 \
        --csv \
        python "$PYTHON_SCRIPT" "${SCRIPT_ARGS[@]}" \
        > "$OUT_DIR/ncu_metrics.csv" 2>"$OUT_DIR/ncu_stderr.log" || true

    echo "ncu profiling done."
fi

echo ""
echo "=========================================="
echo "Profile complete. Output in: $OUT_DIR"
echo ""
echo "Files:"
ls -la "$OUT_DIR/"
echo ""
echo "Next step: python zeyu/analyze_profile.py $OUT_DIR"
echo "=========================================="
