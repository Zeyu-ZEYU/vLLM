#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Cross-node disaggregation launcher for Qwen3-VL-8B-Instruct.
#
# Runs vision encoder + prefill on THIS node (the "prefill node") and
# decode on a REMOTE node over SSH. Collects all metrics on the prefill
# node (rsync'd from decode node at the end) and produces a merged
# summary JSON.
#
# Usage (run on the prefill node, Node 0):
#
#   bash zeyu/disagg_run.sh \
#       --peer-host lj1.zeyu.tw \
#       --peer-ssh-user zeyu \
#       --model /home/zeyu/models/Qwen3-VL-8B-Instruct \
#       --num-prompts 20 \
#       --max-tokens 64
#
# Requires passwordless SSH from THIS node to --peer-host.
# Both nodes must have the repo at the same path (default /home/zeyu/vllm/mono_kernel).
# Both nodes must be inside the fe_rnic container with mono_kernel env active.

set -euo pipefail

# ---------- Defaults ----------
MODEL="/home/zeyu/models/Qwen3-VL-8B-Instruct"
NUM_PROMPTS=4
MAX_TOKENS=64
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.85
IFACE="eth0"
KV_PORT=25555
CTRL_PORT=25500
PREFILL_GPU=0
DECODE_GPU=0
INPUT=""
PEER_HOST=""
PEER_SSH_USER="zeyu"
REMOTE_REPO="/home/zeyu/vllm/mono_kernel"
CONTAINER="fe_rnic"
CONDA_ENV="mono_kernel"
OUTPUT_ROOT=""

# ---------- Parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --peer-host) PEER_HOST="$2"; shift 2;;
        --peer-ssh-user) PEER_SSH_USER="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2;;
        --max-tokens) MAX_TOKENS="$2"; shift 2;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2;;
        --iface) IFACE="$2"; shift 2;;
        --kv-port) KV_PORT="$2"; shift 2;;
        --ctrl-port) CTRL_PORT="$2"; shift 2;;
        --prefill-gpu) PREFILL_GPU="$2"; shift 2;;
        --decode-gpu) DECODE_GPU="$2"; shift 2;;
        --input) INPUT="$2"; shift 2;;
        --remote-repo) REMOTE_REPO="$2"; shift 2;;
        --container) CONTAINER="$2"; shift 2;;
        --conda-env) CONDA_ENV="$2"; shift 2;;
        --output-root) OUTPUT_ROOT="$2"; shift 2;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$PEER_HOST" ]]; then
    echo "ERROR: --peer-host is required (hostname or IP of decode node)" >&2
    exit 1
fi

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_REPO="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}/disagg_${TIMESTAMP}"
mkdir -p "$OUT_DIR/prefill" "$OUT_DIR/decode"

# Remote mirror — same relative path under the remote repo.
REMOTE_OUT_DIR="${REMOTE_REPO}/zeyu/outputs/disagg_${TIMESTAMP}"

# ---------- Detect local eth0 IP ----------
LOCAL_IP="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
if [[ -z "$LOCAL_IP" ]]; then
    echo "ERROR: could not detect IP on interface $IFACE" >&2
    exit 1
fi

echo "============================================================"
echo "  Cross-node disaggregation launcher"
echo "============================================================"
echo "  Prefill (local) : $(hostname -s)  $IFACE=$LOCAL_IP  GPU=$PREFILL_GPU"
echo "  Decode  (remote): $PEER_HOST                  GPU=$DECODE_GPU"
echo "  Model           : $MODEL"
echo "  Prompts         : $NUM_PROMPTS  max_tokens=$MAX_TOKENS"
echo "  KV port         : $KV_PORT  ctrl_port=$CTRL_PORT"
echo "  Output dir      : $OUT_DIR"
echo "============================================================"

# ---------- Remote decode start ----------
REMOTE_DECODE_LOG="$REMOTE_OUT_DIR/decode.log"
REMOTE_DECODE_PID_FILE="$REMOTE_OUT_DIR/decode.pid"

INPUT_ARG=""
if [[ -n "$INPUT" ]]; then
    INPUT_ARG="--input $INPUT"
fi

REMOTE_CMD="\
mkdir -p '$REMOTE_OUT_DIR/decode' && \
cd '$REMOTE_REPO' && \
(nohup bash -lc '\
    source ~/miniforge3/etc/profile.d/conda.sh && \
    conda activate $CONDA_ENV && \
    export VLLM_LOG_ITERATIONS=1 && \
    export VLLM_ITERATION_LOG_DIR=\"$REMOTE_OUT_DIR\" && \
    export CUDA_VISIBLE_DEVICES=$DECODE_GPU && \
    python zeyu/run_qwen35_vision_offline.py \
        --role decode \
        --peer-ip $LOCAL_IP \
        --kv-port $KV_PORT \
        --ctrl-port $CTRL_PORT \
        --iface $IFACE \
        --gpu 0 \
        --model $MODEL \
        --num-prompts $NUM_PROMPTS \
        --max-tokens $MAX_TOKENS \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --output-dir $REMOTE_OUT_DIR \
        $INPUT_ARG \
' >'$REMOTE_DECODE_LOG' 2>&1 </dev/null &) && \
sleep 1 && \
pgrep -f 'role decode' | tail -1 > '$REMOTE_DECODE_PID_FILE'"

SSH_PREFIX=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${PEER_SSH_USER}@${PEER_HOST}")

echo "[launcher] Starting decode on $PEER_HOST ..."
"${SSH_PREFIX[@]}" "docker exec -u $PEER_SSH_USER $CONTAINER bash -lc \"$REMOTE_CMD\"" \
    || { echo "[launcher] FAILED to start remote decode"; exit 2; }

REMOTE_DECODE_PID="$("${SSH_PREFIX[@]}" "docker exec -u $PEER_SSH_USER $CONTAINER bash -lc 'cat $REMOTE_DECODE_PID_FILE 2>/dev/null || echo'" | tr -d '[:space:]')"
echo "[launcher] Remote decode PID=$REMOTE_DECODE_PID, log=$REMOTE_DECODE_LOG"

# Cleanup function: kill remote decode if we die mid-run.
cleanup() {
    echo "[launcher] Cleaning up ..."
    if [[ -n "${REMOTE_DECODE_PID:-}" ]]; then
        "${SSH_PREFIX[@]}" "docker exec -u $PEER_SSH_USER $CONTAINER bash -c 'kill -TERM $REMOTE_DECODE_PID 2>/dev/null; sleep 2; kill -KILL $REMOTE_DECODE_PID 2>/dev/null'" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

# ---------- Local prefill run ----------
export VLLM_LOG_ITERATIONS=1
export VLLM_ITERATION_LOG_DIR="$OUT_DIR"
# Select the prefill GPU via CUDA_VISIBLE_DEVICES BEFORE Python starts so
# that vLLM's engine subprocess (spawned later) inherits the correct mask.
export CUDA_VISIBLE_DEVICES="$PREFILL_GPU"

echo "[launcher] Starting prefill on $(hostname -s) GPU $PREFILL_GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ..."
# shellcheck disable=SC2086
python "$LOCAL_REPO/zeyu/run_qwen35_vision_offline.py" \
    --role prefill \
    --peer-ip "$PEER_HOST" \
    --kv-port "$KV_PORT" \
    --ctrl-port "$CTRL_PORT" \
    --iface "$IFACE" \
    --gpu 0 \
    --model "$MODEL" \
    --num-prompts "$NUM_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --output-dir "$OUT_DIR" \
    $INPUT_ARG \
    2>&1 | tee "$OUT_DIR/prefill.log"

PREFILL_RC=${PIPESTATUS[0]}
echo "[launcher] Prefill exit code: $PREFILL_RC"

# ---------- Wait for remote decode to finish ----------
echo "[launcher] Waiting for remote decode to complete ..."
for i in $(seq 1 600); do
    if ! "${SSH_PREFIX[@]}" "docker exec -u $PEER_SSH_USER $CONTAINER bash -c 'kill -0 $REMOTE_DECODE_PID 2>/dev/null'" >/dev/null 2>&1; then
        echo "[launcher] Remote decode finished."
        break
    fi
    sleep 2
done

# Print last lines of remote decode log.
echo "[launcher] Remote decode log (last 30 lines):"
"${SSH_PREFIX[@]}" "docker exec -u $PEER_SSH_USER $CONTAINER bash -c 'tail -30 $REMOTE_DECODE_LOG 2>/dev/null'" || true

# ---------- Pull remote decode outputs back ----------
# rsync may not be installed in the container; use tar-over-ssh instead.
echo "[launcher] Copying remote decode outputs back ..."
mkdir -p "$OUT_DIR/decode"
"${SSH_PREFIX[@]}" "tar -C '${REMOTE_OUT_DIR}' -cf - decode" \
    | tar -C "$OUT_DIR" -xf - \
    || { echo "[launcher] tar copy FAILED (decode dir may be empty)"; }

# Also pull remote decode.log for reference.
"${SSH_PREFIX[@]}" "cat '${REMOTE_DECODE_LOG}'" > "$OUT_DIR/decode.log" \
    2>/dev/null || true

# ---------- Merge into disagg_summary.json ----------
echo "[launcher] Merging metrics ..."
python "$LOCAL_REPO/zeyu/merge_disagg.py" "$OUT_DIR" \
    || { echo "[launcher] merge_disagg.py failed"; exit 3; }

# ---------- Analyze (optional: iteration+kernel correlation) ----------
# Skip analyze_profile.py if no nsys report is present; disagg_summary
# already has per-request and per-iteration metrics.

echo "============================================================"
echo "  Done. Output: $OUT_DIR"
echo "============================================================"
ls -la "$OUT_DIR"

trap - EXIT
exit $PREFILL_RC
