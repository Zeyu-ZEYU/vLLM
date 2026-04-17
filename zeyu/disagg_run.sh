#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Cross-node disaggregation launcher for Qwen3-VL-8B-Instruct.
#
# Runs vision encoder + prefill on THIS node (the "prefill node") and
# decode on a REMOTE node over SSH. Collects all metrics on the prefill
# node (tar'd from decode node at the end) and produces a merged
# summary JSON.
#
# By default, nsys profiling is ON on both sides. This collects:
# per-iteration GPU utilization, kernel-launch gap, per-phase (VE /
# prefill / decode) GPU / kernel / memory stats. Use --no-nsys to
# disable. Pass --sm-metrics to also collect aggregate "SM active %"
# and "num active SMs" via nsys --gpu-metrics-devices (requires
# unrestricted GPU performance counters on the host; see
# https://developer.nvidia.com/ERR_NVGPUCTRPERM ).
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
ENABLE_NSYS=false              # Requires working nsys; pynvml is the default path
ENABLE_SM_METRICS=false        # Requires GPU perf-counter privilege
NSYS_GPU_METRICS_FREQ=10000    # 10 kHz = 100 us between samples
# Candidate paths for nsys (checked in order, first found wins).
NSYS_PATH_CANDIDATES=(
    "nsys"
    "/usr/local/cuda/bin/nsys"
    "/opt/nvidia/nsight-systems/bin/nsys"
    "/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys"
)

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
        --nsys) ENABLE_NSYS=true; shift 1;;
        --no-nsys) ENABLE_NSYS=false; shift 1;;
        --sm-metrics) ENABLE_SM_METRICS=true; shift 1;;
        --no-sm-metrics) ENABLE_SM_METRICS=false; shift 1;;
        --nsys-freq) NSYS_GPU_METRICS_FREQ="$2"; shift 2;;
        -h|--help)
            sed -n '1,60p' "$0"
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

# ---------- Locate nsys (local) ----------
LOCAL_NSYS=""
if $ENABLE_NSYS; then
    for cand in "${NSYS_PATH_CANDIDATES[@]}"; do
        if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
            LOCAL_NSYS="$cand"
            break
        fi
    done
    if [[ -z "$LOCAL_NSYS" ]]; then
        echo "[launcher] WARN: nsys not found on this node; disabling profiling."
        ENABLE_NSYS=false
    else
        echo "[launcher] Using nsys: $LOCAL_NSYS"
    fi
fi

### nsys path resolution: detect ahead of time on each node so we can
### embed a concrete absolute path into the remote command (avoids
### quoting hell with nested bash -lc invocations).

echo "============================================================"
echo "  Cross-node disaggregation launcher"
echo "============================================================"
echo "  Prefill (local) : $(hostname -s)  $IFACE=$LOCAL_IP  GPU=$PREFILL_GPU"
echo "  Decode  (remote): $PEER_HOST                  GPU=$DECODE_GPU"
echo "  Model           : $MODEL"
echo "  Prompts         : $NUM_PROMPTS  max_tokens=$MAX_TOKENS"
echo "  KV port         : $KV_PORT  ctrl_port=$CTRL_PORT"
echo "  Profiling       : nsys=$ENABLE_NSYS  sm_metrics=$ENABLE_SM_METRICS  (gpu_metrics_freq=$NSYS_GPU_METRICS_FREQ)"
echo "  Output dir      : $OUT_DIR"
echo "============================================================"

# ---------- Remote decode start ----------
REMOTE_DECODE_LOG="$REMOTE_OUT_DIR/decode.log"
REMOTE_DECODE_PID_FILE="$REMOTE_OUT_DIR/decode.pid"
REMOTE_NSYS_REPORT="$REMOTE_OUT_DIR/decode/nsys_report"

INPUT_ARG=""
if [[ -n "$INPUT" ]]; then
    INPUT_ARG="--input $INPUT"
fi

SSH_PREFIX=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${PEER_SSH_USER}@${PEER_HOST}")

### Detect remote nsys path (inside container) via a probe SSH.
REMOTE_NSYS=""
if $ENABLE_NSYS; then
    REMOTE_NSYS="$(
        "${SSH_PREFIX[@]}" "docker exec -u $PEER_SSH_USER $CONTAINER bash -lc 'for c in nsys /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/bin/nsys /opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys; do if command -v \"\$c\" >/dev/null 2>&1 || [[ -x \"\$c\" ]]; then echo \"\$c\"; exit 0; fi; done'" 2>/dev/null | tr -d '[:space:]'
    )"
    if [[ -z "$REMOTE_NSYS" ]]; then
        echo "[launcher] WARN: nsys not found on $PEER_HOST; disabling decode-side profiling."
    else
        echo "[launcher] Remote nsys: $REMOTE_NSYS"
    fi
fi

# Build the python command (possibly wrapped with nsys).
if $ENABLE_NSYS && [[ -n "$REMOTE_NSYS" ]]; then
    SM_METRICS_FLAG=""
    if $ENABLE_SM_METRICS; then
        SM_METRICS_FLAG="--gpu-metrics-devices=all --gpu-metrics-frequency=$NSYS_GPU_METRICS_FREQ"
    fi
    REMOTE_PY_CMD_WRAP="\
        '$REMOTE_NSYS' profile \
            --trace=cuda,nvtx \
            --cuda-graph-trace=node \
            --trace-fork-before-exec=true \
            $SM_METRICS_FLAG \
            --output='$REMOTE_NSYS_REPORT' \
            --force-overwrite=true \
            "
    REMOTE_NVTX_EXPORT=" && \
    '$REMOTE_NSYS' stats -r cuda_gpu_trace --format csv --output '$REMOTE_OUT_DIR/decode/nsys_kernels' '$REMOTE_NSYS_REPORT.nsys-rep' 2>/dev/null || true; \
    '$REMOTE_NSYS' stats -r nvtx_pushpop_trace --format csv --output '$REMOTE_OUT_DIR/decode/nsys_nvtx_pushpop' '$REMOTE_NSYS_REPORT.nsys-rep' 2>/dev/null || true; \
    '$REMOTE_NSYS' stats -r gpu_metrics --format csv --output '$REMOTE_OUT_DIR/decode/nsys_gpu_metrics' '$REMOTE_NSYS_REPORT.nsys-rep' 2>/dev/null || true"
    REMOTE_NVTX_ENV="export VLLM_NVTX_SCOPES_FOR_PROFILING=1 && "
else
    REMOTE_PY_CMD_WRAP=""
    REMOTE_NVTX_EXPORT=""
    REMOTE_NVTX_ENV=""
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
    ${REMOTE_NVTX_ENV}\
    ${REMOTE_PY_CMD_WRAP}python zeyu/run_qwen35_vision_offline.py \
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
        $INPUT_ARG ${REMOTE_NVTX_EXPORT}\
' >'$REMOTE_DECODE_LOG' 2>&1 </dev/null &) && \
sleep 1 && \
pgrep -f 'role decode' | tail -1 > '$REMOTE_DECODE_PID_FILE'"

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

LOCAL_NSYS_REPORT="$OUT_DIR/prefill/nsys_report"

echo "[launcher] Starting prefill on $(hostname -s) GPU $PREFILL_GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ..."
if $ENABLE_NSYS; then
    export VLLM_NVTX_SCOPES_FOR_PROFILING=1
    SM_FLAGS=()
    if $ENABLE_SM_METRICS; then
        SM_FLAGS=(--gpu-metrics-devices=all --gpu-metrics-frequency="$NSYS_GPU_METRICS_FREQ")
    fi
    # shellcheck disable=SC2086
    "$LOCAL_NSYS" profile \
        --trace=cuda,nvtx \
        --cuda-graph-trace=node \
        --trace-fork-before-exec=true \
        "${SM_FLAGS[@]}" \
        --output="$LOCAL_NSYS_REPORT" \
        --force-overwrite=true \
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

    echo "[launcher] Exporting prefill nsys CSVs ..."
    "$LOCAL_NSYS" stats -r cuda_gpu_trace --format csv \
        --output "$OUT_DIR/prefill/nsys_kernels" \
        "$LOCAL_NSYS_REPORT.nsys-rep" 2>/dev/null || true
    "$LOCAL_NSYS" stats -r nvtx_pushpop_trace --format csv \
        --output "$OUT_DIR/prefill/nsys_nvtx_pushpop" \
        "$LOCAL_NSYS_REPORT.nsys-rep" 2>/dev/null || true
    "$LOCAL_NSYS" stats -r gpu_metrics --format csv \
        --output "$OUT_DIR/prefill/nsys_gpu_metrics" \
        "$LOCAL_NSYS_REPORT.nsys-rep" 2>/dev/null || true
else
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
fi

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

# ---------- Run analyze_profile on each role's data (if nsys enabled) ----------
if $ENABLE_NSYS; then
    for role in prefill decode; do
        if [[ -s "$OUT_DIR/$role/iterations.jsonl" ]]; then
            echo "[launcher] Analyzing $role ..."
            python "$LOCAL_REPO/zeyu/analyze_profile.py" "$OUT_DIR/$role" \
                2>&1 | tee -a "$OUT_DIR/analyze_$role.log" \
                || echo "[launcher] analyze_profile for $role failed (non-fatal)"
        fi
    done
fi

# ---------- Merge into disagg_summary.json ----------
echo "[launcher] Merging metrics ..."
python "$LOCAL_REPO/zeyu/merge_disagg.py" "$OUT_DIR" \
    || { echo "[launcher] merge_disagg.py failed"; exit 3; }

echo "============================================================"
echo "  Done. Output: $OUT_DIR"
echo "============================================================"
ls -la "$OUT_DIR"

trap - EXIT
exit $PREFILL_RC
