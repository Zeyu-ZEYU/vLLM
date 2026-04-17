#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One-side launcher for cross-node PD disaggregation.
#
# This script launches EXACTLY ONE side of a PD-disaggregated run. You
# must invoke it twice — once on the prefill node (--role prefill) and
# once on the decode node (--role decode) — in two separate terminals.
# There is NO SSH between the nodes; each terminal acts on its own
# local node only.
#
# The two sides coordinate over a plain TCP ZMQ ctrl channel (prefill
# binds, decode connects to `--peer-ip`) and a NCCL connection for the
# actual KV cache transfer.
#
# ----- Order of operations (do it in this order) -----
#
#   1. On PREFILL NODE, terminal A:
#        bash zeyu/disagg_run.sh --role prefill \
#            --iface eth0 \
#            --gpu 0 \
#            --num-prompts 20 --max-tokens 64
#      (The prefill side prints its own IP and waits for decode.)
#
#   2. On DECODE NODE, terminal B (use prefill's IP from step 1):
#        bash zeyu/disagg_run.sh --role decode \
#            --peer-ip 192.0.2.42 \
#            --iface eth0 \
#            --gpu 0 \
#            --num-prompts 20 --max-tokens 64
#
#   3. Both processes exit when decode finishes. Each writes to its own
#      local output directory:
#         <OUT_DIR>/prefill/    (on the prefill node)
#         <OUT_DIR>/decode/     (on the decode node)
#      where <OUT_DIR> defaults to zeyu/outputs/disagg_<UTC-timestamp>.
#
#   4. Copy the decode side's <OUT_DIR>/decode/ directory into the
#      prefill side's <OUT_DIR>/ (anywhere that gives you a single
#      directory with both prefill/ and decode/ subdirs). If both nodes
#      share an NFS mount, pass --output-dir to the SAME path on both
#      sides and this step is a no-op.
#
#   5. On the node that now has both subdirs, run:
#         python zeyu/merge_disagg.py <OUT_DIR>
#      This produces <OUT_DIR>/disagg_summary.json.
#
# Both sides must use IDENTICAL values for these flags (otherwise the
# handshake / KV exchange won't match):
#    --kv-port   --ctrl-port   --num-prompts   --max-tokens   --model
#    --max-model-len   (and --input if using a custom dataset)
#
# You can differ on: --iface (each side uses its own NIC name),
#    --gpu (each side picks a local GPU), --output-dir (per-side paths).

set -euo pipefail

# ---------- Defaults ----------
ROLE=""                         # MUST be set: prefill | decode
MODEL="/home/zeyu/models/Qwen3-VL-8B-Instruct"
NUM_PROMPTS=4
MAX_TOKENS=64
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.85
IFACE="eth0"
KV_PORT=25555
CTRL_PORT=25500
GPU=0
INPUT=""
DELAY=""                        # Optional global inter-arrival delay (ms).
PEER_IP=""                      # Required for --role decode.
OUTPUT_DIR=""                   # Empty → auto-timestamped subdir.
ENABLE_NSYS=false               # pynvml path is always on; nsys is opt-in.
ENABLE_SM_METRICS=false         # Requires GPU perf-counter privilege.
NSYS_GPU_METRICS_FREQ=10000     # 10 kHz = 100 us between samples.
# Candidate paths for nsys (checked in order, first found wins).
NSYS_PATH_CANDIDATES=(
    "nsys"
    "/usr/local/bin/nsys"
    "/usr/local/cuda/bin/nsys"
    "/opt/nvidia/nsight-systems/bin/nsys"
    "/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys"
)

# ---------- Parse args ----------
usage() {
    # Print the top comment block (up to, but not including, the
    # `set -euo pipefail` line).
    sed -n '1,55p' "$0"
    exit 0
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --role) ROLE="$2"; shift 2;;
        --peer-ip) PEER_IP="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2;;
        --max-tokens) MAX_TOKENS="$2"; shift 2;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2;;
        --iface) IFACE="$2"; shift 2;;
        --kv-port) KV_PORT="$2"; shift 2;;
        --ctrl-port) CTRL_PORT="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        --input) INPUT="$2"; shift 2;;
        --delay) DELAY="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --nsys) ENABLE_NSYS=true; shift 1;;
        --no-nsys) ENABLE_NSYS=false; shift 1;;
        --sm-metrics) ENABLE_SM_METRICS=true; shift 1;;
        --no-sm-metrics) ENABLE_SM_METRICS=false; shift 1;;
        --nsys-freq) NSYS_GPU_METRICS_FREQ="$2"; shift 2;;
        -h|--help) usage;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# ---------- Validate required flags ----------
if [[ -z "$ROLE" ]]; then
    echo "ERROR: --role {prefill|decode} is required." >&2
    echo "Run with --help for usage." >&2
    exit 1
fi
if [[ "$ROLE" != "prefill" && "$ROLE" != "decode" ]]; then
    echo "ERROR: --role must be 'prefill' or 'decode' (got: $ROLE)." >&2
    exit 1
fi
if [[ "$ROLE" == "decode" && -z "$PEER_IP" ]]; then
    echo "ERROR: --peer-ip <PREFILL_NODE_IP> is required for --role decode." >&2
    echo "  Get it from the prefill side's startup log (look for 'local_ip=')." >&2
    exit 1
fi

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_REPO="$(dirname "$SCRIPT_DIR")"

if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="$SCRIPT_DIR/outputs/disagg_${TIMESTAMP}"
fi
# This side's role-specific subdir. The OTHER side will (after you copy
# its data over) populate the sibling subdir, so that merge_disagg.py
# can see both under one parent.
ROLE_DIR="$OUTPUT_DIR/$ROLE"
mkdir -p "$ROLE_DIR"

# ---------- Detect local IP on --iface ----------
LOCAL_IP="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
if [[ -z "$LOCAL_IP" ]]; then
    echo "ERROR: could not detect IPv4 address on interface '$IFACE'." >&2
    echo "  Try: ip -4 -o addr   (then pass --iface <the right name>)" >&2
    exit 1
fi

# ---------- Locate nsys ----------
LOCAL_NSYS=""
if $ENABLE_NSYS; then
    for cand in "${NSYS_PATH_CANDIDATES[@]}"; do
        if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
            LOCAL_NSYS="$cand"
            break
        fi
    done
    if [[ -z "$LOCAL_NSYS" ]]; then
        echo "[launcher] WARN: nsys not found; falling back to pynvml-only path."
        ENABLE_NSYS=false
    fi
fi

# ---------- Print launch banner ----------
echo "============================================================"
echo "  PD-disagg launcher: role=$ROLE"
echo "============================================================"
echo "  Host        : $(hostname -s)"
echo "  Iface/IP    : $IFACE = $LOCAL_IP"
echo "  GPU         : $GPU"
echo "  Peer IP     : ${PEER_IP:-<unset (prefill binds & waits)>}"
echo "  KV port     : $KV_PORT         (decode connects to kv_port+100 on peer)"
echo "  Ctrl port   : $CTRL_PORT"
echo "  Model       : $MODEL"
echo "  Prompts     : $NUM_PROMPTS         max_tokens=$MAX_TOKENS"
echo "  Output dir  : $OUTPUT_DIR"
echo "                (this side will write under $ROLE_DIR)"
if $ENABLE_NSYS; then
    echo "  Profiling   : nsys=ON ($LOCAL_NSYS)   sm_metrics=$ENABLE_SM_METRICS"
else
    echo "  Profiling   : pynvml-only (nsys OFF)"
fi
if [[ "$ROLE" == "prefill" ]]; then
    echo
    echo "  >>> After this side is up and printing 'waiting for decode"
    echo "  >>> READY...', go to the DECODE node and run:"
    echo "  >>>"
    echo "  >>>   bash zeyu/disagg_run.sh --role decode \\"
    echo "  >>>       --peer-ip $LOCAL_IP \\"
    echo "  >>>       --iface <decode-nic> --gpu <decode-gpu-idx> \\"
    echo "  >>>       --kv-port $KV_PORT --ctrl-port $CTRL_PORT \\"
    echo "  >>>       --num-prompts $NUM_PROMPTS --max-tokens $MAX_TOKENS \\"
    echo "  >>>       --model $MODEL"
    echo
fi
echo "============================================================"

# ---------- Assemble optional args ----------
INPUT_ARG=""
if [[ -n "$INPUT" ]]; then
    INPUT_ARG="--input $INPUT"
fi

DELAY_ARG=""
if [[ -n "$DELAY" ]]; then
    DELAY_ARG="--delay $DELAY"
fi

PEER_IP_ARG=""
if [[ -n "$PEER_IP" ]]; then
    PEER_IP_ARG="--peer-ip $PEER_IP"
fi

# ---------- Set up env for this process ----------
export VLLM_LOG_ITERATIONS=1
export VLLM_ITERATION_LOG_DIR="$OUTPUT_DIR"
# CUDA_VISIBLE_DEVICES must be set BEFORE python imports torch so that
# the vLLM engine subprocess inherits the correct mask.
export CUDA_VISIBLE_DEVICES="$GPU"
if $ENABLE_NSYS; then
    export VLLM_NVTX_SCOPES_FOR_PROFILING=1
fi

LOG_FILE="$OUTPUT_DIR/${ROLE}.log"

# ---------- Build the python invocation ----------
PY=(python "$LOCAL_REPO/zeyu/run_qwen35_vision_offline.py"
    --role "$ROLE"
    $PEER_IP_ARG
    --kv-port "$KV_PORT"
    --ctrl-port "$CTRL_PORT"
    --iface "$IFACE"
    --gpu 0    # inside-process GPU index after CUDA_VISIBLE_DEVICES masks it to 1 GPU
    --model "$MODEL"
    --num-prompts "$NUM_PROMPTS"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --output-dir "$OUTPUT_DIR"
    $INPUT_ARG $DELAY_ARG)

# ---------- Run ----------
echo "[launcher] Starting $ROLE ..."
if $ENABLE_NSYS; then
    NSYS_REPORT="$ROLE_DIR/nsys_report"
    SM_FLAGS=()
    if $ENABLE_SM_METRICS; then
        SM_FLAGS=(--gpu-metrics-devices=all --gpu-metrics-frequency="$NSYS_GPU_METRICS_FREQ")
    fi
    "$LOCAL_NSYS" profile \
        --trace=cuda,nvtx \
        --cuda-graph-trace=node \
        --trace-fork-before-exec=true \
        "${SM_FLAGS[@]}" \
        --output="$NSYS_REPORT" \
        --force-overwrite=true \
        "${PY[@]}" \
        2>&1 | tee "$LOG_FILE"
    PY_RC=${PIPESTATUS[0]}
else
    "${PY[@]}" 2>&1 | tee "$LOG_FILE"
    PY_RC=${PIPESTATUS[0]}
fi
echo "[launcher] Python exit code: $PY_RC"

# ---------- Post-processing (this side only) ----------
if $ENABLE_NSYS && [[ -s "$ROLE_DIR/nsys_report.nsys-rep" ]]; then
    echo "[launcher] Exporting ${ROLE} nsys CSVs ..."
    # `nsys stats` exports a .sqlite beside the .nsys-rep on first
    # invocation. Subsequent invocations refuse to re-use the .sqlite
    # with 'File is older than input file, use --force-export=true'
    # unless we pass that flag. (Without the flag the second/third
    # stats command silently produces no CSV.)
    NSYS_STATS_OPTS=(stats --force-export=true --format csv)
    "$LOCAL_NSYS" "${NSYS_STATS_OPTS[@]}" -r cuda_gpu_trace \
        --output "$ROLE_DIR/nsys_kernels" \
        "$ROLE_DIR/nsys_report.nsys-rep" 2>/dev/null || true
    "$LOCAL_NSYS" "${NSYS_STATS_OPTS[@]}" -r nvtx_pushpop_trace \
        --output "$ROLE_DIR/nsys_nvtx_pushpop" \
        "$ROLE_DIR/nsys_report.nsys-rep" 2>/dev/null || true
    if $ENABLE_SM_METRICS; then
        # gpu_metrics is only meaningful when --gpu-metrics-devices
        # was active at profile time.
        "$LOCAL_NSYS" "${NSYS_STATS_OPTS[@]}" -r gpu_metrics \
            --output "$ROLE_DIR/nsys_gpu_metrics" \
            "$ROLE_DIR/nsys_report.nsys-rep" 2>/dev/null || true
    fi

    if [[ -s "$ROLE_DIR/iterations.jsonl" ]]; then
        echo "[launcher] Running analyze_profile.py on ${ROLE} ..."
        python "$LOCAL_REPO/zeyu/analyze_profile.py" "$ROLE_DIR" \
            2>&1 | tee -a "$OUTPUT_DIR/analyze_${ROLE}.log" \
            || echo "[launcher] analyze_profile failed (non-fatal)"
    fi
fi

# ---------- Final instructions ----------
echo "============================================================"
echo "  $ROLE side done. Exit code: $PY_RC"
echo "============================================================"
echo "  Output written to: $ROLE_DIR"
echo
if [[ "$ROLE" == "prefill" ]]; then
    echo "  Next steps:"
    echo "    1. Wait for the decode side to finish on the other node."
    echo "    2. Copy its '<their-OUT>/decode/' directory INTO this"
    echo "       output dir:"
    echo "         $OUTPUT_DIR/"
    echo "       so that it contains BOTH 'prefill/' and 'decode/'"
    echo "       subdirs. (Use scp / rsync / a shared NFS mount.)"
    echo "    3. Produce the merged summary:"
    echo "         python zeyu/merge_disagg.py $OUTPUT_DIR"
elif [[ "$ROLE" == "decode" ]]; then
    echo "  Next steps:"
    echo "    1. Copy THIS output dir back to the prefill node (or to"
    echo "       wherever the prefill-side output lives). The prefill"
    echo "       side expects its own '<their-OUT>/' to end up with a"
    echo "       'decode/' sibling of 'prefill/'. Example:"
    echo "         scp -r $ROLE_DIR prefill-host:/path/to/<their-OUT>/"
    echo "    2. On that node:"
    echo "         python zeyu/merge_disagg.py /path/to/<their-OUT>"
fi
echo "============================================================"

exit $PY_RC
