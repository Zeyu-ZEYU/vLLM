#!/usr/bin/env bash
# BL1 (single-GPU origin) experiment orchestrator.
#
# Runs on a remote node inside the mono_kernel container (mamba env
# `mono_kernel`). Brings up `vllm serve` with the BL1 sidecar metrics
# enabled, runs `vllm bench serve` against it with the custom_mm dataset,
# tears down the server, and merges client + server metrics into the spec
# JSONL at $OUTPUTS_DIR/origin_<workload>_<args>_<time>.jsonl.
#
# Required env / defaults (override before invocation if needed):
#   WORKTREE      vLLM worktree dir, default /home/zeyu/vLLM/mono_kernel_origin
#   INPUTS_DIR    project inputs/ dir,        default $HOME/mono_kernel/inputs
#   OUTPUTS_DIR   metrics output dir,         default $HOME/mono_kernel/outputs/metrics
#   LOGS_DIR      server/client logs dir,     default $HOME/mono_kernel/outputs/logs
#   MODEL         model path,                 default $HOME/models/Qwen3-VL-8B-Instruct
#   WORKLOAD      label,                      default example
#                 Picks $INPUTS_DIR/requests/<WORKLOAD>.jsonl unless
#                 DATASET_FILE is overridden.
#   DATASET_FILE  full path to requests jsonl, default $INPUTS_DIR/requests/<WORKLOAD>.jsonl
#   ARGS_TAG      label suffix,               default qwen3vl8b_n5_rps2
#   PORT          server port,                default 8000
#   NUM_PROMPTS   client prompts,             default 5
#   REQUEST_RATE  client rate,                default 2
#   MAX_NUM_SEQS  server concurrency,         default 16

set -euo pipefail

WORKTREE=${WORKTREE:-/home/zeyu/vLLM/mono_kernel_origin}
INPUTS_DIR=${INPUTS_DIR:-$HOME/mono_kernel/inputs}
OUTPUTS_DIR=${OUTPUTS_DIR:-$HOME/mono_kernel/outputs/metrics}
LOGS_DIR=${LOGS_DIR:-$HOME/mono_kernel/outputs/logs}
MODEL=${MODEL:-$HOME/models/Qwen3-VL-8B-Instruct}
WORKLOAD=${WORKLOAD:-example}
DATASET_FILE=${DATASET_FILE:-$INPUTS_DIR/requests/${WORKLOAD}.jsonl}
ARGS_TAG=${ARGS_TAG:-qwen3vl8b_n5_rps2}
PORT=${PORT:-8000}
NUM_PROMPTS=${NUM_PROMPTS:-5}
REQUEST_RATE=${REQUEST_RATE:-2}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPUS=${GPUS:-0}
# Force visible GPUs early so every child inherits the pin.
export CUDA_VISIBLE_DEVICES="$GPUS"

mkdir -p "$OUTPUTS_DIR" "$LOGS_DIR"

# 1) clean residual processes and free ports.
PORTS="$PORT" SLEEP_AFTER=${CLEAN_SLEEP:-80} \
    bash "$WORKTREE/mk_scripts/clean.sh"

# 2) ensure dataset jsonl exists; auto-gen only for the bundled "example"
#    workload — other workloads must be staged in advance (e.g. via
#    mk_scripts/build_sharegpt4v_inputs.py + rsync).
if [[ ! -f "$DATASET_FILE" ]]; then
    if [[ "$WORKLOAD" == "example" ]]; then
        echo "[run_bl1] generating example inputs at $INPUTS_DIR"
        python "$WORKTREE/mk_scripts/make_example_inputs.py" --out-dir "$INPUTS_DIR"
    else
        echo "[run_bl1] dataset file not found: $DATASET_FILE" >&2
        exit 2
    fi
fi

T=$(date +%Y%m%d_%H%M%S)
SERVER_LOG="$LOGS_DIR/bl1_server_${T}.log"
CLIENT_LOG="$LOGS_DIR/bl1_client_${T}.log"
SERVER_SIDE="$OUTPUTS_DIR/.bl1_server_${T}.jsonl"
CLIENT_FILE=".bl1_client_${T}.json"
FINAL_FILE="origin_${WORKLOAD}_${ARGS_TAG}_${T}.jsonl"

echo "[run_bl1] T=$T"
echo "[run_bl1] launching server: log=$SERVER_LOG sidecar=$SERVER_SIDE"

# 3) start server with BL1 instrumentation enabled
MONO_KERNEL_BL1_METRICS_PATH="$SERVER_SIDE" \
    vllm serve "$MODEL" \
        --tensor-parallel-size 1 \
        --port "$PORT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --allowed-local-media-path "$INPUTS_DIR/assets" \
        > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run_bl1] stopping server (pid=$SERVER_PID)"
        kill -INT "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 4) wait for /health
echo "[run_bl1] waiting for server health on :$PORT"
HEALTH_DEADLINE=$((SECONDS + 600))
while (( SECONDS < HEALTH_DEADLINE )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run_bl1] server died before becoming healthy; tail of log:" >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if grep -q -E "^ERROR|Traceback" "$SERVER_LOG" 2>/dev/null; then
        echo "[run_bl1] server log shows ERROR/traceback; tail:" >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[run_bl1] server healthy"
        break
    fi
    sleep 2
done
if ! curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[run_bl1] server did not become healthy in time" >&2
    exit 1
fi

# 5) run benchmark client
echo "[run_bl1] running vllm bench serve: log=$CLIENT_LOG"
vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/chat/completions \
    --model "$MODEL" \
    --dataset-name custom_mm \
    --dataset-path "$DATASET_FILE" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --disable-shuffle \
    --save-result \
    --save-detailed \
    --result-dir "$OUTPUTS_DIR" \
    --result-filename "$CLIENT_FILE" \
    > "$CLIENT_LOG" 2>&1
CLIENT_RC=$?
if (( CLIENT_RC != 0 )); then
    echo "[run_bl1] bench client failed (rc=$CLIENT_RC); tail:" >&2
    tail -n 80 "$CLIENT_LOG" >&2 || true
    exit "$CLIENT_RC"
fi

# 6) stop server (cleanup trap will also fire)
cleanup
trap - EXIT INT TERM

# 7) merge client + server records into the final spec JSONL
echo "[run_bl1] merging metrics into $OUTPUTS_DIR/$FINAL_FILE"
python "$WORKTREE/mk_scripts/merge_metrics.py" \
    --client  "$OUTPUTS_DIR/$CLIENT_FILE" \
    --server  "$SERVER_SIDE" \
    --inputs  "$DATASET_FILE" \
    --label origin --workload "$WORKLOAD" --args "$ARGS_TAG" --time "$T" \
    --out     "$OUTPUTS_DIR/$FINAL_FILE"

echo "[run_bl1] done: $OUTPUTS_DIR/$FINAL_FILE"
