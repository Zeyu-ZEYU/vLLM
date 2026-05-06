#!/usr/bin/env bash
# BL1 (single-GPU origin) experiment orchestrator.
#
# Runs on a remote node inside the mono_kernel container (mamba env
# `mono_kernel`). Brings up `vllm serve` with the BL1 sidecar metrics
# enabled, runs `vllm bench serve` against it with the custom_mm dataset,
# tears down the server, and merges client + server metrics into the spec
# JSONL at $OUTPUTS_DIR/origin_<workload>_<args>_<time>.jsonl.
#
# When COLLECT_SM_METRICS=1 (env), runs the workload twice:
#   pass 1: default (NVML — fills d_*, gu_*, gmu_*, ko_*, plus client metrics)
#   pass 2: SM-only via DCGM (fills nsm_*, smu_*, sm_occ_*).
# Final JSONL combines both. Only one profiling consumer can run at a
# time so the two passes are physically serial.
#
# Required env / defaults (override before invocation if needed):
#   WORKTREE             vLLM worktree dir,         default /home/zeyu/vLLM/mono_kernel_origin
#   INPUTS_DIR           project inputs/ dir,       default $HOME/mono_kernel/inputs
#   OUTPUTS_DIR          metrics output dir,        default $HOME/mono_kernel/outputs/metrics
#   LOGS_DIR             server/client logs dir,    default $HOME/mono_kernel/outputs/logs
#   MODEL                model path,                default $HOME/models/Qwen3-VL-8B-Instruct
#   WORKLOAD             label,                     default example
#                        Picks $INPUTS_DIR/requests/<WORKLOAD>.jsonl unless
#                        DATASET_FILE is overridden.
#   DATASET_FILE         full path to requests jsonl
#   ARGS_TAG             label suffix,              default qwen3vl8b_n5_rps2
#   PORT                 server port,               default 8000
#   NUM_PROMPTS          client prompts,            default 5
#   REQUEST_RATE         client rate,               default 2
#   MAX_NUM_SEQS         server concurrency,        default 16
#   COLLECT_SM_METRICS   0 (default) or 1 to also run SM pass
#   DCGM_HOST            "ip:port" for nv-hostengine, default 127.0.0.1:5556
#                        Required to be reachable when COLLECT_SM_METRICS=1.

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
COLLECT_SM_METRICS=${COLLECT_SM_METRICS:-0}
DCGM_HOST=${DCGM_HOST:-127.0.0.1:5556}
# Force visible GPUs early so every child inherits the pin.
export CUDA_VISIBLE_DEVICES="$GPUS"

mkdir -p "$OUTPUTS_DIR" "$LOGS_DIR"

# ensure dataset jsonl exists; auto-gen only for the bundled "example"
# workload — other workloads must be staged in advance.
if [[ ! -f "$DATASET_FILE" ]]; then
    if [[ "$WORKLOAD" == "example" ]]; then
        echo "[run_bl1] generating example inputs at $INPUTS_DIR"
        python "$WORKTREE/mk_scripts/make_example_inputs.py" --out-dir "$INPUTS_DIR"
    else
        echo "[run_bl1] dataset file not found: $DATASET_FILE" >&2
        exit 2
    fi
fi

# Pre-flight for SM pass: nv-hostengine must already be listening.
if (( COLLECT_SM_METRICS == 1 )); then
    DCGM_PORT="${DCGM_HOST##*:}"
    if ! ss -ltn 2>/dev/null | grep -q ":${DCGM_PORT}\b"; then
        echo "[run_bl1] DCGM hostengine NOT listening on $DCGM_HOST." >&2
        echo "  Start it inside the container as root:" >&2
        echo "    docker exec -u 0 mono_kernel nohup nv-hostengine -b 127.0.0.1 -p ${DCGM_PORT} > /tmp/nv-hostengine.log 2>&1 &" >&2
        exit 3
    fi
fi

T=$(date +%Y%m%d_%H%M%S)
FINAL_FILE="origin_${WORKLOAD}_${ARGS_TAG}_${T}.jsonl"
CLIENT_FILE_DEFAULT=".bl1_client_${T}.json"
CLIENT_FILE_SM=".bl1_client_sm_${T}.json"
SERVER_SIDE="$OUTPUTS_DIR/.bl1_server_${T}.jsonl"
SERVER_SM_SIDE="$OUTPUTS_DIR/.bl1_server_sm_${T}.jsonl"

# run_pass <label> <env_var_name_for_sidecar> <sidecar_path> <client_file_basename>
run_pass() {
    local label="$1"
    local sidecar_var="$2"
    local sidecar_path="$3"
    local client_file="$4"
    local server_log="$LOGS_DIR/bl1_server_${T}_${label}.log"
    local client_log="$LOGS_DIR/bl1_client_${T}_${label}.log"

    echo "[run_bl1] ===== pass: $label ====="
    PORTS="$PORT" SLEEP_AFTER=${CLEAN_SLEEP:-80} \
        bash "$WORKTREE/mk_scripts/clean.sh"

    # Wait for GPU 0 memory to actually drop (CUDA driver memory release
    # can lag the kill of the holding process by several seconds; pass 2
    # vllm-serve will fail to start if the previous pass's KV cache is
    # still resident).
    echo "[run_bl1] waiting for GPU 0 to be free"
    local GPU_DEADLINE=$((SECONDS + 120))
    while (( SECONDS < GPU_DEADLINE )); do
        local used
        used=$(nvidia-smi --id=0 --query-gpu=memory.used \
                          --format=csv,noheader,nounits 2>/dev/null \
               | tr -d ' ')
        if [[ -n "$used" && "$used" -lt 1000 ]]; then
            echo "[run_bl1] GPU 0 free (${used} MiB used)"
            break
        fi
        sleep 2
    done

    echo "[run_bl1] launching server: log=$server_log sidecar=$sidecar_path"
    env "$sidecar_var=$sidecar_path" \
        MONO_KERNEL_BL1_DCGM_HOST="$DCGM_HOST" \
        vllm serve "$MODEL" \
            --tensor-parallel-size 1 \
            --port "$PORT" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --enforce-eager \
            --no-enable-prefix-caching \
            --mm-processor-cache-gb 0 \
            --enable-request-id-headers \
            --allowed-local-media-path "$INPUTS_DIR/assets" \
            > "$server_log" 2>&1 &
    local SERVER_PID=$!
    local CLEANED_UP=0

    cleanup_pass() {
        if (( CLEANED_UP == 1 )); then return; fi
        CLEANED_UP=1
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
        # Reap any orphaned vLLM EngineCore subprocesses still holding the
        # GPU. The parent vllm-serve dies but the engine_core child can
        # outlive it; without this they keep the model + KV cache resident
        # and the next pass cannot allocate.
        pkill -KILL -u "$(id -u)" -f "python.*vllm" 2>/dev/null || true
        pkill -KILL -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null || true
    }
    trap cleanup_pass EXIT INT TERM

    echo "[run_bl1] waiting for server health on :$PORT"
    local HEALTH_DEADLINE=$((SECONDS + 600))
    while (( SECONDS < HEALTH_DEADLINE )); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[run_bl1] server died before becoming healthy; tail:" >&2
            tail -n 80 "$server_log" >&2 || true
            exit 1
        fi
        if grep -q -E "^ERROR|Traceback" "$server_log" 2>/dev/null; then
            echo "[run_bl1] server log shows ERROR/traceback; tail:" >&2
            tail -n 80 "$server_log" >&2 || true
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

    echo "[run_bl1] running vllm bench serve: log=$client_log"
    set +e
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
        --result-filename "$client_file" \
        > "$client_log" 2>&1
    local rc=$?
    set -e
    if (( rc != 0 )); then
        echo "[run_bl1] bench client failed (rc=$rc); tail:" >&2
        tail -n 80 "$client_log" >&2 || true
        cleanup_pass
        exit "$rc"
    fi
    cleanup_pass
    trap - EXIT INT TERM
}

# pass 1: default NVML metrics (always run)
run_pass default MONO_KERNEL_BL1_METRICS_PATH "$SERVER_SIDE" "$CLIENT_FILE_DEFAULT"

# pass 2: SM metrics via DCGM (only when requested)
SERVER_SM_ARG=()
if (( COLLECT_SM_METRICS == 1 )); then
    run_pass sm MONO_KERNEL_BL1_SM_METRICS_PATH "$SERVER_SM_SIDE" "$CLIENT_FILE_SM"
    SERVER_SM_ARG=(--server-sm "$SERVER_SM_SIDE")
fi

# Merge
echo "[run_bl1] merging metrics into $OUTPUTS_DIR/$FINAL_FILE"
python "$WORKTREE/mk_scripts/merge_metrics.py" \
    --client  "$OUTPUTS_DIR/$CLIENT_FILE_DEFAULT" \
    --server  "$SERVER_SIDE" \
    "${SERVER_SM_ARG[@]}" \
    --inputs  "$DATASET_FILE" \
    --label origin --workload "$WORKLOAD" --args "$ARGS_TAG" --time "$T" \
    --out     "$OUTPUTS_DIR/$FINAL_FILE"

echo "[run_bl1] done: $OUTPUTS_DIR/$FINAL_FILE"
