#!/usr/bin/env bash
# BL2 (vision-text disaggregation) experiment orchestrator.
#
# Runs on node 0 (inside the mono_kernel container, mamba env mono_kernel).
# Spins up:
#   - vision (encoder-only) instance on node 0 GPU 0  (port $VIS_PORT)
#   - text (prefill+decode) instance on node 1 GPU 0  (port $TEXT_PORT, via ssh)
#   - disagg_epd_proxy on node 0 (port $PROXY_PORT) routing fan-out -> vision,
#     forward -> text.
# Both instances use --ec-transfer-config 'NixlECConnector', vision = producer,
# text = consumer. Bond RNIC (PCIe-closest to GPU 0 on each node) is selected
# automatically and exported via UCX_NET_DEVICES inside each instance.
#
# When COLLECT_SM_METRICS=1, the workload is run twice:
#   pass 1 "default": NVML recorder. Fills d_*, gu_*, gmu_*, plus
#                     d_vemb_transfer (BL2 vemb sidecar).
#   pass 2 "sm":      DCGM SM recorder. Fills nsm_*, smu_*, ko_*, sm_occ_*.
# Only one profiling consumer can run at a time, so the passes are physically
# serial. Both passes use the same input file. The merge combines them into
# a single jsonl.
#
# Sidecars produced (all collected on node 0 before merge):
#   .bl2_client_<T>.json                 -- vllm bench serve output, pass 1
#   .bl2_client_sm_<T>.json              -- vllm bench serve output, pass 2 (SM)
#   .bl2_vis_server_<T>.jsonl            -- node 0 NVML, vision phases
#   .bl2_vis_server_sm_<T>.jsonl         -- node 0 DCGM SM, vision phases
#   .bl2_vis_vemb_<T>.jsonl              -- node 0 BL2 producer events (pass 1 only)
#   .bl2_text_server_<T>.jsonl           -- node 1 NVML, prefill+decode phases
#   .bl2_text_server_sm_<T>.jsonl        -- node 1 DCGM SM, prefill+decode phases
#   .bl2_text_vemb_<T>.jsonl             -- node 1 BL2 consumer events (pass 1 only)
#
# Required env / defaults (override before invocation if needed):
#   WORKTREE             vLLM worktree dir,         default /home/zeyu/vLLM/mono_kernel_disaggregation
#   INPUTS_DIR           project inputs/ dir,       default $HOME/mono_kernel/inputs
#   OUTPUTS_DIR          metrics output dir,        default $HOME/mono_kernel/outputs/metrics
#   LOGS_DIR             server/client logs dir,    default $HOME/mono_kernel/outputs/logs
#   MODEL                model path,                default $HOME/models/Qwen3-VL-8B-Instruct
#   WORKLOAD             label,                     default example
#   DATASET_FILE         requests jsonl,            default $INPUTS_DIR/requests/${WORKLOAD}.jsonl
#   ARGS_TAG             label suffix,              default qwen3vl8b_n5_rps2
#   PROXY_PORT           proxy port (client target),default 8000
#   VIS_PORT             vision instance port,      default 8002
#   TEXT_PORT            text instance port,        default 8001
#   NIXL_PORT            NIXL bootstrap TCP port,   default 9100
#   NUM_PROMPTS          client prompts,            default 5
#   REQUEST_RATE         client rate,               default 2
#   MAX_NUM_SEQS         server concurrency,        default 16
#   NODE1_HOST           ssh alias for node 1,      default lj1.zeyu.tw
#   NODE1_USER_UID       container exec uid on n1,  default 1001
#   NODE1_OUTPUTS_DIR    metrics dir on node 1,     default $HOME/mono_kernel/outputs/metrics
#   NODE1_LOGS_DIR       server log dir on node 1,  default $HOME/mono_kernel/outputs/logs
#   BOND_DEV_N0          bond IB device on node 0,  default auto-detect
#   BOND_DEV_N1          bond IB device on node 1,  default auto-detect
#   BOND_IP_N0           IP NIXL listens on,        default first IP from `hostname -I`
#   COLLECT_SM_METRICS   0 (default) or 1 to also run the DCGM SM pass.
#   DCGM_HOST            ip:port for nv-hostengine, default 127.0.0.1:5556
#                        Required to be reachable on BOTH nodes when
#                        COLLECT_SM_METRICS=1.
#   DRY_RUN              1 = print only, 0 = run.   default 0.

set -euo pipefail

WORKTREE=${WORKTREE:-/home/zeyu/vLLM/mono_kernel_disaggregation}
INPUTS_DIR=${INPUTS_DIR:-$HOME/mono_kernel/inputs}
OUTPUTS_DIR=${OUTPUTS_DIR:-$HOME/mono_kernel/outputs/metrics}
LOGS_DIR=${LOGS_DIR:-$HOME/mono_kernel/outputs/logs}
MODEL=${MODEL:-$HOME/models/Qwen3-VL-8B-Instruct}
WORKLOAD=${WORKLOAD:-example}
DATASET_FILE=${DATASET_FILE:-$INPUTS_DIR/requests/${WORKLOAD}.jsonl}
ARGS_TAG=${ARGS_TAG:-qwen3vl8b_n5_rps2}
PROXY_PORT=${PROXY_PORT:-8000}
VIS_PORT=${VIS_PORT:-8002}
TEXT_PORT=${TEXT_PORT:-8001}
NIXL_PORT=${NIXL_PORT:-9100}
NUM_PROMPTS=${NUM_PROMPTS:-5}
REQUEST_RATE=${REQUEST_RATE:-2}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
NODE1_HOST=${NODE1_HOST:-lj1.zeyu.tw}
NODE1_USER_UID=${NODE1_USER_UID:-1001}
NODE1_OUTPUTS_DIR=${NODE1_OUTPUTS_DIR:-$HOME/mono_kernel/outputs/metrics}
NODE1_LOGS_DIR=${NODE1_LOGS_DIR:-$HOME/mono_kernel/outputs/logs}
COLLECT_SM_METRICS=${COLLECT_SM_METRICS:-0}
DCGM_HOST=${DCGM_HOST:-127.0.0.1:5556}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$OUTPUTS_DIR" "$LOGS_DIR"

# ---- bond RNIC discovery ---------------------------------------------------
detect_bond_from() {
    # Stdin = `nvidia-smi topo -m` output. Print the mlx5_* with closest PCIe
    # affinity to GPU0 (PIX > PXB > PHB > NODE > SYS).
    python3 - <<'PY'
import sys, re
txt = sys.stdin.read()
lines = [l for l in txt.splitlines() if l.strip()]
if not lines:
    sys.exit(0)
hdr = re.split(r"\s{2,}", lines[0].strip())
for l in lines[1:]:
    cells = l.split()
    if not cells or cells[0] != "GPU0":
        continue
    row = re.split(r"\s{2,}", l.strip())
    rank = {"PIX":0,"PXB":1,"PHB":2,"NODE":3,"SYS":4}
    best = None
    for c, v in zip(hdr, row[1:]):
        if not c.startswith("mlx"):
            continue
        r = rank.get(v.strip(), 99)
        if best is None or r < best[1]:
            best = (c, r)
    if best:
        print(best[0])
    break
PY
}

if [[ -z "${BOND_DEV_N0:-}" ]]; then
    BOND_DEV_N0=$(nvidia-smi topo -m 2>/dev/null | detect_bond_from || true)
    echo "[run_bl2] auto-detected BOND_DEV_N0=${BOND_DEV_N0:-<unset>}"
fi
if [[ -z "${BOND_DEV_N1:-}" ]]; then
    if topo_n1=$(ssh -o BatchMode=yes "$NODE1_HOST" \
            "docker exec -u $NODE1_USER_UID mono_kernel nvidia-smi topo -m" 2>/dev/null); then
        BOND_DEV_N1=$(echo "$topo_n1" | detect_bond_from || true)
    fi
    echo "[run_bl2] auto-detected BOND_DEV_N1=${BOND_DEV_N1:-<unset>}"
fi

if [[ -z "${BOND_IP_N0:-}" ]]; then
    BOND_IP_N0=$(hostname -I | awk '{print $1}')
fi
PEER_ENDPOINT="${BOND_IP_N0}:${NIXL_PORT}"
echo "[run_bl2] PEER_ENDPOINT=$PEER_ENDPOINT"

# Build EC connector configs (single-line JSON).
# slot_bytes default in connector is 24 MiB; for large multi-image inputs
# (e.g., milebench 1920x1280 native-res images) one encoder output can be
# ~66 MiB. Sized to 128 MiB by default; override via SLOT_BYTES env.
SLOT_BYTES=${SLOT_BYTES:-134217728}
# n_slots default 64 for multi-image throughput. Each slot holds one
# encoder output (~66 MiB for 1920x1280 native-res image). 64 slots ×
# 128 MiB = 8 GiB scratch — fits H20 96 GiB easily and supports several
# concurrent multi-image requests in flight before producer back-pressure.
N_SLOTS=${N_SLOTS:-64}
ec_extra_n0='"peer_endpoint":"'${PEER_ENDPOINT}'","slot_bytes":'${SLOT_BYTES}',"n_slots":'${N_SLOTS}
[[ -n "${BOND_DEV_N0:-}" ]] && ec_extra_n0+=',"nixl_dev":"'${BOND_DEV_N0}'"'
EC_CFG_PRODUCER='{"ec_connector":"NixlECConnector","ec_role":"ec_producer","ec_connector_extra_config":{'${ec_extra_n0}'}}'

ec_extra_n1='"peer_endpoint":"'${PEER_ENDPOINT}'","slot_bytes":'${SLOT_BYTES}',"n_slots":'${N_SLOTS}
[[ -n "${BOND_DEV_N1:-}" ]] && ec_extra_n1+=',"nixl_dev":"'${BOND_DEV_N1}'"'
EC_CFG_CONSUMER='{"ec_connector":"NixlECConnector","ec_role":"ec_consumer","ec_connector_extra_config":{'${ec_extra_n1}'}}'

# ensure dataset jsonl exists; auto-gen only for the bundled "example" workload
if [[ ! -f "$DATASET_FILE" ]]; then
    if [[ "$WORKLOAD" == "example" ]]; then
        echo "[run_bl2] generating example inputs at $INPUTS_DIR"
        python "$WORKTREE/mk_scripts/make_example_inputs.py" --out-dir "$INPUTS_DIR"
    else
        echo "[run_bl2] dataset file not found: $DATASET_FILE" >&2
        exit 2
    fi
fi

# Pre-flight: when COLLECT_SM_METRICS=1, nv-hostengine must be reachable on
# both node 0 and node 1 (the BL1 SM recorder dials it from inside each
# vllm worker via pydcgm).
if (( COLLECT_SM_METRICS == 1 )); then
    DCGM_PORT="${DCGM_HOST##*:}"
    n0_ok=0
    if ss -ltn 2>/dev/null | grep -q ":${DCGM_PORT}\b"; then
        n0_ok=1
    fi
    n1_ok=0
    if ssh -o BatchMode=yes "$NODE1_HOST" \
        "docker exec -u 0 mono_kernel ss -ltn 2>/dev/null | grep -q ':${DCGM_PORT}\\b'" 2>/dev/null; then
        n1_ok=1
    fi
    if (( n0_ok == 0 || n1_ok == 0 )); then
        echo "[run_bl2] nv-hostengine missing: node0=$n0_ok node1=$n1_ok" >&2
        echo "  Start it on each node as root inside the container:" >&2
        echo "    docker exec -u 0 mono_kernel nohup nv-hostengine -b 127.0.0.1 -p ${DCGM_PORT} > /tmp/nv-hostengine.log 2>&1 &" >&2
        exit 3
    fi
    echo "[run_bl2] DCGM hostengine OK on both nodes (port=$DCGM_PORT)"
fi

T=$(date +%Y%m%d_%H%M%S)
FINAL_FILE="disaggregation_${WORKLOAD}_${ARGS_TAG}_${T}.jsonl"

# Default (NVML) pass sidecars
VIS_SERVER_SIDE="$OUTPUTS_DIR/.bl2_vis_server_${T}.jsonl"
VIS_VEMB_SIDE="$OUTPUTS_DIR/.bl2_vis_vemb_${T}.jsonl"
TEXT_SERVER_SIDE_REMOTE="$NODE1_OUTPUTS_DIR/.bl2_text_server_${T}.jsonl"
TEXT_VEMB_SIDE_REMOTE="$NODE1_OUTPUTS_DIR/.bl2_text_vemb_${T}.jsonl"
TEXT_SERVER_SIDE_LOCAL="$OUTPUTS_DIR/.bl2_text_server_${T}.jsonl"
TEXT_VEMB_SIDE_LOCAL="$OUTPUTS_DIR/.bl2_text_vemb_${T}.jsonl"

# SM pass sidecars
VIS_SERVER_SM_SIDE="$OUTPUTS_DIR/.bl2_vis_server_sm_${T}.jsonl"
TEXT_SERVER_SM_SIDE_REMOTE="$NODE1_OUTPUTS_DIR/.bl2_text_server_sm_${T}.jsonl"
TEXT_SERVER_SM_SIDE_LOCAL="$OUTPUTS_DIR/.bl2_text_server_sm_${T}.jsonl"

CLIENT_FILE_DEFAULT=".bl2_client_${T}.json"
CLIENT_FILE_SM=".bl2_client_sm_${T}.json"

cleanup_all() {
    echo "[run_bl2] cleanup"
    pkill -KILL -u "$(id -u)" -f "python.*disagg_epd_proxy" 2>/dev/null || true
    pkill -KILL -u "$(id -u)" -f "python.*vllm" 2>/dev/null || true
    pkill -KILL -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -KILL -u "$(id -u)" -f "vllm serve" 2>/dev/null || true
    # Node 1 cleanup. We escalate to root inside the container because the
    # orchestrator's user-level pkill has been observed to leave a 130 GiB
    # EngineCore zombie that blocks the next run's GPU allocation.
    ssh -o BatchMode=yes "$NODE1_HOST" "docker exec -u 0 mono_kernel \
        bash -lc 'pkill -KILL -f \"python.*vllm\" 2>/dev/null || true ;
                  pkill -KILL -f \"VLLM::EngineCore\" 2>/dev/null || true ;
                  pkill -KILL -f \"vllm serve\" 2>/dev/null || true ;
                  for pid in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
                      kill -KILL \"\$pid\" 2>/dev/null || true ;
                  done'" || true
    sleep 5
}
trap cleanup_all EXIT INT TERM

wait_gpu_free() {
    local label="$1"
    local cmd="$2"
    local deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
        local used
        used=$(eval "$cmd" 2>/dev/null | tr -d ' ')
        if [[ -n "$used" && "$used" -lt 2000 ]]; then
            echo "[run_bl2] $label GPU free (${used} MiB used)"
            return 0
        fi
        sleep 2
    done
    echo "[run_bl2] $label GPU still busy after 60s; proceeding anyway" >&2
    return 0
}

# run_pass <label> <nvml_var> <vis_sidecar> <text_sidecar_remote> <client_file> <enable_vemb>
#   label              - "default" or "sm" (used in log file names)
#   nvml_var           - MONO_KERNEL_BL1_METRICS_PATH or MONO_KERNEL_BL1_SM_METRICS_PATH
#   vis_sidecar        - local path on node 0 for the vision-side sidecar
#   text_sidecar_remote- path on node 1 (used inside the launcher script)
#   client_file        - basename for the bench-serve result JSON
#   enable_vemb        - 1 = also set MONO_KERNEL_BL2_VEMB_PATH; 0 = don't.
#                        Only the default pass sets it; the SM pass would
#                        otherwise generate a duplicate set of vemb events
#                        that conflate with the first pass's measurements.
run_pass() {
    local label="$1"
    local nvml_var="$2"
    local vis_sidecar="$3"
    local text_sidecar_remote="$4"
    local client_file="$5"
    local enable_vemb="$6"

    local vis_log="$LOGS_DIR/bl2_vis_${T}_${label}.log"
    local text_log_remote="$NODE1_LOGS_DIR/bl2_text_${T}_${label}.log"
    local proxy_log="$LOGS_DIR/bl2_proxy_${T}_${label}.log"
    local client_log="$LOGS_DIR/bl2_client_${T}_${label}.log"

    echo "[run_bl2] ===== pass: $label ====="

    # Per-pass cleanup + GPU-free wait. Each pass leaves the previous
    # instance's processes dead before launching the new ones.
    PORTS="$VIS_PORT $PROXY_PORT $NIXL_PORT" SLEEP_AFTER=${CLEAN_SLEEP:-80} \
        bash "$WORKTREE/mk_scripts/clean.sh" || true
    ssh -o BatchMode=yes "$NODE1_HOST" "docker exec -u $NODE1_USER_UID mono_kernel \
        bash -lc 'PORTS=\"$TEXT_PORT\" SLEEP_AFTER=${CLEAN_SLEEP:-80} bash $WORKTREE/mk_scripts/clean.sh'" || true
    cleanup_all
    wait_gpu_free "node 0" \
        "nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits"
    wait_gpu_free "node 1" \
        "ssh -o BatchMode=yes $NODE1_HOST 'docker exec -u 0 mono_kernel \
            nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits'"

    # Materialize the remote launcher script for this pass.
    local remote_launcher_local="$LOGS_DIR/.bl2_text_launcher_${T}_${label}.sh"
    local remote_launcher_home="$HOME/.bl2_text_launcher_${T}_${label}.sh"
    local vemb_export=""
    if (( enable_vemb == 1 )); then
        vemb_export="export MONO_KERNEL_BL2_VEMB_PATH=\"$TEXT_VEMB_SIDE_REMOTE\""
    fi
    cat > "$remote_launcher_local" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if command -v mamba >/dev/null 2>&1; then
    eval "\$(mamba shell hook --shell bash 2>/dev/null || true)" || true
    mamba activate mono_kernel || true
fi
mkdir -p "$NODE1_OUTPUTS_DIR" "$NODE1_LOGS_DIR"
cd "$WORKTREE"
export CUDA_VISIBLE_DEVICES=0
export ${nvml_var}="$text_sidecar_remote"
export MONO_KERNEL_BL1_DCGM_HOST="$DCGM_HOST"
$vemb_export
exec vllm serve "$MODEL" \\
    --port $TEXT_PORT \\
    --tensor-parallel-size 1 \\
    --max-num-seqs $MAX_NUM_SEQS \\
    --enable-request-id-headers \\
    --allowed-local-media-path "$INPUTS_DIR/assets" \\
    --ec-transfer-config '$EC_CFG_CONSUMER'
EOF
    chmod +x "$remote_launcher_local"

    # Push launcher and start text instance on node 1.
    scp -o BatchMode=yes "$remote_launcher_local" "${NODE1_HOST}:${remote_launcher_home}"
    ssh -o BatchMode=yes "$NODE1_HOST" "docker exec -u $NODE1_USER_UID mono_kernel \
        bash -lc 'nohup bash $remote_launcher_home > $text_log_remote 2>&1 &'"

    # Launch vision instance on node 0.
    local vemb_n0=""
    if (( enable_vemb == 1 )); then
        vemb_n0="MONO_KERNEL_BL2_VEMB_PATH=$VIS_VEMB_SIDE"
    fi
    env CUDA_VISIBLE_DEVICES=0 \
        ${nvml_var}="$vis_sidecar" \
        MONO_KERNEL_BL1_DCGM_HOST="$DCGM_HOST" \
        ${vemb_n0:-DUMMY_VAR=ignored} \
        vllm serve "$MODEL" \
            --port "$VIS_PORT" \
            --tensor-parallel-size 1 \
            --enforce-eager \
            --no-enable-prefix-caching \
            --enable-request-id-headers \
            --max-num-batched-tokens 114688 \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --allowed-local-media-path "$INPUTS_DIR/assets" \
            --mm-encoder-only \
            --ec-transfer-config "$EC_CFG_PRODUCER" \
            > "$vis_log" 2>&1 &
    local vis_pid=$!

    echo "[run_bl2] waiting for vision (:$VIS_PORT) and text ($NODE1_HOST:$TEXT_PORT) /health"
    local deadline=$((SECONDS + 600))
    local vis_ok=0 text_ok=0
    while (( SECONDS < deadline )) && (( vis_ok == 0 || text_ok == 0 )); do
        if (( vis_ok == 0 )) && curl -fs "http://127.0.0.1:$VIS_PORT/health" >/dev/null 2>&1; then
            vis_ok=1; echo "[run_bl2] vision healthy"
        fi
        if (( text_ok == 0 )) && \
           ssh -o BatchMode=yes "$NODE1_HOST" \
               "curl -fs http://127.0.0.1:$TEXT_PORT/health >/dev/null 2>&1"; then
            text_ok=1; echo "[run_bl2] text healthy"
        fi
        sleep 2
    done
    if (( vis_ok == 0 || text_ok == 0 )); then
        echo "[run_bl2] one or both instances did not become healthy" >&2
        tail -n 80 "$vis_log" >&2 || true
        ssh -o BatchMode=yes "$NODE1_HOST" "tail -n 80 $text_log_remote" >&2 || true
        return 1
    fi

    # Launch proxy.
    python "$WORKTREE/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
        --host 0.0.0.0 --port "$PROXY_PORT" \
        --encode-servers-urls "http://127.0.0.1:$VIS_PORT" \
        --prefill-servers-urls "disable" \
        --decode-servers-urls "http://${NODE1_HOST}:${TEXT_PORT}" \
        > "$proxy_log" 2>&1 &
    local proxy_pid=$!
    sleep 5

    # Run client.
    set +e
    vllm bench serve \
        --backend openai-chat \
        --base-url "http://127.0.0.1:$PROXY_PORT" \
        --endpoint /v1/chat/completions \
        --model "$MODEL" \
        --dataset-name custom_mm \
        --dataset-path "$DATASET_FILE" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --disable-shuffle \
        --save-result --save-detailed \
        --result-dir "$OUTPUTS_DIR" \
        --result-filename "$client_file" \
        > "$client_log" 2>&1
    local rc=$?
    set -e
    kill -INT "$proxy_pid" 2>/dev/null || true
    if (( rc != 0 )); then
        echo "[run_bl2] bench client failed (rc=$rc); tail:" >&2
        tail -n 80 "$client_log" >&2 || true
        return "$rc"
    fi

    # Pull node-1 sidecars back to node 0.
    scp -o BatchMode=yes "${NODE1_HOST}:${text_sidecar_remote}" "${OUTPUTS_DIR}/" || true
    if (( enable_vemb == 1 )); then
        scp -o BatchMode=yes "${NODE1_HOST}:${TEXT_VEMB_SIDE_REMOTE}" "${OUTPUTS_DIR}/" || true
    fi
    return 0
}

if (( DRY_RUN == 1 )); then
    echo "[run_bl2] DRY_RUN=1 — printing what would run, not invoking."
    echo "  EC_CFG_PRODUCER=$EC_CFG_PRODUCER"
    echo "  EC_CFG_CONSUMER=$EC_CFG_CONSUMER"
    echo "  COLLECT_SM_METRICS=$COLLECT_SM_METRICS"
    echo "  VIS_SERVER_SIDE=$VIS_SERVER_SIDE  / VIS_SERVER_SM_SIDE=$VIS_SERVER_SM_SIDE"
    echo "  TEXT_SERVER_SIDE_REMOTE=$TEXT_SERVER_SIDE_REMOTE"
    echo "  TEXT_SERVER_SM_SIDE_REMOTE=$TEXT_SERVER_SM_SIDE_REMOTE"
    echo "  To actually run: DRY_RUN=0 bash $0"
    exit 0
fi

# Pass 1: NVML metrics + BL2 vemb
run_pass default MONO_KERNEL_BL1_METRICS_PATH \
    "$VIS_SERVER_SIDE" "$TEXT_SERVER_SIDE_REMOTE" "$CLIENT_FILE_DEFAULT" 1

# Pass 2: DCGM SM metrics (no BL2 vemb)
SERVER_VIS_SM_ARG=()
SERVER_TEXT_SM_ARG=()
if (( COLLECT_SM_METRICS == 1 )); then
    run_pass sm MONO_KERNEL_BL1_SM_METRICS_PATH \
        "$VIS_SERVER_SM_SIDE" "$TEXT_SERVER_SM_SIDE_REMOTE" "$CLIENT_FILE_SM" 0
    SERVER_VIS_SM_ARG=(--server-vis-sm "$VIS_SERVER_SM_SIDE")
    SERVER_TEXT_SM_ARG=(--server-text-sm "$TEXT_SERVER_SM_SIDE_LOCAL")
fi

echo "[run_bl2] merging metrics into $OUTPUTS_DIR/$FINAL_FILE"
python "$WORKTREE/mk_scripts/merge_metrics.py" \
    --client     "$OUTPUTS_DIR/$CLIENT_FILE_DEFAULT" \
    --server-vis "$VIS_SERVER_SIDE" \
    --server-vis-vemb "$VIS_VEMB_SIDE" \
    --server-text "$TEXT_SERVER_SIDE_LOCAL" \
    --server-text-vemb "$TEXT_VEMB_SIDE_LOCAL" \
    "${SERVER_VIS_SM_ARG[@]}" \
    "${SERVER_TEXT_SM_ARG[@]}" \
    --inputs  "$DATASET_FILE" \
    --label disaggregation --workload "$WORKLOAD" --args "$ARGS_TAG" --time "$T" \
    --out     "$OUTPUTS_DIR/$FINAL_FILE"

echo "[run_bl2] done: $OUTPUTS_DIR/$FINAL_FILE"
