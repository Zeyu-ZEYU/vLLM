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
# Sidecars produced (all collected on node 0 before merge):
#   .bl2_client_<T>.json            -- vllm bench serve output
#   .bl2_vis_server_<T>.jsonl       -- node 0 NVML (vision phases)
#   .bl2_vis_vemb_<T>.jsonl         -- node 0 BL2 producer events
#   .bl2_text_server_<T>.jsonl      -- node 1 NVML (prefill+decode phases)
#   .bl2_text_vemb_<T>.jsonl        -- node 1 BL2 consumer events
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
#   DRY_RUN              1 = print only, 0 = run.   default 1 (this iteration's
#                        scope is code-only; flip to 0 once you've validated
#                        nodes 0+1 builds and want to actually run it).

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
DRY_RUN=${DRY_RUN:-1}

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
ec_extra_n0='"peer_endpoint":"'${PEER_ENDPOINT}'"'
[[ -n "${BOND_DEV_N0:-}" ]] && ec_extra_n0+=',"nixl_dev":"'${BOND_DEV_N0}'"'
EC_CFG_PRODUCER='{"ec_connector":"NixlECConnector","ec_role":"ec_producer","ec_connector_extra_config":{'${ec_extra_n0}'}}'

ec_extra_n1='"peer_endpoint":"'${PEER_ENDPOINT}'"'
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

T=$(date +%Y%m%d_%H%M%S)
FINAL_FILE="disaggregation_${WORKLOAD}_${ARGS_TAG}_${T}.jsonl"

VIS_SERVER_SIDE="$OUTPUTS_DIR/.bl2_vis_server_${T}.jsonl"
VIS_VEMB_SIDE="$OUTPUTS_DIR/.bl2_vis_vemb_${T}.jsonl"
TEXT_SERVER_SIDE_REMOTE="$NODE1_OUTPUTS_DIR/.bl2_text_server_${T}.jsonl"
TEXT_VEMB_SIDE_REMOTE="$NODE1_OUTPUTS_DIR/.bl2_text_vemb_${T}.jsonl"
TEXT_SERVER_SIDE_LOCAL="$OUTPUTS_DIR/.bl2_text_server_${T}.jsonl"
TEXT_VEMB_SIDE_LOCAL="$OUTPUTS_DIR/.bl2_text_vemb_${T}.jsonl"

CLIENT_FILE_DEFAULT=".bl2_client_${T}.json"
TEXT_LOG_REMOTE="$NODE1_LOGS_DIR/bl2_text_${T}.log"
VIS_LOG="$LOGS_DIR/bl2_vis_${T}.log"
PROXY_LOG="$LOGS_DIR/bl2_proxy_${T}.log"
CLIENT_LOG="$LOGS_DIR/bl2_client_${T}.log"

# Materialize the remote launcher script locally, scp it to node 1, then
# invoke it via `docker exec`. This avoids brittle multi-level shell quoting.
REMOTE_LAUNCHER_LOCAL="$LOGS_DIR/.bl2_text_launcher_${T}.sh"
REMOTE_LAUNCHER_HOME="$HOME/.bl2_text_launcher_${T}.sh"   # path inside the container (= host /home/zeyu mount)

cat > "$REMOTE_LAUNCHER_LOCAL" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# Activate the project mamba env if present (set MAMBA_ROOT_PREFIX in the
# container if you've put it elsewhere).
if command -v mamba >/dev/null 2>&1; then
    eval "\$(mamba shell hook --shell bash 2>/dev/null || true)" || true
    mamba activate mono_kernel || true
fi
mkdir -p "$NODE1_OUTPUTS_DIR" "$NODE1_LOGS_DIR"
cd "$WORKTREE"
export CUDA_VISIBLE_DEVICES=0
export MONO_KERNEL_BL1_METRICS_PATH="$TEXT_SERVER_SIDE_REMOTE"
export MONO_KERNEL_BL2_VEMB_PATH="$TEXT_VEMB_SIDE_REMOTE"
exec vllm serve "$MODEL" \\
    --port $TEXT_PORT \\
    --tensor-parallel-size 1 \\
    --max-num-seqs $MAX_NUM_SEQS \\
    --enable-request-id-headers \\
    --allowed-local-media-path "$INPUTS_DIR/assets" \\
    --ec-transfer-config '$EC_CFG_CONSUMER'
EOF
chmod +x "$REMOTE_LAUNCHER_LOCAL"

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

# Pre-flight: wait for both GPUs to be free (memory < 1 GiB used). Stale
# EngineCore can survive previous runs' SIGINT and hold model weights;
# without this, the next vllm serve fails with "Free memory < desired
# utilization".
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
wait_gpu_free "node 0" \
    "nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits"
wait_gpu_free "node 1" \
    "ssh -o BatchMode=yes $NODE1_HOST 'docker exec -u 0 mono_kernel \
        nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits'"

if (( DRY_RUN == 1 )); then
    echo "[run_bl2] DRY_RUN=1 — printing what would run, not invoking."
    echo "  EC_CFG_PRODUCER=$EC_CFG_PRODUCER"
    echo "  EC_CFG_CONSUMER=$EC_CFG_CONSUMER"
    echo "  VIS_SERVER_SIDE=$VIS_SERVER_SIDE"
    echo "  VIS_VEMB_SIDE=$VIS_VEMB_SIDE"
    echo "  TEXT_SERVER_SIDE_REMOTE=$TEXT_SERVER_SIDE_REMOTE"
    echo "  TEXT_VEMB_SIDE_REMOTE=$TEXT_VEMB_SIDE_REMOTE"
    echo "  Remote launcher staged at: $REMOTE_LAUNCHER_LOCAL"
    echo "  To actually run: DRY_RUN=0 bash $0"
    exit 0
fi

# Ports + 80s TIME_WAIT drain on both ends.
PORTS="$VIS_PORT $PROXY_PORT $NIXL_PORT" SLEEP_AFTER=${CLEAN_SLEEP:-80} \
    bash "$WORKTREE/mk_scripts/clean.sh" || true
ssh -o BatchMode=yes "$NODE1_HOST" "docker exec -u $NODE1_USER_UID mono_kernel \
    bash -lc 'PORTS=\"$TEXT_PORT\" SLEEP_AFTER=${CLEAN_SLEEP:-80} bash $WORKTREE/mk_scripts/clean.sh'" || true

# Push the launcher to node 1's home (host bind-mount) and start it inside the container.
scp -o BatchMode=yes "$REMOTE_LAUNCHER_LOCAL" "${NODE1_HOST}:${REMOTE_LAUNCHER_HOME}"
ssh -o BatchMode=yes "$NODE1_HOST" "docker exec -u $NODE1_USER_UID mono_kernel \
    bash -lc 'nohup bash $REMOTE_LAUNCHER_HOME > $TEXT_LOG_REMOTE 2>&1 &'"

# Launch vision instance on node 0 in the background.
env CUDA_VISIBLE_DEVICES=0 \
    MONO_KERNEL_BL1_METRICS_PATH="$VIS_SERVER_SIDE" \
    MONO_KERNEL_BL2_VEMB_PATH="$VIS_VEMB_SIDE" \
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
        > "$VIS_LOG" 2>&1 &
VIS_PID=$!

# Wait for both /health.
echo "[run_bl2] waiting for vision (:$VIS_PORT) and text ($NODE1_HOST:$TEXT_PORT) /health"
DEADLINE=$((SECONDS + 600))
vis_ok=0 text_ok=0
while (( SECONDS < DEADLINE )) && (( vis_ok == 0 || text_ok == 0 )); do
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
    tail -n 80 "$VIS_LOG" >&2 || true
    ssh -o BatchMode=yes "$NODE1_HOST" "tail -n 80 $TEXT_LOG_REMOTE" >&2 || true
    exit 1
fi

# Launch proxy.
python "$WORKTREE/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host 0.0.0.0 --port "$PROXY_PORT" \
    --encode-servers-urls "http://127.0.0.1:$VIS_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://${NODE1_HOST}:${TEXT_PORT}" \
    > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
sleep 5

# Run the client.
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
    --result-filename "$CLIENT_FILE_DEFAULT" \
    > "$CLIENT_LOG" 2>&1
rc=$?
set -e
kill -INT "$PROXY_PID" 2>/dev/null || true
if (( rc != 0 )); then
    echo "[run_bl2] bench client failed (rc=$rc); tail:" >&2
    tail -n 80 "$CLIENT_LOG" >&2 || true
    exit "$rc"
fi

# Pull node-1 sidecars back to node 0.
scp -o BatchMode=yes "${NODE1_HOST}:${TEXT_SERVER_SIDE_REMOTE}" "${OUTPUTS_DIR}/" || true
scp -o BatchMode=yes "${NODE1_HOST}:${TEXT_VEMB_SIDE_REMOTE}"    "${OUTPUTS_DIR}/" || true

echo "[run_bl2] merging metrics into $OUTPUTS_DIR/$FINAL_FILE"
python "$WORKTREE/mk_scripts/merge_metrics.py" \
    --client     "$OUTPUTS_DIR/$CLIENT_FILE_DEFAULT" \
    --server-vis "$VIS_SERVER_SIDE" \
    --server-vis-vemb "$VIS_VEMB_SIDE" \
    --server-text "$TEXT_SERVER_SIDE_LOCAL" \
    --server-text-vemb "$TEXT_VEMB_SIDE_LOCAL" \
    --inputs  "$DATASET_FILE" \
    --label disaggregation --workload "$WORKLOAD" --args "$ARGS_TAG" --time "$T" \
    --out     "$OUTPUTS_DIR/$FINAL_FILE"

echo "[run_bl2] done: $OUTPUTS_DIR/$FINAL_FILE"
