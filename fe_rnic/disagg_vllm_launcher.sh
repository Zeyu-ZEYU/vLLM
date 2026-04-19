#!/bin/bash
# ============================================================================
# Disaggregated PD Serving — Qwen3-235B-A22B
# vLLM v1 + LMCache + Mooncake (RDMA)
#
# 架构:
#   Prefill : 2 节点 × 8 GPU = 16 GPU   DP=16, TP=1, EP=16 (auto)
#             使用 Ray 作为 DP backend（需先运行 ray_start.sh）
#   Decode  : 1-2 节点 × 8 GPU          DP=8, TP=1, EP=8 (auto)
#             单节点独立运行，不需要 Ray
#   Master  : mooncake_master (通过 master.sh 单独启动)
#   Proxy   : disagg_proxy_server.py
#
# 必需环境变量:
#   MASTER_IP      — mooncake master 节点 IPv4 (eth0)
#   LOCAL_IP       — 当前节点 IPv4 (eth0)
#   LOCAL_RDMA_IP  — 当前节点 IPv6 RDMA 地址 (bond 接口)
#
# 启动顺序:
#   1. 在 node 0:     bash master.sh
#   2. 在 node 0:     bash ray_start.sh head
#   3. 在 node 1:     bash ray_start.sh worker <node0_ip>
#   4. 在 node 0:     bash disagg_vllm_launcher.sh prefill
#   5. 在 node 2:     bash ray_start.sh head (MOONCAKE_ROLE=decoder)
#   5b.在 node 2:     bash disagg_vllm_launcher.sh decode
#   6. (可选) node 3: 同 step 5 (ray_start.sh head + disagg_vllm_launcher.sh decode)
#   7. 任意节点:       bash disagg_vllm_launcher.sh proxy
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======================== 默认配置 ========================
MODEL="${MODEL:-$HOME/models/Qwen3-235B-A22B}"
IB_DEVICES="${IB_DEVICES:-mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3}"

PREFILL_PORT="${PREFILL_PORT:-8100}"
PREFILL_DP="${PREFILL_DP:-16}"
PREFILL_DP_LOCAL="${PREFILL_DP_LOCAL:-8}"
PREFILL_TP="${PREFILL_TP:-1}"

DECODE_PORT="${DECODE_PORT:-8200}"
DECODE_DP="${DECODE_DP:-8}"
DECODE_TP="${DECODE_TP:-1}"

PROXY_PORT="${PROXY_PORT:-9090}"

# KV overlap: layerwise KV transfer overlapping with prefill computation
ENABLE_KV_OVERLAP="${ENABLE_KV_OVERLAP:-false}"

# Head NIC splitting: route some KV chunks via management RNIC (mlx5_0)
ENABLE_HEAD_NIC_SPLIT="${ENABLE_HEAD_NIC_SPLIT:-false}"

export PYTHONHASHSEED="${VLLM_PYTHON_HASH_SEED:-123}"

# ======================== 工具函数 ========================
check_env() {
    local var_name=$1
    if [[ -z "${!var_name:-}" ]]; then
        echo "ERROR: 环境变量 $var_name 未设置"
        exit 1
    fi
}

generate_mooncake_config() {
    local role=$1
    local template="$SCRIPT_DIR/configs/mooncake-${role}-config.yaml"
    local output="/tmp/mooncake-${role}-config.yaml"

    sed -e "s|{MASTER_IP}|${MASTER_IP}|g" \
        -e "s|{LOCAL_RDMA_IP}|${LOCAL_RDMA_IP}|g" \
        -e "s|{IB_DEVICES}|${IB_DEVICES}|g" \
        "$template" > "$output"

    echo "$output"
}

usage() {
    cat <<'EOF'
Usage:
  disagg_vllm_launcher.sh <role>

Roles:
  prefill   在 prefill-node-0 上启动 vLLM prefiller (需先在两节点启动 Ray)
  decode    在 decode 节点上启动 vLLM decoder（每个 decode 节点各运行一次）
  proxy     启动 disagg proxy server

必需环境变量见脚本头部注释。
EOF
    exit 1
}

# ======================== 入口 ========================
if [[ $# -lt 1 ]]; then
    usage
fi

ROLE="$1"

case "$ROLE" in

# ------------------------------------------------------------------
# Prefill（在 prefill-node-0 上运行，Ray 已在两节点启动）
# ------------------------------------------------------------------
prefill)
    check_env MASTER_IP
    check_env LOCAL_IP
    check_env LOCAL_RDMA_IP

    config_file=$(generate_mooncake_config prefiller)
    echo "[prefill] 生成 mooncake config: $config_file"
    echo "[prefill] 模型: $MODEL"
    echo "[prefill] DP=$PREFILL_DP (local=$PREFILL_DP_LOCAL)  TP=$PREFILL_TP  EP=auto"
    echo "[prefill] 管理网络 (IPv4): $LOCAL_IP"
    echo "[prefill] RDMA 网络 (IPv6): $LOCAL_RDMA_IP"
    echo "[prefill] IB 设备: $IB_DEVICES"
    echo "[prefill] DP backend: Ray"

    # Build kv-transfer-config JSON
    KV_EXTRA='{"discard_partial_chunks":false,"lmcache_rpc_port":"producer1"'
    if [[ "$ENABLE_KV_OVERLAP" == "true" ]]; then
        KV_EXTRA="${KV_EXTRA},\"lmcache.use_layerwise\":true"
        # Opt into the zero-copy put/get path (batch_put_from / batch_get_into)
        # so layerwise per-chunk Put doesn't pay the put_parts metadata penalty
        # and can RDMA-WRITE directly to the decode segment via preferred_segment.
        KV_EXTRA="${KV_EXTRA},\"save_chunk_meta\":false"
        echo "[prefill] KV overlap 已启用 (layerwise, zero-copy Mooncake Put)"
    fi
    if [[ "$ENABLE_HEAD_NIC_SPLIT" == "true" ]]; then
        HEAD_CFG="${LMCACHE_HEAD_NIC_CONFIG_FILE:-/tmp/mooncake-head-nic-config.yaml}"
        KV_EXTRA="${KV_EXTRA},\"lmcache.enable_head_nic_split\":true"
        KV_EXTRA="${KV_EXTRA},\"lmcache.head_nic_config_file\":\"${HEAD_CFG}\""
        echo "[prefill] Head NIC 分流已启用 (config=$HEAD_CFG)"
    fi
    KV_EXTRA="${KV_EXTRA}}"
    KV_CONFIG="{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":${KV_EXTRA}}"

    # 环境变量（已由 ray_start.sh export，这里再设一次确保 vllm serve 也有）
    UCX_TLS=all \
    LMCACHE_CONFIG_FILE="$config_file" \
    LMCACHE_USE_EXPERIMENTAL=True \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    vllm serve "$MODEL" \
        --host "$LOCAL_IP" \
        --port "$PREFILL_PORT" \
        --enable-expert-parallel \
        -dp "$PREFILL_DP" \
        -dpl "$PREFILL_DP_LOCAL" \
        --data-parallel-address "$LOCAL_IP" \
        --data-parallel-backend ray \
        -tp "$PREFILL_TP" \
        --trust-remote-code \
        --served-model-name Qwen3-235B \
        --seed 1024 \
        --dtype bfloat16 \
        --enforce-eager \
        --max-model-len 16384 \
        --max-num-batched-tokens 16384 \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.9 \
        --no-enable-prefix-caching \
        --kv-transfer-config "$KV_CONFIG"
    ;;

# ------------------------------------------------------------------
# Decode（每个 decode 节点独立运行，需先在本节点启动 Ray head）
# ------------------------------------------------------------------
decode)
    check_env MASTER_IP
    check_env LOCAL_IP
    check_env LOCAL_RDMA_IP

    config_file=$(generate_mooncake_config decoder)
    echo "[decode] 生成 mooncake config: $config_file"
    echo "[decode] 模型: $MODEL"
    echo "[decode] DP=$DECODE_DP  TP=$DECODE_TP  EP=auto"
    echo "[decode] 管理网络 (IPv4): $LOCAL_IP  端口: $DECODE_PORT"
    echo "[decode] RDMA 网络 (IPv6): $LOCAL_RDMA_IP"
    echo "[decode] IB 设备: $IB_DEVICES"
    echo "[decode] DP backend: Ray (单节点)"

    # Build decode kv-transfer-config
    DEC_KV_EXTRA='{"discard_partial_chunks":false,"lmcache_rpc_port":"consumer1"'
    if [[ "$ENABLE_KV_OVERLAP" == "true" ]]; then
        # Decode must match prefill: per-layer chunks + zero-copy get
        # (batch_get_into) with per-layer meta_shapes.
        DEC_KV_EXTRA="${DEC_KV_EXTRA},\"lmcache.use_layerwise\":true"
        DEC_KV_EXTRA="${DEC_KV_EXTRA},\"save_chunk_meta\":false"
        echo "[decode] KV overlap 已启用 (layerwise, zero-copy Mooncake Get)"
    fi
    if [[ "$ENABLE_HEAD_NIC_SPLIT" == "true" ]]; then
        HEAD_CFG="${LMCACHE_HEAD_NIC_CONFIG_FILE:-/tmp/mooncake-head-nic-config.yaml}"
        DEC_KV_EXTRA="${DEC_KV_EXTRA},\"lmcache.enable_head_nic_split\":true"
        DEC_KV_EXTRA="${DEC_KV_EXTRA},\"lmcache.head_nic_config_file\":\"${HEAD_CFG}\""
        echo "[decode] Head NIC 分流已启用 (config=$HEAD_CFG)"
    fi
    DEC_KV_EXTRA="${DEC_KV_EXTRA}}"
    DEC_KV_CONFIG="{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":${DEC_KV_EXTRA}}"

    UCX_TLS=all \
    LMCACHE_CONFIG_FILE="$config_file" \
    LMCACHE_USE_EXPERIMENTAL=True \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    vllm serve "$MODEL" \
        --host "$LOCAL_IP" \
        --port "$DECODE_PORT" \
        --enable-expert-parallel \
        -dp "$DECODE_DP" \
        -tp "$DECODE_TP" \
        --data-parallel-address "$LOCAL_IP" \
        --data-parallel-backend ray \
        --trust-remote-code \
        --served-model-name Qwen3-235B \
        --seed 1024 \
        --dtype bfloat16 \
        --enforce-eager \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.9 \
        --no-enable-prefix-caching \
        --kv-transfer-config "$DEC_KV_CONFIG"
    ;;

# ------------------------------------------------------------------
# Proxy
# ------------------------------------------------------------------
proxy)
    check_env PREFILL_PRIMARY_IP
    check_env DECODE_IPS

    IFS=',' read -ra DECODE_ARRAY <<< "$DECODE_IPS"
    NUM_DECODERS=${#DECODE_ARRAY[@]}
    DECODE_HOSTS="${DECODE_IPS}"
    DECODE_PORTS=$(printf "${DECODE_PORT}%.0s," $(seq 1 "$NUM_DECODERS") | sed 's/,$//')

    echo "[proxy] Prefiller: ${PREFILL_PRIMARY_IP}:${PREFILL_PORT}"
    echo "[proxy] Decoders ($NUM_DECODERS): ${DECODE_HOSTS} port=${DECODE_PORT}"

    RDMA_HOST_ARG=""
    if [[ -n "${DECODE_RDMA_IPS:-}" ]]; then
        RDMA_HOST_ARG="--decoder-rdma-host $DECODE_RDMA_IPS"
        echo "[proxy] Decode RDMA hosts: ${DECODE_RDMA_IPS}"
    fi

    python3 "$SCRIPT_DIR/disagg_proxy_server.py" \
        --host "0.0.0.0" \
        --port "$PROXY_PORT" \
        --prefiller-host "$PREFILL_PRIMARY_IP" \
        --prefiller-port "$PREFILL_PORT" \
        --num-prefillers 1 \
        --decoder-host "$DECODE_HOSTS" \
        --decoder-port "$DECODE_PORTS" \
        --num-decoders "$NUM_DECODERS" \
        $RDMA_HOST_ARG
    ;;

# ------------------------------------------------------------------
*)
    echo "无效角色: $ROLE"
    usage
    ;;

esac
