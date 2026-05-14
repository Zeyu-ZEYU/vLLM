#!/bin/bash
# ============================================================================
# Ray 集群启动脚本
# 在每个 prefill 节点上运行，导出环境变量后启动 Ray
#
# 用法:
#   prefill-node-0:  bash ray_start.sh head
#   prefill-node-1:  bash ray_start.sh worker <head_node_ip>
#
# 必需环境变量:
#   MASTER_IP      — mooncake master 节点 IPv4
#   LOCAL_IP       — 当前节点 IPv4 (用于 Ray node-ip-address)
#   LOCAL_RDMA_IP  — 当前节点 IPv6 RDMA 地址 (bond 接口)
#
# 关键: 所有环境变量必须在 ray start 前 export，Ray worker 会继承
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======================== 必需环境变量检查 ========================
for var in MASTER_IP LOCAL_IP LOCAL_RDMA_IP; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: 环境变量 $var 未设置"
        exit 1
    fi
done

# ======================== 默认配置 ========================
MODEL="${MODEL:-$HOME/models/Qwen3-235B-A22B}"
IB_DEVICES="${IB_DEVICES:-mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3}"
RAY_PORT="${RAY_PORT:-6379}"
ROLE="${1:-}"

# ======================== 生成 Mooncake config ========================
# 根据角色决定生成 prefiller 还是 decoder config
MOONCAKE_ROLE="${MOONCAKE_ROLE:-prefiller}"
template="$SCRIPT_DIR/configs/mooncake-${MOONCAKE_ROLE}-config.yaml"
output="/tmp/mooncake-${MOONCAKE_ROLE}-config.yaml"

if [[ -f "$template" ]]; then
    sed -e "s|{MASTER_IP}|${MASTER_IP}|g" \
        -e "s|{LOCAL_RDMA_IP}|${LOCAL_RDMA_IP}|g" \
        -e "s|{IB_DEVICES}|${IB_DEVICES}|g" \
        "$template" > "$output"
    echo "[ray_start] 生成 mooncake config: $output"
else
    echo "WARNING: 模板文件 $template 不存在，跳过 mooncake config 生成"
    output=""
fi

# ======================== 导出环境变量（Ray worker 会继承） ========================
# v4 (2026-05-14): 机尾三网卡 v4 — 所有 NCCL/GLOO env 注释掉，让 NCCL 用默认值。
# 配合 node 0 物理禁用 mlx5_bond_3（ip link set reth6/reth7 down）实现"非环境变量方式"
# 限制 node 0 用 3 bond。
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Gloo (Ray DP control plane) 必须显式 export，否则默认 127.0.0.1，跨 node DP>=2 connectFullMesh fail
export GLOO_SOCKET_IFNAME=eth0
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_HCA="${NCCL_IB_HCA:-${IB_DEVICES}}"
# export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_MIN_NCHANNELS=4
# export NCCL_IB_SL=5
# export NCCL_IB_TC=138
# export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
# export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# UCX
export UCX_TLS=all

# LMCache / Mooncake
if [[ -n "$output" ]]; then
    export LMCACHE_CONFIG_FILE="$output"
fi
export LMCACHE_USE_EXPERIMENTAL=True

# Head NIC splitting: generate head config if enabled
ENABLE_HEAD_NIC_SPLIT="${ENABLE_HEAD_NIC_SPLIT:-false}"
if [[ "$ENABLE_HEAD_NIC_SPLIT" == "true" ]]; then
    head_template="$SCRIPT_DIR/configs/mooncake-head-nic-config.yaml"
    head_output="/tmp/mooncake-head-nic-config.yaml"
    LOCAL_HEAD_IP="${LOCAL_HEAD_IP:-${LOCAL_IP}}"
    if [[ -f "$head_template" ]]; then
        sed -e "s|{MASTER_IP}|${MASTER_IP}|g" \
            -e "s|{LOCAL_HEAD_IP}|${LOCAL_HEAD_IP}|g" \
            "$head_template" > "$head_output"
        export LMCACHE_HEAD_NIC_CONFIG_FILE="$head_output"
        echo "[ray_start] Head NIC config: $head_output (device=mlx5_0)"
    fi
fi

# vLLM multiprocessing
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Hash seed（prefill / decode 必须一致）
export PYTHONHASHSEED="${VLLM_PYTHON_HASH_SEED:-123}"

echo "[ray_start] 环境变量已导出（v4: NCCL/GLOO env 全部注释，由 NCCL 取默认值）"
echo "[ray_start] LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE:-not set}"

# ======================== 启动 Ray ========================
case "$ROLE" in
head)
    ray start --head --port="$RAY_PORT" --node-ip-address="$LOCAL_IP"
    echo "[ray_start] Ray head 已启动: $LOCAL_IP:$RAY_PORT"
    ;;
worker)
    HEAD_IP="${2:?ERROR: worker 模式需要指定 head IP，用法: bash ray_start.sh worker <head_ip>}"
    ray start --address="${HEAD_IP}:${RAY_PORT}" --node-ip-address="$LOCAL_IP"
    echo "[ray_start] Ray worker 已加入集群: ${HEAD_IP}:${RAY_PORT}"
    ;;
*)
    echo "用法: bash ray_start.sh <head|worker> [head_ip]"
    echo "  head    — 在 prefill-node-0 上启动 Ray head"
    echo "  worker  — 在 prefill-node-1 上加入 Ray 集群"
    exit 1
    ;;
esac
