#!/bin/bash
# Node-side torchrun launcher (run INSIDE the fe_rnic container).
# Activates the venv, sets the back-end-RDMA NCCL env (mirrors ray_start.sh),
# then torchrun with the given node_rank.
# Usage: bash run_torchrun.sh <node_rank> <script.py> [script args...]
set -uo pipefail
NODE_RANK="$1"; shift
PY="$1"; shift
source /home/zeyu/vLLM/fe_rnic/.venv/bin/activate
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_IB_SL=5
export NCCL_IB_TC=138
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
cd /home/zeyu/vLLM/fe_rnic/fe_rnic
exec torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr=192.168.0.42 --master_port=29517 "$PY" "$@"
