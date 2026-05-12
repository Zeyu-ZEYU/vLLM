#!/bin/bash
# ============================================================================
# 机尾 RNIC HW QoS ratelimit (mlnx_qos)
#
# Usage:
#   sudo bash qos_limit.sh <rate_Gbps>     # e.g. 50 → 50 Gbps per priority
#   sudo bash qos_limit.sh 0               # unlimited (restore)
#
# 仅限速机尾 4 bond 的 8 个 phys port: reth0..reth7
#   bond0 = reth0+reth1,  bond1 = reth2+reth3
#   bond2 = reth4+reth5,  bond3 = reth6+reth7
#
# 机头 mlx5_0 对应 netdev = eth0 (VPC underlay)，本脚本不动 eth0 → 机头永远 unlimited.
#
# mlnx_qos --ratelimit 单位是 Gbps (Gbit/s)。8 个值 = 8 个 priority queue (prio 0..7)，
# 全 0 表示 unlimited。本脚本对 8 prio 都设同一值（任务要求所有 priority 都限速）。
#
# 注意：mlnx_qos 操作的是 NIC firmware 级 HW QoS（DCB netlink → mlx5_core
# driver → NIC ASIC 寄存器 QETCR/QEEC 等），状态在 host kernel 之下，
# **无 namespace 隔离**。host/容器/所有用户共享同一物理 NIC，限速影响所有进程。
# 实验结束必须用 `bash qos_limit.sh 0` 恢复，否则 host 上所有后续用户继续被限速。
# ============================================================================

set -euo pipefail

RATE="${1:?Usage: $0 <rate_Gbps> (0=unlimited)}"
PORTS=(reth0 reth1 reth2 reth3 reth4 reth5 reth6 reth7)
RL="${RATE},${RATE},${RATE},${RATE},${RATE},${RATE},${RATE},${RATE}"

echo "[qos] Setting ratelimit=$RL on: ${PORTS[*]}"
for p in "${PORTS[@]}"; do
    echo "[qos] $p --ratelimit=$RL"
    sudo -n mlnx_qos -i "$p" --ratelimit="$RL"
done

echo ""
echo "[verify] Current state:"
for p in "${PORTS[@]}"; do
    echo "--- $p ---"
    sudo -n mlnx_qos -i "$p" 2>&1 | grep -A1 -E "ratelimit" || true
done
