#!/bin/bash
# ============================================================================
# Host-side cleanup — runs OUTSIDE the fe_rnic container with sudo.
#
# Motivation:
# `clean.sh` (the in-container cleanup) can kill most residual processes,
# but after certain failure modes (vLLM engine OOM / TCPStore crash / NCCL
# abort during DP init) Ray worker children get reparented to host-level
# init and end up in zombie (Z) state holding GPU memory. `kill -9` from
# inside the container doesn't reach them because the container's PID
# namespace has renumbered / detached them. They stay there until
# someone on the HOST side with sudo reaps them — or we reboot docker.
#
# This script is the "nuclear first step" before each experiment:
#   1. SIGKILL host-level Ray workers / DPMoEEngineCoreActor / VLLM::*
#      processes that nvidia-smi still counts as GPU-compute-apps.
#   2. Print the remaining GPU memory so the caller can sanity-check
#      before running clean.sh inside the container.
#
# Usage (runs on HOST, not in container):
#   bash clean_host.sh          # clean this host only
#   bash clean_host.sh --all    # also clean lj1/lj2/lj3 via ssh
#
# The container still needs its own `bash clean.sh` afterwards to drop
# /tmp/ray, /dev/shm/nccl*, /tmp/mooncake*-config.yaml, etc. The
# recommended per-experiment cleanup is:
#
#   # On host of each node, in sequence:
#   bash clean_host.sh    # sudo kill host-level zombies
#   docker exec -u zeyu fe_rnic bash -c 'cd /home/zeyu/vllm/fe_rnic/fe_rnic && bash clean.sh'
#   sleep 15              # let GPU driver reclaim
# ============================================================================

set -euo pipefail

cleanup_host() {
    echo "=== [host $(hostname)] 清理宿主级残留 ==="

    # 1. sudo pkill 各类 Ray / vLLM 子进程名
    local killed=0
    for pat in "ray::RayWorker" "DPMoEEngineCoreActor" "VLLM::" \
               "ray::IDLE" "ray::BaseWorkerTemplate" "default_worker.py" \
               "runtime_env/agent/main.py" "dashboard/agent.py" \
               "mooncake_master"; do
        # pgrep just to check; pkill to actually signal
        if sudo -n pgrep -f "$pat" >/dev/null 2>&1; then
            local n
            n=$(sudo -n pgrep -f "$pat" | wc -l)
            sudo -n pkill -9 -f "$pat" 2>/dev/null || true
            echo "  sudo pkill -9 [$pat]: $n 个"
            killed=$((killed + n))
        fi
    done

    # 2. Kill anything nvidia-smi reports as compute-app (last resort)
    if command -v nvidia-smi &>/dev/null; then
        local gpu_pids
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | tr -d '[:space:]' | tr -s ',' '\n' | grep -E '^[0-9]+$' || true)
        if [[ -n "$gpu_pids" ]]; then
            local n
            n=$(echo "$gpu_pids" | wc -w)
            echo "  sudo kill -9 [GPU-holders from nvidia-smi]: $n 个"
            echo "$gpu_pids" | xargs -r sudo -n kill -9 2>/dev/null || true
            killed=$((killed + n))
        fi
    fi

    # 3. Wait for GPU driver to release memory (matches CLAUDE.md guidance)
    if [[ $killed -gt 0 ]]; then
        echo "  等待 GPU 驱动回收显存 (5s)..."
        sleep 5
    fi

    # 4. Report
    echo "  残留 GPU compute-apps:"
    if command -v nvidia-smi &>/dev/null; then
        local remaining
        remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [[ $remaining -gt 0 ]]; then
            echo "    ⚠ 仍有 $remaining 个进程:"
            nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader 2>/dev/null | head -5
        else
            echo "    无"
        fi

        echo "  GPU 总内存占用:"
        nvidia-smi --query-gpu=index,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | \
            awk -F', ' '{printf "    GPU %s: %s/%s MiB\n", $1, $2, $3}' || true
    fi

    echo "=== [host $(hostname)] 清理完成 (killed=$killed) ==="
}

if [[ "${1:-}" == "--all" ]]; then
    echo "========== 宿主级清理所有节点 =========="
    echo ""
    cleanup_host
    echo ""
    for node in lj1.zeyu.tw lj2.zeyu.tw lj3.zeyu.tw; do
        echo "[${node}]"
        ssh -o ConnectTimeout=5 "zeyu@${node}" \
            "cd /home/zeyu/vllm/fe_rnic/fe_rnic && bash clean_host.sh" 2>/dev/null \
            || echo "  连接失败: $node"
        echo ""
    done
    echo "========== 全部清理完成 =========="
else
    cleanup_host
fi
