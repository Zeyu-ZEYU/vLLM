#!/bin/bash
# ============================================================================
# 清理残留进程
# 当 Ctrl+C 无法完全清理 vllm/mooncake/lmcache 相关进程时使用
#
# 用法:
#   bash clean.sh          # 清理本节点
#   bash clean.sh --all    # 清理所有 4 个节点 (需从 node 0 执行)
# ============================================================================

set -euo pipefail

cleanup_node() {
    echo "=== 清理残留进程 ==="

    # 按优先级从高到低清理
    local patterns=(
        "vllm serve"
        "vllm.entrypoints"
        "mooncake_master"
        "disagg_proxy_server"
        "multiproc_executor"
        "EngineCore"
        "Worker_DP"
        "lmcache"
    )

    local killed=0
    for pat in "${patterns[@]}"; do
        local pids
        pids=$(pgrep -f "$pat" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "  杀掉 [$pat]: $pids"
            echo "$pids" | xargs kill -9 2>/dev/null || true
            killed=$((killed + $(echo "$pids" | wc -w)))
        fi
    done

    # 清理可能残留的 python 多进程 worker（vllm spawn 出来的）
    local orphan_pids
    orphan_pids=$(pgrep -f "from multiprocessing.spawn" 2>/dev/null || true)
    if [[ -n "$orphan_pids" ]]; then
        echo "  杀掉 [multiprocessing spawn orphans]: $orphan_pids"
        echo "$orphan_pids" | xargs kill -9 2>/dev/null || true
        killed=$((killed + $(echo "$orphan_pids" | wc -w)))
    fi

    # 清理 VLLM::APIServer / DPCoordin / resource_tracker 僵尸进程
    # (这些进程不被上面的 pgrep -f 匹配到，因为进程名不含上述 pattern)
    local zombie_pids
    zombie_pids=$(ps aux | grep -E "VLLM::|DPCoordin|resource_tracker" | grep -v grep | awk '{print $2}' || true)
    if [[ -n "$zombie_pids" ]]; then
        echo "  杀掉 [VLLM/DPCoordin/resource_tracker zombies]: $(echo $zombie_pids | wc -w) 个"
        echo "$zombie_pids" | xargs kill -9 2>/dev/null || true
        killed=$((killed + $(echo "$zombie_pids" | wc -w)))
    fi

    # 清理 Ray
    if command -v ray &>/dev/null; then
        echo "  停止 Ray..."
        ray stop --force 2>/dev/null || true
    fi

    # 清理临时文件
    rm -f /tmp/engine_* 2>/dev/null || true

    # 清理共享内存 (NCCL / Mooncake 可能残留)
    local shm_files
    shm_files=$(find /dev/shm -user "$(whoami)" -name "nccl*" -o -user "$(whoami)" -name "mooncake*" 2>/dev/null || true)
    if [[ -n "$shm_files" ]]; then
        echo "  清理共享内存: $(echo "$shm_files" | wc -w) 个文件"
        echo "$shm_files" | xargs rm -f 2>/dev/null || true
    fi

    # 清理临时 mooncake config
    rm -f /tmp/mooncake-prefiller-config.yaml /tmp/mooncake-decoder-config.yaml 2>/dev/null || true

    # 清理上一轮 benchmark 的 server metrics JSONL (prefill worker 写的)
    rm -f /home/zeyu/lmcache_metrics.jsonl 2>/dev/null || true

    if [[ $killed -eq 0 ]]; then
        echo "  没有发现残留进程"
    else
        echo "  共清理 $killed 个进程"
    fi

    # 释放 GPU 显存 (等一下让驱动回收)
    sleep 1
    echo "  GPU 状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "    GPU %s: %s/%s MiB\n", $1, $2, $3}' || echo "    nvidia-smi 不可用"

    echo "=== 清理完成 ==="
}

if [[ "${1:-}" == "--all" ]]; then
    echo "========== 清理所有节点 =========="
    echo ""
    echo "[node 0 - $(hostname)]"
    cleanup_node
    echo ""
    for node in lj1.zeyu.tw lj2.zeyu.tw lj3.zeyu.tw; do
        echo "[${node}]"
        ssh -o ConnectTimeout=5 "zeyu@${node}" "docker exec -u zeyu fe_rnic bash -c 'export PATH=/home/zeyu/miniforge3/envs/fe_rnic/bin:\$PATH && cd /home/zeyu/vllm/fe_rnic/fe_rnic && bash clean.sh'" 2>/dev/null || echo "  连接失败: $node"
        echo ""
    done
    echo "========== 全部清理完成 =========="
else
    cleanup_node
fi
