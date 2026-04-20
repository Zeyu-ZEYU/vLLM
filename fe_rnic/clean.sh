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
        # ray::RayWorkerWrapper — the actual vLLM DP worker subprocesses
        # spawned by Ray. They hold the GPU memory. After a crashed prefill
        # kills the DPCoordinator, these workers get reparented to Ray's
        # raylet but stay alive (Ray doesn't cascade-kill on leader death).
        # Without this they survive `clean.sh` and we can't relaunch.
        "ray::RayWorker"
        "ray::IDLE"
        "DPMoEEngineCoreActor"
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

    # 最后兜底: 根据 nvidia-smi 找任何还占 GPU 显存的进程，强杀。
    # 这是兜底的兜底 —— 有时上面 pgrep/ps 模式匹配不到 Ray 派生的
    # python 子进程（进程名只是 "python3.13" 没有独特标志），但它们
    # 依然持着 CUDA ctx。这种 orphan 不杀，下次 `vllm serve` 申请显存
    # 会被现有占用挤爆甚至直接失败。
    if command -v nvidia-smi &>/dev/null; then
        local gpu_pids
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' | tr -s ',' '\n' | grep -E '^[0-9]+$' || true)
        if [[ -n "$gpu_pids" ]]; then
            echo "  杀掉 [GPU memory holders from nvidia-smi]: $(echo $gpu_pids | wc -w) 个"
            echo "$gpu_pids" | xargs -r kill -9 2>/dev/null || true
            killed=$((killed + $(echo "$gpu_pids" | wc -w)))
        fi
    fi

    # 清理 Ray
    if command -v ray &>/dev/null; then
        echo "  停止 Ray..."
        ray stop --force 2>/dev/null || true
    fi

    # 兜底：`ray stop` 偶尔因为内部 IPC socket 破损不能真正杀掉
    # raylet/gcs_server，它们会以老的 PID 一直挂着。新 `ray start --head`
    # 起来以后又"看见"残留进程带进来的 ghost node 记录，ray status
    # 里就会出现明明只启了一个节点却显示两三个 active 的 node ID。
    # 这个 ghost node 会让 vLLM DP 的 placement group 调度器算错 GPU
    # 总数（多半只落到 head node 8 卡），报
    # "Not enough resources to allocate 16 placement groups,
    #  only created 8 placement groups"。
    # 直接 SIGKILL 掉 raylet / gcs_server / default_worker，彻底
    # 断掉 IPC socket 的持有者，下次 ray start --head 才是真"从零开始"。
    pkill -9 raylet 2>/dev/null || true
    pkill -9 -f gcs_server 2>/dev/null || true
    pkill -9 -f default_worker.py 2>/dev/null || true
    pkill -9 -f dashboard/agent.py 2>/dev/null || true
    pkill -9 -f runtime_env/agent/main.py 2>/dev/null || true
    # Ray autoscaler/_private/monitor.py sometimes outlives ray stop when
    # a prior `kill -9 gcs_server` orphans it. It points at the dead GCS
    # address (192.168.0.42:6379 etc.) forever, eating a tiny bit of CPU
    # and more importantly polluting `ps` so we can't tell stale from live.
    pkill -9 -f "autoscaler/_private/monitor.py" 2>/dev/null || true
    # ray/_private/log_monitor.py also outlives `ray stop` when GCS was
    # killed with -9. Same reasoning as autoscaler monitor above.
    pkill -9 -f "_private/log_monitor.py" 2>/dev/null || true

    # 清理 /tmp/ray GCS 持久状态
    # `ray stop` 只停进程，不删 session_* 目录。残留的 session 会让下次
    # ray start --head 认为上次的 placement group / actor 还活着，从而
    # 在 vLLM 启动时报 "Created 24 DP placement groups, expected 16"
    # (前一次没清干净的 DP workers 留下的 placement group count 累加到了
    # 这次的)。所以一定要 rm -rf /tmp/ray。
    rm -rf /tmp/ray 2>/dev/null || true

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

    # 清理上一轮 benchmark 的 server metrics JSONL
    rm -f /home/zeyu/lmcache_metrics.jsonl 2>/dev/null || true
    rm -f /home/zeyu/lmcache_metrics_producer.jsonl 2>/dev/null || true
    rm -f /home/zeyu/lmcache_metrics_consumer.jsonl 2>/dev/null || true

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
