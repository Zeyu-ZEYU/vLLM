#!/bin/bash
# ============================================================================
# Wait for zeyu's TIME_WAIT sockets to drain before launching vllm.
#
# Motivation:
# Between experiments, ephemeral ports used by previously-killed vllm /
# ZMQ / Ray processes sit in TIME_WAIT for ~60 s. If vLLM's per-engine
# port picker happens to pick one of those for a new ApiServer's
# internal ZMQ socket, bind fails with
#   zmq.error.ZMQError: Address already in use
# and the whole prefill wedges with one dead ApiServer.
#
# We explicitly refuse to patch upstream's bind() code (it's tricky to
# add SO_REUSEADDR safely — pyzmq doesn't expose zmq.REUSEADDR on all
# builds, and the retry-with-new-port variant is a correctness bug).
# Instead, we wait for the window to pass.
#
# Behavior:
#   - Counts TIME_WAIT sockets whose local port is in the ephemeral
#     range and whose uid resolves to zeyu (or whose process, before
#     it died, was zeyu's).
#   - Blocks until count == 0, or until MAX_WAIT seconds (default 90).
#   - Returns 0 regardless — the caller can launch anyway if the wait
#     timed out (we don't want to block the experiment indefinitely).
#
# Usage:
#   bash wait_for_tw_drain.sh            # wait up to 90s on this host
#   bash wait_for_tw_drain.sh --all      # fan out across lj1/lj2/lj3
#
# Designed to be called on the HOST before `docker exec ... vllm serve`.
# ============================================================================

set -euo pipefail

MAX_WAIT="${MAX_WAIT:-90}"
# Ephemeral port range (Linux default 32768-60999). We narrow to the
# range ZMQ / vLLM pickers actually use (which is the whole ephemeral
# range). Broaden as needed.
EPHEM_LOW="${EPHEM_LOW:-32768}"
EPHEM_HIGH="${EPHEM_HIGH:-65535}"

count_tw_zeyu() {
    # ss -H suppresses the header. `state time-wait` filters TIME_WAIT.
    # We look at the local address's port and uid.
    # -n: numeric ports, -t: tcp
    # We can't directly get uid from TIME_WAIT sockets in ss output —
    # the process has exited. But the TIME_WAIT itself is bound to the
    # kernel, not a pid. As a proxy, count all TIME_WAIT in the
    # ephemeral range; these were almost certainly from zeyu on this
    # workload (root processes wouldn't use ephemeral ports this way).
    ss -tan state time-wait 2>/dev/null | awk -v lo="$EPHEM_LOW" -v hi="$EPHEM_HIGH" '
        NR > 1 {
            # Local address is the 3rd column (after State, Recv-Q, Send-Q)
            n = split($3, parts, ":")
            port = parts[n]
            if (port ~ /^[0-9]+$/ && port+0 >= lo && port+0 <= hi) count++
        }
        END { print count + 0 }
    '
}

wait_on_host() {
    local host="$(hostname)"
    local elapsed=0
    local interval=3
    local initial
    initial=$(count_tw_zeyu)
    echo "[wait-tw $host] 初始 TIME_WAIT (ephem $EPHEM_LOW-$EPHEM_HIGH): $initial"

    if [[ "$initial" -eq 0 ]]; then
        echo "[wait-tw $host] 无 TIME_WAIT 端口，直接继续"
        return 0
    fi

    while [[ $elapsed -lt $MAX_WAIT ]]; do
        sleep $interval
        elapsed=$((elapsed + interval))
        local n
        n=$(count_tw_zeyu)
        echo "[wait-tw $host] t=${elapsed}s TIME_WAIT=$n"
        if [[ $n -eq 0 ]]; then
            echo "[wait-tw $host] drain 完成 (用时 ${elapsed}s)"
            return 0
        fi
    done

    local final
    final=$(count_tw_zeyu)
    echo "[wait-tw $host] ⚠ 等了 ${MAX_WAIT}s 仍有 $final 个 TIME_WAIT — 继续（可能仍会撞端口）"
    return 0
}

if [[ "${1:-}" == "--all" ]]; then
    echo "========== 等所有节点 TIME_WAIT drain =========="
    wait_on_host
    for node in lj1.zeyu.tw lj2.zeyu.tw lj3.zeyu.tw; do
        echo ""
        ssh -o ConnectTimeout=5 "zeyu@${node}" \
            "cd /home/zeyu/vllm/fe_rnic/fe_rnic && MAX_WAIT=${MAX_WAIT} bash wait_for_tw_drain.sh" \
            2>/dev/null || echo "[wait-tw $node] 连接失败"
    done
    echo ""
    echo "========== 全部 drain 完成 =========="
else
    wait_on_host
fi
