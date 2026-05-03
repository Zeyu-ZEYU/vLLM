#!/usr/bin/env bash
# Clean up residual vLLM / SGLang processes and free experiment ports.
#
# Run on the experiment node, before every experiment, inside the
# mono_kernel container as user zeyu.
#
# Optional env:
#   PORTS         space-separated port list to free, default "8000 8001"
#   SLEEP_AFTER   seconds to sleep after kill, default 80 (TIME_WAIT drain
#                 per CLAUDE.md). Set to 0 to skip.

set -uo pipefail

PORTS=${PORTS:-"8000 8001"}
SLEEP_AFTER=${SLEEP_AFTER:-80}
ME="$(id -un)"

echo "[clean] running as $ME on $(hostname)"

# 1) Snapshot residuals before killing.
echo "[clean] residual processes (this user only):"
ps -u "$ME" -o pid,stat,etime,cmd 2>/dev/null \
    | grep -E "vllm|sglang|^[ ]*[0-9]+ .*python.* (serve|bench)" \
    | grep -v "grep -E\|clean.sh" || echo "  (none)"

# 2) Kill vLLM and SGLang processes owned by this user.
#    Match patterns: `vllm serve`, `vllm bench`, `python -m vllm.*`, anything
#    with /sglang/ in the cmdline. Kill -INT first, escalate to -KILL.
PATTERNS=("vllm[[:space:]]*serve" "vllm[[:space:]]*bench" "python.*vllm" "sglang")
PIDS=()
for pat in "${PATTERNS[@]}"; do
    while IFS= read -r pid; do
        # don't kill ourselves or our parent shell
        if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "$PPID" ]]; then
            PIDS+=("$pid")
        fi
    done < <(pgrep -u "$ME" -f "$pat" 2>/dev/null || true)
done
# Dedup
if (( ${#PIDS[@]} > 0 )); then
    UNIQ_PIDS=$(printf "%s\n" "${PIDS[@]}" | sort -u)
    echo "[clean] killing pids: $(echo "$UNIQ_PIDS" | tr '\n' ' ')"
    for pid in $UNIQ_PIDS; do kill -INT "$pid" 2>/dev/null || true; done
    sleep 5
    for pid in $UNIQ_PIDS; do kill -TERM "$pid" 2>/dev/null || true; done
    sleep 5
    for pid in $UNIQ_PIDS; do kill -KILL "$pid" 2>/dev/null || true; done
else
    echo "[clean] no matching processes"
fi

# 3) Best-effort: kill anyone still holding the experiment ports (this user).
for port in $PORTS; do
    HOLDERS=$(ss -ltnp "sport = :$port" 2>/dev/null \
        | awk -F'pid=' 'NR>1 {print $2}' | awk -F',' '{print $1}' | sort -u || true)
    if [[ -n "$HOLDERS" ]]; then
        echo "[clean] port $port still held by pid(s): $HOLDERS — killing"
        for pid in $HOLDERS; do kill -KILL "$pid" 2>/dev/null || true; done
    fi
done

# 4) Final residual snapshot.
RESIDUAL=$(ps -u "$ME" -o pid,cmd 2>/dev/null \
    | grep -E "vllm|sglang" \
    | grep -v "grep -E\|clean.sh" || true)
if [[ -n "$RESIDUAL" ]]; then
    echo "[clean] WARNING: residual after cleanup:"
    echo "$RESIDUAL"
else
    echo "[clean] no residual vLLM/SGLang processes"
fi

# 5) GPU sanity check.
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$GPU_PROCS" ]]; then
        echo "[clean] GPU compute apps still attached:"
        echo "$GPU_PROCS"
    else
        echo "[clean] no GPU compute apps attached"
    fi
fi

# 6) Drain TIME_WAIT (per CLAUDE.md: 等待 80s 让 TIME_WAIT 消失).
if (( SLEEP_AFTER > 0 )); then
    echo "[clean] sleeping ${SLEEP_AFTER}s for TIME_WAIT drain"
    sleep "$SLEEP_AFTER"
fi

echo "[clean] done"
