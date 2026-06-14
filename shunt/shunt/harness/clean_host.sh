#!/usr/bin/env bash
# Host-level cleanup, run AFTER clean.sh (which runs inside the container). Sweeps
# orphaned engine/Ray processes the container teardown can miss, then you should
# wait ~60s for TIME_WAIT sockets to drain before the next run (see README).
set -uo pipefail

echo "[clean_host] killing orphan vLLM / planner / proxy processes"
for pat in "vllm serve" "EngineCore" "shunt.serving.proxy" \
           "shunt.harness.node_planner" "shunt.harness.bw_sampler"; do
  pkill -9 -f "$pat" 2>/dev/null || true
done

echo "[clean_host] open sockets still in TIME_WAIT (wait ~60s before re-running):"
ss -tan 2>/dev/null | grep -c TIME-WAIT || true

echo "[clean_host] done"
