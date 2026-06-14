#!/usr/bin/env bash
# Tear down a Shunt run inside the container. Run before every experiment, then
# clean_host.sh, then wait ~60s for TIME_WAIT to drain (see README).
set -uo pipefail

echo "[clean] stopping vLLM / proxy / planner / samplers"
pkill -f "shunt.serving.proxy"        2>/dev/null || true
pkill -f "shunt.harness.node_planner" 2>/dev/null || true
pkill -f "shunt.harness.bw_sampler"   2>/dev/null || true
pkill -f "vllm serve"                 2>/dev/null || true
pkill -f "vllm.entrypoints"           2>/dev/null || true
pkill -f "EngineCore"                 2>/dev/null || true

echo "[clean] freeing GPUs held by stragglers"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
  kill -9 "$pid" 2>/dev/null || true
done

echo "[clean] clearing planner plan files + mooncake metadata"
rm -f /tmp/shunt/kvlb_plan_*.json 2>/dev/null || true
rm -rf /dev/shm/mooncake* 2>/dev/null || true

echo "[clean] done"
