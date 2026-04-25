#!/bin/bash
# ============================================================================
# Matrix runner using vLLM's built-in `vllm bench serve`.
# Loops over input lengths × concurrency levels.
#
# Usage:
#   bash run_vbench_matrix.sh <tail|head>
#
# Output:
#   /home/zeyu/exp_results/fe_rnic/bench_vllm/vbench_<mode>_conc<C>_L<L>.json
# plus a stdout log in /tmp/vbench_<mode>.log when invoked in background.
#
# v2 (2026-04-26): focus on long-context high-load regime where head/tail
#   differs most. Cut to 2 lengths × 5 concurrencies, bumped --num-prompts
#   to 600 for tighter percentile estimates.
#
# Assumptions:
#   - The proxy is up and reachable on localhost:9090.
#   - The tokenizer path is valid locally (used client-side for random tokens).
#   - --random-output-len=5 — ITL gives 4 data points per request (ITL[0]
#     contains the cross-node KV pull; ITL[1..3] are pure decode forward).
#   - --goodput is omitted (we don't gate on SLO here).
#   - TPOT is inflated ~33% by our PD-disagg proxy off-by-one (decode's
#     `usage.completion_tokens` doesn't count the head_chunk token from
#     prefill). Same in both modes, so head/tail comparison is fine.
#     TTFT, ITL, E2EL are unaffected.
# ============================================================================
set -euo pipefail

MODE="${1:?Usage: $0 <tail|head>}"
cd /home/zeyu/vllm/fe_rnic/fe_rnic
mkdir -p /home/zeyu/exp_results/fe_rnic/bench_vllm

LENGTHS=(8192 16384)
CONCURRENCIES=(50 100 150 200 300)
MODEL=Qwen3-235B
TOKENIZER=/home/zeyu/models/Qwen3-235B-A22B
HOST=localhost
PORT=9090
OUTPUT_LEN=5
NUM_PROMPTS=600
OUT_DIR=/home/zeyu/exp_results/fe_rnic/bench_vllm

for C in "${CONCURRENCIES[@]}"; do
    for L in "${LENGTHS[@]}"; do
        TAG="vbench_${MODE}_conc${C}_L${L}"
        OUT_FILE="${TAG}.json"
        echo "========================================================"
        echo "[$(date +%H:%M:%S)] $TAG"
        echo "========================================================"
        vllm bench serve \
            --backend vllm \
            --model "$MODEL" \
            --tokenizer "$TOKENIZER" \
            --dataset-name random \
            --host "$HOST" \
            --port "$PORT" \
            --random-input-len "$L" \
            --random-output-len "$OUTPUT_LEN" \
            --random-range-ratio 0 \
            --burstiness 1 \
            --percentile-metrics "ttft,tpot,itl,e2el" \
            --metric-percentiles "25,50,75,99" \
            --seed "$(date +%s)" \
            --trust-remote-code \
            --request-rate "$C" \
            --max-concurrency "$C" \
            --num-prompts "$NUM_PROMPTS" \
            --save-result \
            --result-dir "$OUT_DIR" \
            --result-filename "$OUT_FILE" 2>&1 | tail -30
        echo ""
    done
done
echo "[$(date +%H:%M:%S)] vbench-${MODE} all done"
