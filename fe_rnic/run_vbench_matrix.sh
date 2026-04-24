#!/bin/bash
# ============================================================================
# Matrix runner using vLLM's built-in `vllm bench serve` (colleague-style).
# Loops over 6 input lengths × 4 concurrency levels.
#
# Usage:
#   bash run_vbench_matrix.sh <tail|head>
#
# Output:
#   /home/zeyu/exp_results/fe_rnic/bench_vllm/vbench_<mode>_conc<C>_L<L>.json
# plus a stdout log in /tmp/vbench_<mode>.log.
#
# Assumptions:
#   - The proxy is up and reachable on localhost:9090.
#   - The tokenizer path is valid locally (used client-side for random tokens).
#   - --num-prompts=1000 matches the colleague's scale.
#   - --random-output-len=5 matches the colleague (keeps the measurement focused
#     on prefill + KV transfer + first decode step).
#   - --goodput is omitted per user instruction.
#   - TPOT numbers will be inflated by ~33% because our proxy emits prefill's
#     first token as a synthesized head_chunk but decode's `usage` only counts
#     decode's own tokens. We intentionally leave this uncorrected to stay
#     apples-to-apples with the colleague's measurement (which has the same
#     off-by-one on his proxy). TTFT, ITL, E2EL are unaffected.
# ============================================================================
set -euo pipefail

MODE="${1:?Usage: $0 <tail|head>}"
cd /home/zeyu/vllm/fe_rnic/fe_rnic
mkdir -p /home/zeyu/exp_results/fe_rnic/bench_vllm

LENGTHS=(512 1024 2048 4096 8192 16384)
CONCURRENCIES=(50 100 150 200)
MODEL=Qwen3-235B
TOKENIZER=/home/zeyu/models/Qwen3-235B-A22B
HOST=localhost
PORT=9090
OUTPUT_LEN=5
NUM_PROMPTS=1000
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
