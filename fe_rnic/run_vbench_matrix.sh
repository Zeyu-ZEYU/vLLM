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
# Run boundaries (mode, L, C, t_start, t_end as Unix epoch with ns):
#   /tmp/vbench_run_boundaries.tsv
#
# v3 (2026-04-30): 机尾三网卡 task — 6 lengths × 3 concurrencies × 2 modes,
#   750 prompts, output_len=2. Added P95 to percentile output. Records run
#   boundaries to /tmp/vbench_run_boundaries.tsv so d_prefill JSONL records
#   (which only carry per-request ts) can be bucketed back to (mode, L, C).
#
# Assumptions:
#   - The proxy is up and reachable on localhost:9090.
#   - The tokenizer path is valid locally (used client-side for random tokens).
#   - --random-output-len=2 — gives at minimum 1 ITL data point per request.
#   - --goodput is omitted (we don't gate on SLO here).
#   - TPOT is inflated ~33% by our PD-disagg proxy off-by-one (decode's
#     `usage.completion_tokens` doesn't count the head_chunk token from
#     prefill). Same in both modes, so head/tail comparison is fine.
#     TTFT, ITL, E2EL are unaffected.
# ============================================================================
set -euo pipefail

MODE="${1:?Usage: $0 <tail|head>}"
cd /home/zeyu/vLLM/v0.11.0/fe_rnic
mkdir -p /home/zeyu/exp_results/fe_rnic/bench_vllm

LENGTHS=(256 512 1024 2048 4096 8192)
CONCURRENCIES=(50 150 250)
MODEL=Qwen3-235B
TOKENIZER=/home/zeyu/models/Qwen3-235B-A22B
HOST=localhost
PORT=9090
OUTPUT_LEN=2
NUM_PROMPTS=750
OUT_DIR=/home/zeyu/exp_results/fe_rnic/bench_vllm
BOUNDARIES=/tmp/vbench_run_boundaries.tsv

# init boundaries file with header if not present
if [ ! -f "$BOUNDARIES" ]; then
    echo -e "mode\tL\tC\tt_start\tt_end" > "$BOUNDARIES"
fi

for C in "${CONCURRENCIES[@]}"; do
    for L in "${LENGTHS[@]}"; do
        TAG="vbench_${MODE}_conc${C}_L${L}"
        OUT_FILE="${TAG}.json"
        echo "========================================================"
        echo "[$(date +%H:%M:%S)] $TAG"
        echo "========================================================"
        T_START=$(date +%s.%N)
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
            --metric-percentiles "25,50,75,95,99" \
            --seed "$(date +%s)" \
            --trust-remote-code \
            --request-rate "$C" \
            --max-concurrency "$C" \
            --num-prompts "$NUM_PROMPTS" \
            --save-result \
            --result-dir "$OUT_DIR" \
            --result-filename "$OUT_FILE" 2>&1 | tail -40
        T_END=$(date +%s.%N)
        echo -e "${MODE}\t${L}\t${C}\t${T_START}\t${T_END}" >> "$BOUNDARIES"
        echo ""
        # short cooldown to avoid TIME_WAIT pile-up between runs
        sleep 30
    done
done
echo "[$(date +%H:%M:%S)] vbench-${MODE} all done"
echo "Run boundaries appended to $BOUNDARIES"
