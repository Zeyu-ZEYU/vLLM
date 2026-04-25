#!/bin/bash
# ============================================================================
# Matrix runner using vLLM's built-in `vllm bench serve`.
# Loops over 6 input lengths × 4 concurrency levels.
#
# Usage:
#   bash run_vbench_matrix.sh <tail|head>
#
# Per-run output (in /home/zeyu/exp_results/fe_rnic/bench_vllm/):
#   vbench_<mode>_conc<C>_L<L>.json                — vllm bench result
#   jsonl/<mode>/conc<C>/L<L>/producer_node0.jsonl — adapter timestamps
#   jsonl/<mode>/conc<C>/L<L>/producer_lj1.jsonl
#   jsonl/<mode>/conc<C>/L<L>/consumer_lj2.jsonl
#   jsonl/<mode>/conc<C>/L<L>/consumer_lj3.jsonl
# Stdout is logged to /tmp/vbench_<mode>.log when invoked in background.
#
# Per-iteration flow:
#   1. Truncate JSONLs on all 4 nodes (clean slate per run).
#   2. Run vllm bench serve.
#   3. Collect JSONLs into the per-run directory above.
# Aggregation (vbench JSON + JSONLs → mean/p25/p50/p75/p99 for both
# client- and server-side metrics) is done post-hoc by
# `agg_vbench_results.py`.
#
# Assumptions:
#   - The proxy is up and reachable on localhost:9090.
#   - SSH from inside this container to `zeyu@<lj{1,2,3} IPv4>` works
#     without password (BatchMode + StrictHostKeyChecking=no, same as
#     benchmark.py used). Internal LAN, no jump host needed.
#   - --num-prompts=400 keeps each (length × concurrency) point's
#     wall-clock under a few minutes; total matrix ~1-1.5 hr per mode.
#   - --random-output-len=5: ITL gives 4 data points per request — ITL[0]
#     is the cross-node KV pull dominated one; ITL[1..3] are pure decode
#     forward steps.
#   - --goodput is omitted (we don't gate on SLO here).
#   - The TPOT off-by-one (decode's `usage.completion_tokens` doesn't
#     count the head_chunk token from prefill) inflates client-side
#     TPOT ~33% in both modes. Kept to match the colleague's reference
#     setup. TTFT/ITL/E2EL unaffected; server-side d_* unaffected.
# ============================================================================
set -euo pipefail

MODE="${1:?Usage: $0 <tail|head>}"
cd /home/zeyu/vllm/fe_rnic/fe_rnic

OUT_DIR=/home/zeyu/exp_results/fe_rnic/bench_vllm
JSONL_BASE=$OUT_DIR/jsonl
mkdir -p "$OUT_DIR" "$JSONL_BASE"

LENGTHS=(512 1024 2048 4096 8192 16384)
CONCURRENCIES=(50 100 150 200)
MODEL=Qwen3-235B
TOKENIZER=/home/zeyu/models/Qwen3-235B-A22B
HOST=localhost
PORT=9090
OUTPUT_LEN=5
NUM_PROMPTS=400

# Per-node JSONL paths (host paths, also visible inside containers
# because /home/zeyu is bind-mounted with the same path).
PROD_JSONL=/home/zeyu/lmcache_metrics_producer.jsonl
CONS_JSONL=/home/zeyu/lmcache_metrics_consumer.jsonl

# Per-node IPv4 (eth0). node 0 is local; lj1/2/3 reached via ssh.
NODE0_IP=192.168.0.42
LJ1_IP=192.168.0.40
LJ2_IP=192.168.0.39
LJ3_IP=192.168.0.41

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

truncate_jsonls() {
    # Local (node 0) — prefiller
    : > "$PROD_JSONL"
    # Remote prefill (lj1) — producer side
    ssh $SSH_OPTS zeyu@${LJ1_IP} ": > $PROD_JSONL" 2>/dev/null || \
        echo "  [warn] truncate lj1 producer failed"
    # Remote decode (lj2, lj3) — consumer side
    ssh $SSH_OPTS zeyu@${LJ2_IP} ": > $CONS_JSONL" 2>/dev/null || \
        echo "  [warn] truncate lj2 consumer failed"
    ssh $SSH_OPTS zeyu@${LJ3_IP} ": > $CONS_JSONL" 2>/dev/null || \
        echo "  [warn] truncate lj3 consumer failed"
}

collect_jsonls() {
    local outdir="$1"
    mkdir -p "$outdir"
    cp "$PROD_JSONL" "$outdir/producer_node0.jsonl" 2>/dev/null || \
        : > "$outdir/producer_node0.jsonl"
    ssh $SSH_OPTS zeyu@${LJ1_IP} "cat $PROD_JSONL 2>/dev/null" \
        > "$outdir/producer_lj1.jsonl" 2>/dev/null || \
        : > "$outdir/producer_lj1.jsonl"
    ssh $SSH_OPTS zeyu@${LJ2_IP} "cat $CONS_JSONL 2>/dev/null" \
        > "$outdir/consumer_lj2.jsonl" 2>/dev/null || \
        : > "$outdir/consumer_lj2.jsonl"
    ssh $SSH_OPTS zeyu@${LJ3_IP} "cat $CONS_JSONL 2>/dev/null" \
        > "$outdir/consumer_lj3.jsonl" 2>/dev/null || \
        : > "$outdir/consumer_lj3.jsonl"
    # Quick line count summary so the log shows something useful.
    echo "  [collect] $(wc -l "$outdir"/*.jsonl 2>/dev/null | tail -1) total lines"
}

for C in "${CONCURRENCIES[@]}"; do
    for L in "${LENGTHS[@]}"; do
        TAG="vbench_${MODE}_conc${C}_L${L}"
        OUT_FILE="${TAG}.json"
        JSONL_DIR="${JSONL_BASE}/${MODE}/conc${C}/L${L}"

        echo "========================================================"
        echo "[$(date +%H:%M:%S)] $TAG"
        echo "========================================================"

        echo "[$(date +%H:%M:%S)] truncating JSONLs on 4 nodes..."
        truncate_jsonls

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

        echo "[$(date +%H:%M:%S)] collecting JSONLs into $JSONL_DIR ..."
        collect_jsonls "$JSONL_DIR"
        echo ""
    done
done
echo "[$(date +%H:%M:%S)] vbench-${MODE} all done"
