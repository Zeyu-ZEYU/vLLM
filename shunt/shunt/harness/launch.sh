#!/usr/bin/env bash
# Launch one role of the Shunt deployment (§4.1).
#
#   ./launch.sh prefill   # one per prefill node (DP=16/EP=16/TP=1 across 2 nodes)
#   ./launch.sh decode    # one per decode node  (DP=8/EP=8/TP=1)
#   ./launch.sh proxy     # the RS proxy
#
# The MoE A2A and KV ride different DSCP classes so the A2A always wins the wire:
# NCCL marks the A2A high (NCCL_IB_TC), Mooncake marks KV low (MC_IB_TC). NICs and
# switches must trust DSCP (mlnx_qos --trust=dscp; ETS strict). DSCP = TC >> 2,
# so an IB traffic class of 96 is DSCP 24 (A2A, high) and 32 is DSCP 8 (KV, low).
set -euo pipefail

ROLE="${1:?usage: launch.sh prefill|decode|proxy}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- cluster knobs (override via env) --------------------------------------
MODEL="${MODEL:-$HOME/models/Qwen3-235B-A22B}"
MASTER_ADDR="${MASTER_ADDR:-PREFILL_NODE0}"
DP_RPC_PORT="${DP_RPC_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"              # 0 or 1 within the role's node pool
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
NCCL_IB_TC="${NCCL_IB_TC:-96}"          # A2A high class  (DSCP 24)
export MC_IB_TC="${MC_IB_TC:-32}"        # KV  low  class  (DSCP 8)
export NCCL_IB_TC
export VLLM_USE_V1=1
export SHUNT_GPU_DIRECT=1                 # KV rides dma-buf GPU-direct, no host staging

lmcache_producer='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false}}'
lmcache_consumer='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false}}'

case "$ROLE" in
  prefill)
    export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-$HERE/configs/lmcache-prefill.yaml}"
    # One DP=16/EP=16 job spans both prefill nodes (8 local GPUs each). The proxy
    # pins each request to a DP rank via X-data-parallel-rank (see dp_pinning).
    exec vllm serve "$MODEL" \
      --data-parallel-size 16 \
      --data-parallel-size-local 8 \
      --data-parallel-address "$MASTER_ADDR" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --data-parallel-start-rank $(( NODE_RANK * 8 )) \
      --enable-expert-parallel \
      --tensor-parallel-size 1 \
      --no-enable-prefix-caching \
      --kv-transfer-config "$lmcache_producer" \
      --port "$PREFILL_PORT"
    ;;
  decode)
    export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-$HERE/configs/lmcache-decode.yaml}"
    # Each decode node is an independent DP=8/EP=8 deployment (intra-node A2A).
    exec vllm serve "$MODEL" \
      --data-parallel-size 8 \
      --enable-expert-parallel \
      --tensor-parallel-size 1 \
      --kv-transfer-config "$lmcache_consumer" \
      --port "$DECODE_PORT"
    ;;
  proxy)
    exec python -m shunt.serving.proxy \
      --config "${PROXY_CONFIG:-$HERE/configs/proxy.example.json}" \
      --host 0.0.0.0 --port "${PROXY_PORT:-8000}"
    ;;
  *)
    echo "unknown role: $ROLE" >&2; exit 1 ;;
esac
