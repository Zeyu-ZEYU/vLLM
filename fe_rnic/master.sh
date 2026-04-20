#!/bin/bash
# ============================================================================
# Mooncake Master Server(s)
#
# Modes:
#   bash master.sh            # single tail master (default)
#   bash master.sh --split    # tail + head masters (for ENABLE_HEAD_NIC_SPLIT)
#
#   - tail master: rpc=50052 http=8080 metrics=9009
#   - head master: rpc=50053 http=8081 metrics=9010
#
# Why two masters when head-NIC split is on:
# A single master tracks one segment pool. Tail and head clients each
# register a segment on whichever master they point to. If both share
# one master, master's PutStart can hand out a head-bound segment to
# a tail client — the tail engine (mlx5_bond_*) has no RDMA QP to a
# head-bound segment (mlx5_0) and the WRITE fails status=6 FAILED. We
# saw this reproduce on this cluster even with mooncake_prefer_local_alloc
# set (the preference is soft and master falls back to any segment on
# contention). Two masters physically isolate the segment pools so
# cross-route assignment is impossible by construction.
# ============================================================================

set -euo pipefail

MODE="${1:-tail-only}"

start_tail_master() {
    echo "[master] starting TAIL master: rpc=50052 http=8080 metrics=9009"
    mooncake_master -port 50052 -max_threads 64 -metrics_port 9009 \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --http_metadata_server_port=8080 "$@"
}

start_head_master() {
    echo "[master] starting HEAD master: rpc=50053 http=8081 metrics=9010"
    mooncake_master -port 50053 -max_threads 64 -metrics_port 9010 \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --http_metadata_server_port=8081 "$@"
}

case "$MODE" in
  --split)
      # Head master in background, tail master in foreground.
      # Exiting the foreground master tears down the background head master.
      start_head_master >/tmp/master_head.log 2>&1 &
      HEAD_PID=$!
      trap 'kill -TERM "$HEAD_PID" 2>/dev/null || true' EXIT
      sleep 0.5
      start_tail_master
      ;;
  tail-only|*)
      start_tail_master
      ;;
esac
