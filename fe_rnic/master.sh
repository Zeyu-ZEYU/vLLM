#!/bin/bash
# ============================================================================
# Mooncake Master Server
# 启动 mooncake metadata + transfer master，所有 prefill/decode 节点都依赖它
# 建议部署在 prefill-node-0 上
#
# 绑定 0.0.0.0 (IPv4 front-end 管理网卡)，RDMA 数据面不经过 master
# ============================================================================

set -euo pipefail

mooncake_master -port 50052 -max_threads 64 -metrics_port 9009 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
