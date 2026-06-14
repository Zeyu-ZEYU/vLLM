# Shunt integration map

Where each Shunt component hooks into the upstream stack. File paths are relative
to each repo's subdirectory below. Captured from a fresh code survey of
vLLM @ `1c607d7b2`, LMCache @ `140990dc`, Mooncake @ `b0bda8c`.

## vLLM (`vllm/`)

### RS — compute-aware request scheduling (proxy)
- Dispatch lives in a **proxy**, not vLLM core. Upstream ships only an example
  (`examples/online_serving/disaggregated_serving/disagg_proxy_demo.py`,
  `Proxy.schedule()` round-robin). Shunt ships its own proxy.
- There is **no built-in "route to DP rank N" header**. To pin a request to a
  chosen prefill DP worker, run one prefill engine per DP rank and have the proxy
  address that engine directly (the deployment the harness launches).
- DP/EP/TP ranks: `vllm/distributed/parallel_state.py` — `get_dp_group()`,
  `get_ep_group()`, `get_tp_group()`, `.rank_in_group`; env `VLLM_DP_RANK`.

### EAP — elastic attention parallelism (model forward)
- Attention block: `vllm/model_executor/models/qwen3_moe.py`
  `Qwen3MoeAttention.forward()` (~L343–361): `qkv_proj` → `attn` → `o_proj`.
- Decoder layer: `Qwen3MoeDecoderLayer.forward()` (~L416 self_attn, L435 mlp).
- MoE A2A (leave untouched): `FusedMoE` → `MoERunner._forward_impl()`
  (`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`) dispatch/combine
  via `get_ep_group()`. EAP must not perturb this.
- Attention weights are **replicated across DP workers** (qkv/o are TP-parallel,
  replicated over DP) → any on-node GPU can compute any head. Helpers for the
  head split are the node's other DP workers, reached over NVLink. The split
  reuses the intra-node TP attention path but with a per-iteration degree.

### KV connector (to LMCache)
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py` `KVConnectorBase_V1`:
  `start_load_kv()`, `wait_for_layer_load()`, `save_kv_layer()`, `wait_for_save()`.
- Runner glue: `vllm/v1/worker/kv_connector_model_runner_mixin.py`.
- LMCache bridge: `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`.
- Per-request routing hints flow through the LMCache adapter's `request_configs`.

### NCCL / DSCP
- No code change. Set `NCCL_IB_TC` (A2A, high class) at worker launch; the A2A
  rides whatever NCCL is told. KV rides `MC_IB_TC` (low class), set on the
  Mooncake side. Both live in the launcher env.

## LMCache (`lmcache/`)

### KVLB — routing backend (NEW; does not exist upstream)
- Build `lmcache/v1/storage_backend/routing_backend.py` implementing
  `StorageBackendInterface` (`lmcache/v1/storage_backend/abstract_backend.py:26`).
- It holds **N device-bound Mooncake backends** (one `RemoteBackend` per RDMA
  device: 4 backend bonds + 1 frontend) and dispatches each KV chunk to a chosen
  backend per the Alg-3 plan.
- One device per Mooncake connector via `device_name` in `MooncakeStoreConfig`
  (`lmcache/v1/storage_backend/connector/mooncakestore_connector.py:152,323`;
  zero-copy paths `_batched_put_zero_copy` L716, `_batch_get_into` L522).
- `StorageManager` already supports a `location` selector and a layerwise path:
  `batched_put`/`batched_get`/`layerwise_batched_get`
  (`lmcache/v1/storage_backend/storage_manager.py:384,480,515`). Backends are
  created in `storage_backend/__init__.py:113 CreateStorageBackends`.
- Layerwise store/retrieve: `lmcache/v1/cache_engine.py` `store_layer` (L568),
  `retrieve_layer` (L902); GPU staging `lmcache/v1/gpu_connector/gpu_connectors.py`.
- Config via `LMCacheEngineConfig` (`lmcache/v1/config.py`): `extra_config`,
  `remote_storage_plugins`, `chunk_size` (256).

## Mooncake (`mooncake/`)

**No source change required.**
- DSCP: `MC_IB_TC` env → `mooncake-transfer-engine/src/config.cpp:352` →
  applied to QP GRH `traffic_class` at `rdma_endpoint.cpp:625`.
- dma-buf GPU-direct: build with `-DWITH_NVIDIA_PEERMEM=OFF -DUSE_CUDA=ON`
  (`mooncake-common/common.cmake:113`) → `ibv_reg_dmabuf_mr` path in
  `rdma_context.cpp:228`.
- One instance per device: single-element device filter (constructor filter or
  `setWhitelistFilters`); run several instances, one per NIC.
