# SPDX-License-Identifier: Apache-2.0
"""Hooks that connect vLLM's prefill workers to the Shunt runtime.

Shunt is active in a process when ``SHUNT_ROLE=prefill`` and ``SHUNT_CONFIG``
is set; otherwise every hook is a no-op and :data:`ENABLED` is ``False``.
The runtime itself lives in the ``shunt`` package (``shunt.runtime``). The call
sites in vLLM are:

- ``v1/worker/gpu_worker.py``: :func:`on_distributed_ready` after the model
  parallel groups exist.
- ``v1/worker/gpu_model_runner.py``: :func:`begin_step` before the per-step
  DP synchronization.
- ``v1/worker/dp_utils.py``: :func:`dp_extra_column` / :func:`on_dp_rows`
  around that synchronization.
- ``model_executor/models/qwen3_moe.py``: :func:`attention_forward`.
- ``model_executor/layers/fused_moe/...`` and the KV-transfer decorator:
  :func:`phase` and :func:`timed_kv_wait` for the timing logs.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext

ENABLED: bool = (os.environ.get("SHUNT_ROLE") == "prefill"
                 and bool(os.environ.get("SHUNT_CONFIG")))

_TIMING = None
_EAP = None


def _state():
    from shunt.runtime import state

    return state.get()


def on_distributed_ready(vllm_config) -> None:
    """Bind the runtime to this DP rank and create the elastic-attention groups."""
    global _TIMING, _EAP
    if not ENABLED:
        return
    from vllm.distributed.parallel_state import get_dp_group

    pc = vllm_config.parallel_config
    rt = _state()
    rank = get_dp_group().rank_in_group if pc.data_parallel_size > 1 else 0
    rt.set_rank(rank, pc.data_parallel_size)
    if pc.tensor_parallel_size > 1:
        from vllm.distributed.parallel_state import get_tp_group

        if get_tp_group().rank_in_group != 0:
            rt.logs.dir = None   # one log per DP rank

    from shunt.runtime import kv_router, timing

    _TIMING = timing.get()
    kv_router.get()
    if rt.opts.eap:
        if pc.tensor_parallel_size != 1:
            raise ValueError("Shunt elastic attention requires TP=1")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("Shunt elastic attention requires --enforce-eager "
                             "on the prefill instance")
        from shunt.runtime import elastic

        node = rt.settings.node_size or rt.deploy.workers_per_node
        _EAP = elastic.init(vllm_config.scheduler_config.max_num_seqs, node,
                            2 if pc.enable_dbo else 1)


def begin_step(runner, scheduler_output, num_scheduled_tokens) -> None:
    """Report this rank's scheduled requests before the DP synchronization."""
    if not ENABLED:
        return
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group, has_kv_transfer_group)

    n = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids[:n]
    computed = runner.input_batch.num_computed_tokens_cpu[:n]
    kv_in: dict = {}
    producer = False
    if has_kv_transfer_group():
        conn = get_kv_transfer_group()
        fn = getattr(conn, "shunt_kv_tokens", None)
        if fn is not None:
            kv_in = fn(scheduler_output.kv_connector_metadata) or {}
        producer = getattr(conn, "is_producer", lambda: False)()
    reqs = []
    for i, rid in enumerate(req_ids):
        fresh = int(num_scheduled_tokens[i])
        reqs.append((rid, int(computed[i]), fresh, int(kv_in.get(rid, 0)),
                     fresh if producer else 0))
    _state().begin_step(reqs)


def dp_extra_column() -> list[int] | None:
    if not ENABLED:
        return None
    return _state().dp_column()


def on_dp_rows(rows: list[list[int]]) -> None:
    if ENABLED:
        _state().on_dp_rows(rows)


def attention_forward(mod, positions, hidden_states, orig):
    """Qwen3-MoE attention with timing and, when planned, elastic attention."""
    if _TIMING is not None:
        _TIMING.set_layer(_layer_index(mod))
        with _TIMING.phase("attn"):
            return _attention(mod, positions, hidden_states, orig)
    return _attention(mod, positions, hidden_states, orig)


def _attention(mod, positions, hidden_states, orig):
    if _EAP is not None:
        return _EAP.forward(mod, positions, hidden_states, orig)
    return orig(positions, hidden_states)


def _layer_index(mod) -> int:
    idx = getattr(mod, "_shunt_layer_idx", None)
    if idx is None:
        from vllm.model_executor.models.utils import extract_layer_index

        idx = extract_layer_index(mod.attn.layer_name)
        mod._shunt_layer_idx = idx
    return idx


def phase(name: str):
    """Timing context for one phase of the current layer (no-op when off)."""
    if _TIMING is None:
        return nullcontext()
    return _TIMING.phase(name)


@contextmanager
def timed_kv_wait():
    """Measures host time blocked waiting for a layer's inbound KV."""
    if _TIMING is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _TIMING.add_host("kv_wait", time.perf_counter() - t0)
