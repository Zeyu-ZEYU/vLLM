"""Engine-side runtime of Shunt, used by the patched vLLM and LMCache.

It is active in prefill workers when ``SHUNT_ROLE=prefill`` and
``SHUNT_CONFIG`` points to a :class:`shunt.config.ShuntConfig` JSON file.

- :mod:`.settings`: environment settings.
- :mod:`.state`: per-iteration planning. Each rank contributes its estimated
  compute and KV volumes to the data-parallel synchronization vLLM already runs
  every step, then derives the group plan locally.
- :mod:`.elastic`: elastic attention (head offload over the node's NVLink).
- :mod:`.kv_router`: chunk-to-port routing for LMCache from the KVLB plan.
- :mod:`.timing`: per-layer phase timing and per-step logs.
"""
