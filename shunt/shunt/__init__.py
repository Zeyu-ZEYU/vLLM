"""Shunt: balancing compute and KV traffic without all-to-all contention in
disaggregated MoE serving.

Subpackages:

- ``shunt.config``, ``shunt.compute_model``: model, deployment, and the two
  profiled compute-time functions.
- ``shunt.algorithms``, ``shunt.native``, ``shunt.planner``: LPT placement,
  attention-head balancing, KV offload allocation, and the per-iteration plan.
- ``shunt.runtime``: the engine-side runtime (per-iteration planning, elastic
  attention, KV routing, timing logs) used by the patched vLLM and LMCache.
- ``shunt.serving``: the proxy (request placement and PD handoff).
- ``shunt.harness``: launch scripts, the trace driver, and samplers.
- ``shunt.bench``, ``shunt.profiling``: microbenchmarks and compute profiling.
- ``shunt.analysis``: turns collected logs into the figures and tables.
"""
__version__ = "1.0.0"
