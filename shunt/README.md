# shunt

Shunt's planner, engine runtime, proxy, experiment harness, microbenchmarks,
and analysis scripts. The README at the root of the repository explains how
to install everything, run the experiments, and produce the figures and
tables.

- `shunt/`: the Python package (`pip install -e shunt`).
  - `config.py`, `compute_model.py`, `algorithms.py`, `planner.py`, `native.py`,
    `trace.py`: the planner and its inputs.
  - `runtime/`: the engine-side runtime (per-iteration planning, elastic
    attention, KV routing, phase timing).
  - `serving/`: the proxy and its placement policies.
  - `harness/`: the experiment harness, the trace driver, and the RNIC sampler.
  - `profiling/`, `tools/`, `bench/`, `analysis/`: compute profiling, the ORS
    oracle, microbenchmarks, and figures and tables.
- `csrc/`: the C++ planner cores and the decision-cost benchmark
  (`make -C shunt/csrc`).
- `configs/`: cluster file examples.
- `experiments/`: experiment files, one per group of figures and tables.
- `tests/`: unit tests (`python -m pytest shunt/tests`).
