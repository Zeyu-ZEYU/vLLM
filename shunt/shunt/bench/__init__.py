"""Single-node microbenchmarks and the decision-cost benchmark.

- :mod:`.pick_iterations`: pick iterations of a measured run by straggler
  severity, as inputs for the two attention benchmarks.
- :mod:`.elastic_overhead`: exposed transfer time of elastic attention.
- :mod:`.ring_attention`: striped ring attention on the same iterations.
- :mod:`.scalability`: per-iteration decision cost of LPT, EAP, and KVLB.
"""
