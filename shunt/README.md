# Shunt — artifact

Source code for *Shunt: Balancing Compute and KV Traffic without All-to-All
Contention in Disaggregated MoE Serving*. Shunt lowers prefill time-to-first-token
(TTFT) for MoE serving by planning each iteration ahead of time:

1. **RS** — compute-aware request scheduling at the proxy (§3.2);
2. **EAP** — straggler-aware elastic attention parallelism (§3.3);
3. **KVLB** — contention- and urgency-aware KV traffic load balancing (§3.4).

It is built on three repositories, each on a `shunt_artifact` branch:

| Repo | Branch | What Shunt adds |
|------|--------|-----------------|
| vLLM | `shunt_artifact` (from `main`) | the `shunt/` package: planner cores, RS proxy, EAP, harness, analyses, plotters |
| LMCache | `shunt_artifact` (from `dev`) | `routing_backend.py` + `routing_policy.py`: the KVLB device-bound KV router |
| Mooncake | `shunt_artifact` (from `main`) | no source change — DSCP is `MC_IB_TC`, dma-buf is a build flag |

Everything Shunt-specific in vLLM lives under `shunt/`; the rest of each repo is
upstream.

## Layout (`vllm/shunt/`)

```
shunt/                  planner package (compute model + Alg 1/2/3, validated on the trace)
  compute_model.py      §3.1 worker/expert compute-time model
  algorithms.py         Alg 1 RS (LPT), oracle, Alg 2 EAP, Alg 3 KVLB
  planner.py            per-iteration RS->EAP->KVLB
  trace.py              Qwen trace loader + prefix-reuse sim
  serving/              RS proxy + scheduler + EAP serving integration
  harness/              launcher, trace driver, node planner, bw sampler, configs, cleanup
  analysis/             figure generators (Figs 5-8, 16-20)
  bench/                GPU microbench (Fig 18)
  plots/                shared figure style
csrc/                   C++ planner cores + ctypes lib + Fig 19 bench
tests/                  unit + parity tests
docs/integration_map.md where each component hooks into the three repos
```

## Setup

```bash
# 1. check out the three branches (worktrees or clones)
git -C vllm    checkout shunt_artifact
git -C LMCache checkout shunt_artifact
git -C Mooncake checkout shunt_artifact

# 2. build the C++ planner cores (needs a C++17 compiler)
make -C vllm/shunt/csrc            # -> libshunt.so (ctypes) + scalability_bench

# 3. Python deps for the analyses/driver (numpy, matplotlib, aiohttp)
pip install numpy matplotlib aiohttp

# 4. for the live system, install the three packages on the testbed as usual
#    (vLLM + LMCache from source on shunt_artifact; Mooncake via pip, built with
#     -DWITH_NVIDIA_PEERMEM=OFF -DUSE_CUDA=ON for dma-buf GPU-direct)
```

Run the tests (no GPU needed):

```bash
cd vllm/shunt
PYTHONPATH=. python tests/test_planner.py
PYTHONPATH=. python tests/test_scheduler.py
PYTHONPATH=. python tests/test_elastic.py
PYTHONPATH=. python tests/test_native_parity.py     # after `make -C csrc`
```

## Reproducing the figures

`TRACE=path/to/qwen_traceB_blksz_16.jsonl.xz` throughout. Three tracks by where
each experiment runs:

### A. Single machine, CPU only — no cluster needed

| Figure | What | Command |
|--------|------|---------|
| 5 | trace I/O length CDF | `python -m shunt.analysis.trace_stats --trace $TRACE --out fig5.pdf` |
| 6, 7, 8 | compute / KV imbalance, round-robin and oracle | `python -m shunt.analysis.imbalance --trace $TRACE --plot-dir figs/` |
| 19 | RS/EAP/KVLB decision cost vs DP workers | `./csrc/scalability_bench > r.txt && python -m shunt.analysis.scalability --results r.txt --out fig19.pdf` |

The imbalance run prints, and Figs 6-8 reproduce, the paper's Measurement numbers
exactly: round-robin compute max/mean 2.56×/11.39×, inbound 1.49×/2.54×, outbound
1.93×/5.49×; under the oracle compute stays 11.17× worst (16.9% of iterations
>3×), outbound drops to 4.52×, and inbound *worsens* to 3.35×.

### B. One multi-GPU node (8 GPUs)

| Figure | What | Command |
|--------|------|---------|
| 18 | elastic-attention NVLink overhead | `torchrun --nproc_per_node=8 -m shunt.bench.elastic_attn_bench --out e.jsonl` then `python -m shunt.analysis.elastic_overhead --results e.jsonl --out fig18.pdf` |

### C. Full 4-node RoCE testbed — the live system

Deploy (see *Running the system* below), drive the trace, then plot.

| Figure | What | How |
|--------|------|-----|
| 9 | backend RNIC near-saturation | `shunt.harness.bw_sampler` on a prefill + a decode node during serving |
| 10 | TTFT cost of A2A↔KV contention | run baseline vs the no-contention KVLB mode (`shunt_no_contention`), `shunt.analysis.ttft box` |
| 11 | frontend RNIC idle | `shunt.harness.bw_sampler` on the frontend NIC |
| 16 | overall TTFT + load (Baseline/ORS/Shunt/Sh-ORS) | drive each config, `shunt.analysis.ttft box` and `... load` |
| 17 | leave-one-out ablation (/RS, /EAP, /KVLB) | drive each arm, `shunt.analysis.ttft` |
| 20 | θ sensitivity | drive Shunt at θ∈{1,1.5,2,2.5,3}, `shunt.analysis.theta` |

## Running the system

The proxy runs RS and publishes the per-iteration EAP+KVLB plans; vLLM serves
prefill (DP=16/EP=16/TP=1 across two nodes) and decode (DP=8/EP=8/TP=1 per node);
LMCache routes KV across the NICs through Mooncake; the MoE A2A and KV ride
different DSCP classes so the A2A always wins the wire.

```bash
# prefill node 0 and 1 (NODE_RANK 0/1), one DP=16/EP=16 job:
MODEL=$HOME/models/Qwen3-235B-A22B MASTER_ADDR=prefill0 NODE_RANK=0 \
  ./shunt/harness/launch.sh prefill
# decode nodes:
./shunt/harness/launch.sh decode
# proxy (edit configs/proxy.example.json with the node URLs + mode):
PROXY_CONFIG=my-proxy.json ./shunt/harness/launch.sh proxy
# drive the trace:
python -m shunt.harness.trace_replay --trace $TRACE --url http://proxy:8000 \
  --concurrency 512 --out shunt.jsonl          # closed loop (TTFT dist)
python -m shunt.harness.trace_replay --trace $TRACE --rate 12 --out shunt_r12.jsonl  # open loop
```

The four configurations and ablation arms are selected by the proxy config
(`mode` = `shunt`/`baseline`/`ors`, plus `enable_eap`/`enable_kvlb`):

| Config | mode | enable_eap | enable_kvlb |
|--------|------|-----------|------------|
| Baseline | baseline | false | false |
| ORS | ors | false | false |
| **Shunt** | shunt | true | true |
| Sh-ORS | ors | true | true |
| /RS | baseline | true | true |
| /EAP | shunt | false | true |
| /KVLB | shunt | true | false |

Always `clean.sh` then `clean_host.sh` and wait ~60 s for TIME_WAIT before the
next run.

### DSCP priority (no Mooncake source change)

The launcher sets `NCCL_IB_TC` (A2A, high class) and `MC_IB_TC` (KV, low class);
DSCP = traffic class >> 2. NICs and switches must trust DSCP (`mlnx_qos
--trust=dscp`, strict ETS). The two classes are lossless under PFC.

## Notes

- **The planner is verified against the paper.** The compute model, trace
  reconstruction, RS/oracle, EAP head split, and KVLB allocation reproduce the
  Measurement numbers exactly and pass C++↔Python parity.
- **Microsecond costs are machine-dependent.** Fig 19's absolute RS/EAP/KVLB
  times depend on the CPU; the paper pins a Xeon core. The shape (RS ~linear to
  8 K workers, EAP/KVLB flat and sub-µs) is the claim and is reproduced.
- **The live-serving figures need the RoCE testbed.** Figs 9-11 and 16-17/20 are
  produced by the deployed system; the analyses here consume its output JSONL.
- The elastic-attention serving path (`serving/elastic.py`) falls back to local
  attention until validated on the multi-GPU testbed; its NVLink cost is measured
  standalone by Fig 18.
