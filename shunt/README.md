# Shunt — artifact

Source code for *Shunt: Balancing Compute and KV Traffic without All-to-All
Contention in Disaggregated MoE Serving*. Shunt lowers prefill time-to-first-token
for MoE serving by planning each iteration ahead of time with three components:

- **RS** — compute-aware request scheduling at the proxy (§3.2);
- **EAP** — straggler-aware elastic attention parallelism inside the prefill
  engines (§3.3);
- **KVLB** — contention- and urgency-aware KV traffic load balancing in LMCache
  over Mooncake (§3.4).

This README explains how to bring the whole system up on the evaluation testbed
and run it end to end.

## Repositories

Shunt spans three repositories, each on a `shunt_artifact` branch:

| Repo | Branch (from) | Shunt's part |
|------|---------------|--------------|
| vLLM | `shunt_artifact` (`main`) | the `shunt/` package: planner cores, RS proxy, EAP, node planner, launcher/configs |
| LMCache | `shunt_artifact` (`dev`) | `routing_backend.py` + `routing_policy.py`: the KVLB device-bound KV router |
| Mooncake | `shunt_artifact` (`main`) | no source change — DSCP is `MC_IB_TC`, dma-buf is a build flag |

Everything Shunt-specific in vLLM lives under `shunt/`; the rest of each repo is
upstream.

## Testbed assumed by this guide

Four nodes, eight NVIDIA H20-3e GPUs each (32 GPUs), two RoCEv2 fabrics per node
(per-GPU backend RNIC ports + one host-attached frontend RNIC). The deployment:

- **2 prefill nodes** form one DP=16, EP=16, TP=1 job (16 GPUs);
- **2 decode nodes**, each an independent DP=8, EP=8, TP=1 job;
- the model is Qwen3-235B-A22B.

## Setup (each node)

```bash
# check out the three branches (worktrees or clones)
git -C vLLM     checkout shunt_artifact
git -C LMCache  checkout shunt_artifact
git -C Mooncake checkout shunt_artifact

# build the planner core (libshunt.so) used at runtime by the RS proxy and the
# node planner; needs a C++17 compiler
make -C vLLM/shunt/csrc

# install the stack: vLLM + LMCache from source on shunt_artifact; Mooncake via
# pip built with dma-buf GPU-direct
#   (cmake ... -DWITH_NVIDIA_PEERMEM=OFF -DUSE_CUDA=ON)
# and the proxy/driver deps:
pip install -e vLLM/shunt[serving]    # numpy, matplotlib, aiohttp
```

## Bringing the system up

Always set the DSCP split so the MoE A2A wins the wire over KV: NCCL marks the
A2A high (`NCCL_IB_TC`), Mooncake marks KV low (`MC_IB_TC`); DSCP = traffic
class >> 2. NICs and switches must trust DSCP (`mlnx_qos --trust=dscp`, strict
ETS); PFC keeps both classes lossless. `launch.sh` sets these for you.

### 1. Prefill (2 nodes, one DP=16/EP=16 job)

On each prefill node (`NODE_RANK` 0 and 1):

```bash
MODEL=$HOME/models/Qwen3-235B-A22B MASTER_ADDR=<prefill-node-0> NODE_RANK=0 \
  ./shunt/harness/launch.sh prefill
```

This serves with `--data-parallel-size 16 --enable-expert-parallel
--tensor-parallel-size 1`, the LMCache KV producer, and the prefill LMCache
config (`configs/lmcache-prefill.yaml`) that enables KVLB.

### 2. Decode (2 nodes, independent DP=8/EP=8 each)

```bash
./shunt/harness/launch.sh decode
```

### 3. Proxy (runs RS, publishes the EAP + KVLB plans)

Copy `configs/proxy.example.json`, fill in the prefill/decode node URLs, and pick
the mode (below), then:

```bash
PROXY_CONFIG=my-proxy.json ./shunt/harness/launch.sh proxy
```

### 4. Drive a run

```bash
python -m shunt.harness.trace_replay \
  --trace qwen_traceB_blksz_16.jsonl.xz --url http://<proxy>:8000 \
  --concurrency 512 --out run.jsonl
```

The driver replays the production trace (reconstructing reuse-preserving prompts
so prefix caching hits), passes each request's prefix/fresh-token counts to the
proxy, and records TTFT.

Between runs: `clean.sh` (in container) then `clean_host.sh`, and wait ~60 s for
TIME_WAIT to drain.

## How each component runs

- **RS (proxy).** The proxy coalesces admitted requests into per-tick batches and
  assigns each to a prefill DP worker with the longest-processing-time rule (the
  C++ core in `libshunt.so`), pricing requests by the compute they add. It pins
  the prefill onto that worker (per-rank URL or the `X-data-parallel-rank` hint).

- **EAP (prefill engines).** Each iteration the proxy's node planner writes
  `eap_heads.json` — the post-split query-head count per worker. Inside the
  prefill forward, a straggler worker offloads its extra query heads to underloaded
  on-node helper GPUs over NVLink, which compute those heads and reduce the result
  back exactly; it triggers only above the straggler threshold θ (default 1.5).
  This runs within each prefill node as part of live serving (`serving/elastic.py`,
  hooked into `Qwen3MoeAttention.forward`).

- **KVLB (LMCache + Mooncake).** The node planner also writes per-worker
  `kvlb_plan_w*.json` — the per-NIC byte budget for inbound and outbound KV. The
  LMCache routing backend (`RoutingBackend`) binds one Mooncake backend per RDMA
  NIC (each worker's own backend port first, the node's other ports next, the
  frontend last) and routes each KV chunk per that plan, keeping KV inside the
  MoE-free compute window and spilling overflow to spare ports and the idle
  frontend. The A2A keeps strict priority via the DSCP classes above.

## Configurations

The proxy config selects the scheduling mode and which node-local components run,
so the same deployment serves the full system and its variants:

| Config | `mode` | `enable_eap` | `enable_kvlb` |
|--------|--------|-------------|--------------|
| Baseline | `baseline` | false | false |
| ORS | `ors` | false | false |
| **Shunt** | `shunt` | true | true |
| Sh-ORS | `ors` | true | true |
| /RS | `baseline` | true | true |
| /EAP | `shunt` | false | true |
| /KVLB | `shunt` | true | false |

`ors` mode reads a precomputed assignment from `ors_rank_file`.

`docs/integration_map.md` records exactly where each component hooks into vLLM,
LMCache, and Mooncake.
