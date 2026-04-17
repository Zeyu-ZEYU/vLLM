# Disaggregated Prefill-Decode (PD) for Qwen3-VL — Cross-Node

This launcher runs a multimodal LLM (default **Qwen3-VL-8B-Instruct**)
in **prefill-decode disaggregated** mode across **two GPUs on two
nodes** and collects per-request / per-phase metrics.

- **Prefill node**: runs vision encoder + text prefill
  (`kv_role=kv_producer`).
- **Decode node**: runs text decode (`kv_role=kv_consumer`).
- KV cache is transferred between the two GPUs via **NCCL over TCP**
  on a user-chosen NIC (any IP-reachable interface works; the fast
  path is whichever one connects the two hosts).

The orchestrator runs on the **prefill** node, SSHes into the decode
node to launch the decode process, runs prefill locally, then copies
decode outputs back and produces a single consolidated summary.

---

## Contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [All command-line flags](#all-command-line-flags)
- [Output layout](#output-layout)
- [Metrics reference](#metrics-reference)
- [Optional: nsys profiling](#optional-nsys-profiling)
- [Optional: per-SM metrics](#optional-per-sm-metrics)
- [Troubleshooting](#troubleshooting)
- [Notes on correctness](#notes-on-correctness)

---

## Prerequisites

Everything below must hold on **both** nodes (unless otherwise noted).

### Hardware

- 1× CUDA-capable GPU per node with ≥ 24 GB free (H20 / H100 / A100 /
  L40 / A6000 all tested-class).
- A network path between the two nodes with at least one IP address
  reachable from each side (any NIC; Ethernet / IB-over-IP / etc.).

### Software

- Linux + CUDA toolkit + a working Python environment (venv / conda).
- This repository cloned at **the same absolute path on both nodes**
  (the launcher assumes identical paths for simplicity; change it via
  `--remote-repo` if you must).
- `pip install -e .` of the vLLM repo, OR an existing vLLM install
  whose source matches the one checked out here. In particular, this
  fork contains:
    - `zeyu/` launcher scripts (the entry point),
    - `vllm/v1/engine/iteration_logger.py` (per-iteration JSONL +
      pynvml sampling),
  so you must run the vLLM in *this* repo, not the pypi build.
- `pip install msgpack pynvml` — both are imported at runtime
  (`P2pNcclConnector` uses msgpack; the iteration logger uses pynvml).
- The model weights at the same absolute path on both nodes. Default
  is `Qwen/Qwen3-VL-8B-Instruct`; pass `--model /your/path` otherwise.

### SSH

- **Passwordless SSH** from the prefill node to the decode node as
  whatever user will run the job (configure with `--peer-ssh-user`).
  Test with: `ssh <decode-user>@<decode-host> echo ok`.
- On the decode side, the remote shell must source the normal login
  profile (so `conda` / `mamba` / `pip` are on `$PATH` under a
  non-interactive SSH). If your setup requires an extra activation
  step, wrap it in a login shell script and point `--conda-env`
  (or the `ssh` command in the launcher) at it.

### Optional: Docker

If both nodes run the workload inside a Docker container, set
`--container <name>`; the launcher will `docker exec -u <user>
<container> bash -lc "..."` to reach the conda env. If you run
bare-metal, pass `--container ''` (empty) — the launcher will
execute the remote command directly via SSH.

### Optional: Nsight Systems (for kernel-level metrics)

Not required for basic metrics. See the
[nsys profiling section](#optional-nsys-profiling) below for install
notes.

---

## Quick start

1. **Pick the network interface** the two nodes should use:

   ```bash
   ip -4 -o addr                   # list IPv4 interfaces
   ping <other-node-ip> -I eth0    # confirm the NIC is routable
   ```

   Use the management NIC (the one you SSH over) unless you
   specifically have a faster RDMA-over-TCP path you want to test.

2. **Pick two free GPUs** (one per node):

   ```bash
   nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
              --format=csv,noheader
   ```

3. **Verify single-node sanity** first:

   ```bash
   python zeyu/run_qwen35_vision_offline.py \
       --model /path/to/Qwen3-VL-8B-Instruct \
       --num-prompts 4 --max-tokens 64
   ```

   If this run fails, fix it before attempting cross-node.

4. **Run the cross-node disagg** from the prefill node:

   ```bash
   bash zeyu/disagg_run.sh \
       --peer-host  decode.example.org \
       --peer-ssh-user alice \
       --model      /path/to/Qwen3-VL-8B-Instruct \
       --iface      eth0 \
       --prefill-gpu 0 \
       --decode-gpu  0 \
       --num-prompts 20 \
       --max-tokens  64
   ```

5. When it finishes, the merged summary is written to:

   ```
   zeyu/outputs/disagg_<UTC-TIMESTAMP>/disagg_summary.json
   ```

   A human-readable per-request table is also printed to stdout.

---

## How it works

```
Prefill node                                Decode node
──────────────                              ──────────────
 1. Launcher binds ZMQ ctrl                  1. Launcher SSHes in and
    socket on ctrl-port.                        starts the decode
                                                process.
 2. Prefill process loads the LLM            2. Decode process loads the
    with kv_role=kv_producer on                 LLM with kv_role=
    <iface IP>:kv_port.                         kv_consumer on
 3. Prefill sends its ZMQ address               <iface IP>:(kv_port+100).
    over ctrl.                               3. Decode sends its ZMQ
 4. Prefill mints request_ids                   address over ctrl.
    that encode BOTH endpoints'              4. Decode receives request
    addresses in the format                     ids; both sides use the
    P2pNcclConnector requires                   SAME ids so KV tensor
    (`reqN___prefill_addr_IP:PORT___            keys match on producer
    decode_addr_IP:PORT_UID`).                  and consumer.
 5. Prefill calls                            5. Decode waits for
    generate(max_tokens=1).                     "prefill_done" signal.
    Encoder + prefill + first                6. Decode calls
    token are computed; KV is                   generate(max_tokens=N).
    pushed to decode via NCCL.                  The consumer sees
 6. Prefill signals "prefill_done"              that KV already arrived
    and waits for the decode to                 and skips prefill
    signal "exit" before tearing                compute — just decodes.
    down NCCL.                               7. Decode writes
 7. Launcher tar-copies decode's                latency.json, signals
    output dir back, runs                       "exit", tears down NCCL.
    analyze_profile.py (if nsys
    was enabled) and
    merge_disagg.py.
```

All GPU-to-GPU KV traffic goes over NCCL. The control channel is
plain TCP ZMQ PAIR sockets (bind on prefill, connect from decode)
and is only used for tiny sync messages (addresses, "done", "exit").

---

## All command-line flags

`bash zeyu/disagg_run.sh --help` prints the header comment. Supported
flags:

| Flag | Default | Description |
|---|---|---|
| `--peer-host HOST` | *(required)* | Hostname/IP of the decode node (whatever you SSH to) |
| `--peer-ssh-user USER` | `$USER` of current shell | SSH user on the decode side |
| `--model PATH` | `/home/$USER/models/Qwen3-VL-8B-Instruct` (example) | Model path, must exist on **both** nodes at the same location |
| `--num-prompts N` | `4` | Number of built-in example prompts to cycle through |
| `--max-tokens N` | `64` | Max tokens to generate per request (decode length) |
| `--max-model-len N` | `4096` | Context length |
| `--gpu-memory-utilization F` | `0.85` | Fraction of GPU memory for KV cache + model |
| `--iface IFACE` | `eth0` | Name of the NIC to bind NCCL/GLOO to on both nodes |
| `--kv-port N` | `25555` | KV ZMQ port on prefill side. Decode uses `N+100`. Must be open on both nodes' firewall / pod policy |
| `--ctrl-port N` | `25500` | Control-channel ZMQ port (prefill binds, decode connects) |
| `--prefill-gpu IDX` | `0` | Physical GPU index on the prefill node (as seen by `nvidia-smi`) |
| `--decode-gpu IDX` | `0` | Physical GPU index on the decode node |
| `--input PATH` | *(none)* | JSONL file of `{prompt, image_url}` records. Overrides `--num-prompts` |
| `--remote-repo PATH` | same as local | vLLM repo path on the decode node |
| `--container NAME` | `fe_rnic` | Docker container on decode node. Set to empty string to skip `docker exec` and run directly via SSH |
| `--conda-env NAME` | `mono_kernel` | Conda env to `conda activate` on decode node |
| `--output-root DIR` | `zeyu/outputs/` | Where to write the `disagg_<timestamp>/` directory |
| `--nsys` / `--no-nsys` | `--no-nsys` | Enable Nsight Systems profiling on both sides (see section below) |
| `--sm-metrics` / `--no-sm-metrics` | `--no-sm-metrics` | Sample `SM Active %` / active-SM count via `nsys --gpu-metrics-devices` (requires host-level GPU counter privilege) |
| `--nsys-freq N` | `10000` | GPU-metrics sampling rate in Hz when `--sm-metrics` is on |

Change the defaults in `zeyu/disagg_run.sh` if your cluster has a
different convention (see the `---- Defaults ----` block near the top).

---

## Output layout

Every run creates a new timestamped directory:

```
zeyu/outputs/disagg_<YYYYMMDD_HHMMSS_UTC>/
├── prefill/
│   ├── iterations.jsonl      # one line per scheduler iteration (prefill side)
│   ├── requests.jsonl        # request_id → iteration indices mapping
│   ├── latency.json          # per-request: VE time, prefill time, wall ts
│   └── nsys_*.csv / *.nsys-rep      (only if --nsys)
├── decode/
│   ├── iterations.jsonl      # copied back from the decode node
│   ├── requests.jsonl
│   ├── latency.json          # per-request: decode time, TBT, KV transfer
│   └── nsys_*.csv / *.nsys-rep      (only if --nsys)
├── disagg_summary.json       # <-- the main output: per-request + aggregates
├── prefill.log               # stdout/stderr from prefill
├── decode.log                # stdout/stderr from decode (tail'd back)
├── analyze_prefill.log       # (only if --nsys) output of analyze_profile.py
└── analyze_decode.log        # (only if --nsys) output of analyze_profile.py
```

A human-readable table is also printed to stdout at the end of the run.

---

## Metrics reference

Two independent collection paths feed `disagg_summary.json`:

1. **In-process timing + pynvml sampling** — *always on*.
   A background thread inside the vLLM engine process samples
   `nvmlDeviceGetUtilizationRates` at 100 Hz. Each scheduler
   iteration gets one `iterations.jsonl` line with the mean/max of
   samples that fell in its time window, plus PyTorch
   `memory_allocated` / `max_memory_allocated` / delta.

2. **Nsight Systems profiling** — *opt-in via `--nsys`*.
   Wraps the process in `nsys profile --trace=cuda,nvtx`, post-
   processes the `.nsys-rep` with `nsys stats` to emit
   `cuda_gpu_trace.csv` (every kernel launch + duration) and
   `nvtx_pushpop_trace.csv` (every `gpu_model_runner: {preprocess,
   forward, sample}` range). `analyze_profile.py` correlates
   kernels to NVTX ranges and computes kernel-busy % per iteration
   and per sub-range (vision_encoder vs text_forward).

### Per request (`disagg_summary.json["requests"][i]`)

| Field | Description | Source |
|---|---|---|
| `vision_encoder_time_ms` | Time inside `embed_multimodal()` (0 if image hit the processor cache) | Prefill side |
| `prefill_time_ms` | `first_token_ts − scheduled_ts` of the prefill iteration | Prefill side |
| `kv_transfer_time_ms` | Approximate wall-clock delay between prefill completion and decode receiving `prefill_done` | Control channel |
| `decode_time_ms` | `last_token_ts − first_token_ts` of the decode iterations | Decode side |
| `num_generation_tokens` | Number of decode iterations (= output tokens) | Decode side |
| `tpot_ms` | Average inter-token latency = `decode_time_ms / (N − 1)` | Decode side |
| `per_token_tbt_ms[]` | Per-step TBT, one entry per decode iteration | `decode/iterations.jsonl` |
| `tbt_stats` | `{count, mean, min, p50, p95, p99, max}` of `per_token_tbt_ms` | Computed |
| `jct_ms` | End-to-end: `vision_encoder + prefill + kv_transfer + decode` | Computed |
| `vision_encoder` / `prefill` / `decode` | Sub-objects, see next table | — |

### Per phase (sub-object of each request)

Each sub-object aggregates over the iterations this request spent in
that phase.

*Always present* (pynvml path):

| Field | Description |
|---|---|
| `num_iterations` | # iterations in this phase for this request |
| `avg_gpu_mem_allocated_MiB` | Mean PyTorch-allocated GPU memory across phase iterations |
| `avg_nvml_gpu_util_pct_mean` | Mean GPU util % from pynvml within phase |
| `avg_nvml_gpu_util_pct_max` | Max per-iter peak GPU util |
| `avg_nvml_mem_util_pct_mean` | Mean memory-bus utilization % |
| `avg_nvml_mem_used_MiB_mean` | Mean total GPU memory used |
| `avg_nvml_mem_used_MiB_max` | Max total GPU memory used |

*Present when `--nsys` is on:*

| Field | Description |
|---|---|
| `avg_gpu_util_pct` | Mean kernel-busy % (nsys kernel timeline) |
| `avg_kernel_launch_gap_pct` | `100 − avg_gpu_util_pct`; time the GPU was idle during the phase |
| `sum_total_kernel_time_ns` | Absolute GPU busy time summed over phase iters |
| `sum_kernel_launch_gap_ns` | Absolute idle time summed over phase iters |
| `avg_vision_encoder_gpu_util_pct` | Kernel-busy % during the `vision_encoder` NVTX sub-range only |
| `avg_text_forward_gpu_util_pct` | Kernel-busy % during the `gpu_model_runner: forward` sub-range only |
| `sum_vision_encoder_kernel_time_ns` / `sum_text_forward_kernel_time_ns` | Absolute times for the two sub-ranges |

*Present when `--nsys --sm-metrics` is on AND GPU counters are
unrestricted* (see notes below):

| Field | Description |
|---|---|
| `avg_sm_active_pct_mean` | Mean "SM Active %" (nsys GPU metrics sampler) |
| `avg_sm_occupancy_pct_mean` | Mean "SM Warp Occupancy %" |
| `avg_num_active_sms_mean` | Mean count of active SMs |

True *per-SM* utilization (each SM's own number) requires Nsight
Compute (`ncu`), which replays kernels and runs 10–100× slower. It's
not part of this launcher; see
[Optional: per-SM metrics](#optional-per-sm-metrics).

### Summary (`disagg_summary.json["summary"]`)

| Field | Description |
|---|---|
| `avg_vision_encoder_time_ms` | Mean over requests that had a VE call (cache hits excluded) |
| `avg_prefill_time_ms` | Mean prefill time |
| `avg_kv_transfer_time_ms` | Mean KV transfer time |
| `avg_decode_time_ms` | Mean decode time |
| `avg_tpot_ms` | Mean TPOT across requests |
| `tbt_stats` | Aggregate TBT stats across **all** tokens of **all** requests |
| `avg_jct_ms` | Mean JCT |
| `total_decode_tokens` | Sum of generated tokens |
| `prefill_wall_time_s` | Prefill side's `llm.generate()` wall clock |
| `decode_wall_time_s` | Decode side's `llm.generate()` wall clock |
| `rps_end_to_end` | `num_requests / (prefill_wall + decode_wall)` |
| `rps_decode_only` | `num_requests / decode_wall` |
| `vision_encoder`, `prefill`, `decode` | Per-phase aggregates (means across requests) |

### Per iteration (`<role>/iterations.jsonl` and `consolidated_iterations.jsonl`)

Every scheduler step produces one record. Fields (pynvml path):

`iter, ts_mono, ts_wall, step_latency_ms, num_reqs, step_rps,
has_encoder, encoder_req_ids[], prefill_req_ids[], decode_req_ids[],
num_prefill_reqs, num_prefill_tokens, num_decode_reqs,
num_decode_tokens, total_tokens, gpu_mem_allocated_MiB,
gpu_mem_peak_MiB, gpu_mem_delta_MiB, nvml_gpu_util_pct_mean/max,
nvml_mem_util_pct_mean, nvml_mem_used_MiB_mean/max`.

Nsys path (added by `analyze_profile.py` into `consolidated_iterations.jsonl`):

`gpu_util_pct, total_kernel_time_ns, num_kernels, kernel_launch_gap_ns,
kernel_launch_gap_pct, vision_encoder_gpu_util_pct,
vision_encoder_kernel_time_ns, text_forward_gpu_util_pct,
text_forward_kernel_time_ns` (+ `sm_*` when `--sm-metrics`).

---

## Optional: nsys profiling

Pass `--nsys` to enable. The launcher wraps both processes in
`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node
--trace-fork-before-exec=true`. Outputs are in each role's dir:

```
<out>/{prefill,decode}/
├── nsys_report.nsys-rep
├── nsys_report.sqlite
├── nsys_kernels_cuda_gpu_trace.csv           # every kernel launch
└── nsys_nvtx_pushpop_nvtx_pushpop_trace.csv  # NVTX ranges (iter boundaries)
```

### Installing a working nsys

Some CUDA / PyTorch base images ship with a **nsys binary bundled with
Nsight Compute** that *cannot finalize .qdstrm files* (you get
`Unable to retrieve the importer version` at the end of the run).
The launcher auto-detects a usable nsys; if it can't find one, it
quietly disables nsys and falls back to the pynvml-only path.

To install a known-good standalone Nsight Systems:

```bash
# Debian/Ubuntu:
distro=$(. /etc/os-release; echo ${ID}${VERSION_ID//./})  # e.g. ubuntu2204
sudo dpkg -i https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y nsight-systems-2025.6.3  # any >= 2024.x works

which nsys     # should now be /usr/local/bin/nsys
nsys --version
```

Repeat on **both** nodes. The launcher probes these candidate paths
and picks the first one that responds to `nsys --version`:

```
nsys
/usr/local/bin/nsys
/usr/local/cuda/bin/nsys
/opt/nvidia/nsight-systems/bin/nsys
```

### Performance cost

Nsys adds ~5–15 % overhead on GPU-heavy phases; NVTX itself is
near-free. The `nsys_report.nsys-rep` is in the tens of MBs per
minute of trace, so long runs may want a smaller `--num-prompts`.

---

## Optional: per-SM metrics

`nsys --gpu-metrics-devices=all` samples GPU performance counters
including `SM Active %`, `SM Warp Occupancy %`, and count of active
SMs. To turn this on, pass `--sm-metrics` in addition to `--nsys`.

**Privilege requirement.** This reads restricted GPU counters and
will fail with `ERR_NVGPUCTRPERM` on most clusters because NVIDIA
gates them behind:

```bash
# On the host (not container), as root:
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
     | sudo tee /etc/modprobe.d/nvidia-perfcounters.conf
# Reboot or reload the nvidia kernel module
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
```

See
https://developer.nvidia.com/ERR_NVGPUCTRPERM.

If your cluster admin won't enable it, the **pynvml path** (on by
default) still gives a 100 Hz sampled GPU utilization % that is good
enough for most phase-level analysis — just not per-SM.

For *true* per-SM breakdown (each SM's utilization, separately) you
need Nsight Compute (`ncu`), which replays kernels and runs
10–100× slower. That's out of scope for this launcher; run `ncu`
once on a single-node, single-GPU reference to get per-kernel SM
numbers that you can then pair with the per-iteration kernel list
from the nsys `cuda_gpu_trace.csv`.

---

## Troubleshooting

**`ValueError: Hybrid KV cache manager is disabled but failed to convert ...`**
The model has a hybrid attention stack (e.g. Qwen3.5). KV transfer
connectors only support uniform full-attention models. Use a dense
full-attention model; Qwen3-VL-8B-Instruct, Qwen2.5-VL, and
LLaVA-class models work.

**`Free memory on device cuda:0 ... less than desired`**
Another process is using the GPU, or `--gpu-memory-utilization` is
too high. Check `nvidia-smi`, pick a different `--prefill-gpu` /
`--decode-gpu`, or lower `--gpu-memory-utilization 0.7`.

**`ModuleNotFoundError: No module named 'msgpack'` (or `pynvml`)**
```bash
pip install msgpack pynvml
```
In whichever env you use on *both* nodes.

**`Address already in use`**
A previous run left a socket in `TIME_WAIT`. Pick a different
`--ctrl-port` / `--kv-port` (any high free port) or wait ~60 s.

**Decode hangs at `Waiting for prefill to complete ...`**
The ZMQ ctrl channel didn't connect. Check:
- Is `--peer-host` reachable from the prefill side? (`ping`, `nc -zv`)
- Is `--ctrl-port` open on both sides' firewall / k8s NetworkPolicy?
- Did the prefill process actually bind? Look at `prefill.log` for
  `[Prefill] Ctrl bound on 0.0.0.0:<port>`.

**NCCL hangs at engine init / `ncclCommInitRank`**
The NCCL transport can't connect. Verify:
```bash
# Both nodes, inside the conda env:
echo $NCCL_SOCKET_IFNAME  # should be what you passed as --iface
ping <peer-iface-ip>      # must succeed over that NIC
```
The launcher exports `NCCL_SOCKET_IFNAME=$IFACE` and
`NCCL_IB_DISABLE=1` by default. If you want to use InfiniBand /
RDMA, pass `--iface <ib-name>` AND unset the IB-disable in
`disagg_run.sh` manually.

**`Request id ... does not contain hostname and port`**
P2pNcclConnector's regex couldn't parse the request ID. Make sure
both nodes are on the **same** commit (`git log -1` should match).
The launcher sets `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`
automatically; if something else overrides it, the IDs will get
mangled.

**Garbage / mismatched output from decode**
The KV producer-consumer handshake is timing out silently and the
consumer falls back to its own prefill. This shouldn't normally
happen; see
[Notes on correctness](#notes-on-correctness). Usually it's a
symptom of the two nodes running different model weights or a
network flake during NCCL setup.

**`Unable to retrieve the importer version` at end of nsys run**
The bundled Nsight-Compute nsys cannot finalize `.qdstrm` files.
Install standalone Nsight Systems — see
[Installing a working nsys](#installing-a-working-nsys).

**`Illegal --gpu-metrics-devices usage: Insufficient privilege`**
GPU performance counters are not accessible to non-admin users.
See [Optional: per-SM metrics](#optional-per-sm-metrics) for how the
cluster admin enables this; otherwise drop `--sm-metrics` and rely
on the pynvml path.

**`tar: ...: Cannot open: No such file or directory` when copying decode back**
The remote decode process never wrote its output dir. Inspect
`decode.log` for the real error — typical causes are module import
failures, model-path mismatch between the two nodes, or NCCL timeout.

---

## Notes on correctness

1. The model **must** be full-attention throughout (no Mamba, no
   linear-attention, no mixer layers). Hybrid models either crash at
   engine init (best case) or silently produce garbage because the
   recurrent state isn't transferred.
2. For multimodal prompts, the decode side still passes the image
   tokens through its own input processor — the `kv_consumer` short-
   circuits the *attention* compute but not the tokenization /
   placeholder substitution. Both nodes must therefore have the same
   model and the same image-preprocessor config.
3. `kv_transfer_time_ms` is a wall-clock delta measured across the
   two machines. It **is not** NTP-corrected. If your nodes' clocks
   drift by milliseconds, you'll see small negative numbers or jitter
   on fast runs; that's normal. Prefer `prefill_wall_time_s +
   decode_wall_time_s` and `rps_end_to_end` for throughput analysis.
4. To sanity-check decode output, run a **single-GPU reference** with
   the same model and compare a few prompts:

   ```bash
   python zeyu/run_qwen35_vision_offline.py \
       --model /path/to/Qwen3-VL-8B-Instruct \
       --num-prompts 4 --max-tokens 64
   ```

   Tokens should match the disagg run on the same prompts
   (deterministic because seed = 42 by default).
