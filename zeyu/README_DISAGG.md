# Disaggregated Prefill-Decode (PD) Mode — Cross-Node

This document describes how to run **Qwen3-VL-8B-Instruct** in
prefill-decode disaggregated mode across **two nodes**:

- **Node 0 GPU**: vision encoder + text prefill (`kv_producer`)
- **Node 1 GPU**: text decode (`kv_consumer`)

KV cache is transferred between the two GPUs over NCCL, going out
over the management NIC (default `eth0`, TCP). The launcher is meant
to be run from Node 0; it SSHes into Node 1 to start the decode
process, runs the prefill locally, copies the remote metrics back,
and produces a consolidated summary.

## Why not Qwen3.5-9B?

Qwen3.5-9B has a **hybrid** attention architecture (layers alternate
between `linear_attention` and `full_attention`). vLLM's
`P2pNcclConnector` — and all standard KV transfer connectors —
only support uniform full-attention models. Running
`--kv-transfer-config` with Qwen3.5-9B crashes at engine init:

    ValueError: Hybrid KV cache manager is disabled but failed to
    convert the KV cache specs to one unified type.

Qwen3-VL-8B-Instruct is the closest equivalent (pure full-attention,
8B dense, latest Qwen3 VL series, multimodal) that works with
P2pNcclConnector.

## How It Works

```
Node 0 (prefill role)                     Node 1 (decode role)
───────────────────────                   ─────────────────────
 1. Launcher binds ZMQ ctrl                1. Launcher starts decode
    socket on ctrl-port.                      via SSH; decode connects
 2. Prefill loads LLM with                    to Node 0 ctrl socket.
    kv_role=kv_producer on                 2. Decode loads LLM with
    eth0 IP:kv_port.                          kv_role=kv_consumer on
 3. Prefill sends its ZMQ                     eth0 IP:(kv_port+100).
    address over ctrl.                     3. Decode sends its ZMQ
 4. Prefill generates                         address over ctrl.
    request_ids that encode                4. Decode receives request
    BOTH endpoints' ZMQ                       ids from prefill; both
    addresses in the special                  sides use the SAME ids
    format expected by                        so KV tensor keys match.
    P2pNcclConnector.                      5. Decode waits for prefill
 5. Prefill runs                              to signal "done".
    generate(max_tokens=1).                6. Decode runs
    Encoder + prefill +                       generate(max_tokens=N).
    first token are computed.                 Consumer side skips prefill
    KV is pushed to decode                    work because producer's KV
    via NCCL over eth0.                       has already arrived.
 6. Prefill sends "done"                   7. Decode produces tokens and
    signal over ctrl.                         writes latency.json.
 7. Prefill waits for                      8. Decode sends "exit" signal;
    decode's "exit" signal,                   tears down NCCL.
    then exits.
 8. Launcher tars decode
    outputs back, then runs
    merge_disagg.py to build
    disagg_summary.json.
```

## Prerequisites

Same on **both** nodes:

- Linux with CUDA + an H20/H100/A100 class GPU (≥24 GB free)
- The repo cloned at the same absolute path (default
  `/home/zeyu/vllm/mono_kernel`)
- Conda env (default `mono_kernel`) with vLLM installed editable
  from this repo
- `msgpack` installed (`pip install msgpack`) — required by
  `P2pNcclConnector`
- Qwen3-VL-8B-Instruct weights at the same absolute path on both
  nodes (default `/home/zeyu/models/Qwen3-VL-8B-Instruct`)
- Network: both nodes reachable to each other over the NIC named by
  `--iface` (default `eth0`). Use the management NIC IP; the launcher
  auto-detects it.
- Passwordless SSH from **Node 0** to **Node 1** (either direct or
  via the docker container, depending on your setup)
- `nvidia-smi`, `tar`, `ssh` available inside the container
  (rsync is NOT required)

## Usage

Run on Node 0 (the prefill node):

```bash
bash zeyu/disagg_run.sh \
    --peer-host lj1.zeyu.tw \
    --prefill-gpu 4 \
    --decode-gpu 4 \
    --num-prompts 20 \
    --max-tokens 64
```

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--peer-host HOST` | *(required)* | Hostname/IP of the decode node |
| `--peer-ssh-user USER` | `zeyu` | SSH user for the peer |
| `--model PATH` | `/home/zeyu/models/Qwen3-VL-8B-Instruct` | Model path on BOTH nodes |
| `--num-prompts N` | 4 | Number of built-in example prompts to cycle through |
| `--max-tokens N` | 64 | Max tokens to generate per request |
| `--max-model-len N` | 4096 | Maximum context length |
| `--gpu-memory-utilization F` | 0.85 | Fraction of GPU memory for KV cache |
| `--iface IFACE` | `eth0` | Network interface for NCCL/GLOO |
| `--kv-port N` | 25555 | Prefill KV port; decode uses N+100 |
| `--ctrl-port N` | 25500 | Control-channel ZMQ port on prefill node |
| `--prefill-gpu IDX` | 0 | Physical GPU index on Node 0 |
| `--decode-gpu IDX` | 0 | Physical GPU index on Node 1 |
| `--input PATH` | *(none)* | JSONL input file (overrides `--num-prompts`) |
| `--remote-repo PATH` | `/home/zeyu/vllm/mono_kernel` | Repo path on peer |
| `--container NAME` | `fe_rnic` | Docker container name on peer |
| `--conda-env NAME` | `mono_kernel` | Conda env name on peer |

## Output structure

```
zeyu/outputs/disagg_<YYYYMMDD_HHMMSS>/
├── prefill/
│   ├── iterations.jsonl          # per-iteration log (from IterationLogger)
│   ├── requests.jsonl            # request→iteration mapping
│   └── latency.json              # per-request: VE, prefill, timestamps
├── decode/
│   ├── iterations.jsonl          # copied back from Node 1
│   ├── requests.jsonl
│   └── latency.json              # per-request: decode, TPOT, KV transfer, TTFT
├── disagg_summary.json           # consolidated per-request + summary
├── prefill.log                   # stdout from prefill side
└── decode.log                    # stdout from decode side
```

## Metrics pipeline

Two independent collection paths feed into `disagg_summary.json`:

1. **In-process timing + pynvml sampling** (default, always on).
   The iteration logger runs inside each role's engine process,
   writes one line per scheduler step to `iterations.jsonl`. A
   background thread samples `nvmlDeviceGetUtilizationRates` at
   100 Hz, and each iteration's record carries the mean/max of
   samples that fell within its ts_mono window.

   Metrics: per-iteration GPU util %, memory util %, memory used
   MiB, memory allocated (torch), request phase classification,
   step latency, step RPS.

2. **nsys profiling** (opt-in via `--nsys`). When enabled and the
   environment has a working standalone nsys, collects:
   - Per-kernel CUDA timeline (analyze_profile.py derives per-
     iteration "kernel-busy %" and kernel-launch gap from this).
   - Per-NVTX-range breakdown (vision_encoder / text_forward).
   - (Optional) `--sm-metrics` enables `nsys profile
     --gpu-metrics-devices=all` for aggregate SM active %. This
     requires GPU performance counters to be unrestricted on
     the host (ERR_NVGPUCTRPERM). See NVIDIA documentation.

   When nsys is on, analyze_profile.py is invoked by the launcher
   to produce `consolidated_iterations.jsonl` with GPU util, kernel
   time, SM metrics. merge_disagg.py then picks these up.

**Note**: the Nsight-Compute-bundled nsys in some environments fails
to finalize `.qdstrm` files (cannot retrieve importer version).
Standalone Nsight Systems ≥ 2024.x is recommended if you want
nsys-derived kernel / SM metrics. Otherwise pynvml-based GPU-util
sampling (path 1) is the primary source and is always available.

All metrics are in `disagg_summary.json`. Per-iteration raw data is
additionally in `prefill/iterations.jsonl`,
`decode/iterations.jsonl`, and `<role>/consolidated_iterations.jsonl`
(the latter two include nsys-derived GPU/SM metrics when profiling
is on — off by default, use `--nsys` to enable).

### Per request (in `disagg_summary.json["requests"]`)

| Field | Description | Source |
|---|---|---|
| `vision_encoder_time_ms` | Time in `embed_multimodal()` (first-use images only) | Node 0 |
| `prefill_time_ms` | `first_token_ts − scheduled_ts` during prefill | Node 0 |
| `kv_transfer_time_ms` | Wall-clock delta between prefill `time.time()` at wall_end and decode `time.time()` at signal arrival (approximate, depends on NTP sync) | Launcher |
| `decode_time_ms` | `last_token_ts − first_token_ts` during decode | Node 1 |
| `num_generation_tokens` | Number of output tokens produced (= # decode iterations) | Node 1 |
| `tpot_ms` | Average time per output token = decode_time / (N − 1) | Node 1 |
| `per_token_tbt_ms` | Array of inter-token latencies (ms), one per decode step | From decode iterations.jsonl |
| `tbt_stats` | `{count, mean_ms, min_ms, p50_ms, p95_ms, p99_ms, max_ms}` of TBT for this request | Computed |
| `jct_ms` | Job completion time ≈ VE + prefill + KV transfer + decode | Computed |

### Per phase, per request (sub-objects `vision_encoder`, `prefill`, `decode`)

Each sub-object contains averages/sums across the iterations in that
phase of the request.

Always present (pynvml path):

| Field | Description |
|---|---|
| `num_iterations` | Number of iterations in this phase for this request |
| `avg_gpu_mem_allocated_MiB` | Mean PyTorch-allocated GPU memory during the phase |
| `avg_nvml_gpu_util_pct_mean` | Mean GPU utilization % from pynvml samples in this phase |
| `avg_nvml_gpu_util_pct_max` | Max per-iteration peak GPU utilization |
| `avg_nvml_mem_util_pct_mean` | Mean memory-bus utilization % |
| `avg_nvml_mem_used_MiB_mean` | Mean total GPU memory used (MiB) |
| `avg_nvml_mem_used_MiB_max` | Max total memory used (MiB) |

Present when nsys is enabled:

| Field | Description |
|---|---|
| `avg_gpu_util_pct` | Mean kernel-busy % from nsys kernel timeline |
| `avg_kernel_launch_gap_pct` | Mean kernel-launch gap % (= 100 − gpu_util) |
| `sum_total_kernel_time_ns` | Total GPU busy time across all phase iterations (absolute) |
| `sum_kernel_launch_gap_ns` | Total GPU idle time (absolute) |
| `avg_vision_encoder_gpu_util_pct` / `avg_text_forward_gpu_util_pct` | Sub-range GPU util when the iteration ran the vision encoder or text forward kernel |
| `sum_vision_encoder_kernel_time_ns` / `sum_text_forward_kernel_time_ns` | Same, absolute |

Present when nsys + `--sm-metrics` is enabled AND GPU counters are
unrestricted on the host:

| Field | Description |
|---|---|
| `avg_sm_active_pct_mean` | Mean "SM Active %" (fraction of SMs busy) sampled by nsys GPU metrics |
| `avg_sm_occupancy_pct_mean` | Mean "SM Warp Occupancy %" sampled by nsys |
| `avg_num_active_sms_mean` | Mean count of active SMs sampled by nsys |

True per-SM breakdown (each SM's own utilization %) requires ncu and
is out of scope for this launcher; see "per-SM breakdown" section.

### Per iteration (in `<role>/consolidated_iterations.jsonl`)

Already documented in `zeyu/README.md`. When nsys is on, each record
additionally carries `gpu_util_pct`, `kernel_launch_gap_pct`,
`kernel_launch_gap_ns`, `total_kernel_time_ns`,
`vision_encoder_gpu_util_pct`/`text_forward_gpu_util_pct`,
`sm_active_pct_mean`, `sm_occupancy_pct_mean`,
`num_active_sms_mean`.

### Summary (in `disagg_summary.json["summary"]`)

| Field | Description |
|---|---|
| `avg_vision_encoder_time_ms` | Mean across requests that had a VE call (cache-hit requests excluded) |
| `avg_prefill_time_ms` | Mean prefill time |
| `avg_decode_time_ms` | Mean decode time |
| `avg_tpot_ms` | Mean TPOT (average inter-token latency) |
| `tbt_stats` | Aggregate `{count, mean_ms, min_ms, p50_ms, p95_ms, p99_ms, max_ms}` across all TBT samples from all requests |
| `avg_kv_transfer_time_ms` | Mean KV coordination transfer time |
| `avg_jct_ms` | Mean JCT |
| `total_decode_tokens` | Sum of all generated tokens |
| `prefill_wall_time_s` | Node 0's `llm.generate()` wall clock |
| `decode_wall_time_s` | Node 1's `llm.generate()` wall clock |
| `rps_end_to_end` | N / (prefill_wall + decode_wall) |
| `rps_decode_only` | N / decode_wall |
| `vision_encoder` / `prefill` / `decode` | Per-phase aggregated iteration metrics (means over all requests' phase aggregates) |

## nsys profiling (on by default)

The launcher profiles **both** prefill and decode with `nsys profile
--trace=cuda,nvtx --gpu-metrics-devices=cuda-visible`. Outputs:

```
<out_dir>/prefill/
├── nsys_report.nsys-rep
├── nsys_kernels_cuda_gpu_trace.csv
├── nsys_nvtx_pushpop_nvtx_pushpop_trace.csv
├── nsys_gpu_metrics_gpu_metrics.csv          # aggregate SM metrics
└── consolidated_iterations.jsonl             # enriched by analyze_profile.py
```

Same structure under `<out_dir>/decode/`. `merge_disagg.py` reads
these at the end of the run and produces per-request per-phase
aggregates in `disagg_summary.json`.

Flags:
- `--no-nsys` disables profiling (faster, but you lose GPU util /
  SM metrics / kernel overhead numbers).
- `--nsys-freq N` sets GPU-metrics sampling rate in Hz. Default
  10000 (= 100 µs between samples). Lower if you see GPU slowdown;
  higher for finer-grained SM tracking.

## Using ncu for per-SM breakdown (optional, very slow)

`nsys --gpu-metrics-devices` gives aggregate SM activity but not a
per-SM utilization breakdown. For true per-SM numbers (each SM's
utilization separately) you need `ncu`, which replays kernels and
runs 10–100× slower. To collect it, run `zeyu/profile_run.sh` on a
single node with the same model — the kernel set is the same as in
disagg mode, so the per-kernel SM numbers transfer over.

```bash
bash zeyu/profile_run.sh --model /home/zeyu/models/Qwen3-VL-8B-Instruct
```

(requires NVIDIA GPU performance counters enabled on the host; see
`docs/NCU_PERMISSIONS.md` in your cluster.)

## Troubleshooting

**`ValueError: Hybrid KV cache manager is disabled ...`**
You're trying to use a hybrid model (e.g. Qwen3.5). Switch to
Qwen3-VL-8B-Instruct or Qwen2.5-VL.

**`ValueError: Free memory on device cuda:0 ... less than desired`**
The selected GPU is already occupied. Pick a different GPU with
`--prefill-gpu` / `--decode-gpu` or lower `--gpu-memory-utilization`.

**`ModuleNotFoundError: No module named 'msgpack'`**
Run `pip install msgpack` inside the conda env on both nodes.

**`Address already in use`** on the ctrl port
A previous run left a socket in TIME_WAIT. Pick a different
`--ctrl-port` (any high port) or wait ~60s.

**`NCCL error` / hangs at engine init on the decode side**
Check that both nodes can reach each other on the NIC given by
`--iface` (`ping 192.168.0.42` from Node 1 should work). Confirm
`NCCL_SOCKET_IFNAME` gets set (the script exports `eth0` by
default). For dense cluster setups with back-end NICs, you can
override `--iface` to match your topology.

**`decode.log`: `Request id ... does not contain hostname and port`**
This indicates the producer/consumer got an ID that doesn't match
the P2pNcclConnector regex. Ensure both sides are running the same
version of `run_qwen35_vision_offline.py` and that
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` is set (the script sets
it automatically).

**`rsync FAILED` in the launcher (older logs only)**
The launcher now uses `tar`-over-SSH; rsync is no longer required.

## Notes on correctness

Qwen3-VL-8B-Instruct is a pure full-attention transformer, so the KV
cache transferred from the producer is usable by the consumer
without any linear-attention recurrent-state issues. If you observe
garbage output on the decode side, the likely causes are:

1. NCCL connectivity problem — the consumer times out or silently
   drops KV, then re-runs prefill itself (falling back to text).
2. Image preprocessing mismatch — the consumer processed its own
   copy of the image before realizing the KV was already there.
   For multimodal models, the consumer still passes the images
   through (they feed input_ids), but P2pNcclConnector short-
   circuits the actual attention compute.

For a sanity check, run a non-disaggregated single-GPU reference:

```bash
python zeyu/run_qwen35_vision_offline.py \
    --model /home/zeyu/models/Qwen3-VL-8B-Instruct \
    --num-prompts 4 \
    --max-tokens 64
```

and compare generated outputs.
