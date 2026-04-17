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

**No SSH between nodes is required.** The launcher is invoked
SEPARATELY on each node — once with `--role prefill`, once with
`--role decode` — in two independent terminals. The two sides
coordinate over a plain TCP ZMQ ctrl channel (prefill binds, decode
connects) and a NCCL peer-to-peer connection for the actual KV
transfer. After both sides finish, you copy the decode side's output
directory next to the prefill side's (scp / rsync / shared NFS — any
path that ends up with both subdirs under one parent works), and run
a merge command to produce a single `disagg_summary.json`.

---

## Contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [All command-line flags](#all-command-line-flags)
- [Providing your own dataset](#providing-your-own-dataset)
- [Output layout](#output-layout)
- [Viewing and processing metrics](#viewing-and-processing-metrics)
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

### Software (same on **both** nodes)

- Linux + CUDA toolkit + a working Python environment (venv / conda).
- This repository cloned somewhere (doesn't have to be the same path
  on both nodes — each invocation stays local).
- `pip install -e .` of this vLLM repo, OR an existing vLLM install
  whose source matches the one checked out here. In particular, this
  fork contains:
    - `zeyu/` launcher scripts (the entry point),
    - `vllm/v1/engine/iteration_logger.py` (per-iteration JSONL +
      pynvml sampling),
  so you must run the vLLM in *this* repo, not the pypi build.
- `pip install msgpack pynvml pyzmq` — all imported at runtime
  (`P2pNcclConnector` uses msgpack; the iteration logger uses pynvml;
  the cross-node ctrl channel uses pyzmq).
- The model weights available on both nodes. The two sides don't
  need *identical* paths, but each side has to point at a local copy
  via `--model`. Default is `Qwen/Qwen3-VL-8B-Instruct`.

### Network

- One NIC on each node with an IPv4 address the other node can reach.
  The launcher auto-detects the local IP on the `--iface` you pass.
- Two TCP ports must be open between the nodes (defaults `25500` for
  ctrl, `25555` and `25655` for KV-NCCL). Set these via
  `--ctrl-port` / `--kv-port` if your cluster firewall or pod network
  policy requires specific ones.
- Cross-node SSH is **not** needed for running the experiment. You
  only need it once, after the run, to copy the decode side's output
  directory back to the prefill side (and even that is optional if
  both nodes share a common filesystem).

### Optional: Nsight Systems (for kernel-level metrics)

Not required for basic metrics. See
[Optional: nsys profiling](#optional-nsys-profiling) for install
notes.

---

## Quick start

You will open **two terminals** — one on each node — and run the
same script with a different `--role`. Then you'll run a third
(short) command to merge the outputs.

### Step 0 — on BOTH nodes, find the NIC + a free GPU

```bash
# Find the NIC you'll use for NCCL/ZMQ. Pick any IPv4 interface
# that can reach the other node.
ip -4 -o addr

# Confirm reachability (replace with the OTHER node's IP):
ping -c 2 -I eth0 <other-node-ip>

# Pick a free GPU:
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
```

### Step 1 — sanity-check single-node first

Always verify the model + environment work on one GPU before
attempting disagg (saves hours of debugging):

```bash
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3-VL-8B-Instruct \
    --num-prompts 4 --max-tokens 64
```

### Step 2 — on the PREFILL node, terminal A

```bash
bash zeyu/disagg_run.sh --role prefill \
    --iface eth0 \
    --gpu 0 \
    --num-prompts 20 --max-tokens 64 \
    --model /path/to/Qwen3-VL-8B-Instruct
```

The prefill side loads the model (~30 s), prints its own IP address,
binds the ZMQ ctrl socket, and waits for decode. Note down two
things from its banner:

- **Prefill IP** (line labelled `Iface/IP   : eth0 = A.B.C.D`)
- **Output dir** (line labelled `Output dir  : ...`)

### Step 3 — on the DECODE node, terminal B

Use the prefill IP from step 2 as `--peer-ip`:

```bash
bash zeyu/disagg_run.sh --role decode \
    --peer-ip A.B.C.D \
    --iface eth0 \
    --gpu 0 \
    --num-prompts 20 --max-tokens 64 \
    --model /path/to/Qwen3-VL-8B-Instruct
```

**IMPORTANT**: the following flags must be **identical on both
sides**, otherwise the handshake / KV exchange won't match:
`--num-prompts`, `--max-tokens`, `--model`, `--max-model-len`,
`--kv-port`, `--ctrl-port`, and `--input` (if used).

Each side can differ on: `--iface` (each uses its own NIC name),
`--gpu` (each picks its own local GPU), `--output-dir` (per-side
output paths — see step 4).

### Step 4 — merge the outputs

When both terminals finish, each side has written to its own local
output dir:

```
PREFILL node: <prefill-OUT>/prefill/
DECODE  node: <decode-OUT>/decode/
```

Copy the decode side's `decode/` directory next to the prefill
side's `prefill/` so they become siblings under one parent. Pick
whichever method applies to your cluster:

```bash
# Option A — you have SSH / scp working between the nodes:
scp -r <decode-user>@<decode-host>:<decode-OUT>/decode \
       <prefill-OUT>/

# Option B — you have a shared filesystem (NFS / Lustre / GPFS):
# Pass the SAME --output-dir /mnt/shared/disagg_run1 on BOTH sides
# in steps 2 and 3. Nothing to copy — skip straight to the merge.

# Option C — use an intermediate host you can reach from both nodes
# (e.g. the cluster's login node):
#   login> scp <decode-host>:<decode-OUT>/decode <prefill-host>:<prefill-OUT>/
```

Then, on the node that now has both `prefill/` and `decode/`
subdirs:

```bash
python zeyu/merge_disagg.py <prefill-OUT>
```

The merged summary is written to `<prefill-OUT>/disagg_summary.json`
and a human-readable per-request table is printed to stdout.

---

## How it works

Two independent processes, one per node, coordinate over two channels:

```
 Prefill node (terminal A)                   Decode node (terminal B)
 ─────────────────────────                   ────────────────────────
 1. You launch:                              1. You launch:
       disagg_run.sh --role prefill             disagg_run.sh --role decode
       (no --peer-ip needed)                    --peer-ip <prefill-IP>
 2. Prefill loads Qwen3-VL with              2. Decode loads Qwen3-VL with
    kv_role=kv_producer on                      kv_role=kv_consumer on
    <iface IP>:kv_port.                         <iface IP>:(kv_port+100).
 3. Prefill binds a ZMQ PAIR                 3. Decode connects its
    ctrl socket on 0.0.0.0:ctrl-                ZMQ PAIR ctrl socket to
    port and waits for READY.                   <peer-IP>:ctrl-port and
                                                 sends a READY with its
 4. Prefill receives READY, learns              own IP + kv_port.
    decode's IP + kv_port, replies           4. Decode receives the ACK
    with an ACK carrying its own                 with prefill's IP +
    IP + kv_port.                                kv_port.
 5. Prefill mints request_ids of the form
    `reqN___prefill_addr_PIP:PPORT___decode_addr_DIP:DPORT_UID`
    (the format P2pNcclConnector's regex requires) and sends them
    to decode over ctrl. Both sides now use the SAME request IDs
    so the KV tensor keys match.
 6. Prefill runs                             6. Decode waits for
    generate(max_tokens=1). The                "prefill_done" on ctrl.
    vision encoder + prefill +              7. Decode runs
    first token are computed.                  generate(max_tokens=N).
    KV cache is pushed to decode               The consumer sees that KV
    over NCCL.                                 has already arrived and
                                                skips prefill compute —
 7. Prefill signals "prefill_done"             just decodes.
    over ctrl and waits for                  8. Decode writes latency.json
    "exit" before tearing down                  and iterations.jsonl
    NCCL.                                       to <OUT>/decode/, signals
 8. Prefill writes latency.json                 "exit" over ctrl, tears
    and iterations.jsonl to                     down NCCL, exits.
    <OUT>/prefill/. Exits.
```

**Two inter-node channels**, both over ordinary TCP:

| Channel | Use | Traffic volume |
|---|---|---|
| ZMQ PAIR (ctrl-port) | Handshake + tiny "done"/"exit" messages | < 1 KB total |
| NCCL (kv-port / kv-port+100) | GPU-to-GPU KV cache transfer | MBs–GBs per request |

Nothing else crosses the node boundary during the run. After the
run, the user (not the scripts) copies decode's output directory
back to the prefill side and runs `merge_disagg.py` locally.

---

## All command-line flags

`bash zeyu/disagg_run.sh --help` prints the full header comment from
the script. Supported flags:

### Required

| Flag | Applies to | Description |
|---|---|---|
| `--role {prefill\|decode}` | both sides | Which half of the pipeline this invocation runs. Mandatory. |
| `--peer-ip IP` | **decode only** | IP of the prefill node (the one decode connects to over ZMQ). **Not** required on `--role prefill` (prefill just binds and waits). |

### Must be identical on both sides

Mismatches cause the handshake to fail or the KV exchange to deadlock.

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | `/home/zeyu/models/Qwen3-VL-8B-Instruct` | Model path. The two sides load their own local copy; the PATHS don't need to match, but the MODEL weights must. |
| `--num-prompts N` | `4` | Number of built-in prompts to cycle through (ignored if `--input`). |
| `--max-tokens N` | `64` | Max tokens to generate per request. |
| `--max-model-len N` | `4096` | Context length. |
| `--input PATH` | *(none)* | JSONL dataset, see [Providing your own dataset](#providing-your-own-dataset). |
| `--kv-port N` | `25555` | Prefill side binds `N`; decode side binds `N+100` and connects to prefill's `N`. Must be open in both directions. |
| `--ctrl-port N` | `25500` | Control-channel ZMQ port (prefill binds, decode connects). Must be open decode→prefill. |

### Per-side (can differ between the two terminals)

| Flag | Default | Description |
|---|---|---|
| `--iface IFACE` | `eth0` | NIC to bind NCCL/ZMQ to. Each side uses its own local NIC name; the IPs will differ. |
| `--gpu IDX` | `0` | Local GPU index (as seen in `nvidia-smi` on *this* node). |
| `--gpu-memory-utilization F` | `0.85` | Fraction of GPU memory for KV cache + model. |
| `--output-dir DIR` | `zeyu/outputs/disagg_<UTC-timestamp>` | Base output directory. The role-specific subdir (`prefill/` or `decode/`) is created under it. |
| `--nsys` | off | Wrap this side's python in `nsys profile --trace=cuda,nvtx`. See [Optional: nsys profiling](#optional-nsys-profiling). Each side is independent — you can turn nsys on for one side only. |
| `--sm-metrics` | off | Additionally enable `--gpu-metrics-devices=all` (requires GPU counter privilege). |
| `--nsys-freq N` | `10000` | GPU-metrics sampling rate in Hz when `--sm-metrics` is on. |
| `--delay N` | *(none)* | Global per-request inter-arrival delay in ms. |

Change the defaults in the `---- Defaults ----` block at the top of
`zeyu/disagg_run.sh` if your cluster has a stable convention.

---

## Providing your own dataset

If you don't pass `--input`, the launcher uses 4 built-in toy prompts
cycled over `--num-prompts` requests. Good for a smoke test, useless
for real measurement (all requests have near-identical input length /
image size and the same two images, so you get no workload diversity).

### JSONL format

Pass `--input /path/to/data.jsonl`. One request per line. Each line
is a JSON object with these fields:

| Field | Type | Required? | Description |
|---|---|---|---|
| `text` | string | **yes** | The user-side prompt. The launcher wraps it in the Qwen3 chat template automatically — just put the raw user question. |
| `images` | string **or** list of strings | no | One or more **local file paths** to images (JPEG/PNG/WebP — anything PIL can open). If omitted or empty, the request is text-only. HTTP/S URLs are **not** fetched; download them to disk first. |
| `delay` | integer | no | Milliseconds to sleep before submitting this request (simulates arrival spacing). Default `0`. Overridden by `--delay` if that flag is set. |

### Example — `my_dataset.jsonl`

```jsonl
{"text": "What is in this picture?", "images": "/data/images/cat.jpg"}
{"text": "Summarize the chart.", "images": ["/data/images/chart.png"]}
{"text": "Compare these two screenshots.", "images": ["/data/a.png", "/data/b.png"]}
{"text": "Write a haiku about autumn.", "delay": 50}
{"text": "Describe the key finding.", "images": "/data/figure2.webp", "delay": 100}
```

Notes:
- **Each image path must be readable by BOTH the prefill and the
  decode process** (each side reads the dataset independently — the
  prefill side to compute the vision encoder, the decode side to
  tokenize the image placeholders). The paths don't have to be
  literally identical between the two nodes as long as each side
  gets the same bytes at the path you pass it; common options are a
  shared NFS / GPFS mount (both sides see the same path) or a
  per-node copy of the dataset directory (each side sees the same
  path locally).
- **The same `--input PATH` must be passed to both sides.** If the
  JSONL lists are different, the two sides will run different
  workloads and the KV transfer will deadlock or produce garbage.
- **Multi-image requests** are supported (pass a list). Each image
  is emitted as one `<image>` placeholder in the prompt.
- **Text-only requests** (no `images`) are allowed and go through
  the text-only fast path — no vision encoder, no multimodal cache.
- The chat template is fixed to Qwen3's `<|im_start|>user ...
  <|im_end|>` format. If you want a different template, edit
  `build_prompt()` in `run_qwen35_vision_offline.py`.

### Validate your dataset before running disagg

Don't debug a broken dataset across two nodes — validate it locally
on a single GPU first:

```bash
python zeyu/run_qwen35_vision_offline.py \
    --model /path/to/Qwen3-VL-8B-Instruct \
    --input /path/to/my_dataset.jsonl \
    --max-tokens 32
```

This prints per-request metrics and writes latency to
`zeyu/outputs/run_<timestamp>/latency.json`. If every request
completes with non-empty output and sensible `prefill_time_ms`,
you're good.

### Building a dataset from a common multimodal benchmark

There's no benchmark-specific loader in the repo — convert your
benchmark of choice into the JSONL schema above. Quick recipes:

- **COCO VQA / VQAv2**: for each question JSON record, emit one line
  with `text = question`, `images = image_path`.
- **MMBench / MME / MMMU**: flatten the multiple-choice question plus
  options into one string: `text = f"{question}\nA. {a}\nB. {b}\n
  ..."`, then optionally append `"Answer with the letter only."`.
- **ShareGPT4V / LLaVA-Bench**: just use the `question` + `image`
  fields directly.
- **TextVQA / DocVQA** (long prompts): no special handling needed;
  watch that `--max-model-len` covers the concatenated prompt +
  image-token count.

For throughput measurement it's important the requests have
**varied** text length and image size — identical requests let the
scheduler batch too cleanly and don't stress chunked prefill /
continuous batching.

---

## Output layout

During the run, **each side writes only its own subdir on its own
node**:

```
# On the PREFILL node:
<prefill-OUT>/
├── prefill/
│   ├── iterations.jsonl      # one line per scheduler iteration (prefill)
│   ├── requests.jsonl        # request_id → iteration indices mapping
│   ├── latency.json          # per-request: VE time, prefill time, wall ts
│   └── nsys_*.csv / *.nsys-rep      (only if --nsys)
├── prefill.log               # stdout/stderr from prefill
└── analyze_prefill.log       # (only if --nsys) output of analyze_profile.py

# On the DECODE node:
<decode-OUT>/
├── decode/
│   ├── iterations.jsonl      # one line per scheduler iteration (decode)
│   ├── requests.jsonl
│   ├── latency.json          # per-request: decode time, TBT
│   └── nsys_*.csv / *.nsys-rep      (only if --nsys)
├── decode.log                # stdout/stderr from decode
└── analyze_decode.log        # (only if --nsys) output of analyze_profile.py
```

After you copy the decode side's `decode/` directory into the prefill
side's `<prefill-OUT>/`, it looks like:

```
<prefill-OUT>/
├── prefill/      (was already here)
├── decode/       (copied from decode node — NEW)
├── prefill.log
├── analyze_prefill.log
└── (now run: python zeyu/merge_disagg.py <prefill-OUT>)
```

And `merge_disagg.py` adds:

```
<prefill-OUT>/
├── disagg_summary.json   ← merged per-request + aggregates (the main output)
├── prefill/consolidated_iterations.jsonl    (if nsys was on for prefill)
├── prefill/consolidated_requests.jsonl
├── decode/consolidated_iterations.jsonl     (if nsys was on for decode)
└── decode/consolidated_requests.jsonl
```

A human-readable table is also printed to stdout by `merge_disagg.py`.

---

## Viewing and processing metrics

After a run finishes, **`disagg_summary.json`** is the file you should
open first. It has two top-level sections:

```json
{
  "model": "/path/to/Qwen3-VL-8B-Instruct",
  "mode": "disagg",
  "summary": {
     "num_requests": 20,
     "avg_vision_encoder_time_ms": 61.28,
     "avg_prefill_time_ms": 312.75,
     "avg_kv_transfer_time_ms": 2.53,
     "avg_decode_time_ms": 1262.79,
     "avg_tpot_ms": 20.04,
     "avg_jct_ms": 1584.19,
     "rps_end_to_end": 1.95,
     "rps_decode_only": 2.88,
     "tbt_stats": {"count": 1240, "mean_ms": 20.02, "p50_ms": 19.91, "p95_ms": 21.36, "p99_ms": 22.67, ...},
     "vision_encoder": { ...per-phase averages... },
     "prefill":        { ...per-phase averages... },
     "decode":         { ...per-phase averages... }
  },
  "requests": [
     {"request_index": 0, "image_source": "cherry_blossom.jpg",
      "question": "What is in this picture?",
      "vision_encoder_time_ms": 68.19, "prefill_time_ms": 846.02,
      "kv_transfer_time_ms": 2.53, "decode_time_ms": 1284.0,
      "num_generation_tokens": 64, "tpot_ms": 20.38, "jct_ms": 2200.73,
      "per_token_tbt_ms": [60.9, 19.1, 19.0, ...],
      "tbt_stats": { ... },
      "vision_encoder": { ... }, "prefill": { ... }, "decode": { ... },
      "generated_text": "This image shows a cherry blossom tree..."},
     ... one per request ...
  ]
}
```

See the [Metrics reference](#metrics-reference) below for every field.

### The stdout table

At end-of-run the launcher prints a compact per-request table:

```
# | Image Source | VE(ms) | Pref(ms) | KV(ms) | Dec(ms) | GenTok | TPOT | p50TBT | JCT(ms)
0 | cherry.jpg   |  68.19 |   846.02 |   2.53 | 1284.00 |     64 | 20.38 |  20.05 |  2200.73
...
AVG|              |  61.28 |   312.75 |   2.53 | 1262.79 |     64 | 20.04 |  19.91 |  1584.19
RPS (end-to-end) = 1.946 | RPS (decode-only) = 2.877 | ...
```

That same data lives in `disagg_summary.json["requests"]` — useful
for eyeballing; export from the JSON if you need to plot or report.

### Quick queries with `jq`

```bash
OUT=zeyu/outputs/disagg_20260417_141119   # ← your run's dir

# Top-line numbers
jq '.summary | {num_requests, avg_prefill_time_ms, avg_decode_time_ms,
                avg_tpot_ms, avg_jct_ms, rps_end_to_end, rps_decode_only}' \
   $OUT/disagg_summary.json

# p50 / p95 / p99 TBT across all tokens
jq '.summary.tbt_stats' $OUT/disagg_summary.json

# Per-request table as CSV (idx, prefill_ms, decode_ms, jct_ms)
jq -r '.requests[] | [.request_index, .prefill_time_ms,
                      .decode_time_ms, .jct_ms] | @csv' \
   $OUT/disagg_summary.json

# Requests slower than 2 s end-to-end (by request_index)
jq '.requests | map(select(.jct_ms > 2000)) | [.[].request_index]' \
   $OUT/disagg_summary.json

# GPU utilization during the decode phase (pynvml + nsys)
jq '.summary.decode | {avg_nvml_gpu_util_pct_mean, avg_gpu_util_pct,
                       avg_text_forward_gpu_util_pct}' \
   $OUT/disagg_summary.json
```

### Python recipes for plotting / analysis

```python
import json, numpy as np, matplotlib.pyplot as plt

s = json.load(open("zeyu/outputs/disagg_20260417_141119/disagg_summary.json"))

# 1. TBT distribution across all tokens
tbts = [t for r in s["requests"] for t in r["per_token_tbt_ms"]]
print(f"TBT mean={np.mean(tbts):.2f} p50={np.median(tbts):.2f} "
      f"p95={np.percentile(tbts,95):.2f} p99={np.percentile(tbts,99):.2f}")
plt.hist(tbts, bins=80); plt.xlabel("TBT (ms)"); plt.show()

# 2. JCT breakdown stacked bar, per request
import numpy as np
reqs = s["requests"]
N = len(reqs)
ve   = [r["vision_encoder_time_ms"] for r in reqs]
pre  = [r["prefill_time_ms"]        for r in reqs]
kv   = [r["kv_transfer_time_ms"]    for r in reqs]
dec  = [r["decode_time_ms"]         for r in reqs]
x = np.arange(N)
plt.bar(x, ve, label="VE")
plt.bar(x, pre, bottom=ve, label="Prefill")
plt.bar(x, kv, bottom=np.add(ve, pre), label="KV transfer")
plt.bar(x, dec, bottom=np.add(np.add(ve, pre), kv), label="Decode")
plt.legend(); plt.xlabel("Request #"); plt.ylabel("Time (ms)"); plt.show()

# 3. Per-phase GPU utilization comparison (pynvml vs nsys if present)
for phase in ("vision_encoder", "prefill", "decode"):
    p = s["summary"].get(phase, {})
    print(f"{phase:15s}  nvml_util={p.get('avg_nvml_gpu_util_pct_mean','-'):>6}  "
          f"nsys_util={p.get('avg_gpu_util_pct','-'):>6}")
```

### Raw per-iteration data

`disagg_summary.json` aggregates; if you want every scheduler step,
open `<role>/iterations.jsonl`:

```bash
# Step count per phase (decode side)
jq -s 'group_by(.prefill_req_ids | length > 0) | map({n: length})' \
   $OUT/decode/iterations.jsonl

# Step latency of every iteration (decode side) as CSV
jq -r '[.iter, .step_latency_ms, .num_decode_tokens,
        .nvml_gpu_util_pct_mean] | @csv' \
   $OUT/decode/iterations.jsonl > decode_iter_timeline.csv
```

With `--nsys`, `<role>/consolidated_iterations.jsonl` has the same
shape plus kernel-derived fields (`gpu_util_pct`,
`kernel_launch_gap_pct`, `total_kernel_time_ns`,
`vision_encoder_gpu_util_pct`, `text_forward_gpu_util_pct`).

### Raw per-request data

`<role>/latency.json` is the canonical per-request output from each
side before merging. Use it if you need fields that `disagg_summary`
rolls up:

```bash
jq '.requests[0]' $OUT/prefill/latency.json    # prefill-side raw record
jq '.requests[0]' $OUT/decode/latency.json     # decode-side raw record
```

### Re-running the merger

If you tweak `merge_disagg.py` (or if the run finished but the final
merge failed for any reason) you can rebuild `disagg_summary.json`
without re-running the workload:

```bash
python zeyu/merge_disagg.py zeyu/outputs/disagg_20260417_141119/
```

Same for re-running the nsys analysis separately:

```bash
python zeyu/analyze_profile.py zeyu/outputs/disagg_20260417_141119/prefill/
python zeyu/analyze_profile.py zeyu/outputs/disagg_20260417_141119/decode/
```

### Comparing multiple runs

The directory-per-run layout makes A/B comparison straightforward.
Small helper:

```python
import json, glob, pandas as pd

rows = []
for d in sorted(glob.glob("zeyu/outputs/disagg_*/")):
    try:
        s = json.load(open(f"{d}disagg_summary.json"))["summary"]
    except FileNotFoundError:
        continue
    rows.append({
        "run": d.rstrip("/").split("/")[-1],
        **{k: s[k] for k in ("avg_vision_encoder_time_ms",
                             "avg_prefill_time_ms",
                             "avg_kv_transfer_time_ms",
                             "avg_decode_time_ms",
                             "avg_tpot_ms", "avg_jct_ms",
                             "rps_end_to_end", "rps_decode_only")},
    })
df = pd.DataFrame(rows)
print(df.to_string(index=False))
```

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

Pass `--nsys` to this side's invocation. The launcher wraps the
python process in `nsys profile --trace=cuda,nvtx
--cuda-graph-trace=node --trace-fork-before-exec=true`, then
exports the kernel + NVTX CSVs, then runs `analyze_profile.py` to
produce a `consolidated_iterations.jsonl` — all locally. Outputs
land under the role subdir:

```
<OUT>/{prefill|decode}/
├── nsys_report.nsys-rep
├── nsys_report.sqlite
├── nsys_kernels_cuda_gpu_trace.csv           # every kernel launch
├── nsys_nvtx_pushpop_nvtx_pushpop_trace.csv  # NVTX ranges (iter boundaries)
└── consolidated_iterations.jsonl             # iteration logs enriched with nsys data
```

You can turn `--nsys` on for one side only. `merge_disagg.py` uses
whatever data is available on each side.

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
too high. Check `nvidia-smi`, pick a different `--gpu` index, or
lower `--gpu-memory-utilization 0.7`.

**`ModuleNotFoundError: No module named 'msgpack'` (or `pynvml` / `zmq`)**
```bash
pip install msgpack pynvml pyzmq
```
In whichever env you use on **both** nodes.

**`Address already in use`**
A previous run left a socket in `TIME_WAIT`. Pick a different
`--ctrl-port` / `--kv-port` (any high free port) or wait ~60 s. If
BOTH previous sides were killed mid-run, also check for leftover
python processes with `pgrep -fa run_qwen35_vision_offline`.

**Decode side hangs at `Waiting for prefill to complete ...`**
The ZMQ ctrl channel connected but no `prefill_done` signal ever
arrived. Almost always a prefill-side crash. Look at the prefill
terminal for the real error; common causes are OOM on the prefill
GPU or a model-load failure.

**Decode errors with `Connection refused` on `ctrl-port`**
Prefill hasn't started yet, or `--peer-ip` on the decode side is
wrong. Confirm by:
```bash
nc -zv <prefill-ip> <ctrl-port>     # from the decode node
```
Prefill must be running first (or at least far enough along that
it's past the "Ctrl bound on 0.0.0.0:..." line). You can start
decode AFTER prefill is waiting — the scripts don't require tight
synchronization.

**NCCL hangs at `ncclCommInitRank`**
The NCCL TCP transport can't establish a peer-to-peer connection.
Verify from each node:
```bash
echo $NCCL_SOCKET_IFNAME        # must match what you passed as --iface
ping <peer-iface-ip>
nc -zv <peer-iface-ip> <kv-port>  # from decode, to prefill's kv_port
nc -zv <peer-iface-ip> <kv-port+100>  # from prefill, to decode's
```
The launcher exports `NCCL_SOCKET_IFNAME=$IFACE` and
`NCCL_IB_DISABLE=1` by default. If you want to use InfiniBand /
RDMA, pass `--iface <ib-name>` AND remove the `NCCL_IB_DISABLE=1`
line from `disagg_run.sh` manually.

**`Request id ... does not contain hostname and port`**
P2pNcclConnector's regex couldn't parse the request ID. Make sure
both nodes are on the **same commit** of this repo (`git log -1`
should match). The script sets
`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` automatically; if your
environment overrides it, the IDs will get mangled.

**Garbage / mismatched output from decode**
The KV producer-consumer handshake silently failed and the consumer
ran its own prefill fallback. Usually a symptom of the two nodes
having different model weights, or a network flake during NCCL
setup. Re-run and watch for `ncclCommInitRank Success` on both
sides — if you don't see it, the KV transfer never happened.

**`merge_disagg.py` complains that `decode/latency.json` is missing**
You ran the merge on the prefill side before copying decode's
output over. See step 4 in [Quick start](#quick-start).

**`Unable to retrieve the importer version` at end of nsys run**
The bundled Nsight-Compute nsys cannot finalize `.qdstrm` files.
Install standalone Nsight Systems — see
[Installing a working nsys](#installing-a-working-nsys).

**`Illegal --gpu-metrics-devices usage: Insufficient privilege`**
GPU performance counters are not accessible to non-admin users.
See [Optional: per-SM metrics](#optional-per-sm-metrics); otherwise
drop `--sm-metrics` and rely on the pynvml path.

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
