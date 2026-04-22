#!/usr/bin/env python3
"""
PD Disaggregated Serving Benchmark

Sends requests to the proxy endpoint and collects:
  - RPS (throughput)
  - TTFT (time to first token)
  - TBT (time between tokens, a.k.a. ITL)
  - JCT (job completion time, end-to-end latency)
  - Decode count (output tokens per request)

Usage:
  python benchmark.py --url http://192.168.0.42:9090 --num-requests 10
  python benchmark.py --url http://192.168.0.42:9090 --synthetic --input-len 1024
  python benchmark.py --url http://192.168.0.42:9090 --synthetic --input-len 1024 --qps 2.0
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Per-request timings, organized around the user's metric spec.

    Naming conventions:
      - ``t_*``  = epoch-seconds wall-clock timestamps (single instant)
      - ``d_*``  = millisecond durations derived from two ``t_*`` values

    Clock skew across the 4 nodes is NTP-bounded to <1ms (verified via
    chrony), so cross-node subtraction (e.g. client's t_2nd_token_recv
    minus decode-server's t_kv_end) is valid at the low-ms precision
    we care about.
    """

    req_id: int
    prompt_len: int
    output_tokens: int = 0
    ttft: float = 0.0    # seconds (legacy; equals d_ttft_ms/1000)
    itl: list[float] = field(default_factory=list)  # inter-token latencies
    jct: float = 0.0     # job completion time (seconds; = d_jct_ms/1000)
    server_id: str = ""   # completion id returned by proxy (cmpl-...)

    # ── Prefill / KV timestamps (server-side) ───────────────────────
    # Filled by LMCache adapter via producer JSONL on prefill node and
    # consumer JSONL on decode node, then joined by req_id.
    t_prefill_start: float = 0.0     # before first CUDA kernel of model forward
    t_prefill_end: float = 0.0       # after forward, cuda.synchronize()-ed
    t_kv_start: float = 0.0          # prefill dispatches KV save (CPU event)
    t_kv_mnck_in_end: float = 0.0    # all KV chunks landed in Mooncake segment
    t_kv_mnck_out_start: float = 0.0 # decode begins fetching from Mooncake
    t_kv_end: float = 0.0            # all KV copied to decode GPU; load_stream.synchronize()-ed

    # ── Client-side timestamps (benchmark.py, NTP-synced) ───────────
    t_req_sent: float = 0.0          # benchmark about to HTTP-POST the request
    t_1st_token_recv: float = 0.0    # chunk 1 arrived (prefill's sampled token)
    t_2nd_token_recv: float = 0.0    # chunk 2 arrived (decode's first produced token)
    t_decode_end: float = 0.0        # last chunk arrived at client

    # ── Proxy-side timestamp (piggy-backed via head_chunk server_metrics) ─
    t_dec_req_sent: float = 0.0      # proxy dispatched decode HTTP POST

    kv_mode: str = ""     # "layerwise" or "non_layerwise"
    success: bool = True
    error: str = ""

    # ═══════════════════════════════════════════════════════════════
    #  Derived durations (ms). All named ``d_<metric>_ms``.
    # ═══════════════════════════════════════════════════════════════

    @property
    def num_decode_tokens(self) -> int:
        """Tokens produced by the decode side.

        PD-disagg flow: proxy forces prefill's ``max_tokens=1`` (so it
        samples and returns exactly one token — the first token the
        client sees via ``choices[0].text``). Decode is then asked for
        ``original_max_tokens - 1`` tokens. So
        ``output_tokens = 1 + num_decode_tokens``.
        """
        return max(0, self.output_tokens - 1)

    # ── Prefill / KV durations ──────────────────────────────────────

    @property
    def d_prefill_ms(self) -> float:
        """Prefill compute: t_prefill_end − t_prefill_start."""
        return max(0.0, (self.t_prefill_end - self.t_prefill_start) * 1000)

    @property
    def d_kv_mnck_in_ms(self) -> float:
        """KV into Mooncake: t_kv_mnck_in_end − t_kv_start.
        GPU→CPU offload + Mooncake Put (local segment or RDMA WRITE to
        the decode segment's `preferred_segment`).
        """
        return max(0.0, (self.t_kv_mnck_in_end - self.t_kv_start) * 1000)

    @property
    def d_kv_mnck_ms(self) -> float:
        """Dwell in Mooncake segment: t_kv_mnck_out_start − t_kv_mnck_in_end.
        Includes prefill→proxy response + proxy→decode HTTP dispatch +
        decode scheduler pickup + first Get call entry.
        """
        return max(0.0, (self.t_kv_mnck_out_start - self.t_kv_mnck_in_end) * 1000)

    @property
    def d_kv_mnck_out_ms(self) -> float:
        """KV out of Mooncake: t_kv_end − t_kv_mnck_out_start.
        Mooncake Get (RDMA READ cross-node) + CPU→GPU H2D memcpy.
        Measured up to ``load_stream.synchronize()``.
        """
        return max(0.0, (self.t_kv_end - self.t_kv_mnck_out_start) * 1000)

    @property
    def d_total_kv_ms(self) -> float:
        """End-to-end KV: d_kv_mnck_in + d_kv_mnck + d_kv_mnck_out
        (== t_kv_end − t_kv_start).
        """
        return max(0.0, (self.t_kv_end - self.t_kv_start) * 1000)

    @property
    def d_total_kv_no_mnck_ms(self) -> float:
        """KV transfer only: d_kv_mnck_in + d_kv_mnck_out (no dwell)."""
        return max(0.0, self.d_kv_mnck_in_ms + self.d_kv_mnck_out_ms)

    @property
    def d_exposed_kv_ms(self) -> float:
        """KV time exposed on critical path: t_kv_end − t_prefill_end.
        In no-overlap this equals d_total_kv (t_kv_start == t_prefill_end).
        In overlap it's less, because KV save starts during prefill.
        """
        return max(0.0, (self.t_kv_end - self.t_prefill_end) * 1000)

    @property
    def d_exposed_kv_no_mnck_ms(self) -> float:
        """Exposed KV minus dwell: only the actual transfer latency
        that sits on the critical path.
        """
        return max(0.0, self.d_exposed_kv_ms - self.d_kv_mnck_ms)

    # ── Client-visible latency ──────────────────────────────────────

    @property
    def d_ttft_ms(self) -> float:
        """Time to first token: t_1st_token_recv − t_req_sent."""
        if self.t_1st_token_recv <= 0 or self.t_req_sent <= 0:
            return 0.0
        return max(0.0, (self.t_1st_token_recv - self.t_req_sent) * 1000)

    @property
    def d_jct_ms(self) -> float:
        """Job completion: t_decode_end − t_req_sent."""
        if self.t_decode_end <= 0 or self.t_req_sent <= 0:
            return 0.0
        return max(0.0, (self.t_decode_end - self.t_req_sent) * 1000)

    @property
    def d_jct_no_prefill_q_ms(self) -> float:
        """JCT without prefill queueing: t_decode_end − t_prefill_start."""
        if self.t_decode_end <= 0 or self.t_prefill_start <= 0:
            return 0.0
        return max(0.0, (self.t_decode_end - self.t_prefill_start) * 1000)

    # ── Decode-phase TBT metrics ────────────────────────────────────

    @property
    def d_1st_tbt_ms(self) -> float:
        """First TBT with proxy anchor: t_2nd_token_recv − t_dec_req_sent.
        Covers: decode HTTP in-flight + decode scheduler + KV transfer
        + first decode forward pass + decode→proxy→client response.
        Structurally symmetric with d_ttft (req-sent → token-recv).
        """
        if self.t_2nd_token_recv <= 0 or self.t_dec_req_sent <= 0:
            return 0.0
        return max(0.0, (self.t_2nd_token_recv - self.t_dec_req_sent) * 1000)

    @property
    def d_1st_tbt_no_kv_ms(self) -> float:
        """First TBT without KV transfer: t_2nd_token_recv − t_kv_end.

        Anchored at decode's "KV fully on GPU" moment rather than at
        the proxy's decode-request dispatch, so it isolates the
        duration of decode's first forward pass + response-back
        network. Should be ≈ d_tbt in no-overlap (first iter ≈ steady)
        but << d_tbt in layerwise overlap (t_kv_end fires after
        compute has been running layer-by-layer, capturing only the
        tail).
        """
        if self.t_2nd_token_recv <= 0 or self.t_kv_end <= 0:
            return 0.0
        return max(0.0, (self.t_2nd_token_recv - self.t_kv_end) * 1000)

    @property
    def d_decode_ms(self) -> float:
        """Decode phase: t_decode_end − t_kv_end.
        t_decode_start == t_kv_end by definition (decode can't start
        before KV is on GPU). So this is the decode-only span, no KV
        transfer time included.
        """
        if self.t_decode_end <= 0 or self.t_kv_end <= 0:
            return 0.0
        return max(0.0, (self.t_decode_end - self.t_kv_end) * 1000)

    @property
    def d_tbt_ms(self) -> float:
        """Average TBT across decode tokens: d_decode / num_decode_tokens.
        The (N-1) segments are: (t_kv_end → t_2nd_token_recv) and the
        (N-2) inter-decode-token gaps. Averaged over all N-1.
        """
        n = self.num_decode_tokens
        if n <= 0:
            return 0.0
        return self.d_decode_ms / n


PCTL_KEYS = ("p25", "p50", "p75", "p90", "p95", "p99")
PCTL_VALS = (25, 50, 75, 90, 95, 99)


def _pcts(data: list[float]) -> dict:
    """Compute P25/P50/P75/P90/P95/P99 for a list of values."""
    if not data:
        return {k: 0.0 for k in PCTL_KEYS}
    return {
        k: float(np.percentile(data, v))
        for k, v in zip(PCTL_KEYS, PCTL_VALS)
    }


@dataclass
class BenchmarkResult:
    num_requests: int
    num_completed: int
    num_failed: int
    total_time: float            # seconds
    rps: float                   # requests per second

    # TTFT — Time To First Token (ms)
    ttft: dict  # {p25, p50, p75, p90, p95, p99}

    # vLLM-convention reference metrics (client-side percentiles)
    #  - ttft should match d_ttft
    #  - itl  should approximate d_tbt (includes KV transfer on itl[0])
    #  - e2el should match d_jct
    ttft: dict = field(default_factory=dict)
    itl: dict = field(default_factory=dict)
    e2el: dict = field(default_factory=dict)

    # Input / output token counts
    avg_output_tokens: float = 0.0
    avg_input_tokens: float = 0.0

    # ── Mean of each duration metric (ms) ──────────────────────────
    #
    # Naming: ``avg_<metric>_ms`` where ``<metric>`` is the exact
    # name from the user spec (d_prefill, d_ttft, d_kv_mnck_in, ...).
    # Percentiles are omitted for now — single-request runs dominate
    # the test matrix, so mean is the only meaningful statistic.
    #
    # Prefill / KV
    avg_d_prefill_ms: float = 0.0
    avg_d_kv_mnck_in_ms: float = 0.0
    avg_d_kv_mnck_ms: float = 0.0
    avg_d_kv_mnck_out_ms: float = 0.0
    avg_d_total_kv_ms: float = 0.0
    avg_d_total_kv_no_mnck_ms: float = 0.0
    avg_d_exposed_kv_ms: float = 0.0
    avg_d_exposed_kv_no_mnck_ms: float = 0.0
    # Client-visible
    avg_d_ttft_ms: float = 0.0
    avg_d_jct_ms: float = 0.0
    avg_d_jct_no_prefill_q_ms: float = 0.0
    # TBT family
    avg_d_1st_tbt_ms: float = 0.0
    avg_d_1st_tbt_no_kv_ms: float = 0.0
    avg_d_decode_ms: float = 0.0
    avg_d_tbt_ms: float = 0.0
    # Reference (client-side, vLLM convention)
    avg_itl_ms: float = 0.0
    avg_itl_steady_ms: float = 0.0   # mean(itl[1:]) — cross-check for d_tbt

    # Raw per-request data
    requests: list[dict] = field(default_factory=list)

    # Per-request rows (RequestResult) joined with server-side metrics.
    # Populated for small runs so print_results can show raw t_*/d_*
    # timestamps for manual inspection.
    joined_rows: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt,
    max_tokens: int,
    model: str,
    req_id: int,
) -> RequestResult:
    """Send a streaming completions request and collect timing.

    `prompt` may be a string or a list of token IDs. When a list is
    supplied, the proxy skips its /tokenize step and we know exactly
    how many tokens go in.
    """
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "model": model,
        "stream": True,
    }

    if isinstance(prompt, list):
        prompt_len = len(prompt)
    else:
        prompt_len = len(prompt.split())
    result = RequestResult(req_id=req_id, prompt_len=prompt_len)
    # Parallel lists: ``token_times`` uses perf_counter (high-res
    # monotonic, used for TTFT/ITL/JCT durations). ``token_epochs``
    # uses wall-clock time.time() of the same instant, used to join
    # with server-side epoch timestamps (t_kv_end etc.) on other
    # nodes via NTP sync. Pair them so we don't call time.time()
    # twice per chunk (would double the jitter).
    token_times: list[float] = []
    token_epochs: list[float] = []

    try:
        # Anchor: capture both clocks at the same instant so we can
        # reconstruct each chunk's epoch time from its perf_counter
        # delta without another time.time() call per chunk.
        t_start = time.perf_counter()
        t_start_epoch = time.time()
        # t_req_sent: the "user client sent the request" moment. We
        # stamp just BEFORE session.post() returns the response header,
        # which is the closest we can get in aiohttp to "wire send
        # begins" without hooking the transport.
        result.t_req_sent = t_start_epoch
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            resp.raise_for_status()
            first_chunk = True
            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                now_perf = time.perf_counter()
                token_times.append(now_perf)
                token_epochs.append(t_start_epoch + (now_perf - t_start))
                # Capture completion id + proxy-side t_dec_req_sent
                # from chunk 1's server_metrics payload.
                if first_chunk:
                    first_chunk = False
                    try:
                        chunk_data = json.loads(data_str)
                        result.server_id = chunk_data.get("id", "")
                        sm = chunk_data.get("server_metrics") or {}
                        if "t_dec_req_sent" in sm:
                            result.t_dec_req_sent = float(sm["t_dec_req_sent"])
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        pass

        t_end = time.perf_counter()

        if not token_times:
            result.success = False
            result.error = "no tokens received"
            return result

        result.ttft = token_times[0] - t_start
        result.jct = t_end - t_start
        result.output_tokens = len(token_times)
        result.itl = [
            token_times[i] - token_times[i - 1]
            for i in range(1, len(token_times))
        ]
        # Decode-phase boundaries (epoch seconds, NTP-synced to
        # server-side t_kv_end etc.). The first streamed chunk is
        # prefill's sampled token (via ret_first_tok); decode's first
        # produced token is chunk 2.
        result.t_1st_token_recv = token_epochs[0]
        result.t_decode_end = token_epochs[-1]
        if len(token_epochs) >= 2:
            result.t_2nd_token_recv = token_epochs[1]

    except Exception as e:
        result.success = False
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Synthetic prompt generation
# ---------------------------------------------------------------------------

def generate_synthetic_prompt(input_len: int) -> str:
    """Generate a *deterministic* synthetic prompt whose length is
    approximately input_len tokens. Used only when the caller cannot
    pre-tokenize. Callers that need exact token count should use
    ``generate_exact_token_prompt`` (falls back to hitting the server's
    /tokenize endpoint once).
    """
    # Deterministic: same seed word repeated, no random.choice.
    num_words = max(1, int(input_len * 0.85))
    return " ".join(["hello"] * num_words)


def generate_exact_token_prompt(
    url: str, input_len: int, model: str,
    tokenize_cache: dict | None = None,
) -> list[int]:
    """Return a list of exactly ``input_len`` token IDs by calling the
    server's /tokenize endpoint once and slicing the result.

    We build a long deterministic seed string (no randomness), tokenize
    it, and return the first ``input_len`` token IDs. If the seed
    tokenizes to fewer tokens than requested, we double the seed and
    retry. The list is cached per (input_len) in ``tokenize_cache`` so
    repeated calls don't hit the server.

    The returned list goes straight into the /v1/completions request
    body as ``prompt`` — the proxy detects a list prompt and skips its
    own /tokenize round trip, so the exact count arrives at prefill.
    """
    if tokenize_cache is not None and input_len in tokenize_cache:
        return tokenize_cache[input_len]

    import urllib.request

    seed_word = "hello"
    # Start with ~1 word per token; grow seed until we have >= input_len.
    # Prepend a length-specific salt so different input_len values do NOT
    # share a common token prefix — otherwise decode's local prefix
    # cache (enable_prefix_caching=True by default) hits on the shared
    # prefix and short-circuits the Mooncake retrieve we are trying to
    # measure, leaving consumer JSONL empty for the longer sequences.
    words_factor = 2
    while True:
        num_words = max(input_len * words_factor, 16)
        salt = f"length_{input_len}_marker "
        seed = salt + " ".join([seed_word] * num_words)
        req = urllib.request.Request(
            f"{url}/tokenize",
            data=json.dumps({
                "prompt": seed,
                "model": model,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
        except Exception as e:
            raise RuntimeError(
                f"Failed to call /tokenize at {url}: {e}"
            )
        tokens = body.get("tokens") or body.get("input_ids") or []
        if len(tokens) >= input_len:
            exact = tokens[:input_len]
            if tokenize_cache is not None:
                tokenize_cache[input_len] = exact
            return exact
        # Seed too short, expand and retry.
        words_factor *= 2
        if words_factor > 1024:
            raise RuntimeError(
                f"Could not produce {input_len} tokens from seed expansion "
                f"(max factor {words_factor})"
            )


# ---------------------------------------------------------------------------
# Server-side metrics collection (from prefill worker JSONL files)
# ---------------------------------------------------------------------------

def _fetch_metrics_source(source: str) -> str:
    """Fetch JSONL content from one source. Returns "" on failure.

    source formats:
      "local:/path/to/file"
      "ssh:user@host:/path/to/file"
    """
    try:
        if source.startswith("local:"):
            path = source[len("local:"):]
            if not os.path.exists(path):
                return ""
            with open(path) as f:
                return f.read()
        if source.startswith("ssh:"):
            rest = source[len("ssh:"):]
            # rest = "user@host:/path"
            host, _, path = rest.partition(":")
            cmd = ["ssh", "-o", "BatchMode=yes", "-o",
                   "StrictHostKeyChecking=no", host, f"cat {path}"]
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=30)
            return out.stdout
    except Exception as e:
        print(f"  [warn] failed to fetch {source}: {e}")
    return ""


def _normalize_metric_req_id(rid: str) -> str:
    """Normalize to vLLM's two-segment completion id form
    ``cmpl-<hex>``. vLLM's internal Request.req_id is actually
    ``cmpl-<hex>-<rank>-<uuid>`` (four dash-separated parts; e.g.
    ``cmpl-b2dae2d7d6089fe7-0-beea237f``). The client's first streaming
    chunk ``id`` is the two-part prefix. Consumer JSONL (after the
    proxy correlation_id fix) writes the two-part form directly;
    producer JSONL writes the full four-part form. Collapsing both to
    two parts lets ``collect_server_metrics`` merge them under a single
    key.
    """
    if rid is None or not rid.startswith("cmpl-"):
        return rid
    parts = rid.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return rid


def collect_server_metrics(sources: list[str]) -> dict[str, dict]:
    """Fetch + parse JSONL from all sources; merge by normalized req_id.

    Producer and consumer JSONLs carry disjoint field sets (producer:
    t_prefill_*, t_kv_start, t_kv_mnck_in_end; consumer:
    t_kv_mnck_out_start, t_kv_end). Merging by ``dict.update`` keeps
    both sides' fields on the same req_id entry.

    The normalization step ensures that old producer JSONLs (with the
    long 'cmpl-<hex>-dp<N>-<uuid>' form) still join with new consumer
    JSONLs (short 'cmpl-<hex>' form from proxy correlation_id).
    """
    entries: dict[str, dict] = {}
    for src in sources:
        content = _fetch_metrics_source(src)
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_rid = rec.get("req_id")
            if not raw_rid:
                continue
            rid = _normalize_metric_req_id(raw_rid)
            existing = entries.setdefault(rid, {"req_id": rid})
            existing.update(rec)
            # Overwrite the stored req_id with the normalized form
            # (rec.update above may have re-written it from the file).
            existing["req_id"] = rid
    return entries


def attach_server_metrics(
    results: list[RequestResult], metrics: dict[str, dict],
) -> tuple[int, int]:
    """Populate each result with server-side timestamps.

    Matching strategy: adapter req_id looks like
    "cmpl-<hex>-<dp_rank>-<uuid>"; client sees server_id "cmpl-<hex>"
    (or the full form). Match by prefix in either direction.

    Returns ``(matched_producer, matched_consumer)``.
    """
    matched_prod = 0
    matched_cons = 0
    for r in results:
        if not r.success or not r.server_id:
            continue
        hit = None
        if r.server_id in metrics:
            hit = metrics[r.server_id]
        else:
            for rid, rec in metrics.items():
                if rid.startswith(r.server_id) or r.server_id.startswith(rid):
                    hit = rec
                    break
        if hit is None:
            continue
        has_producer = "t_prefill_start" in hit
        has_consumer = "t_kv_end" in hit
        if has_producer:
            r.t_prefill_start = float(hit["t_prefill_start"])
            r.t_prefill_end = float(hit["t_prefill_end"])
            r.t_kv_start = float(hit["t_kv_start"])
            r.t_kv_mnck_in_end = float(hit["t_kv_mnck_in_end"])
            if "mode" in hit:
                r.kv_mode = hit["mode"]
            matched_prod += 1
        if has_consumer:
            r.t_kv_mnck_out_start = float(hit["t_kv_mnck_out_start"])
            r.t_kv_end = float(hit["t_kv_end"])
            if "mode" in hit and not r.kv_mode:
                r.kv_mode = hit["mode"]
            matched_cons += 1
    return matched_prod, matched_cons


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def _send_warmup(
    session: aiohttp.ClientSession, url: str, model: str,
    input_len: int, max_tokens: int, count: int,
    prompt=None,
) -> None:
    """Fire ``count`` warmup requests and await completion.

    Results are discarded. Used to get past cold-start autotuning and
    first-request compilation so the measured run sees steady-state
    timing. ``prompt`` can be a pre-tokenized list (exact input_len)
    or None (falls back to deterministic string of approx length).
    """
    if count <= 0:
        return
    print(f"Warmup: {count} requests (input~{input_len}, out={max_tokens}) ...")
    if prompt is None:
        prompt = generate_synthetic_prompt(input_len)
    tasks = [
        asyncio.create_task(
            send_request(
                session, url,
                prompt,
                max_tokens, model, -(i + 1),
            )
        )
        for i in range(count)
    ]
    warm_results = await asyncio.gather(*tasks)
    ok = sum(1 for r in warm_results if r.success)
    print(f"  Warmup done: {ok}/{count} succeeded")


def _truncate_metrics_sources(sources: list[str]) -> None:
    """Best-effort truncate of every metrics JSONL source.

    Called after warmup so that the main run's metrics don't mix with
    warmup entries.
    """
    for src in sources:
        try:
            if src.startswith("local:"):
                path = src[len("local:"):]
                if os.path.exists(path):
                    open(path, "w").close()
            elif src.startswith("ssh:"):
                rest = src[len("ssh:"):]
                host, _, path = rest.partition(":")
                cmd = [
                    "ssh", "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=no",
                    host, f": > {path}",
                ]
                subprocess.run(cmd, capture_output=True, timeout=15)
        except Exception as e:
            print(f"  [warn] truncate {src} failed: {e}")


async def run_benchmark(args) -> BenchmarkResult:
    """Run the benchmark and return results."""

    # Build prompts
    prompts = []
    max_tokens_list = []
    if args.dataset:
        import jsonlines  # noqa: F811
        with open(args.dataset, "r") as f:
            for line in f:
                req = json.loads(line.strip())
                prompts.append(req["prompt"])
                max_tokens_list.append(req.get("max_tokens", args.max_tokens))
        args.num_requests = len(prompts)
    elif args.synthetic:
        if args.exact_tokens:
            # Unique token list per request: start from the common exact
            # token list, then mutate one interior position per request
            # (req index i → token index len//2, replace with i % vocab).
            # This keeps input_len exact while making each prompt's SHA
            # differ, so LMCache can't short-circuit prefill via KV cache
            # reuse from a prior iteration. The first ~3 leading tokens
            # are preserved so the tokenizer's BOS handling is stable.
            cache: dict = {}
            base_toks = generate_exact_token_prompt(
                args.url, args.input_len, args.model,
                tokenize_cache=cache,
            )
            print(f"  Synthetic input (exact): {len(base_toks)} tokens")
            prompts = []
            mid = len(base_toks) // 2
            for i in range(args.num_requests):
                toks = list(base_toks)
                # Swap one token near middle to a request-dependent id.
                # 100 + i stays safely inside Qwen3's vocab (>150K).
                if mid < len(toks):
                    toks[mid] = 100 + i
                prompts.append(toks)
        else:
            for _ in range(args.num_requests):
                prompts.append(generate_synthetic_prompt(args.input_len))
    elif args.prompt:
        prompts = [args.prompt] * args.num_requests
    else:
        prompts = ["Hello, my name is"] * args.num_requests

    if not max_tokens_list:
        max_tokens_list = [args.max_tokens] * len(prompts)

    # Poisson arrival if qps > 0
    arrival_times: list[float] = []
    if args.qps and args.qps > 0:
        for i in range(args.num_requests):
            if i == 0:
                arrival_times.append(0.0)
            else:
                interval = np.random.exponential(1.0 / args.qps)
                arrival_times.append(arrival_times[-1] + interval)
    else:
        arrival_times = [0.0] * args.num_requests  # all at once

    print(f"Benchmark: {args.num_requests} requests, "
          f"max_tokens={args.max_tokens}, model={args.model}")
    if args.synthetic:
        print(f"  Synthetic input: ~{args.input_len} tokens")
    if args.qps:
        print(f"  Poisson arrival: QPS={args.qps}")

    connector = aiohttp.TCPConnector(limit=max(args.concurrency, args.warmup))
    async with aiohttp.ClientSession(connector=connector) as session:
        if args.warmup > 0:
            warm_input_len = args.input_len if args.synthetic else 512
            warm_max_tokens = args.max_tokens
            # Reuse the same exact token list for warmup so proxy
            # skips /tokenize there too, and the warmup hits the same
            # exact prefill path as the measured run.
            warm_prompt = prompts[0] if (
                args.synthetic and args.exact_tokens and prompts
            ) else None
            await _send_warmup(
                session, args.url, args.model,
                warm_input_len, warm_max_tokens, args.warmup,
                prompt=warm_prompt,
            )
            if args.metrics_sources:
                srcs = [s.strip() for s in args.metrics_sources.split(",")
                        if s.strip()]
                _truncate_metrics_sources(srcs)
            # Let any tail callbacks finish before measuring.
            await asyncio.sleep(0.5)

        # Actually gate concurrent in-flight requests. Before, --concurrency
        # was only a TCP connection-pool hint; send_request tasks were all
        # created at once and executed in parallel regardless. For
        # measurements where the user wants strict sequential execution
        # (concurrency=1, isolating each request's full lifecycle without
        # queueing interactions) or a fixed max-in-flight window, gate here.
        sem = asyncio.Semaphore(max(1, args.concurrency))

        async def _gated_send(prompt, max_tok, req_id):
            async with sem:
                return await send_request(
                    session, args.url, prompt, max_tok, args.model, req_id,
                )

        tasks = []
        t_bench_start = time.perf_counter()

        for i in range(args.num_requests):
            # Wait for arrival time
            wait = arrival_times[i] - (time.perf_counter() - t_bench_start)
            if wait > 0:
                await asyncio.sleep(wait)

            task = asyncio.create_task(
                _gated_send(prompts[i], max_tokens_list[i], i)
            )
            tasks.append(task)

        results: list[RequestResult] = await asyncio.gather(*tasks)
        t_bench_end = time.perf_counter()

    total_time = t_bench_end - t_bench_start

    # Collect per-request server-side metrics from prefill worker JSONLs
    if args.metrics_sources:
        sources = [s.strip() for s in args.metrics_sources.split(",")
                   if s.strip()]
        metrics_map = collect_server_metrics(sources)
        m_prod, m_cons = attach_server_metrics(results, metrics_map)
        print(f"  Server metrics: producer={m_prod}/{len(results)} "
              f"consumer={m_cons}/{len(results)} "
              f"(from {len(metrics_map)} merged entries)")

    # Aggregate metrics
    completed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not completed:
        print("ERROR: All requests failed!")
        return BenchmarkResult(
            num_requests=args.num_requests,
            num_completed=0, num_failed=len(failed),
            total_time=total_time, rps=0,
        )

    # vLLM-convention references (client-only, no server metrics needed)
    ttfts_ms = [r.ttft * 1000 for r in completed]
    e2els_ms = [r.jct * 1000 for r in completed]
    all_itls_ms = []
    for r in completed:
        all_itls_ms.extend([x * 1000 for x in r.itl])
    itl_steady_means = [
        sum(r.itl[1:]) * 1000 / len(r.itl[1:])
        for r in completed if len(r.itl) > 1
    ]

    # Only include requests where BOTH producer JSONL, consumer JSONL
    # and proxy t_dec_req_sent arrived — otherwise d_* are meaningless.
    def _has_full(r: RequestResult) -> bool:
        return (r.t_prefill_start > 0 and r.t_kv_end > 0
                and r.t_req_sent > 0)

    ok = [r for r in completed if _has_full(r)]

    def _collect(attr: str, pred=None) -> list:
        """Collect a per-request duration only when the timestamps
        feeding it exist (property returns 0 otherwise)."""
        vals = []
        for r in ok:
            v = getattr(r, attr)
            if v > 0:
                vals.append(v)
        return vals

    def _mean(xs) -> float:
        return float(np.mean(xs)) if xs else 0.0

    result = BenchmarkResult(
        num_requests=args.num_requests,
        num_completed=len(completed),
        num_failed=len(failed),
        total_time=total_time,
        rps=len(completed) / total_time if total_time > 0 else 0,

        ttft=_pcts(ttfts_ms),
        itl=_pcts(all_itls_ms),
        e2el=_pcts(e2els_ms),

        avg_output_tokens=float(np.mean([r.output_tokens for r in completed])),
        avg_input_tokens=float(np.mean([r.prompt_len for r in completed])),

        # Prefill / KV
        avg_d_prefill_ms=_mean(_collect("d_prefill_ms")),
        avg_d_kv_mnck_in_ms=_mean(_collect("d_kv_mnck_in_ms")),
        avg_d_kv_mnck_ms=_mean(_collect("d_kv_mnck_ms")),
        avg_d_kv_mnck_out_ms=_mean(_collect("d_kv_mnck_out_ms")),
        avg_d_total_kv_ms=_mean(_collect("d_total_kv_ms")),
        avg_d_total_kv_no_mnck_ms=_mean(_collect("d_total_kv_no_mnck_ms")),
        avg_d_exposed_kv_ms=_mean(_collect("d_exposed_kv_ms")),
        avg_d_exposed_kv_no_mnck_ms=_mean(_collect("d_exposed_kv_no_mnck_ms")),
        # Client-visible
        avg_d_ttft_ms=_mean(_collect("d_ttft_ms")),
        avg_d_jct_ms=_mean(_collect("d_jct_ms")),
        avg_d_jct_no_prefill_q_ms=_mean(_collect("d_jct_no_prefill_q_ms")),
        # TBT family
        avg_d_1st_tbt_ms=_mean(_collect("d_1st_tbt_ms")),
        avg_d_1st_tbt_no_kv_ms=_mean(_collect("d_1st_tbt_no_kv_ms")),
        avg_d_decode_ms=_mean(_collect("d_decode_ms")),
        avg_d_tbt_ms=_mean(_collect("d_tbt_ms")),
        # Reference
        avg_itl_ms=_mean(all_itls_ms),
        avg_itl_steady_ms=_mean(itl_steady_means),

        requests=[asdict(r) for r in results],
        joined_rows=ok,
    )

    return result


def print_results(result: BenchmarkResult):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Completed:    {result.num_completed}/{result.num_requests} "
          f"({result.num_failed} failed)")
    print(f"  Total time:   {result.total_time:.2f}s")
    print(f"  RPS:          {result.rps:.2f} req/s")
    print(f"  Avg input:    {result.avg_input_tokens:.0f} tokens")
    print(f"  Avg output:   {result.avg_output_tokens:.0f} tokens")
    print()
    def _fmt(d: dict) -> str:
        if not d:
            return "  ".join(f"{k.upper()}=0.0" for k in PCTL_KEYS)
        return "  ".join(f"{k.upper()}={d.get(k, 0.0):.1f}" for k in PCTL_KEYS)

    # vLLM-convention references (client-only, always available)
    print(f"  TTFT  (ms):       {_fmt(result.ttft)}")
    print(f"  ITL   (ms):       {_fmt(result.itl)}")
    print(f"  E2EL  (ms):       {_fmt(result.e2el)}")

    has_server = result.avg_d_total_kv_ms > 0
    if not has_server:
        print("  (Server-side KV metrics: no joined producer+consumer "
              "records — check metrics_sources)")
    print()
    print("  Averages (ms) — per user spec (t_*/d_* naming):")
    # ── Prefill / KV ────────────────────────────────────────────────
    print(f"    d_prefill              {result.avg_d_prefill_ms:9.2f}  "
          "(t_prefill_end − t_prefill_start, cuda-synced)")
    print(f"    d_kv_mnck_in           {result.avg_d_kv_mnck_in_ms:9.2f}  "
          "(t_kv_start → t_kv_mnck_in_end: GPU→CPU + Mooncake Put)")
    print(f"    d_kv_mnck              {result.avg_d_kv_mnck_ms:9.2f}  "
          "(t_kv_mnck_in_end → t_kv_mnck_out_start: dwell in Mooncake)")
    print(f"    d_kv_mnck_out          {result.avg_d_kv_mnck_out_ms:9.2f}  "
          "(t_kv_mnck_out_start → t_kv_end: Mooncake Get + CPU→GPU)")
    print(f"    d_total_kv             {result.avg_d_total_kv_ms:9.2f}  "
          "(t_kv_start → t_kv_end: in + dwell + out)")
    print(f"    d_total_kv_no_mnck     {result.avg_d_total_kv_no_mnck_ms:9.2f}  "
          "(in + out only, excludes dwell)")
    print(f"    d_exposed_kv           {result.avg_d_exposed_kv_ms:9.2f}  "
          "(t_kv_end − t_prefill_end; = d_total_kv in no-overlap)")
    print(f"    d_exposed_kv_no_mnck   "
          f"{result.avg_d_exposed_kv_no_mnck_ms:9.2f}  "
          "(d_exposed_kv − d_kv_mnck)")
    # ── Client-visible ──────────────────────────────────────────────
    print(f"    d_ttft                 {result.avg_d_ttft_ms:9.2f}  "
          "(t_1st_token_recv − t_req_sent)")
    print(f"    d_jct                  {result.avg_d_jct_ms:9.2f}  "
          "(t_decode_end − t_req_sent)")
    print(f"    d_jct_no_prefill_q     "
          f"{result.avg_d_jct_no_prefill_q_ms:9.2f}  "
          "(t_decode_end − t_prefill_start)")
    # ── TBT family ──────────────────────────────────────────────────
    print(f"    d_1st_tbt              {result.avg_d_1st_tbt_ms:9.2f}  "
          "(t_2nd_token_recv − t_dec_req_sent: proxy-anchored)")
    print(f"    d_1st_tbt_no_kv        {result.avg_d_1st_tbt_no_kv_ms:9.2f}  "
          "(t_2nd_token_recv − t_kv_end: ≈ d_tbt in no-overlap)")
    print(f"    d_decode               {result.avg_d_decode_ms:9.2f}  "
          "(t_decode_end − t_kv_end: decode-only, no KV transfer)")
    print(f"    d_tbt                  {result.avg_d_tbt_ms:9.2f}  "
          "(d_decode / num_decode_tokens)")
    # ── Reference (client-side vLLM) ────────────────────────────────
    print(f"    ITL mean (all)         {result.avg_itl_ms:9.2f}  "
          "(client-side, includes ITL[0])")
    print(f"    ITL mean (steady)      {result.avg_itl_steady_ms:9.2f}  "
          "(mean(itl[1:]); cross-check for d_tbt)")

    if has_server:
        # Per-request raw timestamps for analysis. Only shown when the
        # joined metric set is small (<= 10 requests) so the output
        # stays readable. Millisecond-resolution deltas from a common
        # baseline (t_req_sent of the first completed request).
        joined = [r for r in result.joined_rows] if hasattr(result, "joined_rows") else []
        if joined and len(joined) <= 10:
            t0 = joined[0].t_req_sent or joined[0].t_prefill_start
            print()
            print("  Per-request timeline (ms from t_req_sent of req 0):")
            hdr = ("    {:>3}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  "
                   "{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}")
            print(hdr.format(
                "#", "t_req_s", "t_pre_s", "t_pre_e", "t_kv_s",
                "t_in_end", "t_out_st", "t_kv_e", "t_1st_r",
                "t_dec_s", "t_2nd_r", "t_dec_e",
            ))
            row = ("    {:>3}  {:8.1f}  {:8.1f}  {:8.1f}  {:8.1f}  "
                   "{:8.1f}  {:8.1f}  {:8.1f}  {:>8}  {:>8}  {:>8}  "
                   "{:>8}")
            def _fmt_t(t):
                return f"{(t - t0) * 1000:8.1f}" if t > 0 else "       -"
            for idx, r in enumerate(joined):
                print(row.format(
                    idx,
                    (r.t_req_sent - t0) * 1000 if r.t_req_sent > 0 else 0.0,
                    (r.t_prefill_start - t0) * 1000,
                    (r.t_prefill_end - t0) * 1000,
                    (r.t_kv_start - t0) * 1000,
                    (r.t_kv_mnck_in_end - t0) * 1000,
                    (r.t_kv_mnck_out_start - t0) * 1000,
                    (r.t_kv_end - t0) * 1000,
                    _fmt_t(r.t_1st_token_recv),
                    _fmt_t(r.t_dec_req_sent),
                    _fmt_t(r.t_2nd_token_recv),
                    _fmt_t(r.t_decode_end),
                ))
    print("=" * 60)


def save_results(result: BenchmarkResult, output_dir: str, tag: str = ""):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"bench_{tag}_{ts}.json" if tag else f"bench_{ts}.json"
    path = os.path.join(output_dir, fname)

    # Save without raw per-request data for summary
    summary = {k: v for k, v in asdict(result).items() if k != "requests"}
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {path}")

    # Also save full data with per-request details
    full_path = path.replace(".json", "_full.json")
    with open(full_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Full data saved to: {full_path}")

    return path


def main():
    parser = argparse.ArgumentParser(description="PD Disagg Serving Benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:9090",
                        help="Proxy URL")
    parser.add_argument("--model", type=str, default="Qwen3-235B",
                        help="Model name")
    parser.add_argument("--num-requests", type=int, default=10,
                        help="Number of requests")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max output tokens per request")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent requests")
    parser.add_argument("--qps", type=float, default=0,
                        help="Target QPS (Poisson arrival). 0 = send all at once")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Fire N warmup requests first (not counted in "
                             "results). Metrics JSONL is truncated after "
                             "warmup so only main-run entries are aggregated.")

    # Synthetic data
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic prompts")
    parser.add_argument("--input-len", type=int, default=1024,
                        help="Input length in tokens (with --synthetic). "
                             "With --exact-tokens the value is exact.")
    parser.add_argument("--exact-tokens", action="store_true",
                        help="Pre-tokenize via server /tokenize and pass "
                             "a token list to /v1/completions so the "
                             "prompt is exactly --input-len tokens (no "
                             "random, no approximation). Requires the "
                             "proxy to skip /tokenize for list prompts.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Fixed prompt to use for all requests")
    parser.add_argument("--dataset", type=str, default=None,
                        help="JSONL file from gen_synthetic_data.py")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results JSON")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for result filename")

    # Server-side metrics: comma-separated list of JSONL sources.
    # Each source is "local:/path" or "ssh:user@host:/path". Producer
    # JSONL is written by prefill workers; consumer JSONL by decode
    # workers. Both are merged by adapter req_id and matched to the
    # client's server_id by prefix.
    parser.add_argument(
        "--metrics-sources", type=str,
        default=(
            # producer (prefill) — this node + node-1
            "local:/home/zeyu/lmcache_metrics_producer.jsonl,"
            "ssh:zeyu@lj1.zeyu.tw:/home/zeyu/lmcache_metrics_producer.jsonl,"
            # consumer (decode) — node-2 + node-3
            "ssh:zeyu@lj2.zeyu.tw:/home/zeyu/lmcache_metrics_consumer.jsonl,"
            "ssh:zeyu@lj3.zeyu.tw:/home/zeyu/lmcache_metrics_consumer.jsonl"
        ),
        help="Comma-separated list of JSONL metric sources "
             "(local:/path or ssh:user@host:/path). Empty to disable.",
    )

    args = parser.parse_args()

    result = asyncio.run(run_benchmark(args))
    print_results(result)

    if args.output_dir:
        save_results(result, args.output_dir, args.tag)


if __name__ == "__main__":
    main()
