"""Shunt disaggregated-serving proxy (§4.1).

Replaces vLLM's round-robin disagg proxy with compute-aware request scheduling.
Incoming OpenAI-compatible requests are coalesced into per-tick batches, assigned
to prefill DP workers by :class:`RequestScheduler`, then driven through the
standard 1P/1D prefill->decode handshake (vLLM v1 + LMCache + Mooncake).

The driver (``shunt.harness.trace_replay``) sends each request's reused-prefix
and fresh-token counts in ``X-Shunt-Prefix-Tokens`` / ``X-Shunt-Fresh-Tokens`` so
the scheduler can price it exactly; without them we fall back to the prompt
length (prefix unknown).

Run::

    python -m shunt.serving.proxy --config proxy.json

where the JSON gives the prefill URLs (one per DP rank), decode URLs, mode, and
the compute profile. See shunt/harness for the launcher that starts the engines.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
import uuid

import aiohttp
from aiohttp import web

from ..compute_model import ComputeModel
from ..config import TESTBED
from .scheduler import RequestScheduler, SchedItem


class Proxy:
    def __init__(self, cfg: dict):
        self.prefill_urls: list[str] = cfg["prefill_urls"]
        self.decode_urls: list[str] = cfg["decode_urls"]
        self.num_workers = cfg.get("num_workers", TESTBED.ep_group_workers)
        model = (ComputeModel.from_profile(cfg["compute_profile"])
                 if cfg.get("compute_profile") else ComputeModel.default())
        ors_ranks = None
        if cfg.get("ors_rank_file"):
            with open(cfg["ors_rank_file"]) as f:
                ors_ranks = [int(x) for x in f.read().split()]
        self.scheduler = RequestScheduler(
            self.num_workers, model, mode=cfg.get("mode", "shunt"),
            ors_ranks=ors_ranks)
        self.sched_tick_s = cfg.get("sched_tick_ms", 2.0) / 1000.0
        self.sched_batch = cfg.get("sched_batch", 512)
        self.queue: asyncio.Queue = asyncio.Queue()
        self._decode_rr = itertools.cycle(range(len(self.decode_urls)))
        self.session: aiohttp.ClientSession | None = None
        # node-local planning (EAP + KVLB); switch the arms of §4.3 here
        self.model = model
        self.publish_plan = cfg.get("publish_plan", cfg.get("mode") != "baseline")
        self.enable_eap = cfg.get("enable_eap", True)
        self.enable_kvlb = cfg.get("enable_kvlb", True)
        self.theta = cfg.get("theta", 1.5)
        self.plan_dir = cfg.get("plan_dir", "/tmp/shunt")

    # --- per-tick batched scheduling (one RS pass per tick) --------------------

    async def scheduler_loop(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            await asyncio.sleep(self.sched_tick_s)
            while not self.queue.empty() and len(batch) < self.sched_batch:
                batch.append(self.queue.get_nowait())
            items = [b[0] for b in batch]
            assign = self.scheduler.assign_batch(items)
            if self.publish_plan:
                self._publish(items, assign)
            for item, fut in batch:
                if not fut.done():
                    fut.set_result(assign[item.req_id])

    def _publish(self, items: list[SchedItem], assign: dict[str, int]) -> None:
        """Turn the assignment into the node-local EAP + KVLB plan files."""
        from ..harness import node_planner
        from ..types import Request
        reqs = [Request(it.req_id, it.prefix_tokens, it.fresh_tokens)
                for it in items]
        worker_of = [assign[it.req_id] for it in items]
        try:
            node_planner.publish(reqs, worker_of, self.model, self.plan_dir,
                                 enable_eap=self.enable_eap,
                                 enable_kvlb=self.enable_kvlb, theta=self.theta)
        except Exception:  # planning must never stall request routing
            pass

    async def _schedule(self, item: SchedItem) -> int:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    # --- request handling ------------------------------------------------------

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        req_id = body.get("request_id") or f"shunt-{uuid.uuid4().hex[:16]}"
        prefix = int(request.headers.get("X-Shunt-Prefix-Tokens", 0))
        fresh = int(request.headers.get("X-Shunt-Fresh-Tokens", 0)) or \
            self._estimate_fresh(body)
        dp_rank = await self._schedule(SchedItem(req_id, prefix, fresh))
        return await self._forward_pd(request, body, req_id, dp_rank)

    def _estimate_fresh(self, body: dict) -> int:
        p = body.get("prompt") or body.get("messages")
        if isinstance(p, str):
            return max(1, len(p) // 4)   # ~4 chars/token, prefix unknown
        return 1

    def _prefill_target(self, dp_rank: int) -> tuple[str, dict]:
        """URL + extra headers to land the prefill on the chosen DP worker."""
        if len(self.prefill_urls) == self.num_workers:
            return self.prefill_urls[dp_rank], {}        # one engine per rank
        url = self.prefill_urls[dp_rank % len(self.prefill_urls)]
        return url, {"X-data-parallel-rank": str(dp_rank)}  # DP engine honors hint

    async def _forward_pd(self, request: web.Request, body: dict, req_id: str,
                          dp_rank: int) -> web.StreamResponse:
        assert self.session is not None
        path = request.path
        prefill_url, extra = self._prefill_target(dp_rank)
        decode_url = self.decode_urls[next(self._decode_rr)]

        # Phase 1: prefill one token, hand KV to the decode side via Mooncake.
        pf_body = dict(body)
        pf_body["max_tokens"] = 1
        pf_body["request_id"] = req_id
        pf_body["kv_transfer_params"] = {
            "do_remote_decode": True, "do_remote_prefill": False,
            "remote_engine_id": None, "remote_block_ids": None,
        }
        async with self.session.post(prefill_url + path, json=pf_body,
                                     headers=extra) as pf:
            pf_json = await pf.json()
        kvt = pf_json.get("kv_transfer_params", {})

        # Phase 2: decode reads the KV and streams the completion to the client.
        dec_body = dict(body)
        dec_body["request_id"] = req_id
        dec_body["kv_transfer_params"] = {
            "do_remote_decode": False, "do_remote_prefill": True, **kvt,
        }
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        await resp.prepare(request)
        async with self.session.post(decode_url + path, json=dec_body) as dec:
            async for chunk in dec.content.iter_any():
                await resp.write(chunk)
        await resp.write_eof()
        return resp

    # --- lifecycle -------------------------------------------------------------

    async def on_startup(self, app: web.Application) -> None:
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600))
        app["sched_task"] = asyncio.create_task(self.scheduler_loop())

    async def on_cleanup(self, app: web.Application) -> None:
        app["sched_task"].cancel()
        if self.session:
            await self.session.close()


def build_app(cfg: dict) -> web.Application:
    proxy = Proxy(cfg)
    app = web.Application()
    app.router.add_post("/v1/completions", proxy.handle)
    app.router.add_post("/v1/chat/completions", proxy.handle)
    app.on_startup.append(proxy.on_startup)
    app.on_cleanup.append(proxy.on_cleanup)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Shunt RS disaggregation proxy")
    ap.add_argument("--config", required=True, help="proxy JSON config")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    web.run_app(build_app(cfg), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
