"""The Shunt proxy for prefill-decode disaggregated serving.

For every OpenAI ``/v1/completions`` request it:

1. places the request on a prefill DP rank (:mod:`shunt.serving.placement`) and
   pins it there with vLLM's ``X-data-parallel-rank`` header;
2. runs the prefill with ``max_tokens=1`` and streams that first token to the
   client as soon as the prefill returns it;
3. hands the request to a decode instance (round-robin) and streams the rest
   of the output, dropping the decode's repeat of the first token.

Each request is logged as one JSON line (``log`` in the config). The trace
driver passes a request's estimated prefix and fresh tokens in
``X-Shunt-Prefix-Tokens`` / ``X-Shunt-Fresh-Tokens`` and its trace index in
``X-Shunt-Request-Id``.

Usage::

    python -m shunt.serving.proxy --config proxy.json

Config keys: ``prefill_url``, ``prefill_ranks``, ``decode_urls``,
``placement``, ``shunt_config`` (model and deployment for the compute
estimates), and optionally ``tick_ms``, ``oracle_file``, ``kv_events``
(one ZMQ endpoint per prefill DP rank), ``lb_factor``, ``log``, ``port``.
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

from .. import algorithms, native
from ..compute_model import ComputeModel
from ..config import ShuntConfig
from .placement import Placement


class Proxy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prefill_url = cfg["prefill_url"].rstrip("/")
        self.num_ranks = int(cfg["prefill_ranks"])
        self.decode_urls = [u.rstrip("/") for u in cfg["decode_urls"]]
        self._decode_rr = itertools.cycle(range(len(self.decode_urls)))
        sc = ShuntConfig.from_json(cfg["shunt_config"]) if cfg.get("shunt_config") \
            else ShuntConfig()
        self.cm = ComputeModel.from_profile(sc.compute_profile, sc.model, sc.deploy)
        lpt = native.lpt_schedule if native.available() else algorithms.lpt_schedule
        oracle = None
        if cfg.get("oracle_file"):
            with open(cfg["oracle_file"]) as f:
                oracle = {str(k): int(v) for k, v in json.load(f).items()}
        kv_index = None
        if cfg.get("kv_events"):
            from .kv_index import KVIndex

            kv_index = KVIndex(self.num_ranks)
            kv_index.subscribe(cfg["kv_events"], cfg.get("kv_events_topic", ""))
        self.placement = Placement(
            cfg.get("placement", "rr"), self.num_ranks, lpt,
            tick_s=float(cfg.get("tick_ms", 2.0)) / 1e3, oracle=oracle,
            kv_index=kv_index, lb_factor=float(cfg.get("lb_factor", 1.5)))
        self.log = open(cfg["log"], "a", buffering=1) if cfg.get("log") else None
        self.session: aiohttp.ClientSession | None = None

    # --- lifecycle -----------------------------------------------------------------

    async def on_startup(self, app) -> None:
        conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(
            connector=conn, timeout=aiohttp.ClientTimeout(total=None))
        self.placement.start()

    async def on_cleanup(self, app) -> None:
        if self.session is not None:
            await self.session.close()
        if self.log is not None:
            self.log.close()

    # --- request path --------------------------------------------------------------

    async def handle(self, request: web.Request) -> web.StreamResponse:
        t_arrive = time.time()
        body = await request.json()
        rid = request.headers.get("X-Shunt-Request-Id") or uuid.uuid4().hex[:16]
        prompt = body.get("prompt")
        tokens = prompt if isinstance(prompt, list) and prompt and \
            isinstance(prompt[0], int) else None
        prefix = int(request.headers.get("X-Shunt-Prefix-Tokens", 0))
        fresh = int(request.headers.get("X-Shunt-Fresh-Tokens", 0)) or \
            (len(tokens) if tokens else 1)
        est = self.cm.worker_time(prefix, fresh)
        rank = await self.placement.place(rid, est, tokens)
        rec = {"id": rid, "rank": rank, "prefix": prefix, "fresh": fresh,
               "est_s": est, "t_arrive": t_arrive, "t_placed": time.time()}

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        self.placement.started(rank, est)
        try:
            first, kvp = await self._prefill(body, rid, rank)
        except Exception as e:  # noqa: BLE001
            self.placement.finished(rank, est)
            rec["error"] = f"prefill: {e}"
            self._write(rec)
            await resp.write_eof()
            return resp
        self.placement.finished(rank, est)
        rec["t_first"] = time.time()
        await resp.write(_sse({"id": rid, "object": "text_completion",
                               "choices": [{"index": 0, "text": first,
                                            "finish_reason": None}]}))
        n_out = 1
        max_tokens = int(body.get("max_tokens", 16))
        if max_tokens > 1:
            try:
                n_out += await self._decode(body, rid, kvp, resp, rec)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"decode: {e}"
        rec["t_last"] = time.time()
        rec["n_out"] = n_out
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        self._write(rec)
        return resp

    async def _prefill(self, body: dict, rid: str, rank: int) -> tuple[str, dict]:
        pf = dict(body)
        pf.update({"max_tokens": 1, "stream": False, "request_id": rid + "-p"})
        pf.pop("stream_options", None)
        pf["kv_transfer_params"] = {"do_remote_decode": True,
                                    "do_remote_prefill": False,
                                    "remote_engine_id": None,
                                    "remote_block_ids": None}
        headers = {"X-data-parallel-rank": str(rank)}
        async with self.session.post(self.prefill_url + "/v1/completions", json=pf,
                                     headers=headers) as r:
            r.raise_for_status()
            out = await r.json()
        text = out["choices"][0].get("text", "")
        return text, out.get("kv_transfer_params") or {}

    async def _decode(self, body: dict, rid: str, kvp: dict, resp, rec: dict) -> int:
        url = self.decode_urls[next(self._decode_rr)]
        dec = dict(body)
        dec.update({"stream": True, "request_id": rid + "-d"})
        dec["kv_transfer_params"] = {"do_remote_decode": False,
                                     "do_remote_prefill": True, **kvp}
        n = 0
        skipped_first = False
        async with self.session.post(url + "/v1/completions", json=dec) as r:
            r.raise_for_status()
            async for line in r.content:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                if not skipped_first:
                    skipped_first = True      # the prefill already returned it
                    rec["t_decode_first"] = time.time()
                    continue
                n += 1
                await resp.write(line + b"\n")
        return n

    def _write(self, rec: dict) -> None:
        if self.log is not None:
            self.log.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def build_app(cfg: dict) -> web.Application:
    proxy = Proxy(cfg)
    app = web.Application(client_max_size=64 << 20)
    app.router.add_post("/v1/completions", proxy.handle)
    app.router.add_get("/health", lambda r: web.Response(text="ok"))
    app.on_startup.append(proxy.on_startup)
    app.on_cleanup.append(proxy.on_cleanup)
    app["proxy"] = proxy
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Shunt PD proxy")
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    web.run_app(build_app(cfg), host=a.host, port=a.port or int(cfg.get("port", 8000)),
                access_log=None)


if __name__ == "__main__":
    main()
