# SPDX-License-Identifier: Apache-2.0
"""NIXL-backed EC connector for vision-text disaggregation (BL2).

Producer side runs on the vision instance: after `_execute_mm_encoder`
populates `encoder_cache[mm_hash]`, the runner calls `save_caches(...)` which
registers that GPU tensor with NIXL, advertises it to the consumer via a
notif, and waits for the consumer to read it.

Consumer side runs on the text (prefill+decode) instance: when the scheduler
allocates a request that needs an mm_hash we don't have locally,
`update_state_after_alloc` queues it; `build_connector_meta` ships the queue
to the worker; worker's `start_load_caches` blocks on the producer's
"PUSH|<mm_hash>|..." notif, NIXL-pulls the tensor, and drops it into the
local `encoder_cache`.

Bootstrap (one-shot per process): the producer opens a small TCP listener on
`peer_endpoint` (host:port from `ec_connector_extra_config`). The consumer
dials in once, reads the producer's NIXL agent metadata bytes, and registers
the remote agent. After bootstrap, all data and control flow through NIXL.

Per-transfer protocol (after bootstrap):
  Producer    → notif to consumer: "PUSH|<mm_hash>|<addr>|<n_bytes>|<shape>|<dtype>"
  Consumer    → NIXL READ from producer's address into local buffer
                with notif_msg="ACK|<mm_hash>"; NIXL delivers ACK to producer
                automatically when the read completes.
  Producer    ← unblocks `save_caches` on the ACK notif, then deregisters.

Both sides additionally emit one event per transfer to the BL2 vemb sidecar
(see `vllm.v1.observability.bl2`) so that `merge_metrics.py` can compute
`d_vemb_transfer = t_recv_done - t_send_done` per mm_hash. Cross-node host
clocks are assumed NTP-synced.

Limitations (acceptable for a baseline):
- Synchronous: each `save_caches` blocks until the consumer's READ completes.
  vLLM's `wait_for_save` semantics tolerate this.
- One mm_hash per call, no batching.
- Memory is registered/deregistered per-transfer rather than from a
  pre-registered pool. Adds ~µs latency that's irrelevant at MB-scale
  embeddings over 200 Gbps RDMA.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger
from vllm.v1.observability import bl2 as _bl2

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


# Sentinel used by NixlWrapper for self-side prepped descs.
_NIXL_INIT_AGENT = "NIXL_INIT_AGENT"

# 4-byte big-endian length-prefix for the bootstrap TCP frames.
_LEN_PREFIX = struct.Struct(">I")

# Length-bound for a single boot handshake message (sanity).
_MAX_BOOT_BYTES = 1 << 24  # 16 MiB

# Polling cadence while waiting on producer notifs / xfer completion.
_POLL_INTERVAL_S = 0.001

# Producer slot-reclaim safety net. Typical pull latency is sub-second; even
# under congestion we've never observed pulls beyond a few seconds. Any slot
# inflight beyond this is overwhelmingly likely to indicate that BOTH the
# NIXL auto-ACK and the consumer's explicit app-level ACK were dropped on the
# notif AM control-plane. Reclaiming the slot is then safe — the consumer
# either already pulled the data or will eventually retry via its own
# wait_for_notif timeout, which will land in a subsequent re-advertise cycle.
_SLOT_RECLAIM_TIMEOUT_S = 180.0


# ---- metadata shipped scheduler -> worker -----------------------------------


@dataclass
class _MMMeta:
    mm_hash: str
    num_token: int


@dataclass
class NixlECConnectorMetadata(ECConnectorMetadata):
    """Per-step metadata: the mm_hashes the consumer needs to load this step,
    plus a side map of mm_hash -> req_ids consuming it (for sidecar logging)."""

    mm_datas: list[_MMMeta] = field(default_factory=list)
    req_ids_by_hash: dict[str, list[str]] = field(default_factory=dict)


# ---- NIXL endpoint wrapper ---------------------------------------------------


class _NixlEndpoint:
    """One NIXL agent + one TCP bootstrap channel.

    Producer mode: hosts a TCP listener that ships agent metadata to any
    consumer that connects, then drains incoming notifications.

    Consumer mode: dials the producer's TCP listener once, reads the producer's
    metadata, registers it as a remote agent, and queues incoming PUSH notifs
    keyed by mm_hash.
    """

    def __init__(self, role: ECConnectorRole, is_producer: bool, extra_cfg: dict[str, Any], device_id: int):
        if NixlWrapper is None:
            raise RuntimeError(
                "NIXL is not installed in this environment but the "
                "NixlECConnector requires it. Install nixl or pick a "
                "different ec_connector."
            )

        self._is_producer = is_producer
        self._device_id = int(device_id)
        self._extra_cfg = extra_cfg

        # NIXL backend selection. Defaults to UCX (the bond RNICs are RDMA-
        # capable; UCX picks IB by default when available).
        self._nixl_backends = list(extra_cfg.get("backends", ["UCX"]))
        # Optional explicit device pin (the bond RNIC name, e.g. "mlx5_0").
        # If unset, NIXL/UCX picks by topology — usually fine on a dual-bond
        # setup, but we surface the choice for traceability.
        self._nixl_dev = extra_cfg.get("nixl_dev")
        if self._nixl_dev:
            # UCX honours UCX_NET_DEVICES env to pin the HCA.
            import os
            os.environ.setdefault("UCX_NET_DEVICES", self._nixl_dev + ":1")

        config = None
        if nixl_agent_config is not None:
            config = nixl_agent_config(backends=self._nixl_backends)

        self._engine_id = str(uuid.uuid4())
        self._wrapper = NixlWrapper(self._engine_id, config)

        # Pre-allocate a ring of scratch slots on the device and register them
        # all with NIXL ONCE. NIXL only knows about memory that was registered
        # before `get_agent_metadata()` returned the bytes the peer
        # `add_remote_agent`'d. With multiple slots, save_caches doesn't have
        # to block on the consumer's ACK before returning — the next save
        # claims a different slot, the consumer reads them in any order, and
        # the notif drain thread releases each slot when its ACK arrives.
        #
        # Each slot is sized for one encoder output. Default 24 MiB
        # comfortably covers Qwen3-VL deep-stacked embeddings (~10 MB).
        self._slot_size = int(extra_cfg.get("slot_bytes", 24 * 1024 * 1024))
        self._n_slots = int(extra_cfg.get("n_slots", 16))
        self._scratch_n_bytes = self._slot_size * self._n_slots
        self._scratch = torch.empty(
            self._scratch_n_bytes,
            dtype=torch.uint8,
            device=f"cuda:{self._device_id}",
        )
        self._scratch_addr = int(self._scratch.data_ptr())
        self._scratch_descs = self._wrapper.get_reg_descs(
            [(self._scratch_addr, self._scratch_n_bytes,
              self._device_id, "")],
            "VRAM",
        )
        self._wrapper.register_memory(
            self._scratch_descs, backends=self._nixl_backends)
        # Producer ring management: free queue + per-mm_hash slot map. Notif
        # drain thread auto-releases a slot when its matching ACK arrives.
        self._free_slots: deque[int] = deque(range(self._n_slots))
        self._slots_cv = threading.Condition()
        self._inflight_slots: dict[str, int] = {}
        # Producer-only: per-mm_hash PUSH msg, kept until ACK so the periodic
        # resender thread can retransmit if NIXL/UCX dropped the original.
        # Empirically, after ~5–10 k advertises the AM control-plane drops
        # individual notifs; without resend, the consumer's wait_for_notif
        # times out and the engine asserts on cache miss.
        self._inflight_msgs: dict[str, str] = {}
        # Per-slot acquisition time for the slot-timeout reclaim safety net.
        # If neither NIXL's auto-ACK nor the consumer's explicit app-level
        # ACK arrives within _SLOT_RECLAIM_TIMEOUT_S, the producer assumes
        # the consumer must be done by now and reclaims the slot. Bounds
        # the slot ring against unbounded ACK loss.
        self._inflight_since: dict[str, float] = {}
        # Consumer-side serialization: only one transfer at a time uses the
        # local scratch slot 0 (consumers only need one because READs are
        # serialized within a single start_load_caches call).
        self._consumer_lock = threading.Lock()

        # Bootstrap endpoint (peer's host:port). Producer binds; consumer
        # dials. Format: "host:port".
        peer = extra_cfg.get("peer_endpoint")
        if not peer or ":" not in peer:
            raise ValueError(
                "ec_connector_extra_config.peer_endpoint must be 'host:port' "
                "(producer binds, consumer dials)"
            )
        self._peer_host, peer_port = peer.rsplit(":", 1)
        self._peer_port = int(peer_port)

        # Notification queue (producer + consumer): mm_hash -> list[str messages].
        # Producer tracks ACK|<hash>; consumer tracks PUSH|<hash>|...
        self._notif_q: dict[str, list[str]] = defaultdict(list)
        self._notif_lock = threading.Lock()
        self._stop = threading.Event()

        # Remote agent name + remote scratch addr (populated at bootstrap).
        self._remote_agent: str | None = None
        self._remote_scratch_addr: int | None = None
        self._remote_scratch_size: int | None = None

        if self._is_producer:
            self._start_producer_listener()
        else:
            self._dial_producer_bootstrap()

        # Notification drain thread (both sides).
        self._notif_thread = threading.Thread(
            target=self._drain_notifs, name="bl2-nixl-notif", daemon=True
        )
        self._notif_thread.start()

        # Producer-only: periodic resender for unACK'd PUSH msgs.
        if self._is_producer:
            self._resender_thread = threading.Thread(
                target=self._resend_pushes,
                name="bl2-nixl-resend",
                daemon=True,
            )
            self._resender_thread.start()

    @property
    def remote_agent(self) -> str:
        if self._remote_agent is None:
            raise RuntimeError(
                "Remote NIXL agent not yet bootstrapped (consumer-side only)"
            )
        return self._remote_agent

    # ---- bootstrap ---------------------------------------------------------

    def _start_producer_listener(self) -> None:
        """Producer: listen on (peer_host, peer_port) for a single consumer
        connection and ship agent_metadata + remote_agent_name."""
        meta_bytes = self._wrapper.get_agent_metadata()
        agent_name_bytes = self._engine_id.encode("utf-8")

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to the host portion the user supplied (could be a specific
        # bond IP). Empty host means listen on all interfaces.
        bind_host = "" if self._peer_host in ("0.0.0.0", "") else self._peer_host
        srv.bind((bind_host, self._peer_port))
        srv.listen(4)
        srv.settimeout(1.0)
        self._listener_sock = srv
        logger.info(
            "BL2 NIXL producer listening on %s:%d (engine_id=%s)",
            bind_host or "0.0.0.0", self._peer_port, self._engine_id,
        )

        # Pack our scratch addr/size so the consumer can target it for READ.
        scratch_payload = struct.pack(
            ">QQ", self._scratch_addr, self._scratch_n_bytes
        )

        def accept_loop():
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                try:
                    payload = _LEN_PREFIX.pack(len(agent_name_bytes)) + agent_name_bytes
                    payload += _LEN_PREFIX.pack(len(meta_bytes)) + meta_bytes
                    payload += scratch_payload
                    conn.sendall(payload)
                    # Read the consumer's agent_name bytes so we can address it.
                    n = _LEN_PREFIX.unpack(_recv_n(conn, 4))[0]
                    if n > _MAX_BOOT_BYTES:
                        conn.close()
                        continue
                    consumer_name = _recv_n(conn, n).decode("utf-8")
                    n2 = _LEN_PREFIX.unpack(_recv_n(conn, 4))[0]
                    consumer_meta = _recv_n(conn, n2)
                    cons_scratch = _recv_n(conn, 16)
                    cons_addr, cons_size = struct.unpack(">QQ", cons_scratch)
                    self._remote_scratch_addr = cons_addr
                    self._remote_scratch_size = cons_size
                    # Register the consumer as a remote agent so we can
                    # send_notif to it (notifs are keyed by the recipient).
                    name = self._wrapper.add_remote_agent(consumer_meta)
                    if name != consumer_name:
                        logger.warning(
                            "BL2 NIXL: consumer-declared name %r differs "
                            "from add_remote_agent result %r; using the "
                            "wrapper's value",
                            consumer_name, name,
                        )
                    self._remote_agent = name
                    logger.info(
                        "BL2 NIXL producer accepted consumer at %s, "
                        "remote_agent=%s", addr, name,
                    )
                except Exception:
                    logger.exception("BL2 NIXL producer handshake failed")
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
                # We only need one consumer for this baseline.
                return

        self._accept_thread = threading.Thread(
            target=accept_loop, name="bl2-nixl-accept", daemon=True
        )
        self._accept_thread.start()

    def _dial_producer_bootstrap(self) -> None:
        """Consumer: dial the producer once, exchange agent metadata."""
        deadline = time.time() + 600.0  # match vllm bench serve timeout
        last_err: Exception | None = None
        sock: socket.socket | None = None
        while time.time() < deadline:
            try:
                sock = socket.create_connection(
                    (self._peer_host, self._peer_port), timeout=5.0
                )
                break
            except OSError as e:
                last_err = e
                time.sleep(1.0)
        if sock is None:
            raise RuntimeError(
                f"BL2 NIXL consumer failed to dial {self._peer_host}:{self._peer_port}: {last_err}"
            )
        try:
            n = _LEN_PREFIX.unpack(_recv_n(sock, 4))[0]
            if n > _MAX_BOOT_BYTES:
                raise RuntimeError("BL2 NIXL bootstrap: oversized name")
            producer_name = _recv_n(sock, n).decode("utf-8")
            n2 = _LEN_PREFIX.unpack(_recv_n(sock, 4))[0]
            producer_meta = _recv_n(sock, n2)
            prod_scratch = _recv_n(sock, 16)
            prod_addr, prod_size = struct.unpack(">QQ", prod_scratch)
            self._remote_scratch_addr = prod_addr
            self._remote_scratch_size = prod_size
            self._remote_agent = self._wrapper.add_remote_agent(producer_meta)
            if self._remote_agent != producer_name:
                logger.warning(
                    "BL2 NIXL: producer-declared name %r differs from "
                    "add_remote_agent result %r; using the wrapper's value",
                    producer_name, self._remote_agent,
                )

            # Send our own metadata back so the producer can address us.
            our_name = self._engine_id.encode("utf-8")
            our_meta = self._wrapper.get_agent_metadata()
            sock.sendall(_LEN_PREFIX.pack(len(our_name)) + our_name)
            sock.sendall(_LEN_PREFIX.pack(len(our_meta)) + our_meta)
            sock.sendall(struct.pack(
                ">QQ", self._scratch_addr, self._scratch_n_bytes))
            logger.info(
                "BL2 NIXL consumer bootstrap done; producer=%s remote_scratch=0x%x/%d",
                self._remote_agent, prod_addr, prod_size,
            )
        finally:
            sock.close()

    # ---- notif plumbing ----------------------------------------------------

    def _drain_notifs(self) -> None:
        while not self._stop.is_set():
            try:
                pending = self._wrapper.get_new_notifs()
            except Exception:
                time.sleep(_POLL_INTERVAL_S)
                continue
            if pending:
                ack_hashes: list[str] = []
                req_push_hashes: list[str] = []
                with self._notif_lock:
                    for _agent, msgs in pending.items():
                        for raw in msgs:
                            try:
                                msg = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                            except Exception:
                                continue
                            # Key by mm_hash: ACK|<hash>, PUSH|<hash>|..., or
                            # REQ_PUSH|<hash> (consumer-initiated retry).
                            parts = msg.split("|")
                            if len(parts) >= 2:
                                self._notif_q[parts[1]].append(msg)
                                if parts[0] == "ACK" and self._is_producer:
                                    ack_hashes.append(parts[1])
                                elif (
                                    parts[0] == "REQ_PUSH"
                                    and self._is_producer
                                ):
                                    req_push_hashes.append(parts[1])
                # Release outside the notif lock to avoid lock-order issues
                # with _slots_cv.
                for mh in ack_hashes:
                    self._release_slot(mh)
                # Consumer-initiated targeted retry: re-send PUSH from the
                # cached msg if we still have it (slot not yet ACK'd).
                if req_push_hashes and self._remote_agent is not None:
                    with self._slots_cv:
                        snap = {
                            h: self._inflight_msgs.get(h)
                            for h in req_push_hashes
                        }
                    for h, msg in snap.items():
                        if msg is None:
                            continue
                        try:
                            self._wrapper.send_notif(
                                self._remote_agent, notif_msg=msg
                            )
                            logger.info(
                                "BL2 NIXL producer re-sent PUSH on REQ_PUSH "
                                "for mm_hash=%s", h,
                            )
                        except Exception:
                            pass
            time.sleep(_POLL_INTERVAL_S)

    def wait_for_notif(
        self, mm_hash: str, prefix: str, timeout_s: float = 600.0
    ) -> str:
        """Block until a notif starting with ``prefix`` arrives for mm_hash.
        Returns the full message string.

        Consumer-side: if waiting for a PUSH from a producer for >req_interval
        seconds, send a REQ_PUSH|<hash> notif to ask the producer to
        re-advertise. Defends against NIXL/UCX AM control-plane drops that
        would otherwise stall the engine for the full timeout.
        """
        deadline = time.time() + timeout_s
        # Consumer kicks targeted retries every 30 s if no PUSH has arrived.
        # Producer mode (waiting for ACK) skips this — ACKs are auto-emitted
        # by NIXL on transfer completion, so re-sending the read isn't safe.
        next_retry = time.time() + 30.0 if (
            not self._is_producer and prefix == "PUSH"
        ) else None
        retries = 0
        while time.time() < deadline:
            with self._notif_lock:
                q = self._notif_q.get(mm_hash, [])
                for i, m in enumerate(q):
                    if m.startswith(prefix):
                        del q[i]
                        return m
            now = time.time()
            if next_retry is not None and now >= next_retry:
                if self._remote_agent is not None:
                    try:
                        self._wrapper.send_notif(
                            self._remote_agent,
                            notif_msg=f"REQ_PUSH|{mm_hash}",
                        )
                        retries += 1
                        logger.info(
                            "BL2 NIXL consumer sent REQ_PUSH for "
                            "mm_hash=%s (retry #%d after %.0fs wait)",
                            mm_hash, retries, now - (deadline - timeout_s),
                        )
                    except Exception:
                        pass
                next_retry = now + 30.0
            time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(
            f"BL2 NIXL: timed out waiting for {prefix} notif for {mm_hash}"
        )

    # ---- transfer primitives -----------------------------------------------

    def producer_advertise(self, mm_hash: str, tensor: torch.Tensor) -> int:
        """Producer-side: claim a free scratch slot, copy the tensor into it,
        and send a PUSH notif advertising offset+length+shape+dtype.

        Returns immediately after the PUSH is sent (does NOT wait for ACK).
        The slot remains claimed until the consumer's auto-ACK arrives at
        the producer's notif drain thread, which calls _release_slot to
        return the slot to the free queue.

        Returns the number of bytes copied.
        """
        if not tensor.is_cuda:
            raise RuntimeError("BL2 NIXL: tensor must be on CUDA device")
        t = tensor.contiguous()
        n_bytes = int(t.numel() * t.element_size())
        if n_bytes > self._slot_size:
            raise RuntimeError(
                f"BL2 NIXL: tensor for {mm_hash} ({n_bytes} bytes) exceeds "
                f"slot_size={self._slot_size}; raise slot_bytes in "
                f"ec_connector_extra_config"
            )

        # Wait for a free slot. Producer blocks here when consumer is
        # behind (slots can't free until consumer pulls + ACKs). For
        # multi-image requests under contention, raise n_slots in
        # ec_connector_extra_config to avoid sustained back-pressure.
        with self._slots_cv:
            while not self._free_slots:
                self._slots_cv.wait(timeout=600.0)
                if not self._free_slots:
                    raise RuntimeError(
                        f"BL2 NIXL: no free scratch slot for {mm_hash} "
                        f"after 600s (n_slots={self._n_slots})"
                    )
            slot = self._free_slots.popleft()
            self._inflight_slots[mm_hash] = slot
            self._inflight_since[mm_hash] = time.time()

        try:
            offset = slot * self._slot_size
            flat = t.view(torch.uint8).flatten()
            self._scratch[offset:offset + n_bytes].copy_(flat)
            torch.cuda.synchronize()

            shape = ",".join(str(x) for x in t.shape)
            dtype = str(t.dtype).replace("torch.", "")
            msg = (
                f"PUSH|{mm_hash}|{offset}|{n_bytes}|{shape}|{dtype}"
            )
            self._wrapper.send_notif(self.remote_agent, notif_msg=msg)
            with self._slots_cv:
                self._inflight_msgs[mm_hash] = msg
        except Exception:
            # Slot wasn't successfully published; reclaim it.
            with self._slots_cv:
                self._inflight_slots.pop(mm_hash, None)
                self._inflight_msgs.pop(mm_hash, None)
                self._inflight_since.pop(mm_hash, None)
                self._free_slots.append(slot)
                self._slots_cv.notify()
            raise
        return n_bytes

    def _release_slot(self, mm_hash: str) -> None:
        """Return the slot held by mm_hash back to the free queue.

        Called by the notif drain thread when an ACK|<mm_hash> notif
        arrives, and by the resender thread for slots that have exceeded
        the reclaim timeout.
        """
        with self._slots_cv:
            slot = self._inflight_slots.pop(mm_hash, None)
            self._inflight_msgs.pop(mm_hash, None)
            self._inflight_since.pop(mm_hash, None)
            if slot is None:
                return
            self._free_slots.append(slot)
            self._slots_cv.notify()

    def _resend_pushes(self) -> None:
        """Producer-only: every 30s, re-send PUSH for any unACK'd hash.

        Two layers of defense against NIXL/UCX AM control-plane drops:
        - Producer-side blind periodic resend (this method).
        - Consumer-initiated REQ_PUSH retries (handled in _drain_notifs).
        If the original PUSH made it, duplicates just linger in the
        consumer's _notif_q (small leak, fine for one run). If dropped,
        the next tick re-delivers within 30 s — well inside the consumer's
        600 s wait window.
        """
        tick = 0
        while not self._stop.is_set():
            self._stop.wait(30.0)
            if self._stop.is_set():
                return
            tick += 1
            if self._remote_agent is None:
                continue
            with self._slots_cv:
                pairs = list(self._inflight_msgs.items())
            n_resent = 0
            for _hash, msg in pairs:
                try:
                    self._wrapper.send_notif(
                        self._remote_agent, notif_msg=msg
                    )
                    n_resent += 1
                except Exception:
                    pass
            # Slot timeout reclaim: any slot inflight beyond
            # _SLOT_RECLAIM_TIMEOUT_S is overwhelmingly likely to have lost
            # both NIXL's auto-ACK and the consumer's explicit app-level
            # ACK. The consumer either has the data already (cache populated)
            # or will retry via wait_for_notif on a later re-advertise.
            now = time.time()
            with self._slots_cv:
                stale = [
                    h for h, t in self._inflight_since.items()
                    if now - t > _SLOT_RECLAIM_TIMEOUT_S
                ]
            n_reclaimed = 0
            for h in stale:
                logger.warning(
                    "BL2 NIXL forcing slot reclaim for stale hash=%s "
                    "(no ACK after %.0fs)", h, _SLOT_RECLAIM_TIMEOUT_S,
                )
                self._release_slot(h)
                n_reclaimed += 1
            # One log line per minute (every other tick) for visibility.
            if tick % 2 == 0:
                logger.info(
                    "BL2 NIXL producer resender tick=%d inflight=%d "
                    "resent=%d reclaimed=%d", tick, len(pairs),
                    n_resent, n_reclaimed,
                )

    def consumer_pull(
        self,
        mm_hash: str,
        push_msg: str,
    ) -> torch.Tensor:
        """Consumer-side: parse the PUSH msg, NIXL-READ from the producer's
        scratch into our scratch, copy out to a fresh tensor, return it.

        The READ's notif_msg is set so the producer's save_caches unblocks
        on the implicit ACK that NIXL sends when the READ completes.
        """
        # PUSH|<mm_hash>|<offset>|<n_bytes>|<shape>|<dtype>
        parts = push_msg.split("|")
        if len(parts) < 6 or parts[0] != "PUSH" or parts[1] != mm_hash:
            raise RuntimeError(f"BL2 NIXL: malformed PUSH notif {push_msg!r}")
        remote_offset = int(parts[2])
        n_bytes = int(parts[3])
        shape = [int(x) for x in parts[4].split(",")] if parts[4] else []
        dtype_name = parts[5]
        dtype = _parse_dtype(dtype_name)

        if self._remote_scratch_addr is None:
            raise RuntimeError(
                "BL2 NIXL: remote scratch addr not yet bootstrapped"
            )
        if n_bytes > self._slot_size:
            raise RuntimeError(
                f"BL2 NIXL: incoming tensor for {mm_hash} ({n_bytes} bytes) "
                f"exceeds slot_size={self._slot_size}"
            )

        # Consumer uses slot 0 of its own scratch as the receive buffer for
        # one transfer at a time; the actual remote offset comes from the
        # PUSH notif (producer's slot index).
        with self._consumer_lock:
            local_xfer = self._wrapper.get_xfer_descs(
                [(self._scratch_addr, n_bytes, self._device_id)], "VRAM"
            )
            local_handle = self._wrapper.prep_xfer_dlist(
                _NIXL_INIT_AGENT, local_xfer
            )
            remote_xfer = self._wrapper.get_xfer_descs(
                [(self._remote_scratch_addr + remote_offset, n_bytes,
                  self._device_id)],
                "VRAM",
            )
            remote_handle = self._wrapper.prep_xfer_dlist(
                self.remote_agent, remote_xfer
            )
            try:
                ack_msg = f"ACK|{mm_hash}"
                xfer_handle = self._wrapper.make_prepped_xfer(
                    "READ",
                    local_handle, [0],
                    remote_handle, [0],
                    notif_msg=ack_msg,
                )
                self._wrapper.transfer(xfer_handle)
                deadline = time.time() + 30.0
                state = "PROC"
                while time.time() < deadline:
                    state = self._wrapper.check_xfer_state(xfer_handle)
                    if state in ("DONE", "ERR"):
                        break
                    if isinstance(state, str) and "DONE" in state.upper():
                        state = "DONE"
                        break
                    time.sleep(_POLL_INTERVAL_S)
                else:
                    state = "TIMEOUT"
                self._wrapper.release_xfer_handle(xfer_handle)
                if state != "DONE":
                    raise RuntimeError(
                        f"BL2 NIXL: READ transfer for {mm_hash} ended "
                        f"in state={state}"
                    )
            finally:
                try:
                    self._wrapper.release_dlist_handle(local_handle)
                except Exception:
                    pass
                try:
                    self._wrapper.release_dlist_handle(remote_handle)
                except Exception:
                    pass

            # Copy bytes out of scratch into a fresh tensor that survives the
            # next transfer (which would overwrite scratch).
            tensor = torch.empty(
                shape, dtype=dtype, device=f"cuda:{self._device_id}"
            )
            tensor.view(torch.uint8).flatten()[:n_bytes].copy_(
                self._scratch[:n_bytes]
            )

        # Explicit app-level ACK in addition to the NIXL auto-ACK above.
        # Both go through the same notif AM channel that drops messages
        # under sustained load, but two independent sends roughly halves
        # the probability of *both* being lost. Combined with the
        # producer's _SLOT_RECLAIM_TIMEOUT_S safety net, slot exhaustion
        # is bounded even if the channel is highly lossy.
        try:
            self._wrapper.send_notif(
                self.remote_agent, notif_msg=f"ACK|{mm_hash}"
            )
        except Exception:
            pass
        return tensor

    def stop(self) -> None:
        self._stop.set()
        try:
            sock = getattr(self, "_listener_sock", None)
            if sock is not None:
                sock.close()
        except Exception:
            pass


# ---- helpers ----------------------------------------------------------------


def _recv_n(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("BL2 NIXL bootstrap: peer closed early")
        buf.extend(chunk)
    return bytes(buf)


_DTYPE_TABLE = {
    "float32": torch.float32, "float": torch.float32,
    "float64": torch.float64, "double": torch.float64,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64, "long": torch.int64,
    "int32": torch.int32, "int": torch.int32,
    "int16": torch.int16, "short": torch.int16,
    "int8": torch.int8, "uint8": torch.uint8,
    "bool": torch.bool,
}


def _parse_dtype(name: str) -> torch.dtype:
    name = name.strip()
    if name in _DTYPE_TABLE:
        return _DTYPE_TABLE[name]
    # Some torch.dtype reprs include the namespace.
    for k, v in _DTYPE_TABLE.items():
        if name.endswith(k):
            return v
    raise RuntimeError(f"BL2 NIXL: unsupported dtype name {name!r}")


# ---- the connector ---------------------------------------------------------


class NixlECConnector(ECConnectorBase):
    """Cross-node EC connector backed by NIXL RDMA."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        cfg = vllm_config.ec_transfer_config
        if cfg is None:
            raise ValueError("ec_transfer_config must be set for NixlECConnector")
        extra = dict(cfg.ec_connector_extra_config or {})

        # Device id: prefer extra_cfg override; else first CUDA visible.
        device_id = int(extra.get("device_id", 0))

        # Only the worker-side connector instantiates the NIXL agent and
        # opens the bootstrap TCP listener / dial. The scheduler-side
        # connector lives in a different process; it doesn't move data and
        # would conflict on the bind port if it tried to start the listener.
        self._endpoint: _NixlEndpoint | None = None
        self._bl2 = None
        if role == ECConnectorRole.WORKER:
            self._endpoint = _NixlEndpoint(
                role=role,
                is_producer=self.is_producer,
                extra_cfg=extra,
                device_id=device_id,
            )
            # BL2 sidecar recorder. Only emits when MONO_KERNEL_BL2_VEMB_PATH
            # is set in the worker process; harmless otherwise.
            self._bl2 = _bl2.create_recorder(
                "producer" if self.is_producer else "consumer"
            )

        # Scheduler-side state: mm_hash -> num_token, populated by
        # update_state_after_alloc and drained by build_connector_meta.
        self._pending_loads: dict[str, int] = {}
        self._pending_req_ids: dict[str, list[str]] = {}

        # Cache of mm_hashes that this side has already produced (producer)
        # or already loaded (consumer), so has_cache_item can short-circuit.
        self._known_hashes: set[str] = set()

    # ---- worker-side methods ----------------------------------------------

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        if not self.is_consumer or self._endpoint is None:
            return
        meta = self._get_connector_metadata()
        assert isinstance(meta, NixlECConnectorMetadata)
        for mm in meta.mm_datas:
            if mm.mm_hash in encoder_cache:
                continue
            # Single-clock instrumentation around the two phases the consumer
            # executes serially: wait_for_notif (block until PUSH arrives) and
            # consumer_pull (NIXL READ + scratch copy-out). perf_counter is
            # monotonic and high-resolution; using it on the consumer side
            # alone avoids NTP-dependent cross-node clock subtraction.
            t0 = time.perf_counter()
            try:
                push = self._endpoint.wait_for_notif(mm.mm_hash, "PUSH")
            except Exception:
                logger.exception(
                    "BL2 NIXL wait_for_notif failed for mm_hash=%s",
                    mm.mm_hash,
                )
                continue
            t1 = time.perf_counter()
            try:
                tensor = self._endpoint.consumer_pull(mm.mm_hash, push)
            except Exception:
                logger.exception(
                    "BL2 NIXL consumer_pull failed for mm_hash=%s", mm.mm_hash
                )
                continue
            t2 = time.perf_counter()
            encoder_cache[mm.mm_hash] = tensor
            self._known_hashes.add(mm.mm_hash)
            if self._bl2 is not None:
                self._bl2.record_recv_done(
                    mm.mm_hash,
                    meta.req_ids_by_hash.get(mm.mm_hash, []),
                    int(tensor.numel() * tensor.element_size()),
                    d_wait=t1 - t0,
                    d_pull=t2 - t1,
                )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        if not self.is_producer or self._endpoint is None:
            return
        tensor = encoder_cache.get(mm_hash)
        if tensor is None:
            return
        try:
            # Returns as soon as the PUSH notif is sent. The slot stays
            # claimed in the endpoint's ring until ACK arrives (auto-
            # released by the notif drain thread). save_caches doesn't
            # block, which lets the encoder pipeline keep flowing
            # independently of the consumer's read latency.
            n_bytes = self._endpoint.producer_advertise(mm_hash, tensor)
        except Exception:
            logger.exception(
                "BL2 NIXL producer_advertise failed for mm_hash=%s", mm_hash
            )
            return
        self._known_hashes.add(mm_hash)
        if self._bl2 is not None:
            self._bl2.record_send_done(
                mm_hash, kwargs.get("req_ids", []), n_bytes
            )

    # ---- scheduler-side methods --------------------------------------------

    def has_cache_item(self, identifier: str) -> bool:
        # Consumer: optimistically assume the producer will deliver this hash.
        # The actual blocking wait happens in start_load_caches; if the
        # producer never sends, that wait times out and the load fails.
        return self.is_consumer or identifier in self._known_hashes

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        if not self.is_consumer:
            return
        mm_hash = request.mm_features[index].identifier
        # Skip if we already loaded it this run.
        if mm_hash in self._known_hashes:
            return
        num_token = request.get_num_encoder_embeds(index)
        self._pending_loads[mm_hash] = num_token
        self._pending_req_ids.setdefault(mm_hash, []).append(request.request_id)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECConnectorMetadata:
        meta = NixlECConnectorMetadata()
        for mm_hash, num_token in self._pending_loads.items():
            meta.mm_datas.append(_MMMeta(mm_hash=mm_hash, num_token=num_token))
            meta.req_ids_by_hash[mm_hash] = list(
                self._pending_req_ids.get(mm_hash, [])
            )
        self._pending_loads.clear()
        self._pending_req_ids.clear()
        return meta

    # ---- lifecycle ---------------------------------------------------------

    def __del__(self):
        try:
            ep = getattr(self, "_endpoint", None)
            if ep is not None:
                ep.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_bl2", None) is not None:
                self._bl2.stop()
        except Exception:
            pass
