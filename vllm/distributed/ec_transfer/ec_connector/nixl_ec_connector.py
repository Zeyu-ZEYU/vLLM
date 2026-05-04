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
from collections import defaultdict
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

        # Remote agent name (consumer-side only, populated at bootstrap).
        self._remote_agent: str | None = None

        if self._is_producer:
            self._start_producer_listener()
        else:
            self._dial_producer_bootstrap()

        # Notification drain thread (both sides).
        self._notif_thread = threading.Thread(
            target=self._drain_notifs, name="bl2-nixl-notif", daemon=True
        )
        self._notif_thread.start()

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
                    conn.sendall(payload)
                    # Read the consumer's agent_name bytes so we can address it.
                    n = _LEN_PREFIX.unpack(_recv_n(conn, 4))[0]
                    if n > _MAX_BOOT_BYTES:
                        conn.close()
                        continue
                    consumer_name = _recv_n(conn, n).decode("utf-8")
                    n2 = _LEN_PREFIX.unpack(_recv_n(conn, 4))[0]
                    consumer_meta = _recv_n(conn, n2)
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
            logger.info(
                "BL2 NIXL consumer bootstrap done; producer=%s",
                self._remote_agent,
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
                with self._notif_lock:
                    for _agent, msgs in pending.items():
                        for raw in msgs:
                            try:
                                msg = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                            except Exception:
                                continue
                            # Key by mm_hash: ACK|<hash> or PUSH|<hash>|...
                            parts = msg.split("|")
                            if len(parts) >= 2:
                                self._notif_q[parts[1]].append(msg)
            time.sleep(_POLL_INTERVAL_S)

    def wait_for_notif(
        self, mm_hash: str, prefix: str, timeout_s: float = 600.0
    ) -> str:
        """Block until a notif starting with ``prefix`` arrives for mm_hash.
        Returns the full message string."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._notif_lock:
                q = self._notif_q.get(mm_hash, [])
                for i, m in enumerate(q):
                    if m.startswith(prefix):
                        del q[i]
                        return m
            time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(
            f"BL2 NIXL: timed out waiting for {prefix} notif for {mm_hash}"
        )

    # ---- transfer primitives -----------------------------------------------

    def producer_advertise(self, mm_hash: str, tensor: torch.Tensor) -> tuple[Any, int, int]:
        """Producer-side: register tensor memory, send PUSH notif.

        Returns (registered_descs, addr, n_bytes) for later deregistration.
        Caller must call ``producer_release(...)`` after the consumer ACKs.
        """
        if not tensor.is_cuda:
            raise RuntimeError("BL2 NIXL: tensor must be on CUDA device")
        t = tensor.contiguous()
        addr = int(t.data_ptr())
        n_bytes = int(t.numel() * t.element_size())
        descs = self._wrapper.get_reg_descs([(addr, n_bytes, self._device_id, "")], "VRAM")
        self._wrapper.register_memory(descs, backends=self._nixl_backends)

        shape = ",".join(str(x) for x in t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        msg = f"PUSH|{mm_hash}|{addr}|{n_bytes}|{shape}|{dtype}"
        self._wrapper.send_notif(self.remote_agent, notif_msg=msg)
        return descs, addr, n_bytes

    def producer_release(self, descs: Any) -> None:
        try:
            self._wrapper.deregister_memory(descs)
        except Exception:
            logger.exception("BL2 NIXL: deregister_memory failed; leaking")

    def consumer_pull(
        self,
        mm_hash: str,
        push_msg: str,
    ) -> torch.Tensor:
        """Consumer-side: parse a PUSH msg, allocate local tensor, NIXL-READ
        from the producer's pointer, return the populated tensor. The READ's
        notif_msg is set so the producer can unblock its `save_caches`."""
        # PUSH|<mm_hash>|<addr>|<n_bytes>|<shape>|<dtype>
        parts = push_msg.split("|")
        if len(parts) < 6 or parts[0] != "PUSH" or parts[1] != mm_hash:
            raise RuntimeError(f"BL2 NIXL: malformed PUSH notif {push_msg!r}")
        remote_addr = int(parts[2])
        n_bytes = int(parts[3])
        shape = [int(x) for x in parts[4].split(",")] if parts[4] else []
        dtype_name = parts[5]
        dtype = _parse_dtype(dtype_name)

        local = torch.empty(shape, dtype=dtype, device=f"cuda:{self._device_id}")
        local_addr = int(local.data_ptr())
        local_n = int(local.numel() * local.element_size())
        if local_n != n_bytes:
            raise RuntimeError(
                f"BL2 NIXL: size mismatch for {mm_hash}: producer={n_bytes} "
                f"consumer={local_n} (shape={shape}, dtype={dtype})"
            )

        local_reg = self._wrapper.get_reg_descs(
            [(local_addr, local_n, self._device_id, "")], "VRAM"
        )
        self._wrapper.register_memory(local_reg, backends=self._nixl_backends)
        local_handle = None
        remote_handle = None
        try:
            local_xfer = self._wrapper.get_xfer_descs(
                [(local_addr, local_n, self._device_id, "")], "VRAM"
            )
            local_handle = self._wrapper.prep_xfer_dlist(_NIXL_INIT_AGENT, local_xfer)
            remote_xfer = self._wrapper.get_xfer_descs(
                [(remote_addr, n_bytes, self._device_id, "")], "VRAM"
            )
            remote_handle = self._wrapper.prep_xfer_dlist(self.remote_agent, remote_xfer)

            ack_msg = f"ACK|{mm_hash}"
            xfer_handle = self._wrapper.make_prepped_xfer(
                "READ",
                local_handle,
                [0],
                remote_handle,
                [0],
                notif_msg=ack_msg,
            )
            self._wrapper.transfer(xfer_handle)

            # Spin until the transfer reports DONE (NIXL state strings vary
            # across versions; treat anything containing "DONE" or matching
            # boolean True as completion).
            deadline = time.time() + 600.0
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
            try:
                self._wrapper.release_dlist_handle(local_handle)
                self._wrapper.release_dlist_handle(remote_handle)
            except Exception:
                pass
            if state != "DONE":
                raise RuntimeError(
                    f"BL2 NIXL: READ transfer for {mm_hash} ended in "
                    f"state={state}"
                )
        finally:
            try:
                self._wrapper.deregister_memory(local_reg)
            except Exception:
                logger.exception("BL2 NIXL: consumer deregister_memory failed")

        return local

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
            try:
                push = self._endpoint.wait_for_notif(mm.mm_hash, "PUSH")
                tensor = self._endpoint.consumer_pull(mm.mm_hash, push)
            except Exception:
                logger.exception(
                    "BL2 NIXL consumer_pull failed for mm_hash=%s", mm.mm_hash
                )
                continue
            encoder_cache[mm.mm_hash] = tensor
            self._known_hashes.add(mm.mm_hash)
            if self._bl2 is not None:
                self._bl2.record_recv_done(
                    mm.mm_hash,
                    meta.req_ids_by_hash.get(mm.mm_hash, []),
                    int(tensor.numel() * tensor.element_size()),
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
            descs, _addr, n_bytes = self._endpoint.producer_advertise(
                mm_hash, tensor
            )
        except Exception:
            logger.exception(
                "BL2 NIXL producer_advertise failed for mm_hash=%s", mm_hash
            )
            return
        try:
            # Block until the consumer has READ our memory.
            self._endpoint.wait_for_notif(mm_hash, "ACK")
        except Exception:
            logger.exception(
                "BL2 NIXL: ACK wait failed for mm_hash=%s", mm_hash
            )
        finally:
            self._endpoint.producer_release(descs)
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
