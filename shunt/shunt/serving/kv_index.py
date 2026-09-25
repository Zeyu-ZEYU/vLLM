"""Which prefill DP rank holds the longest cached prefix of a prompt.

Subscribes to the KV-cache events vLLM publishes per DP rank
(``--kv-events-config``, ZMQ). Each ``BlockStored`` event carries the token ids
and the parent block of the stored blocks, so the index keeps, per rank, a map
``(parent block, block tokens) -> block``. A prompt is matched by walking its
full blocks from the root; ``BlockRemoved`` and ``AllBlocksCleared`` keep the
map in sync with evictions.
"""
from __future__ import annotations

import threading

ROOT = None


class RankIndex:
    def __init__(self):
        self.child: dict[tuple, object] = {}
        self.key_of: dict[object, tuple] = {}
        self.block_size = 16

    def stored(self, hashes, parent, tokens, block_size: int) -> None:
        self.block_size = block_size
        for i, h in enumerate(hashes):
            key = (parent, tuple(tokens[i * block_size:(i + 1) * block_size]))
            self.child[key] = h
            self.key_of[h] = key
            parent = h

    def removed(self, hashes) -> None:
        for h in hashes:
            key = self.key_of.pop(h, None)
            if key is not None and self.child.get(key) == h:
                del self.child[key]

    def cleared(self) -> None:
        self.child.clear()
        self.key_of.clear()

    def match_blocks(self, blocks: list[tuple]) -> int:
        parent = ROOT
        n = 0
        for b in blocks:
            h = self.child.get((parent, b))
            if h is None:
                break
            parent = h
            n += 1
        return n


class KVIndex:
    """Longest-prefix lookup over all prefill DP ranks."""

    def __init__(self, num_ranks: int):
        self.ranks = [RankIndex() for _ in range(num_ranks)]
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def apply(self, rank: int, events) -> None:
        from vllm.distributed.kv_events import (
            AllBlocksCleared, BlockRemoved, BlockStored)

        idx = self.ranks[rank]
        with self._lock:
            for ev in events:
                if isinstance(ev, BlockStored):
                    idx.stored(ev.block_hashes, ev.parent_block_hash, ev.token_ids,
                               ev.block_size)
                elif isinstance(ev, BlockRemoved):
                    idx.removed(ev.block_hashes)
                elif isinstance(ev, AllBlocksCleared):
                    idx.cleared()

    def longest_prefix(self, tokens: list[int]) -> tuple[int, int]:
        """``(rank, cached tokens)`` of the best rank; rank -1 if none."""
        best, best_n = -1, 0
        with self._lock:
            bs = self.ranks[0].block_size if self.ranks else 16
            blocks = [tuple(tokens[i:i + bs])
                      for i in range(0, len(tokens) - bs + 1, bs)]
            for r, idx in enumerate(self.ranks):
                n = idx.match_blocks(blocks)
                if n > best_n:
                    best, best_n = r, n
        return best, best_n * bs

    def subscribe(self, endpoints: list[str], topic: str = "") -> None:
        """One subscriber thread per DP rank; ``endpoints[r]`` is rank r's
        publisher address (e.g. ``tcp://prefill-host:5557``)."""
        for r, ep in enumerate(endpoints):
            t = threading.Thread(target=self._run, args=(r, ep, topic), daemon=True)
            t.start()
            self._threads.append(t)

    def _run(self, rank: int, endpoint: str, topic: str) -> None:
        import msgspec
        import zmq
        from vllm.distributed.kv_events import KVEventBatch

        dec = msgspec.msgpack.Decoder(type=KVEventBatch)
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.connect(endpoint)
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        while True:
            parts = sock.recv_multipart()
            batch = dec.decode(parts[-1])
            self.apply(rank, batch.events)
