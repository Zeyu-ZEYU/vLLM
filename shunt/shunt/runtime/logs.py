"""Line-buffered JSONL writers, one file per kind and rank."""
from __future__ import annotations

import json
import os
import threading


class JsonlWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            self._f.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._f.close()


class LogSet:
    """Writers keyed by kind (``plan``, ``step``, ``kv``) for one rank."""

    def __init__(self, log_dir: str | None, rank: int):
        self.dir = log_dir
        self.rank = rank
        self._w: dict[str, JsonlWriter] = {}

    def enabled(self) -> bool:
        return self.dir is not None

    def write(self, kind: str, rec: dict) -> None:
        if self.dir is None:
            return
        w = self._w.get(kind)
        if w is None:
            w = self._w[kind] = JsonlWriter(
                os.path.join(self.dir, f"{kind}.rank{self.rank:03d}.jsonl"))
        w.write(rec)
