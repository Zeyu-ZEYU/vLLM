"""Runtime settings read from the environment.

- ``SHUNT_ROLE``: ``prefill`` activates the runtime in this process.
- ``SHUNT_CONFIG``: path of the :class:`shunt.config.ShuntConfig` JSON.
- ``SHUNT_LOG_DIR``: directory for per-rank JSONL logs (unset: no logs).
- ``SHUNT_TIMING``: ``0`` off, ``1`` per-layer phase durations, ``2`` also
  the per-layer start and end offsets of every phase.
- ``SHUNT_PLANNER``: ``native`` (C++ cores, default) or ``python``.
- ``SHUNT_NODE_SIZE``: override the workers per node used for elastic
  attention groups (default: ``deploy.workers_per_node`` of the config).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    role: str
    config_path: str | None
    log_dir: str | None
    timing: int
    planner: str
    node_size: int | None

    @property
    def active(self) -> bool:
        return self.role == "prefill" and bool(self.config_path)


def load() -> Settings:
    node = os.environ.get("SHUNT_NODE_SIZE")
    return Settings(
        role=os.environ.get("SHUNT_ROLE", ""),
        config_path=os.environ.get("SHUNT_CONFIG"),
        log_dir=os.environ.get("SHUNT_LOG_DIR") or None,
        timing=int(os.environ.get("SHUNT_TIMING", "0") or 0),
        planner=os.environ.get("SHUNT_PLANNER", "native"),
        node_size=int(node) if node else None,
    )
