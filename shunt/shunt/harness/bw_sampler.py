"""Sample RNIC and network-interface throughput at a fixed interval.

RDMA devices are read from ``/sys/class/infiniband/<dev>/ports/<port>/counters``
(``port_xmit_data`` / ``port_rcv_data``, in 4-byte units); network interfaces
from ``/sys/class/net/<if>/statistics`` (``tx_bytes`` / ``rx_bytes``). Every
sample is one JSON line per device with the bytes moved in each direction over
the measured interval and the utilization against the given capacity.

Example (a prefill node's backend bonds and frontend)::

    python -m shunt.harness.bw_sampler --interval-ms 5 --duration 60 \\
        --ib mlx5_bond_0:1@400 --ib mlx5_bond_1:1@400 \\
        --ib mlx5_bond_2:1@400 --ib mlx5_bond_3:1@400 \\
        --ib mlx5_0:1@200 --netdev eth0@200 --out bw.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

IB = Path("/sys/class/infiniband")
NET = Path("/sys/class/net")


def _read(p: Path) -> int:
    try:
        return int(p.read_text())
    except (OSError, ValueError):
        return 0


class Device:
    def __init__(self, kind: str, spec: str):
        name, _, cap = spec.partition("@")
        self.kind = kind
        self.cap_Bps = float(cap or 200) * 1e9 / 8
        if kind == "ib":
            dev, _, port = name.partition(":")
            self.name, self.port = dev, port or "1"
            base = IB / dev / "ports" / self.port / "counters"
            self._tx, self._rx, self._scale = (base / "port_xmit_data",
                                               base / "port_rcv_data", 4)
        else:
            self.name, self.port = name, ""
            base = NET / name / "statistics"
            self._tx, self._rx, self._scale = base / "tx_bytes", base / "rx_bytes", 1

    def read(self) -> tuple[int, int]:
        return _read(self._tx) * self._scale, _read(self._rx) * self._scale


def main() -> None:
    ap = argparse.ArgumentParser(description="RNIC / network interface sampler")
    ap.add_argument("--ib", action="append", default=[],
                    help="RDMA device dev:port@Gbps (repeatable)")
    ap.add_argument("--netdev", action="append", default=[],
                    help="network interface name@Gbps (repeatable)")
    ap.add_argument("--interval-ms", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    devs = [Device("ib", s) for s in a.ib] + [Device("net", s) for s in a.netdev]
    if not devs:
        ap.error("give at least one --ib or --netdev")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dt = a.interval_ms / 1e3
    prev = [d.read() for d in devs]
    t_prev = time.perf_counter()
    t_next = t_prev + dt
    t_end = t_prev + a.duration
    with open(a.out, "w") as f:
        while t_next <= t_end:
            while True:           # sleep most of the interval, spin the rest
                left = t_next - time.perf_counter()
                if left <= 0:
                    break
                if left > 0.002:
                    time.sleep(left - 0.001)
            now = time.perf_counter()
            cur = [d.read() for d in devs]
            span = now - t_prev
            wall = time.time()
            for d, (tx, rx), (ptx, prx) in zip(devs, cur, prev):
                f.write(json.dumps({
                    "t": wall, "dt": span, "dev": d.name, "port": d.port,
                    "kind": d.kind, "tx_bytes": tx - ptx, "rx_bytes": rx - prx,
                    "tx_util": (tx - ptx) / span / d.cap_Bps,
                    "rx_util": (rx - prx) / span / d.cap_Bps,
                }, separators=(",", ":")) + "\n")
            prev, t_prev = cur, now
            t_next += dt


if __name__ == "__main__":
    main()
