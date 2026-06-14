"""Sample RNIC bandwidth utilization from the IB port counters (Figs 9, 11).

Reads ``port_xmit_data`` / ``port_rcv_data`` (4-octet units) under
``/sys/class/infiniband/<dev>/ports/<port>/counters`` and reports per-direction
GB/s and utilization against the port's capacity. Used to show the backend is
near-saturated (Fig 9) while the frontend stays under 0.1% (Fig 11).

Example::

    python -m shunt.harness.bw_sampler \
        --device mlx5_bond_0:1 --device mlx5_bond_1:1 ... \
        --interval-ms 5 --duration 45 --capacity-gbps 200 --out backend_bw.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CTR = Path("/sys/class/infiniband")


def _read(dev: str, port: str, name: str) -> int:
    try:
        return int((CTR / dev / "ports" / port / "counters" / name).read_text())
    except OSError:
        return 0


def _xmit_rcv(dev: str, port: str) -> tuple[int, int]:
    # counters are in units of 4 octets
    return (_read(dev, port, "port_xmit_data") * 4,
            _read(dev, port, "port_rcv_data") * 4)


def main() -> None:
    ap = argparse.ArgumentParser(description="RNIC bandwidth sampler")
    ap.add_argument("--device", action="append", required=True,
                    help="dev:port, e.g. mlx5_bond_0:1 (repeatable)")
    ap.add_argument("--interval-ms", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--capacity-gbps", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    devs = [d.split(":") if ":" in d else (d, "1") for d in a.device]
    cap_GBps = a.capacity_gbps * 1e9 / 8 / 1e9   # GB/s capacity
    dt = a.interval_ms / 1000.0
    prev = {tuple(d): _xmit_rcv(*d) for d in devs}
    t_end = time.time() + a.duration

    with open(a.out, "w") as f:
        while time.time() < t_end:
            time.sleep(dt)
            now = time.time()
            for d in devs:
                k = tuple(d)
                tx, rx = _xmit_rcv(*d)
                ptx, prx = prev[k]
                tx_GBps = (tx - ptx) / 1e9 / dt
                rx_GBps = (rx - prx) / 1e9 / dt
                prev[k] = (tx, rx)
                f.write(json.dumps({
                    "t": now, "dev": d[0], "port": d[1],
                    "tx_GBps": tx_GBps, "rx_GBps": rx_GBps,
                    "tx_util": tx_GBps / cap_GBps, "rx_util": rx_GBps / cap_GBps,
                }) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
