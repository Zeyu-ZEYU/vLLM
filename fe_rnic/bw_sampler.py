#!/usr/bin/env python3
"""
Backend-network bandwidth / utilization sampler for the fe_rnic measurement.

Samples the RoCE port counters of the BACKEND RNICs on THIS node (the 机尾
fabric: mlx5_bond_0..3, each one 200 Gb/s port) and reports aggregate inbound
(RX) and outbound (TX) throughput + utilization over time.

We measure the WHOLE backend network, not just KV: everything that crosses the
机尾 fabric (MoE all-to-all + inbound prefix-KV + outbound new-KV) is counted.
The front-end RNIC (mlx5_0) is intentionally excluded.

Run ON the node (host side; /sys/class/infiniband is host-level), e.g.:
    python3 bw_sampler.py --label prefill-node0 --interval 0.5 --out /tmp/bw_node0.jsonl

Counter units: port_xmit_data / port_rcv_data are in 4-byte words (IB spec),
so bytes = counter * 4. Per-port capacity 200 Gb/s; node backend capacity per
direction = num_ports * 200 Gb/s (links are full-duplex, so RX and TX each get
the full per-port rate).
"""
import argparse
import json
import os
import signal
import time

DEFAULT_DEVS = "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"
COUNTER_WORD_BYTES = 4  # port_{xmit,rcv}_data are in units of 4 octets


def read_counter(dev, port, name):
    p = f"/sys/class/infiniband/{dev}/ports/{port}/counters/{name}"
    try:
        with open(p) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=DEFAULT_DEVS, help="comma list of backend IB devices")
    ap.add_argument("--port", default="1", help="IB port number (bond devices expose port 1)")
    ap.add_argument("--interval", type=float, default=0.5, help="sample interval (s)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N s (0 = until SIGINT/SIGTERM)")
    ap.add_argument("--port-rate-gbps", type=float, default=200.0, help="per-port line rate")
    ap.add_argument("--label", default=os.uname().nodename)
    ap.add_argument("--out", default="", help="append JSONL samples here (optional)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-sample stdout")
    args = ap.parse_args()

    devs = [d.strip() for d in args.devices.split(",") if d.strip()]
    cap_dir = len(devs) * args.port_rate_gbps  # Gb/s aggregate, one direction

    outf = open(args.out, "a") if args.out else None
    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))

    def snapshot():
        s = {}
        for d in devs:
            s[d] = (read_counter(d, args.port, "port_xmit_data"),
                    read_counter(d, args.port, "port_rcv_data"))
        return s, time.time()

    prev, prev_t = snapshot()
    t0 = prev_t
    if not args.quiet:
        print(f"# label={args.label} devs={devs} cap/dir={cap_dir:.0f} Gb/s interval={args.interval}s")
        print(f"# {'t_s':>7} {'rx_gbps':>9} {'tx_gbps':>9} {'rx_util':>8} {'tx_util':>8} {'tot_util':>8}")

    while not stop["v"]:
        time.sleep(args.interval)
        cur, cur_t = snapshot()
        dt = cur_t - prev_t
        if dt <= 0:
            prev, prev_t = cur, cur_t
            continue
        per, tx_tot, rx_tot = {}, 0.0, 0.0
        for d in devs:
            ptx, prx = prev[d]
            ctx, crx = cur[d]
            if None in (ptx, prx, ctx, crx) or ctx < ptx or crx < prx:
                per[d] = None  # missing counter or reset/wrap -> skip this port this tick
                continue
            dtx = (ctx - ptx) * COUNTER_WORD_BYTES * 8 / dt / 1e9  # Gb/s
            drx = (crx - prx) * COUNTER_WORD_BYTES * 8 / dt / 1e9
            per[d] = {"tx_gbps": round(dtx, 3), "rx_gbps": round(drx, 3)}
            tx_tot += dtx
            rx_tot += drx
        rec = {
            "t": round(cur_t - t0, 3), "ts": round(cur_t, 3), "label": args.label,
            "rx_gbps": round(rx_tot, 3), "tx_gbps": round(tx_tot, 3),
            "rx_util": round(rx_tot / cap_dir, 4) if cap_dir else 0.0,
            "tx_util": round(tx_tot / cap_dir, 4) if cap_dir else 0.0,
            "tot_util": round((rx_tot + tx_tot) / (2 * cap_dir), 4) if cap_dir else 0.0,
            "per_dev": per,
        }
        if outf:
            outf.write(json.dumps(rec) + "\n")
            outf.flush()
        if not args.quiet:
            print(f"{rec['t']:>7.1f} {rx_tot:>9.2f} {tx_tot:>9.2f} "
                  f"{rec['rx_util']:>8.1%} {rec['tx_util']:>8.1%} {rec['tot_util']:>8.1%}")
        prev, prev_t = cur, cur_t
        if args.duration and (cur_t - t0) >= args.duration:
            break

    if outf:
        outf.close()


if __name__ == "__main__":
    main()
