#!/usr/bin/env python3
"""Print the mlx5_bond_* InfiniBand device whose PCIe placement is closest
to GPU 0. Used by run_bl2.sh to pick the right `nixl_dev` per node so NIXL
RDMA traffic goes through the GPU's 机尾 bond, not through `mlx5_0` (the
机头 management RNIC) or a 不同-NUMA bond.

Algorithm:
  - Read GPU 0's PCIe bus ID via `nvidia-smi`.
  - For each device under `/sys/class/infiniband/` whose name starts with
    `mlx5_bond_` (skipping `mlx5_0` which is the management RNIC), follow
    its `device` symlink to obtain the PCIe domain:bus.
  - Pick the bond with smallest PCIe distance to GPU 0:
      * same domain wins outright over different domain
      * within same domain, smaller |bond_bus - gpu_bus| wins
  - Print the chosen bond name on stdout (or nothing if none found).

Used both locally (node 0) and remotely (node 1 via ssh) by run_bl2.sh.
"""

from __future__ import annotations

import os
import subprocess
import sys


def parse_pci(s: str) -> tuple[str | None, int | None]:
    """Parse e.g. '0000:7f:00.0' or '00000000:08:00.0' → (domain_str, bus_int).
    Returns (None, None) on parse failure."""
    parts = s.strip().split(":")
    if len(parts) < 2:
        return None, None
    domain = parts[0].lstrip("0") or "0"
    try:
        bus = int(parts[1], 16)
    except ValueError:
        return None, None
    return domain, bus


def main() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=pci.bus_id",
             "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as e:
        print(f"detect_closest_bond: nvidia-smi failed: {e}", file=sys.stderr)
        return 1
    gpu_dom, gpu_bus = parse_pci(out)
    if gpu_bus is None:
        print(f"detect_closest_bond: cannot parse GPU 0 PCI '{out}'",
              file=sys.stderr)
        return 1

    ib_root = "/sys/class/infiniband"
    if not os.path.isdir(ib_root):
        print(f"detect_closest_bond: no {ib_root}", file=sys.stderr)
        return 1

    best: tuple[str, int] | None = None
    for dev in sorted(os.listdir(ib_root)):
        # Only consider mlx5_bond_* (机尾 inference bonds). mlx5_0 is the
        # 机头 management RNIC and must NOT be selected here.
        if not dev.startswith("mlx5_bond_"):
            continue
        link_path = os.path.join(ib_root, dev, "device")
        try:
            link = os.readlink(link_path)
        except OSError:
            continue
        dom, bus = parse_pci(os.path.basename(link))
        if bus is None:
            continue
        # Same PCIe domain wins; within same domain, smaller bus distance wins.
        if dom != gpu_dom:
            dist = 100000  # different domain → far
        else:
            dist = abs(bus - gpu_bus)
        if best is None or dist < best[1]:
            best = (dev, dist)

    if best is None:
        print("detect_closest_bond: no mlx5_bond_* device found",
              file=sys.stderr)
        return 1
    print(best[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
