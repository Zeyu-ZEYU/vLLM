#!/usr/bin/env python3
"""Minimal 16-rank (8/node x2) distributed sanity test: all_reduce correctness
+ cross-node all_to_all timing over the back-end RDMA. Launched via torchrun."""
import os, socket, time
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    dev = torch.device("cuda", local)
    print(f"rank {rank}/{world} local {local} host {socket.gethostname()}", flush=True)

    # correctness: all_reduce of per-rank constant -> sum(0..world-1)
    x = torch.full((4,), float(rank), device=dev)
    dist.all_reduce(x)
    expect = world * (world - 1) // 2

    # cross-node all_to_all timing (128 MB bf16)
    n = 64 * 1024 * 1024
    send = torch.randn(n, device=dev, dtype=torch.bfloat16)
    recv = torch.empty_like(send)
    for _ in range(3):
        dist.all_to_all_single(recv, send)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(20):
        dist.all_to_all_single(recv, send)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    dist.barrier()

    if rank == 0:
        ok = "OK" if abs(x[0].item() - expect) < 1e-3 else "FAIL"
        ms = (t1 - t0) / 20 * 1000
        print(f"[RESULT] world={world} allreduce={ok}({x[0].item():.0f}/{expect}) "
              f"| all_to_all 128MB: {ms:.3f} ms/iter", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
