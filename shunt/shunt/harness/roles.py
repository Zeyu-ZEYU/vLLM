"""Configuration files and launch commands of every role for one run.

Roles: the Mooncake master, the decode instances (one per decode node), the
prefill instance (one DP deployment across the prefill nodes: the first node
runs the API server, the others run headless), the proxy, and optional RNIC
samplers. :func:`build` returns everything the orchestrator needs.
"""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

import yaml

from .cluster import Cluster
from .systems import System


@dataclass
class Role:
    name: str
    host: str
    cmd: str
    env: dict = field(default_factory=dict)
    health: str | None = None          # URL that answers 200 when ready

    def shell(self) -> str:
        exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in self.env.items())
        return f"export {exports}; exec {self.cmd}" if exports else f"exec {self.cmd}"


@dataclass
class Plan:
    files: dict[tuple[str, str], str]    # (host, path) -> content
    master: Role
    decode: list[Role]
    prefill: list[Role]                  # head last: headless ranks start first
    proxy: Role
    shunt_config: dict


def _net(c: Cluster, key: str, default=None):
    return c.network.get(key, default)


def _mooncake_extra(c: Cluster, host: str, group, segment_bytes: int,
                    buffer_bytes: int, devices: str) -> dict:
    return {
        "local_hostname": group.rdma_ip.get(host, c.ip(host)),
        "metadata_server": f"http://{c.ip(c.master_host)}:{c.master_http_port}/metadata",
        "master_server_address": f"{c.ip(c.master_host)}:{c.master_rpc_port}",
        "protocol": _net(c, "protocol", "rdma"),
        "device_name": devices,
        "global_segment_size": segment_bytes,
        "local_buffer_size": buffer_bytes,
        "transfer_timeout": int(c.lmcache.get("transfer_timeout", 10)),
        "save_chunk_meta": False,
    }


def lmcache_decode(c: Cluster, host: str) -> dict:
    devs = list(dict.fromkeys(_net(c, "backend_devices", [])))
    fe = _net(c, "frontend_device")
    # The pool is registered on every NIC, so a prefill client bound to the
    # frontend reaches it over the frontend network (MC_ENABLE_DEST_DEVICE_AFFINITY).
    all_devs = ",".join(devs + ([fe] if fe else []))
    return {
        "chunk_size": int(c.lmcache.get("chunk_size", 256)),
        "local_cpu": False,
        "max_local_cpu_size": float(c.lmcache.get("decode_cpu_gb", 32)),
        "use_layerwise": True,
        "remote_url": f"mooncakestore://{c.ip(c.master_host)}:{c.master_rpc_port}/",
        "remote_serde": "naive",
        "extra_config": _mooncake_extra(
            c, host, c.decode, int(c.lmcache.get("decode_segment_gb", 64)) << 30,
            int(c.lmcache.get("buffer_gb", 4)) << 30, all_devs),
    }


def lmcache_prefill(c: Cluster, host: str, s: System) -> dict:
    if s.kv_mode == "local":
        return {"chunk_size": int(c.lmcache.get("chunk_size", 256)),
                "local_cpu": True,
                "max_local_cpu_size": float(c.lmcache.get("local_mode_cpu_gb", 200)),
                "use_layerwise": True}
    devs = _net(c, "backend_devices", [])
    cfg = {
        "chunk_size": int(c.lmcache.get("chunk_size", 256)),
        "local_cpu": False,
        "max_local_cpu_size": float(c.lmcache.get("prefill_cpu_gb", 32)),
        "use_layerwise": True,
        "remote_url": f"mooncakestore://{c.ip(c.master_host)}:{c.master_rpc_port}/",
        "remote_serde": "naive",
        # the prefill nodes hold no segment: every KV transfer crosses the fabric
        "extra_config": _mooncake_extra(
            c, host, c.prefill, 0, int(c.lmcache.get("buffer_gb", 4)) << 30,
            ",".join(dict.fromkeys(devs))),
    }
    if s.kv_router:
        ex = cfg["extra_config"]
        ex["shunt_kvlb_enabled"] = True
        ex["shunt_kvlb_port_devices"] = list(devs)
        if _net(c, "frontend_device"):
            ex["shunt_kvlb_frontend_device"] = _net(c, "frontend_device")
        ex["shunt_gpu_direct"] = bool(c.shunt.get("gpu_direct", True))
        ex["shunt_gpu_pool_mb"] = int(c.shunt.get("gpu_pool_mb", 2048))
        ex["shunt_host_pool_mb"] = int(c.shunt.get("host_pool_mb", 4096))
        ex["local_buffer_size"] = int(c.shunt.get("client_buffer_mb", 64)) << 20
        if s.kv_priority and _net(c, "kv_tc") is not None:
            ex["shunt_kv_tc_own"] = int(_net(c, "kv_tc"))
            ex["shunt_kv_tc_borrow"] = int(_net(c, "kv_borrow_tc", _net(c, "kv_tc")))
    return cfg


def shunt_config(c: Cluster, s: System) -> dict:
    n_prefill = len(c.prefill.hosts) * c.prefill.gpus_per_node
    return {
        "model": c.shunt.get("model_config", c.model),
        "deploy": {
            "ep_group_workers": n_prefill // s.tp,
            "workers_per_node": c.prefill.gpus_per_node // s.tp,
            "bw_port": float(_net(c, "port_gbps", 200)) * 1e9 / 8,
            "bw_fe": float(_net(c, "frontend_gbps", 200)) * 1e9 / 8,
            "kv_chunk_tokens": int(c.lmcache.get("chunk_size", 256)),
            "pcie_distance": c.shunt.get("pcie_distance"),
        },
        "options": s.options(),
        "compute_profile": c.shunt.get("compute_profile"),
    }


def _common_env(c: Cluster) -> dict:
    env = {"PYTHONHASHSEED": "123", "VLLM_ENABLE_V1_MULTIPROCESSING": "1"}
    if _net(c, "nccl_ib_hca"):
        env["NCCL_IB_HCA"] = _net(c, "nccl_ib_hca")
    if _net(c, "gid_index") is not None:
        env["NCCL_IB_GID_INDEX"] = str(_net(c, "gid_index"))
        env["MC_GID_INDEX"] = str(_net(c, "gid_index"))
    env.update({k: str(v) for k, v in c.env.items()})
    return env


def build(c: Cluster, s: System, run_dir: str, timing: int = 1,
          oracle_file: str | None = None) -> Plan:
    files: dict[tuple[str, str], str] = {}
    cfg_dir = f"{run_dir}/config"
    shunt_cfg = shunt_config(c, s)
    shunt_path = f"{cfg_dir}/shunt.json"
    for h in c.prefill.hosts + [c.proxy_host]:
        files[(h, shunt_path)] = json.dumps(shunt_cfg, indent=1)

    master = Role("master", c.master_host,
                  f"mooncake_master -port {c.master_rpc_port} "
                  f"-enable_http_metadata_server=true "
                  f"-http_metadata_server_host={c.bind_host} "
                  f"-http_metadata_server_port={c.master_http_port}",
                  env=_common_env(c))

    base_args = ["--enforce-eager", "--dtype", "bfloat16",
                 "--served-model-name", c.served_model_name] + list(c.vllm_args)

    # decode: one independent DP deployment per decode node
    decode = []
    for h in c.decode.hosts:
        path = f"{cfg_dir}/lmcache-decode.{h}.yaml"
        files[(h, path)] = yaml.safe_dump(lmcache_decode(c, h))
        extra = {"discard_partial_chunks": False,
                 "shunt_assume_hit": s.kv_mode == "local"}
        kvt = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_consumer",
               "kv_connector_extra_config": extra}
        args = ["vllm", "serve", c.model, "--host", c.bind_host,
                "--port", str(c.decode.port),
                "--data-parallel-size", str(c.decode.gpus_per_node),
                "--enable-expert-parallel", "--tensor-parallel-size", "1",
                "--kv-transfer-config", json.dumps(kvt)] + base_args
        if not s.prefix_cache:
            args.append("--no-enable-prefix-caching")
        env = _common_env(c) | {"LMCACHE_CONFIG_FILE": path,
                                "MC_ENABLE_DEST_DEVICE_AFFINITY": "1"}
        if h in c.decode.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = c.decode.cuda_visible_devices[h]
        decode.append(Role(f"decode.{h}", h, " ".join(shlex.quote(a) for a in args),
                           env, health=f"http://{c.ip(h)}:{c.decode.port}/health"))

    # prefill: one DP deployment over all prefill nodes
    tp = s.tp
    dp = len(c.prefill.hosts) * c.prefill.gpus_per_node // tp
    dpl = c.prefill.gpus_per_node // tp
    head = c.prefill.hosts[0]
    prefill = []
    for i, h in enumerate(c.prefill.hosts):
        path = f"{cfg_dir}/lmcache-prefill.{h}.yaml"
        files[(h, path)] = yaml.safe_dump(lmcache_prefill(c, h, s))
        kvt = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_producer",
               "kv_connector_extra_config": {"discard_partial_chunks": False}}
        args = ["vllm", "serve", c.model,
                "--data-parallel-size", str(dp), "--data-parallel-size-local", str(dpl),
                "--data-parallel-address", c.ip(head),
                "--data-parallel-rpc-port", str(c.prefill.dp_rpc_port),
                "--tensor-parallel-size", str(tp), "--enable-expert-parallel",
                "--kv-transfer-config", json.dumps(kvt)] + base_args
        if s.placement in ("kva", "kva_lb"):
            # each DP rank publishes on port 5557 + rank (vLLM adds the rank)
            args += ["--kv-events-config", json.dumps({
                "enable_kv_cache_events": True, "publisher": "zmq",
                "endpoint": "tcp://*:5557"})]
        if i == 0:
            args += ["--host", c.bind_host, "--port", str(c.prefill.port)]
        else:
            args += ["--headless", "--data-parallel-start-rank", str(i * dpl)]
        if s.a2a == "deepep":
            args += ["--all2all-backend", "deepep_high_throughput"]
        if s.dbo:
            args.append("--enable-dbo")
        if not s.prefix_cache:
            args.append("--no-enable-prefix-caching")
        if s.chunked_prefill_tokens:
            args += ["--max-num-batched-tokens", str(s.chunked_prefill_tokens)]
        env = _common_env(c) | {"LMCACHE_CONFIG_FILE": path,
                                "MC_ENABLE_DEST_DEVICE_AFFINITY": "1"}
        a2a_tc = _net(c, "a2a_tc")
        if a2a_tc is not None:
            env["NCCL_IB_TC"] = str(a2a_tc)
            # KV shares the A2A's class unless the system gives KV lower priority
            env["MC_IB_TC"] = str(_net(c, "kv_tc") if s.kv_priority else a2a_tc)
        # The runtime runs on every prefill rank: in systems without Shunt's
        # mechanisms it only logs requests, plans, and (with timing) phase times.
        env |= {"SHUNT_ROLE": "prefill", "SHUNT_CONFIG": shunt_path,
                "SHUNT_LOG_DIR": f"{run_dir}/shunt/{h}",
                "SHUNT_TIMING": str(timing)}
        if h in c.prefill.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = c.prefill.cuda_visible_devices[h]
        health = f"http://{c.ip(h)}:{c.prefill.port}/health" if i == 0 else None
        prefill.append(Role(f"prefill.{h}", h, " ".join(shlex.quote(a) for a in args),
                            env, health=health))
    prefill = prefill[1:] + prefill[:1]

    # proxy
    ranks_per_host = {h: range(i * dpl, (i + 1) * dpl)
                      for i, h in enumerate(c.prefill.hosts)}
    kv_events = [f"tcp://{c.ip(h)}:{5557 + r}" for h, rr in ranks_per_host.items()
                 for r in rr] if s.placement in ("kva", "kva_lb") else []
    proxy_cfg = {
        "prefill_url": f"http://{c.ip(head)}:{c.prefill.port}",
        "prefill_ranks": dp,
        "decode_urls": [f"http://{c.ip(h)}:{c.decode.port}" for h in c.decode.hosts],
        "placement": s.placement, "shunt_config": shunt_path,
        "tick_ms": float(c.shunt.get("tick_ms", 2.0)),
        "oracle_file": oracle_file, "kv_events": kv_events,
        "lb_factor": float(c.shunt.get("lb_factor", 1.5)),
        "log": f"{run_dir}/proxy.jsonl", "port": c.proxy_port,
    }
    proxy_path = f"{cfg_dir}/proxy.json"
    files[(c.proxy_host, proxy_path)] = json.dumps(proxy_cfg, indent=1)
    proxy = Role("proxy", c.proxy_host,
                 f"python -m shunt.serving.proxy --config {shlex.quote(proxy_path)} "
                 f"--host {c.bind_host}",
                 _common_env(c), health=f"http://{c.ip(c.proxy_host)}:{c.proxy_port}/health")
    return Plan(files=files, master=master, decode=decode, prefill=prefill,
                proxy=proxy, shunt_config=shunt_cfg)
