"""Cluster description and remote execution for the experiment harness.

The cluster file (YAML) names the hosts of every role, the addresses and RDMA
devices, and how to run commands on a host. See
``shunt/configs/cluster.example.yaml``.
Commands run through ``ssh`` (or locally when ``ssh: local``) in a non-login
``bash`` after sourcing ``activate`` (the Python environment), optionally
wrapped by ``exec_prefix`` (for example a ``docker exec`` into the serving
container); ``{cmd}`` in the prefix is replaced by the quoted command.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class NodeGroup:
    hosts: list[str]
    gpus_per_node: int = 8
    port: int = 8000
    mgmt_ip: dict[str, str] = field(default_factory=dict)
    rdma_ip: dict[str, str] = field(default_factory=dict)
    cuda_visible_devices: dict[str, str] = field(default_factory=dict)
    dp_rpc_port: int = 13345

    def ip(self, host: str) -> str:
        return self.mgmt_ip.get(host, host)


@dataclass
class Cluster:
    raw: dict
    model: str
    served_model_name: str
    workdir: str
    rundir: str
    activate: str
    ssh: str
    exec_prefix: str
    prefill: NodeGroup
    decode: NodeGroup
    master_host: str
    master_rpc_port: int
    master_http_port: int
    proxy_host: str
    proxy_port: int
    driver_host: str
    network: dict
    env: dict
    vllm_args: list[str]
    lmcache: dict
    shunt: dict
    bind_host: str = "0.0.0.0"

    @classmethod
    def load(cls, path: str) -> "Cluster":
        with open(path) as f:
            c = yaml.safe_load(f)

        def group(d: dict) -> NodeGroup:
            return NodeGroup(hosts=list(d["hosts"]),
                             gpus_per_node=int(d.get("gpus_per_node", 8)),
                             port=int(d.get("port", 8000)),
                             mgmt_ip=dict(d.get("mgmt_ip") or {}),
                             rdma_ip=dict(d.get("rdma_ip") or {}),
                             cuda_visible_devices=dict(
                                 d.get("cuda_visible_devices") or {}),
                             dp_rpc_port=int(d.get("dp_rpc_port", 13345)))

        m = c.get("master", {})
        p = c.get("proxy", {})
        return cls(
            raw=c, model=c["model"],
            served_model_name=c.get("served_model_name", "model"),
            workdir=c["workdir"], rundir=c["rundir"],
            activate=c.get("activate", ""), ssh=c.get("ssh", "ssh -o BatchMode=yes"),
            exec_prefix=c.get("exec_prefix", ""),
            prefill=group(c["prefill"]), decode=group(c["decode"]),
            master_host=m.get("host", c["prefill"]["hosts"][0]),
            master_rpc_port=int(m.get("rpc_port", 50051)),
            master_http_port=int(m.get("http_port", 8080)),
            proxy_host=p.get("host", c["prefill"]["hosts"][0]),
            proxy_port=int(p.get("port", 9000)),
            driver_host=c.get("driver", {}).get("host", c["prefill"]["hosts"][0]),
            network=dict(c.get("network") or {}), env=dict(c.get("env") or {}),
            vllm_args=list(c.get("vllm_args") or []),
            lmcache=dict(c.get("lmcache") or {}), shunt=dict(c.get("shunt") or {}),
            bind_host=c.get("bind_host", "0.0.0.0"),
        )

    # --- addresses ------------------------------------------------------------

    def ip(self, host: str) -> str:
        for g in (self.prefill, self.decode):
            if host in g.mgmt_ip:
                return g.mgmt_ip[host]
        return self.raw.get("mgmt_ip", {}).get(host, host)

    @property
    def is_local(self) -> bool:
        return self.ssh == "local"

    # --- execution --------------------------------------------------------------

    def _wrap(self, cmd: str) -> str:
        inner = f"source {self.activate} && {cmd}" if self.activate else cmd
        if self.exec_prefix:
            return self.exec_prefix.replace("{cmd}", shlex.quote(inner))
        return f"bash -c {shlex.quote(inner)}"

    def _argv(self, host: str, cmd: str) -> list[str]:
        wrapped = self._wrap(cmd)
        if self.is_local:
            return ["bash", "-c", wrapped]
        return shlex.split(self.ssh) + [host, wrapped]

    def run(self, host: str, cmd: str, check: bool = True, timeout: float | None = None,
            capture: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(self._argv(host, cmd), check=check, timeout=timeout,
                              capture_output=capture, text=True)

    def start_bg(self, host: str, cmd: str, log: str, pidfile: str) -> None:
        """Start ``cmd`` in its own process group; record its group id."""
        bg = (f"mkdir -p {shlex.quote(str(Path(log).parent))} && "
              f"(setsid bash -c {shlex.quote(cmd)} > {shlex.quote(log)} 2>&1 "
              f"< /dev/null & echo $! > {shlex.quote(pidfile)})")
        self.run(host, bg)

    def stop_bg(self, host: str, pidfile: str, grace: float = 10.0) -> None:
        q = shlex.quote(pidfile)
        self.run(host, f"[ -f {q} ] && kill -TERM -- -$(cat {q}) 2>/dev/null; true",
                 check=False)
        time.sleep(grace)
        self.run(host, f"[ -f {q} ] && kill -KILL -- -$(cat {q}) 2>/dev/null; "
                       f"rm -f {q}; true", check=False)

    def write(self, host: str, path: str, text: str) -> None:
        """Write a file on ``host`` (outside any container; bind-mount it)."""
        if self.is_local:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
            return
        d = shlex.quote(str(Path(path).parent))
        argv = shlex.split(self.ssh) + [host, f"mkdir -p {d} && cat > {shlex.quote(path)}"]
        subprocess.run(argv, input=text, text=True, check=True)

    def fetch(self, host: str, remote_dir: str, local_dir: str) -> None:
        """Copy a directory from ``host`` into ``local_dir``."""
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        if self.is_local:
            subprocess.run(["bash", "-c", f"cp -r {shlex.quote(remote_dir)}/. "
                                          f"{shlex.quote(local_dir)}/ 2>/dev/null; true"])
            return
        argv = shlex.split(self.ssh) + [host, f"tar -C {shlex.quote(remote_dir)} -cf - ."]
        with subprocess.Popen(argv, stdout=subprocess.PIPE) as src:
            subprocess.run(["tar", "-C", local_dir, "-xf", "-"], stdin=src.stdout,
                           check=False)

    def http_ok(self, host: str, url: str) -> bool:
        r = self.run(host, f"curl -s -m 3 -o /dev/null -w '%{{http_code}}' {url}",
                     check=False, capture=True)
        return r.stdout.strip() == "200"
