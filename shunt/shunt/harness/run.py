"""Run experiments on the cluster and collect their logs.

One run = one system (``shunt.harness.systems``) under one load. The harness
cleans every host, starts the Mooncake master, the decode instances, the proxy,
and the prefill instance, waits until all are healthy while watching their logs
for errors and stalls, replays the trace, and copies every log into the run's
output directory:

- ``requests.jsonl``: one line per request from the trace driver;
- ``proxy.jsonl``: per-request placement and timestamps from the proxy;
- ``shunt/<host>/``: per-rank runtime logs of the prefill engines;
- ``bw/``: RNIC samples (with ``--sample-bw``);
- ``logs/``: the stdout and stderr of every role; ``meta.json``.

Examples::

    # one run
    python -m shunt.harness.run --cluster cluster.yaml --system shunt \\
        --load closed:512 --trace traces/qwen_traceB_blksz_16.jsonl --requests 20000 \\
        --out results/fig13/shunt

    # every run of an experiment file (see shunt/experiments/)
    python -m shunt.harness.run --cluster cluster.yaml --plan shunt/experiments/closed.yaml

    python -m shunt.harness.run --list-systems
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import yaml

from . import roles as R
from . import systems as S
from .cluster import Cluster

ERROR_MARKERS = ("Traceback (most recent call last)", "EngineCore failed",
                 "EngineDeadError", "CUDA error", "NCCL error")


class RunError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Runner:
    def __init__(self, cluster: Cluster, stall_s: float = 180.0,
                 ready_timeout_s: float = 1800.0, drain_s: float = 60.0):
        self.c = cluster
        self.stall_s = stall_s
        self.ready_timeout_s = ready_timeout_s
        self.drain_s = drain_s

    # --- host management --------------------------------------------------------------

    def hosts(self) -> list[str]:
        c = self.c
        hs = c.prefill.hosts + c.decode.hosts + [c.master_host, c.proxy_host,
                                                 c.driver_host]
        return list(dict.fromkeys(hs))

    def clean(self) -> None:
        script = f"{self.c.workdir}/shunt/shunt/harness/clean.sh"
        for h in self.hosts():
            self.c.run(h, f"bash {shlex.quote(script)}", check=False)

    def _start(self, role: R.Role, run_dir: str) -> None:
        log = f"{run_dir}/logs/{role.name}.log"
        pid = f"{run_dir}/logs/{role.name}.pid"
        self.c.start_bg(role.host, f"cd {shlex.quote(self.c.workdir)} && {role.shell()}",
                        log, pid)
        role.log, role.pid = log, pid  # type: ignore[attr-defined]

    def _stop(self, role: R.Role) -> None:
        pid = getattr(role, "pid", None)
        if pid:
            self.c.stop_bg(role.host, pid, grace=5.0)

    def _check(self, role: R.Role, sizes: dict) -> None:
        """Raise on an error in the role's log or when it stopped growing."""
        log = getattr(role, "log")
        r = self.c.run(role.host, f"stat -c %s {shlex.quote(log)} 2>/dev/null; "
                                  f"tail -n 200 {shlex.quote(log)} 2>/dev/null",
                       check=False, capture=True)
        lines = r.stdout.splitlines()
        size = int(lines[0]) if lines and lines[0].isdigit() else 0
        tail = "\n".join(lines[1:])
        for m in ERROR_MARKERS:
            if m in tail:
                raise RunError(f"{role.name} on {role.host}: '{m}' in {log}")
        now = time.time()
        last = sizes.get(role.name)
        if last is None or size != last[0]:
            sizes[role.name] = (size, now)
        elif now - last[1] > self.stall_s:
            raise RunError(f"{role.name} on {role.host}: no log output for "
                           f"{self.stall_s:.0f} s ({log})")

    def _wait_ready(self, roles: list[R.Role]) -> None:
        t0 = time.time()
        sizes: dict = {}
        pending = [r for r in roles if r.health]
        while pending:
            for r in roles:
                self._check(r, sizes)
            pending = [r for r in pending if not self.c.http_ok(r.host, r.health)]
            if not pending:
                return
            if time.time() - t0 > self.ready_timeout_s:
                raise RunError("not ready: " + ", ".join(r.name for r in pending))
            time.sleep(5)

    # --- one run -------------------------------------------------------------------

    def run(self, system: str, load: str, trace: str, out: str, requests: int | None,
            start: int, duration: float, max_output: int | None, sample_bw: list[str],
            bw_seconds: float, timing: int, window: int) -> None:
        c = self.c
        s = S.get(system)
        out_dir = Path(out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = f"{c.rundir}/{out_dir.name}-{int(time.time())}"
        _log(f"run {system} ({load}) -> {out_dir}  [remote dir {run_dir}]")

        oracle_path = None
        if s.placement == "oracle":
            oracle_path = f"{run_dir}/config/ors_ranks.json"
            local = out_dir / "ors_ranks.json"
            cfg_json = out_dir / "shunt_for_oracle.json"
            cfg_json.write_text(json.dumps(R.shunt_config(c, s)))
            argv = [sys.executable, "-m", "shunt.tools.oracle", "--trace", trace,
                    "--config", str(cfg_json), "--start", str(start),
                    "--window", str(window), "--out", str(local)]
            if requests:
                argv += ["--num-requests", str(requests)]
            _run_local(argv)
        plan = R.build(c, s, run_dir, timing=timing, oracle_file=oracle_path)
        if oracle_path:
            plan.files[(c.proxy_host, oracle_path)] = (out_dir / "ors_ranks.json").read_text()
        (out_dir / "config").mkdir(exist_ok=True)
        (out_dir / "config" / "shunt.json").write_text(json.dumps(plan.shunt_config,
                                                                  indent=1))

        self.clean()
        _log(f"waiting {self.drain_s:.0f} s for sockets to drain")
        time.sleep(self.drain_s)
        for (h, path), text in plan.files.items():
            c.write(h, path, text)

        started: list[R.Role] = []
        samplers: list[R.Role] = []
        try:
            for role in [plan.master] + plan.decode + [plan.proxy] + plan.prefill:
                self._start(role, run_dir)
                started.append(role)
                if role is plan.master:
                    time.sleep(3)
            self._wait_ready(started)
            _log("all roles ready")
            passes = ["warm", "measure"] if s.kv_mode == "local" else ["measure"]
            for p in passes:
                if p == "measure" and sample_bw:
                    samplers = self._start_samplers(sample_bw, run_dir, bw_seconds)
                self._drive(s, load, trace, run_dir, requests, start, duration,
                            max_output, started, tag=p)
        finally:
            for role in samplers + started[::-1]:
                self._stop(role)
            self.clean()
            self._collect(run_dir, out_dir)
        meta = {"system": system, "load": load, "trace": trace, "start": start,
                "requests": requests, "duration": duration, "max_output": max_output,
                "timing": timing, "remote_dir": run_dir, "finished": time.time(),
                "system_config": s.__dict__}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        _log(f"done: {out_dir}")

    def _drive(self, s: S.System, load: str, trace: str, run_dir: str,
               requests: int | None, start: int, duration: float,
               max_output: int | None, roles: list[R.Role], tag: str) -> None:
        c = self.c
        kind, _, val = load.partition(":")
        args = ["python", "-m", "shunt.harness.replay", "--trace", trace,
                "--url", f"http://{c.ip(c.proxy_host)}:{c.proxy_port}",
                "--model", c.served_model_name, "--start", str(start),
                "--out", f"{run_dir}/requests.{tag}.jsonl", "--progress", "500"]
        args += ["--concurrency", val] if kind == "closed" else ["--rate", val]
        if requests:
            args += ["--num-requests", str(requests)]
        if duration:
            args += ["--duration", str(duration)]
        if max_output:
            args += ["--max-output", str(max_output)]
        if not s.prefix_cache:
            args.append("--no-reuse")
        driver = R.Role(f"driver.{tag}", c.driver_host,
                        " ".join(shlex.quote(a) for a in args))
        self._start(driver, run_dir)
        sizes: dict = {}
        _log(f"driver ({tag}) started: {load}")
        while True:
            r = c.run(c.driver_host, f"kill -0 $(cat {shlex.quote(driver.pid)}) "
                                     f"2>/dev/null && echo alive", check=False,
                      capture=True)
            if "alive" not in r.stdout:
                break
            for role in roles + [driver]:
                self._check(role, sizes)
            time.sleep(10)
        _log(f"driver ({tag}) finished")

    def _start_samplers(self, hosts: list[str], run_dir: str,
                        seconds: float) -> list[R.Role]:
        c = self.c
        net = c.network
        out = []
        for h in hosts:
            devs = []
            for d in dict.fromkeys(net.get("backend_devices", [])):
                devs += ["--ib", f"{d}:1@{net.get('backend_port_gbps', 400)}"]
            if net.get("frontend_device"):
                devs += ["--ib", f"{net['frontend_device']}:1@{net.get('frontend_gbps', 200)}"]
            if net.get("frontend_netdev"):
                devs += ["--netdev", f"{net['frontend_netdev']}@{net.get('frontend_gbps', 200)}"]
            cmd = " ".join(["python", "-m", "shunt.harness.bw_sampler",
                            "--interval-ms", str(net.get("sample_ms", 5)),
                            "--duration", str(seconds),
                            "--out", f"{run_dir}/bw/{h}.jsonl"] + devs)
            role = R.Role(f"bw.{h}", h, cmd)
            self._start(role, run_dir)
            out.append(role)
        return out

    def _collect(self, run_dir: str, out_dir: Path) -> None:
        c = self.c
        for h in self.hosts():
            c.fetch(h, run_dir, str(out_dir / "remote" / h))
        # flatten the usual files
        for h in self.hosts():
            base = out_dir / "remote" / h
            for name in ("requests.measure.jsonl", "requests.measure.jsonl.meta.json",
                         "proxy.jsonl"):
                p = base / name
                if p.exists():
                    dst = out_dir / name.replace("requests.measure", "requests")
                    dst.write_bytes(p.read_bytes())
            if (base / "shunt").exists():
                for src in (base / "shunt").rglob("*.jsonl"):
                    dst = out_dir / "shunt" / src.relative_to(base / "shunt")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())
            if (base / "bw").exists():
                for src in (base / "bw").glob("*.jsonl"):
                    (out_dir / "bw").mkdir(exist_ok=True)
                    (out_dir / "bw" / src.name).write_bytes(src.read_bytes())
            if (base / "logs").exists():
                for src in (base / "logs").glob("*.log"):
                    (out_dir / "logs").mkdir(exist_ok=True)
                    (out_dir / "logs" / src.name).write_bytes(src.read_bytes())


def _run_local(argv: list[str]) -> None:
    import subprocess

    subprocess.run(argv, check=True)


def _run_plan(runner: Runner, plan_path: str, only: set[str] | None,
              results: str) -> None:
    with open(plan_path) as f:
        spec = yaml.safe_load(f)
    defaults = spec.get("defaults", {})
    for run in spec["runs"]:
        r = dict(defaults) | run
        name = r["name"]
        if only and name not in only:
            continue
        out = Path(results) / spec.get("name", Path(plan_path).stem) / name
        if (out / "meta.json").exists():
            _log(f"skip {name}: already done ({out})")
            continue
        runner.run(r["system"], r["load"], r["trace"], str(out), r.get("requests"),
                   int(r.get("start", 0)), float(r.get("duration", 0)),
                   r.get("max_output"), list(r.get("sample_bw", [])),
                   float(r.get("bw_seconds", 60)), int(r.get("timing", 1)),
                   int(r.get("window", 512)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cluster")
    ap.add_argument("--system")
    ap.add_argument("--load", help="closed:<in flight> or open:<requests/s>")
    ap.add_argument("--trace")
    ap.add_argument("--requests", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--max-output", type=int, default=None)
    ap.add_argument("--sample-bw", default="", help="comma-separated hosts")
    ap.add_argument("--bw-seconds", type=float, default=60.0)
    ap.add_argument("--timing", type=int, default=1, help="SHUNT_TIMING level")
    ap.add_argument("--window", type=int, default=512,
                    help="requests per iteration for the ORS oracle")
    ap.add_argument("--out")
    ap.add_argument("--plan", help="experiment file with a list of runs")
    ap.add_argument("--only", default="", help="comma-separated run names")
    ap.add_argument("--results", default="results")
    ap.add_argument("--drain-s", type=float, default=60.0)
    ap.add_argument("--list-systems", action="store_true")
    a = ap.parse_args()
    if a.list_systems:
        print(S.describe())
        return
    if not a.cluster:
        ap.error("--cluster is required")
    runner = Runner(Cluster.load(a.cluster), drain_s=a.drain_s)
    if a.plan:
        _run_plan(runner, a.plan, set(filter(None, a.only.split(","))) or None,
                  a.results)
        return
    for k in ("system", "load", "trace", "out"):
        if not getattr(a, k):
            ap.error(f"--{k} is required for a single run")
    runner.run(a.system, a.load, a.trace, a.out, a.requests, a.start, a.duration,
               a.max_output, [h for h in a.sample_bw.split(",") if h],
               a.bw_seconds, a.timing, a.window)


if __name__ == "__main__":
    main()
