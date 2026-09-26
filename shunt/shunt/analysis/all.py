"""Produce every figure and table from a results directory.

Expects the layout the experiment files create
(``<results>/<experiment>/<run>``, see ``shunt/experiments``) and the
microbenchmark outputs under ``<bench>``. Items whose inputs are missing are
skipped with a message.

Example::

    python -m shunt.analysis.all --results results --bench bench --traces traces \\
        --prefill-host p0 --decode-host d0 \\
        --backend-devices mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \\
        --frontend-device eth0 --out out
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--bench", default="bench")
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--prefill-host", default=None)
    ap.add_argument("--decode-host", default=None)
    ap.add_argument("--backend-devices", default=None)
    ap.add_argument("--frontend-device", default=None)
    ap.add_argument("--warmup-s", type=float, default=0.0)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    R, B, T = Path(a.results), Path(a.bench), Path(a.traces)
    F, TB = Path(a.out) / "figures", Path(a.out) / "tables"

    def have(*paths) -> bool:
        return all(Path(p).exists() for p in paths)

    def run(name: str, module: str, *args, need=()):
        missing = [str(p) for p in need if not Path(p).exists()]
        if missing:
            print(f"-- skip {name}: missing {', '.join(missing)}")
            return
        print(f"== {name}")
        subprocess.run([sys.executable, "-m", f"shunt.analysis.{module}", *map(str, args)],
                       check=False)

    mot, cl, sw, a2a, wl, acc = (R / x for x in ("motivation", "closed", "sweep", "a2a",
                                                 "workloads", "accuracy"))
    w = ["--warmup-s", str(a.warmup_s)]

    # Sec. 2
    base_t = mot / "baseline-timing"
    run("Fig. 5", "imbalance", "plot", "--run", base_t, "--kind", "compute",
        "--out", F / "fig5_straggler_imbalance", need=[base_t])
    run("Fig. 6", "imbalance", "plot", "--run", base_t, "--kind", "kv",
        "--out", F / "fig6_kv_imbalance", need=[base_t])
    run("Fig. 7", "imbalance", "plot", "--run", mot / "ors-timing", "--kind", "all",
        "--out", F / "fig7_oracle_imbalance", need=[mot / "ors-timing"])
    if a.prefill_host and a.decode_host:
        bw = base_t / "bw"
        p, d = bw / f"{a.prefill_host}.jsonl", bw / f"{a.decode_host}.jsonl"
        if a.backend_devices:
            run("Fig. 8", "bandwidth", "--prefill", p, "--decode", d, "--devices",
                a.backend_devices, "--window", "45", "--out", F / "fig8_backend_bw",
                need=[p, d])
        if a.frontend_device:
            run("Fig. S2", "bandwidth", "--prefill", p, "--decode", d, "--devices",
                a.frontend_device, "--window", "50", "--percent-max", "0.1",
                "--out", F / "figS2_frontend_bw", need=[p, d])
    run("Fig. 9", "ttft", "box", *w, f"Baseline={mot / 'baseline'}",
        f"No contention={mot / 'no-contention'}", "--out", F / "fig9_contention",
        need=[mot / "baseline", mot / "no-contention"])
    run("Sec. 2.3 prompt tokens", "trace_stats", "split", f"Baseline={base_t}",
        "--out", TB / "sec2_prompt_tokens", need=[base_t])
    run("Sec. 3.2 chunked prefill", "imbalance", "table", f"no chunking={base_t}",
        f"chunked={mot / 'baseline-chunked'}", "--out", TB / "sec3_chunked_prefill",
        need=[base_t, mot / "baseline-chunked"])

    # Sec. 4.2
    main5 = [("Baseline", "baseline"), ("ORS", "ors"), ("Combo", "combo"),
             ("Shunt", "shunt"), ("Sh-ORS", "sh-ors")]
    run("Fig. 13a", "ttft", "box", *w, *[f"{l}={cl / n}" for l, n in main5],
        "--out", F / "fig13a_ttft", need=[cl / n for _, n in main5])
    series = [f"{l}={sw / n}-r*" for l, n in main5] + \
        [f"Baseline DBO={sw / 'baseline+dbo'}-r*", f"Shunt DBO={sw / 'shunt+dbo'}-r*"]
    run("Fig. 13b", "sweep", *sum((["--series", s] for s in series), []),
        "--dashed", "Baseline DBO,Shunt DBO", "--out", F / "fig13b_load",
        need=[sw])
    stacks = [("NCCL Baseline", cl / "baseline"), ("NCCL Shunt", cl / "shunt"),
              ("DeepEP Baseline", a2a / "baseline+deepep"),
              ("DeepEP Shunt", a2a / "shunt+deepep"),
              ("DeepEP+DBO Baseline", a2a / "baseline+dbo"),
              ("DeepEP+DBO Shunt", a2a / "shunt+dbo")]
    run("Table 1", "ttft", "table", *w, *[f"{l}={p}" for l, p in stacks],
        "--out", TB / "tab1_a2a_stacks", need=[p for _, p in stacks])
    run("Sec. 4.2 contention under DBO", "ttft", "table", *w,
        f"Baseline DBO={a2a / 'baseline+dbo'}",
        f"No contention DBO={a2a / 'no-contention+dbo'}",
        "--out", TB / "sec4_dbo_contention",
        need=[a2a / "baseline+dbo", a2a / "no-contention+dbo"])
    run("Sec. 4.2 idle and A2A occupancy", "barrier", f"NCCL Baseline={base_t}",
        f"DBO Baseline={a2a / 'baseline+dbo-timing'}",
        f"DBO Shunt={a2a / 'shunt+dbo-timing'}", "--out", TB / "sec4_barrier",
        need=[base_t, a2a / "baseline+dbo-timing", a2a / "shunt+dbo-timing"])
    coder = [f"Baseline={wl}/coder-baseline-r*", f"Shunt={wl}/coder-shunt-r*"]
    run("Table 2a", "sweep", *sum((["--series", s] for s in coder), []),
        "--out", TB / "tab2a_coder", need=[wl])
    run("Table 2b", "ttft", "table", *w, f"Baseline={wl / 'nocache-baseline'}",
        f"Shunt={wl / 'nocache-shunt'}", "--out", TB / "tab2b_no_reuse",
        need=[wl / "nocache-baseline", wl / "nocache-shunt"])

    # Sec. 4.3
    abl = [("Shunt", "shunt"), ("/RS", "no-rs"), ("/EAP", "no-eap"), ("/KVLB", "no-kvlb")]
    run("Fig. 14a", "ttft", "box", *w, *[f"{l}={cl / n}" for l, n in abl],
        "--out", F / "fig14a_ablation", need=[cl / n for _, n in abl])
    run("Fig. 14b", "sweep", *sum((["--series", f"{l}={sw / n}-r*"] for l, n in abl), []),
        "--out", F / "fig14b_ablation_load", need=[sw])
    kv = [("Shunt", "shunt"), ("/prio", "no-prio"), ("/budget", "no-budget"),
          ("/borrow", "no-borrow"), ("/frontend", "no-frontend"), ("/KVLB", "no-kvlb")]
    run("Fig. 15a", "ttft", "box", *w, *[f"{l}={cl / n}" for l, n in kv],
        "--out", F / "fig15a_kvlb", need=[cl / n for _, n in kv])
    run("Fig. 15b", "sweep", *sum((["--series", f"{l}={sw / n}-r*"] for l, n in
                                   [("Shunt", "shunt"), ("/budget", "no-budget"),
                                    ("/KVLB", "no-kvlb")]), []),
        "--out", F / "fig15b_kvlb_load", need=[sw])
    run("Table S4", "ttft", "table", *w, *[f"{l}={cl / n}" for l, n in kv],
        f"Combo={cl / 'combo'}", f"Shunt DBO={a2a / 'shunt+dbo'}",
        f"/frontend DBO={a2a / 'no-frontend+dbo'}", "--out", TB / "tabS4_kvlb_arms",
        need=[cl / n for _, n in kv])
    run("Sec. 4.3 and S5 overflow", "overflow", f"NCCL={acc / 'shunt-timing'}",
        f"DBO={acc / 'shunt+dbo-timing'}", "--out", TB / "sec4_overflow",
        need=[acc / "shunt-timing", acc / "shunt+dbo-timing"])
    disp = [("Baseline", "baseline"), ("LPT alone", "lpt"), ("KVA alone", "kva"),
            ("KVA-LB alone", "kva-lb"), ("Shunt (LPT)", "shunt"),
            ("Shunt (KVA)", "shunt-kva"), ("Shunt (KVA-LB)", "shunt-kva-lb")]
    run("Table S5", "dispatch", *[f"{l}={cl / n}" for l, n in disp],
        "--out", TB / "tabS5_dispatch", need=[cl / n for _, n in disp])
    s3 = [("Shunt", "shunt"), ("/budget", "no-budget"), ("/KVLB", "no-kvlb"),
          ("Combo", "combo"), ("Shunt (KVA)", "shunt-kva")]
    run("Table S3", "sweep", *sum((["--series", f"{l}={sw / n}-r*"] for l, n in s3), []),
        "--out", TB / "tabS3_sweep", need=[sw])
    s2 = [("NCCL Baseline", "baseline"), ("NCCL Shunt", "shunt"),
          ("DeepEP Baseline", "baseline+deepep"), ("DeepEP Shunt", "shunt+deepep"),
          ("DBO Baseline", "baseline+dbo"), ("DBO Shunt", "shunt+dbo")]
    run("Table S2", "sweep", *sum((["--series", f"{l}={sw / n}-r*"] for l, n in s2), []),
        "--out", TB / "tabS2_sweep_a2a", need=[sw])

    # Sec. 4.4 and 4.5
    run("Fig. S3", "bench", "overhead", B / "eap_overhead.json",
        "--out", F / "figS3_eap_overhead", need=[B / "eap_overhead.json"])
    run("Table S6", "bench", "ring", B / "ring.json", B / "eap_overhead.json",
        "--out", TB / "tabS6_ring", need=[B / "ring.json", B / "eap_overhead.json"])
    run("Fig. S4", "bench", "scalability", B / "scalability.csv",
        "--out", F / "figS4_scalability", need=[B / "scalability.csv"])
    theta = [("1.0", "shunt-theta1.0"), ("1.25", "shunt-theta1.25"), ("1.5", "shunt"),
             ("2.0", "shunt-theta2.0"), ("2.5", "shunt-theta2.5"), ("3.0", "shunt-theta3.0")]
    run("Fig. S5", "ttft", "box", *w, *[f"theta {l}={cl / n}" for l, n in theta],
        "--out", F / "figS5_theta", need=[cl / n for _, n in theta])
    run("Table 3", "accuracy", f"NCCL={acc / 'shunt-timing'}",
        f"DeepEP+DBO={acc / 'shunt+dbo-timing'}:dbo", "--out", TB / "tab3_accuracy",
        need=[acc / "shunt-timing", acc / "shunt+dbo-timing"])

    # Supplement: traces and TP
    traces = [("business", T / "qwen_traceB_blksz_16.jsonl"),
              ("consumer", T / "qwen_traceA_blksz_16.jsonl"),
              ("reasoning", T / "qwen_thinking_blksz_16.jsonl"),
              ("coding", T / "qwen_coder_blksz_16.jsonl")]
    have_traces = [(l, p) for l, p in traces if p.exists()]
    if have_traces:
        run("Table S1 (traces)", "trace_stats", "stats",
            *[f"{l}={p}" for l, p in have_traces], "--out", TB / "tabS1_traces")
    run("Table S1 (imbalance)", "imbalance", "table", f"business={base_t}",
        f"consumer={mot / 'baseline-traceA'}", f"reasoning={mot / 'baseline-thinking'}",
        f"coding={mot / 'baseline-coder'}", "--out", TB / "tabS1_imbalance",
        need=[base_t, mot / "baseline-traceA", mot / "baseline-thinking",
              mot / "baseline-coder"])
    run("Fig. S1", "trace_stats", "lengths", traces[0][1], "--out",
        F / "figS1_trace_lengths", need=[traces[0][1]])
    run("Table S7", "imbalance", "table", f"TP=1={base_t}",
        f"TP=2={mot / 'baseline-tp2'}", f"TP=4={mot / 'baseline-tp4'}",
        f"TP=8={mot / 'baseline-tp8'}", "--out", TB / "tabS7_tp",
        need=[base_t, mot / "baseline-tp2", mot / "baseline-tp4", mot / "baseline-tp8"])


if __name__ == "__main__":
    main()
