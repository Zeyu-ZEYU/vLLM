#!/usr/bin/env python3
"""Build a MileBench multi-image workload for BL1.

MileBench (FreedomIntelligence/MileBench on HF) provides multi-image,
multi-turn-style benchmarks. We pick subsets that combine
'Realistic Temporal' + 'Diagnostic Long Text with Image' (per the user's
Motivation experiment goal), cap each request to N images so the vision
encoder cost scales meaningfully without blowing past Qwen3-VL-8B's
single-card budget, and emit a jsonl in the mono_kernel BL1 multi-image
format.

Input layout (after extraction of MileBench_part*.tar.gz):
    <assets-root>/<Subset>/<Subset>.json
    <assets-root>/<Subset>/images/<rel_path>          # raw images
    <assets-root>/<Subset>/combined_1_images/...      # ignored

Output:
    <out-dir>/requests/milebench.jsonl

Each row carries the multi-image schema:
    {"id": <int>, "prompt": "...", "images": ["...", ...],
     "output": "groundtruth", "output_tokens": 32}

`images` paths are written as absolute paths under
``--image-path-prefix`` (defaults to the local extracted dir but can be
overridden so the jsonl carries paths that resolve on a remote node).

Usage:
    python mk_scripts/build_milebench_inputs.py \
        --assets-root ~/zeyu/mono_kernel/inputs/assets/milebench \
        --out-dir ~/zeyu/mono_kernel/inputs \
        --image-path-prefix /home/zeyu/mono_kernel/inputs/assets/milebench \
        --num 1000 --max-imgs-per-req 13 --output-tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Subset selection: per-subset cap. Keys must match top-level dirs under
# assets-root and the JSON file name pattern <Subset>/<Subset>.json.
# The pick favours subsets that combine many images per sample with
# medium-or-large image dimensions (`max_dim >= 1280`).
_DEFAULT_QUOTAS = [
    ("CharacterOrder", 200),         # ~27 imgs/sample, 1280x1920 frames
    ("StateChange", 200),            # ~26 imgs/sample, 1280x1920 frames
    ("ImageNeedleInAHaystack", 300), # ~33 imgs/sample, 1300x1300
    ("TextNeedleInAHaystack", 300),  # ~33 imgs/sample, 800-1800
]


def _resolve_image_path(asset_subset_dir: Path, rel: str) -> Path | None:
    """MileBench json paths are relative to <Subset>/images/. Some odd ones
    are relative to <Subset>/. Try both."""
    p1 = asset_subset_dir / "images" / rel
    if p1.exists():
        return p1
    p2 = asset_subset_dir / rel
    if p2.exists():
        return p2
    return None


def _build_prompt(meta_data: dict, sample: dict) -> str:
    """Compose a clean text prompt from MileBench meta_data + sample.

    The original `context` includes positional `{image#N}` placeholders.
    vLLM's chat path injects images as separate content items after the
    text, so we strip the placeholders to keep the text body readable.
    Choices, when present, are appended as 'A. <opt>' lines.
    """
    instr_id = sample.get("task_instruction_id", 0) or 0
    instructions = meta_data.get("task_instruction") or []
    if isinstance(instructions, list) and 0 <= instr_id < len(instructions):
        instr = instructions[instr_id]
    else:
        instr = ""
    ti = sample["task_instance"]
    ctx = ti.get("context", "") or ""
    # Strip {image#N} placeholders. vLLM's chat handler appends images
    # after the text body anyway.
    import re

    ctx = re.sub(r"\{image#\d+\}", " ", ctx)
    ctx = "\n".join(ln.strip() for ln in ctx.split("\n")).strip()

    parts = []
    if instr:
        parts.append(instr.strip())
    if ctx:
        parts.append(ctx)
    choices = ti.get("choice_list")
    if choices:
        parts.append("Choices:")
        for j, c in enumerate(choices):
            parts.append(f"  {chr(ord('A') + j)}. {c}")
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", required=True,
                    help="Local extracted MileBench root (e.g. "
                         "<inputs>/assets/milebench)")
    ap.add_argument("--out-dir", required=True,
                    help="Project inputs/ dir; writes requests/milebench.jsonl")
    ap.add_argument("--image-path-prefix", default=None,
                    help="Path prefix written into the jsonl `images` field. "
                         "Defaults to --assets-root. Set to the remote-side "
                         "absolute path when staging for a remote node.")
    ap.add_argument("--num", type=int, default=1000,
                    help="Total number of rows to emit.")
    ap.add_argument("--max-imgs-per-req", type=int, default=13,
                    help="Cap on number of images forwarded per request.")
    ap.add_argument("--output-tokens", type=int, default=64,
                    help="output_tokens budget written into each row.")
    args = ap.parse_args()

    assets_root = Path(os.path.expanduser(args.assets_root)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    req_dir = out_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)
    image_prefix = (
        Path(os.path.expanduser(args.image_path_prefix)).resolve()
        if args.image_path_prefix else assets_root
    )

    # Distribute --num across the default quotas proportionally, in case
    # the user overrides --num to something other than the sum (1000).
    quotas = list(_DEFAULT_QUOTAS)
    quota_total = sum(q for _, q in quotas)
    if args.num != quota_total:
        scale = args.num / quota_total
        adjusted = [(s, max(1, int(round(q * scale)))) for s, q in quotas]
        delta = args.num - sum(q for _, q in adjusted)
        # Add/remove from the first subset to balance.
        adjusted[0] = (adjusted[0][0], max(1, adjusted[0][1] + delta))
        quotas = adjusted

    out_path = req_dir / "milebench.jsonl"
    next_id = 0
    skipped_missing = 0
    written_per_subset: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fout:
        for subset, quota in quotas:
            subset_dir = assets_root / subset
            json_path = subset_dir / f"{subset}.json"
            if not json_path.exists():
                # Some subsets are stored with hyphens vs underscores.
                alt = list(subset_dir.glob("*.json"))
                alt = [p for p in alt if "adv" not in p.name]
                if not alt:
                    print(f"WARN: {subset} has no source json under {subset_dir}",
                          file=sys.stderr)
                    continue
                json_path = alt[0]
            with open(json_path) as f:
                d = json.load(f)
            samples = d["data"] if isinstance(d, dict) and "data" in d else d
            meta_data = d.get("meta_data", {}) if isinstance(d, dict) else {}
            written = 0
            for s in samples:
                if written >= quota:
                    break
                ti = s["task_instance"]
                rel_paths = ti.get("images_path") or []
                if not rel_paths:
                    skipped_missing += 1
                    continue
                rel_paths = rel_paths[: args.max_imgs_per_req]
                # Verify all chosen images exist locally; skip the row if
                # any one fails to resolve.
                resolved_local: list[Path] = []
                ok = True
                for rel in rel_paths:
                    p = _resolve_image_path(subset_dir, rel)
                    if p is None:
                        ok = False
                        break
                    resolved_local.append(p)
                if not ok:
                    skipped_missing += 1
                    continue
                # Map each local path to the configured image_prefix.
                # rel_to_assets is the path of the file relative to the
                # assets-root (so it works with arbitrary prefixes).
                images_out: list[str] = []
                for p_local in resolved_local:
                    rel_to_root = p_local.resolve().relative_to(assets_root)
                    images_out.append(str(image_prefix / rel_to_root))

                prompt = _build_prompt(meta_data, s)
                obj = {
                    "id": next_id,
                    "prompt": prompt,
                    "images": images_out,
                    "output": s.get("response"),
                    "output_tokens": args.output_tokens,
                    "subset": subset,
                    "milebench_sample_id": s.get("sample_id"),
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                next_id += 1
                written += 1
            written_per_subset[subset] = written
            print(f"  {subset:30s}  wrote {written}/{quota}")

    print(f"total written: {next_id} rows -> {out_path}")
    if skipped_missing:
        print(f"  skipped {skipped_missing} samples for missing image files")
    print("per-subset:", written_per_subset)
    return 0 if next_id > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
