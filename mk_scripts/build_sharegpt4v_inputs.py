#!/usr/bin/env python3
"""Pull a slice of Lin-Chen/ShareGPT4V (instruct, COCO-only rows) into the
project's inputs/ layout.

Output:
    <out-dir>/assets/sharegpt4v/<coco_id>.jpg     # raw images
    <out-dir>/requests/sharegpt4v.jsonl           # one row per request

Each JSONL row matches the BL1 input spec:
    {id, prompt, image, output, output_tokens, sg4v_id}
- id           sequential 0..N-1
- prompt       first human turn from the conversation, with the literal
               "<image>" placeholder stripped (vLLM's chat path injects the
               image as a separate content item)
- image        absolute path to the saved jpg
- output       first gpt turn (kept for reference, optional)
- output_tokens default 256, capped at config; bench-serve uses this as
               max_completion_tokens for the request.
- sg4v_id      original ShareGPT4V row id (12-digit COCO id) for traceability.

Usage:
    HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \\
        python mk_scripts/build_sharegpt4v_inputs.py \\
            --instruct-json /tmp/_sg4v_inst.json \\
            --out-dir ~/zeyu/mono_kernel/inputs \\
            --num 1000 --output-tokens 256 \\
            [--workers 16]

If --instruct-json is missing or unreadable, the script aborts; download it
first via:
    curl -L https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json -o /tmp/_sg4v_inst.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen


_COCO_BASE = "http://images.cocodataset.org/train2017"


def _strip_image_tokens(text: str) -> str:
    """Remove <image> placeholders and collapse surrounding whitespace."""
    out = text.replace("<image>", "")
    return "\n".join(line.strip() for line in out.split("\n")).strip()


def _download_image(coco_id: str, dest: Path) -> tuple[str, bool, str]:
    """Returns (coco_id, ok, error_message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return coco_id, True, ""
    url = f"{_COCO_BASE}/{coco_id}.jpg"
    try:
        req = Request(url, headers={"User-Agent": "mono_kernel/bl1"})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        if not data:
            return coco_id, False, "empty"
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(dest)
        return coco_id, True, ""
    except Exception as e:
        return coco_id, False, repr(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct-json", required=True,
                    help="Path to sharegpt4v_instruct_gpt4-vision_cap100k.json")
    ap.add_argument("--out-dir", required=True,
                    help="Project inputs/ directory (will create assets/sharegpt4v/ "
                         "and requests/)")
    ap.add_argument("--num", type=int, default=1000, help="number of samples")
    ap.add_argument("--output-tokens", type=int, default=256,
                    help="max output tokens to write into each row")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel image downloads")
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.instruct_json)).resolve()
    if not src.exists():
        print(f"ERROR: instruct-json not found at {src}", file=sys.stderr)
        return 2

    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    asset_dir = out_dir / "assets" / "sharegpt4v"
    req_dir = out_dir / "requests"
    asset_dir.mkdir(parents=True, exist_ok=True)
    req_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {src}")
    with open(src) as f:
        all_rows = json.load(f)
    print(f"  {len(all_rows)} rows total")

    # Filter to COCO rows only (to use direct cocodataset.org download).
    coco_rows = [r for r in all_rows if r.get("image", "").startswith("coco/")]
    print(f"  {len(coco_rows)} rows under coco/")

    selected = coco_rows[: args.num]
    print(f"selected {len(selected)} samples")

    # 1) parallel image fetch
    def _coco_id_of(row):
        # image like "coco/train2017/000000000009.jpg"
        return Path(row["image"]).stem

    fetches = [(r, _coco_id_of(r), asset_dir / f"{_coco_id_of(r)}.jpg")
               for r in selected]
    needed = [(cid, dest) for _, cid, dest in fetches
              if not (dest.exists() and dest.stat().st_size > 0)]
    print(f"need to download {len(needed)} images "
          f"({len(fetches) - len(needed)} already cached)")
    fail: list[tuple[str, str]] = []
    if needed:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_download_image, cid, dest) for cid, dest in needed]
            done = 0
            for fut in cf.as_completed(futs):
                cid, ok, err = fut.result()
                done += 1
                if not ok:
                    fail.append((cid, err))
                if done % 100 == 0 or done == len(needed):
                    print(f"  fetched {done}/{len(needed)} "
                          f"(failures so far: {len(fail)})")
    if fail:
        print(f"WARN: {len(fail)} failed image fetches; first 5:")
        for cid, err in fail[:5]:
            print(f"  {cid}: {err}")

    # 2) write jsonl, skipping rows whose image fetch failed
    failed_ids = {cid for cid, _ in fail}
    out_path = req_dir / "sharegpt4v.jsonl"
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        next_id = 0
        for row, coco_id, dest in fetches:
            if coco_id in failed_ids or not dest.exists():
                continue
            convs = row.get("conversations", []) or []
            human = next((c["value"] for c in convs if c.get("from") == "human"), None)
            gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), None)
            if not human:
                continue
            prompt = _strip_image_tokens(str(human))
            if not prompt:
                continue
            obj = {
                "id": next_id,
                "prompt": prompt,
                "image": str(dest),
                "output": gpt or None,
                "output_tokens": args.output_tokens,
                "sg4v_id": coco_id,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            next_id += 1
            written += 1
    print(f"wrote {written} rows to {out_path}")
    print(f"images under {asset_dir}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
