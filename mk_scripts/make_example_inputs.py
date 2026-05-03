#!/usr/bin/env python3
"""Generate the BL1 example multimodal inputs.

Writes 5 small PNGs to <out-dir>/assets/ and one JSONL with 5 requests to
<out-dir>/requests/example.jsonl. Image paths in the JSONL are absolute so
that any working directory on the experiment host can resolve them.

Usage:
    python mk_scripts/make_example_inputs.py --out-dir ~/zeyu/mono_kernel/inputs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


_BLOCKS = [
    # (filename, rgb color, digit)
    ("block_red.png", (200, 50, 50), "1"),
    ("block_blue.png", (50, 90, 200), "2"),
    ("block_green.png", (50, 170, 80), "3"),
    ("block_orange.png", (230, 140, 40), "4"),
    ("block_purple.png", (140, 60, 180), "5"),
]


_REQUESTS = [
    {"prompt": "What color is the block in the image?",
     "image_idx": 0, "output": "red", "output_tokens": 32},
    {"prompt": "What digit is shown in the image?",
     "image_idx": 0, "output": "1", "output_tokens": 16},
    {"prompt": "Describe the image briefly.",
     "image_idx": 1, "output": "blue block with digit 2", "output_tokens": 32},
    {"prompt": "What digit is shown?",
     "image_idx": 2, "output": "3", "output_tokens": 16},
    {"prompt": "Name the color and digit.",
     "image_idx": 3, "output": "orange 4", "output_tokens": 24},
]


def _draw_block(out_path: Path, rgb: tuple[int, int, int], digit: str) -> None:
    img = Image.new("RGB", (256, 256), rgb)
    draw = ImageDraw.Draw(img)
    # Try to use a default truetype font; fall back to PIL default.
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(candidate):
            try:
                font = ImageFont.truetype(candidate, 160)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), digit, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (256 - w) // 2 - bbox[0]
    y = (256 - h) // 2 - bbox[1]
    draw.text((x, y), digit, fill=(255, 255, 255), font=font)
    img.save(out_path, format="PNG")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="Project inputs/ directory (contains assets/ and requests/).")
    args = ap.parse_args()

    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    assets_dir = out_dir / "assets"
    requests_dir = out_dir / "requests"
    assets_dir.mkdir(parents=True, exist_ok=True)
    requests_dir.mkdir(parents=True, exist_ok=True)

    # Generate PNGs.
    image_paths: list[Path] = []
    for fname, rgb, digit in _BLOCKS:
        p = assets_dir / fname
        _draw_block(p, rgb, digit)
        image_paths.append(p)

    # Write example.jsonl with absolute image paths.
    jsonl_path = requests_dir / "example.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, req in enumerate(_REQUESTS):
            row = {
                "id": i,
                "prompt": req["prompt"],
                "image": str(image_paths[req["image_idx"]]),
                "output": req["output"],
                "output_tokens": req["output_tokens"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(image_paths)} PNG(s) to {assets_dir}")
    print(f"Wrote {len(_REQUESTS)} request(s) to {jsonl_path}")


if __name__ == "__main__":
    main()
