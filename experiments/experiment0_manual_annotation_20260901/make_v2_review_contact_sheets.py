#!/usr/bin/env python3
"""Build lossless review contact sheets for Experiment 0 v2 packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


BACKGROUND = (245, 246, 248)
PANEL_BACKGROUND = (255, 255, 255)
TEXT = (25, 28, 33)
BORDER = (170, 176, 186)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--panel-width", type=int, default=820)
    parser.add_argument("--panel-height", type=int, default=500)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def panel(path: Path, title: str, width: int, height: int) -> Image.Image:
    title_height = 42
    canvas = Image.new("RGB", (width, height + title_height), PANEL_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), title, fill=TEXT, font=load_font(22))
    with Image.open(path) as source:
        source = source.convert("RGB")
        fitted = ImageOps.contain(source, (width - 16, height - 16))
        x = (width - fitted.width) // 2
        y = title_height + (height - fitted.height) // 2
        canvas.paste(fitted, (x, y))
    draw.rectangle((0, 0, width - 1, height + title_height - 1), outline=BORDER, width=2)
    return canvas


def build_case_sheet(case_dir: Path, output_path: Path, width: int, height: int) -> None:
    public = json.loads((case_dir / "case_public.json").read_text(encoding="utf-8"))
    case_uid = public["case_uid"]
    header_height = 74
    gap = 18
    rows: list[tuple[tuple[Path, str], tuple[Path, str]]] = [
        (
            (case_dir / public["current"]["context_asset"], "Current context + mask"),
            (case_dir / public["current"]["crop_asset"], "Current masked crop"),
        )
    ]
    for candidate in public["candidates"]:
        code = candidate["code"]
        rows.append(
            (
                (case_dir / candidate["history_asset"], f"Candidate {code}: history"),
                (case_dir / candidate["pcd_asset"], f"Candidate {code}: 3D views"),
            )
        )

    panel_height = height + 42
    sheet_width = width * 2 + gap * 3
    sheet_height = header_height + len(rows) * panel_height + (len(rows) + 1) * gap
    sheet = Image.new("RGB", (sheet_width, sheet_height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (gap, 17),
        f"{case_uid} | frame {public['event_frame_idx']} | {public['source_frame']}",
        fill=TEXT,
        font=load_font(30),
    )
    y = header_height + gap
    for left_spec, right_spec in rows:
        left = panel(*left_spec, width, height)
        right = panel(*right_spec, width, height)
        sheet.paste(left, (gap, y))
        sheet.paste(right, (gap * 2 + width, y))
        y += panel_height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    packet_root = args.packet_root.resolve()
    output_dir = (args.output_dir or packet_root / "review_contact_sheets").resolve()
    case_dirs = sorted(path for path in (packet_root / "cases").iterdir() if path.is_dir())
    for case_dir in case_dirs:
        output_path = output_dir / f"{case_dir.name}.png"
        build_case_sheet(case_dir, output_path, args.panel_width, args.panel_height)
        print(output_path)


if __name__ == "__main__":
    main()
