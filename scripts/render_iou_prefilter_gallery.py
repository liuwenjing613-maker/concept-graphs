#!/usr/bin/env python3
"""Render every IoU-prefiltered candidate pair as an inspectable HTML gallery.

The gallery is purely post-hoc: it reads an existing online run's evidence and
does not modify a map or invoke a VLM.  Each card shows the exact shared
historical frame used by score-ordered NMS: cyan is the kept higher-score
candidate, red is the dropped candidate, and yellow is their overlap.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


KEPT_RGB = np.array([22, 201, 210], dtype=np.uint8)  # cyan
DROPPED_RGB = np.array([244, 80, 80], dtype=np.uint8)  # red
OVERLAP_RGB = np.array([255, 218, 64], dtype=np.uint8)  # yellow
PANEL_BG = np.array([19, 25, 36], dtype=np.uint8)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        key = "mask" if "mask" in archive else archive.files[0]
        return np.asarray(archive[key]).astype(bool, copy=False)


def _fit(image: np.ndarray, width: int, height: int, background: np.ndarray) -> np.ndarray:
    canvas = np.broadcast_to(background, (height, width, 3)).copy()
    if image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    out_w = max(1, int(round(image.shape[1] * scale)))
    out_h = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (out_w, out_h), interpolation=interpolation)
    x0, y0 = (width - out_w) // 2, (height - out_h) // 2
    canvas[y0:y0 + out_h, x0:x0 + out_w] = resized
    return canvas


def _put(canvas: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.58,
         color: tuple[int, int, int] = (235, 240, 248), thickness: int = 1) -> None:
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _overlay_rgb(image: np.ndarray, kept: np.ndarray, dropped: np.ndarray) -> np.ndarray:
    base = image.copy()
    overlap = kept & dropped
    kept_only = kept & ~dropped
    dropped_only = dropped & ~kept
    for mask, color in ((kept_only, KEPT_RGB), (dropped_only, DROPPED_RGB), (overlap, OVERLAP_RGB)):
        if mask.any():
            base[mask] = (0.48 * base[mask] + 0.52 * color).astype(np.uint8)
    return base


def _mask_panel(shape: tuple[int, int], kept: np.ndarray, dropped: np.ndarray) -> np.ndarray:
    panel = np.broadcast_to(PANEL_BG, (*shape, 3)).copy()
    overlap = kept & dropped
    panel[kept & ~dropped] = KEPT_RGB
    panel[dropped & ~kept] = DROPPED_RGB
    panel[overlap] = OVERLAP_RGB
    return panel


def _draw_legend(canvas: np.ndarray, x: int, y: int) -> None:
    entries = ((KEPT_RGB, "KEPT (higher score)"), (DROPPED_RGB, "DROPPED"), (OVERLAP_RGB, "OVERLAP"))
    for index, (color, label) in enumerate(entries):
        px = x + index * 220
        cv2.rectangle(canvas, (px, y - 13), (px + 18, y + 5), tuple(int(v) for v in color), -1)
        _put(canvas, label, (px + 26, y + 2), scale=0.47)


def _render_card(
    *, record: dict[str, Any], drop: dict[str, Any], observations: dict[str, dict[str, Any]],
    run_root: Path, output_path: Path,
) -> dict[str, Any]:
    evidence = drop["shared_frame_evidence"]
    dropped_obs = observations[evidence["left_obs_uid"]]
    kept_obs = observations[evidence["right_obs_uid"]]
    dropped_mask_path = run_root / dropped_obs["processed_mask_ref"]["path"]
    kept_mask_path = run_root / kept_obs["processed_mask_ref"]["path"]
    dropped_mask = _read_mask(dropped_mask_path)
    kept_mask = _read_mask(kept_mask_path)
    if kept_mask.shape != dropped_mask.shape:
        kept_mask = cv2.resize(
            kept_mask.astype(np.uint8), (dropped_mask.shape[1], dropped_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    source_path = Path(evidence["shared_rgb_path"])
    source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        source_rgb = np.broadcast_to(PANEL_BG, (*dropped_mask.shape, 3)).copy()
        source_missing = True
    else:
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        if source_rgb.shape[:2] != dropped_mask.shape:
            source_rgb = cv2.resize(source_rgb, (dropped_mask.shape[1], dropped_mask.shape[0]), interpolation=cv2.INTER_AREA)
        source_missing = False
    overlap = kept_mask & dropped_mask
    union = kept_mask | dropped_mask
    measured_iou = float(overlap.sum() / union.sum()) if union.any() else 0.0
    left = _fit(_overlay_rgb(source_rgb, kept_mask, dropped_mask), 780, 490, PANEL_BG)
    right = _fit(_mask_panel(dropped_mask.shape, kept_mask, dropped_mask), 780, 490, PANEL_BG)
    canvas = np.full((720, 1600, 3), (13, 18, 28), dtype=np.uint8)
    canvas[125:615, 15:795] = left
    canvas[125:615, 805:1585] = right
    _put(canvas, f"IoU prefilter: {record['source_frame_id']} | detection {record['detected_obj_idx']}", (20, 35), 0.82, thickness=2)
    raw = record["raw_trigger_before_iou"]
    trigger_detail = f"raw {raw['kind']} trigger"
    if raw["kind"] == "association":
        trigger_detail += f" | margin={raw['margin']:.6f}"
    else:
        trigger_detail += f" | distance={raw['threshold_distance']:.6f}"
    _put(canvas, trigger_detail + " -> SUPPRESSED AFTER NMS", (20, 68), 0.57, (255, 219, 120), 2)
    _draw_legend(canvas, 20, 105)
    _put(canvas, "Shared historical RGB + masks", (20, 652), 0.56)
    _put(canvas, "Binary masks only (same pixels)", (810, 652), 0.56)
    _put(canvas, f"KEEP  idx={drop['representative_object_index']}  score={drop['representative_score']:.6f}", (20, 684), 0.52, tuple(int(v) for v in KEPT_RGB), 2)
    _put(canvas, f"DROP  idx={drop['object_index']}  score={drop['score']:.6f}", (535, 684), 0.52, tuple(int(v) for v in DROPPED_RGB), 2)
    _put(canvas, f"recorded IoU={drop['same_frame_mask_iou']:.6f} | rendered IoU={measured_iou:.6f}", (1010, 684), 0.47, tuple(int(v) for v in OVERLAP_RGB), 1)
    _put(canvas, f"shared frame={evidence['shared_frame_idx']} | keep obs={evidence['right_obs_uid']} | drop obs={evidence['left_obs_uid']}", (20, 712), 0.41, (184, 195, 212), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"failed writing {output_path}")
    return {
        "image": output_path.name,
        "frame_idx": record["frame_idx"],
        "source_frame_id": record["source_frame_id"],
        "detected_obj_idx": record["detected_obj_idx"],
        "raw_trigger": raw,
        "kept_index": drop["representative_object_index"],
        "dropped_index": drop["object_index"],
        "kept_uid": drop.get("representative_object_uid"),
        "dropped_uid": drop.get("object_uid"),
        "recorded_iou": drop["same_frame_mask_iou"],
        "rendered_iou": measured_iou,
        "shared_frame_idx": evidence["shared_frame_idx"],
        "shared_rgb_path": str(source_path),
        "source_missing": source_missing,
    }


def _write_html(cards: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for card in cards:
        raw = card["raw_trigger"]
        raw_value = raw.get("margin", raw.get("threshold_distance"))
        rows.append(
            "<article>"
            f"<h2>{html.escape(card['source_frame_id'])} · det {card['detected_obj_idx']} · {html.escape(raw['kind'])}</h2>"
            f"<p>Raw trigger value={raw_value:.6f} · kept index={card['kept_index']} · dropped index={card['dropped_index']} · IoU={card['recorded_iou']:.6f}</p>"
            f"<a href=\"images/{html.escape(card['image'])}\"><img src=\"images/{html.escape(card['image'])}\" alt=\"IoU prefilter comparison\"></a>"
            f"<details><summary>Evidence metadata</summary><pre>{html.escape(json.dumps(card, ensure_ascii=False, indent=2))}</pre></details>"
            "</article>"
        )
    document = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>IoU prefilter gallery</title>
<style>body{{font-family:system-ui,sans-serif;background:#0d121c;color:#edf2f7;margin:24px}}article{{border:1px solid #344055;border-radius:8px;padding:14px;margin:16px 0;background:#151c29}}h1,h2{{margin:0 0 8px}}p{{color:#cbd5e1}}img{{width:min(100%,1200px);border:1px solid #475569}}pre{{white-space:pre-wrap;color:#cbd5e1}}summary{{cursor:pointer;color:#7dd3fc}}</style></head><body>
<h1>IoU prefilter: filtered candidate comparisons</h1><p>{len(cards)} dropped candidate comparisons. Cyan = kept higher-score candidate; red = filtered candidate; yellow = exact overlap. Click an image for full size.</p>
{''.join(rows)}</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path, help="Experiment root containing blocking_association_gate and evidence")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    gate_dir = run_root / "blocking_association_gate"
    output_dir = (args.output_dir or gate_dir / "iou_prefilter_gallery").resolve()
    prefilter_path = gate_dir / "iou_prefilter.jsonl"
    observations = {row["obs_uid"]: row for row in _read_jsonl(run_root / "evidence" / "observations.jsonl")}
    cards: list[dict[str, Any]] = []
    sequence = 0
    for record in _read_jsonl(prefilter_path):
        if record.get("outcome") != "trigger_suppressed":
            continue
        for drop in record["candidate_iou_prefilter_hidden_from_vlm"].get("dropped", []):
            sequence += 1
            image_name = f"{sequence:03d}_{record['source_frame_id']}_d{record['detected_obj_idx']:03d}_drop{drop['object_index']}.jpg"
            cards.append(_render_card(
                record=record, drop=drop, observations=observations, run_root=run_root,
                output_path=output_dir / "images" / image_name,
            ))
    (output_dir / "manifest.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_html(cards, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "comparisons": len(cards)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
