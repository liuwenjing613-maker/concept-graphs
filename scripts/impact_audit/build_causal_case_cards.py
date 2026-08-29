#!/usr/bin/env python3
"""Render compact evidence cards for the two highest-impact reviewed families."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        scene, path = value.split("=", 1)
        result[scene] = Path(path)
    return result


def load_map(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_mask(obj: dict[str, Any], frame: int) -> np.ndarray:
    for image_idx, mask in zip(obj.get("image_idx", []), obj.get("mask", [])):
        if int(image_idx) == frame:
            return np.asarray(mask, dtype=bool)
    raise KeyError(f"object has no observation at processed frame {frame}")


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    color = np.array([255.0, 40.0, 40.0], dtype=np.float32)
    result[mask] = 0.45 * result[mask] + 0.55 * color
    return np.clip(result, 0, 255).astype(np.uint8)


def crop_bounds(mask: np.ndarray, margin: int = 30) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(mask.shape[1], int(xs.max()) + margin + 1),
        min(mask.shape[0], int(ys.max()) + margin + 1),
    )


def signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-cases", required=True, type=Path)
    parser.add_argument("--case-summary", required=True, type=Path)
    parser.add_argument("--geometry-audit", required=True, type=Path)
    parser.add_argument("--b0-map", required=True, action="append")
    parser.add_argument("--o3-map", required=True, action="append")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    frozen = json.loads(args.frozen_cases.read_text(encoding="utf-8"))
    cases = [case for case in frozen if case["family"] in {"geometry", "association"}]
    summary = json.loads(args.case_summary.read_text(encoding="utf-8"))
    primary = {row["case_id"]: row for row in summary["primary_5cm"]}
    geometry = {
        row["case_id"]: row
        for row in json.loads(args.geometry_audit.read_text(encoding="utf-8"))["cases"]
    }
    b0_paths = named_paths(args.b0_map)
    o3_paths = named_paths(args.o3_map)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for scene in sorted({case["scene_id"] for case in cases}):
        b0 = load_map(b0_paths[scene])
        o3 = load_map(o3_paths[scene])
        gt_objects = {
            int(obj["oracle_gt_id"]): obj
            for obj in o3["objects"]
            if obj.get("oracle_gt_id") is not None
        }
        for case in [item for item in cases if item["scene_id"] == scene]:
            frame = int(case["s_processed_frame"])
            rgb_path = Path(case["i1_event"]["rgb_path"])
            image = np.asarray(Image.open(rgb_path).convert("RGB"))
            if case["family"] == "geometry":
                source = gt_objects[int(case["gt_instance_id"])]
                source_identity = f"O3 GT instance {case['gt_instance_id']}"
            else:
                source_index = int(case["i1_event"].get("predicted_index", case["predicted_index"]))
                source = b0["objects"][source_index]
                source_identity = f"B0 predicted object {source_index}"
            mask = object_mask(source, frame)
            if mask.shape != image.shape[:2]:
                raise ValueError(f"mask/image shape mismatch for {case['case_id']}")
            blended = overlay(image, mask)
            left, top, right, bottom = crop_bounds(mask)
            row = primary[case["case_id"]]
            lines = [
                f"case: {case['case_id']}  scene: {scene}",
                f"family/subtype: {case['family']} / {case['subtype']}",
                f"label: {case.get('predicted_label') or '-'} -> {case.get('gt_label') or '-'}",
                f"timeline: s={case['s_processed_frame']} <= d={case['d_processed_frame']} <= h={case['h_processed_frame']}; c=N/A",
                f"I1 source: {source_identity}; raw frame={case['i1_event']['raw_frame']}",
                "5 cm deltas:",
                f"  F1 {signed(row['delta_object_f1_iou0p25'])}",
                f"  purity {signed(row['delta_mean_maximum_purity'])}",
                f"  coverage {signed(row['delta_mean_maximum_coverage'])}",
                f"  semantic {signed(row['delta_semantic_accuracy_hungarian_positive'])}",
                f"verdict: {row['verdict']}",
            ]
            if case["family"] == "geometry":
                audit = geometry[case["case_id"]]
                lines.extend(
                    [
                        f"root cause: {audit['root_cause_classification']}",
                        "human audit used as truth: no",
                    ]
                )
            else:
                obs = case["observation_identity_audit"]
                count = obs.get("significant_gt_count", obs.get("significant_prediction_count"))
                purity = obs.get(
                    "median_confident_observation_gt_purity",
                    obs.get("median_assigned_observation_gt_purity"),
                )
                lines.extend(
                    [
                        f"identity groups/fragments: {count}",
                        f"median mask identity purity: {purity:.3f}",
                        "human audit used as truth: no",
                    ]
                )

            figure, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.5, 1.0, 1.25]})
            axes[0].imshow(blended)
            axes[0].set_title("I1 at error-start frame (red mask)")
            axes[1].imshow(blended[top:bottom, left:right])
            axes[1].set_title("Target crop")
            axes[2].axis("off")
            axes[2].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
            for axis in axes[:2]:
                axis.axis("off")
            figure.tight_layout()
            output = args.output_dir / f"{case['case_id']}.png"
            figure.savefig(output, dpi=130, bbox_inches="tight")
            plt.close(figure)
            manifest.append(
                {
                    "case_id": case["case_id"],
                    "scene_id": scene,
                    "family": case["family"],
                    "image": str(output),
                    "image_sha256": sha256(output),
                    "i1_rgb": str(rgb_path),
                    "i1_processed_frame": frame,
                    "mask_source": source_identity,
                    "verdict_5cm": row["verdict"],
                }
            )
        del b0, o3, gt_objects
        gc.collect()

    manifest_path = args.output_dir / "case_cards_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"card_count": len(manifest), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
