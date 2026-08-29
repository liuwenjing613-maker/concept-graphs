#!/usr/bin/env python3
"""Summarize two-scene, three-scale case-oracle metrics and parity."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCALES = ("voxel0p025", "voxel0p05", "voxel0p10")
VARIANTS = ("baseline", "semantic", "spurious", "geometry", "association", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--original-matching-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def semantic_accuracy(path: Path) -> tuple[float | None, int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [row for row in rows if as_bool(row.get("semantic_eligible"))]
    correct = sum(as_bool(row.get("semantic_consistent")) for row in eligible)
    return (correct / len(eligible) if eligible else None, correct, len(eligible))


def timing(path: Path) -> dict[str, float | int | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"wall_s=([0-9.]+) max_rss_kb=(\d+)", text)
    return {
        "wall_seconds": float(match.group(1)) if match else None,
        "max_rss_kb": int(match.group(2)) if match else None,
    }


def flatten(report: dict[str, Any], semantic: tuple[float | None, int, int]) -> dict[str, Any]:
    threshold = report["object_metrics_by_iou_threshold"]["0.25"]
    purity = report["distributions"]["maximum_purity_per_prediction"]
    coverage = report["distributions"]["maximum_coverage_per_gt"]
    return {
        "predicted_objects": report["counts"]["predicted_objects"],
        "gt_instances": report["counts"]["observable_gt_instances"],
        "positive_overlap_pairs": report["counts"]["positive_overlap_pairs"],
        "object_precision_iou0p25": threshold["precision"],
        "object_recall_iou0p25": threshold["recall"],
        "object_f1_iou0p25": threshold["f1"],
        "mean_maximum_purity": purity["mean"],
        "median_maximum_purity": purity["median"],
        "mean_maximum_coverage": coverage["mean"],
        "median_maximum_coverage": coverage["median"],
        "semantic_accuracy_hungarian_positive": semantic[0],
        "semantic_correct": semantic[1],
        "semantic_denominator": semantic[2],
    }


def nested_numeric_differences(left: Any, right: Any, prefix: str = "") -> list[tuple[str, float]]:
    differences: list[tuple[str, float]] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) & set(right)):
            if key in {"prediction_map", "prediction_map_sha256", "gt_map", "gt_map_sha256"}:
                continue
            differences.extend(nested_numeric_differences(left[key], right[key], f"{prefix}.{key}" if prefix else key))
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        differences.append((prefix, abs(float(left) - float(right))))
    return differences


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    parity = []
    for scene in ("room0", "office0"):
        for scale in SCALES:
            original = read_json(args.original_matching_root / scene / scale / "audit_summary.json")
            baseline = read_json(args.metrics_root / scene / "baseline" / scale / "audit_summary.json")
            differences = nested_numeric_differences(original, baseline)
            max_difference = max((value for _key, value in differences), default=0.0)
            parity.append({
                "scene_id": scene,
                "scale": scale,
                "numeric_field_count": len(differences),
                "maximum_absolute_difference": max_difference,
                "pass_tolerance_1e_12": max_difference <= 1e-12,
            })
            for variant in VARIANTS:
                base = args.metrics_root / scene / variant / scale
                report = read_json(base / "audit_summary.json")
                sem = semantic_accuracy(base / "hungarian_matches.csv")
                row = {
                    "scene_id": scene,
                    "scale": scale,
                    "variant": variant,
                    **flatten(report, sem),
                    **timing(base / "run.log"),
                }
                rows.append(row)

    by_key = {(row["scene_id"], row["scale"], row["variant"]): row for row in rows}
    metric_names = (
        "object_precision_iou0p25",
        "object_recall_iou0p25",
        "object_f1_iou0p25",
        "mean_maximum_purity",
        "mean_maximum_coverage",
        "semantic_accuracy_hungarian_positive",
    )
    deltas = []
    robustness: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    epsilon = 1e-12
    for row in rows:
        if row["variant"] == "baseline":
            continue
        baseline = by_key[(row["scene_id"], row["scale"], "baseline")]
        delta_row = {"scene_id": row["scene_id"], "scale": row["scale"], "variant": row["variant"]}
        for metric in metric_names:
            value = row[metric]
            base_value = baseline[metric]
            delta = None if value is None or base_value is None else float(value) - float(base_value)
            delta_row[f"delta_{metric}"] = delta
        deltas.append(delta_row)
    for variant in VARIANTS[1:]:
        variant_rows = [row for row in deltas if row["variant"] == variant]
        for metric in metric_names:
            values = [row[f"delta_{metric}"] for row in variant_rows if row[f"delta_{metric}"] is not None]
            robustness[variant][metric] = {
                "improved_cells": sum(value > epsilon for value in values),
                "unchanged_cells": sum(abs(value) <= epsilon for value in values),
                "degraded_cells": sum(value < -epsilon for value in values),
                "cell_count": len(values),
            }

    macro = []
    for scale in SCALES:
        for variant in VARIANTS:
            selected = [row for row in rows if row["scale"] == scale and row["variant"] == variant]
            item: dict[str, Any] = {"scale": scale, "variant": variant}
            for metric in metric_names:
                values = [float(row[metric]) for row in selected if row[metric] is not None]
                item[metric] = sum(values) / len(values) if values else None
            macro.append(item)
    macro_by_key = {(row["scale"], row["variant"]): row for row in macro}
    macro_deltas = []
    for row in macro:
        if row["variant"] == "baseline":
            continue
        baseline = macro_by_key[(row["scale"], "baseline")]
        macro_deltas.append({
            "scale": row["scale"],
            "variant": row["variant"],
            **{
                f"delta_{metric}": None if row[metric] is None or baseline[metric] is None else row[metric] - baseline[metric]
                for metric in metric_names
            },
        })

    wall_values = [row["wall_seconds"] for row in rows if row["wall_seconds"] is not None]
    summary = {
        "schema_version": "1.0.0",
        "baseline_light_parity": parity,
        "baseline_light_parity_all_pass": all(item["pass_tolerance_1e_12"] for item in parity),
        "per_scene_scale_variant": rows,
        "per_scene_scale_delta": deltas,
        "macro_two_scene": macro,
        "macro_delta_vs_baseline": macro_deltas,
        "robustness_sign_counts": robustness,
        "timing": {
            "job_count": len(rows),
            "sum_job_wall_seconds": sum(wall_values),
            "mean_job_wall_seconds": sum(wall_values) / len(wall_values) if wall_values else None,
            "maximum_job_wall_seconds": max(wall_values) if wall_values else None,
            "note": "Jobs ran with up to four-way overlap; sum is not elapsed wall time.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = [row for row in macro_deltas if row["scale"] == "voxel0p05"]
    print(json.dumps({"parity": summary["baseline_light_parity_all_pass"], "primary_macro_deltas": primary, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
