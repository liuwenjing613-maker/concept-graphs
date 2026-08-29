#!/usr/bin/env python3
"""Select a scale-stable, GT-derived candidate pool without human labels.

This script deliberately selects *observable phenomena*, not causal error labels.
Causal labels (for example association-rooted) must be assigned only after the
online observation history has been reviewed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCALE_DIRS = ("voxel0p025", "voxel0p05", "voxel0p10")
SCALE_NAMES = {"voxel0p025": "0.025", "voxel0p05": "0.05", "voxel0p10": "0.10"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-map",
        action="append",
        required=True,
        metavar="SCENE=PATH",
        help="Fresh B0 map for one scene; may be passed more than once.",
    )
    parser.add_argument("--per-scene-per-phenomenon", type=int, default=3)
    parser.add_argument("--controls-per-scene", type=int, default=3)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None, "nan", "NaN"):
        return default
    return float(value)


def i(row: dict[str, str], key: str, default: int = -1) -> int:
    value = row.get(key, "")
    if value in ("", None, "nan", "NaN"):
        return default
    return int(float(value))


def b(row: dict[str, str], key: str, default: bool = False) -> bool:
    value = row.get(key, "")
    if value in ("", None, "nan", "NaN"):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value for {key}: {value!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    middle = n // 2
    if n % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def load_map_metadata(path: Path) -> dict[int, dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    result: dict[int, dict[str, Any]] = {}
    for index, obj in enumerate(payload["objects"]):
        frames = [int(value) for value in obj.get("image_idx", [])]
        masks = [int(value) for value in obj.get("mask_idx", [])]
        raw_labels = obj.get("class_name", [])
        if isinstance(raw_labels, str):
            labels = [raw_labels]
        else:
            labels = [str(value) for value in raw_labels]
        result[index] = {
            "predicted_uuid": str(obj.get("id", "")),
            "map_label": str(obj.get("consolidated_caption") or (Counter(labels).most_common(1)[0][0] if labels else "")),
            "first_frame": min(frames) if frames else None,
            "last_frame": max(frames) if frames else None,
            "observation_count": len(frames),
            "first_mask_idx": masks[frames.index(min(frames))] if frames and len(masks) == len(frames) else None,
            "all_frames": frames,
            "all_mask_indices": masks,
        }
    return result


def add_scale_fields(output: dict[str, Any], rows: dict[str, dict[str, str]], fields: Iterable[str]) -> None:
    for scale_dir, row in rows.items():
        suffix = SCALE_NAMES[scale_dir].replace(".", "p")
        for field in fields:
            output[f"{field}_{suffix}"] = f(row, field)


def top_unique(rows: list[dict[str, Any]], count: int, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in sorted(rows, key=lambda item: (-float(item["stable_score"]), str(item["case_key"]))):
        key = tuple(row.get(field) for field in key_fields)
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) == count:
            break
    return selected


def top_disjoint_pairs(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Greedily keep both prediction and GT identities distinct."""
    selected: list[dict[str, Any]] = []
    seen_predictions: set[int] = set()
    seen_gt: set[int] = set()
    for row in sorted(rows, key=lambda item: (-float(item["stable_score"]), str(item["case_key"]))):
        predicted_index = int(row["predicted_index"])
        gt_id = int(row["gt_instance_id"])
        if predicted_index in seen_predictions or gt_id in seen_gt:
            continue
        selected.append(row)
        seen_predictions.add(predicted_index)
        seen_gt.add(gt_id)
        if len(selected) == count:
            break
    return selected


def main() -> None:
    args = parse_args()
    scene_maps: dict[str, Path] = {}
    for item in args.scene_map:
        scene, separator, raw_path = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid --scene-map: {item}")
        scene_maps[scene] = Path(raw_path)

    candidates: list[dict[str, Any]] = []
    input_files: list[Path] = []

    for scene, map_path in sorted(scene_maps.items()):
        map_metadata = load_map_metadata(map_path)
        input_files.append(map_path)
        pred_by_scale: dict[str, dict[int, dict[str, str]]] = {}
        gt_by_scale: dict[str, dict[int, dict[str, str]]] = {}
        overlaps_by_scale: dict[str, list[dict[str, str]]] = {}

        for scale_dir in SCALE_DIRS:
            base = args.matching_root / scene / scale_dir
            pred_path = base / "predicted_object_summary.csv"
            gt_path = base / "gt_instance_summary.csv"
            overlap_path = base / "object_gt_overlaps.csv"
            input_files.extend((pred_path, gt_path, overlap_path))
            pred_by_scale[scale_dir] = {i(row, "predicted_index"): row for row in read_csv(pred_path)}
            gt_by_scale[scale_dir] = {i(row, "gt_instance_id"): row for row in read_csv(gt_path)}
            overlaps_by_scale[scale_dir] = read_csv(overlap_path)

        common_pred = set.intersection(*(set(rows) for rows in pred_by_scale.values()))
        common_gt = set.intersection(*(set(rows) for rows in gt_by_scale.values()))

        # Total GT-supported fraction is different from maximum purity: it stays
        # low only when a prediction is genuinely unsupported, not merely merged.
        total_support: dict[str, dict[int, float]] = {}
        for scale_dir, rows in overlaps_by_scale.items():
            intersections: defaultdict[int, float] = defaultdict(float)
            predicted_voxels: dict[int, float] = {}
            for row in rows:
                pred_index = i(row, "predicted_index")
                intersections[pred_index] += f(row, "intersection_voxels")
                predicted_voxels[pred_index] = f(row, "predicted_voxels")
            total_support[scale_dir] = {
                pred_index: intersections[pred_index] / predicted_voxels[pred_index]
                for pred_index in predicted_voxels
                if predicted_voxels[pred_index] > 0
            }

        contamination: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        for pred_index in common_pred:
            rows = {scale: pred_by_scale[scale][pred_index] for scale in SCALE_DIRS}
            voxels = [f(row, "predicted_voxels") for row in rows.values()]
            purities = [f(row, "maximum_purity") for row in rows.values()]
            degree_05 = [f(row, "overlap_gt_count_purity_0p05") for row in rows.values()]
            supports = [total_support[scale].get(pred_index, 0.0) for scale in SCALE_DIRS]
            metadata = map_metadata.get(pred_index, {})

            base = {
                "scene_id": scene,
                "entity_type": "prediction",
                "predicted_index": pred_index,
                "gt_instance_id": None,
                "predicted_label": rows["voxel0p05"].get("predicted_label", ""),
                "gt_label": None,
                **metadata,
            }

            if median(voxels) >= 25 and min(degree_05) >= 2:
                row = {
                    **base,
                    "phenomenon": "contamination",
                    "case_key": f"{scene}:pred:{pred_index}:contamination",
                    "stable_score": min(1.0 - value for value in purities)
                    * (1.0 + min(min(degree_05), 5.0) / 5.0),
                    "median_maximum_purity": median(purities),
                    "minimum_overlap_gt_count_purity_0p05": min(degree_05),
                    "median_total_gt_support": median(supports),
                }
                add_scale_fields(row, rows, ("predicted_voxels", "maximum_purity", "overlap_gt_count_purity_0p05"))
                contamination.append(row)

            if median(voxels) >= 25:
                row = {
                    **base,
                    "phenomenon": "unsupported_prediction",
                    "case_key": f"{scene}:pred:{pred_index}:unsupported",
                    "stable_score": 1.0 - max(supports),
                    "median_maximum_purity": median(purities),
                    "minimum_overlap_gt_count_purity_0p05": min(degree_05),
                    "median_total_gt_support": median(supports),
                }
                add_scale_fields(row, rows, ("predicted_voxels", "maximum_purity"))
                for scale_dir, value in zip(SCALE_DIRS, supports):
                    row[f"total_gt_support_{SCALE_NAMES[scale_dir].replace('.', 'p')}"] = value
                unsupported.append(row)

        incomplete: list[dict[str, Any]] = []
        fragmentation: list[dict[str, Any]] = []
        for gt_id in common_gt:
            rows = {scale: gt_by_scale[scale][gt_id] for scale in SCALE_DIRS}
            voxels = [f(row, "gt_voxels") for row in rows.values()]
            coverage = [f(row, "maximum_coverage") for row in rows.values()]
            degree_05 = [f(row, "overlap_prediction_count_coverage_0p05") for row in rows.values()]
            best_pred = i(rows["voxel0p05"], "best_predicted_index")
            metadata = map_metadata.get(best_pred, {}) if best_pred >= 0 else {}
            base = {
                "scene_id": scene,
                "entity_type": "gt_instance",
                "predicted_index": best_pred if best_pred >= 0 else None,
                "gt_instance_id": gt_id,
                "predicted_label": rows["voxel0p05"].get("best_predicted_label", ""),
                "gt_label": rows["voxel0p05"].get("gt_label", ""),
                **metadata,
            }
            if median(voxels) >= 25:
                row = {
                    **base,
                    "phenomenon": "incompleteness",
                    "case_key": f"{scene}:gt:{gt_id}:incompleteness",
                    "stable_score": 1.0 - max(coverage),
                    "median_maximum_coverage": median(coverage),
                    "minimum_overlap_prediction_count_coverage_0p05": min(degree_05),
                }
                add_scale_fields(row, rows, ("gt_voxels", "maximum_coverage", "overlap_prediction_count_coverage_0p05"))
                incomplete.append(row)

            if median(voxels) >= 25 and min(degree_05) >= 2:
                row = {
                    **base,
                    "phenomenon": "fragmentation",
                    "case_key": f"{scene}:gt:{gt_id}:fragmentation",
                    "stable_score": min(degree_05) * (1.0 - max(coverage)),
                    "median_maximum_coverage": median(coverage),
                    "minimum_overlap_prediction_count_coverage_0p05": min(degree_05),
                }
                add_scale_fields(row, rows, ("gt_voxels", "maximum_coverage", "overlap_prediction_count_coverage_0p05"))
                fragmentation.append(row)

        # Pair-level phenomena. Requiring the same pair at every scale prevents
        # a scale-dependent nearest neighbour from masquerading as a stable case.
        pair_rows: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
        for scale_dir, rows in overlaps_by_scale.items():
            for row in rows:
                pair_rows[(i(row, "predicted_index"), i(row, "gt_instance_id"))][scale_dir] = row

        semantic: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        for (pred_index, gt_id), rows in pair_rows.items():
            if set(rows) != set(SCALE_DIRS):
                continue
            ious = [f(rows[scale], "voxel_iou") for scale in SCALE_DIRS]
            purities = [f(rows[scale], "purity") for scale in SCALE_DIRS]
            coverage = [f(rows[scale], "coverage") for scale in SCALE_DIRS]
            semantic_consistent = [b(rows[scale], "semantic_consistent") for scale in SCALE_DIRS]
            metadata = map_metadata.get(pred_index, {})
            base = {
                "scene_id": scene,
                "entity_type": "prediction_gt_pair",
                "predicted_index": pred_index,
                "gt_instance_id": gt_id,
                "predicted_label": rows["voxel0p05"].get("predicted_label", ""),
                "gt_label": rows["voxel0p05"].get("gt_label", ""),
                **metadata,
            }
            if not any(semantic_consistent) and min(ious) >= 0.15 and median(purities) >= 0.45 and median(coverage) >= 0.25:
                row = {
                    **base,
                    "phenomenon": "semantic_mismatch",
                    "case_key": f"{scene}:pair:{pred_index}:{gt_id}:semantic",
                    "stable_score": min(ious) + 0.25 * min(purities) + 0.25 * min(coverage),
                    "median_voxel_iou": median(ious),
                    "minimum_purity": min(purities),
                    "minimum_coverage": min(coverage),
                }
                add_scale_fields(row, rows, ("voxel_iou", "purity", "coverage", "bbox_iou", "center_distance_m"))
                semantic.append(row)

            if all(semantic_consistent):
                row = {
                    **base,
                    "phenomenon": "relative_clean_control",
                    "case_key": f"{scene}:pair:{pred_index}:{gt_id}:relative_control",
                    "stable_score": min(ious) + 0.5 * min(purities) + 0.5 * min(coverage),
                    "median_voxel_iou": median(ious),
                    "minimum_purity": min(purities),
                    "minimum_coverage": min(coverage),
                    "absolute_clean_gate_pass": min(ious) >= 0.25 and min(purities) >= 0.50 and min(coverage) >= 0.50,
                }
                add_scale_fields(row, rows, ("voxel_iou", "purity", "coverage", "bbox_iou", "center_distance_m"))
                controls.append(row)

        pools = {
            "semantic_mismatch": semantic,
            "contamination": contamination,
            "incompleteness": incomplete,
            "fragmentation": fragmentation,
            "unsupported_prediction": unsupported,
        }
        for phenomenon, rows in pools.items():
            chosen = top_unique(rows, args.per_scene_per_phenomenon, ("predicted_index", "gt_instance_id"))
            for rank, row in enumerate(chosen, 1):
                row["rank_within_scene_phenomenon"] = rank
                candidates.append(row)

        chosen_controls = top_disjoint_pairs(controls, args.controls_per_scene)
        for rank, row in enumerate(chosen_controls, 1):
            row["rank_within_scene_phenomenon"] = rank
            candidates.append(row)

    candidates.sort(key=lambda row: (row["phenomenon"], row["scene_id"], row["rank_within_scene_phenomenon"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Keep frame lists in JSON; the CSV is intentionally compact for review.
    json_path = args.output_dir / "objective_candidate_pool.json"
    json_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    compact_rows = []
    for row in candidates:
        compact_rows.append({key: value for key, value in row.items() if key not in {"all_frames", "all_mask_indices"}})
    fieldnames = sorted({key for row in compact_rows for key in row})
    csv_path = args.output_dir / "objective_candidate_pool.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compact_rows)

    manifest = {
        "selection_stage": "pre_causal_objective_candidate_pool",
        "human_labels_used": False,
        "oracle_repair_results_used": False,
        "scales_m": [0.025, 0.05, 0.10],
        "selection_policy": {
            "per_scene_per_phenomenon": args.per_scene_per_phenomenon,
            "controls_per_scene": args.controls_per_scene,
            "note": "Phenomena are GT-derived observations, not causal error labels.",
        },
        "counts": dict(Counter(row["phenomenon"] for row in candidates)),
        "scene_counts": dict(Counter(row["scene_id"] for row in candidates)),
        "input_sha256": {str(path): sha256(path) for path in sorted(set(input_files))},
        "outputs": {str(csv_path): sha256(csv_path), str(json_path): sha256(json_path)},
    }
    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates), "counts": manifest["counts"], "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
