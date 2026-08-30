#!/usr/bin/env python3
"""Evaluate object structure with FROSS-style 0.1 m point correspondence.

This is an evaluation-only audit.  It never changes a map.  Points are first
deduplicated on a frozen voxel grid, then each predicted point is assigned to
its nearest observable-GT point within ``radius``.  The FROSS correspondence
gate is applied before a deterministic one-to-one selection.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("map must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("map must be NAME=PATH")
    return name, Path(raw_path)


def load_map(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def voxel_centers(points: np.ndarray, voxel_size: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected Nx3 points, got {values.shape}")
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if not len(values):
        return np.empty((0, 3), dtype=np.float32)
    quantized = np.floor(values / voxel_size).astype(np.int32)
    quantized = np.unique(quantized, axis=0)
    return ((quantized.astype(np.float64) + 0.5) * voxel_size).astype(np.float32)


def prepare_objects(objects: list[dict], voxel_size: float) -> list[dict]:
    prepared: list[dict] = []
    for source_index, obj in enumerate(objects):
        points = voxel_centers(np.asarray(obj["pcd_np"]), voxel_size)
        if not len(points):
            continue
        prepared.append(
            {
                "source_index": int(source_index),
                "label": str(obj.get("class_name", "unknown")),
                "oracle_gt_id": obj.get("oracle_gt_id"),
                "points": points,
            }
        )
    return prepared


def concatenate_points(objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not objects:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
    points = np.concatenate([obj["points"] for obj in objects], axis=0)
    owners = np.concatenate(
        [np.full(len(obj["points"]), index, dtype=np.int32) for index, obj in enumerate(objects)]
    )
    return points, owners


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def evaluate_prepared(
    predicted: list[dict],
    ground_truth: list[dict],
    radius: float,
    workers: int,
) -> dict:
    gt_points, gt_point_owners = concatenate_points(ground_truth)
    pred_points, pred_point_owners = concatenate_points(predicted)
    gt_tree = cKDTree(gt_points) if len(gt_points) else None
    pred_tree = cKDTree(pred_points) if len(pred_points) else None

    # For every GT surface point, record which prediction is its nearest support.
    coverage_counts = np.zeros((len(ground_truth), len(predicted)), dtype=np.int64)
    any_covered_counts = np.zeros(len(ground_truth), dtype=np.int64)
    if pred_tree is not None:
        for gt_index, gt in enumerate(ground_truth):
            distances, neighbors = pred_tree.query(
                gt["points"], k=1, distance_upper_bound=radius, workers=workers
            )
            valid = np.isfinite(distances) & (neighbors < len(pred_point_owners))
            any_covered_counts[gt_index] = int(valid.sum())
            if valid.any():
                owner_counts = np.bincount(
                    pred_point_owners[neighbors[valid]], minlength=len(predicted)
                )
                coverage_counts[gt_index, :] = owner_counts[: len(predicted)]

    rows: list[dict] = []
    for pred_index, pred in enumerate(predicted):
        point_count = len(pred["points"])
        identity_counts = np.zeros(len(ground_truth), dtype=np.int64)
        matched_count = 0
        dominant_distance_mean = None
        if gt_tree is not None:
            distances, neighbors = gt_tree.query(
                pred["points"], k=1, distance_upper_bound=radius, workers=workers
            )
            valid = np.isfinite(distances) & (neighbors < len(gt_point_owners))
            matched_count = int(valid.sum())
            if valid.any():
                identity_counts = np.bincount(
                    gt_point_owners[neighbors[valid]], minlength=len(ground_truth)
                )[: len(ground_truth)]
        ranked = np.argsort(identity_counts)[::-1] if len(identity_counts) else np.array([], dtype=int)
        dominant_gt = int(ranked[0]) if len(ranked) and identity_counts[ranked[0]] else None
        largest = int(identity_counts[dominant_gt]) if dominant_gt is not None else 0
        second = int(identity_counts[ranked[1]]) if len(ranked) > 1 else 0
        purity = largest / point_count if point_count else 0.0
        support_fraction = matched_count / point_count if point_count else 0.0
        second_to_first = second / largest if largest else None
        fross_candidate = bool(
            dominant_gt is not None
            and purity > 0.5
            and (second_to_first is None or second_to_first < 0.75)
        )
        if dominant_gt is not None and gt_tree is not None:
            # ``cKDTree.query`` returns ``tree.n`` as the neighbor index for an
            # unmatched point.  Never index that sentinel before applying the
            # finite/range mask.
            owned = np.zeros_like(valid, dtype=bool)
            owned[valid] = gt_point_owners[neighbors[valid]] == dominant_gt
            if owned.any():
                dominant_distance_mean = float(np.mean(distances[owned]))
        threshold_count = max(1, int(math.ceil(0.05 * point_count)))
        rows.append(
            {
                "predicted_position": pred_index,
                "predicted_index": pred["source_index"],
                "predicted_label": pred["label"],
                "predicted_voxel_points": point_count,
                "nearest_gt_support_points": matched_count,
                "nearest_gt_support_fraction": float(support_fraction),
                "dominant_gt_position": dominant_gt,
                "dominant_gt_index": (
                    ground_truth[dominant_gt]["source_index"] if dominant_gt is not None else None
                ),
                "dominant_gt_instance_id": (
                    ground_truth[dominant_gt]["oracle_gt_id"] if dominant_gt is not None else None
                ),
                "dominant_gt_label": (
                    ground_truth[dominant_gt]["label"] if dominant_gt is not None else None
                ),
                "dominant_points": largest,
                "dominant_fraction_of_prediction": float(purity),
                "second_points": second,
                "second_to_first_ratio": (
                    float(second_to_first) if second_to_first is not None else None
                ),
                "identity_degree_any": int(np.count_nonzero(identity_counts)),
                "identity_degree_share_ge_0p05": int(np.count_nonzero(identity_counts >= threshold_count)),
                "dominant_mean_distance_m": dominant_distance_mean,
                "fross_candidate": fross_candidate,
                "selected_one_to_one": False,
                "selected_gt_coverage": None,
            }
        )

    # FROSS requires one-to-one object correspondence.  Each prediction has at
    # most one candidate owner, so deterministic maximum-quality selection per
    # GT is sufficient and avoids order dependence.
    candidate_order = sorted(
        (index for index, row in enumerate(rows) if row["fross_candidate"]),
        key=lambda index: (
            rows[index]["dominant_fraction_of_prediction"],
            rows[index]["dominant_points"],
            -(rows[index]["dominant_mean_distance_m"] or radius),
            -rows[index]["predicted_index"],
        ),
        reverse=True,
    )
    used_gt: set[int] = set()
    for pred_index in candidate_order:
        gt_index = rows[pred_index]["dominant_gt_position"]
        if gt_index in used_gt:
            continue
        used_gt.add(gt_index)
        rows[pred_index]["selected_one_to_one"] = True
        denominator = len(ground_truth[gt_index]["points"])
        rows[pred_index]["selected_gt_coverage"] = (
            float(coverage_counts[gt_index, pred_index] / denominator) if denominator else None
        )

    per_gt: list[dict] = []
    for gt_index, gt in enumerate(ground_truth):
        candidates = [
            row["predicted_position"]
            for row in rows
            if row["fross_candidate"] and row["dominant_gt_position"] == gt_index
        ]
        selected = [
            row["predicted_position"]
            for row in rows
            if row["selected_one_to_one"] and row["dominant_gt_position"] == gt_index
        ]
        denominator = len(gt["points"])
        candidate_coverage = (
            float(coverage_counts[gt_index, candidates].sum() / denominator)
            if denominator and candidates
            else 0.0
        )
        any_coverage = float(any_covered_counts[gt_index] / denominator) if denominator else 0.0
        per_gt.append(
            {
                "gt_position": gt_index,
                "gt_index": gt["source_index"],
                "gt_instance_id": gt["oracle_gt_id"],
                "gt_label": gt["label"],
                "gt_voxel_points": denominator,
                "fross_candidate_predictions": candidates,
                "fragmentation_degree": len(candidates),
                "selected_prediction": selected[0] if selected else None,
                "candidate_combined_coverage": candidate_coverage,
                "any_prediction_coverage": any_coverage,
            }
        )

    true_positive = sum(bool(row["selected_one_to_one"]) for row in rows)
    false_positive = len(predicted) - true_positive
    false_negative = len(ground_truth) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 1.0
    selected = [row for row in rows if row["selected_one_to_one"]]
    all_purities = [float(row["dominant_fraction_of_prediction"]) for row in rows]
    selected_purities = [float(row["dominant_fraction_of_prediction"]) for row in selected]
    selected_coverages = [
        float(row["selected_gt_coverage"])
        for row in selected
        if row["selected_gt_coverage"] is not None
    ]
    any_coverages = [float(row["any_prediction_coverage"]) for row in per_gt]
    return {
        "predicted_nodes": len(predicted),
        "observable_gt_nodes": len(ground_truth),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / len(predicted) if predicted else (1.0 if not ground_truth else 0.0),
        "recall": true_positive / len(ground_truth) if ground_truth else 1.0,
        "f1": float(f1),
        "candidate_prediction_count_before_one_to_one": sum(bool(row["fross_candidate"]) for row in rows),
        "fragmented_gt_count": sum(row["fragmentation_degree"] > 1 for row in per_gt),
        "contaminated_prediction_count_by_fross_ratio": sum(
            row["second_to_first_ratio"] is not None and row["second_to_first_ratio"] >= 0.75
            for row in rows
        ),
        "mean_prediction_dominant_purity": safe_mean(all_purities),
        "median_prediction_dominant_purity": safe_median(all_purities),
        "mean_selected_purity": safe_mean(selected_purities),
        "median_selected_purity": safe_median(selected_purities),
        "mean_selected_gt_coverage": safe_mean(selected_coverages),
        "median_selected_gt_coverage": safe_median(selected_coverages),
        "mean_gt_any_prediction_coverage": safe_mean(any_coverages),
        "median_gt_any_prediction_coverage": safe_median(any_coverages),
        "predictions": rows,
        "ground_truth": per_gt,
    }


def run_self_test() -> None:
    def obj(index: int, origin: float, count: int = 8) -> dict:
        points = np.stack(
            [origin + np.arange(count) * 0.01, np.zeros(count), np.zeros(count)], axis=1
        ).astype(np.float32)
        return {
            "source_index": index,
            "label": f"o{index}",
            "oracle_gt_id": index,
            "points": points,
        }

    gt = [obj(0, 0.0), obj(1, 2.0)]
    pred = [obj(10, 0.0), obj(11, 0.02), obj(12, 10.0)]
    result = evaluate_prepared(pred, gt, radius=0.1, workers=1)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 2
    assert result["false_negative"] == 1
    assert abs(result["f1"] - 0.4) < 1e-12
    assert result["fragmented_gt_count"] == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-map", type=Path)
    parser.add_argument("--map", action="append", type=parse_named_path)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        print("self-test passed")
        return 0
    if args.gt_map is None or not args.map or args.output is None:
        parser.error("--gt-map, at least one --map, and --output are required")
    if args.voxel_size <= 0 or args.radius <= 0 or args.workers == 0:
        raise ValueError("voxel size/radius must be positive and workers non-zero")

    gt_path = args.gt_map.resolve()
    gt_payload = load_map(gt_path)
    ground_truth = prepare_objects(gt_payload["objects"], args.voxel_size)
    del gt_payload
    gt_ids = [obj["oracle_gt_id"] for obj in ground_truth]
    if any(value is None for value in gt_ids) or len(set(gt_ids)) != len(gt_ids):
        raise ValueError("GT map must contain one unique oracle_gt_id per object")

    results = []
    for name, raw_path in args.map:
        path = raw_path.resolve()
        payload = load_map(path)
        predicted = prepare_objects(payload["objects"], args.voxel_size)
        del payload
        metrics = evaluate_prepared(predicted, ground_truth, args.radius, args.workers)
        metrics.update({"name": name, "map": str(path), "map_sha256": sha256_file(path)})
        results.append(metrics)
        del predicted
        gc.collect()

    report = {
        "schema_version": "1.0.0",
        "protocol": {
            "source": "FROSS ICCV 2025 section 4.1.3",
            "point_preparation": "unique voxel centers per object",
            "voxel_size_m": args.voxel_size,
            "nearest_gt_radius_m": args.radius,
            "candidate_gate": "dominant GT fraction > 0.5 and second/largest < 0.75",
            "object_matching": "deterministic one-to-one after candidate gate",
            "coverage": "GT points whose nearest prediction within radius belongs to the matched prediction",
            "identity_degree_share_threshold": 0.05,
        },
        "gt_map": str(gt_path),
        "gt_map_sha256": sha256_file(gt_path),
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete = output.with_suffix(output.suffix + ".incomplete")
    incomplete.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    incomplete.replace(output)
    print(
        json.dumps(
            {
                row["name"]: {
                    "f1": row["f1"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "mean_selected_gt_coverage": row["mean_selected_gt_coverage"],
                    "mean_selected_purity": row["mean_selected_purity"],
                }
                for row in results
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
