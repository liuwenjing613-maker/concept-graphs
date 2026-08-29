#!/usr/bin/env python3
"""Build a continuous Predicted Object <-> GT Instance audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "oracle"))
from evaluate_geometry_semantics import (  # noqa: E402
    Canonicalizer,
    hungarian_matches,
    iou_matrix,
    load_map,
    prepare_objects,
    sha256_file,
)


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def bbox_iou(left: dict, right: dict) -> float:
    lo = np.maximum(left["lower"], right["lower"])
    hi = np.minimum(left["upper"], right["upper"])
    intersection = int(np.prod(np.maximum(hi - lo + 1, 0)))
    left_volume = int(np.prod(left["upper"] - left["lower"] + 1))
    right_volume = int(np.prod(right["upper"] - right["lower"] + 1))
    union = left_volume + right_volume - intersection
    return intersection / union if union else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prediction-map", type=Path, required=True)
    parser.add_argument("--gt-map", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=Path("/home/chenkejun/beauty/conceptgraphs/code/third_party/ReplicaSSG/files/replica_to_visual_genome.json"),
    )
    args = parser.parse_args()
    if args.voxel_size <= 0 or args.top_k <= 0:
        raise ValueError("voxel size and top-k must be positive")

    prediction_map = args.prediction_map.resolve()
    gt_map = args.gt_map.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    canonicalize = Canonicalizer(args.label_mapping.resolve())
    predicted = prepare_objects(load_map(prediction_map)["objects"], args.voxel_size)
    ground_truth = prepare_objects(load_map(gt_map)["objects"], args.voxel_size)
    gt_ids = [item["oracle_gt_id"] for item in ground_truth]
    if any(item is None for item in gt_ids) or len(gt_ids) != len(set(gt_ids)):
        raise ValueError("GT map must contain one unique oracle_gt_id per node")

    shape = (len(predicted), len(ground_truth))
    intersection = np.zeros(shape, dtype=np.int32)
    purity = np.zeros(shape, dtype=np.float64)
    coverage = np.zeros(shape, dtype=np.float64)
    iou = iou_matrix(predicted, ground_truth)
    box_iou = np.zeros(shape, dtype=np.float64)
    center_distance = np.zeros(shape, dtype=np.float64)
    pred_centers = [np.asarray(list(item["voxels"]), dtype=np.float64).mean(axis=0) for item in predicted]
    gt_centers = [np.asarray(list(item["voxels"]), dtype=np.float64).mean(axis=0) for item in ground_truth]
    overlap_rows = []
    for pi, pred in enumerate(predicted):
        pred_label = canonicalize(pred["label"])
        for gi, gt in enumerate(ground_truth):
            count = len(pred["voxels"] & gt["voxels"])
            intersection[pi, gi] = count
            purity[pi, gi] = count / len(pred["voxels"])
            coverage[pi, gi] = count / len(gt["voxels"])
            box_iou[pi, gi] = bbox_iou(pred, gt)
            center_distance[pi, gi] = float(np.linalg.norm(pred_centers[pi] - gt_centers[gi]) * args.voxel_size)
            if count:
                gt_label = canonicalize(gt["label"])
                overlap_rows.append({
                    "predicted_index": pred["index"], "predicted_label": pred["label"],
                    "gt_index": gt["index"], "gt_instance_id": gt["oracle_gt_id"], "gt_label": gt["label"],
                    "intersection_voxels": count, "predicted_voxels": len(pred["voxels"]), "gt_voxels": len(gt["voxels"]),
                    "purity": purity[pi, gi], "coverage": coverage[pi, gi], "voxel_iou": iou[pi, gi],
                    "bbox_iou": box_iou[pi, gi], "center_distance_m": center_distance[pi, gi],
                    "semantic_eligible": gt_label != "unknown", "semantic_consistent": gt_label != "unknown" and pred_label == gt_label,
                })

    pred_rows = []
    for pi, pred in enumerate(predicted):
        gi = int(np.argmax(iou[pi])) if ground_truth and float(iou[pi].max()) > 0 else None
        best_purity = float(purity[pi].max()) if ground_truth else 0.0
        pred_rows.append({
            "predicted_index": pred["index"], "predicted_label": pred["label"], "predicted_voxels": len(pred["voxels"]),
            "best_gt_instance_id": ground_truth[gi]["oracle_gt_id"] if gi is not None else "",
            "best_gt_label": ground_truth[gi]["label"] if gi is not None else "", "best_voxel_iou": float(iou[pi, gi]) if gi is not None else 0.0,
            "purity_at_best_iou": float(purity[pi, gi]) if gi is not None else 0.0, "coverage_at_best_iou": float(coverage[pi, gi]) if gi is not None else 0.0,
            "maximum_purity": best_purity, "overlap_gt_count": int((intersection[pi] > 0).sum()),
            "overlap_gt_count_purity_0p01": int((purity[pi] >= 0.01).sum()), "overlap_gt_count_purity_0p05": int((purity[pi] >= 0.05).sum()),
        })

    gt_rows = []
    for gi, gt in enumerate(ground_truth):
        pi = int(np.argmax(iou[:, gi])) if predicted and float(iou[:, gi].max()) > 0 else None
        gt_rows.append({
            "gt_index": gt["index"], "gt_instance_id": gt["oracle_gt_id"], "gt_label": gt["label"], "gt_voxels": len(gt["voxels"]),
            "best_predicted_index": predicted[pi]["index"] if pi is not None else "", "best_predicted_label": predicted[pi]["label"] if pi is not None else "",
            "best_voxel_iou": float(iou[pi, gi]) if pi is not None else 0.0, "maximum_coverage": float(coverage[:, gi].max()) if predicted else 0.0,
            "overlap_prediction_count": int((intersection[:, gi] > 0).sum()), "overlap_prediction_count_coverage_0p01": int((coverage[:, gi] >= 0.01).sum()),
            "overlap_prediction_count_coverage_0p05": int((coverage[:, gi] >= 0.05).sum()),
        })

    positive_matches = hungarian_matches(iou, 1e-12)
    match_rows = []
    for pi, gi, value in positive_matches:
        pred_label = canonicalize(predicted[pi]["label"])
        gt_label = canonicalize(ground_truth[gi]["label"])
        match_rows.append({
            "predicted_index": predicted[pi]["index"], "gt_instance_id": ground_truth[gi]["oracle_gt_id"], "voxel_iou": value,
            "purity": float(purity[pi, gi]), "coverage": float(coverage[pi, gi]), "bbox_iou": float(box_iou[pi, gi]),
            "center_distance_m": float(center_distance[pi, gi]), "semantic_eligible": gt_label != "unknown",
            "semantic_consistent": gt_label != "unknown" and pred_label == gt_label,
        })
    thresholds = {}
    for threshold in (0.10, 0.25, 0.50):
        matches = hungarian_matches(iou, threshold)
        tp = len(matches)
        thresholds[str(threshold)] = {
            "true_positive": tp, "precision": tp / len(predicted) if predicted else None,
            "recall": tp / len(ground_truth) if ground_truth else None,
            "f1": 2 * tp / (len(predicted) + len(ground_truth)) if predicted or ground_truth else 1.0,
        }

    pred_purity = np.asarray([row["maximum_purity"] for row in pred_rows])
    gt_coverage = np.asarray([row["maximum_coverage"] for row in gt_rows])
    report = {
        "schema_version": "1.0.0", "scene": args.scene, "voxel_size_m": args.voxel_size,
        "prediction_map": str(prediction_map), "prediction_map_sha256": sha256_file(prediction_map),
        "gt_map": str(gt_map), "gt_map_sha256": sha256_file(gt_map),
        "protocol": {"gt_reference": "O3 observable surfaces from identical online frames/depth/poses", "matching": "class-agnostic Hungarian on voxel IoU", "threshold_policy": "continuous distributions plus IoU sensitivity; no single error threshold"},
        "counts": {"predicted_objects": len(predicted), "observable_gt_instances": len(ground_truth), "positive_overlap_pairs": len(overlap_rows), "positive_hungarian_matches": len(positive_matches)},
        "object_metrics_by_iou_threshold": thresholds,
        "distributions": {"best_iou_per_prediction": stats(iou.max(axis=1) if ground_truth else np.array([])), "maximum_purity_per_prediction": stats(pred_purity), "maximum_coverage_per_gt": stats(gt_coverage), "best_iou_per_gt": stats(iou.max(axis=0) if predicted else np.array([]))},
        "tails": {
            "worst_prediction_purity": sorted(pred_rows, key=lambda row: (row["maximum_purity"], row["predicted_index"]))[:args.top_k],
            "worst_gt_coverage": sorted(gt_rows, key=lambda row: (row["maximum_coverage"], row["gt_instance_id"]))[:args.top_k],
            "most_contaminated_predictions": sorted(pred_rows, key=lambda row: (-row["overlap_gt_count_purity_0p01"], row["maximum_purity"]))[:args.top_k],
            "most_fragmented_gt": sorted(gt_rows, key=lambda row: (-row["overlap_prediction_count_coverage_0p01"], row["maximum_coverage"]))[:args.top_k],
        },
    }
    fields = list(overlap_rows[0]) if overlap_rows else []
    write_csv(output / "object_gt_overlaps.csv", overlap_rows, fields)
    write_csv(output / "predicted_object_summary.csv", pred_rows, list(pred_rows[0]))
    write_csv(output / "gt_instance_summary.csv", gt_rows, list(gt_rows[0]))
    write_csv(output / "hungarian_matches.csv", match_rows, list(match_rows[0]) if match_rows else [])
    matrix_tmp = output / "overlap_matrices.npz.incomplete"
    with matrix_tmp.open("wb") as handle:
        np.savez_compressed(handle, intersection=intersection, purity=purity, coverage=coverage, iou=iou, bbox_iou=box_iou, center_distance_m=center_distance)
    matrix_tmp.replace(output / "overlap_matrices.npz")
    report_tmp = output / "audit_summary.json.incomplete"
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_tmp.replace(output / "audit_summary.json")
    print(json.dumps({"scene": args.scene, "voxel_size_m": args.voxel_size, "counts": report["counts"], "thresholds": thresholds, "purity": report["distributions"]["maximum_purity_per_prediction"], "coverage": report["distributions"]["maximum_coverage_per_gt"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
