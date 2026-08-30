#!/usr/bin/env python3
"""Evaluate a simple class-query -> localized-instance retrieval proxy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.oracle.evaluate_geometry_semantics import (  # noqa: E402
    Canonicalizer,
    hungarian_matches,
    iou_matrix,
    load_map,
    parse_named_path,
    prepare_objects,
    sha256_file,
)


def voxel_centroid(item: dict, voxel_size: float) -> np.ndarray:
    return np.asarray(list(item["voxels"]), dtype=np.float64).mean(axis=0) * voxel_size


def evaluate_one(
    *,
    name: str,
    path: Path,
    ground_truth: list[dict],
    voxel_size: float,
    threshold: float,
    canonicalize: Canonicalizer,
) -> dict:
    payload = load_map(path.resolve())
    predicted = prepare_objects(payload["objects"], voxel_size)
    matrix = iou_matrix(predicted, ground_truth)
    pred_labels = [canonicalize(item["label"]) for item in predicted]
    gt_labels = [canonicalize(item["label"]) for item in ground_truth]
    eligible_pred = [index for index, label in enumerate(pred_labels) if label != "unknown"]
    eligible_gt = [index for index, label in enumerate(gt_labels) if label != "unknown"]

    semantic_matrix = np.zeros_like(matrix)
    for pred_index in eligible_pred:
        for gt_index in eligible_gt:
            if pred_labels[pred_index] == gt_labels[gt_index]:
                semantic_matrix[pred_index, gt_index] = matrix[pred_index, gt_index]
    matches = hungarian_matches(semantic_matrix, threshold)
    true_positive = len(matches)
    precision = true_positive / len(eligible_pred) if eligible_pred else None
    recall = true_positive / len(eligible_gt) if eligible_gt else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision is not None and recall is not None else None)
    )

    gt_label_counts = Counter(gt_labels[index] for index in eligible_gt)
    unique_queries = [
        gt_index
        for gt_index in eligible_gt
        if gt_label_counts[gt_labels[gt_index]] == 1
    ]
    query_rows = []
    for gt_index in unique_queries:
        label = gt_labels[gt_index]
        candidates = [
            pred_index
            for pred_index in eligible_pred
            if pred_labels[pred_index] == label
        ]
        ranked_candidates = sorted(
            candidates,
            key=lambda index: (
                predicted[index]["score"],
                predicted[index]["num_detections"],
                -predicted[index]["index"],
            ),
            reverse=True,
        )
        if candidates:
            top = ranked_candidates[0]
            overlap = float(matrix[top, gt_index])
            success = overlap >= threshold
            top1_centroid_error = float(
                np.linalg.norm(
                    voxel_centroid(predicted[top], voxel_size)
                    - voxel_centroid(ground_truth[gt_index], voxel_size)
                )
            )
            centroid_error = top1_centroid_error if success else None
            predicted_index = predicted[top]["index"]
            top3_overlaps = [float(matrix[index, gt_index]) for index in ranked_candidates[:3]]
            top3_success = any(value >= threshold for value in top3_overlaps)
        else:
            overlap = 0.0
            success = False
            centroid_error = None
            top1_centroid_error = None
            predicted_index = None
            top3_overlaps = []
            top3_success = False
        query_rows.append(
            {
                "canonical_query_label": label,
                "gt_index": ground_truth[gt_index]["index"],
                "gt_instance_id": ground_truth[gt_index]["oracle_gt_id"],
                "candidate_count": len(candidates),
                "top_predicted_index": predicted_index,
                "top_voxel_iou": overlap,
                "top1_success": success,
                "top3_voxel_ious": top3_overlaps,
                "top3_success": top3_success,
                "top1_centroid_error_m": top1_centroid_error,
                "top1_within_1m": bool(
                    top1_centroid_error is not None and top1_centroid_error <= 1.0
                ),
                "successful_centroid_error_m": centroid_error,
            }
        )
    successful_errors = [
        row["successful_centroid_error_m"]
        for row in query_rows
        if row["successful_centroid_error_m"] is not None
    ]
    top1_candidate_errors = [
        row["top1_centroid_error_m"]
        for row in query_rows
        if row["top1_centroid_error_m"] is not None
    ]
    pred_label_counts = Counter(pred_labels[index] for index in eligible_pred)
    count_labels = sorted(set(gt_label_counts) | set(pred_label_counts))
    count_rows = [
        {
            "canonical_label": label,
            "gt_count": int(gt_label_counts[label]),
            "predicted_count": int(pred_label_counts[label]),
            "absolute_error": abs(int(gt_label_counts[label]) - int(pred_label_counts[label])),
        }
        for label in count_labels
    ]
    return {
        "name": name,
        "map": str(path.resolve()),
        "map_sha256": sha256_file(path.resolve()),
        "voxel_size_m": voxel_size,
        "iou_threshold": threshold,
        "predicted_nodes": len(predicted),
        "semantic_evaluable_predicted_nodes": len(eligible_pred),
        "semantic_evaluable_gt_nodes": len(eligible_gt),
        "class_conditioned_true_positive": true_positive,
        "class_conditioned_precision": precision,
        "class_conditioned_recall": recall,
        "class_conditioned_f1": f1,
        "unique_class_query_count": len(query_rows),
        "unique_class_top1_success_count": sum(row["top1_success"] for row in query_rows),
        "unique_class_top1_success_rate": (
            sum(row["top1_success"] for row in query_rows) / len(query_rows)
            if query_rows
            else None
        ),
        "unique_class_top3_success_count": sum(row["top3_success"] for row in query_rows),
        "unique_class_top3_success_rate": (
            sum(row["top3_success"] for row in query_rows) / len(query_rows)
            if query_rows
            else None
        ),
        "unique_class_top1_within_1m_count": sum(row["top1_within_1m"] for row in query_rows),
        "unique_class_top1_within_1m_rate": (
            sum(row["top1_within_1m"] for row in query_rows) / len(query_rows)
            if query_rows
            else None
        ),
        "unique_class_top1_candidate_coverage": (
            len(top1_candidate_errors) / len(query_rows) if query_rows else None
        ),
        "unique_class_top1_candidate_mean_centroid_error_m": (
            float(np.mean(top1_candidate_errors)) if top1_candidate_errors else None
        ),
        "successful_query_mean_centroid_error_m": (
            float(np.mean(successful_errors)) if successful_errors else None
        ),
        "semantic_class_count_mae": (
            float(np.mean([row["absolute_error"] for row in count_rows])) if count_rows else None
        ),
        "semantic_class_exact_count_rate": (
            sum(row["absolute_error"] == 0 for row in count_rows) / len(count_rows)
            if count_rows
            else None
        ),
        "semantic_class_counts": count_rows,
        "class_conditioned_matches": [
            {
                "predicted_index": predicted[pred_index]["index"],
                "gt_index": ground_truth[gt_index]["index"],
                "canonical_label": pred_labels[pred_index],
                "voxel_iou": overlap,
            }
            for pred_index, gt_index, overlap in matches
        ],
        "unique_class_queries": query_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-map", type=Path, required=True)
    parser.add_argument("--map", action="append", type=parse_named_path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gt_payload = load_map(args.gt_map.resolve())
    ground_truth = prepare_objects(gt_payload["objects"], args.voxel_size)
    canonicalize = Canonicalizer(args.label_mapping.resolve())
    results = [
        evaluate_one(
            name=name,
            path=path,
            ground_truth=ground_truth,
            voxel_size=args.voxel_size,
            threshold=args.iou_threshold,
            canonicalize=canonicalize,
        )
        for name, path in args.map
    ]
    payload = {
        "schema_version": "1.0.0",
        "evaluation_role": (
            "proxy for class query to localized object; not an official relation metric"
        ),
        "scene_is_independent_unit": True,
        "voxel_size_m": args.voxel_size,
        "iou_threshold": args.iou_threshold,
        "gt_map": str(args.gt_map.resolve()),
        "gt_map_sha256": sha256_file(args.gt_map.resolve()),
        "label_mapping_sha256": sha256_file(args.label_mapping.resolve()),
        "ranking": "highest max detection confidence, then detection count, then index",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            [
                {
                    "name": row["name"],
                    "class_conditioned_f1": row["class_conditioned_f1"],
                    "unique_class_top1_success_rate": row[
                        "unique_class_top1_success_rate"
                    ],
                }
                for row in results
            ],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
