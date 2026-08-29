#!/usr/bin/env python3
"""Evaluate observable-surface instance, node, and semantic oracle metrics."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ALIASES = {
    "arm chair": "chair",
    "armchair": "chair",
    "blinds": "curtain",
    "closet door": "door",
    "couch": "chair",
    "end table": "table",
    "coffee table": "table",
    "dining table": "table",
    "paper bag": "bag",
    "potted plant": "plant",
    "sofa": "chair",
    "sofa chair": "chair",
    "stool": "chair",
    "television": "screen",
    "tv": "screen",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_map(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("map must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("map must be NAME=PATH")
    return name, Path(raw_path)


def normalize_label(label: object) -> str:
    return " ".join(
        str(label).strip().lower().replace("_", " ").replace("-", " ").split()
    )


class Canonicalizer:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.replica = {
            normalize_label(source): str(target).lower()
            for source, target in payload["Replica2VisualGenome"].items()
        }
        self.visual_genome = {
            normalize_label(item) for item in payload["VisualGenome_list"]
        }

    def __call__(self, label: object) -> str:
        normalized = normalize_label(label)
        if normalized in ALIASES:
            return ALIASES[normalized]
        if normalized in self.replica:
            return self.replica[normalized]
        if normalized in self.visual_genome:
            return normalized
        return "unknown"


def voxel_set(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    quantized = np.floor(np.asarray(points, dtype=np.float64) / voxel_size).astype(
        np.int32
    )
    return set(map(tuple, np.unique(quantized, axis=0).tolist()))


def voxel_bounds(voxels: set[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(list(voxels), dtype=np.int32)
    return values.min(axis=0), values.max(axis=0)


def prepare_objects(objects: list[dict], voxel_size: float) -> list[dict]:
    prepared = []
    for index, obj in enumerate(objects):
        voxels = voxel_set(np.asarray(obj["pcd_np"]), voxel_size)
        if not voxels:
            continue
        lower, upper = voxel_bounds(voxels)
        confidences = [float(item) for item in obj.get("conf", [])]
        prepared.append(
            {
                "index": index,
                "label": str(obj.get("class_name", "unknown")),
                "oracle_gt_id": obj.get("oracle_gt_id"),
                "num_detections": int(obj.get("num_detections", 0)),
                "score": max(confidences) if confidences else 1.0,
                "voxels": voxels,
                "lower": lower,
                "upper": upper,
            }
        )
    return prepared


def iou_matrix(predicted: list[dict], ground_truth: list[dict]) -> np.ndarray:
    matrix = np.zeros((len(predicted), len(ground_truth)), dtype=np.float64)
    for pred_index, pred in enumerate(predicted):
        for gt_index, gt in enumerate(ground_truth):
            if np.any(pred["upper"] < gt["lower"]) or np.any(gt["upper"] < pred["lower"]):
                continue
            intersection = len(pred["voxels"] & gt["voxels"])
            if intersection == 0:
                continue
            union = len(pred["voxels"]) + len(gt["voxels"]) - intersection
            matrix[pred_index, gt_index] = intersection / union
    return matrix


def hungarian_matches(matrix: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    if matrix.size == 0:
        return []
    rows, columns = linear_sum_assignment(-matrix)
    return [
        (int(row), int(column), float(matrix[row, column]))
        for row, column in zip(rows.tolist(), columns.tolist())
        if matrix[row, column] >= threshold
    ]


def interpolated_ap(
    matrix: np.ndarray,
    predicted: list[dict],
    ground_truth_count: int,
    threshold: float,
) -> dict:
    order = sorted(
        range(len(predicted)),
        key=lambda index: (
            predicted[index]["score"],
            predicted[index]["num_detections"],
            -predicted[index]["index"],
        ),
        reverse=True,
    )
    used_gt: set[int] = set()
    true_positive = []
    false_positive = []
    for pred_index in order:
        if ground_truth_count == 0:
            best_gt, best_iou = None, 0.0
        else:
            ranked = np.argsort(matrix[pred_index])[::-1]
            best_gt, best_iou = None, 0.0
            for gt_index in ranked.tolist():
                if gt_index not in used_gt:
                    best_gt = int(gt_index)
                    best_iou = float(matrix[pred_index, gt_index])
                    break
        is_true = best_gt is not None and best_iou >= threshold
        true_positive.append(1 if is_true else 0)
        false_positive.append(0 if is_true else 1)
        if is_true:
            used_gt.add(best_gt)
    if ground_truth_count == 0:
        return {"ap": None, "true_positive": 0, "false_positive": len(order)}
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / ground_truth_count
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    samples = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= recall_level]
        samples.append(float(eligible.max()) if eligible.size else 0.0)
    return {
        "ap": float(np.mean(samples)),
        "true_positive": int(cumulative_tp[-1]) if len(cumulative_tp) else 0,
        "false_positive": int(cumulative_fp[-1]) if len(cumulative_fp) else 0,
    }


def evaluate_one(
    name: str,
    path: Path,
    gt_prepared: list[dict],
    voxel_size: float,
    node_iou_threshold: float,
    canonicalize: Canonicalizer,
) -> dict:
    payload = load_map(path)
    predicted = prepare_objects(payload["objects"], voxel_size)
    matrix = iou_matrix(predicted, gt_prepared)
    matches = hungarian_matches(matrix, node_iou_threshold)
    true_positive = len(matches)
    false_positive = len(predicted) - true_positive
    false_negative = len(gt_prepared) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    node_f1 = 2 * true_positive / denominator if denominator else 1.0

    semantic_rows = []
    semantic_correct = 0
    semantic_denominator = 0
    for pred_index, gt_index, iou in matches:
        pred = predicted[pred_index]
        gt = gt_prepared[gt_index]
        pred_label = canonicalize(pred["label"])
        gt_label = canonicalize(gt["label"])
        eligible = gt_label != "unknown"
        correct = eligible and pred_label == gt_label
        if eligible:
            semantic_denominator += 1
            semantic_correct += int(correct)
        semantic_rows.append(
            {
                "predicted_index": pred["index"],
                "predicted_label": pred["label"],
                "predicted_canonical_label": pred_label,
                "gt_index": gt["index"],
                "gt_instance_id": gt["oracle_gt_id"],
                "gt_label": gt["label"],
                "gt_canonical_label": gt_label,
                "voxel_iou": iou,
                "semantic_eligible": eligible,
                "semantic_correct": bool(correct),
            }
        )
    ap25 = interpolated_ap(matrix, predicted, len(gt_prepared), 0.25)
    ap50 = interpolated_ap(matrix, predicted, len(gt_prepared), 0.50)
    instance_ap = (
        (ap25["ap"] + ap50["ap"]) / 2
        if ap25["ap"] is not None and ap50["ap"] is not None
        else None
    )
    max_iou = matrix.max(axis=1) if len(gt_prepared) and len(predicted) else np.array([])
    return {
        "name": name,
        "map": str(path.resolve()),
        "map_sha256": sha256_file(path.resolve()),
        "predicted_nodes": len(predicted),
        "observable_gt_nodes": len(gt_prepared),
        "node_iou_threshold": node_iou_threshold,
        "node_true_positive": true_positive,
        "node_false_positive": false_positive,
        "node_false_negative": false_negative,
        "node_f1": float(node_f1),
        "ap25": ap25,
        "ap50": ap50,
        "instance_ap_mean_25_50": instance_ap,
        "semantic_correct": semantic_correct,
        "semantic_denominator": semantic_denominator,
        "semantic_accuracy": (
            semantic_correct / semantic_denominator
            if semantic_denominator
            else None
        ),
        "semantic_evaluable_match_fraction": (
            semantic_denominator / len(matches) if matches else None
        ),
        "mean_best_voxel_iou_per_prediction": (
            float(max_iou.mean()) if max_iou.size else None
        ),
        "matches": semantic_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-map", type=Path, required=True)
    parser.add_argument("--map", action="append", type=parse_named_path, required=True)
    parser.add_argument(
        "--label-mapping",
        type=Path,
        default=Path(
            "/home/chenkejun/beauty/conceptgraphs/code/third_party/"
            "ReplicaSSG/files/replica_to_visual_genome.json"
        ),
    )
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--node-iou-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.voxel_size <= 0:
        raise ValueError("voxel size must be positive")

    canonicalize = Canonicalizer(args.label_mapping.resolve())
    gt_payload = load_map(args.gt_map.resolve())
    gt_prepared = prepare_objects(gt_payload["objects"], args.voxel_size)
    gt_ids = [item["oracle_gt_id"] for item in gt_prepared]
    if any(item is None for item in gt_ids) or len(set(gt_ids)) != len(gt_ids):
        raise ValueError("O3 must contain exactly one node per observable GT instance")

    results = [
        evaluate_one(
            name,
            path.resolve(),
            gt_prepared,
            args.voxel_size,
            args.node_iou_threshold,
            canonicalize,
        )
        for name, path in args.map
    ]
    report = {
        "schema_version": "1.0.0",
        "protocol": {
            "gt_reference": "O3 observable surfaces from identical frames/depth/poses",
            "class_agnostic_instance_ap": "101-point interpolated AP",
            "ap_iou_thresholds": [0.25, 0.50],
            "instance_ap_primary": "arithmetic mean of AP25 and AP50",
            "node_matching": "one-to-one Hungarian on voxel IoU",
            "node_iou_threshold": args.node_iou_threshold,
            "voxel_size_m": args.voxel_size,
            "prediction_score": "maximum source detection confidence; ties by view count",
            "semantic_vocabulary": "ReplicaSSG labels canonicalized to Visual Genome",
            "semantic_unknown_gt_policy": "exclude from semantic denominator",
        },
        "gt_map": str(args.gt_map.resolve()),
        "gt_map_sha256": sha256_file(args.gt_map.resolve()),
        "observable_gt_nodes": len(gt_prepared),
        "results": results,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.resolve().with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output.resolve())
    print(
        json.dumps(
            {
                row["name"]: {
                    "instance_ap_mean_25_50": row["instance_ap_mean_25_50"],
                    "node_f1": row["node_f1"],
                    "semantic_accuracy": row["semantic_accuracy"],
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
