#!/usr/bin/env python3
"""Locate whether frozen zero-coverage GT cases disappear before or after mapping."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        scene, path = value.split("=", 1)
        result[scene] = Path(path)
    return result


def load_map(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def frame_masks(objects: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for object_index, obj in enumerate(objects):
        for image_idx, mask in zip(obj.get("image_idx", []), obj.get("mask", [])):
            result[int(image_idx)].append(
                {
                    "source_index": object_index,
                    "label": str(obj.get("class_name") or obj.get("predicted_class_name") or ""),
                    "mask": np.asarray(mask, dtype=bool),
                }
            )
    return result


def overlap(gt_mask: np.ndarray, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    gt = np.asarray(gt_mask, dtype=bool)
    gt_area = int(gt.sum())
    best = {
        "source_index": None,
        "label": None,
        "intersection": 0,
        "gt_recall": 0.0,
        "candidate_purity": 0.0,
        "iou": 0.0,
    }
    for candidate in candidates:
        mask = candidate["mask"]
        if mask.shape != gt.shape:
            continue
        intersection = int(np.logical_and(gt, mask).sum())
        if intersection <= 0:
            continue
        candidate_area = int(mask.sum())
        union = gt_area + candidate_area - intersection
        metrics = {
            "source_index": candidate["source_index"],
            "label": candidate["label"],
            "intersection": intersection,
            "gt_recall": intersection / gt_area if gt_area else 0.0,
            "candidate_purity": intersection / candidate_area if candidate_area else 0.0,
            "iou": intersection / union if union else 0.0,
        }
        if (metrics["iou"], metrics["gt_recall"]) > (best["iou"], best["gt_recall"]):
            best = metrics
    return best


def raw_candidates(root: Path | None, raw_frame: int) -> list[dict[str, Any]] | None:
    if root is None:
        return None
    folder = root / f"frame{raw_frame:06d}"
    mask_path = folder / "mask.npz"
    labels_path = folder / "detection_class_labels.pkl.gz"
    if not mask_path.is_file() or not labels_path.is_file():
        return None
    masks = np.load(mask_path)["arr_0"]
    with gzip.open(labels_path, "rb") as handle:
        labels = pickle.load(handle)
    return [
        {"source_index": index, "label": labels[index] if index < len(labels) else "", "mask": mask}
        for index, mask in enumerate(masks)
    ]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "observation_count": len(rows),
        "best_iou": distribution([row["iou"] for row in rows]),
        "best_gt_recall": distribution([row["gt_recall"] for row in rows]),
        "best_candidate_purity": distribution([row["candidate_purity"] for row in rows]),
        "support_rate_iou_0p10": sum(row["iou"] >= 0.10 for row in rows) / len(rows) if rows else None,
        "support_rate_recall_0p50": sum(row["gt_recall"] >= 0.50 for row in rows) / len(rows) if rows else None,
        "top_labels": sorted(
            (
                {"label": label, "count": sum(row["label"] == label for row in rows)}
                for label in {row["label"] for row in rows if row["label"]}
            ),
            key=lambda item: (-item["count"], item["label"]),
        )[:5],
    }


def classify(raw: dict[str, Any] | None, retained: dict[str, Any]) -> str:
    retained_rate = retained["support_rate_iou_0p10"] or 0.0
    if raw is None:
        return "detection_or_graph_retention_unresolved"
    raw_rate = raw["support_rate_iou_0p10"] or 0.0
    if raw_rate < 0.20:
        return "detector_or_mask_miss_dominant"
    if raw_rate >= 0.50 and retained_rate < 0.20:
        return "graph_ingestion_association_or_retention_loss_dominant"
    return "mixed_detection_and_mapping_loss"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-cases", required=True, type=Path)
    parser.add_argument("--b0-map", required=True, action="append")
    parser.add_argument("--o3-map", required=True, action="append")
    parser.add_argument("--raw-detections", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frozen = json.loads(args.frozen_cases.read_text(encoding="utf-8"))
    geometry_cases = [case for case in frozen if case["family"] == "geometry"]
    b0_paths = named_paths(args.b0_map)
    o3_paths = named_paths(args.o3_map)
    raw_roots = named_paths(args.raw_detections)
    reports = []
    for scene in sorted({case["scene_id"] for case in geometry_cases}):
        b0 = load_map(b0_paths[scene])
        o3 = load_map(o3_paths[scene])
        retained_by_frame = frame_masks(b0["objects"])
        gt_objects = {
            int(obj["oracle_gt_id"]): obj
            for obj in o3["objects"]
            if obj.get("oracle_gt_id") is not None
        }
        for case in [item for item in geometry_cases if item["scene_id"] == scene]:
            gt_obj = gt_objects[int(case["gt_instance_id"])]
            retained_rows = []
            raw_rows = []
            raw_available = True
            i1_frame = int(case["s_processed_frame"])
            i1 = {"processed_frame": i1_frame, "raw_frame": i1_frame * 5}
            for image_idx, mask in zip(gt_obj.get("image_idx", []), gt_obj.get("mask", [])):
                processed = int(image_idx)
                retained = overlap(mask, retained_by_frame.get(processed, []))
                retained_rows.append({"processed_frame": processed, **retained})
                raw = raw_candidates(raw_roots.get(scene), processed * 5)
                if raw is None:
                    raw_available = False
                else:
                    raw_rows.append({"processed_frame": processed, **overlap(mask, raw)})
                if processed == i1_frame:
                    i1["retained_best"] = retained
                    i1["raw_best"] = None if raw is None else overlap(mask, raw)
            retained_summary = summarize(retained_rows)
            raw_summary = summarize(raw_rows) if raw_available and raw_rows else None
            reports.append(
                {
                    "case_id": case["case_id"],
                    "scene_id": scene,
                    "gt_instance_id": case["gt_instance_id"],
                    "gt_label": case["gt_label"],
                    "o3_observation_count_frozen": case["o3_observation_count"],
                    "retained_b0_observation_overlap": retained_summary,
                    "raw_detector_observation_overlap": raw_summary,
                    "i1": i1,
                    "root_cause_classification": classify(raw_summary, retained_summary),
                }
            )
        del b0, o3, retained_by_frame, gt_objects
        gc.collect()

    output = {
        "schema_version": "1.0.0",
        "protocol": {
            "unit": "O3 GT mask observation",
            "retained_stage": "all observations retained by final B0 graph nodes",
            "raw_stage": "saved per-frame B0 detector masks when available",
            "support_threshold": "best pixel IoU >= 0.10",
            "human_labels_used_as_truth": False,
        },
        "cases": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
