#!/usr/bin/env python3
"""Build geometry-preserving label-only oracles with explicit match strictness."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.oracle.evaluate_geometry_semantics import (
    Canonicalizer,
    hungarian_matches,
    iou_matrix,
    prepare_objects,
    sha256_file,
)


def load_map(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def atomic_pickle(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with gzip.open(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def geometry_hash(objects: list[dict]) -> str:
    digest = hashlib.sha256()
    for obj in objects:
        digest.update(np.asarray(obj["pcd_np"], dtype=np.float32).tobytes())
        digest.update(np.asarray(obj["bbox_np"], dtype=np.float32).tobytes())
        digest.update(str(int(obj.get("num_detections", 0))).encode("utf-8"))
        for color_path, mask_idx in zip(obj.get("color_path", []), obj.get("mask_idx", [])):
            digest.update(f"{Path(str(color_path)).stem}:{int(mask_idx)}\n".encode("utf-8"))
    return digest.hexdigest()


def degree_counts(
    predicted: list[dict], ground_truth: list[dict], threshold: float
) -> tuple[list[int], list[int]]:
    pred_degree = []
    for pred in predicted:
        pred_degree.append(
            sum(
                len(pred["voxels"] & gt["voxels"]) / len(pred["voxels"]) >= threshold
                for gt in ground_truth
            )
        )
    gt_degree = []
    for gt in ground_truth:
        gt_degree.append(
            sum(
                len(pred["voxels"] & gt["voxels"]) / len(gt["voxels"]) >= threshold
                for pred in predicted
            )
        )
    return pred_degree, gt_degree


def apply_labels(
    payload: dict,
    predicted: list[dict],
    ground_truth: list[dict],
    matches: list[tuple[int, int, float]],
    allowed: set[tuple[int, int]],
    canonicalize: Canonicalizer,
) -> tuple[dict, list[dict]]:
    output = copy.deepcopy(payload)
    rows = []
    for pred_index, gt_index, iou in matches:
        pred = predicted[pred_index]
        gt = ground_truth[gt_index]
        source_label = str(output["objects"][pred["index"]].get("class_name", "unknown"))
        target_label = str(gt["label"])
        applied = (pred_index, gt_index) in allowed
        if applied:
            output["objects"][pred["index"]]["class_name"] = target_label
            output["objects"][pred["index"]]["semantic_only_oracle"] = True
            output["objects"][pred["index"]]["semantic_only_source_label"] = source_label
            output["objects"][pred["index"]]["semantic_only_gt_id"] = gt["oracle_gt_id"]
        rows.append(
            {
                "predicted_index": pred["index"],
                "gt_index": gt["index"],
                "gt_id": gt["oracle_gt_id"],
                "voxel_iou": iou,
                "source_label": source_label,
                "source_canonical": canonicalize(source_label),
                "target_label": target_label,
                "target_canonical": canonicalize(target_label),
                "semantic_evaluable": canonicalize(target_label) != "unknown",
                "label_was_wrong": (
                    canonicalize(target_label) != "unknown"
                    and canonicalize(source_label) != canonicalize(target_label)
                ),
                "oracle_applied": applied,
            }
        )
    return output, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--gt-map", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--match-iou-threshold", type=float, default=0.25)
    parser.add_argument("--degree-threshold", type=float, default=0.10)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "INCOMPLETE").write_text("building\n", encoding="utf-8")

    baseline = load_map(args.baseline_map.resolve())
    gt_payload = load_map(args.gt_map.resolve())
    predicted = prepare_objects(baseline["objects"], args.voxel_size)
    ground_truth = prepare_objects(gt_payload["objects"], args.voxel_size)
    matrix = iou_matrix(predicted, ground_truth)
    matches = hungarian_matches(matrix, args.match_iou_threshold)
    pred_degree, gt_degree = degree_counts(predicted, ground_truth, args.degree_threshold)
    strict_pairs = {
        (pred_index, gt_index)
        for pred_index, gt_index, _ in matches
        if pred_degree[pred_index] == 1 and gt_degree[gt_index] == 1
    }
    hungarian_pairs = {(pred_index, gt_index) for pred_index, gt_index, _ in matches}
    canonicalize = Canonicalizer(args.label_mapping.resolve())

    source_geometry = geometry_hash(baseline["objects"])
    variants = {}
    for name, allowed in (("os_strict", strict_pairs), ("os_hungarian", hungarian_pairs)):
        mapped, rows = apply_labels(
            baseline, predicted, ground_truth, matches, allowed, canonicalize
        )
        if geometry_hash(mapped["objects"]) != source_geometry:
            raise RuntimeError(f"{name} changed geometry/provenance")
        mapped["oracle_condition"] = name
        map_path = output / f"pcd_{name}.pkl.gz"
        atomic_pickle(map_path, mapped)
        applied_rows = [row for row in rows if row["oracle_applied"]]
        variants[name] = {
            "map": str(map_path),
            "map_sha256": sha256_file(map_path),
            "geometry_sha256": source_geometry,
            "applied_match_count": len(applied_rows),
            "applied_wrong_label_count": sum(row["label_was_wrong"] for row in applied_rows),
            "matched_rows": rows,
        }

    manifest = {
        "schema_version": "1.0.0",
        "baseline_map": str(args.baseline_map.resolve()),
        "baseline_map_sha256": sha256_file(args.baseline_map.resolve()),
        "gt_map": str(args.gt_map.resolve()),
        "gt_map_sha256": sha256_file(args.gt_map.resolve()),
        "label_mapping": str(args.label_mapping.resolve()),
        "label_mapping_sha256": sha256_file(args.label_mapping.resolve()),
        "voxel_size_m": args.voxel_size,
        "match_iou_threshold": args.match_iou_threshold,
        "degree_threshold": args.degree_threshold,
        "matching": "one-to-one Hungarian on voxel IoU",
        "strict_definition": (
            "Hungarian match plus predicted identity degree=1 and GT fragmentation "
            "degree=1 at the degree threshold"
        ),
        "predicted_nodes": len(predicted),
        "gt_nodes": len(ground_truth),
        "hungarian_match_count": len(matches),
        "strict_match_count": len(strict_pairs),
        "geometry_and_provenance_unchanged": True,
        "variants": variants,
    }
    atomic_json(output / "manifest.json", manifest)
    (output / "INCOMPLETE").unlink(missing_ok=True)
    (output / "READY").write_text("ready\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "hungarian_match_count": len(matches),
                "strict_match_count": len(strict_pairs),
                "os_hungarian_applied_wrong": variants["os_hungarian"][
                    "applied_wrong_label_count"
                ],
                "os_strict_applied_wrong": variants["os_strict"][
                    "applied_wrong_label_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
