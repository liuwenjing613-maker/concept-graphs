#!/usr/bin/env python3
"""Freeze a 16-case GT-derived audit set before any repair result is inspected."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCALE_DIRS = ("voxel0p025", "voxel0p05", "voxel0p10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--matching-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-b0", action="append", required=True, metavar="SCENE=PATH")
    parser.add_argument("--scene-o3", action="append", required=True, metavar="SCENE=PATH")
    return parser.parse_args()


def named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"Expected SCENE=PATH, got {value!r}")
        result[name] = Path(raw_path).resolve()
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value in ("", "nan", "NaN", None) else float(value)


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return -1 if value in ("", "nan", "NaN", None) else int(float(value))


def as_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"true", "1", "yes"}


def raw_frame_number(path: object) -> int | None:
    match = re.search(r"frame(\d+)", str(path))
    return int(match.group(1)) if match else None


def event_for_object(obj: dict[str, Any], processed_frame: int | None) -> dict[str, Any]:
    if processed_frame is None:
        return {"processed_frame": None, "raw_frame": None, "rgb_path": None, "mask_idx": None}
    frames = [int(value) for value in obj.get("image_idx", [])]
    try:
        position = frames.index(int(processed_frame))
    except ValueError:
        return {"processed_frame": int(processed_frame), "raw_frame": None, "rgb_path": None, "mask_idx": None}
    paths = obj.get("color_path", [])
    masks = obj.get("mask_idx", [])
    path = paths[position] if position < len(paths) else None
    return {
        "processed_frame": int(processed_frame),
        "raw_frame": raw_frame_number(path),
        "rgb_path": str(path) if path is not None else None,
        "mask_idx": int(masks[position]) if position < len(masks) else None,
    }


def build_observation_audit(b0: dict[str, Any], o3: dict[str, Any]) -> dict[str, Any]:
    gt_meta: dict[int, dict[str, Any]] = {}
    o3_by_frame: defaultdict[int, list[tuple[int, Any]]] = defaultdict(list)
    for obj in o3["objects"]:
        gt_id = int(obj["oracle_gt_id"])
        gt_meta[gt_id] = obj
        for image_idx, mask in zip(obj.get("image_idx", []), obj.get("mask", [])):
            o3_by_frame[int(image_idx)].append((gt_id, mask))

    b0_by_frame: defaultdict[int, list[tuple[int, int, Any]]] = defaultdict(list)
    for pred_index, obj in enumerate(b0["objects"]):
        for position, (image_idx, mask) in enumerate(zip(obj.get("image_idx", []), obj.get("mask", []))):
            b0_by_frame[int(image_idx)].append((pred_index, position, mask))

    # Match each B0 mask to O3 GT masks by actual pixel overlap. mask_idx is only
    # a local index within each condition and must never be treated as identity.
    observation_matches: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    ambiguous_gt_pixels = 0
    total_gt_pixels = 0
    for image_idx in sorted(set(o3_by_frame) | set(b0_by_frame)):
        gt_entries = o3_by_frame.get(image_idx, [])
        pred_entries = b0_by_frame.get(image_idx, [])
        if not pred_entries:
            continue
        if not gt_entries:
            for pred_index, position, mask in pred_entries:
                observation_matches[pred_index].append({
                    "processed_frame": image_idx, "position": position, "gt_instance_id": None,
                    "best_gt_purity": 0.0, "mask_pixels": int(mask.sum()),
                })
            continue
        shape = gt_entries[0][1].shape
        occupancy = None
        for _gt_id, mask in gt_entries:
            if mask.shape != shape:
                raise ValueError(f"O3 mask shape mismatch at frame {image_idx}")
            binary = mask.astype(bool, copy=False)
            if occupancy is None:
                occupancy = binary.astype("uint16")
            else:
                occupancy += binary
        assert occupancy is not None
        ambiguous_gt_pixels += int((occupancy > 1).sum())
        total_gt_pixels += int((occupancy > 0).sum())
        label_image = occupancy.copy()
        label_image.fill(0)
        slot_to_gt: dict[int, int] = {}
        for slot, (gt_id, mask) in enumerate(gt_entries, 1):
            label_image[(occupancy == 1) & mask.astype(bool, copy=False)] = slot
            slot_to_gt[slot] = gt_id
        for pred_index, position, mask in pred_entries:
            binary = mask.astype(bool, copy=False)
            if binary.shape != shape:
                raise ValueError(f"B0/O3 mask shape mismatch at frame {image_idx}")
            area = int(binary.sum())
            if area == 0:
                best_slot, best_count = 0, 0
            else:
                counts = np.bincount(label_image[binary].ravel())
                if len(counts) <= 1:
                    best_slot, best_count = 0, 0
                else:
                    best_slot = int(counts[1:].argmax()) + 1
                    best_count = int(counts[best_slot])
            observation_matches[pred_index].append({
                "processed_frame": image_idx,
                "position": position,
                "gt_instance_id": slot_to_gt.get(best_slot),
                "best_gt_purity": best_count / area if area else 0.0,
                "mask_pixels": area,
            })

    pred_stats: dict[int, dict[str, Any]] = {}
    for pred_index, obj in enumerate(b0["objects"]):
        all_matches = observation_matches.get(pred_index, [])
        confident = [
            item for item in all_matches
            if item["gt_instance_id"] is not None and item["best_gt_purity"] >= 0.50
        ]
        counts = Counter(int(item["gt_instance_id"]) for item in confident)
        mapped = sum(counts.values())
        threshold = max(2, int(math.ceil(0.05 * mapped))) if mapped else 2
        significant = {gt_id: count for gt_id, count in counts.items() if count >= threshold}
        dominant_gt, dominant_count = (counts.most_common(1)[0] if counts else (None, 0))
        first_mixed = None
        observed: set[int] = set()
        for item in sorted(confident, key=lambda value: (value["processed_frame"], value["position"])):
            observed.add(int(item["gt_instance_id"]))
            if len(observed) >= 2:
                first_mixed = int(item["processed_frame"])
                break
        purities = sorted(float(item["best_gt_purity"]) for item in confident)
        median_purity = purities[len(purities) // 2] if purities else 0.0
        pred_stats[pred_index] = {
            "mapped_observations": mapped,
            "unmapped_or_low_purity_observations": len(all_matches) - mapped,
            "observation_match_purity_threshold": 0.50,
            "median_confident_observation_gt_purity": median_purity,
            "dominant_gt_instance_id": dominant_gt,
            "dominant_gt_fraction": dominant_count / mapped if mapped else 0.0,
            "gt_observation_counts": dict(sorted(counts.items())),
            "significant_gt_observation_counts": dict(sorted(significant.items())),
            "significant_gt_count": len(significant),
            "first_mixed_processed_frame": first_mixed,
        }

    gt_stats: dict[int, dict[str, Any]] = {}
    for gt_id, obj in gt_meta.items():
        counts: Counter[int] = Counter()
        frame_by_prediction: defaultdict[int, list[int]] = defaultdict(list)
        assigned_purities: list[float] = []
        for pred_index, matches in observation_matches.items():
            for item in matches:
                if item["gt_instance_id"] == gt_id and item["best_gt_purity"] >= 0.50:
                    counts[pred_index] += 1
                    frame_by_prediction[pred_index].append(int(item["processed_frame"]))
                    assigned_purities.append(float(item["best_gt_purity"]))
        mapped_observations = sum(counts.values())
        threshold = max(2, int(math.ceil(0.05 * mapped_observations))) if mapped_observations else 2
        significant = {pred_index: count for pred_index, count in counts.items() if count >= threshold}
        dominant_pred, dominant_count = (counts.most_common(1)[0] if counts else (None, 0))
        creation_order = sorted(
            ((min(frames), pred_index) for pred_index, frames in frame_by_prediction.items() if pred_index in significant),
            key=lambda item: (item[0], item[1]),
        )
        first_split = creation_order[1][0] if len(creation_order) >= 2 else None
        assigned_purities.sort()
        gt_stats[gt_id] = {
            "mapped_observations": mapped_observations,
            "o3_visible_observations": len(obj.get("image_idx", [])),
            "median_assigned_observation_gt_purity": assigned_purities[len(assigned_purities) // 2] if assigned_purities else 0.0,
            "dominant_predicted_index": dominant_pred,
            "dominant_prediction_fraction": dominant_count / sum(counts.values()) if counts else 0.0,
            "prediction_observation_counts": dict(sorted(counts.items())),
            "significant_prediction_observation_counts": dict(sorted(significant.items())),
            "significant_prediction_count": len(significant),
            "first_split_processed_frame": first_split,
        }

    return {
        "observation_matches": observation_matches,
        "pred_stats": pred_stats,
        "gt_stats": gt_stats,
        "gt_meta": gt_meta,
        "ambiguous_gt_pixel_fraction": ambiguous_gt_pixels / total_gt_pixels if total_gt_pixels else 0.0,
    }


def balanced_select(rows: list[dict[str, Any]], count: int, usable: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-float(row["selection_score"]), str(row["selection_key"])))
    selected: list[dict[str, Any]] = []
    scenes = sorted({str(row["scene_id"]) for row in ordered})
    if count >= len(scenes):
        for scene in scenes:
            candidate = next((row for row in ordered if row["scene_id"] == scene and usable(row)), None)
            if candidate is not None:
                selected.append(candidate)
    for row in ordered:
        if len(selected) >= count:
            break
        if row in selected or not usable(row):
            continue
        selected.append(row)
    return selected


def case_entities(row: dict[str, Any]) -> set[tuple[str, str, int]]:
    scene = str(row["scene_id"])
    result: set[tuple[str, str, int]] = set()
    if row.get("predicted_index") is not None:
        result.add((scene, "pred", int(row["predicted_index"])))
    if row.get("gt_instance_id") is not None:
        result.add((scene, "gt", int(row["gt_instance_id"])))
    for pred_index in row.get("associated_predicted_indices", []):
        result.add((scene, "pred", int(pred_index)))
    for gt_id in row.get("associated_gt_instance_ids", []):
        result.add((scene, "gt", int(gt_id)))
    return result


def main() -> None:
    args = parse_args()
    b0_paths = named_paths(args.scene_b0)
    o3_paths = named_paths(args.scene_o3)
    if set(b0_paths) != set(o3_paths):
        raise ValueError("B0 and O3 scenes differ")
    candidates = json.loads(args.candidate_pool.read_text(encoding="utf-8"))
    input_paths = [args.candidate_pool.resolve()]
    scene_data: dict[str, dict[str, Any]] = {}
    online_latest_frame: dict[str, int] = {}
    for scene in sorted(b0_paths):
        b0 = load_pickle(b0_paths[scene])
        o3 = load_pickle(o3_paths[scene])
        audit = build_observation_audit(b0, o3)
        scene_data[scene] = {"b0": b0, "o3": o3, **audit}
        online_latest_frame[scene] = max(
            int(value) for obj in b0["objects"] for value in obj.get("image_idx", [])
        )
        input_paths.extend((b0_paths[scene], o3_paths[scene]))

    used: set[tuple[str, str, int]] = set()
    frozen: list[dict[str, Any]] = []

    # Reserve the rare strict S1 entity before association sampling so that a
    # confounded case cannot displace the only interpretable semantic oracle.
    strict_s1_entities: set[tuple[str, str, int]] = set()
    for scene, data in sorted(scene_data.items()):
        pair_by_scale: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
        for scale_dir in SCALE_DIRS:
            path = args.matching_root / scene / scale_dir / "object_gt_overlaps.csv"
            input_paths.append(path.resolve())
            for overlap_row in read_csv(path):
                pair_by_scale[(as_int(overlap_row, "predicted_index"), as_int(overlap_row, "gt_instance_id"))][scale_dir] = overlap_row
        for (pred_index, gt_id), rows in pair_by_scale.items():
            if (
                set(rows) != set(SCALE_DIRS)
                or any(as_bool(rows[scale], "semantic_consistent") for scale in SCALE_DIRS)
                or not all(as_bool(rows[scale], "semantic_eligible") for scale in SCALE_DIRS)
            ):
                continue
            pred_stats = data["pred_stats"][pred_index]
            gt_stats = data["gt_stats"][gt_id]
            if (
                pred_stats["dominant_gt_instance_id"] == gt_id
                and pred_stats["dominant_gt_fraction"] >= 0.75
                and pred_stats["significant_gt_count"] <= 1
                and pred_stats["median_confident_observation_gt_purity"] >= 0.75
                and gt_stats["dominant_predicted_index"] == pred_index
                and gt_stats["dominant_prediction_fraction"] >= 0.75
                and gt_stats["significant_prediction_count"] <= 1
                and gt_stats["median_assigned_observation_gt_purity"] >= 0.75
            ):
                strict_s1_entities.update({(scene, "pred", pred_index), (scene, "gt", gt_id)})

    def available(row: dict[str, Any]) -> bool:
        return not (case_entities(row) & used)

    def association_available(row: dict[str, Any]) -> bool:
        return available(row) and not (case_entities(row) & strict_s1_entities)

    def commit(rows: list[dict[str, Any]], family: str, expected: int) -> None:
        if len(rows) != expected:
            raise RuntimeError(f"Could freeze only {len(rows)}/{expected} cases for {family}")
        for row in rows:
            row["family"] = family
            frozen.append(row)
            used.update(case_entities(row))

    # Association cases are derived from GT-linked observation identity, not
    # from endpoint appearance or human labels.
    association_rows: list[dict[str, Any]] = []
    for scene, data in sorted(scene_data.items()):
        pred_scales: dict[str, dict[int, dict[str, str]]] = {}
        gt_scales: dict[str, dict[int, dict[str, str]]] = {}
        for scale_dir in SCALE_DIRS:
            pred_path = args.matching_root / scene / scale_dir / "predicted_object_summary.csv"
            gt_path = args.matching_root / scene / scale_dir / "gt_instance_summary.csv"
            input_paths.extend((pred_path.resolve(), gt_path.resolve()))
            pred_scales[scale_dir] = {as_int(row, "predicted_index"): row for row in read_csv(pred_path)}
            gt_scales[scale_dir] = {as_int(row, "gt_instance_id"): row for row in read_csv(gt_path)}
        for pred_index, stats in data["pred_stats"].items():
            if pred_index not in pred_scales["voxel0p05"]:
                continue
            stable_degree = min(
                as_int(pred_scales[scale][pred_index], "overlap_gt_count_purity_0p05")
                for scale in SCALE_DIRS
            )
            if (
                stable_degree >= 2
                and stats["significant_gt_count"] >= 2
                and stats["median_confident_observation_gt_purity"] >= 0.75
            ):
                significant = stats["significant_gt_observation_counts"]
                score = (1.0 - stats["dominant_gt_fraction"]) * math.log1p(stats["mapped_observations"]) * (len(significant) - 1)
                association_rows.append({
                    "scene_id": scene,
                    "phenomenon": "contamination",
                    "predicted_index": pred_index,
                    "gt_instance_id": None,
                    "predicted_label": pred_scales["voxel0p05"][pred_index]["predicted_label"],
                    "gt_label": None,
                    "subtype": "A_false_merge_observation_identity",
                    "selection_key": f"{scene}:A:merge:{pred_index}",
                    "selection_score": score,
                    "stable_overlap_gt_degree_0p05": stable_degree,
                    "associated_gt_instance_ids": [int(value) for value in significant],
                    "observation_identity_audit": stats,
                    "s_processed_frame": stats["first_mixed_processed_frame"],
                })
        for gt_id, stats in data["gt_stats"].items():
            if gt_id not in gt_scales["voxel0p05"]:
                continue
            stable_degree = min(
                as_int(gt_scales[scale][gt_id], "overlap_prediction_count_coverage_0p05")
                for scale in SCALE_DIRS
            )
            if (
                stable_degree >= 2
                and stats["significant_prediction_count"] >= 2
                and stats["median_assigned_observation_gt_purity"] >= 0.75
            ):
                significant = stats["significant_prediction_observation_counts"]
                score = (1.0 - stats["dominant_prediction_fraction"]) * math.log1p(stats["mapped_observations"]) * (len(significant) - 1)
                association_rows.append({
                    "scene_id": scene,
                    "phenomenon": "fragmentation",
                    "predicted_index": stats["dominant_predicted_index"],
                    "gt_instance_id": gt_id,
                    "predicted_label": gt_scales["voxel0p05"][gt_id]["best_predicted_label"],
                    "gt_label": gt_scales["voxel0p05"][gt_id]["gt_label"],
                    "subtype": "A_false_split_observation_identity",
                    "selection_key": f"{scene}:A:split:{gt_id}",
                    "selection_score": score,
                    "stable_overlap_prediction_degree_0p05": stable_degree,
                    "associated_predicted_indices": [int(value) for value in significant],
                    "observation_identity_audit": stats,
                    "s_processed_frame": stats["first_split_processed_frame"],
                })
    # Preserve both merge and split when the evidence contains both.
    association_selected: list[dict[str, Any]] = []
    for subtype in ("A_false_merge_observation_identity", "A_false_split_observation_identity"):
        subtype_rows = [row for row in association_rows if row["subtype"] == subtype and association_available(row)]
        if subtype_rows:
            best = sorted(subtype_rows, key=lambda row: (-row["selection_score"], row["selection_key"]))[0]
            association_selected.append(best)
            used.update(case_entities(best))
    # Enforce two-scene coverage if a third slot can supply the missing scene.
    missing_scenes = set(scene_data) - {row["scene_id"] for row in association_selected}
    for scene in sorted(missing_scenes):
        if len(association_selected) >= 3:
            break
        choices = [row for row in association_rows if row["scene_id"] == scene and row not in association_selected and association_available(row)]
        if choices:
            best = sorted(choices, key=lambda row: (-row["selection_score"], row["selection_key"]))[0]
            association_selected.append(best)
            used.update(case_entities(best))
    for row in sorted(association_rows, key=lambda item: (-item["selection_score"], item["selection_key"])):
        if len(association_selected) >= 3:
            break
        if row not in association_selected and association_available(row):
            association_selected.append(row)
            used.update(case_entities(row))
    # Undo temporary reservation; commit performs the authoritative reservation.
    for row in association_selected:
        used.difference_update(case_entities(row))
    commit(association_selected, "association", 3)

    semantic_rows: list[dict[str, Any]] = []
    semantic_composite_rows: list[dict[str, Any]] = []
    for scene, data in sorted(scene_data.items()):
        pair_by_scale: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
        for scale_dir in SCALE_DIRS:
            path = args.matching_root / scene / scale_dir / "object_gt_overlaps.csv"
            input_paths.append(path.resolve())
            for overlap_row in read_csv(path):
                pair_by_scale[(as_int(overlap_row, "predicted_index"), as_int(overlap_row, "gt_instance_id"))][scale_dir] = overlap_row
        for (pred_index, gt_id), rows in pair_by_scale.items():
            if set(rows) != set(SCALE_DIRS) or any(as_bool(rows[scale], "semantic_consistent") for scale in SCALE_DIRS):
                continue
            pred_stats = data["pred_stats"][pred_index]
            gt_stats = data["gt_stats"][gt_id]
            semantic_eligible = all(as_bool(rows[scale], "semantic_eligible") for scale in SCALE_DIRS)
            strict_s1 = (
                semantic_eligible
                and
                pred_stats["dominant_gt_instance_id"] == gt_id
                and pred_stats["dominant_gt_fraction"] >= 0.75
                and pred_stats["significant_gt_count"] <= 1
                and pred_stats["median_confident_observation_gt_purity"] >= 0.75
                and gt_stats["dominant_predicted_index"] == pred_index
                and gt_stats["dominant_prediction_fraction"] >= 0.75
                and gt_stats["significant_prediction_count"] <= 1
                and gt_stats["median_assigned_observation_gt_purity"] >= 0.75
            )
            ious = [as_float(rows[scale], "voxel_iou") for scale in SCALE_DIRS]
            purities = [as_float(rows[scale], "purity") for scale in SCALE_DIRS]
            coverages = [as_float(rows[scale], "coverage") for scale in SCALE_DIRS]
            obj = data["b0"]["objects"][pred_index]
            first_frame = min((int(value) for value in obj.get("image_idx", [])), default=None)
            semantic_row = {
                "scene_id": scene,
                "phenomenon": "semantic_mismatch",
                "predicted_index": pred_index,
                "gt_instance_id": gt_id,
                "predicted_label": rows["voxel0p05"]["predicted_label"],
                "gt_label": rows["voxel0p05"]["gt_label"],
                "subtype": (
                    "S1_geometry_link_stable_label_mismatch"
                    if strict_s1
                    else ("S3_ontology_or_unknown_mismatch" if not semantic_eligible else "S2_geometry_or_identity_confounded_label_mismatch")
                ),
                "selection_key": f"{scene}:S:{pred_index}:{gt_id}",
                "selection_score": (min(ious) + 0.25 * min(purities) + 0.25 * min(coverages)) * pred_stats["dominant_gt_fraction"] * gt_stats["dominant_prediction_fraction"],
                "minimum_voxel_iou": min(ious),
                "minimum_purity": min(purities),
                "minimum_coverage": min(coverages),
                "prediction_observation_identity_audit": pred_stats,
                "gt_observation_identity_audit": gt_stats,
                "s_processed_frame": first_frame,
                "isolated_semantic_oracle_interpretable": strict_s1,
            }
            (semantic_rows if strict_s1 else semantic_composite_rows).append(semantic_row)
    semantic_selected = balanced_select(semantic_rows, min(3, len(semantic_rows)), available)
    for row in semantic_selected:
        used.update(case_entities(row))
    semantic_needed = 3 - len(semantic_selected)
    semantic_fill = balanced_select(semantic_composite_rows, semantic_needed, available)
    for row in semantic_selected:
        used.difference_update(case_entities(row))
    semantic_selected.extend(semantic_fill)
    commit(semantic_selected, "semantic", 3)

    spurious_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["phenomenon"] != "unsupported_prediction":
            continue
        observations = int(candidate.get("observation_count") or 0)
        voxels = float(candidate.get("predicted_voxels_0p05") or 0.0)
        spurious_rows.append({
            **candidate,
            "subtype": "unsupported_prediction_all_three_scales",
            "selection_key": f"{candidate['scene_id']}:P:{candidate['predicted_index']}",
            "selection_score": float(candidate["stable_score"]) * math.log1p(max(observations, 1)) * math.log1p(max(voxels, 1)),
            "s_processed_frame": candidate.get("first_frame"),
        })
    spurious_selected = balanced_select(spurious_rows, 3, available)
    commit(spurious_selected, "spurious", 3)

    geometry_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["phenomenon"] != "incompleteness":
            continue
        scene = candidate["scene_id"]
        gt_id = int(candidate["gt_instance_id"])
        gt_obj = scene_data[scene]["gt_meta"][gt_id]
        observations = len(gt_obj.get("image_idx", []))
        voxels = float(candidate.get("gt_voxels_0p05") or 0.0)
        s_frame = min((int(value) for value in gt_obj.get("image_idx", [])), default=None)
        geometry_rows.append({
            **candidate,
            "subtype": "G_observable_gt_incomplete_or_missed",
            "selection_key": f"{scene}:G:{gt_id}",
            "selection_score": float(candidate["stable_score"]) * math.log1p(max(observations, 1)) * math.log1p(max(voxels, 1)),
            "o3_observation_count": observations,
            "s_processed_frame": s_frame,
        })
    geometry_selected = balanced_select(geometry_rows, 3, available)
    commit(geometry_selected, "geometry", 3)

    # Build controls from every stable pair, then require clean observation
    # identity in both directions. If none pass the strict identity gate, keep
    # explicitly tiered best-available non-target controls; never call them
    # absolute clean controls.
    control_rows: list[dict[str, Any]] = []
    for scene in sorted(scene_data):
        pair_by_scale: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
        for scale_dir in SCALE_DIRS:
            path = args.matching_root / scene / scale_dir / "object_gt_overlaps.csv"
            input_paths.append(path.resolve())
            for row in read_csv(path):
                pair_by_scale[(as_int(row, "predicted_index"), as_int(row, "gt_instance_id"))][scale_dir] = row
        for (pred_index, gt_id), rows in pair_by_scale.items():
            if set(rows) != set(SCALE_DIRS) or not all(as_bool(rows[scale], "semantic_consistent") for scale in SCALE_DIRS):
                continue
            pred_stats = scene_data[scene]["pred_stats"][pred_index]
            gt_stats = scene_data[scene]["gt_stats"][gt_id]
            strict_identity_gate = (
                pred_stats["dominant_gt_instance_id"] == gt_id
                and pred_stats["dominant_gt_fraction"] >= 0.90
                and pred_stats["significant_gt_count"] <= 1
                and pred_stats["median_confident_observation_gt_purity"] >= 0.75
                and gt_stats["dominant_predicted_index"] == pred_index
                and gt_stats["dominant_prediction_fraction"] >= 0.90
                and gt_stats["significant_prediction_count"] <= 1
                and gt_stats["median_assigned_observation_gt_purity"] >= 0.75
            )
            relative_identity_gate = (
                pred_stats["dominant_gt_instance_id"] == gt_id
                and pred_stats["dominant_gt_fraction"] >= 0.50
                and pred_stats["median_confident_observation_gt_purity"] >= 0.50
                and gt_stats["dominant_predicted_index"] == pred_index
                and gt_stats["dominant_prediction_fraction"] >= 0.50
                and gt_stats["median_assigned_observation_gt_purity"] >= 0.50
            )
            ious = [as_float(rows[scale], "voxel_iou") for scale in SCALE_DIRS]
            purities = [as_float(rows[scale], "purity") for scale in SCALE_DIRS]
            coverages = [as_float(rows[scale], "coverage") for scale in SCALE_DIRS]
            control_rows.append({
                "scene_id": scene,
                "family": "control",
                "subtype": (
                    "strict_identity_control"
                    if strict_identity_gate
                    else ("relative_identity_control" if relative_identity_gate else "geometry_ranked_non_target_control")
                ),
                "phenomenon": "relative_non_target_control",
                "predicted_index": pred_index,
                "gt_instance_id": gt_id,
                "predicted_label": rows["voxel0p05"]["predicted_label"],
                "gt_label": rows["voxel0p05"]["gt_label"],
                "selection_key": f"{scene}:C:{pred_index}:{gt_id}",
                "selection_score": min(ious) + 0.5 * min(purities) + 0.5 * min(coverages),
                "minimum_voxel_iou": min(ious),
                "minimum_purity": min(purities),
                "minimum_coverage": min(coverages),
                "absolute_clean_gate_pass": min(ious) >= 0.25 and min(purities) >= 0.50 and min(coverages) >= 0.50,
                "strict_identity_gate_pass": strict_identity_gate,
                "relative_identity_gate_pass": relative_identity_gate,
                "control_tier": (
                    "strict_identity"
                    if strict_identity_gate
                    else ("relative_identity" if relative_identity_gate else "geometry_ranked_non_target")
                ),
                "prediction_observation_identity_audit": pred_stats,
                "gt_observation_identity_audit": gt_stats,
                "s_processed_frame": min((int(value) for value in scene_data[scene]["b0"]["objects"][pred_index].get("image_idx", [])), default=None),
            })
    strict_controls = [row for row in control_rows if row["strict_identity_gate_pass"]]
    relative_controls = [
        row for row in control_rows
        if not row["strict_identity_gate_pass"] and row["relative_identity_gate_pass"]
    ]
    ranked_controls = [row for row in control_rows if not row["relative_identity_gate_pass"]]
    control_selected: list[dict[str, Any]] = []
    for tier_rows in (strict_controls, relative_controls, ranked_controls):
        needed = 4 - len(control_selected)
        if needed <= 0:
            break
        chosen = balanced_select(tier_rows, needed, available)
        control_selected.extend(chosen)
        for row in chosen:
            used.update(case_entities(row))
    for row in control_selected:
        used.difference_update(case_entities(row))
    commit(control_selected, "control", 4)

    family_codes = {"semantic": "S", "geometry": "G", "spurious": "P", "association": "A", "control": "C"}
    family_counters: Counter[str] = Counter()
    final_rows: list[dict[str, Any]] = []
    for row in frozen:
        family = row["family"]
        family_counters[family] += 1
        scene = row["scene_id"]
        row["case_id"] = f"{family_codes[family]}{family_counters[family]:02d}_{scene}"
        row["d_processed_frame"] = online_latest_frame[scene]
        row["h_processed_frame"] = online_latest_frame[scene]
        row["c_processed_frame"] = None
        row["online_latest_processed_frame"] = online_latest_frame[scene]
        pred_index = row.get("predicted_index")
        if pred_index is not None:
            row["i1_event"] = event_for_object(scene_data[scene]["b0"]["objects"][int(pred_index)], row.get("s_processed_frame"))
            if row["i1_event"]["rgb_path"] is None:
                for alternative in row.get("associated_predicted_indices", []):
                    event = event_for_object(scene_data[scene]["b0"]["objects"][int(alternative)], row.get("s_processed_frame"))
                    if event["rgb_path"] is not None:
                        row["i1_event"] = event
                        row["i1_event"]["predicted_index"] = int(alternative)
                        break
        else:
            gt_id = int(row["gt_instance_id"])
            row["i1_event"] = event_for_object(scene_data[scene]["gt_meta"][gt_id], row.get("s_processed_frame"))
        row["selection_used_human_labels"] = False
        row["selection_used_oracle_repair_outcomes"] = False
        final_rows.append(row)

    expected_counts = {"semantic": 3, "geometry": 3, "spurious": 3, "association": 3, "control": 4}
    actual_counts = Counter(row["family"] for row in final_rows)
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(f"Wrong family counts: {dict(actual_counts)}")
    if len({row["case_id"] for row in final_rows}) != 16:
        raise RuntimeError("Case IDs are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "frozen_16_cases.json"
    json_path.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    compact_fields = [
        "case_id", "scene_id", "family", "subtype", "phenomenon", "predicted_index",
        "gt_instance_id", "predicted_label", "gt_label", "selection_score",
        "s_processed_frame", "d_processed_frame", "h_processed_frame", "online_latest_processed_frame",
    ]
    csv_path = args.output_dir / "frozen_16_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)
    manifest = {
        "status": "frozen_pre_oracle",
        "case_count": 16,
        "family_counts": expected_counts,
        "scene_counts": dict(Counter(row["scene_id"] for row in final_rows)),
        "human_labels_used": False,
        "oracle_repair_outcomes_used": False,
        "selection_rules": {
            "semantic": "prefer strict S1: stable mismatch + bidirectional identity + mask purity; if fewer than 3 exist, freeze S2/S3 confounded mismatches as explicit non-interpretable negatives",
            "geometry": "stable three-scale worst GT coverage; task weight=log(observations)*log(GT voxels)",
            "spurious": "stable three-scale unsupported prediction; task weight=log(observations)*log(predicted voxels)",
            "association": "same observation keys map to >=2 significant O3 identities, or one O3 identity maps to >=2 significant B0 objects",
            "control": "prefer strict bidirectional >=0.90 identity, then relative >=0.50 identity, then explicit geometry-ranked non-target pairs; all require three-scale semantic consistency and no entity reuse",
            "balance": "each family covers both scenes when eligible candidates exist",
        },
        "control_limitation": "No assumption of absolute cleanliness: strict identity and absolute geometry gates are reported per case, and fallback controls are explicitly tiered.",
        "input_sha256": {str(path): sha256(path) for path in sorted(set(input_paths))},
        "output_sha256": {str(json_path): sha256(json_path), str(csv_path): sha256(csv_path)},
    }
    manifest_path = args.output_dir / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"case_count": 16, "family_counts": expected_counts, "scene_counts": manifest["scene_counts"], "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
