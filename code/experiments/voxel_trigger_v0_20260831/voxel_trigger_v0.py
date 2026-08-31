#!/usr/bin/env python3
"""Build and evaluate the simple voxel trigger on frozen ali-dev B0 maps.

The voxel payload is intentionally minimal:

    seen_count + label_hist + obs_ids

All semantic, fragmentation, duplicate, and GT diagnostics are derived outside
the voxel payload.  GT is never used to build voxel evidence or anomaly scores.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


RUN_DATE = "20260831"
SCHEMA_VERSION = "voxel-trigger-v0/1.0"
PRIMARY_SCALE = 0.05
SCALES = (0.025, 0.05, 0.10)
DECISION_GATES = {
    "go": (
        "At 5 cm, combined object anomaly AP exceeds error prevalence by at least "
        "0.10 and top-20% error rate is at least 2x bottom-20% in both scenes; "
        "direction agrees at two or more voxel scales."
    ),
    "modify": (
        "Separation appears in only one scene, one scale, or one error family; "
        "retain the useful component but do not claim a unified trigger."
    ),
    "stop": (
        "No stable top-vs-bottom separation, or voxel scoring does not improve over "
        "the non-spatial object label-histogram baseline."
    ),
}
BITS = 21
OFFSET = 1 << (BITS - 1)
FIELD_MASK = (1 << BITS) - 1
SHIFT_X = BITS * 2
SHIFT_Y = BITS
BG_LABELS = {"wall", "floor", "ceiling"}

BASE = Path("/home/chenkejun/beauty/conceptgraphs")
EXPERIMENT = BASE / "results/experiments/oracle_three_error_20260828/pilot"
CODE_ROOT_DEFAULT = BASE / "code/experiments/oracle_three_error_20260828"
GT_SIDECARS_DEFAULT = EXPERIMENT / "gt_full"
OBJECTS_JSON_DEFAULT = BASE / "code/third_party/ReplicaSSG/files/objects.json"

SCENE_SPECS = {
    "room0": {
        "source_scene": "room_0",
        "baseline_map": EXPERIMENT
        / "b0_dataset/Replica/room0/exps/b0_room0_fresh/pcd_b0_room0_fresh.pkl.gz",
        "gt_map": EXPERIMENT / "room0/o3/pcd_o3.pkl.gz",
    },
    "office0": {
        "source_scene": "office_0",
        "baseline_map": EXPERIMENT
        / "b0_dataset/Replica/office0/exps/b0_office0_fresh_final/pcd_b0_office0_fresh_final.pkl.gz",
        "gt_map": EXPERIMENT / "office0/o3/pcd_o3.pkl.gz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(SCENE_SPECS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT_DEFAULT)
    parser.add_argument("--gt-sidecars", type=Path, default=GT_SIDECARS_DEFAULT)
    parser.add_argument("--objects-json", type=Path, default=OBJECTS_JSON_DEFAULT)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def load_pickle_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def pack_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"expected Nx3 coordinates, got {coords.shape}")
    shifted = coords + OFFSET
    if np.any(shifted < 0) or np.any(shifted > FIELD_MASK):
        low = coords.min(axis=0).tolist() if len(coords) else []
        high = coords.max(axis=0).tolist() if len(coords) else []
        raise ValueError(f"voxel coordinate outside 21-bit range: min={low}, max={high}")
    return (
        (shifted[:, 0] << SHIFT_X)
        | (shifted[:, 1] << SHIFT_Y)
        | shifted[:, 2]
    ).astype(np.int64, copy=False)


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    x = ((keys >> SHIFT_X) & FIELD_MASK) - OFFSET
    y = ((keys >> SHIFT_Y) & FIELD_MASK) - OFFSET
    z = (keys & FIELD_MASK) - OFFSET
    return np.column_stack((x, y, z)).astype(np.int32, copy=False)


def voxel_keys(points: np.ndarray, scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return np.empty(0, dtype=np.int64)
    finite = np.all(np.isfinite(points), axis=1)
    quantized = np.floor(points[finite] / float(scale)).astype(np.int64)
    return np.unique(pack_coords(quantized))


def entropy_from_counts(counts: Iterable[int]) -> float:
    values = np.asarray([int(value) for value in counts if int(value) > 0], dtype=float)
    if len(values) <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(values)))


def cosine_counts(left: Counter[int], right: Counter[int]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    a = np.asarray([left.get(key, 0) for key in keys], dtype=float)
    b = np.asarray([right.get(key, 0) for key in keys], dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def component_sizes(keys: np.ndarray, connectivity: int = 26) -> list[int]:
    remaining = set(int(value) for value in np.asarray(keys, dtype=np.int64))
    if not remaining:
        return []
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                if connectivity == 6 and abs(dx) + abs(dy) + abs(dz) != 1:
                    continue
                offsets.append((dx << SHIFT_X) + (dy << SHIFT_Y) + dz)
    sizes = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        size = 1
        while stack:
            current = stack.pop()
            for delta in offsets:
                neighbor = current + delta
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        sizes.append(size)
    return sorted(sizes, reverse=True)


def percentile_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return np.empty(0, dtype=float)
    finite = np.isfinite(array)
    output = np.zeros(len(array), dtype=float)
    if finite.any():
        output[finite] = rankdata(array[finite], method="average") / int(finite.sum())
    return output


def binary_metrics(labels: list[bool], scores: list[float]) -> dict:
    y = np.asarray(labels, dtype=np.uint8)
    score = np.asarray(scores, dtype=float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if not len(y):
        return {"n": 0}
    order = np.argsort(-score, kind="stable")
    prevalence = float(y.mean())
    result = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
    }
    if len(np.unique(y)) == 2:
        result["average_precision"] = float(average_precision_score(y, score))
        result["auroc"] = float(roc_auc_score(y, score))
    else:
        result["average_precision"] = None
        result["auroc"] = None
    for name, count in (
        ("top5", min(5, len(y))),
        ("top10", min(10, len(y))),
        ("top20pct", max(1, int(math.ceil(0.2 * len(y))))),
    ):
        selected = y[order[:count]]
        precision = float(selected.mean())
        result[name] = {
            "k": int(count),
            "true_errors": int(selected.sum()),
            "precision": precision,
            "lift_vs_prevalence": float(precision / prevalence) if prevalence else None,
        }
    bottom_count = max(1, int(math.ceil(0.2 * len(y))))
    result["bottom20pct"] = {
        "k": int(bottom_count),
        "true_errors": int(y[order[-bottom_count:]].sum()),
        "error_rate": float(y[order[-bottom_count:]].mean()),
    }
    top_rate = result["top20pct"]["precision"]
    bottom_rate = result["bottom20pct"]["error_rate"]
    result["top_to_bottom_ratio"] = float(top_rate / bottom_rate) if bottom_rate else None
    return result


def load_instance_labels(path: Path, source_scene: str) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scans = [item for item in payload["scans"] if item["scan"] == source_scene]
    if len(scans) != 1:
        raise ValueError(f"expected one objects entry for {source_scene}")
    return {int(item["id"]): str(item["label"]) for item in scans[0]["objects"]}


def mask_gt_assignment(
    mask: np.ndarray,
    semantic: np.ndarray,
    labels: dict[int, str],
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return {
            "gt_top_id": None,
            "gt_top_label": None,
            "gt_top_pixels": 0,
            "gt_second_pixels": 0,
            "mask_area": 0,
            "gt_purity": 0.0,
            "gt_second_fraction": 0.0,
            "gt_supported_fraction": 0.0,
            "mask_mixed": False,
            "gt_assignment_eligible": False,
        }
    ids, counts = np.unique(semantic[mask], return_counts=True)
    candidates = sorted(
        (
            (int(count), int(instance_id))
            for instance_id, count in zip(ids.tolist(), counts.tolist())
            if int(instance_id) in labels
        ),
        reverse=True,
    )
    if not candidates:
        return {
            "gt_top_id": None,
            "gt_top_label": None,
            "gt_top_pixels": 0,
            "gt_second_pixels": 0,
            "mask_area": area,
            "gt_purity": 0.0,
            "gt_second_fraction": 0.0,
            "gt_supported_fraction": 0.0,
            "mask_mixed": False,
            "gt_assignment_eligible": False,
        }
    top_count, top_id = candidates[0]
    second_count = candidates[1][0] if len(candidates) > 1 else 0
    supported = sum(count for count, _ in candidates)
    purity = top_count / area
    second_fraction = second_count / area
    return {
        "gt_top_id": int(top_id),
        "gt_top_label": labels[int(top_id)],
        "gt_top_pixels": int(top_count),
        "gt_second_pixels": int(second_count),
        "mask_area": area,
        "gt_purity": float(purity),
        "gt_second_fraction": float(second_fraction),
        "gt_supported_fraction": float(supported / area),
        "mask_mixed": bool(purity < 0.8 or second_fraction >= 0.1),
        "gt_assignment_eligible": bool(top_count >= 25),
    }


@dataclass
class VoxelMap:
    voxel_keys: np.ndarray
    voxel_coords: np.ndarray
    seen_count: np.ndarray
    obs_offsets: np.ndarray
    obs_ids: np.ndarray
    label_offsets: np.ndarray
    label_ids: np.ndarray
    label_counts: np.ndarray


def build_voxel_map(
    *,
    observation_voxels: list[np.ndarray],
    observation_labels: np.ndarray,
    selected: np.ndarray,
    output: Path,
) -> VoxelMap:
    selected_indices = np.flatnonzero(selected)
    key_chunks = [observation_voxels[index] for index in selected_indices]
    nonempty = [position for position, chunk in enumerate(key_chunks) if len(chunk)]
    if not nonempty:
        raise ValueError("selected voxel map has no entries")
    key_entries = np.concatenate([key_chunks[position] for position in nonempty])
    obs_entries = np.concatenate(
        [
            np.full(len(key_chunks[position]), selected_indices[position], dtype=np.int32)
            for position in nonempty
        ]
    )
    label_entries = observation_labels[obs_entries].astype(np.int32, copy=False)

    order = np.argsort(key_entries, kind="stable")
    key_entries = key_entries[order]
    obs_entries = obs_entries[order]
    label_entries_by_voxel = label_entries[order]
    unique_keys, first, counts = np.unique(
        key_entries, return_index=True, return_counts=True
    )
    obs_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )

    pair_order = np.lexsort((label_entries_by_voxel, key_entries))
    pair_keys_sorted = key_entries[pair_order]
    pair_labels_sorted = label_entries_by_voxel[pair_order]
    pair_change = np.ones(len(pair_keys_sorted), dtype=bool)
    pair_change[1:] = (pair_keys_sorted[1:] != pair_keys_sorted[:-1]) | (
        pair_labels_sorted[1:] != pair_labels_sorted[:-1]
    )
    pair_starts = np.flatnonzero(pair_change)
    pair_counts = np.diff(np.append(pair_starts, len(pair_keys_sorted))).astype(np.int32)
    pair_keys = pair_keys_sorted[pair_starts]
    pair_labels = pair_labels_sorted[pair_starts].astype(np.int32, copy=False)
    pair_voxel = np.searchsorted(unique_keys, pair_keys)
    labels_per_voxel = np.bincount(pair_voxel, minlength=len(unique_keys))
    label_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(labels_per_voxel, dtype=np.int64))
    )

    result = VoxelMap(
        voxel_keys=unique_keys,
        voxel_coords=unpack_keys(unique_keys),
        seen_count=counts.astype(np.int32, copy=False),
        obs_offsets=obs_offsets,
        obs_ids=obs_entries.astype(np.int32, copy=False),
        label_offsets=label_offsets,
        label_ids=pair_labels,
        label_counts=pair_counts,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray([SCHEMA_VERSION]),
            voxel_keys=result.voxel_keys,
            voxel_coords=result.voxel_coords,
            seen_count=result.seen_count,
            obs_offsets=result.obs_offsets,
            obs_ids=result.obs_ids,
            label_offsets=result.label_offsets,
            label_ids=result.label_ids,
            label_counts=result.label_counts,
        )
    temporary.replace(output)
    return result


def object_evidence(
    object_keys: np.ndarray,
    voxel_map: VoxelMap,
) -> tuple[dict, Counter[int]]:
    positions = np.searchsorted(voxel_map.voxel_keys, object_keys)
    within = positions < len(voxel_map.voxel_keys)
    valid = np.zeros(len(object_keys), dtype=bool)
    valid[within] = (
        voxel_map.voxel_keys[positions[within]] == object_keys[within]
    )
    matched_object_keys = object_keys[valid]
    voxel_indices = positions[valid]
    vote_counts: Counter[int] = Counter()
    majority_counts: Counter[int] = Counter()
    stable_majority: dict[int, list[int]] = defaultdict(list)
    disagreements = []
    support_values = []
    for key, voxel_index in zip(matched_object_keys.tolist(), voxel_indices.tolist()):
        begin = int(voxel_map.label_offsets[voxel_index])
        end = int(voxel_map.label_offsets[voxel_index + 1])
        labels = voxel_map.label_ids[begin:end]
        counts = voxel_map.label_counts[begin:end]
        if not len(labels):
            continue
        local = Counter({int(label): int(count) for label, count in zip(labels, counts)})
        vote_counts.update(local)
        maximum = max(local.values())
        dominant = min(label for label, count in local.items() if count == maximum)
        majority_counts[dominant] += 1
        total = sum(local.values())
        seen = int(voxel_map.seen_count[voxel_index])
        support_values.append(seen)
        if seen >= 2:
            stable_majority[dominant].append(int(key))
            disagreements.append(1.0 - maximum / total)

    supported = int(len(voxel_indices))
    stable = int(sum(len(values) for values in stable_majority.values()))
    primary_label = majority_counts.most_common(1)[0][0] if majority_counts else None
    ordered_majority = majority_counts.most_common()
    second_label = ordered_majority[1][0] if len(ordered_majority) > 1 else None
    second_fraction = (
        ordered_majority[1][1] / supported if supported and len(ordered_majority) > 1 else 0.0
    )
    largest_second_region = 0
    if second_label is not None and stable_majority.get(second_label):
        sizes = component_sizes(np.asarray(stable_majority[second_label], dtype=np.int64))
        largest_second_region = sizes[0] if sizes else 0
    major_region_threshold = max(3, int(math.ceil(0.02 * max(stable, 1))))
    major_regions = 0
    for keys in stable_majority.values():
        major_regions += sum(
            size >= major_region_threshold
            for size in component_sizes(np.asarray(keys, dtype=np.int64))
        )
    result = {
        "evidence_supported_voxels": supported,
        "evidence_coverage": float(supported / max(len(object_keys), 1)),
        "stable_evidence_voxels": stable,
        "mean_seen_count": float(np.mean(support_values)) if support_values else 0.0,
        "median_seen_count": float(np.median(support_values)) if support_values else 0.0,
        "label_vote_entropy": entropy_from_counts(vote_counts.values()),
        "voxel_majority_entropy": entropy_from_counts(majority_counts.values()),
        "voxel_majority_top_fraction": (
            float(ordered_majority[0][1] / supported) if supported and ordered_majority else 0.0
        ),
        "voxel_majority_second_fraction": float(second_fraction),
        "mean_voxel_disagreement": float(np.mean(disagreements)) if disagreements else 0.0,
        "primary_evidence_label_id": int(primary_label) if primary_label is not None else None,
        "second_evidence_label_id": int(second_label) if second_label is not None else None,
        "second_label_largest_region_fraction": float(largest_second_region / max(stable, 1)),
        "major_label_region_count": int(major_regions),
        "label_hist": {str(key): int(value) for key, value in vote_counts.most_common()},
    }
    return result, vote_counts


def object_gt_metrics(
    object_keys: np.ndarray,
    gt_objects: list[dict],
    gt_global_keys: np.ndarray,
) -> dict:
    overlaps = []
    for gt in gt_objects:
        count = int(
            np.intersect1d(object_keys, gt["voxel_keys"], assume_unique=True).size
        )
        if count:
            overlaps.append(
                {
                    "gt_id": int(gt["gt_id"]),
                    "gt_label": str(gt["gt_label"]),
                    "voxels": count,
                    "pred_fraction": float(count / max(len(object_keys), 1)),
                }
            )
    overlaps.sort(key=lambda item: (-item["voxels"], item["gt_id"]))
    supported = int(np.intersect1d(object_keys, gt_global_keys, assume_unique=True).size)
    dominant = overlaps[0] if overlaps else None
    second = overlaps[1] if len(overlaps) > 1 else None
    substantial_threshold = max(3, int(math.ceil(0.05 * max(len(object_keys), 1))))
    substantial = [item for item in overlaps if item["voxels"] >= substantial_threshold]
    overmerge = bool(
        len(substantial) >= 2
        and second is not None
        and second["pred_fraction"] >= 0.05
    )
    cross_class = bool(
        overmerge and len({item["gt_label"] for item in substantial}) >= 2
    )
    return {
        "gt_supported_voxels": supported,
        "gt_supported_fraction": float(supported / max(len(object_keys), 1)),
        "gt_dominant_id": dominant["gt_id"] if dominant else None,
        "gt_dominant_label": dominant["gt_label"] if dominant else None,
        "gt_dominant_fraction": dominant["pred_fraction"] if dominant else 0.0,
        "gt_second_id": second["gt_id"] if second else None,
        "gt_second_label": second["gt_label"] if second else None,
        "gt_second_fraction": second["pred_fraction"] if second else 0.0,
        "gt_substantial_instances": int(len(substantial)),
        "gt_overmerge": overmerge,
        "gt_overmerge_cross_class": cross_class,
        "gt_overmerge_same_class": bool(overmerge and not cross_class),
        "gt_overlap_hist": overlaps,
    }


def bbox_gap(left: np.ndarray, right: np.ndarray) -> int:
    left_min, left_max = left.min(axis=0), left.max(axis=0)
    right_min, right_max = right.min(axis=0), right.max(axis=0)
    gap = np.maximum(np.maximum(right_min - left_max - 1, left_min - right_max - 1), 0)
    return int(gap.max())


def pair_rows_for_scale(
    object_rows: list[dict],
    object_keys: list[np.ndarray],
    label_vectors: list[Counter[int]],
    scale: float,
) -> list[dict]:
    rows = []
    foreground = [
        index
        for index, row in enumerate(object_rows)
        if not row["is_background"] and row["object_voxels"] >= 3
    ]
    coords = {index: unpack_keys(object_keys[index]) for index in foreground}
    for left_position, left_index in enumerate(foreground):
        left_keys = object_keys[left_index]
        left_coords = coords[left_index]
        for right_index in foreground[left_position + 1 :]:
            right_keys = object_keys[right_index]
            right_coords = coords[right_index]
            gap = bbox_gap(left_coords, right_coords)
            if gap > 2:
                continue
            intersection = int(
                np.intersect1d(left_keys, right_keys, assume_unique=True).size
            )
            union = len(left_keys) + len(right_keys) - intersection
            exact_iou = intersection / max(union, 1)
            if len(left_coords) <= len(right_coords):
                query, reference = left_coords, right_coords
            else:
                query, reference = right_coords, left_coords
            distances = cKDTree(reference).query(query, k=1, workers=-1)[0]
            minimum_grid = float(distances.min()) if len(distances) else float("inf")
            contact1 = float(np.mean(distances <= math.sqrt(3) + 1e-9))
            contact2 = float(np.mean(distances <= 2 * math.sqrt(3) + 1e-9))
            label_similarity = cosine_counts(
                label_vectors[left_index], label_vectors[right_index]
            )
            spatial1 = max(exact_iou, contact1)
            spatial2 = max(exact_iou, contact2)
            left = object_rows[left_index]
            right = object_rows[right_index]
            same_gt = bool(
                left["gt_dominant_id"] is not None
                and left["gt_dominant_id"] == right["gt_dominant_id"]
                and left["gt_dominant_fraction"] >= 0.25
                and right["gt_dominant_fraction"] >= 0.25
                and left["gt_supported_voxels"] >= 3
                and right["gt_supported_voxels"] >= 3
            )
            rows.append(
                {
                    "scene": left["scene"],
                    "voxel_size_m": scale,
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_uid": left["object_uid"],
                    "right_uid": right["object_uid"],
                    "left_label": left["predicted_label"],
                    "right_label": right["predicted_label"],
                    "bbox_gap_voxels": gap,
                    "exact_voxel_iou": float(exact_iou),
                    "min_distance_m": float(minimum_grid * scale),
                    "contact_fraction_r1": contact1,
                    "contact_fraction_r2": contact2,
                    "label_hist_cosine": label_similarity,
                    "duplicate_score_r1": float(label_similarity * spatial1),
                    "duplicate_score_r2": float(label_similarity * spatial2),
                    "gt_false_split_pair": same_gt,
                    "gt_instance_id": left["gt_dominant_id"] if same_gt else None,
                    "gt_instance_label": left["gt_dominant_label"] if same_gt else None,
                }
            )
    return rows


def summarize_observation_causes(
    observations: list[dict],
    object_rows: list[dict],
) -> None:
    by_owner = defaultdict(list)
    for observation in observations:
        owner = observation.get("owner_index")
        if owner is not None:
            by_owner[int(owner)].append(observation)
    for index, row in enumerate(object_rows):
        members = by_owner.get(index, [])
        eligible = [item for item in members if item["gt_assignment_eligible"]]
        mixed = [item for item in eligible if item["mask_mixed"]]
        pure = [item for item in eligible if not item["mask_mixed"] and item["gt_purity"] >= 0.8]
        wrong = [
            item
            for item in pure
            if row["gt_dominant_id"] is not None
            and item["gt_top_id"] != row["gt_dominant_id"]
        ]
        row.update(
            {
                "member_observations": int(len(members)),
                "gt_eligible_member_observations": int(len(eligible)),
                "mixed_mask_observations": int(len(mixed)),
                "mixed_mask_fraction": float(len(mixed) / max(len(eligible), 1)),
                "pure_member_observations": int(len(pure)),
                "wrong_association_observations": int(len(wrong)),
                "wrong_association_fraction": float(len(wrong) / max(len(pure), 1)),
                "has_mask_evidence_error": bool(mixed),
                "has_association_evidence_error": bool(
                    len(wrong) >= 2 or (len(wrong) >= 1 and len(wrong) / max(len(pure), 1) >= 0.05)
                ),
            }
        )


def finalize_scores_and_metrics(
    object_rows: list[dict],
    pair_rows: list[dict],
) -> dict:
    incident_score = np.zeros(len(object_rows), dtype=float)
    split_incident = np.zeros(len(object_rows), dtype=bool)
    for pair in pair_rows:
        left, right = int(pair["left_index"]), int(pair["right_index"])
        score = float(pair["duplicate_score_r1"])
        incident_score[left] = max(incident_score[left], score)
        incident_score[right] = max(incident_score[right], score)
        if pair["gt_false_split_pair"]:
            split_incident[left] = True
            split_incident[right] = True

    foreground = [index for index, row in enumerate(object_rows) if not row["is_background"]]
    semantic_features = {
        "voxel_majority_entropy": percentile_ranks(
            [object_rows[index]["voxel_majority_entropy"] for index in foreground]
        ),
        "second_label_largest_region_fraction": percentile_ranks(
            [
                object_rows[index]["second_label_largest_region_fraction"]
                for index in foreground
            ]
        ),
        "mean_voxel_disagreement": percentile_ranks(
            [object_rows[index]["mean_voxel_disagreement"] for index in foreground]
        ),
    }
    fragmentation_rank = percentile_ranks(
        [object_rows[index]["fragmentation_score"] for index in foreground]
    )
    duplicate_rank = percentile_ranks([incident_score[index] for index in foreground])
    nonspatial_rank = percentile_ranks(
        [object_rows[index]["owner_label_entropy"] for index in foreground]
    )
    for position, index in enumerate(foreground):
        semantic_rank = max(values[position] for values in semantic_features.values())
        object_rows[index].update(
            {
                "semantic_conflict_score": float(semantic_rank),
                "fragmentation_rank": float(fragmentation_rank[position]),
                "best_duplicate_pair_score": float(incident_score[index]),
                "duplicate_conflict_score": float(duplicate_rank[position]),
                "nonspatial_label_score": float(nonspatial_rank[position]),
                "gt_false_split_incident": bool(split_incident[index]),
            }
        )
        object_rows[index]["combined_anomaly_score"] = float(
            max(
                semantic_rank,
                fragmentation_rank[position],
                duplicate_rank[position],
            )
        )
        object_rows[index]["gt_identity_error"] = bool(
            object_rows[index]["gt_overmerge"]
            or object_rows[index]["has_association_evidence_error"]
            or split_incident[index]
        )
    for index, row in enumerate(object_rows):
        if index not in foreground:
            row.update(
                {
                    "semantic_conflict_score": 0.0,
                    "fragmentation_rank": 0.0,
                    "best_duplicate_pair_score": 0.0,
                    "duplicate_conflict_score": 0.0,
                    "nonspatial_label_score": 0.0,
                    "combined_anomaly_score": 0.0,
                    "gt_false_split_incident": False,
                    "gt_identity_error": False,
                }
            )

    selected = [object_rows[index] for index in foreground]
    targets = {
        "gt_identity_error": [row["gt_identity_error"] for row in selected],
        "gt_overmerge": [row["gt_overmerge"] for row in selected],
        "gt_cross_class_overmerge": [row["gt_overmerge_cross_class"] for row in selected],
        "gt_same_class_overmerge": [row["gt_overmerge_same_class"] for row in selected],
        "gt_wrong_association": [row["has_association_evidence_error"] for row in selected],
        "gt_split_incident": [row["gt_false_split_incident"] for row in selected],
    }
    scores = {
        "nonspatial_label_entropy": [row["nonspatial_label_score"] for row in selected],
        "voxel_semantic_conflict": [row["semantic_conflict_score"] for row in selected],
        "fragmentation": [row["fragmentation_rank"] for row in selected],
        "duplicate_incident": [row["duplicate_conflict_score"] for row in selected],
        "combined": [row["combined_anomaly_score"] for row in selected],
    }
    evaluation = {
        score_name: {
            target_name: binary_metrics(target_values, score_values)
            for target_name, target_values in targets.items()
        }
        for score_name, score_values in scores.items()
    }
    pair_evaluation = {
        "r1": binary_metrics(
            [row["gt_false_split_pair"] for row in pair_rows],
            [row["duplicate_score_r1"] for row in pair_rows],
        ),
        "r2": binary_metrics(
            [row["gt_false_split_pair"] for row in pair_rows],
            [row["duplicate_score_r2"] for row in pair_rows],
        ),
    }
    return {
        "foreground_objects": len(selected),
        "object_evaluation": evaluation,
        "pair_evaluation": pair_evaluation,
    }


def build_scale_analysis(
    *,
    scene: str,
    scale: float,
    baseline: dict,
    gt_payload: dict,
    observations: list[dict],
    observation_voxels: list[np.ndarray],
    output_root: Path,
    scope_name: str,
    selected: np.ndarray,
) -> dict:
    scale_name = f"voxel_{scale:.3f}".replace(".", "p")
    scale_root = output_root / scale_name / scope_name
    observation_labels = np.asarray(
        [int(item["label_id"]) for item in observations], dtype=np.int32
    )
    voxel_map = build_voxel_map(
        observation_voxels=observation_voxels,
        observation_labels=observation_labels,
        selected=selected,
        output=scale_root / "voxel_map.npz",
    )

    gt_objects = []
    for obj in gt_payload["objects"]:
        gt_id = obj.get("oracle_gt_id")
        if gt_id is None:
            continue
        keys = voxel_keys(np.asarray(obj["pcd_np"], dtype=np.float64), scale)
        if not len(keys):
            continue
        gt_objects.append(
            {
                "gt_id": int(gt_id),
                "gt_label": str(obj.get("oracle_gt_label") or obj.get("class_name")),
                "voxel_keys": keys,
            }
        )
    gt_global_keys = np.unique(
        np.concatenate([item["voxel_keys"] for item in gt_objects])
    )

    object_rows = []
    object_key_sets = []
    label_vectors = []
    class_names = list(baseline["class_names"])
    for object_index, obj in enumerate(baseline["objects"]):
        keys = voxel_keys(np.asarray(obj["pcd_np"], dtype=np.float64), scale)
        object_key_sets.append(keys)
        evidence, label_vector = object_evidence(keys, voxel_map)
        label_vectors.append(label_vector)
        components = component_sizes(keys)
        major_threshold = max(3, int(math.ceil(0.01 * max(len(keys), 1))))
        major_components = [size for size in components if size >= major_threshold]
        owner_hist = Counter(int(value) for value in obj.get("class_id", []))
        predicted_label = str(obj.get("class_name") or "unknown")
        if predicted_label == "unknown" and owner_hist:
            predicted_label = class_names[owner_hist.most_common(1)[0][0]]
        row = {
            "scene": scene,
            "voxel_size_m": scale,
            "evidence_scope": scope_name,
            "object_index": object_index,
            "object_uid": str(obj.get("id")),
            "predicted_label": predicted_label,
            "is_background": bool(obj.get("is_background", False) or predicted_label in BG_LABELS),
            "object_voxels": int(len(keys)),
            "num_detections": int(obj.get("num_detections", len(obj.get("class_id", [])))),
            "owner_label_entropy": entropy_from_counts(owner_hist.values()),
            "connected_components_26": int(len(components)),
            "major_connected_components_26": int(len(major_components)),
            "largest_component_fraction": float(components[0] / max(len(keys), 1)) if components else 0.0,
            "fragmentation_score": float(1.0 - components[0] / max(len(keys), 1)) if components else 1.0,
            **evidence,
            **object_gt_metrics(keys, gt_objects, gt_global_keys),
        }
        primary_id = row.get("primary_evidence_label_id")
        second_id = row.get("second_evidence_label_id")
        row["primary_evidence_label"] = (
            class_names[int(primary_id)] if primary_id is not None and int(primary_id) < len(class_names) else None
        )
        row["second_evidence_label"] = (
            class_names[int(second_id)] if second_id is not None and int(second_id) < len(class_names) else None
        )
        object_rows.append(row)

    summarize_observation_causes(observations, object_rows)
    pair_rows = pair_rows_for_scale(object_rows, object_key_sets, label_vectors, scale)
    evaluation = finalize_scores_and_metrics(object_rows, pair_rows)
    write_jsonl(scale_root / "objects.jsonl", object_rows)
    write_jsonl(scale_root / "pairs.jsonl", pair_rows)
    write_csv(scale_root / "objects.csv", object_rows)
    write_csv(scale_root / "pairs.csv", pair_rows)
    summary = {
        "scene": scene,
        "voxel_size_m": scale,
        "evidence_scope": scope_name,
        "voxel_payload_fields": ["seen_count", "label_hist", "obs_ids"],
        "voxel_count": int(len(voxel_map.voxel_keys)),
        "voxel_observation_links": int(len(voxel_map.obs_ids)),
        "selected_observations": int(selected.sum()),
        "object_count": len(object_rows),
        "neighbor_pair_count": len(pair_rows),
        "evaluation": evaluation,
    }
    atomic_json(scale_root / "summary.json", summary)
    return {
        "summary": summary,
        "objects": object_rows,
        "pairs": pair_rows,
    }


def backend_imports(code_root: Path):
    sys.path.insert(0, str(code_root.resolve()))
    from omegaconf import OmegaConf
    from conceptgraph.dataset.datasets_common import get_dataset
    from conceptgraph.slam.utils import (
        detections_to_obj_pcd_and_bbox,
        filter_gobs,
        init_process_pcd,
        resize_gobs,
    )
    from conceptgraph.utils.general_utils import ObjectClasses, load_saved_detections
    from conceptgraph.utils.ious import mask_subtract_contained

    return {
        "OmegaConf": OmegaConf,
        "get_dataset": get_dataset,
        "detections_to_obj_pcd_and_bbox": detections_to_obj_pcd_and_bbox,
        "filter_gobs": filter_gobs,
        "init_process_pcd": init_process_pcd,
        "resize_gobs": resize_gobs,
        "ObjectClasses": ObjectClasses,
        "load_saved_detections": load_saved_detections,
        "mask_subtract_contained": mask_subtract_contained,
    }


def final_membership_lookup(baseline: dict) -> tuple[dict, dict]:
    lookup = {}
    duplicates = []
    for object_index, obj in enumerate(baseline["objects"]):
        lengths = {
            len(obj.get("color_path", [])),
            len(obj.get("mask_idx", [])),
            len(obj.get("mask", [])),
            len(obj.get("class_id", [])),
        }
        if len(lengths) != 1:
            raise ValueError(f"object {object_index} member length mismatch: {lengths}")
        for position, (color_path, mask_index, mask, class_id) in enumerate(
            zip(obj["color_path"], obj["mask_idx"], obj["mask"], obj["class_id"])
        ):
            key = (Path(str(color_path)).stem, int(mask_index))
            row = {
                "owner_index": object_index,
                "owner_uid": str(obj.get("id")),
                "position": position,
                "mask": np.asarray(mask, dtype=bool),
                "class_id": int(class_id),
            }
            if key in lookup:
                previous = lookup[key]
                duplicates.append(
                    {
                        "frame": key[0],
                        "mask_idx": key[1],
                        "previous_owner": previous["owner_index"],
                        "owner": object_index,
                        "same_owner": previous["owner_index"] == object_index,
                        "same_mask": bool(np.array_equal(previous["mask"], row["mask"])),
                        "same_label": previous["class_id"] == row["class_id"],
                    }
                )
                if previous["owner_index"] != object_index:
                    raise ValueError(f"observation {key} belongs to multiple final objects")
            else:
                lookup[key] = row
    return lookup, {"duplicates": duplicates, "duplicate_count": len(duplicates)}


def process_scene(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    scene = args.scene
    spec = SCENE_SPECS[scene]
    output = args.output_root.resolve() / scene
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix outputs in non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "INCOMPLETE").write_text(f"started={time.time()}\n", encoding="utf-8")

    backend = backend_imports(args.code_root)
    baseline_path = Path(spec["baseline_map"]).resolve()
    gt_map_path = Path(spec["gt_map"]).resolve()
    baseline = load_pickle_gz(baseline_path)
    gt_payload = load_pickle_gz(gt_map_path)
    cfg = backend["OmegaConf"].create(baseline["cfg"])
    cfg.device = args.device
    dataset_cfg = backend["OmegaConf"].load(Path(str(cfg.dataset_config)).resolve())
    if cfg.image_height is None:
        cfg.image_height = dataset_cfg.camera_params.image_height
    if cfg.image_width is None:
        cfg.image_width = dataset_cfg.camera_params.image_width
    dataset = backend["get_dataset"](
        dataconfig=Path(str(cfg.dataset_config)).resolve(),
        start=int(cfg.start),
        end=int(cfg.end),
        stride=int(cfg.stride),
        basedir=Path(str(cfg.dataset_root)).resolve(),
        sequence=scene,
        desired_height=int(cfg.image_height),
        desired_width=int(cfg.image_width),
        device="cpu",
        dtype=__import__("torch").float,
    )
    frame_count = min(len(dataset), int(args.max_frames))
    if args.smoke:
        frame_count = min(frame_count, 3)
    detection_root = (
        Path(str(cfg.dataset_root)).resolve()
        / scene
        / "exps"
        / str(cfg.detections_exp_suffix)
        / "detections"
    )
    if not detection_root.is_dir():
        raise FileNotFoundError(detection_root)
    gt_sidecars = args.gt_sidecars.resolve() / scene
    if not gt_sidecars.is_dir():
        raise FileNotFoundError(gt_sidecars)
    instance_labels = load_instance_labels(args.objects_json.resolve(), spec["source_scene"])
    object_classes = backend["ObjectClasses"](
        classes_file_path=Path(str(cfg.classes_file)),
        bg_classes=list(cfg.bg_classes),
        skip_bg=bool(cfg.skip_bg),
    )
    final_lookup, duplicate_audit = final_membership_lookup(baseline)

    observations = []
    observation_voxels_by_scale: dict[float, list[np.ndarray]] = {
        scale: [] for scale in SCALES
    }
    parity = Counter()
    seen_final_keys = set()
    counters = Counter()
    frame_rows = []
    for frame_index in range(frame_count):
        frame_started = time.perf_counter()
        color_path = Path(dataset.color_paths[frame_index])
        color_stem = color_path.stem
        raw_frame = int(color_stem.replace("frame", ""))
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_index]
        image_rgb = color_tensor.cpu().numpy().astype(np.uint8)
        depth_array = depth_tensor[..., 0].cpu().numpy()
        pose = dataset.poses[frame_index].cpu().numpy()
        intrinsics_np = intrinsics.cpu().numpy()
        loaded = backend["load_saved_detections"](detection_root / color_stem)
        raw_gobs = backend["resize_gobs"](copy.deepcopy(loaded), image_rgb)
        counters["raw_detections"] += len(raw_gobs["mask"])
        filtered = copy.deepcopy(raw_gobs)
        filtered["source_mask_index"] = np.arange(len(raw_gobs["mask"]), dtype=np.int32)
        filtered = backend["filter_gobs"](
            filtered,
            image_rgb,
            skip_bg=bool(cfg.skip_bg),
            BG_CLASSES=object_classes.get_bg_classes_arr(),
            mask_area_threshold=float(cfg.mask_area_threshold),
            max_bbox_area_ratio=float(cfg.max_bbox_area_ratio),
            mask_conf_threshold=float(cfg.mask_conf_threshold),
        )
        if len(filtered["mask"]):
            filtered["mask"] = backend["mask_subtract_contained"](
                filtered["xyxy"], filtered["mask"]
            )
        masks = np.asarray(filtered["mask"], dtype=bool)
        counters["filtered_2d_observations"] += len(masks)
        pcds = (
            backend["detections_to_obj_pcd_and_bbox"](
                depth_array=depth_array,
                masks=masks,
                cam_K=intrinsics_np[:3, :3],
                image_rgb=image_rgb,
                trans_pose=pose,
                min_points_threshold=int(cfg.min_points_threshold),
                spatial_sim_type=str(cfg.spatial_sim_type),
                obj_pcd_max_points=int(cfg.obj_pcd_max_points),
                device=str(cfg.device),
            )
            if len(masks)
            else []
        )
        with np.load(gt_sidecars / f"frame{raw_frame:06d}.npz") as handle:
            semantic = np.asarray(handle["semantic"], dtype=np.uint16)
        if semantic.shape != masks.shape[1:] and len(masks):
            raise ValueError(
                f"frame {raw_frame}: GT {semantic.shape} != masks {masks.shape[1:]}"
            )
        accepted_this_frame = 0
        for mask_index, mask in enumerate(masks):
            key = (color_stem, int(mask_index))
            final = final_lookup.get(key)
            if final is not None:
                seen_final_keys.add(key)
                parity["final_members_seen"] += 1
                if np.array_equal(mask, final["mask"]):
                    parity["mask_exact"] += 1
                else:
                    parity["mask_mismatch"] += 1
                if int(filtered["class_id"][mask_index]) == final["class_id"]:
                    parity["label_exact"] += 1
                else:
                    parity["label_mismatch"] += 1
            item = pcds[mask_index] if mask_index < len(pcds) else None
            if item is None:
                counters["rejected_3d"] += 1
                continue
            processed_pcd = backend["init_process_pcd"](
                pcd=item["pcd"],
                downsample_voxel_size=float(cfg.downsample_voxel_size),
                dbscan_remove_noise=bool(cfg.dbscan_remove_noise),
                dbscan_eps=float(cfg.dbscan_eps),
                dbscan_min_points=int(cfg.dbscan_min_points),
            )
            points = np.asarray(processed_pcd.points, dtype=np.float64)
            if not len(points):
                counters["rejected_empty_after_dbscan"] += 1
                continue
            label_id = int(filtered["class_id"][mask_index])
            label_name = str(baseline["class_names"][label_id])
            assignment = mask_gt_assignment(mask, semantic, instance_labels)
            observation_index = len(observations)
            observation = {
                "observation_index": observation_index,
                "obs_uid": f"{scene}:{color_stem}:m{mask_index:04d}",
                "scene": scene,
                "processed_frame": frame_index,
                "raw_frame": raw_frame,
                "color_stem": color_stem,
                "mask_idx": int(mask_index),
                "source_raw_mask_idx": int(filtered["source_mask_index"][mask_index]),
                "label_id": label_id,
                "label": label_name,
                "confidence": float(filtered["confidence"][mask_index]),
                "point_count_after_dbscan": int(len(points)),
                "owner_index": int(final["owner_index"]) if final is not None else None,
                "owner_uid": final["owner_uid"] if final is not None else None,
                **assignment,
            }
            observations.append(observation)
            for scale in SCALES:
                observation_voxels_by_scale[scale].append(voxel_keys(points, scale))
            counters["accepted_3d_observations"] += 1
            if final is not None:
                counters["accepted_final_members"] += 1
            else:
                counters["accepted_nonfinal_observations"] += 1
            accepted_this_frame += 1
        frame_rows.append(
            {
                "processed_frame": frame_index,
                "raw_frame": raw_frame,
                "raw_detections": int(len(raw_gobs["mask"])),
                "filtered_2d": int(len(masks)),
                "accepted_3d": int(accepted_this_frame),
                "elapsed_seconds": float(time.perf_counter() - frame_started),
            }
        )
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == frame_count:
            print(
                f"{scene}: frames {frame_index + 1}/{frame_count}, "
                f"accepted={counters['accepted_3d_observations']}, "
                f"final={counters['accepted_final_members']}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    expected_final = {
        key for key in final_lookup if int(key[0].replace("frame", "")) <= (frame_count - 1) * int(cfg.stride)
    }
    missing_final = sorted(expected_final - seen_final_keys)
    parity.update(
        {
            "unique_final_member_keys": len(final_lookup),
            "expected_final_member_keys": len(expected_final),
            "missing_final_member_keys": len(missing_final),
        }
    )
    if not args.smoke:
        if parity["mask_mismatch"] or parity["label_mismatch"] or missing_final:
            raise RuntimeError(f"final-member parity failed: {dict(parity)}")

    write_jsonl(output / "observations.jsonl", observations)
    write_csv(output / "frames.csv", frame_rows)
    atomic_json(
        output / "observation_index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "labels": {str(index): str(name) for index, name in enumerate(baseline["class_names"])},
            "observations": [item["obs_uid"] for item in observations],
        },
    )

    if args.smoke:
        smoke_summary = {
            "scene": scene,
            "frame_count": frame_count,
            "counters": dict(counters),
            "parity": dict(parity),
            "duplicate_audit": duplicate_audit,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(output / "smoke_summary.json", smoke_summary)
        (output / "INCOMPLETE").unlink(missing_ok=True)
        (output / "READY").write_text("smoke\n", encoding="utf-8")
        print(json.dumps(smoke_summary, indent=2, ensure_ascii=False), flush=True)
        return 0

    all_selected = np.ones(len(observations), dtype=bool)
    final_selected = np.asarray(
        [item["owner_index"] is not None for item in observations], dtype=bool
    )
    analyses = {}
    for scale in SCALES:
        analyses[str(scale)] = {}
        for scope_name, selected in (
            ("all_history", all_selected),
            ("final_members_only", final_selected),
        ):
            analyses[str(scale)][scope_name] = build_scale_analysis(
                scene=scene,
                scale=scale,
                baseline=baseline,
                gt_payload=gt_payload,
                observations=observations,
                observation_voxels=observation_voxels_by_scale[scale],
                output_root=output,
                scope_name=scope_name,
                selected=selected,
            )["summary"]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_date": RUN_DATE,
        "scene": scene,
        "source_scene": spec["source_scene"],
        "frame_count": frame_count,
        "voxel_scales_m": list(SCALES),
        "primary_voxel_scale_m": PRIMARY_SCALE,
        "voxel_payload_fields": ["seen_count", "label_hist", "obs_ids"],
        "gt_usage": "evaluation_only_not_used_in_voxel_content_or_scores",
        "pre_registered_decision_gates": DECISION_GATES,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "baseline_map": str(baseline_path),
        "baseline_map_sha256": sha256_file(baseline_path),
        "gt_map": str(gt_map_path),
        "gt_map_sha256": sha256_file(gt_map_path),
        "detection_root": str(detection_root),
        "gt_sidecars": str(gt_sidecars),
        "code_root": str(args.code_root.resolve()),
        "device": str(cfg.device),
        "counters": dict(counters),
        "parity": dict(parity),
        "final_membership_duplicates": duplicate_audit,
        "analyses": analyses,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    atomic_json(output / "manifest.json", manifest)
    (output / "INCOMPLETE").unlink(missing_ok=True)
    (output / "READY").write_text("ready\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


def run_self_test() -> dict:
    coords = np.asarray([[-2, 3, 4], [0, 0, 0], [8, -7, 6]], dtype=np.int32)
    assert np.array_equal(unpack_keys(pack_coords(coords)), coords)
    connected = pack_coords(np.asarray([[0, 0, 0], [1, 1, 1], [5, 5, 5]]))
    assert component_sizes(connected) == [2, 1]
    assert component_sizes(connected, connectivity=6) == [1, 1, 1]
    assignment = mask_gt_assignment(
        np.asarray([[1, 1], [1, 0]], dtype=bool),
        np.asarray([[1, 1], [2, 0]], dtype=np.uint16),
        {1: "chair", 2: "table"},
    )
    assert assignment["gt_top_id"] == 1
    metrics = binary_metrics([True, False, True, False], [0.9, 0.1, 0.8, 0.2])
    assert metrics["average_precision"] == 1.0
    return {
        "pack_roundtrip": "PASS",
        "components_26": "PASS",
        "components_6": "PASS",
        "mask_assignment": "PASS",
        "binary_metrics": "PASS",
    }


def main() -> int:
    args = parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0
    return process_scene(args)


if __name__ == "__main__":
    raise SystemExit(main())
