#!/usr/bin/env python3
"""Independent re-audit of the simple voxel trigger V0.

The script never rebuilds or mutates the source maps.  It checks the frozen
voxel ledgers, derives alternative statistics from the same minimal payload,
and evaluates them against both stricter observation-sidecar targets and an
exclusive nearest-GT 3D attribution.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


SCENES = ("room0", "office0")
SCALES = (0.025, 0.05, 0.10)
PRIMARY_SCALE = 0.05
INVALID_INSTANCE_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "unknown",
    "undefined",
    "background",
    "none",
    "",
}
BITS = 21
OFFSET = 1 << (BITS - 1)
FIELD_MASK = (1 << BITS) - 1
SHIFT_X = BITS * 2
SHIFT_Y = BITS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
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
        writer.writerows(rows)
    temporary.replace(path)


def load_pickle_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def scale_tag(scale: float) -> str:
    return f"voxel_{scale:0.3f}".replace(".", "p")


def percentile(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    output = np.zeros(len(array), dtype=float)
    finite = np.isfinite(array)
    if finite.any():
        output[finite] = rankdata(array[finite], method="average") / int(finite.sum())
    return output


def entropy(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if float(value) > 0], dtype=float)
    if len(array) <= 1:
        return 0.0
    probability = array / array.sum()
    return float(-(probability * np.log(probability)).sum() / np.log(len(array)))


def pack_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    shifted = coords + OFFSET
    if np.any(shifted < 0) or np.any(shifted > FIELD_MASK):
        raise ValueError("coordinate outside packed range")
    return (
        (shifted[:, 0] << SHIFT_X)
        | (shifted[:, 1] << SHIFT_Y)
        | shifted[:, 2]
    ).astype(np.int64, copy=False)


def voxel_keys(points: np.ndarray, scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    quantized = np.floor(points[finite] / scale).astype(np.int64)
    return np.unique(pack_coords(quantized))


def binary_metrics(labels: Iterable[bool], scores: Iterable[float]) -> dict:
    y = np.asarray(list(labels), dtype=np.uint8)
    score = np.asarray(list(scores), dtype=float)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    output = {
        "n": int(len(y)),
        "positives": int(y.sum()) if len(y) else 0,
        "prevalence": float(y.mean()) if len(y) else None,
        "average_precision": None,
        "auroc": None,
        "ap_minus_prevalence": None,
    }
    if len(y) and len(np.unique(y)) == 2:
        output["average_precision"] = float(average_precision_score(y, score))
        output["auroc"] = float(roc_auc_score(y, score))
        output["ap_minus_prevalence"] = float(
            output["average_precision"] - output["prevalence"]
        )
    if len(y):
        order = np.argsort(-score, kind="stable")
        count = max(1, int(math.ceil(0.2 * len(y))))
        top = float(y[order[:count]].mean())
        bottom = float(y[order[-count:]].mean())
        output.update(
            {
                "top20_n": int(count),
                "top20_errors": int(y[order[:count]].sum()),
                "top20_error_rate": top,
                "bottom20_errors": int(y[order[-count:]].sum()),
                "bottom20_error_rate": bottom,
                "top_bottom_ratio": float(top / bottom) if bottom else None,
            }
        )
    return output


def valid_foreground_label(label: object) -> bool:
    return str(label or "").strip().lower() not in INVALID_INSTANCE_LABELS


def build_sidecar_targets(
    observations: list[dict],
    object_rows: list[dict],
    *,
    strict: bool,
    conservative: bool = False,
) -> tuple[dict[int, dict], set[int]]:
    by_owner: dict[int, list[dict]] = defaultdict(list)
    for item in observations:
        owner = item.get("owner_index")
        if owner is not None:
            by_owner[int(owner)].append(item)

    audits: dict[int, dict] = {}
    reliable_by_gt: dict[int, list[int]] = defaultdict(list)
    for row in object_rows:
        index = int(row["object_index"])
        members = by_owner.get(index, [])
        eligible = [
            item
            for item in members
            if bool(item.get("gt_assignment_eligible"))
            and item.get("gt_top_id") is not None
        ]
        if strict:
            eligible = [
                item
                for item in eligible
                if float(item.get("gt_supported_fraction", 0.0)) >= (0.9 if conservative else 0.8)
                and valid_foreground_label(item.get("gt_top_label"))
            ]
        if conservative:
            mixed = [
                item
                for item in eligible
                if float(item.get("gt_purity", 0.0)) < 0.7
                or float(item.get("gt_second_fraction", 0.0)) >= 0.2
            ]
        else:
            mixed = [item for item in eligible if bool(item.get("mask_mixed"))]
        pure = [
            item
            for item in eligible
            if not bool(item.get("mask_mixed"))
            and float(item.get("gt_purity", 0.0)) >= (0.9 if conservative else 0.8)
        ]
        counts = Counter(int(item["gt_top_id"]) for item in pure)
        label_by_id = {
            int(item["gt_top_id"]): str(item.get("gt_top_label") or "unknown")
            for item in pure
        }
        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        dominant_id = ordered[0][0] if ordered else None
        dominant_count = ordered[0][1] if ordered else 0
        wrong_count = int(len(pure) - dominant_count)
        wrong_fraction = float(wrong_count / max(len(pure), 1))
        mixed_fraction = float(len(mixed) / max(len(eligible), 1))
        fraction_threshold = 0.10 if conservative else 0.05
        mask_repeated = bool(len(mixed) >= 2 and mixed_fraction >= fraction_threshold)
        association_repeated = bool(wrong_count >= 2 and wrong_fraction >= fraction_threshold)
        dominant_label = label_by_id.get(dominant_id) if dominant_id is not None else None
        reliable_label = (
            valid_foreground_label(dominant_label)
            if strict
            else str(dominant_label or "").strip().lower() not in {"wall", "floor", "ceiling"}
        )
        reliable = bool(
            dominant_id is not None
            and reliable_label
            and dominant_count >= (3 if conservative else 2)
            and dominant_count / max(len(pure), 1) >= (0.7 if conservative else 0.5)
        )
        if reliable:
            reliable_by_gt[int(dominant_id)].append(index)
        audits[index] = {
            "eligible": int(len(eligible)),
            "mixed": int(len(mixed)),
            "pure": int(len(pure)),
            "dominant_gt_id": int(dominant_id) if dominant_id is not None else None,
            "dominant_gt_label": dominant_label,
            "dominant_fraction": float(dominant_count / max(len(pure), 1)),
            "wrong": wrong_count,
            "wrong_fraction": wrong_fraction,
            "mask_repeated": mask_repeated,
            "association_repeated": association_repeated,
            "reliable_foreground": reliable,
        }

    split_incidents: set[int] = set()
    for indices in reliable_by_gt.values():
        if len(indices) >= 2:
            split_incidents.update(indices)
    for index, audit in audits.items():
        audit["split_incident"] = bool(index in split_incidents)
        audit["unified"] = bool(
            audit["mask_repeated"]
            or audit["association_repeated"]
            or audit["split_incident"]
        )
    return audits, split_incidents


def gt_geometry_index(gt_payload: dict, downsample: float = 0.025) -> tuple[cKDTree, np.ndarray, dict[int, str]]:
    coords: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    labels: dict[int, str] = {}
    for position, obj in enumerate(gt_payload["objects"]):
        gt_id = obj.get("oracle_gt_id")
        if gt_id is None:
            gt_id = position + 1
        gt_id = int(gt_id)
        label = str(obj.get("oracle_gt_label") or obj.get("class_name") or "unknown")
        labels[gt_id] = label
        points = np.asarray(obj["pcd_np"], dtype=np.float64)
        if not len(points):
            continue
        quantized = np.unique(np.floor(points / downsample).astype(np.int32), axis=0)
        centers = (quantized.astype(np.float64) + 0.5) * downsample
        coords.append(centers)
        ids.append(np.full(len(centers), gt_id, dtype=np.int32))
    all_coords = np.concatenate(coords, axis=0)
    all_ids = np.concatenate(ids, axis=0)
    return cKDTree(all_coords), all_ids, labels


def exclusive_geometry_queries(
    baseline: dict,
    gt_payload: dict,
    *,
    query_scale: float = 0.025,
) -> tuple[list[dict], dict[int, str]]:
    """Query nearest GT once; distance thresholds are applied later."""
    tree, gt_ids, gt_labels = gt_geometry_index(gt_payload, downsample=query_scale)
    queries: list[dict] = []
    for index, obj in enumerate(baseline["objects"]):
        points = np.asarray(obj["pcd_np"], dtype=np.float64)
        quantized = np.unique(np.floor(points / query_scale).astype(np.int32), axis=0)
        centers = (quantized.astype(np.float64) + 0.5) * query_scale
        if len(centers):
            distances, neighbors = tree.query(centers, k=1, workers=-1)
        else:
            distances = np.zeros(0, dtype=np.float32)
            neighbors = np.zeros(0, dtype=np.int64)
        queries.append(
            {
                "object_index": index,
                "point_count": int(len(centers)),
                "distances": np.asarray(distances, dtype=np.float32),
                "nearest_gt_ids": gt_ids[np.asarray(neighbors, dtype=np.int64)],
            }
        )
    return queries, gt_labels


def exclusive_geometry_targets(
    queries: list[dict],
    gt_labels: dict[int, str],
    *,
    max_distance: float = 0.05,
) -> tuple[dict[int, dict], set[int]]:
    audits: dict[int, dict] = {}
    reliable_by_gt: dict[int, list[int]] = defaultdict(list)
    for query in queries:
        index = int(query["object_index"])
        distances = np.asarray(query["distances"], dtype=np.float32)
        nearest_ids = np.asarray(query["nearest_gt_ids"], dtype=np.int32)
        assigned = nearest_ids[distances <= max_distance]
        point_count = int(query["point_count"])
        counts = Counter(int(value) for value in assigned.tolist())
        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        dominant_id = ordered[0][0] if ordered else None
        dominant_count = ordered[0][1] if ordered else 0
        second_count = ordered[1][1] if len(ordered) > 1 else 0
        support_fraction = float(len(assigned) / max(point_count, 1))
        dominant_fraction = float(dominant_count / max(len(assigned), 1))
        second_fraction = float(second_count / max(len(assigned), 1))
        dominant_label = gt_labels.get(dominant_id, "unknown") if dominant_id is not None else None
        reliable = bool(
            dominant_id is not None
            and valid_foreground_label(dominant_label)
            and len(assigned) >= 20
            and support_fraction >= 0.5
            and dominant_fraction >= 0.6
        )
        if reliable:
            reliable_by_gt[int(dominant_id)].append(index)
        overmerge = bool(
            reliable
            and second_count >= 10
            and second_fraction >= 0.10
            and valid_foreground_label(gt_labels.get(ordered[1][0]) if len(ordered) > 1 else None)
        )
        audits[index] = {
            "supported_voxels": int(len(assigned)),
            "support_fraction": support_fraction,
            "dominant_gt_id": int(dominant_id) if dominant_id is not None else None,
            "dominant_gt_label": dominant_label,
            "dominant_fraction": dominant_fraction,
            "second_fraction": second_fraction,
            "reliable_foreground": reliable,
            "overmerge": overmerge,
        }
    split_incidents: set[int] = set()
    for indices in reliable_by_gt.values():
        if len(indices) >= 2:
            split_incidents.update(indices)
    for index, audit in audits.items():
        audit["split_incident"] = bool(index in split_incidents)
        audit["partition_error"] = bool(audit["overmerge"] or audit["split_incident"])
    return audits, split_incidents


def audit_voxel_map(
    npz_path: Path,
    observations: list[dict],
) -> tuple[dict, dict[str, np.ndarray | list[Counter[int]]]]:
    with np.load(npz_path) as data:
        keys = np.asarray(data["voxel_keys"], dtype=np.int64)
        coords = np.asarray(data["voxel_coords"], dtype=np.int32)
        seen = np.asarray(data["seen_count"], dtype=np.int64)
        obs_offsets = np.asarray(data["obs_offsets"], dtype=np.int64)
        obs_ids = np.asarray(data["obs_ids"], dtype=np.int32)
        label_offsets = np.asarray(data["label_offsets"], dtype=np.int64)
        label_ids = np.asarray(data["label_ids"], dtype=np.int32)
        label_counts = np.asarray(data["label_counts"], dtype=np.int64)

    obs_label = np.asarray([int(item["label_id"]) for item in observations], dtype=np.int32)
    obs_frame = np.asarray([int(item["raw_frame"]) for item in observations], dtype=np.int32)
    obs_owner = np.asarray(
        [int(item["owner_index"]) if item.get("owner_index") is not None else -1 for item in observations],
        dtype=np.int32,
    )
    if not np.array_equal(seen, np.diff(obs_offsets)):
        raise AssertionError("seen_count does not equal obs offset lengths")
    if not np.array_equal(keys, pack_coords(coords)):
        raise AssertionError("voxel key/coordinate mismatch")

    unique_obs_ok = True
    label_hist_ok = True
    same_frame_extra = 0
    voxels_same_frame_multiobs = 0
    voxels_same_frame_multilabel = 0
    voxels_multi_owner = 0
    frame_counts = np.zeros(len(keys), dtype=np.int32)
    fb_dominant = np.full(len(keys), -1, dtype=np.int32)
    fb_disagreement = np.zeros(len(keys), dtype=np.float32)
    owner_entropy = np.zeros(len(keys), dtype=np.float32)
    owner_counts: list[Counter[int]] = []
    for voxel_index in range(len(keys)):
        begin, end = int(obs_offsets[voxel_index]), int(obs_offsets[voxel_index + 1])
        local_obs = obs_ids[begin:end]
        if len(np.unique(local_obs)) != len(local_obs):
            unique_obs_ok = False
        expected = Counter(int(obs_label[item]) for item in local_obs.tolist())
        lb, le = int(label_offsets[voxel_index]), int(label_offsets[voxel_index + 1])
        stored = Counter(
            {int(label): int(count) for label, count in zip(label_ids[lb:le], label_counts[lb:le])}
        )
        if expected != stored or sum(stored.values()) != int(seen[voxel_index]):
            label_hist_ok = False
        by_frame: dict[int, set[int]] = defaultdict(set)
        for item in local_obs.tolist():
            by_frame[int(obs_frame[item])].add(int(obs_label[item]))
        frame_counts[voxel_index] = len(by_frame)
        extra = len(local_obs) - len(by_frame)
        same_frame_extra += extra
        if extra > 0:
            voxels_same_frame_multiobs += 1
        if any(len(labels) > 1 for labels in by_frame.values()):
            voxels_same_frame_multilabel += 1
        balanced = Counter()
        for labels in by_frame.values():
            weight = 1.0 / len(labels)
            for label in labels:
                balanced[label] += weight
        if balanced:
            dominant_label, maximum = max(balanced.items(), key=lambda pair: (pair[1], -pair[0]))
            total = sum(balanced.values())
            fb_dominant[voxel_index] = int(dominant_label)
            fb_disagreement[voxel_index] = float(1.0 - maximum / total)
        owners = Counter(int(obs_owner[item]) for item in local_obs.tolist() if int(obs_owner[item]) >= 0)
        owner_counts.append(owners)
        owner_entropy[voxel_index] = entropy(owners.values())
        if len(owners) > 1:
            voxels_multi_owner += 1

    summary = {
        "voxel_count": int(len(keys)),
        "links": int(len(obs_ids)),
        "seen_matches_offsets": True,
        "voxel_keys_match_coords": True,
        "unique_observation_per_voxel": bool(unique_obs_ok),
        "label_hist_exact": bool(label_hist_ok),
        "same_frame_extra_links": int(same_frame_extra),
        "same_frame_extra_link_fraction": float(same_frame_extra / max(len(obs_ids), 1)),
        "voxels_with_same_frame_multiobs_fraction": float(voxels_same_frame_multiobs / max(len(keys), 1)),
        "voxels_with_same_frame_multilabel_fraction": float(voxels_same_frame_multilabel / max(len(keys), 1)),
        "multi_owner_voxel_fraction": float(voxels_multi_owner / max(len(keys), 1)),
        "median_unique_frames": float(np.median(frame_counts)),
        "mean_unique_frames": float(np.mean(frame_counts)),
    }
    derived: dict[str, np.ndarray | list[Counter[int]]] = {
        "keys": keys,
        "frame_counts": frame_counts,
        "fb_dominant": fb_dominant,
        "fb_disagreement": fb_disagreement,
        "owner_entropy": owner_entropy,
        "owner_counts": owner_counts,
    }
    return summary, derived


def derive_object_voxel_features(
    baseline: dict,
    scale: float,
    derived: dict[str, np.ndarray | list[Counter[int]]],
) -> dict[int, dict]:
    global_keys = np.asarray(derived["keys"], dtype=np.int64)
    frame_counts = np.asarray(derived["frame_counts"], dtype=np.int32)
    fb_dominant = np.asarray(derived["fb_dominant"], dtype=np.int32)
    fb_disagreement = np.asarray(derived["fb_disagreement"], dtype=np.float32)
    owner_entropy = np.asarray(derived["owner_entropy"], dtype=np.float32)
    owner_counts = derived["owner_counts"]
    output: dict[int, dict] = {}
    for index, obj in enumerate(baseline["objects"]):
        keys = voxel_keys(np.asarray(obj["pcd_np"], dtype=np.float64), scale)
        positions = np.searchsorted(global_keys, keys)
        valid = positions < len(global_keys)
        matched = positions[valid]
        matched = matched[global_keys[matched] == keys[valid]]
        labels = fb_dominant[matched]
        labels = labels[labels >= 0]
        majority_hist = Counter(int(value) for value in labels.tolist())
        stable = matched[frame_counts[matched] >= 3]
        persistent = (
            float(np.mean(fb_disagreement[stable] >= 0.20)) if len(stable) else 0.0
        )
        owner_total = Counter()
        for position in matched.tolist():
            owner_total.update(owner_counts[position])
        total_votes = sum(owner_total.values())
        foreign = total_votes - owner_total.get(index, 0)
        output[index] = {
            "framebalanced_majority_entropy": entropy(majority_hist.values()),
            "framebalanced_mean_disagreement": float(np.mean(fb_disagreement[matched])) if len(matched) else 0.0,
            "persistent_conflict_fraction": persistent,
            "multi_owner_voxel_fraction": float(
                np.mean(owner_entropy[matched] > 0) if len(matched) else 0.0
            ),
            "mean_owner_entropy": float(np.mean(owner_entropy[matched])) if len(matched) else 0.0,
            "foreign_owner_link_fraction": float(foreign / max(total_votes, 1)),
            "matched_evidence_voxels": int(len(matched)),
        }
    return output


def target_score_rows(
    scene: str,
    scale: float,
    rows: list[dict],
    targets: dict[str, dict[int, bool]],
    extra: dict[int, dict] | None,
    object_output: list[dict] | None = None,
) -> list[dict]:
    foreground = [row for row in rows if not bool(row.get("is_background"))]
    indices = [int(row["object_index"]) for row in foreground]
    feature_values: dict[str, np.ndarray] = {}
    raw_fields = {
        "voxel_majority_entropy": "voxel_majority_entropy",
        "voxel_vote_entropy": "label_vote_entropy",
        "voxel_mean_disagreement": "mean_voxel_disagreement",
        "voxel_second_region": "second_label_largest_region_fraction",
        "voxel_second_fraction": "voxel_majority_second_fraction",
        "fragmentation": "fragmentation_score",
        "pair_conflict": "best_duplicate_pair_score",
        "nonspatial_owner_label_entropy": "owner_label_entropy",
        "frozen_semantic_max": "semantic_conflict_score",
        "frozen_combined_max": "combined_anomaly_score",
    }
    for name, field in raw_fields.items():
        values = np.asarray([float(row.get(field, 0.0)) for row in foreground], dtype=float)
        if name.startswith("frozen_"):
            feature_values[name] = values
        else:
            feature_values[name] = percentile(values)
    if extra is not None:
        for field in (
            "framebalanced_majority_entropy",
            "framebalanced_mean_disagreement",
            "persistent_conflict_fraction",
            "multi_owner_voxel_fraction",
            "mean_owner_entropy",
            "foreign_owner_link_fraction",
        ):
            feature_values[field] = percentile([extra[index][field] for index in indices])
    semantic_parts = [
        feature_values["voxel_majority_entropy"],
        feature_values["voxel_mean_disagreement"],
        feature_values["voxel_second_region"],
    ]
    feature_values["semantic_mean"] = np.mean(semantic_parts, axis=0)
    core_parts = [
        feature_values["semantic_mean"],
        feature_values["fragmentation"],
        feature_values["pair_conflict"],
    ]
    feature_values["combined_mean"] = np.mean(core_parts, axis=0)
    feature_values["combined_top2_mean"] = np.mean(
        np.sort(np.stack(core_parts, axis=1), axis=1)[:, -2:], axis=1
    )
    if extra is not None:
        feature_values["framebalanced_semantic_mean"] = np.mean(
            [
                feature_values["framebalanced_majority_entropy"],
                feature_values["framebalanced_mean_disagreement"],
                feature_values["persistent_conflict_fraction"],
                feature_values["voxel_second_region"],
            ],
            axis=0,
        )
        feature_values["owner_overlap_mean"] = np.mean(
            [
                feature_values["multi_owner_voxel_fraction"],
                feature_values["mean_owner_entropy"],
                feature_values["foreign_owner_link_fraction"],
            ],
            axis=0,
        )
    feature_values["size_num_detections"] = percentile(
        [math.log1p(float(row.get("num_detections", 0))) for row in foreground]
    )
    feature_values["size_object_voxels"] = percentile(
        [math.log1p(float(row.get("object_voxels", 0))) for row in foreground]
    )
    design = np.column_stack(
        (
            np.ones(len(foreground)),
            feature_values["size_num_detections"],
            feature_values["size_object_voxels"],
        )
    )
    for source in (
        "framebalanced_mean_disagreement",
        "framebalanced_majority_entropy",
        "mean_owner_entropy",
        "voxel_second_region",
    ):
        if source not in feature_values:
            continue
        beta = np.linalg.lstsq(design, feature_values[source], rcond=None)[0]
        feature_values[source + "_size_residual"] = percentile(
            feature_values[source] - design @ beta
        )

    if object_output is not None:
        for position, index in enumerate(indices):
            record = {
                "scene": scene,
                "scale": scale,
                "object_index": index,
                "map_label": foreground[position].get("predicted_label"),
                "num_detections": foreground[position].get("num_detections"),
                "object_voxels": foreground[position].get("object_voxels"),
            }
            record.update(
                {f"score__{name}": float(values[position]) for name, values in feature_values.items()}
            )
            record.update(
                {
                    f"target__{name}": (
                        int(bool(values[index])) if index in values else ""
                    )
                    for name, values in targets.items()
                }
            )
            object_output.append(record)

    output = []
    for target_name, target_by_index in targets.items():
        selected_positions = [
            position for position, index in enumerate(indices) if index in target_by_index
        ]
        labels = [bool(target_by_index[indices[position]]) for position in selected_positions]
        for score_name, values in feature_values.items():
            scores = [float(values[position]) for position in selected_positions]
            output.append(
                {
                    "scene": scene,
                    "scale": scale,
                    "target": target_name,
                    "score": score_name,
                    **binary_metrics(labels, scores),
                }
            )
    return output


def agreement(left: dict[int, bool], right: dict[int, bool]) -> dict:
    keys = sorted(set(left) & set(right))
    a = np.asarray([bool(left[key]) for key in keys], dtype=bool)
    b = np.asarray([bool(right[key]) for key in keys], dtype=bool)
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return {
        "n": int(len(keys)),
        "left_positive": int(a.sum()),
        "right_positive": int(b.sum()),
        "both_positive": intersection,
        "agreement": float(np.mean(a == b)) if len(a) else None,
        "positive_jaccard": float(intersection / union) if union else None,
    }


def plot_results(metrics: list[dict], construction: list[dict], output: Path) -> None:
    chosen_scores = [
        "frozen_combined_max",
        "combined_mean",
        "framebalanced_semantic_mean",
        "owner_overlap_mean",
        "voxel_vote_entropy",
        "nonspatial_owner_label_entropy",
    ]
    chosen_targets = ["sidecar_original_unified", "sidecar_strict_unified", "geometry_partition_5cm"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = {"room0": "#3973ac", "office0": "#d65f5f"}
    width = 0.34
    for target_index, target in enumerate(chosen_targets[:2]):
        axis = axes[0, target_index]
        x = np.arange(len(chosen_scores))
        for offset, scene in enumerate(SCENES):
            values = []
            for score in chosen_scores:
                matches = [
                    row
                    for row in metrics
                    if row["scene"] == scene
                    and row["scale"] == PRIMARY_SCALE
                    and row["target"] == target
                    and row["score"] == score
                ]
                values.append(matches[0]["auroc"] if matches else np.nan)
            axis.bar(x + (offset - 0.5) * width, values, width, label=scene, color=colors[scene])
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_xticks(x, [name.replace("_", "\n") for name in chosen_scores], fontsize=8)
        axis.set_ylabel("AUROC")
        axis.set_title(target)
        axis.legend()

    axis = axes[1, 0]
    x = np.arange(len(SCENES))
    for offset, target in enumerate(chosen_targets):
        values = []
        for scene in SCENES:
            matches = [
                row
                for row in metrics
                if row["scene"] == scene
                and row["scale"] == PRIMARY_SCALE
                and row["target"] == target
                and row["score"] == "framebalanced_semantic_mean"
            ]
            values.append(matches[0]["auroc"] if matches else np.nan)
        axis.bar(x + (offset - 1) * 0.24, values, 0.24, label=target)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, SCENES)
    axis.set_ylim(0, 1)
    axis.set_ylabel("AUROC")
    axis.set_title("Frame-balanced voxel semantic score across GT definitions")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    labels = []
    same_frame = []
    multi_owner = []
    for row in construction:
        if row["scale"] != PRIMARY_SCALE:
            continue
        labels.append(row["scene"])
        same_frame.append(row["same_frame_extra_link_fraction"])
        multi_owner.append(row["multi_owner_voxel_fraction"])
    x = np.arange(len(labels))
    axis.bar(x - 0.18, same_frame, 0.36, label="same-frame extra links")
    axis.bar(x + 0.18, multi_owner, 0.36, label="multi-owner voxels")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, max(same_frame + multi_owner + [0.1]) * 1.25)
    axis.set_ylabel("fraction")
    axis.set_title("Construction confounds at 5 cm")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "01_reaudit_overview.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "INCOMPLETE").write_text("running\n", encoding="utf-8")
    started = time.perf_counter()
    all_metrics: list[dict] = []
    construction_rows: list[dict] = []
    agreements: list[dict] = []
    target_counts: list[dict] = []
    case_rows: list[dict] = []
    object_score_rows: list[dict] = []

    for scene in SCENES:
        print(f"[{scene}] loading frozen artifacts", flush=True)
        scene_root = root / "full" / scene
        manifest = json.loads((scene_root / "manifest.json").read_text(encoding="utf-8"))
        observations = read_jsonl(scene_root / "observations.jsonl")
        baseline = load_pickle_gz(Path(manifest["baseline_map"]))
        gt_payload = load_pickle_gz(Path(manifest["gt_map"]))
        primary_rows = read_jsonl(
            scene_root / scale_tag(PRIMARY_SCALE) / "all_history" / "objects.jsonl"
        )
        original_audit, _ = build_sidecar_targets(observations, primary_rows, strict=False)
        strict_audit, _ = build_sidecar_targets(observations, primary_rows, strict=True)
        conservative_audit, _ = build_sidecar_targets(
            observations, primary_rows, strict=True, conservative=True
        )
        print(f"[{scene}] building one exclusive geometry query", flush=True)
        geometry_queries, geometry_labels = exclusive_geometry_queries(baseline, gt_payload)
        geometry_by_distance = {}
        for distance in (0.025, 0.05, 0.075):
            print(f"[{scene}] exclusive geometry GT at {distance:.3f} m", flush=True)
            geometry_by_distance[distance], _ = exclusive_geometry_targets(
                geometry_queries, geometry_labels, max_distance=distance
            )

        for scale in SCALES:
            print(f"[{scene}] auditing voxel scale {scale:.3f}", flush=True)
            scale_root = scene_root / scale_tag(scale) / "all_history"
            rows = read_jsonl(scale_root / "objects.jsonl")
            construction, derived = audit_voxel_map(scale_root / "voxel_map.npz", observations)
            construction_rows.append({"scene": scene, "scale": scale, **construction})
            extra = derive_object_voxel_features(baseline, scale, derived)
            targets: dict[str, dict[int, bool]] = {
                "sidecar_original_unified": {
                    index: bool(audit["unified"]) for index, audit in original_audit.items()
                },
                "sidecar_strict_unified": {
                    index: bool(audit["unified"]) for index, audit in strict_audit.items()
                },
                "sidecar_original_mask": {
                    index: bool(audit["mask_repeated"]) for index, audit in original_audit.items()
                },
                "sidecar_strict_mask": {
                    index: bool(audit["mask_repeated"]) for index, audit in strict_audit.items()
                },
                "sidecar_original_association": {
                    index: bool(audit["association_repeated"])
                    for index, audit in original_audit.items()
                    if audit["pure"] >= 2
                },
                "sidecar_strict_association": {
                    index: bool(audit["association_repeated"])
                    for index, audit in strict_audit.items()
                    if audit["pure"] >= 2
                },
                "sidecar_conservative_unified": {
                    index: bool(audit["unified"])
                    for index, audit in conservative_audit.items()
                },
                "sidecar_conservative_mask": {
                    index: bool(audit["mask_repeated"])
                    for index, audit in conservative_audit.items()
                },
                "sidecar_conservative_association": {
                    index: bool(audit["association_repeated"])
                    for index, audit in conservative_audit.items()
                    if audit["pure"] >= 3
                },
            }
            for distance, audits in geometry_by_distance.items():
                targets[f"geometry_partition_{int(distance * 100):d}cm"] = {
                    index: bool(audit["partition_error"])
                    for index, audit in audits.items()
                    if audit["reliable_foreground"]
                }
                targets[f"geometry_overmerge_{int(distance * 100):d}cm"] = {
                    index: bool(audit["overmerge"])
                    for index, audit in audits.items()
                    if audit["reliable_foreground"]
                }
            all_metrics.extend(
                target_score_rows(
                    scene,
                    scale,
                    rows,
                    targets,
                    extra,
                    object_score_rows if scale == PRIMARY_SCALE else None,
                )
            )
            if scale == PRIMARY_SCALE:
                for name, values in targets.items():
                    target_counts.append(
                        {
                            "scene": scene,
                            "target": name,
                            "n": len(values),
                            "positives": int(sum(values.values())),
                            "prevalence": float(np.mean(list(values.values()))) if values else None,
                        }
                    )

        original_unified = {index: bool(audit["unified"]) for index, audit in original_audit.items()}
        strict_unified = {index: bool(audit["unified"]) for index, audit in strict_audit.items()}
        geom_5 = {
            index: bool(audit["partition_error"])
            for index, audit in geometry_by_distance[0.05].items()
            if audit["reliable_foreground"]
        }
        agreements.extend(
            [
                {"scene": scene, "left": "sidecar_original", "right": "sidecar_strict", **agreement(original_unified, strict_unified)},
                {"scene": scene, "left": "sidecar_original", "right": "geometry_5cm", **agreement(original_unified, geom_5)},
                {"scene": scene, "left": "sidecar_strict", "right": "geometry_5cm", **agreement(strict_unified, geom_5)},
            ]
        )
        rows_by_index = {int(row["object_index"]): row for row in primary_rows}
        for index in sorted(set(original_unified) | set(strict_unified) | set(geom_5)):
            values = (
                original_unified.get(index),
                strict_unified.get(index),
                geom_5.get(index),
            )
            if len(set(values)) <= 1:
                continue
            row = rows_by_index[index]
            case_rows.append(
                {
                    "scene": scene,
                    "object_index": index,
                    "map_label": row.get("predicted_label"),
                    "combined_score": row.get("combined_anomaly_score"),
                    "original_unified": values[0],
                    "strict_unified": values[1],
                    "geometry_partition_5cm": values[2],
                    "sidecar_gt": original_audit[index].get("dominant_gt_label"),
                    "geometry_gt": geometry_by_distance[0.05][index].get("dominant_gt_label"),
                }
            )

    write_csv(output / "construction_audit.csv", construction_rows)
    write_csv(output / "metrics.csv", all_metrics)
    write_csv(output / "target_counts.csv", target_counts)
    write_csv(output / "gt_agreement.csv", agreements)
    write_csv(output / "gt_disagreement_cases.csv", case_rows)
    write_csv(output / "object_scores_5cm.csv", object_score_rows)
    plot_results(all_metrics, construction_rows, output)

    stable_candidates = []
    score_names = sorted({row["score"] for row in all_metrics})
    target_names = sorted({row["target"] for row in all_metrics})
    for target in target_names:
        for score in score_names:
            rows = [
                row
                for row in all_metrics
                if row["scale"] == PRIMARY_SCALE
                and row["target"] == target
                and row["score"] == score
                and row["auroc"] is not None
            ]
            if len(rows) != 2:
                continue
            stable_candidates.append(
                {
                    "target": target,
                    "score": score,
                    "room0_auroc": next(row["auroc"] for row in rows if row["scene"] == "room0"),
                    "office0_auroc": next(row["auroc"] for row in rows if row["scene"] == "office0"),
                    "min_auroc": min(row["auroc"] for row in rows),
                    "mean_auroc": float(np.mean([row["auroc"] for row in rows])),
                    "both_positive_ap_lift": bool(all(row["ap_minus_prevalence"] > 0 for row in rows)),
                }
            )
    stable_candidates.sort(key=lambda row: (-row["min_auroc"], -row["mean_auroc"], row["target"], row["score"]))
    write_csv(output / "cross_scene_candidates_5cm.csv", stable_candidates)
    summary = {
        "elapsed_seconds": float(time.perf_counter() - started),
        "construction": construction_rows,
        "target_counts": target_counts,
        "gt_agreement": agreements,
        "best_cross_scene_candidates_5cm": stable_candidates[:30],
        "interpretation_rule": (
            "Only a pre-defined score with AUROC>0.60 and positive AP lift in both scenes, "
            "stable across reasonable GT definitions, is evidence for a usable trigger. "
            "Single-scene or post-hoc best features are exploratory only."
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "INCOMPLETE").unlink(missing_ok=True)
    (output / "READY").write_text("ready\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
