#!/usr/bin/env python3
"""Post-analysis and visual diagnostics for the simple voxel-trigger experiment.

The frozen run is never modified.  This script:

1. audits the pre-registered gate exactly as written by the frozen runner;
2. replaces the invalid many-to-many coarse-GT-voxel overlap diagnostic with
   observation-sidecar attribution for mask/association/split diagnostics;
3. reports scale, non-spatial baseline, negative, storage, and candidate-recall
   analyses; and
4. creates static scientific figures and small case tables.

GT sidecars are used only for post-hoc evaluation.  The voxel payload and every
anomaly score remain unchanged.
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
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


BASE = Path("/home/chenkejun/beauty/conceptgraphs")
DEFAULT_ROOT = BASE / "results/experiments/voxel_trigger_v0_20260831/full"
DEFAULT_OUTPUT = BASE / "results/experiments/voxel_trigger_v0_20260831/analysis"
SCENES = ("room0", "office0")
SCALES = (0.025, 0.05, 0.10)
PRIMARY_SCALE = 0.05
BG_LABELS = {"wall", "floor", "ceiling"}

BITS = 21
OFFSET = 1 << (BITS - 1)
FIELD_MASK = (1 << BITS) - 1
SHIFT_X = BITS * 2
SHIFT_Y = BITS

SCORE_FIELDS = {
    "nonspatial": "nonspatial_label_score",
    "voxel_semantic": "semantic_conflict_score",
    "fragmentation": "fragmentation_rank",
    "duplicate_incident": "duplicate_conflict_score",
    "combined": "combined_anomaly_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def scale_dir(scale: float) -> str:
    return f"voxel_{scale:.3f}".replace(".", "p")


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


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(labels: Iterable[bool], scores: Iterable[float]) -> dict:
    y = np.asarray(list(labels), dtype=np.uint8)
    score = np.asarray(list(scores), dtype=float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if not len(y):
        return {"n": 0}
    order = np.argsort(-score, kind="stable")
    prevalence = float(y.mean())
    output = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "average_precision": None,
        "auroc": None,
    }
    if len(np.unique(y)) == 2:
        output["average_precision"] = float(average_precision_score(y, score))
        output["auroc"] = float(roc_auc_score(y, score))
    for name, count in (
        ("top5", min(5, len(y))),
        ("top10", min(10, len(y))),
        ("top20pct", max(1, int(math.ceil(0.2 * len(y))))),
    ):
        subset = y[order[:count]]
        precision = float(subset.mean())
        output[name] = {
            "k": int(count),
            "true_errors": int(subset.sum()),
            "precision": precision,
            "lift_vs_prevalence": float(precision / prevalence) if prevalence else None,
        }
    bottom_count = max(1, int(math.ceil(0.2 * len(y))))
    bottom_rate = float(y[order[-bottom_count:]].mean())
    output["bottom20pct"] = {
        "k": int(bottom_count),
        "true_errors": int(y[order[-bottom_count:]].sum()),
        "error_rate": bottom_rate,
    }
    top_rate = output["top20pct"]["precision"]
    output["top_to_bottom_ratio"] = (
        float(top_rate / bottom_rate) if bottom_rate else None
    )
    output["ap_minus_prevalence"] = (
        float(output["average_precision"] - prevalence)
        if output["average_precision"] is not None
        else None
    )
    return output


def bootstrap_interval(
    labels: Iterable[bool],
    scores: Iterable[float],
    *,
    seed: int,
    repetitions: int = 1000,
) -> dict:
    y = np.asarray(list(labels), dtype=np.uint8)
    score = np.asarray(list(scores), dtype=float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return {"repetitions": 0, "ap_95ci": None, "auroc_95ci": None}
    random = np.random.default_rng(seed)
    ap_values = []
    auc_values = []
    for _ in range(repetitions):
        indices = random.integers(0, len(y), len(y))
        sample_y = y[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_score = score[indices]
        ap_values.append(average_precision_score(sample_y, sample_score))
        auc_values.append(roc_auc_score(sample_y, sample_score))
    return {
        "repetitions": int(len(ap_values)),
        "ap_95ci": [float(x) for x in np.quantile(ap_values, [0.025, 0.975])],
        "auroc_95ci": [float(x) for x in np.quantile(auc_values, [0.025, 0.975])],
    }


def observation_audit(
    object_rows: list[dict], observations: list[dict]
) -> dict[int, dict]:
    """Build collision-free GT diagnostics from per-frame instance sidecars."""
    by_owner: dict[int, list[dict]] = defaultdict(list)
    for observation in observations:
        owner = observation.get("owner_index")
        if owner is not None:
            by_owner[int(owner)].append(observation)

    output: dict[int, dict] = {}
    for row in object_rows:
        index = int(row["object_index"])
        members = by_owner.get(index, [])
        eligible = [
            item
            for item in members
            if item.get("gt_assignment_eligible") and item.get("gt_top_id") is not None
        ]
        mixed = [item for item in eligible if bool(item.get("mask_mixed"))]
        pure = [
            item
            for item in eligible
            if not bool(item.get("mask_mixed"))
            and float(item.get("gt_purity", 0.0)) >= 0.8
        ]
        counts = Counter(int(item["gt_top_id"]) for item in pure)
        label_by_id = {
            int(item["gt_top_id"]): str(item.get("gt_top_label") or "unknown")
            for item in pure
        }
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        dominant_id = ordered[0][0] if ordered else None
        dominant_count = ordered[0][1] if ordered else 0
        second_count = ordered[1][1] if len(ordered) > 1 else 0
        wrong_count = int(len(pure) - dominant_count)
        wrong_fraction = float(wrong_count / max(len(pure), 1))
        mixed_fraction = float(len(mixed) / max(len(eligible), 1))
        substantial = [
            (instance_id, count)
            for instance_id, count in ordered
            if count >= 2 and count / max(len(pure), 1) >= 0.05
        ]
        substantial_labels = {label_by_id.get(instance_id) for instance_id, _ in substantial}
        audit = {
            "audit_eligible_observations": int(len(eligible)),
            "audit_mixed_mask_observations": int(len(mixed)),
            "audit_mixed_mask_fraction": mixed_fraction,
            "audit_pure_observations": int(len(pure)),
            "audit_gt_hist": {str(key): int(value) for key, value in ordered},
            "audit_gt_label_hist": {
                label_by_id.get(key, "unknown"): int(value) for key, value in ordered
            },
            "audit_dominant_gt_id": int(dominant_id) if dominant_id is not None else None,
            "audit_dominant_gt_label": (
                label_by_id.get(dominant_id) if dominant_id is not None else None
            ),
            "audit_dominant_count": int(dominant_count),
            "audit_dominant_fraction": float(dominant_count / max(len(pure), 1)),
            "audit_second_count": int(second_count),
            "audit_wrong_association_observations": wrong_count,
            "audit_wrong_association_fraction": wrong_fraction,
            "audit_mask_any": bool(len(mixed) >= 1),
            "audit_mask_repeated": bool(len(mixed) >= 2 and mixed_fraction >= 0.05),
            "audit_mask_severe": bool(len(mixed) >= 2 and mixed_fraction >= 0.10),
            "audit_association_loose": bool(
                wrong_count >= 2 or (wrong_count >= 1 and wrong_fraction >= 0.05)
            ),
            "audit_association_repeated": bool(
                wrong_count >= 2 and wrong_fraction >= 0.05
            ),
            "audit_overmerge_repeated": bool(len(substantial) >= 2),
            "audit_overmerge_cross_class": bool(
                len(substantial) >= 2 and len(substantial_labels) >= 2
            ),
            "audit_overmerge_same_class": bool(
                len(substantial) >= 2 and len(substantial_labels) == 1
            ),
            "audit_spurious_background": bool(
                dominant_id is not None
                and label_by_id.get(dominant_id) in BG_LABELS
                and dominant_count / max(len(pure), 1) >= 0.5
            ),
            "audit_observation_coverage": bool(len(eligible) >= 1),
            "audit_association_coverage": bool(len(pure) >= 2),
        }
        output[index] = audit
    return output


def split_audit(
    object_rows: list[dict],
    object_audits: dict[int, dict],
    pair_rows: list[dict],
) -> dict:
    foreground = [row for row in object_rows if not row["is_background"]]
    eligible_groups: dict[int, list[int]] = defaultdict(list)
    reliable_indices = set()
    for row in foreground:
        index = int(row["object_index"])
        audit = object_audits[index]
        gt_id = audit["audit_dominant_gt_id"]
        gt_label = audit["audit_dominant_gt_label"]
        if (
            gt_id is not None
            and gt_label not in BG_LABELS
            and audit["audit_dominant_count"] >= 2
            and audit["audit_dominant_fraction"] >= 0.5
        ):
            reliable_indices.add(index)
            eligible_groups[int(gt_id)].append(index)

    all_true_pairs = set()
    split_incidents = set()
    for indices in eligible_groups.values():
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                key = (min(left, right), max(left, right))
                all_true_pairs.add(key)
                split_incidents.update(key)

    corrected_pairs = []
    candidate_true_pairs = set()
    for row in pair_rows:
        copied = dict(row)
        key = (
            min(int(row["left_index"]), int(row["right_index"])),
            max(int(row["left_index"]), int(row["right_index"])),
        )
        copied["audit_pair_evaluable"] = bool(
            key[0] in reliable_indices and key[1] in reliable_indices
        )
        copied["audit_false_split_pair"] = bool(
            copied["audit_pair_evaluable"] and key in all_true_pairs
        )
        if copied["audit_false_split_pair"]:
            candidate_true_pairs.add(key)
            copied["audit_gt_instance_id"] = object_audits[key[0]][
                "audit_dominant_gt_id"
            ]
            copied["audit_gt_instance_label"] = object_audits[key[0]][
                "audit_dominant_gt_label"
            ]
        else:
            copied["audit_gt_instance_id"] = None
            copied["audit_gt_instance_label"] = None
        corrected_pairs.append(copied)

    foreground_count = len(foreground)
    all_pair_count = foreground_count * (foreground_count - 1) // 2
    return {
        "pairs": corrected_pairs,
        "split_incidents": split_incidents,
        "all_true_pair_count": int(len(all_true_pairs)),
        "candidate_true_pair_count": int(len(candidate_true_pairs)),
        "candidate_recall": float(len(candidate_true_pairs) / max(len(all_true_pairs), 1)),
        "candidate_pair_count": int(len(corrected_pairs)),
        "candidate_evaluable_pair_count": int(
            sum(row["audit_pair_evaluable"] for row in corrected_pairs)
        ),
        "candidate_unresolved_pair_count": int(
            sum(not row["audit_pair_evaluable"] for row in corrected_pairs)
        ),
        "all_foreground_pair_count": int(all_pair_count),
        "candidate_reduction": float(1.0 - len(corrected_pairs) / max(all_pair_count, 1)),
    }


def add_object_targets(
    object_rows: list[dict],
    object_audits: dict[int, dict],
    split_incidents: set[int],
) -> list[dict]:
    output = []
    for row in object_rows:
        copied = dict(row)
        index = int(row["object_index"])
        copied.update(object_audits[index])
        copied["audit_split_incident"] = bool(index in split_incidents)
        copied["audit_unified_repeated"] = bool(
            copied["audit_mask_repeated"]
            or copied["audit_association_repeated"]
            or copied["audit_split_incident"]
        )
        copied["audit_unified_plus_spurious"] = bool(
            copied["audit_unified_repeated"]
            or copied["audit_spurious_background"]
        )
        copied["audit_merge_membership_repeated"] = bool(
            copied["audit_mask_repeated"] or copied["audit_association_repeated"]
        )
        output.append(copied)
    return output


def metrics_row(
    *,
    scene: str,
    scale: float,
    task: str,
    score_name: str,
    labels: list[bool],
    scores: list[float],
    target_level: str,
) -> dict:
    result = binary_metrics(labels, scores)
    return {
        "scene": scene,
        "voxel_size_m": scale,
        "target_level": target_level,
        "task": task,
        "score": score_name,
        **result,
    }


def evaluate_scale(
    *,
    scene: str,
    scale: float,
    object_rows: list[dict],
    corrected_pairs: list[dict],
) -> list[dict]:
    output = []
    foreground = [row for row in object_rows if not row["is_background"]]
    object_tasks = {
        "mask_any": ("audit_observation_coverage", "audit_mask_any"),
        "mask_repeated": ("audit_observation_coverage", "audit_mask_repeated"),
        "mask_severe": ("audit_observation_coverage", "audit_mask_severe"),
        "association_loose": (
            "audit_association_coverage",
            "audit_association_loose",
        ),
        "association_repeated": (
            "audit_association_coverage",
            "audit_association_repeated",
        ),
        "overmerge_repeated": (
            "audit_association_coverage",
            "audit_overmerge_repeated",
        ),
        "split_incident": ("audit_observation_coverage", "audit_split_incident"),
        "merge_membership_repeated": (
            "audit_observation_coverage",
            "audit_merge_membership_repeated",
        ),
        "unified_repeated": (
            "audit_observation_coverage",
            "audit_unified_repeated",
        ),
        "unified_plus_spurious": (
            "audit_observation_coverage",
            "audit_unified_plus_spurious",
        ),
        "spurious_background": (
            "audit_association_coverage",
            "audit_spurious_background",
        ),
    }
    for task, (coverage_field, target_field) in object_tasks.items():
        selected = [row for row in foreground if bool(row[coverage_field])]
        labels = [bool(row[target_field]) for row in selected]
        for score_name, score_field in SCORE_FIELDS.items():
            output.append(
                metrics_row(
                    scene=scene,
                    scale=scale,
                    task=task,
                    score_name=score_name,
                    labels=labels,
                    scores=[float(row[score_field]) for row in selected],
                    target_level="object",
                )
            )

    evaluable_pairs = [row for row in corrected_pairs if row["audit_pair_evaluable"]]
    pair_labels = [bool(row["audit_false_split_pair"]) for row in evaluable_pairs]
    for score_name, score_field in (
        ("pair_r1", "duplicate_score_r1"),
        ("pair_r2", "duplicate_score_r2"),
    ):
        output.append(
            metrics_row(
                scene=scene,
                scale=scale,
                task="false_split_pair",
                score_name=score_name,
                labels=pair_labels,
                scores=[float(row[score_field]) for row in evaluable_pairs],
                target_level="pair_candidate",
            )
        )
    return output


def voxel_map_stats(path: Path, scene: str, scale: float) -> tuple[dict, list[dict]]:
    payload = np.load(path)
    seen = np.asarray(payload["seen_count"], dtype=np.int64)
    label_offsets = np.asarray(payload["label_offsets"], dtype=np.int64)
    label_counts = np.asarray(payload["label_counts"], dtype=np.int64)
    labels_per_voxel = np.diff(label_offsets)
    disagreement = np.zeros(len(seen), dtype=float)
    for index in range(len(seen)):
        counts = label_counts[label_offsets[index] : label_offsets[index + 1]]
        disagreement[index] = 1.0 - float(counts.max() / max(counts.sum(), 1))
    array_bytes = int(sum(np.asarray(payload[key]).nbytes for key in payload.files))
    result = {
        "scene": scene,
        "voxel_size_m": scale,
        "voxel_count": int(len(seen)),
        "observation_links": int(len(payload["obs_ids"])),
        "label_entries": int(len(payload["label_ids"])),
        "seen_mean": float(seen.mean()),
        "seen_median": float(np.median(seen)),
        "seen_p90": float(np.quantile(seen, 0.9)),
        "seen_ge2_fraction": float(np.mean(seen >= 2)),
        "seen_ge5_fraction": float(np.mean(seen >= 5)),
        "multi_label_voxel_fraction": float(np.mean(labels_per_voxel >= 2)),
        "mean_voxel_disagreement": float(disagreement.mean()),
        "conflict_voxel_fraction_ge_0p25": float(np.mean(disagreement >= 0.25)),
        "npz_bytes": int(path.stat().st_size),
        "array_bytes": array_bytes,
        "array_bytes_per_voxel": float(array_bytes / max(len(seen), 1)),
        "obs_id_bytes_fraction": float(
            np.asarray(payload["obs_ids"]).nbytes / max(array_bytes, 1)
        ),
    }

    labels = json.loads((path.parents[2] / "observation_index.json").read_text())["labels"]
    ranking = np.lexsort((-disagreement, -seen))
    examples = []
    chosen = list(ranking[:3]) + list(np.argsort(-disagreement, kind="stable")[:3])
    for index in dict.fromkeys(int(value) for value in chosen):
        start, end = label_offsets[index], label_offsets[index + 1]
        hist = {
            labels.get(str(int(label_id)), str(int(label_id))): int(count)
            for label_id, count in zip(
                payload["label_ids"][start:end], payload["label_counts"][start:end]
            )
        }
        obs_start, obs_end = payload["obs_offsets"][index : index + 2]
        examples.append(
            {
                "scene": scene,
                "voxel_size_m": scale,
                "voxel_coord": payload["voxel_coords"][index].astype(int).tolist(),
                "seen_count": int(seen[index]),
                "label_hist": hist,
                "obs_ids_preview": payload["obs_ids"][obs_start : min(obs_end, obs_start + 12)]
                .astype(int)
                .tolist(),
                "obs_ids_total": int(obs_end - obs_start),
                "disagreement": float(disagreement[index]),
            }
        )
    return result, examples


def frozen_gate(root: Path) -> dict:
    scene_rows = []
    scale_directions = {}
    for scene in SCENES:
        manifest = json.loads((root / scene / "manifest.json").read_text())
        for scale in SCALES:
            summary = manifest["analyses"][str(scale)]["all_history"]
            metric = summary["evaluation"]["object_evaluation"]["combined"][
                "gt_identity_error"
            ]
            row = {
                "scene": scene,
                "voxel_size_m": scale,
                "prevalence": metric["prevalence"],
                "average_precision": metric["average_precision"],
                "ap_minus_prevalence": metric["average_precision"]
                - metric["prevalence"],
                "top20_error_rate": metric["top20pct"]["precision"],
                "bottom20_error_rate": metric["bottom20pct"]["error_rate"],
                "top_to_bottom_ratio": metric["top_to_bottom_ratio"],
            }
            scene_rows.append(row)
            scale_directions[(scene, scale)] = bool(
                row["ap_minus_prevalence"] > 0
                and (row["top_to_bottom_ratio"] or 0) > 1
            )
    primary = [row for row in scene_rows if row["voxel_size_m"] == PRIMARY_SCALE]
    go_primary = all(
        row["ap_minus_prevalence"] >= 0.10
        and (row["top_to_bottom_ratio"] or 0) >= 2.0
        for row in primary
    )
    direction_count = {
        scene: sum(scale_directions[(scene, scale)] for scale in SCALES)
        for scene in SCENES
    }
    return {
        "rows": scene_rows,
        "primary_gate_pass": bool(go_primary),
        "direction_count_by_scene": direction_count,
        "go_pass": bool(go_primary and all(value >= 2 for value in direction_count.values())),
        "diagnostic_validity": "invalid_for_final_claim_due_to_coarse_gt_voxel_collision",
    }


def scope_equivalence(root: Path) -> list[dict]:
    output = []
    for scene in SCENES:
        for scale in SCALES:
            directory = root / scene / scale_dir(scale)
            left = directory / "all_history" / "voxel_map.npz"
            right = directory / "final_members_only" / "voxel_map.npz"
            left_summary = json.loads((left.parent / "summary.json").read_text())
            right_summary = json.loads((right.parent / "summary.json").read_text())
            output.append(
                {
                    "scene": scene,
                    "voxel_size_m": scale,
                    "all_history_observations": left_summary["selected_observations"],
                    "final_members_observations": right_summary["selected_observations"],
                    "voxel_map_sha256_equal": sha256(left) == sha256(right),
                    "voxel_map_sha256": sha256(left),
                }
            )
    return output


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    x = ((keys >> SHIFT_X) & FIELD_MASK) - OFFSET
    y = ((keys >> SHIFT_Y) & FIELD_MASK) - OFFSET
    z = (keys & FIELD_MASK) - OFFSET
    return np.column_stack((x, y, z)).astype(np.int32, copy=False)


def pack_coords(coords: np.ndarray) -> np.ndarray:
    shifted = np.asarray(coords, dtype=np.int64) + OFFSET
    return (
        (shifted[:, 0] << SHIFT_X)
        | (shifted[:, 1] << SHIFT_Y)
        | shifted[:, 2]
    ).astype(np.int64, copy=False)


def voxel_keys(points: np.ndarray, scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    return np.unique(pack_coords(np.floor(points[finite] / scale).astype(np.int64)))


def load_pickle(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def object_case_payload(
    root: Path,
    scene: str,
    object_index: int,
    scale: float,
    cache: dict,
) -> dict:
    key = (scene, scale)
    if key not in cache:
        manifest = json.loads((root / scene / "manifest.json").read_text())
        baseline = load_pickle(Path(manifest["baseline_map"]))
        voxel_path = root / scene / scale_dir(scale) / "all_history" / "voxel_map.npz"
        voxel_map = np.load(voxel_path)
        labels = json.loads((root / scene / "observation_index.json").read_text())["labels"]
        cache[key] = (baseline, voxel_map, labels)
    baseline, voxel_map, labels = cache[key]
    points = np.asarray(baseline["objects"][object_index]["pcd_np"], dtype=np.float64)
    keys = voxel_keys(points, scale)
    positions = np.searchsorted(voxel_map["voxel_keys"], keys)
    valid = positions < len(voxel_map["voxel_keys"])
    valid[valid] &= voxel_map["voxel_keys"][positions[valid]] == keys[valid]
    keys = keys[valid]
    positions = positions[valid]
    coords = unpack_keys(keys).astype(float) * scale
    majority_ids = np.full(len(positions), -1, dtype=int)
    confidence = np.zeros(len(positions), dtype=float)
    for output_index, map_index in enumerate(positions):
        start, end = voxel_map["label_offsets"][map_index : map_index + 2]
        counts = voxel_map["label_counts"][start:end]
        ids = voxel_map["label_ids"][start:end]
        if len(counts):
            winner = int(np.argmax(counts))
            majority_ids[output_index] = int(ids[winner])
            confidence[output_index] = float(counts[winner] / counts.sum())
    label_names = np.asarray(
        [labels.get(str(int(value)), "no evidence") if value >= 0 else "no evidence" for value in majority_ids],
        dtype=object,
    )
    return {
        "coords": coords,
        "majority_ids": majority_ids,
        "majority_labels": label_names,
        "confidence": confidence,
        "baseline_object": baseline["objects"][object_index],
    }


def palette_for_labels(labels: np.ndarray) -> tuple[dict[str, object], list[str]]:
    counts = Counter(str(value) for value in labels)
    ordered = [key for key, _ in counts.most_common()]
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(index % 10) for index, label in enumerate(ordered)}
    colors["no evidence"] = (0.65, 0.65, 0.65, 0.75)
    return colors, ordered


def plot_object_cases(
    root: Path,
    output: Path,
    scene: str,
    rows: list[dict],
    case_rows: list[dict],
) -> None:
    cache = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for axis, case in zip(axes.flat, case_rows):
        row = rows[int(case["row_position"])]
        payload = object_case_payload(
            root, scene, int(row["object_index"]), PRIMARY_SCALE, cache
        )
        coords = payload["coords"]
        colors, ordered = palette_for_labels(payload["majority_labels"])
        for label in ordered[:8]:
            selected = payload["majority_labels"] == label
            axis.scatter(
                coords[selected, 0],
                coords[selected, 2],
                s=18,
                alpha=0.85,
                color=colors[label],
                label=f"{label} ({int(selected.sum())})",
                linewidths=0,
            )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m), top view")
        axis.grid(alpha=0.15)
        axis.set_title(
            f"{case['case']} | object {row['object_index']} | map label: {row['predicted_label']}\n"
            f"score={row['combined_anomaly_score']:.3f}, semantic={row['semantic_conflict_score']:.3f}, "
            f"fragment={row['fragmentation_rank']:.3f}, pair={row['duplicate_conflict_score']:.3f}\n"
            f"audit: mask={int(row['audit_mask_repeated'])}, assoc={int(row['audit_association_repeated'])}, "
            f"split={int(row['audit_split_incident'])}, GT={row['audit_dominant_gt_label']}"
        )
        axis.legend(loc="best", fontsize=7, frameon=True)
    for axis in axes.flat[len(case_rows) :]:
        axis.axis("off")
    fig.suptitle(
        f"{scene}: representative 5 cm object voxels colored by observation-label majority",
        fontsize=15,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pair_cases(
    root: Path,
    output: Path,
    selected_by_scene: dict[str, list[dict]],
) -> None:
    cache = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for scene_position, scene in enumerate(SCENES):
        for case_position, pair in enumerate(selected_by_scene[scene]):
            axis = axes[scene_position, case_position]
            for index, color, name in (
                (int(pair["left_index"]), "#e24a33", "left"),
                (int(pair["right_index"]), "#348abd", "right"),
            ):
                payload = object_case_payload(root, scene, index, PRIMARY_SCALE, cache)
                coords = payload["coords"]
                axis.scatter(
                    coords[:, 0],
                    coords[:, 2],
                    s=16,
                    alpha=0.72,
                    color=color,
                    label=f"{name}: object {index}",
                    linewidths=0,
                )
            label = (
                "TP"
                if pair["audit_false_split_pair"]
                else "FP"
                if pair["audit_pair_evaluable"]
                else "UNRESOLVED"
            )
            axis.set_title(
                f"{scene} {label} | pair score r1={pair['duplicate_score_r1']:.3f}\n"
                f"labels: {pair['left_label']} + {pair['right_label']} | "
                f"GT={pair.get('audit_gt_instance_label')}\n"
                f"min distance={pair['min_distance_m']:.3f} m, label cosine={pair['label_hist_cosine']:.3f}"
            )
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xlabel("x (m)")
            axis.set_ylabel("z (m), top view")
            axis.grid(alpha=0.15)
            axis.legend(fontsize=8)
    fig.suptitle("5 cm false-split candidate examples", fontsize=15)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gate(summary: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {"room0": "#e24a33", "office0": "#348abd"}
    rows = summary["frozen_gate"]["rows"]
    for scene in SCENES:
        selected = sorted(
            [row for row in rows if row["scene"] == scene],
            key=lambda row: row["voxel_size_m"],
        )
        x = [row["voxel_size_m"] * 100 for row in selected]
        axes[0, 0].plot(
            x,
            [row["ap_minus_prevalence"] for row in selected],
            "o-",
            color=colors[scene],
            label=scene,
            linewidth=2,
        )
        axes[0, 1].plot(
            x,
            [row["top_to_bottom_ratio"] for row in selected],
            "o-",
            color=colors[scene],
            label=scene,
            linewidth=2,
        )
    axes[0, 0].axhline(0.10, color="black", linestyle="--", label="GO threshold")
    axes[0, 0].axhline(0.0, color="gray", linewidth=0.8)
    axes[0, 0].set_title("Frozen combined object trigger: AP lift")
    axes[0, 0].set_xlabel("voxel size (cm)")
    axes[0, 0].set_ylabel("AP - error prevalence")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)
    axes[0, 1].axhline(2.0, color="black", linestyle="--", label="GO threshold")
    axes[0, 1].axhline(1.0, color="gray", linewidth=0.8)
    axes[0, 1].set_title("Frozen combined object trigger: rank separation")
    axes[0, 1].set_xlabel("voxel size (cm)")
    axes[0, 1].set_ylabel("top-20% / bottom-20% error rate")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    primary_rows = summary["primary_corrected_metrics"]
    natural = [
        ("mask_repeated", "voxel_semantic"),
        ("association_repeated", "voxel_semantic"),
        ("association_repeated", "fragmentation"),
        ("unified_repeated", "combined"),
        ("false_split_pair", "pair_r1"),
    ]
    labels = ["mask / semantic", "assoc / semantic", "assoc / fragment", "unified / combined", "split / pair"]
    width = 0.34
    positions = np.arange(len(natural))
    for scene_position, scene in enumerate(SCENES):
        values = []
        aucs = []
        for task, score in natural:
            row = next(
                item
                for item in primary_rows
                if item["scene"] == scene
                and item["task"] == task
                and item["score"] == score
            )
            values.append(row.get("ap_minus_prevalence") or 0.0)
            aucs.append(row.get("auroc") or 0.5)
        offset = (scene_position - 0.5) * width
        axes[1, 0].bar(
            positions + offset,
            values,
            width,
            color=colors[scene],
            alpha=0.85,
            label=scene,
        )
        axes[1, 1].bar(
            positions + offset,
            aucs,
            width,
            color=colors[scene],
            alpha=0.85,
            label=scene,
        )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Corrected observation audit at 5 cm: AP lift")
    axes[1, 0].set_ylabel("AP - prevalence")
    axes[1, 0].set_xticks(positions, labels, rotation=23, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.2)
    axes[1, 1].axhline(0.5, color="gray", linewidth=0.8)
    axes[1, 1].set_title("Corrected observation audit at 5 cm: AUROC")
    axes[1, 1].set_ylabel("AUROC")
    axes[1, 1].set_ylim(0.2, 1.0)
    axes[1, 1].set_xticks(positions, labels, rotation=23, ha="right")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.2)
    fig.suptitle("Simple voxel trigger V0: gate and error-family diagnostics", fontsize=16)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_map_stats(summary: dict, output: Path) -> None:
    rows = summary["voxel_map_stats"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"room0": "#e24a33", "office0": "#348abd"}
    fields = [
        ("voxel_count", "voxel count", True),
        ("seen_median", "median seen_count", False),
        ("multi_label_voxel_fraction", "multi-label voxel fraction", False),
        ("npz_bytes", "compressed NPZ size (MB)", True),
    ]
    for axis, (field, title, log_scale) in zip(axes.flat, fields):
        for scene in SCENES:
            selected = sorted(
                [row for row in rows if row["scene"] == scene],
                key=lambda row: row["voxel_size_m"],
            )
            values = [row[field] for row in selected]
            if field == "npz_bytes":
                values = [value / 1024**2 for value in values]
            axis.plot(
                [row["voxel_size_m"] * 100 for row in selected],
                values,
                "o-",
                color=colors[scene],
                label=scene,
                linewidth=2,
            )
        axis.set_title(title)
        axis.set_xlabel("voxel size (cm)")
        if log_scale:
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Minimal voxel payload: scale and storage statistics", fontsize=16)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(output: Path, rows_by_scene: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for axis, scene in zip(axes, SCENES):
        rows = sorted(
            [row for row in rows_by_scene[scene] if not row["is_background"]],
            key=lambda row: (-float(row["combined_anomaly_score"]), int(row["object_index"])),
        )
        x = np.arange(1, len(rows) + 1)
        score = np.asarray([row["combined_anomaly_score"] for row in rows], dtype=float)
        target = np.asarray([row["audit_unified_repeated"] for row in rows], dtype=bool)
        axis.scatter(x[~target], score[~target], color="#4c9f70", s=45, label="audit clean", alpha=0.85)
        axis.scatter(x[target], score[target], color="#d1495b", s=45, label="audit error", alpha=0.85)
        for rank, row in enumerate(rows[:5], 1):
            axis.annotate(
                f"obj {row['object_index']}\n{row['predicted_label']}",
                (rank, row["combined_anomaly_score"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_ylim(-0.03, 1.08)
        axis.set_xlabel("rank by frozen combined score")
        axis.set_ylabel("combined score")
        axis.set_title(f"{scene}: 5 cm ranked objects")
        axis.grid(alpha=0.18)
        axis.legend(loc="lower left")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frozen = frozen_gate(root)
    scope = scope_equivalence(root)
    metrics = []
    map_stats = []
    voxel_examples = []
    scale_details = []
    primary_rows_by_scene: dict[str, list[dict]] = {}
    primary_pairs_by_scene: dict[str, list[dict]] = {}

    for scene in SCENES:
        observations = read_jsonl(root / scene / "observations.jsonl")
        for scale in SCALES:
            directory = root / scene / scale_dir(scale) / "all_history"
            object_rows = read_jsonl(directory / "objects.jsonl")
            pair_rows = read_jsonl(directory / "pairs.jsonl")
            audits = observation_audit(object_rows, observations)
            split = split_audit(object_rows, audits, pair_rows)
            split_incidents = split.pop("split_incidents")
            corrected_objects = add_object_targets(
                object_rows, audits, split_incidents
            )
            corrected_pairs = split.pop("pairs")
            metrics.extend(
                evaluate_scale(
                    scene=scene,
                    scale=scale,
                    object_rows=corrected_objects,
                    corrected_pairs=corrected_pairs,
                )
            )
            stat, examples = voxel_map_stats(directory / "voxel_map.npz", scene, scale)
            map_stats.append(stat)
            if scale == PRIMARY_SCALE:
                voxel_examples.extend(examples)
                primary_rows_by_scene[scene] = corrected_objects
                primary_pairs_by_scene[scene] = corrected_pairs
                write_jsonl(output / f"{scene}_objects_5cm_audited.jsonl", corrected_objects)
                write_jsonl(output / f"{scene}_pairs_5cm_audited.jsonl", corrected_pairs)
            foreground = [row for row in corrected_objects if not row["is_background"]]
            scale_details.append(
                {
                    "scene": scene,
                    "voxel_size_m": scale,
                    **split,
                    "foreground_objects": len(foreground),
                    "object_error_counts": {
                        field: int(sum(bool(row[field]) for row in foreground))
                        for field in (
                            "audit_mask_any",
                            "audit_mask_repeated",
                            "audit_mask_severe",
                            "audit_association_loose",
                            "audit_association_repeated",
                            "audit_overmerge_repeated",
                            "audit_overmerge_cross_class",
                            "audit_overmerge_same_class",
                            "audit_split_incident",
                            "audit_merge_membership_repeated",
                            "audit_unified_repeated",
                            "audit_unified_plus_spurious",
                            "audit_spurious_background",
                        )
                    },
                    "coverage": {
                        "observation_covered": int(
                            sum(row["audit_observation_coverage"] for row in foreground)
                        ),
                        "association_covered": int(
                            sum(row["audit_association_coverage"] for row in foreground)
                        ),
                    },
                }
            )

    primary_metrics = [row for row in metrics if row["voxel_size_m"] == PRIMARY_SCALE]
    bootstrap = []
    bootstrap_specs = (
        ("mask_repeated", "voxel_semantic"),
        ("association_repeated", "voxel_semantic"),
        ("association_repeated", "fragmentation"),
        ("unified_repeated", "combined"),
        ("false_split_pair", "pair_r1"),
    )
    for scene_index, scene in enumerate(SCENES):
        objects = [row for row in primary_rows_by_scene[scene] if not row["is_background"]]
        pairs = primary_pairs_by_scene[scene]
        for spec_index, (task, score_name) in enumerate(bootstrap_specs):
            if task == "false_split_pair":
                selected_pairs = [row for row in pairs if row["audit_pair_evaluable"]]
                labels = [row["audit_false_split_pair"] for row in selected_pairs]
                scores = [row["duplicate_score_r1"] for row in selected_pairs]
            else:
                coverage = (
                    "audit_association_coverage"
                    if task == "association_repeated"
                    else "audit_observation_coverage"
                )
                selected = [row for row in objects if row[coverage]]
                labels = [row["audit_" + task] for row in selected]
                scores = [row[SCORE_FIELDS[score_name]] for row in selected]
            bootstrap.append(
                {
                    "scene": scene,
                    "voxel_size_m": PRIMARY_SCALE,
                    "task": task,
                    "score": score_name,
                    **bootstrap_interval(
                        labels,
                        scores,
                        seed=20260831 + scene_index * 100 + spec_index,
                    ),
                }
            )

    top_objects = []
    object_cases_by_scene = {}
    for scene in SCENES:
        foreground = [
            row for row in primary_rows_by_scene[scene] if not row["is_background"]
        ]
        ranked = sorted(
            foreground,
            key=lambda row: (-float(row["combined_anomaly_score"]), int(row["object_index"])),
        )
        for rank, row in enumerate(ranked[:15], 1):
            top_objects.append({"rank": rank, **row})
        case_specs = []
        categories = (
            ("TP: high score, audited error", lambda row: row["audit_unified_repeated"], False),
            ("FP: high score, audited clean", lambda row: not row["audit_unified_repeated"], False),
            ("FN: low score, audited error", lambda row: row["audit_unified_repeated"], True),
            ("TN: low score, audited clean", lambda row: not row["audit_unified_repeated"], True),
        )
        for name, predicate, reverse in categories:
            candidates = list(reversed(ranked)) if reverse else ranked
            chosen = next((row for row in candidates if predicate(row)), None)
            if chosen is not None:
                case_specs.append(
                    {
                        "case": name,
                        "row_position": primary_rows_by_scene[scene].index(chosen),
                        "object_index": chosen["object_index"],
                    }
                )
        object_cases_by_scene[scene] = case_specs

    top_pairs = []
    pair_cases_by_scene = {}
    for scene in SCENES:
        ranked = sorted(
            primary_pairs_by_scene[scene],
            key=lambda row: (
                -float(row["duplicate_score_r1"]),
                int(row["left_index"]),
                int(row["right_index"]),
            ),
        )
        for rank, row in enumerate(ranked[:20], 1):
            top_pairs.append({"rank": rank, **row})
        true_case = next((row for row in ranked if row["audit_false_split_pair"]), None)
        false_case = next(
            (
                row
                for row in ranked
                if row["audit_pair_evaluable"] and not row["audit_false_split_pair"]
            ),
            None,
        )
        unresolved_case = next(
            (row for row in ranked if not row["audit_pair_evaluable"]), None
        )
        pair_cases_by_scene[scene] = [
            row for row in (true_case, false_case or unresolved_case) if row
        ]

    primary_selected = [
        row
        for row in primary_metrics
        if (row["task"], row["score"])
        in {
            ("mask_repeated", "nonspatial"),
            ("mask_repeated", "voxel_semantic"),
            ("association_repeated", "nonspatial"),
            ("association_repeated", "voxel_semantic"),
            ("association_repeated", "fragmentation"),
            ("unified_repeated", "combined"),
            ("unified_plus_spurious", "combined"),
            ("false_split_pair", "pair_r1"),
            ("false_split_pair", "pair_r2"),
        }
    ]

    pair_signal_stable = all(
        (
            next(
                row
                for row in metrics
                if row["scene"] == scene
                and row["voxel_size_m"] == scale
                and row["task"] == "false_split_pair"
                and row["score"] == "pair_r1"
            ).get("auroc")
            or 0.0
        )
        > 0.5
        for scene in SCENES
        for scale in SCALES
    )
    summary = {
        "schema_version": "voxel-trigger-v0-post-analysis/1.0",
        "root": str(root),
        "output": str(output),
        "gt_usage": "evaluation_only; scores and voxel payload unchanged",
        "gt_voxel_overlap_warning": {
            "status": "excluded_from_final_claim",
            "reason": (
                "Independent coarse voxelization lets one predicted voxel intersect "
                "multiple neighboring GT instances; per-instance intersections are "
                "therefore not mutually exclusive and inflated the frozen identity-error target."
            ),
            "replacement": (
                "Per-observation 2D Replica instance sidecars, with repeated-evidence "
                "and coverage sensitivity reported explicitly."
            ),
        },
        "frozen_gate": frozen,
        "scope_equivalence": scope,
        "voxel_map_stats": map_stats,
        "voxel_examples_5cm": voxel_examples,
        "scale_details": scale_details,
        "primary_corrected_metrics": primary_selected,
        "bootstrap_5cm": bootstrap,
        "decision": {
            "overall": "STOP",
            "unified_object_trigger": "STOP_AS_CURRENT_MAX_COMBINATION",
            "false_split_pair_component": "STOP_AS_CURRENT_LABEL_TIMES_PROXIMITY_SCORE",
            "pair_signal_auroc_above_random_all_scenes_scales": pair_signal_stable,
            "reason": (
                "The pre-registered unified GO gate fails. After collision-free "
                "observation-sidecar attribution, no voxel-derived component shows "
                "stable cross-scene and cross-scale improvement; the non-spatial "
                "owner-label entropy baseline is stronger for repeated mixed masks."
            ),
        },
    }

    write_json(output / "analysis_summary.json", summary)
    write_csv(output / "metrics_long.csv", metrics)
    write_csv(output / "primary_metrics_5cm.csv", primary_selected)
    write_csv(output / "voxel_map_stats.csv", map_stats)
    write_json(output / "voxel_examples_5cm.json", voxel_examples)
    write_csv(output / "top_objects_5cm.csv", top_objects)
    write_csv(output / "top_pairs_5cm.csv", top_pairs)
    write_json(output / "object_cases_5cm.json", object_cases_by_scene)

    plot_gate(summary, output / "01_gate_and_family_diagnostics.png")
    plot_map_stats(summary, output / "02_voxel_map_scale_storage.png")
    plot_ranking(output / "03_ranked_objects_5cm.png", primary_rows_by_scene)
    for scene in SCENES:
        plot_object_cases(
            root,
            output / f"04_{scene}_voxel_object_cases.png",
            scene,
            primary_rows_by_scene[scene],
            object_cases_by_scene[scene],
        )
    if all(len(pair_cases_by_scene[scene]) == 2 for scene in SCENES):
        plot_pair_cases(
            root,
            output / "05_false_split_pair_cases.png",
            pair_cases_by_scene,
        )
    (output / "READY").write_text("ready\n", encoding="utf-8")
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
