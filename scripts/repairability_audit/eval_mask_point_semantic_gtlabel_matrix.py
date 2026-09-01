#!/usr/bin/env python3
"""CPU Replica semantic-segmentation evaluation with isolated label controls.

The metric definitions and class prompts mirror ConceptGraphs'
``conceptgraph/scripts/eval_replica_semseg.py``.  The only implementation
change is replacing CUDA KNN calls with SciPy cKDTree exact-nearest queries.

The default ``native_clip`` path is unchanged. ``oracle_gt_label`` is a
diagnostic control: it replaces only the semantic feature used by the readout
with the exact Replica prompt feature for an object's stored oracle label. It
does not modify geometry, associations, point ownership, or the GT denominator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--replica-semantic-root", type=Path, required=True)
    parser.add_argument(
        "--slam-root",
        type=Path,
        required=True,
        help="Replica root containing per-scene rgb_cloud reconstructions",
    )
    parser.add_argument(
        "--scene",
        action="append",
        nargs=3,
        metavar=("SCENE_ID", "SEMANTIC_SCENE_ID", "MAP_PICKLE"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-exclude", type=int, nargs="+", default=[1, 4, 6])
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--clip-cache-dir", type=Path, required=True)
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--feature-source",
        choices=(
            "native_clip",
            "oracle_gt_label",
            "oracle_eval_majority_label",
        ),
        default="native_clip",
    )
    parser.add_argument(
        "--scope",
        default="frozen_ali_my_room0_office0_200_frames_stride10",
    )
    parser.add_argument("--aggregate-label", default="all_available_ali_my")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_map(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    objects = payload.get("objects") or []
    if not objects:
        raise ValueError(f"Map contains no objects: {path}")
    return objects


def encode_class_features(
    class_names: list[str],
    model_name: str,
    pretrained: str,
    cache_dir: Path,
    torch_threads: int,
    device: str,
) -> np.ndarray:
    import open_clip
    import torch

    if device == "cpu":
        torch.set_num_threads(torch_threads)
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=str(cache_dir),
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    prompts = [f"an image of {name}" for name in class_names]
    with torch.inference_mode():
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        result = features.float().cpu().numpy()
    del model, tokens, features
    gc.collect()
    return result


def exact_metrics(confmatrix: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Mirror conceptgraph.utils.eval.compute_metrics, including its F1 formula."""
    num_classes = len(class_names)
    ious = np.zeros(num_classes, dtype=np.float64)
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1score = np.zeros(num_classes, dtype=np.float64)
    for index in range(num_classes):
        true_positive = float(confmatrix[index, index])
        union = float(
            confmatrix[index, :].sum()
            + confmatrix[:, index].sum()
            - confmatrix[index, index]
        )
        ious[index] = true_positive / max(1.0, union)
        recall[index] = true_positive / max(1.0, float(confmatrix[index, :].sum()))
        precision[index] = true_positive / max(
            1.0, float(confmatrix[:, index].sum())
        )
        f1score[index] = (
            2.0
            * precision[index]
            * recall[index]
            / max(1.0, precision[index] + recall[index])
        )
    total = float(confmatrix.sum())
    fmiou = float((ious * confmatrix.sum(axis=1) / max(1.0, total)).sum())
    return {
        "class_names": class_names,
        "num_classes": num_classes,
        "point_count": int(confmatrix.sum()),
        "iou": ious.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1score": f1score.tolist(),
        "miou": float(ious.mean()),
        "mrecall": float(recall.mean()),
        "mprecision": float(precision.mean()),
        "mf1score": float(f1score.mean()),
        "fmiou": fmiou,
        "point_accuracy": float(np.trace(confmatrix) / max(1.0, total)),
        "acc0.15": int((ious > 0.15).sum()),
        "acc0.25": int((ious > 0.25).sum()),
        "acc0.50": int((ious > 0.50).sum()),
        "acc0.75": int((ious > 0.75).sum()),
    }


def row_from_metrics(scope: str, n_exclude: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": scope,
        "n_exclude": n_exclude,
        "num_classes": metrics["num_classes"],
        "point_count": metrics["point_count"],
        "miou": 100.0 * metrics["miou"],
        "mrecall": 100.0 * metrics["mrecall"],
        "mprecision": 100.0 * metrics["mprecision"],
        "mf1score": 100.0 * metrics["mf1score"],
        "fmiou": 100.0 * metrics["fmiou"],
        "point_accuracy": 100.0 * metrics["point_accuracy"],
        "acc0.15": metrics["acc0.15"],
        "acc0.25": metrics["acc0.25"],
        "acc0.50": metrics["acc0.50"],
        "acc0.75": metrics["acc0.75"],
    }


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.project_root))

    from gradslam.structures.pointclouds import Pointclouds
    from conceptgraph.dataset.replica_constants import (
        REPLICA_CLASSES,
        REPLICA_EXISTING_CLASSES,
    )

    class_all2existing = np.full(len(REPLICA_CLASSES), -1, dtype=np.int64)
    for existing_index, all_index in enumerate(REPLICA_EXISTING_CLASSES):
        class_all2existing[int(all_index)] = existing_index
    class_names = [REPLICA_CLASSES[index] for index in REPLICA_EXISTING_CLASSES]
    excluded_names = {
        1: ["other"],
        4: ["other", "floor", "wall", "ceiling"],
        6: ["other", "floor", "wall", "ceiling", "door", "window"],
    }
    invalid = sorted(set(args.n_exclude) - set(excluded_names))
    if invalid:
        raise ValueError(f"Unsupported n_exclude values: {invalid}")

    class_features = None
    if args.feature_source != "oracle_eval_majority_label":
        print("Encoding Replica class prompts on CPU", flush=True)
        class_features = encode_class_features(
            class_names,
            args.clip_model,
            args.clip_pretrained,
            args.clip_cache_dir,
            args.torch_threads,
            args.clip_device,
        )

    conf_by_exclusion: dict[int, dict[str, np.ndarray]] = {
        value: {} for value in args.n_exclude
    }
    diagnostics: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    label_control_records: list[dict[str, Any]] = []

    for scene_id, semantic_scene_id, map_name in args.scene:
        scene_started = time.time()
        map_path = Path(map_name).resolve()
        print(f"Loading {scene_id} map and point clouds", flush=True)
        objects = load_map(map_path)
        if args.feature_source == "native_clip":
            object_features = np.stack(
                [np.asarray(obj["clip_ft"], dtype=np.float32) for obj in objects]
            )
            source_labels = None
            adapted_labels = None
        elif args.feature_source == "oracle_gt_label":
            source_labels = []
            adapted_labels = []
            for object_index, obj in enumerate(objects):
                label = obj.get("oracle_gt_label")
                if label is None:
                    raise ValueError(
                        f"{scene_id} object {object_index} lacks oracle_gt_label"
                    )
                label = str(label)
                source_labels.append(label)
                adapted_labels.append(label if label in class_names else "other")
            class_index = {name: index for index, name in enumerate(class_names)}
            object_features = np.stack(
                [class_features[class_index[label]] for label in adapted_labels]
            )
            from collections import Counter

            label_control_records.append(
                {
                    "scene_id": scene_id,
                    "source_label_counts": dict(sorted(Counter(source_labels).items())),
                    "adapted_label_counts": dict(sorted(Counter(adapted_labels).items())),
                    "out_of_ontology_labels_mapped_to_other": dict(
                        sorted(
                            Counter(
                                source
                                for source, adapted in zip(source_labels, adapted_labels)
                                if adapted == "other" and source != "other"
                            ).items()
                        )
                    ),
                }
            )
        else:
            source_labels = None
            adapted_labels = None
            object_features = None

        if object_features is not None:
            object_features /= np.clip(
                np.linalg.norm(object_features, axis=1, keepdims=True), 1e-12, None
            )
            assert class_features is not None
            object_scores = object_features @ class_features.T
        else:
            object_scores = None

        pred_points: list[np.ndarray] = []
        pred_owners: list[np.ndarray] = []
        for object_index, obj in enumerate(objects):
            points = np.asarray(obj["pcd_np"], dtype=np.float32)
            if not len(points):
                continue
            pred_points.append(points)
            pred_owners.append(np.full(len(points), object_index, dtype=np.int32))
        pred_xyz = np.concatenate(pred_points, axis=0)
        pred_owner = np.concatenate(pred_owners, axis=0)

        gt_root = args.replica_semantic_root / semantic_scene_id / "Sequence_1"
        gt_map = Pointclouds.load_pointcloud_from_h5(gt_root / "saved-maps-gt")
        gt_poses = np.loadtxt(gt_root / "traj_w_c.txt").reshape(-1, 4, 4)
        gt_xyz = gt_map.points_padded[0].cpu().numpy().astype(np.float32, copy=False)
        gt_xyz = gt_xyz @ gt_poses[0, :3, :3].T + gt_poses[0, :3, 3]
        gt_embedding = gt_map.embeddings_padded[0].cpu().numpy()
        gt_class_all = gt_embedding.argmax(axis=1)
        gt_class = class_all2existing[gt_class_all]
        if int(gt_class.min()) < 0:
            raise ValueError(f"Unmapped GT semantic class in {scene_id}")

        slam_path = args.slam_root / scene_id / "rgb_cloud"
        slam_cloud = Pointclouds.load_pointcloud_from_h5(slam_path)
        slam_xyz = slam_cloud.points_padded[0].cpu().numpy().astype(np.float32, copy=False)

        print(
            f"{scene_id}: exact CPU nearest-neighbor queries "
            f"slam={len(slam_xyz)} pred={len(pred_xyz)} gt={len(gt_xyz)}",
            flush=True,
        )
        pred_distance, slam_to_pred = cKDTree(pred_xyz).query(
            slam_xyz, k=1, workers=args.workers
        )
        gt_distance, slam_to_gt = cKDTree(gt_xyz).query(
            slam_xyz, k=1, workers=args.workers
        )
        slam_pred_owner = pred_owner[slam_to_pred]
        slam_gt_class = gt_class[slam_to_gt]
        existing_scene_classes = np.unique(gt_class)

        for n_exclude in args.n_exclude:
            exclude_index = np.asarray(
                [class_names.index(name) for name in excluded_names[n_exclude]],
                dtype=np.int64,
            )
            non_existing = np.setdiff1d(
                np.arange(len(class_names), dtype=np.int64), existing_scene_classes
            )
            ignore_index = np.unique(np.concatenate([exclude_index, non_existing]))
            keep_index = np.setdiff1d(
                np.arange(len(class_names), dtype=np.int64), ignore_index
            )
            keep_points = np.isin(slam_gt_class, keep_index)
            labels_gt = slam_gt_class[keep_points]
            if args.feature_source == "oracle_eval_majority_label":
                # One semantic label per predicted object. Votes come from the exact
                # SLAM->prediction and SLAM->GT correspondences used by the official
                # metric. This is the optimal object-level label oracle conditional
                # on the current geometry/partition; it is deliberately not a
                # pointwise GT replacement.
                vote_matrix = np.zeros(
                    (len(objects), len(class_names)), dtype=np.int64
                )
                np.add.at(
                    vote_matrix,
                    (slam_pred_owner[keep_points], labels_gt),
                    1,
                )
                kept_votes = vote_matrix[:, keep_index]
                vote_totals = kept_votes.sum(axis=1)
                best_local = kept_votes.argmax(axis=1)
                object_class = keep_index[best_local]
                # Zero-vote objects receive an arbitrary kept class, but cannot
                # influence this metric because no retained evaluation point is
                # assigned to them.
                zero_vote = vote_totals == 0
                object_class[zero_vote] = keep_index[0]
                winning_votes = kept_votes[
                    np.arange(len(objects)), best_local
                ]
                nonzero = ~zero_vote
                purities = (
                    winning_votes[nonzero] / vote_totals[nonzero]
                    if np.any(nonzero)
                    else np.empty(0, dtype=np.float64)
                )
                from collections import Counter

                label_control_records.append(
                    {
                        "scene_id": scene_id,
                        "n_exclude": n_exclude,
                        "definition": (
                            "one majority GT class per predicted object using the "
                            "same retained SLAM evaluation points and exact nearest "
                            "correspondences as the official point metric"
                        ),
                        "predicted_objects": len(objects),
                        "objects_with_retained_votes": int(nonzero.sum()),
                        "objects_without_retained_votes": int(zero_vote.sum()),
                        "assigned_label_counts_nonzero": dict(
                            sorted(
                                Counter(
                                    class_names[index]
                                    for index in object_class[nonzero]
                                ).items()
                            )
                        ),
                        "object_vote_purity_mean": (
                            float(purities.mean()) if len(purities) else None
                        ),
                        "object_vote_purity_median": (
                            float(np.median(purities)) if len(purities) else None
                        ),
                        "object_vote_purity_min": (
                            float(purities.min()) if len(purities) else None
                        ),
                    }
                )
            else:
                assert object_scores is not None
                selected_scores = object_scores.copy()
                selected_scores[:, ignore_index] = -1e10
                object_class = selected_scores.argmax(axis=1)
            slam_pred_class = object_class[slam_pred_owner]
            labels_pred = slam_pred_class[keep_points]
            flat = labels_gt * len(class_names) + labels_pred
            conf = np.bincount(
                flat, minlength=len(class_names) * len(class_names)
            ).reshape(len(class_names), len(class_names))
            conf_by_exclusion[n_exclude][scene_id] = conf

        diagnostics.append(
            {
                "scene_id": scene_id,
                "semantic_scene_id": semantic_scene_id,
                "predicted_objects": len(objects),
                "predicted_object_points": len(pred_xyz),
                "slam_points": len(slam_xyz),
                "gt_points": len(gt_xyz),
                "slam_to_pred_distance_mean_m": float(pred_distance.mean()),
                "slam_to_pred_distance_p95_m": float(np.percentile(pred_distance, 95)),
                "slam_to_gt_distance_mean_m": float(gt_distance.mean()),
                "slam_to_gt_distance_p95_m": float(np.percentile(gt_distance, 95)),
                "runtime_seconds": round(time.time() - scene_started, 3),
            }
        )
        input_records.append(
            {
                "scene_id": scene_id,
                "map_pickle": str(map_path),
                "map_pickle_sha256": sha256_file(map_path),
                "semantic_gt": str(gt_root / "saved-maps-gt"),
                "slam_rgb_cloud": str(slam_path),
            }
        )
        del objects, gt_map, slam_cloud, pred_xyz, gt_xyz, gt_embedding
        gc.collect()

    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    npz_payload: dict[str, np.ndarray] = {}
    for n_exclude in args.n_exclude:
        detail[str(n_exclude)] = {}
        exclude_index = np.asarray(
            [class_names.index(name) for name in excluded_names[n_exclude]],
            dtype=np.int64,
        )
        for scene_id, semantic_scene_id, _map_name in args.scene:
            conf = conf_by_exclusion[n_exclude][scene_id]
            present = np.flatnonzero(conf.sum(axis=1))
            keep = np.setdiff1d(present, exclude_index)
            reduced = conf[np.ix_(keep, keep)]
            metrics = exact_metrics(reduced, [class_names[i] for i in keep])
            detail[str(n_exclude)][scene_id] = metrics
            rows.append(row_from_metrics(scene_id, n_exclude, metrics))
            npz_payload[f"nexclude{n_exclude}_{scene_id}"] = conf

        combined = sum(conf_by_exclusion[n_exclude].values())
        present = np.flatnonzero(combined.sum(axis=1))
        keep = np.setdiff1d(present, exclude_index)
        reduced = combined[np.ix_(keep, keep)]
        metrics = exact_metrics(reduced, [class_names[i] for i in keep])
        detail[str(n_exclude)][args.aggregate_label] = metrics
        rows.append(row_from_metrics(args.aggregate_label, n_exclude, metrics))
        npz_payload[f"nexclude{n_exclude}_{args.aggregate_label}"] = combined

    result = {
        "format_version": 1,
        "scope": args.scope,
        "protocol": {
            "reference": "conceptgraph/scripts/eval_replica_semseg.py",
            "class_prompt": "an image of {class}",
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "clip_device": args.clip_device,
            "nearest_neighbor_backend": "scipy.spatial.cKDTree exact k=1 on CPU",
            "feature_source": args.feature_source,
            "label_control_definition": (
                (
                    "exact stored oracle_gt_label -> matching Replica class prompt; "
                    "out-of-ontology labels -> other; official exclusion suppression retained"
                )
                if args.feature_source == "oracle_gt_label"
                else (
                    "one majority GT class per predicted object, derived from the "
                    "same retained SLAM evaluation points and exact nearest "
                    "correspondences used by the official metric; no pointwise GT replacement"
                    if args.feature_source == "oracle_eval_majority_label"
                    else None
                )
            ),
            "slam_reconstruction_source": "frozen main stride5 rgb_cloud, shared scene geometry",
            "metric_formula_compatibility": "mirrors conceptgraph.utils.eval.compute_metrics",
            "n_exclude": args.n_exclude,
        },
        "inputs": input_records,
        "diagnostics": diagnostics,
        "label_control_records": label_control_records,
        "summary_rows_percent": rows,
        "details_fraction": detail,
        "runtime_seconds": round(time.time() - started, 3),
    }
    result_path = args.output_dir / "semseg_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "semseg_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "semseg_conf_matrices.npz", **npz_payload)

    for row in rows:
        print(
            f"nexclude={row['n_exclude']} {row['scope']}: "
            f"mIoU={row['miou']:.2f}% mRecall={row['mrecall']:.2f}% "
            f"mPrecision={row['mprecision']:.2f}% mF1={row['mf1score']:.2f}% "
            f"fwIoU={row['fmiou']:.2f}%",
            flush=True,
        )
    print(f"wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
