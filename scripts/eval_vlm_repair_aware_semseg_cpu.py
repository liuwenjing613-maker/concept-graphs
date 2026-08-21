#!/usr/bin/env python3
"""Repair-aware CPU Replica semantic-segmentation evaluation.

The evaluator reports the same geometry under two score sources:

``native_clip_ft``
    The frozen visual feature used by the standard ConceptGraphs protocol.
``map_class_name``
    A text embedding of the label stored in the map, including VLM repairs.

The second source is deliberately reported as a repair-aware diagnostic, not as
a replacement for the standard visual-feature benchmark.  No manual labels are
introduced; both sources are evaluated against the existing Replica semantic GT.
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


def load_text_encoder(
    model_name: str,
    pretrained: str,
    cache_dir: Path,
    torch_threads: int,
    device: str,
) -> tuple[Any, Any, Any]:
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
    return model, tokenizer, torch


def encode_text_features(
    prompts: list[str], model: Any, tokenizer: Any, torch: Any, device: str
) -> np.ndarray:
    with torch.inference_mode():
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        result = features.float().cpu().numpy()
    del tokens, features
    return result


def map_label_scores(
    objects: list[dict[str, Any]],
    class_features: np.ndarray,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    labels = [str(obj.get("class_name") or "").strip() for obj in objects]
    if any(not label for label in labels):
        raise ValueError("map_class_name scoring requires every object to have a label")
    unique_labels = list(dict.fromkeys(labels))
    label_features = encode_text_features(
        [f"an image of {label}" for label in unique_labels],
        model,
        tokenizer,
        torch,
        device,
    )
    label_to_feature = {
        label: label_features[index] for index, label in enumerate(unique_labels)
    }
    object_features = np.stack([label_to_feature[label] for label in labels])
    return object_features @ class_features.T, unique_labels


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


def row_from_metrics(
    score_source: str, scope: str, n_exclude: int, metrics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "score_source": score_source,
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

    print("Loading OpenCLIP text encoder on CPU", flush=True)
    model, tokenizer, torch = load_text_encoder(
        args.clip_model,
        args.clip_pretrained,
        args.clip_cache_dir,
        args.torch_threads,
        args.clip_device,
    )
    class_features = encode_text_features(
        [f"an image of {name}" for name in class_names],
        model,
        tokenizer,
        torch,
        args.clip_device,
    )

    score_sources = ("native_clip_ft", "map_class_name")
    conf_by_source: dict[str, dict[int, dict[str, np.ndarray]]] = {
        source: {value: {} for value in args.n_exclude} for source in score_sources
    }
    diagnostics: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []

    for scene_id, semantic_scene_id, map_name in args.scene:
        scene_started = time.time()
        map_path = Path(map_name).resolve()
        print(f"Loading {scene_id} map and point clouds", flush=True)
        objects = load_map(map_path)
        object_features = np.stack(
            [np.asarray(obj["clip_ft"], dtype=np.float32) for obj in objects]
        )
        object_features /= np.clip(
            np.linalg.norm(object_features, axis=1, keepdims=True), 1e-12, None
        )
        native_scores = object_features @ class_features.T
        label_scores, unique_labels = map_label_scores(
            objects,
            class_features,
            model,
            tokenizer,
            torch,
            args.clip_device,
        )
        scores_by_source = {
            "native_clip_ft": native_scores,
            "map_class_name": label_scores,
        }

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
            for score_source, object_scores in scores_by_source.items():
                selected_scores = object_scores.copy()
                selected_scores[:, ignore_index] = -1e10
                object_class = selected_scores.argmax(axis=1)
                slam_pred_class = object_class[slam_pred_owner]
                labels_pred = slam_pred_class[keep_points]
                flat = labels_gt * len(class_names) + labels_pred
                conf = np.bincount(
                    flat, minlength=len(class_names) * len(class_names)
                ).reshape(len(class_names), len(class_names))
                conf_by_source[score_source][n_exclude][scene_id] = conf

        diagnostics.append(
            {
                "scene_id": scene_id,
                "semantic_scene_id": semantic_scene_id,
                "predicted_objects": len(objects),
                "unique_map_labels": len(unique_labels),
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
    detail: dict[str, Any] = {source: {} for source in score_sources}
    npz_payload: dict[str, np.ndarray] = {}
    for score_source in score_sources:
        for n_exclude in args.n_exclude:
            detail[score_source][str(n_exclude)] = {}
            exclude_index = np.asarray(
                [class_names.index(name) for name in excluded_names[n_exclude]],
                dtype=np.int64,
            )
            for scene_id, semantic_scene_id, _map_name in args.scene:
                conf = conf_by_source[score_source][n_exclude][scene_id]
                present = np.flatnonzero(conf.sum(axis=1))
                keep = np.setdiff1d(present, exclude_index)
                reduced = conf[np.ix_(keep, keep)]
                metrics = exact_metrics(reduced, [class_names[i] for i in keep])
                detail[score_source][str(n_exclude)][scene_id] = metrics
                rows.append(
                    row_from_metrics(score_source, scene_id, n_exclude, metrics)
                )
                npz_payload[f"{score_source}_nexclude{n_exclude}_{scene_id}"] = conf

            combined = sum(conf_by_source[score_source][n_exclude].values())
            present = np.flatnonzero(combined.sum(axis=1))
            keep = np.setdiff1d(present, exclude_index)
            reduced = combined[np.ix_(keep, keep)]
            metrics = exact_metrics(reduced, [class_names[i] for i in keep])
            detail[score_source][str(n_exclude)][args.aggregate_label] = metrics
            rows.append(
                row_from_metrics(
                    score_source, args.aggregate_label, n_exclude, metrics
                )
            )
            npz_payload[
                f"{score_source}_nexclude{n_exclude}_{args.aggregate_label}"
            ] = combined

    result = {
        "format_version": 2,
        "scope": args.scope,
        "protocol": {
            "reference": "conceptgraph/scripts/eval_replica_semseg.py",
            "class_prompt": "an image of {class}",
            "map_label_prompt": "an image of {map class_name}",
            "score_sources": {
                "native_clip_ft": "standard frozen visual feature",
                "map_class_name": "repair-aware text embedding of saved map label",
            },
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "clip_device": args.clip_device,
            "nearest_neighbor_backend": "scipy.spatial.cKDTree exact k=1 on CPU",
            "slam_reconstruction_source": "frozen main stride5 rgb_cloud, shared scene geometry",
            "metric_formula_compatibility": "mirrors conceptgraph.utils.eval.compute_metrics",
            "n_exclude": args.n_exclude,
        },
        "inputs": input_records,
        "diagnostics": diagnostics,
        "summary_rows_percent": rows,
        "details_fraction": detail,
        "runtime_seconds": round(time.time() - started, 3),
    }
    del model
    gc.collect()
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
            f"{row['score_source']} nexclude={row['n_exclude']} {row['scope']}: "
            f"mIoU={row['miou']:.2f}% mRecall={row['mrecall']:.2f}% "
            f"mPrecision={row['mprecision']:.2f}% mF1={row['mf1score']:.2f}% "
            f"fwIoU={row['fmiou']:.2f}%",
            flush=True,
        )
    print(f"wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
