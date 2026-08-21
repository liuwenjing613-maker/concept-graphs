#!/usr/bin/env python3
"""Repair-aware evaluation of one ConceptGraphs map on ReplicaSSG.

The geometry matching and recall definitions follow FROSS's public ReplicaSSG
evaluator.  Object metrics are reported for both the standard native visual
feature and a text embedding of the map's saved ``class_name``.  The latter is
the repair-aware track and consumes no new manual annotations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="ReplicaSSG scene name, e.g. room_0")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--map-pickle", type=Path, required=True)
    parser.add_argument("--scene-graph", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--stage7-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap-threshold", type=float, default=0.1)
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--clip-cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_scan(scans: list[dict[str, Any]], scene: str) -> dict[str, Any]:
    for scan in scans:
        if scan["scan"] == scene:
            return scan
    raise KeyError(f"Scene {scene!r} is absent from annotations")


def load_ground_truth(
    scene: str, annotations_dir: Path
) -> tuple[list[int], list[str], list[tuple[int, int, str]], list[str], list[str]]:
    mapping = read_json(annotations_dir / "replica_to_visual_genome.json")
    object_scan = select_scan(read_json(annotations_dir / "objects.json")["scans"], scene)
    relation_scan = select_scan(
        read_json(annotations_dir / "relationships.json")["scans"], scene
    )

    replica_to_vg = mapping["Replica2VisualGenome"]
    object_classes = list(mapping["VisualGenome_list"])
    predicate_classes = list(mapping["VisualGenome_rel"])

    object_ids: list[int] = []
    labels: list[str] = []
    valid_ids: set[int] = set()
    for obj in object_scan["objects"]:
        replica_label = obj["label"]
        if replica_label not in replica_to_vg:
            continue
        vg_label = replica_to_vg[replica_label]
        if vg_label not in object_classes:
            continue
        object_id = int(obj["id"])
        object_ids.append(object_id)
        labels.append(vg_label)
        valid_ids.add(object_id)

    relations: list[tuple[int, int, str]] = []
    for subject, object_, _predicate_id, predicate in relation_scan["relationships"]:
        if subject not in valid_ids or object_ not in valid_ids:
            raise ValueError(
                f"Relation endpoint absent after object filtering: {(subject, object_, predicate)}"
            )
        if predicate not in predicate_classes:
            raise ValueError(f"Unknown ReplicaSSG predicate: {predicate!r}")
        relations.append((int(subject), int(object_), predicate))
    return object_ids, labels, relations, object_classes, predicate_classes


def load_prediction_objects(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    objects = payload["objects"]
    if not objects:
        raise ValueError("ConceptGraphs map contains no predicted objects")
    return objects


def match_instances(
    gt_ply: Path,
    gt_object_ids: list[int],
    pred_objects: list[dict[str, Any]],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror FROSS's one-way 0.1 m KD-tree matching exactly."""
    mesh = PlyData.read(str(gt_ply))
    vertices = mesh["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    point_object_ids = np.asarray(vertices["objectId"])

    valid_mask = np.isin(point_object_ids, np.asarray(gt_object_ids))
    points = points[valid_mask]
    point_object_ids = point_object_ids[valid_mask]
    object_id_to_index = {object_id: index for index, object_id in enumerate(gt_object_ids)}

    overlap_count = np.zeros((len(gt_object_ids), len(pred_objects)), dtype=np.int64)
    tree = KDTree(points)
    for pred_index, pred_object in enumerate(pred_objects):
        segment = np.asarray(pred_object["pcd_np"])
        if len(segment) == 0:
            continue
        _distances, indices = tree.query(segment, distance_upper_bound=threshold)
        matched_indices = indices[indices != tree.n]
        matched_gt = np.fromiter(
            (object_id_to_index[int(point_object_ids[i])] for i in matched_indices),
            dtype=np.int64,
            count=len(matched_indices),
        )
        if len(matched_gt):
            overlap_count[:, pred_index] = np.bincount(
                matched_gt, minlength=len(gt_object_ids)
            )

        sorted_gt = np.flip(np.argsort(overlap_count[:, pred_index], kind="stable"))
        best = int(sorted_gt[0])
        best_fraction = overlap_count[best, pred_index] / len(segment)
        second_fraction = 0.0
        if len(sorted_gt) > 1:
            second = int(sorted_gt[1])
            second_fraction = overlap_count[second, pred_index] / len(segment)
        ambiguity = second_fraction / best_fraction if best_fraction > 0 else np.inf
        if best_fraction < 0.5 or ambiguity > 0.75:
            overlap_count[:, pred_index] = 0
        else:
            overlap_count[np.arange(len(gt_object_ids)) != best, pred_index] = 0

    gt_to_pred = np.full(len(gt_object_ids), -1, dtype=np.int64)
    matched_points = np.zeros(len(gt_object_ids), dtype=np.int64)
    for gt_index in range(len(gt_object_ids)):
        pred_index = int(np.argmax(overlap_count[gt_index]))
        if overlap_count[gt_index, pred_index] > 0:
            gt_to_pred[gt_index] = pred_index
            matched_points[gt_index] = overlap_count[gt_index, pred_index]
    return gt_to_pred, matched_points, overlap_count


def encode_texts(
    texts: list[str], model: Any, tokenizer: Any, device: str, batch_size: int = 256
) -> np.ndarray:
    import torch

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start : start + batch_size]).to(device)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            result.append(features.float().cpu().numpy())
    return np.concatenate(result, axis=0)


def build_object_scores(
    pred_objects: list[dict[str, Any]],
    object_classes: list[str],
    scene_graph: list[dict[str, Any]],
    model_name: str,
    pretrained: str,
    cache_dir: Path | None,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=str(cache_dir) if cache_dir else None,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    class_prompts = [f"a photo of a {name}" for name in object_classes]
    class_features = encode_texts(class_prompts, model, tokenizer, device)

    visual_features = np.stack([np.asarray(obj["clip_ft"]) for obj in pred_objects])
    visual_norms = np.linalg.norm(visual_features, axis=1, keepdims=True)
    visual_features = visual_features / np.clip(visual_norms, 1e-12, None)
    main_clip_scores = visual_features @ class_features.T

    map_labels = [str(obj.get("class_name") or "").strip() for obj in pred_objects]
    if any(not label for label in map_labels):
        raise ValueError("map_class_name scoring requires every object to have a label")
    unique_map_labels = list(dict.fromkeys(map_labels))
    map_label_features = encode_texts(
        [f"a photo of a {label}" for label in unique_map_labels],
        model,
        tokenizer,
        device,
    )
    label_to_feature = {
        label: map_label_features[index]
        for index, label in enumerate(unique_map_labels)
    }
    map_label_scores = np.stack(
        [label_to_feature[label] @ class_features.T for label in map_labels]
    )

    graph_by_original_id = {int(node["id"]): node for node in scene_graph}
    tag_scores = np.full(
        (len(pred_objects), len(object_classes)), -np.inf, dtype=np.float32
    )
    possible_tag_scores = np.full_like(tag_scores, -np.inf)
    unique_tags: list[str] = []
    for node in scene_graph:
        tags = [node.get("object_tag", "")] + list(node.get("possible_tags", []))
        for tag in tags:
            tag = str(tag).strip()
            if tag and tag not in unique_tags:
                unique_tags.append(tag)
    tag_features = encode_texts(
        [f"a photo of a {tag}" for tag in unique_tags], model, tokenizer, device
    )
    tag_to_feature = {tag: tag_features[i] for i, tag in enumerate(unique_tags)}

    for original_id, node in graph_by_original_id.items():
        if not (0 <= original_id < len(pred_objects)):
            raise ValueError(f"Scene graph original id is out of range: {original_id}")
        final_tag = str(node.get("object_tag", "")).strip()
        if final_tag:
            tag_scores[original_id] = tag_to_feature[final_tag] @ class_features.T
        tags = [final_tag] + [str(t).strip() for t in node.get("possible_tags", [])]
        tags = [tag for tag in tags if tag]
        if tags:
            scores = np.stack([tag_to_feature[tag] @ class_features.T for tag in tags])
            possible_tag_scores[original_id] = scores.max(axis=0)

    metadata = {
        "clip_model": model_name,
        "clip_pretrained": pretrained,
        "class_prompt": "a photo of a {class}",
        "tag_prompt": "a photo of a {tag}",
        "map_label_prompt": "a photo of a {map class_name}",
        "unique_map_labels": len(unique_map_labels),
        "scene_graph_nodes": len(scene_graph),
        "unique_scene_graph_tags": len(unique_tags),
    }
    return {
        "main_native_clip_ft": main_clip_scores,
        "map_class_name": map_label_scores,
        "scene_graph_object_tag": tag_scores,
        "llava_llm_possible_tags_max": possible_tag_scores,
    }, metadata


def stable_rank(scores: np.ndarray, class_index: int) -> int:
    order = np.flip(np.argsort(scores, kind="stable"))
    return int(np.nonzero(order == class_index)[0].item())


def object_metrics(
    scores: np.ndarray,
    gt_to_pred: np.ndarray,
    gt_labels: list[str],
    object_classes: list[str],
) -> tuple[dict[str, Any], list[int | None]]:
    class_to_index = {name: i for i, name in enumerate(object_classes)}
    ranks: list[int | None] = []
    per_class_ranks: dict[str, list[int | None]] = defaultdict(list)
    for gt_index, label in enumerate(gt_labels):
        pred_index = int(gt_to_pred[gt_index])
        rank: int | None = None
        if pred_index >= 0 and np.isfinite(scores[pred_index]).any():
            rank = stable_rank(scores[pred_index], class_to_index[label])
        ranks.append(rank)
        per_class_ranks[label].append(rank)

    def recall(values: list[int | None], k: int) -> float:
        return sum(rank is not None and rank < k for rank in values) / len(values)

    per_class = {
        label: {
            "count": len(values),
            "hits_at_1": sum(rank is not None and rank < 1 for rank in values),
            "hits_at_5": sum(rank is not None and rank < 5 for rank in values),
            "recall_at_1": recall(values, 1),
            "recall_at_5": recall(values, 5),
        }
        for label, values in sorted(per_class_ranks.items())
    }
    metrics = {
        "count": len(ranks),
        "matched_geometry": int(np.sum(gt_to_pred >= 0)),
        "hits_at_1": sum(rank is not None and rank < 1 for rank in ranks),
        "hits_at_5": sum(rank is not None and rank < 5 for rank in ranks),
        "recall_at_1": recall(ranks, 1),
        "recall_at_5": recall(ranks, 5),
        "mean_recall_at_1": float(np.mean([v["recall_at_1"] for v in per_class.values()])),
        "mean_recall_at_5": float(np.mean([v["recall_at_5"] for v in per_class.values()])),
        "classes_present": len(per_class),
        "per_class": per_class,
    }
    return metrics, ranks


def predicted_relations(
    relation_payload: list[dict[str, Any]], audit: dict[str, Any]
) -> tuple[dict[tuple[int, int], set[str]], dict[str, int]]:
    pruned_to_original = {
        int(node["pruned_id"]): int(node["original_id"]) for node in audit["nodes"]
    }
    result: dict[tuple[int, int], set[str]] = defaultdict(set)
    counts = defaultdict(int)
    for item in relation_payload:
        a_pruned = int(item["object1"]["id"])
        b_pruned = int(item["object2"]["id"])
        if a_pruned not in pruned_to_original or b_pruned not in pruned_to_original:
            raise ValueError(f"Relation uses unknown pruned ids: {(a_pruned, b_pruned)}")
        a = pruned_to_original[a_pruned]
        b = pruned_to_original[b_pruned]
        relation = str(item["object_relation"]).strip().lower()
        counts[relation] += 1
        if relation == "a on b":
            result[(a, b)].add("on")
        elif relation == "b on a":
            result[(b, a)].add("on")
        elif relation == "a in b":
            result[(a, b)].add("in")
        elif relation == "b in a":
            result[(b, a)].add("in")
        elif relation == "none of these":
            continue
        else:
            raise ValueError(f"Unsupported Stage-7 relation label: {relation!r}")
    return result, dict(sorted(counts.items()))


def predicate_metrics(
    gt_relations: list[tuple[int, int, str]],
    gt_object_ids: list[int],
    gt_to_pred: np.ndarray,
    predicted: dict[tuple[int, int], set[str]],
) -> tuple[dict[str, Any], list[bool]]:
    object_id_to_gt = {object_id: index for index, object_id in enumerate(gt_object_ids)}
    hits: list[bool] = []
    per_class: dict[str, list[bool]] = defaultdict(list)
    for subject_id, object_id, predicate in gt_relations:
        subject_pred = int(gt_to_pred[object_id_to_gt[subject_id]])
        object_pred = int(gt_to_pred[object_id_to_gt[object_id]])
        hit = (
            subject_pred >= 0
            and object_pred >= 0
            and predicate in predicted.get((subject_pred, object_pred), set())
        )
        hits.append(hit)
        per_class[predicate].append(hit)
    per_class_summary = {
        label: {
            "count": len(values),
            "hits_at_1": sum(values),
            "recall_at_1": sum(values) / len(values),
        }
        for label, values in sorted(per_class.items())
    }
    return {
        "count": len(hits),
        "hits_at_1": sum(hits),
        "recall_at_1": sum(hits) / len(hits),
        "mean_recall_at_1": float(
            np.mean([values["recall_at_1"] for values in per_class_summary.values()])
        ),
        "classes_present": len(per_class_summary),
        "per_class": per_class_summary,
    }, hits


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_ids, gt_labels, gt_relations, object_classes, predicate_classes = load_ground_truth(
        args.scene, args.annotations_dir
    )
    pred_objects = load_prediction_objects(args.map_pickle)
    gt_ply = (
        args.dataset_root
        / "data"
        / args.scene
        / "labels.instances.annotated.v2.ply"
    )
    gt_to_pred, matched_points, overlap_count = match_instances(
        gt_ply, gt_ids, pred_objects, args.overlap_threshold
    )
    valid_predicted = overlap_count.sum(axis=0) > 0
    fragments_per_gt = (overlap_count > 0).sum(axis=1)
    covered_gt = fragments_per_gt > 0
    valid_predicted_count = int(valid_predicted.sum())
    covered_gt_count = int(covered_gt.sum())

    scene_graph = read_json(args.scene_graph)
    score_methods, score_metadata = build_object_scores(
        pred_objects,
        object_classes,
        scene_graph,
        args.clip_model,
        args.clip_pretrained,
        args.clip_cache_dir,
        args.device,
    )
    object_results: dict[str, Any] = {}
    object_ranks: dict[str, list[int | None]] = {}
    for method, scores in score_methods.items():
        metrics, ranks = object_metrics(
            scores, gt_to_pred, gt_labels, object_classes
        )
        object_results[method] = metrics
        object_ranks[method] = ranks

    relation_payload = read_json(args.relations)
    stage7_audit = read_json(args.stage7_audit)
    pred_edges, raw_relation_counts = predicted_relations(relation_payload, stage7_audit)
    pred_metrics, predicate_hits = predicate_metrics(
        gt_relations, gt_ids, gt_to_pred, pred_edges
    )

    result = {
        "format_version": 1,
        "scope": "single_scene",
        "scene": args.scene,
        "protocol": {
            "reference": "FROSS Merging/evaluate.py (ReplicaSSG mode)",
            "geometry_threshold_m": args.overlap_threshold,
            "predicted_segment_min_best_overlap_fraction": 0.5,
            "predicted_segment_max_second_to_best_ratio": 0.75,
            "mean_recall": "unweighted mean over GT classes present in this evaluation scope",
            "object_k": [1, 5],
            "predicate_k": [1],
            "predicate_ignores_object_class": True,
            "map_class_name_track": (
                "repair-aware text-to-text classification; not a replacement "
                "for the native visual-feature benchmark"
            ),
            "geometry_diagnostic_scope": (
                "ReplicaSSG annotated-object scope; unmatched predictions may "
                "include valid map objects outside this GT scope"
            ),
        },
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "annotations_dir": str(args.annotations_dir),
            "map_pickle": str(args.map_pickle),
            "scene_graph": str(args.scene_graph),
            "relations": str(args.relations),
            "stage7_audit": str(args.stage7_audit),
        },
        "integrity": {
            "gt_objects": len(gt_ids),
            "gt_object_classes_present": len(set(gt_labels)),
            "gt_relations": len(gt_relations),
            "gt_predicate_classes_present": len(set(r[2] for r in gt_relations)),
            "predicted_objects": len(pred_objects),
            "scene_graph_nodes_after_pruning": len(scene_graph),
            "geometry_matched_gt_objects": int(np.sum(gt_to_pred >= 0)),
            "geometry_valid_predicted_objects": valid_predicted_count,
            "geometry_unmatched_predicted_objects": len(pred_objects)
            - valid_predicted_count,
            "geometry_valid_prediction_rate": valid_predicted_count
            / len(pred_objects),
            "geometry_one_to_one_coverage_per_prediction": covered_gt_count
            / len(pred_objects),
            "geometry_gt_objects_with_multiple_fragments": int(
                np.sum(fragments_per_gt > 1)
            ),
            "geometry_fragmentation_excess": int(
                np.maximum(fragments_per_gt - 1, 0).sum()
            ),
            "geometry_mean_valid_fragments_per_covered_gt": (
                valid_predicted_count / covered_gt_count
                if covered_gt_count
                else 0.0
            ),
            "positive_predicted_directed_edges": sum(len(v) for v in pred_edges.values()),
            "raw_stage7_relation_counts": raw_relation_counts,
        },
        "object_score_metadata": score_metadata,
        "object": object_results,
        "predicate": pred_metrics,
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "object_recall_at_1",
                "object_recall_at_5",
                "object_mean_recall_at_1",
                "object_mean_recall_at_5",
                "predicate_recall_at_1",
                "predicate_mean_recall_at_1",
            ]
        )
        for method, metrics in object_results.items():
            writer.writerow(
                [
                    method,
                    metrics["recall_at_1"],
                    metrics["recall_at_5"],
                    metrics["mean_recall_at_1"],
                    metrics["mean_recall_at_5"],
                    pred_metrics["recall_at_1"],
                    pred_metrics["mean_recall_at_1"],
                ]
            )

    with (args.output_dir / "gt_matches.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gt_index",
                "gt_object_id",
                "gt_class",
                "pred_original_id",
                "matched_points",
                *[f"{method}_rank_1based" for method in object_results],
            ]
        )
        for gt_index, (object_id, label) in enumerate(zip(gt_ids, gt_labels)):
            writer.writerow(
                [
                    gt_index,
                    object_id,
                    label,
                    int(gt_to_pred[gt_index]),
                    int(matched_points[gt_index]),
                    *[
                        "" if object_ranks[method][gt_index] is None else object_ranks[method][gt_index] + 1
                        for method in object_results
                    ],
                ]
            )

    with (args.output_dir / "predicate_matches.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gt_relation_index",
                "subject_object_id",
                "object_object_id",
                "predicate",
                "subject_pred_original_id",
                "object_pred_original_id",
                "hit_at_1",
            ]
        )
        id_to_gt = {object_id: index for index, object_id in enumerate(gt_ids)}
        for i, (subject, object_, predicate) in enumerate(gt_relations):
            writer.writerow(
                [
                    i,
                    subject,
                    object_,
                    predicate,
                    int(gt_to_pred[id_to_gt[subject]]),
                    int(gt_to_pred[id_to_gt[object_]]),
                    int(predicate_hits[i]),
                ]
            )

    print(f"scope={args.scene} gt_objects={len(gt_ids)} geometry_matched={np.sum(gt_to_pred >= 0)}")
    for method, metrics in object_results.items():
        print(
            f"{method}: Object R@1={pct(metrics['recall_at_1'])} "
            f"R@5={pct(metrics['recall_at_5'])} "
            f"mR@1={pct(metrics['mean_recall_at_1'])} "
            f"mR@5={pct(metrics['mean_recall_at_5'])}"
        )
    print(
        f"current_stage7: Predicate R@1={pct(pred_metrics['recall_at_1'])} "
        f"mR@1={pct(pred_metrics['mean_recall_at_1'])}"
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
