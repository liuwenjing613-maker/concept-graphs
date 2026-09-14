from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_relation_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load relation pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def sorted_triples(values: Iterable[tuple[int, int, str]]) -> list[list[Any]]:
    return [[source, target, predicate] for source, target, predicate in sorted(values)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an aggregated make_edges graph with ReplicaSSG matching"
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--clean-state", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--relation-pipeline", type=Path, required=True)
    parser.add_argument("--overlap-threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    relation_pipeline = load_module(args.relation_pipeline.resolve())
    if args.scene not in relation_pipeline.SCENE_ALIASES:
        raise ValueError(f"unsupported scene alias: {args.scene}")
    objects, _payload = relation_pipeline.load_map(args.map.resolve())
    object_id_to_index = {str(item["id"]): index for index, item in enumerate(objects)}
    if len(object_id_to_index) != len(objects):
        raise ValueError("map contains duplicate object UUIDs")

    with args.clean_state.open(encoding="utf-8") as handle:
        clean_state = json.load(handle)
    raw_edges = list(clean_state.get("edges") or ())
    missing_endpoints = sorted(
        {
            str(edge[key])
            for edge in raw_edges
            for key in ("source_entity_uid", "target_entity_uid")
            if str(edge[key]) not in object_id_to_index
        }
    )
    if missing_endpoints:
        raise ValueError(
            f"relation endpoints do not bind to formal map UUIDs: {missing_endpoints[:5]}"
        )

    predicate_map = {"on top of": "on", "under": "under"}
    predicted: set[tuple[int, int, str]] = set()
    predicted_support = {}
    for edge in raw_edges:
        relation = str(edge["relation"])
        if relation not in predicate_map:
            raise ValueError(f"unsupported baseline relation type: {relation}")
        triple = (
            object_id_to_index[str(edge["source_entity_uid"])],
            object_id_to_index[str(edge["target_entity_uid"])],
            predicate_map[relation],
        )
        predicted.add(triple)
        predicted_support[triple] = int(edge.get("num_detections", 1))
    if len(predicted) != len(raw_edges):
        raise ValueError("aggregated relation graph contains duplicate semantic triples")

    gt_scene = relation_pipeline.SCENE_ALIASES[args.scene]
    gt_ids, gt_relations = relation_pipeline.load_ground_truth(
        gt_scene, args.annotations_dir.resolve()
    )
    gt_ply = (
        args.dataset_root.resolve()
        / "data"
        / gt_scene
        / "labels.instances.annotated.v2.ply"
    )
    gt_to_pred, matched_points, overlap = relation_pipeline.match_instances(
        gt_ply,
        gt_ids,
        objects,
        args.overlap_threshold,
    )
    gt_index = {object_id: index for index, object_id in enumerate(gt_ids)}
    mapped_gt = set()
    endpoint_matched_relations = 0
    for source_id, target_id, predicate in gt_relations:
        source = int(gt_to_pred[gt_index[source_id]])
        target = int(gt_to_pred[gt_index[target_id]])
        if source >= 0 and target >= 0:
            endpoint_matched_relations += 1
            mapped_gt.add((source, target, predicate))

    true_positives = predicted & mapped_gt
    false_positives = predicted - mapped_gt
    false_negatives = mapped_gt - predicted
    precision = ratio(len(true_positives), len(predicted))
    recall_mapped = ratio(len(true_positives), len(mapped_gt))

    support_thresholds = [1, 2, 3, 5, 10, 20, 50, 100]
    support_sweep = []
    for threshold in support_thresholds:
        thresholded = {
            triple for triple, support in predicted_support.items() if support >= threshold
        }
        threshold_hits = thresholded & mapped_gt
        threshold_precision = ratio(len(threshold_hits), len(thresholded))
        threshold_recall = ratio(len(threshold_hits), len(mapped_gt))
        support_sweep.append(
            {
                "minimum_support": threshold,
                "predicted_edges": len(thresholded),
                "true_positive_edges": len(threshold_hits),
                "false_positive_edges": len(thresholded - mapped_gt),
                "false_negative_mapped_gt_edges": len(mapped_gt - thresholded),
                "closed_world_precision": threshold_precision,
                "closed_world_recall_on_mapped_gt": threshold_recall,
                "closed_world_f1": f1_score(threshold_precision, threshold_recall),
            }
        )
    diagnostic_best = max(
        support_sweep,
        key=lambda row: (
            -1.0 if row["closed_world_f1"] is None else row["closed_world_f1"],
            -row["minimum_support"],
        ),
    )

    predicates = sorted({item[2] for item in predicted | mapped_gt})
    per_predicate = {}
    for predicate in predicates:
        pred_subset = {item for item in predicted if item[2] == predicate}
        gt_subset = {item for item in mapped_gt if item[2] == predicate}
        hits = pred_subset & gt_subset
        pred_precision = ratio(len(hits), len(pred_subset))
        gt_recall = ratio(len(hits), len(gt_subset))
        per_predicate[predicate] = {
            "predicted_edges": len(pred_subset),
            "mapped_gt_edges": len(gt_subset),
            "true_positive_edges": len(hits),
            "closed_world_precision": pred_precision,
            "recall_on_mapped_gt": gt_recall,
            "f1": f1_score(pred_precision, gt_recall),
        }

    result = {
        "schema_version": "0.1.0",
        "status": "PASS",
        "scene": args.scene,
        "replicassg_scene": gt_scene,
        "protocol": {
            "geometry_match": (
                "FROSS ReplicaSSG one-way KD-tree; 0.1m, >=50% best overlap, "
                "second/best <=0.75"
            ),
            "overlap_threshold": args.overlap_threshold,
            "predicate_directional": True,
            "predicate_mapping": predicate_map,
            "scope": (
                "strict closed-world evaluation against sparse ReplicaSSG labels; "
                "unlabelled but plausible predictions count as false positives"
            ),
            "support_counts_used_for_accuracy": False,
            "support_threshold_sweep_role": (
                "post-hoc diagnostic only; the primary result uses the unchanged "
                "baseline graph and no added support threshold"
            ),
        },
        "inputs": {
            "map": str(args.map.resolve()),
            "map_sha256": sha256_file(args.map.resolve()),
            "clean_state": str(args.clean_state.resolve()),
            "clean_state_sha256": sha256_file(args.clean_state.resolve()),
            "relation_pipeline": str(args.relation_pipeline.resolve()),
            "relation_pipeline_sha256": sha256_file(args.relation_pipeline.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "annotations_dir": str(args.annotations_dir.resolve()),
        },
        "object_matching": {
            "predicted_map_objects": len(objects),
            "gt_objects": len(gt_ids),
            "matched_gt_objects": sum(int(value) >= 0 for value in gt_to_pred),
            "matched_point_counts_nonzero": sum(int(value) > 0 for value in matched_points),
            "overlap_rows": len(overlap),
        },
        "relations": {
            "gt_relations_all": len(gt_relations),
            "endpoint_matched_gt_relations": endpoint_matched_relations,
            "mapped_gt_unique_edges": len(mapped_gt),
            "predicted_directed_edges": len(predicted),
            "predicted_total_support": sum(predicted_support.values()),
            "true_positive_edges": len(true_positives),
            "false_positive_edges": len(false_positives),
            "false_negative_mapped_gt_edges": len(false_negatives),
            "closed_world_precision": precision,
            "closed_world_recall_on_mapped_gt": recall_mapped,
            "closed_world_f1": f1_score(precision, recall_mapped),
            "recall_on_all_gt": ratio(len(true_positives), len(gt_relations)),
            "predicted_relation_counts": dict(Counter(item[2] for item in predicted)),
            "mapped_gt_relation_counts": dict(Counter(item[2] for item in mapped_gt)),
            "per_predicate": per_predicate,
            "support_threshold_sweep": support_sweep,
            "diagnostic_best_support_threshold": diagnostic_best,
        },
        "audit": {
            "missing_map_endpoint_count": 0,
            "duplicate_predicted_triples": 0,
            "predicted_triples": sorted_triples(predicted),
            "mapped_gt_triples": sorted_triples(mapped_gt),
            "true_positive_triples": sorted_triples(true_positives),
            "false_positive_triples": sorted_triples(false_positives),
            "false_negative_triples": sorted_triples(false_negatives),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
