#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCENES = ("room_0", "office_0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the two-scene VLM-repaired evaluation against ali-my."
    )
    parser.add_argument("--repaired-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _rank_hit(raw: str, k: int) -> bool:
    return bool(raw) and int(raw) <= k


def _pool_objects(root: Path, subdir: str) -> dict[str, Any]:
    per_scene: list[dict[str, Any]] = []
    total = predicted = geometry = hits_1 = hits_5 = 0
    class_count: dict[str, int] = defaultdict(int)
    class_hits_1: dict[str, int] = defaultdict(int)
    class_hits_5: dict[str, int] = defaultdict(int)
    gt_relations = positive_edges = 0
    for scene in SCENES:
        scene_root = root / subdir / scene
        result = json.loads((scene_root / "results.json").read_text(encoding="utf-8"))
        native = result["object"]["main_native_clip_ft"]
        integrity = result["integrity"]
        row = {
            "scene": scene,
            "gt_objects": int(integrity["gt_objects"]),
            "predicted_objects": int(integrity["predicted_objects"]),
            "geometry_matched": int(integrity["geometry_matched_gt_objects"]),
            "geometry_coverage": int(integrity["geometry_matched_gt_objects"])
            / int(integrity["gt_objects"]),
            "recall_at_1": float(native["recall_at_1"]),
            "recall_at_5": float(native["recall_at_5"]),
            "mean_recall_at_1": float(native["mean_recall_at_1"]),
            "mean_recall_at_5": float(native["mean_recall_at_5"]),
        }
        per_scene.append(row)
        predicted += row["predicted_objects"]
        gt_relations += int(integrity["gt_relations"])
        positive_edges += int(integrity["positive_predicted_directed_edges"])
        with (scene_root / "gt_matches.csv").open(newline="", encoding="utf-8") as handle:
            for match in csv.DictReader(handle):
                label = match["gt_class"]
                hit_1 = _rank_hit(match["main_native_clip_ft_rank_1based"], 1)
                hit_5 = _rank_hit(match["main_native_clip_ft_rank_1based"], 5)
                total += 1
                geometry += int(match["pred_original_id"]) >= 0
                hits_1 += hit_1
                hits_5 += hit_5
                class_count[label] += 1
                class_hits_1[label] += hit_1
                class_hits_5[label] += hit_5
    pooled = {
        "scene": "all_2_scenes",
        "gt_objects": total,
        "predicted_objects": predicted,
        "geometry_matched": geometry,
        "geometry_coverage": geometry / total,
        "recall_at_1": hits_1 / total,
        "recall_at_5": hits_5 / total,
        "mean_recall_at_1": sum(
            class_hits_1[label] / count for label, count in class_count.items()
        )
        / len(class_count),
        "mean_recall_at_5": sum(
            class_hits_5[label] / count for label, count in class_count.items()
        )
        / len(class_count),
        "classes_present": len(class_count),
    }
    return {
        "per_scene_fraction": per_scene,
        "pooled_fraction": pooled,
        "relation_structure": {
            "gt_relations": gt_relations,
            "positive_predicted_directed_edges": positive_edges,
            "valid_model_metric": False,
            "reason": "make_edges=false; zero is structural, not predicate performance",
        },
    }


def _semseg_rows(path: Path) -> list[dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return list(result["summary_rows_percent"])


def main() -> None:
    args = parse_args()
    repaired_objects = _pool_objects(args.repaired_root, "replicassg_cpu")
    baseline_objects = _pool_objects(args.baseline_root, "replicassg")
    repaired_semseg = _semseg_rows(args.repaired_root / "semseg_cpu/semseg_results.json")
    baseline_semseg = _semseg_rows(args.baseline_root / "semseg_cpu/semseg_results.json")
    semseg_metrics = ("miou", "mrecall", "mprecision", "mf1score", "fmiou", "point_accuracy")
    repaired_by_key = {
        (row["scope"] if row["scope"] in {"room0", "office0"} else "all_2_scenes", row["n_exclude"]): row
        for row in repaired_semseg
    }
    baseline_by_key = {
        (row["scope"] if row["scope"] in {"room0", "office0"} else "all_2_scenes", row["n_exclude"]): row
        for row in baseline_semseg
    }
    semseg_comparison = []
    for key in sorted(repaired_by_key, key=lambda value: (value[1], value[0])):
        repaired = repaired_by_key[key]
        baseline = baseline_by_key[key]
        semseg_comparison.append(
            {
                "scope": key[0],
                "n_exclude": key[1],
                "repaired_percent": {metric: repaired[metric] for metric in semseg_metrics},
                "baseline_percent": {metric: baseline[metric] for metric in semseg_metrics},
                "delta_percentage_points": {
                    metric: repaired[metric] - baseline[metric] for metric in semseg_metrics
                },
            }
        )
    object_metrics = (
        "geometry_coverage",
        "recall_at_1",
        "recall_at_5",
        "mean_recall_at_1",
        "mean_recall_at_5",
    )
    repaired_pooled = repaired_objects["pooled_fraction"]
    baseline_pooled = baseline_objects["pooled_fraction"]
    object_delta = {
        metric: 100.0 * (repaired_pooled[metric] - baseline_pooled[metric])
        for metric in object_metrics
    }
    payload = {
        "format_version": 1,
        "scope": "Replica room0 + office0, ali-my frozen 200-frame maps",
        "repaired_root": str(args.repaired_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "semseg": {
            "repaired_rows_percent": repaired_semseg,
            "baseline_rows_percent": baseline_semseg,
            "comparison": semseg_comparison,
        },
        "objects": {
            "repaired": repaired_objects,
            "baseline": baseline_objects,
            "pooled_delta_percentage_points": object_delta,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = []
    for item in semseg_comparison:
        for metric in semseg_metrics:
            rows.append(
                {
                    "family": "semseg",
                    "scope": item["scope"],
                    "protocol": f'n_exclude={item["n_exclude"]}',
                    "metric": metric,
                    "repaired_percent": item["repaired_percent"][metric],
                    "baseline_percent": item["baseline_percent"][metric],
                    "delta_percentage_points": item["delta_percentage_points"][metric],
                }
            )
    for metric in object_metrics:
        rows.append(
            {
                "family": "object",
                "scope": "all_2_scenes",
                "protocol": "ReplicaSSG main_native_clip_ft",
                "metric": metric,
                "repaired_percent": 100.0 * repaired_pooled[metric],
                "baseline_percent": 100.0 * baseline_pooled[metric],
                "delta_percentage_points": object_delta[metric],
            }
        )
    with (args.output_dir / "evaluation_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"object_pooled": repaired_pooled, "object_delta_pp": object_delta}, indent=2))


if __name__ == "__main__":
    main()
