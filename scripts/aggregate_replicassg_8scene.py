#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCENES = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rank_hit(raw: str, k: int) -> bool:
    return bool(raw) and int(raw) <= k


def load_method(root: Path, method: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_scene: list[dict[str, Any]] = []
    pooled_count = 0
    pooled_predicted = 0
    pooled_geometry = 0
    pooled_hits_1 = 0
    pooled_hits_5 = 0
    per_class_count: dict[str, int] = defaultdict(int)
    per_class_hits_1: dict[str, int] = defaultdict(int)
    per_class_hits_5: dict[str, int] = defaultdict(int)

    for scene in SCENES:
        scene_dir = root / f"replicassg_{method}_8scene" / scene
        result = json.loads((scene_dir / "results.json").read_text(encoding="utf-8"))
        native = result["object"]["main_native_clip_ft"]
        integrity = result["integrity"]
        row = {
            "method": method,
            "scene": scene,
            "gt_objects": int(integrity["gt_objects"]),
            "predicted_objects": int(integrity["predicted_objects"]),
            "geometry_matched": int(integrity["geometry_matched_gt_objects"]),
            "geometry_coverage": int(integrity["geometry_matched_gt_objects"]) / int(integrity["gt_objects"]),
            "recall_at_1": float(native["recall_at_1"]),
            "recall_at_5": float(native["recall_at_5"]),
            "mean_recall_at_1": float(native["mean_recall_at_1"]),
            "mean_recall_at_5": float(native["mean_recall_at_5"]),
            "classes_present": int(native["classes_present"]),
        }
        per_scene.append(row)
        pooled_predicted += row["predicted_objects"]

        with (scene_dir / "gt_matches.csv").open(newline="", encoding="utf-8") as handle:
            for match in csv.DictReader(handle):
                label = match["gt_class"]
                hit_1 = rank_hit(match["main_native_clip_ft_rank_1based"], 1)
                hit_5 = rank_hit(match["main_native_clip_ft_rank_1based"], 5)
                pooled_count += 1
                pooled_geometry += int(match["pred_original_id"]) >= 0
                pooled_hits_1 += hit_1
                pooled_hits_5 += hit_5
                per_class_count[label] += 1
                per_class_hits_1[label] += hit_1
                per_class_hits_5[label] += hit_5

    pooled = {
        "method": method,
        "scene": "all_8_scenes",
        "gt_objects": pooled_count,
        "predicted_objects": pooled_predicted,
        "geometry_matched": pooled_geometry,
        "geometry_coverage": pooled_geometry / pooled_count,
        "recall_at_1": pooled_hits_1 / pooled_count,
        "recall_at_5": pooled_hits_5 / pooled_count,
        "mean_recall_at_1": sum(per_class_hits_1[label] / count for label, count in per_class_count.items()) / len(per_class_count),
        "mean_recall_at_5": sum(per_class_hits_5[label] / count for label, count in per_class_count.items()) / len(per_class_count),
        "classes_present": len(per_class_count),
        "hits_at_1": pooled_hits_1,
        "hits_at_5": pooled_hits_5,
    }
    return per_scene, pooled


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pooled_by_method: dict[str, dict[str, Any]] = {}
    for method in ("ali", "main"):
        per_scene, pooled = load_method(args.root, method)
        rows.extend(per_scene)
        rows.append(pooled)
        pooled_by_method[method] = pooled

    by_key = {(row["method"], row["scene"]): row for row in rows}
    comparisons = []
    for scene in [*SCENES, "all_8_scenes"]:
        ali = by_key[("ali", scene)]
        main_row = by_key[("main", scene)]
        comparisons.append({
            "scene": scene,
            "delta_geometry_coverage_pp": 100 * (ali["geometry_coverage"] - main_row["geometry_coverage"]),
            "delta_recall_at_1_pp": 100 * (ali["recall_at_1"] - main_row["recall_at_1"]),
            "delta_recall_at_5_pp": 100 * (ali["recall_at_5"] - main_row["recall_at_5"]),
            "delta_mean_recall_at_1_pp": 100 * (ali["mean_recall_at_1"] - main_row["mean_recall_at_1"]),
            "delta_mean_recall_at_5_pp": 100 * (ali["mean_recall_at_5"] - main_row["mean_recall_at_5"]),
        })

    payload = {
        "format_version": 1,
        "scope": "ReplicaSSG 8 scenes, 400 frames, stride 5",
        "valid_method": "main_native_clip_ft",
        "invalid_methods": {
            "predicate": "empty relation inputs; zero is structural and not a model score",
            "scene_graph_object_tag": "no scene graph input",
            "llava_llm_possible_tags_max": "no possible-tag input",
        },
        "rows_fraction": rows,
        "pooled_fraction": pooled_by_method,
        "ali_minus_main_percentage_points": comparisons,
    }
    (args.output_dir / "replicassg_8scene_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    row_fields = [
        "method", "scene", "gt_objects", "predicted_objects", "geometry_matched",
        "geometry_coverage", "recall_at_1", "recall_at_5", "mean_recall_at_1",
        "mean_recall_at_5", "classes_present",
    ]
    with (args.output_dir / "replicassg_8scene_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "replicassg_8scene_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    print(json.dumps({"pooled": pooled_by_method, "comparison": comparisons[-1]}, indent=2))


if __name__ == "__main__":
    main()
