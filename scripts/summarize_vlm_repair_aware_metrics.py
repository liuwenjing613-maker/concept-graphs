#!/usr/bin/env python3
"""Summarize baseline/repaired metrics without new human annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "miou",
    "mrecall",
    "mprecision",
    "mf1score",
    "fmiou",
    "point_accuracy",
)
OBJECT_METHODS = ("main_native_clip_ft", "map_class_name")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_scene_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        scene, separator, raw_path = value.partition("=")
        if not separator or not scene or not raw_path:
            raise ValueError(f"expected SCENE=PATH, received {value!r}")
        result[scene] = Path(raw_path).expanduser().resolve()
    return result


def semseg_index(payload: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (str(row["score_source"]), int(row["n_exclude"]), str(row["scope"])): row
        for row in payload["summary_rows_percent"]
    }


def summarize_semseg(
    baseline: dict[str, Any], repaired: dict[str, Any], aggregate_label: str
) -> dict[str, Any]:
    before = semseg_index(baseline)
    after = semseg_index(repaired)
    comparisons = []
    for source in ("native_clip_ft", "map_class_name"):
        for n_exclude in (1, 4, 6):
            key = (source, n_exclude, aggregate_label)
            baseline_row = before[key]
            repaired_row = after[key]
            comparisons.append(
                {
                    "score_source": source,
                    "n_exclude": n_exclude,
                    "scope": aggregate_label,
                    "baseline_percent": {
                        metric: baseline_row[metric] for metric in METRIC_KEYS
                    },
                    "repaired_percent": {
                        metric: repaired_row[metric] for metric in METRIC_KEYS
                    },
                    "delta_percentage_points": {
                        metric: repaired_row[metric] - baseline_row[metric]
                        for metric in METRIC_KEYS
                    },
                }
            )

    per_class = {}
    for source in ("native_clip_ft", "map_class_name"):
        baseline_detail = baseline["details_fraction"][source]["6"][aggregate_label]
        repaired_detail = repaired["details_fraction"][source]["6"][aggregate_label]
        if baseline_detail["class_names"] != repaired_detail["class_names"]:
            raise ValueError(f"semantic class order changed for {source}")
        deltas = []
        for index, label in enumerate(baseline_detail["class_names"]):
            deltas.append(
                {
                    "class_name": label,
                    "baseline_iou_percent": 100.0 * baseline_detail["iou"][index],
                    "repaired_iou_percent": 100.0 * repaired_detail["iou"][index],
                    "delta_iou_percentage_points": 100.0
                    * (
                        repaired_detail["iou"][index]
                        - baseline_detail["iou"][index]
                    ),
                }
            )
        per_class[source] = sorted(
            deltas, key=lambda row: abs(row["delta_iou_percentage_points"]), reverse=True
        )
    return {"aggregate_comparison": comparisons, "n_exclude_6_per_class": per_class}


def pool_object_method(
    payloads: dict[str, dict[str, Any]], method: str
) -> dict[str, Any]:
    count = 0
    matched = 0
    hits_at_1 = 0
    hits_at_5 = 0
    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "hits_at_1": 0, "hits_at_5": 0}
    )
    for payload in payloads.values():
        metrics = payload["object"][method]
        count += int(metrics["count"])
        matched += int(metrics["matched_geometry"])
        hits_at_1 += int(metrics["hits_at_1"])
        hits_at_5 += int(metrics["hits_at_5"])
        for label, values in metrics["per_class"].items():
            for key in ("count", "hits_at_1", "hits_at_5"):
                per_class[label][key] += int(values[key])
    per_class_result = {
        label: {
            **values,
            "recall_at_1": values["hits_at_1"] / values["count"],
            "recall_at_5": values["hits_at_5"] / values["count"],
        }
        for label, values in sorted(per_class.items())
    }
    return {
        "gt_objects": count,
        "geometry_matched": matched,
        "geometry_coverage": matched / count,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "recall_at_1": hits_at_1 / count,
        "recall_at_5": hits_at_5 / count,
        "mean_recall_at_1": sum(
            values["recall_at_1"] for values in per_class_result.values()
        )
        / len(per_class_result),
        "mean_recall_at_5": sum(
            values["recall_at_5"] for values in per_class_result.values()
        )
        / len(per_class_result),
        "classes_present": len(per_class_result),
        "per_class": per_class_result,
    }


def pool_geometry(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gt_objects = sum(int(p["integrity"]["gt_objects"]) for p in payloads.values())
    predicted_objects = sum(
        int(p["integrity"]["predicted_objects"]) for p in payloads.values()
    )
    covered_gt = sum(
        int(p["integrity"]["geometry_matched_gt_objects"])
        for p in payloads.values()
    )
    valid_predicted = sum(
        int(p["integrity"]["geometry_valid_predicted_objects"])
        for p in payloads.values()
    )
    fragment_excess = sum(
        int(p["integrity"]["geometry_fragmentation_excess"])
        for p in payloads.values()
    )
    multiple = sum(
        int(p["integrity"]["geometry_gt_objects_with_multiple_fragments"])
        for p in payloads.values()
    )
    return {
        "gt_objects": gt_objects,
        "predicted_objects": predicted_objects,
        "prediction_minus_gt_count": predicted_objects - gt_objects,
        "covered_gt_objects": covered_gt,
        "geometry_coverage": covered_gt / gt_objects,
        "geometry_valid_predicted_objects": valid_predicted,
        "geometry_valid_prediction_rate": valid_predicted / predicted_objects,
        "geometry_one_to_one_coverage_per_prediction": covered_gt
        / predicted_objects,
        "gt_objects_with_multiple_fragments": multiple,
        "fragmentation_excess": fragment_excess,
        "mean_valid_fragments_per_covered_gt": valid_predicted / covered_gt,
        "scope_warning": (
            "ReplicaSSG does not annotate every map object; prediction-side rates "
            "are closed-scope diagnostics, not unrestricted real-world precision."
        ),
    }


def read_ranks(result_path: Path, method: str) -> dict[tuple[str, str], int | None]:
    column = f"{method}_rank_1based"
    with (result_path.parent / "gt_matches.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in rows[0]:
        raise ValueError(f"missing {column} beside {result_path}")
    return {
        (str(row["gt_object_id"]), str(row["gt_class"])): (
            int(row[column]) if row[column] else None
        )
        for row in rows
    }


def rank_delta(
    baseline_paths: dict[str, Path], repaired_paths: dict[str, Path], method: str
) -> dict[str, Any]:
    before: dict[tuple[str, str, str], int | None] = {}
    after: dict[tuple[str, str, str], int | None] = {}
    for scene, path in baseline_paths.items():
        before.update(
            {(scene, *key): value for key, value in read_ranks(path, method).items()}
        )
    for scene, path in repaired_paths.items():
        after.update(
            {(scene, *key): value for key, value in read_ranks(path, method).items()}
        )
    if before.keys() != after.keys():
        raise ValueError(f"GT rank keys changed for {method}")

    improved = worsened = same = 0
    top1_gained = top1_lost = top5_gained = top5_lost = 0
    reciprocal_before = reciprocal_after = 0.0
    for key in before:
        before_rank = before[key]
        after_rank = after[key]
        before_rr = 1.0 / before_rank if before_rank else 0.0
        after_rr = 1.0 / after_rank if after_rank else 0.0
        reciprocal_before += before_rr
        reciprocal_after += after_rr
        if after_rr > before_rr:
            improved += 1
        elif after_rr < before_rr:
            worsened += 1
        else:
            same += 1
        before_top1 = before_rank == 1
        after_top1 = after_rank == 1
        before_top5 = before_rank is not None and before_rank <= 5
        after_top5 = after_rank is not None and after_rank <= 5
        top1_gained += int(after_top1 and not before_top1)
        top1_lost += int(before_top1 and not after_top1)
        top5_gained += int(after_top5 and not before_top5)
        top5_lost += int(before_top5 and not after_top5)
    count = len(before)
    return {
        "gt_objects": count,
        "rank_improved": improved,
        "rank_worsened": worsened,
        "rank_unchanged": same,
        "top1_gained": top1_gained,
        "top1_lost": top1_lost,
        "top5_gained": top5_gained,
        "top5_lost": top5_lost,
        "mean_reciprocal_rank_baseline": reciprocal_before / count,
        "mean_reciprocal_rank_repaired": reciprocal_after / count,
        "mean_reciprocal_rank_delta": (reciprocal_after - reciprocal_before)
        / count,
    }


def object_comparison(
    baseline_paths: dict[str, Path], repaired_paths: dict[str, Path]
) -> dict[str, Any]:
    baseline = {scene: load_json(path) for scene, path in baseline_paths.items()}
    repaired = {scene: load_json(path) for scene, path in repaired_paths.items()}
    methods = {}
    for method in OBJECT_METHODS:
        before = pool_object_method(baseline, method)
        after = pool_object_method(repaired, method)
        methods[method] = {
            "baseline": before,
            "repaired": after,
            "delta_percentage_points": {
                metric: 100.0 * (after[metric] - before[metric])
                for metric in (
                    "geometry_coverage",
                    "recall_at_1",
                    "recall_at_5",
                    "mean_recall_at_1",
                    "mean_recall_at_5",
                )
            },
            "gt_rank_changes": rank_delta(
                baseline_paths, repaired_paths, method
            ),
        }
    before_geometry = pool_geometry(baseline)
    after_geometry = pool_geometry(repaired)
    return {
        "methods": methods,
        "geometry_and_fragmentation": {
            "baseline": before_geometry,
            "repaired": after_geometry,
            "delta": {
                key: after_geometry[key] - before_geometry[key]
                for key in (
                    "predicted_objects",
                    "prediction_minus_gt_count",
                    "covered_gt_objects",
                    "geometry_coverage",
                    "geometry_valid_predicted_objects",
                    "geometry_valid_prediction_rate",
                    "geometry_one_to_one_coverage_per_prediction",
                    "gt_objects_with_multiple_fragments",
                    "fragmentation_excess",
                    "mean_valid_fragments_per_covered_gt",
                )
            },
        },
    }


def repair_counts(paths: list[Path]) -> dict[str, Any]:
    action_counts: dict[str, int] = defaultdict(int)
    applied = 0
    for path in paths:
        manifest = load_json(path)
        for report in manifest["reports"]:
            if report.get("apply_status") != "APPLIED":
                continue
            applied += 1
            action_counts[str(report["action"])] += 1
    return {"applied": applied, "by_action": dict(sorted(action_counts.items()))}


def render_markdown(payload: dict[str, Any]) -> str:
    semantic_rows = {
        row["score_source"]: row
        for row in payload["semantic"]["aggregate_comparison"]
        if row["n_exclude"] == 6
    }
    object_methods = payload["object"]["methods"]
    geometry = payload["object"]["geometry_and_fragmentation"]
    rank_changes = object_methods["map_class_name"]["gt_rank_changes"]
    per_class = payload["semantic"]["n_exclude_6_per_class"]["map_class_name"]
    changed_classes = [
        row for row in per_class if abs(row["delta_iou_percentage_points"]) > 1e-12
    ]
    lines = [
        "# ali-my-VLM 修复感知指标（room0 + office0）",
        "",
        "> 未使用任何新增人工标注。`native_clip_ft` 是标准视觉特征对照；",
        "> `map_class_name` 直接评估地图保存标签，包括 VLM 修复结果。",
        "",
        "## 主语义口径（n_exclude=6）",
        "",
        "| 轨道 | mIoU 前→后 | Δ | mRecall Δ | mPrecision Δ | mF1 Δ | fwIoU Δ | 点准确率 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in ("native_clip_ft", "map_class_name"):
        row = semantic_rows[source]
        before = row["baseline_percent"]
        after = row["repaired_percent"]
        delta = row["delta_percentage_points"]
        lines.append(
            f"| `{source}` | {before['miou']:.2f}% → {after['miou']:.2f}% | "
            f"{delta['miou']:+.2f} pp | {delta['mrecall']:+.2f} pp | "
            f"{delta['mprecision']:+.2f} pp | {delta['mf1score']:+.2f} pp | "
            f"{delta['fmiou']:+.2f} pp | {delta['point_accuracy']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## ReplicaSSG 对象分类（两场景 pooled）",
            "",
            "| 轨道 | R@1 前→后 | Δ | R@5 Δ | mR@1 Δ | mR@5 Δ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source in ("main_native_clip_ft", "map_class_name"):
        result = object_methods[source]
        before = result["baseline"]
        after = result["repaired"]
        delta = result["delta_percentage_points"]
        lines.append(
            f"| `{source}` | {100*before['recall_at_1']:.2f}% → "
            f"{100*after['recall_at_1']:.2f}% | {delta['recall_at_1']:+.2f} pp | "
            f"{delta['recall_at_5']:+.2f} pp | "
            f"{delta['mean_recall_at_1']:+.2f} pp | "
            f"{delta['mean_recall_at_5']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "标签感知的 74 个 GT 排名变化："
            f"改善 {rank_changes['rank_improved']}、恶化 {rank_changes['rank_worsened']}、"
            f"不变 {rank_changes['rank_unchanged']}；Top-1 新增 "
            f"{rank_changes['top1_gained']}、丢失 {rank_changes['top1_lost']}。",
            "",
            "## 几何与合并诊断",
            "",
            f"- 预测对象：{geometry['baseline']['predicted_objects']} → "
            f"{geometry['repaired']['predicted_objects']}。",
            f"- GT 几何覆盖：{100*geometry['baseline']['geometry_coverage']:.2f}% → "
            f"{100*geometry['repaired']['geometry_coverage']:.2f}%（不变）。",
            f"- 闭集几何有效预测率："
            f"{100*geometry['baseline']['geometry_valid_prediction_rate']:.2f}% → "
            f"{100*geometry['repaired']['geometry_valid_prediction_rate']:.2f}%。",
            f"- 碎片化 excess：{geometry['baseline']['fragmentation_excess']} → "
            f"{geometry['repaired']['fragmentation_excess']}（不变）。",
            "",
            "## n_exclude=6 类别 IoU 变化（按绝对变化排序）",
            "",
            "| 类别 | 前 | 后 | Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in changed_classes:
        lines.append(
            f"| {row['class_name']} | {row['baseline_iou_percent']:.2f}% | "
            f"{row['repaired_iou_percent']:.2f}% | "
            f"{row['delta_iou_percentage_points']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- `map_class_name` 是通过同一 OpenCLIP 文本词表完成的标签到闭集类别映射，",
            "  衡量标签修复的下游可用性，不等同于原生视觉表征提升。",
            "- ReplicaSSG 未标注地图中的全部对象，预测侧比率只能作为闭集诊断，不能写成真实世界精度。",
            "- 独立修复正确率、beneficial/neutral/harmful 人工裁决和未触发对象召回需要新增标注，本次未测。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-semseg", type=Path, required=True)
    parser.add_argument("--repaired-semseg", type=Path, required=True)
    parser.add_argument("--baseline-object", action="append", default=[], required=True)
    parser.add_argument("--repaired-object", action="append", default=[], required=True)
    parser.add_argument("--repair-manifest", action="append", type=Path, default=[])
    parser.add_argument("--aggregate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    baseline_object_paths = parse_scene_paths(args.baseline_object)
    repaired_object_paths = parse_scene_paths(args.repaired_object)
    if baseline_object_paths.keys() != repaired_object_paths.keys():
        raise SystemExit("baseline/repaired object scenes differ")
    payload = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "ali-my-VLM-only-repair-v1-repair-aware-evaluation",
        "new_manual_annotations_used": False,
        "repair_actions": repair_counts(args.repair_manifest),
        "semantic": summarize_semseg(
            load_json(args.baseline_semseg),
            load_json(args.repaired_semseg),
            args.aggregate_label,
        ),
        "object": object_comparison(
            baseline_object_paths, repaired_object_paths
        ),
        "interpretation": {
            "native_clip_ft": "standard visual-feature benchmark control",
            "map_class_name": (
                "repair-aware score from the saved object label through the same "
                "OpenCLIP text vocabulary"
            ),
            "manual_only_metrics_omitted": [
                "independent repair correctness",
                "human beneficial/neutral/harmful adjudication",
                "unflagged-object error recall",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
