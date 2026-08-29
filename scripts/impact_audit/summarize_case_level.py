#!/usr/bin/env python3
"""Compare frozen one-case Oracles with their scene/scale baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCALES = ("voxel0p025", "voxel0p05", "voxel0p10")
METRICS = (
    "object_precision_iou0p25",
    "object_recall_iou0p25",
    "object_f1_iou0p25",
    "mean_maximum_purity",
    "mean_maximum_coverage",
    "semantic_accuracy_hungarian_positive",
)
EPSILON = 1e-12


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def semantic_accuracy(path: Path) -> tuple[float | None, int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [row for row in rows if as_bool(row.get("semantic_eligible"))]
    correct = sum(as_bool(row.get("semantic_consistent")) for row in eligible)
    return (correct / len(eligible) if eligible else None, correct, len(eligible))


def flatten(report: dict[str, Any], semantic: tuple[float | None, int, int]) -> dict[str, Any]:
    threshold = report["object_metrics_by_iou_threshold"]["0.25"]
    purity = report["distributions"]["maximum_purity_per_prediction"]
    coverage = report["distributions"]["maximum_coverage_per_gt"]
    return {
        "predicted_objects": report["counts"]["predicted_objects"],
        "object_precision_iou0p25": threshold["precision"],
        "object_recall_iou0p25": threshold["recall"],
        "object_f1_iou0p25": threshold["f1"],
        "mean_maximum_purity": purity["mean"],
        "mean_maximum_coverage": coverage["mean"],
        "semantic_accuracy_hungarian_positive": semantic[0],
        "semantic_correct": semantic[1],
        "semantic_denominator": semantic[2],
    }


def classify(family: str, deltas: dict[str, float | None]) -> tuple[str, str]:
    def value(metric: str) -> float:
        raw = deltas[metric]
        return 0.0 if raw is None else raw

    if family == "semantic":
        structural = [abs(value(metric)) <= EPSILON for metric in METRICS[:-1]]
        semantic = value("semantic_accuracy_hungarian_positive")
        if semantic > EPSILON and all(structural):
            return "improved", "semantic accuracy improved with structure unchanged"
        if semantic < -EPSILON or not all(structural):
            return "degraded_or_confounded", "semantic accuracy degraded or structure changed"
        return "no_effect", "semantic accuracy unchanged"
    if family == "spurious":
        precision = value("object_precision_iou0p25")
        f1 = value("object_f1_iou0p25")
        if precision >= -EPSILON and f1 > EPSILON:
            return "improved", "F1 improved without precision loss"
        if precision < -EPSILON or f1 < -EPSILON:
            return "degraded", "precision or F1 degraded"
        return "no_effect", "precision and F1 unchanged"
    if family == "geometry":
        recall = value("object_recall_iou0p25")
        coverage = value("mean_maximum_coverage")
        if recall > EPSILON and coverage > EPSILON:
            return "improved", "recall and coverage both improved"
        if recall < -EPSILON or coverage < -EPSILON:
            return "degraded", "recall or coverage degraded"
        return "partial_or_no_effect", "did not jointly improve recall and coverage"
    if family == "association":
        f1 = value("object_f1_iou0p25")
        purity = value("mean_maximum_purity")
        if f1 > EPSILON and purity >= -EPSILON:
            return "improved", "F1 improved without mean-purity loss"
        if f1 < -EPSILON or purity < -EPSILON:
            return "degraded", "F1 or mean purity degraded"
        return "no_effect", "F1 and mean purity unchanged"
    return "not_scored", "control/no-op"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-cases", required=True, type=Path)
    parser.add_argument("--case-metrics-root", required=True, type=Path)
    parser.add_argument("--family-metrics-root", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = read_json(args.frozen_cases)
    rows = []
    for case in cases:
        scene = case["scene_id"]
        case_id = case["case_id"]
        family = case["family"]
        for scale in SCALES:
            baseline_dir = args.family_metrics_root / scene / "baseline" / scale
            case_dir = baseline_dir if family == "control" else args.case_metrics_root / case_id / scale
            baseline = flatten(
                read_json(baseline_dir / "audit_summary.json"),
                semantic_accuracy(baseline_dir / "hungarian_matches.csv"),
            )
            current = flatten(
                read_json(case_dir / "audit_summary.json"),
                semantic_accuracy(case_dir / "hungarian_matches.csv"),
            )
            deltas = {
                metric: None
                if baseline[metric] is None or current[metric] is None
                else float(current[metric]) - float(baseline[metric])
                for metric in METRICS
            }
            verdict, reason = classify(family, deltas)
            rows.append(
                {
                    "case_id": case_id,
                    "scene_id": scene,
                    "family": family,
                    "subtype": case.get("subtype"),
                    "predicted_label": case.get("predicted_label"),
                    "gt_label": case.get("gt_label"),
                    "scale": scale,
                    "verdict": verdict,
                    "verdict_reason": reason,
                    "baseline_predicted_objects": baseline["predicted_objects"],
                    "current_predicted_objects": current["predicted_objects"],
                    **{f"baseline_{metric}": baseline[metric] for metric in METRICS},
                    **{f"current_{metric}": current[metric] for metric in METRICS},
                    **{f"delta_{metric}": deltas[metric] for metric in METRICS},
                }
            )

    primary = [row for row in rows if row["scale"] == "voxel0p05"]
    primary_by_family: dict[str, Counter] = defaultdict(Counter)
    for row in primary:
        primary_by_family[row["family"]][row["verdict"]] += 1
    cross_scale = {}
    for case in cases:
        case_rows = [row for row in rows if row["case_id"] == case["case_id"]]
        cross_scale[case["case_id"]] = dict(Counter(row["verdict"] for row in case_rows))

    summary = {
        "schema_version": "1.0.0",
        "protocol": {
            "primary_scale": "voxel0p05",
            "selection_blinded_to_oracle_outcomes": True,
            "human_labels_used_as_truth": False,
            "verdict_rules": {
                "semantic": "semantic accuracy improves and all structural metrics remain unchanged",
                "spurious": "IoU@0.25 F1 improves without precision loss",
                "geometry": "IoU@0.25 recall and mean GT coverage both improve",
                "association": "IoU@0.25 F1 improves without mean-purity loss",
                "control": "exact baseline reuse/no-op",
            },
        },
        "primary_5cm": primary,
        "primary_verdict_counts_by_family": {
            family: dict(counts) for family, counts in sorted(primary_by_family.items())
        },
        "cross_scale_verdict_counts_by_case": cross_scale,
        "all_scene_case_scale_rows": rows,
        "run_manifest": read_json(args.run_manifest),
        "limitations": [
            "No strict S1 pure semantic-only case was available in the frozen objective pool.",
            "Controls are no-op or relative controls; no absolute clean control passed the objective criteria.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "primary_verdict_counts_by_family": summary["primary_verdict_counts_by_family"],
                "primary_5cm": primary,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
