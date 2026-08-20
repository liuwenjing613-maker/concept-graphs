#!/usr/bin/env python3
"""Compute Audit Validity Gate v1 metrics from completed human labels.

This script deliberately refuses incomplete or placeholder labels.  It keeps the
calibration and diagnostic cohorts separate, applies adjudicated labels as the
final authority, and writes the four metric files plus a human-readable decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


REQUIRED_LABEL_FIELDS = (
    "reviewer_id",
    "evidence_sufficient",
    "finding_correct",
    "root_stage_correct",
    "physical_interpretation",
    "downstream_harm",
    "harm_confidence",
    "repair_action",
    "repair_locality",
    "repair_confidence",
    "review_seconds",
)
LABEL_FIELDS = REQUIRED_LABEL_FIELDS + (
    "alternative_explanation",
    "notes",
)
ACTIONABLE_REPAIR_EXCLUSIONS = {"NONE", "UNKNOWN", "NEED_MORE_VIEW"}
ACTIONABLE_HARM_EXCLUSIONS = {"NONE", "UNKNOWN"}
LABEL_ENUMS = {
    "evidence_sufficient": {"YES", "NO", "PARTIAL"},
    "finding_correct": {"YES", "NO", "UNCERTAIN"},
    "root_stage_correct": {"YES", "NO", "UNCERTAIN", "NOT_APPLICABLE"},
    "downstream_harm": {
        "NONE",
        "LOCAL_WEIGHTING_BIAS",
        "WRONG_OBSERVATION_MEMBERSHIP",
        "FALSE_SPLIT_DUPLICATE_NODE",
        "FALSE_MERGE_IDENTITY_POLLUTION",
        "GEOMETRY_CORRUPTION",
        "RELATION_POLLUTION",
        "UNKNOWN",
    },
    "repair_action": {
        "NONE",
        "DROP_OBSERVATION",
        "REASSIGN_OBSERVATION",
        "MERGE_OBJECTS",
        "SPLIT_OBJECT",
        "RECOMPUTE_GEOMETRY",
        "DOWNWEIGHT_EVIDENCE",
        "NEED_MORE_VIEW",
        "UNKNOWN",
    },
    "repair_locality": {"LOCAL", "MULTI_OBJECT", "GLOBAL", "NOT_APPLICABLE"},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise ValueError(f"missing required file: {path}")
        return []
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    scene = row.get("scene_id")
    uid = row.get("case_uid") or row.get("finding_uid")
    if not scene or not uid:
        raise ValueError("each row needs scene_id and case_uid/finding_uid")
    return str(scene), str(uid)


def index_unique(rows: Iterable[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = case_key(row)
        if key in indexed:
            raise ValueError(f"duplicate case in {name}: {key[0]}/{key[1]}")
        indexed[key] = row
    return indexed


def validate_label_values(row: dict[str, Any], key: tuple[str, str], name: str) -> None:
    absent = [field for field in REQUIRED_LABEL_FIELDS if row.get(field) is None]
    if absent:
        raise ValueError(f"{name} has incomplete label {key[0]}/{key[1]}: {','.join(absent)}")
    for field, allowed in LABEL_ENUMS.items():
        if row.get(field) not in allowed:
            raise ValueError(f"invalid {field} for {key[0]}/{key[1]}: {row.get(field)!r}")
    for field in ("harm_confidence", "repair_confidence"):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
            raise ValueError(f"invalid {field} for {key[0]}/{key[1]}: {value!r}")
    seconds = row.get("review_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
        raise ValueError(f"invalid review_seconds for {key[0]}/{key[1]}: {seconds!r}")
    for field in ("reviewer_id", "physical_interpretation"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"invalid {field} for {key[0]}/{key[1]}: {value!r}")


def validate_completed_labels(rows: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    actual_index = index_unique(rows, "labels_r1")
    expected_index = index_unique(expected, "r1_worklist")
    missing = sorted(set(expected_index) - set(actual_index))
    extra = sorted(set(actual_index) - set(expected_index))
    if missing or extra:
        raise ValueError(
            f"R1 label coverage mismatch: expected={len(expected_index)}, actual={len(actual_index)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    incomplete: list[str] = []
    for key, row in actual_index.items():
        absent = [field for field in REQUIRED_LABEL_FIELDS if row.get(field) is None]
        if absent:
            incomplete.append(f"{key[0]}/{key[1]}:{','.join(absent)}")
            continue
        validate_label_values(row, key, "labels_r1")
    if incomplete:
        preview = "; ".join(incomplete[:5])
        raise ValueError(f"R1 contains {len(incomplete)} incomplete labels; first: {preview}")


def merge_metadata_and_labels(
    worklist: list[dict[str, Any]],
    r1: list[dict[str, Any]],
    adjudicated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    r1_index = index_unique(r1, "labels_r1")
    adj_index = index_unique(adjudicated, "labels_adjudicated")
    worklist_keys = set(index_unique(worklist, "r1_worklist"))
    if not set(adj_index) <= worklist_keys:
        raise ValueError(f"labels_adjudicated contains {len(set(adj_index) - worklist_keys)} unknown cases")
    for key, row in adj_index.items():
        validate_label_values(row, key, "labels_adjudicated")
    rows: list[dict[str, Any]] = []
    for meta in worklist:
        key = case_key(meta)
        merged = dict(meta)
        merged.update({field: r1_index[key].get(field) for field in LABEL_FIELDS if field in r1_index[key]})
        if key in adj_index:
            merged.update({field: adj_index[key].get(field) for field in LABEL_FIELDS if field in adj_index[key]})
            merged["label_source"] = "ADJUDICATED"
        else:
            merged["label_source"] = "R1"
        rows.append(merged)
    return rows


def is_actionable(row: dict[str, Any]) -> bool:
    return (
        row.get("finding_correct") == "YES"
        and row.get("downstream_harm") not in ACTIONABLE_HARM_EXCLUSIONS
        and row.get("repair_action") not in ACTIONABLE_REPAIR_EXCLUSIONS
    )


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def weighted_rate(rows: Iterable[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = float(row.get("sampling_weight", 0.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"invalid sampling_weight for {case_key(row)}: {weight}")
        denominator += weight
        if predicate(row):
            numerator += weight
    return safe_ratio(numerator, denominator)


def rate(rows: Iterable[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    values = list(rows)
    return safe_ratio(sum(1 for row in values if predicate(row)), len(values))


def priority_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("cohort") == "diagnostic_priority"]
    return sorted(
        selected,
        key=lambda row: (
            -float(row.get("review_score", 0.0)),
            str(row.get("scene_id", "")),
            str(row.get("case_uid") or row.get("finding_uid", "")),
        ),
    )


def p_at_k(rows: list[dict[str, Any]], k: int, predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    top = rows[:k]
    return rate(top, predicate)


def agreement(
    r1: list[dict[str, Any]], r2: list[dict[str, Any]], expected_subset: list[dict[str, Any]]
) -> dict[str, Any]:
    if not r2:
        return {"status": "PENDING", "expected_cases": len(expected_subset), "completed_cases": 0}
    r1_index = index_unique(r1, "labels_r1")
    r2_index = index_unique(r2, "labels_r2_subset")
    for key, row in r2_index.items():
        validate_label_values(row, key, "labels_r2_subset")
    expected_keys = set(index_unique(expected_subset, "r2_subset_manifest"))
    if set(r2_index) != expected_keys:
        raise ValueError(
            f"R2 coverage mismatch: expected={len(expected_keys)}, actual={len(r2_index)}, "
            f"missing={len(expected_keys - set(r2_index))}, extra={len(set(r2_index) - expected_keys)}"
        )
    fields = ("finding_correct", "downstream_harm", "repair_action")
    result: dict[str, Any] = {"status": "COMPLETE", "expected_cases": len(expected_keys), "completed_cases": len(r2_index)}
    for field in fields:
        same = sum(1 for key in expected_keys if r1_index[key].get(field) == r2_index[key].get(field))
        result[f"{field}_raw_agreement"] = safe_ratio(same, len(expected_keys))
    return result


def group_record(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = [row for row in rows if row.get("cohort") == "calibration_random"]
    priority = priority_rows(rows)
    review_seconds = sum(float(row.get("review_seconds", 0.0)) for row in rows)
    return {
        "group": name,
        "labeled_count": len(rows),
        "calibration_count": len(calibration),
        "weighted_finding_precision": weighted_rate(calibration, lambda row: row.get("finding_correct") == "YES") if calibration else None,
        "weighted_actionable_precision": weighted_rate(calibration, is_actionable) if calibration else None,
        "priority_count": len(priority),
        "priority_finding_precision": rate(priority, lambda row: row.get("finding_correct") == "YES"),
        "priority_actionable_precision": rate(priority, is_actionable),
        "actionable_count": sum(1 for row in rows if is_actionable(row)),
        "evidence_sufficiency": rate(rows, lambda row: row.get("evidence_sufficient") == "YES"),
        "review_hours": review_seconds / 3600.0,
    }


def grouped(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field, "UNKNOWN"))].append(row)
    return [group_record(name, buckets[name]) for name in sorted(buckets)]


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = list(group_record("", []).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def system_gates(root: Path) -> dict[str, Any]:
    parity = read_json(root / "parity" / "parity_report.json")
    run_checks: dict[str, Any] = {}
    for scene in ("room0", "office0"):
        formal = root / "runs" / scene / "formal"
        evidence_validation = read_json(formal / "audit" / "validation.json")
        audit_summary = read_json(formal / "audit_validity_gate_v1" / "audit_summary.json")
        run_checks[scene] = {
            "evidence_gate": evidence_validation.get("gate_status"),
            "audit_gate": audit_summary.get("validation_gate_status"),
            "population_censored": audit_summary.get("population_censored"),
            "weighted_precision_allowed": audit_summary.get("weighted_precision_allowed"),
        }
    parity_pass = parity.get("status") == "PASS" and all(parity.get("checks", {}).values())
    runs_pass = all(
        check["evidence_gate"] == "PASS"
        and check["audit_gate"] == "PASS"
        and check["population_censored"] is False
        and check["weighted_precision_allowed"] is True
        for check in run_checks.values()
    )
    return {
        "parity_non_interference": "PASS" if parity_pass else "FAIL",
        "formal_runs": run_checks,
        "all_system_gates_pass": parity_pass and runs_pass,
    }


def decision(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    by_scene = {record["group"]: record for record in metrics["by_scene"]}
    by_checker = metrics["by_checker"]
    room_actionable = by_scene["room0"]["weighted_actionable_precision"]
    office_actionable = by_scene["office0"]["weighted_actionable_precision"]
    cross_scene_drop = None
    if room_actionable is not None and office_actionable is not None:
        cross_scene_drop = room_actionable - office_actionable
    actionable_counts = sorted((record["actionable_count"] for record in by_checker), reverse=True)
    concentration_pass = bool(actionable_counts and actionable_counts[0] >= 20) or sum(
        value >= 8 for value in actionable_counts
    ) >= 2
    criteria = {
        "system_gates": metrics["system_gates"]["all_system_gates_pass"],
        "priority_actionable_p_at_20": overall["priority_actionable_p_at_20"] is not None
        and overall["priority_actionable_p_at_20"] >= 0.70,
        "weighted_finding_precision": overall["weighted_finding_precision"] is not None
        and overall["weighted_finding_precision"] >= 0.50,
        "weighted_actionable_precision": overall["weighted_actionable_precision"] is not None
        and overall["weighted_actionable_precision"] >= 0.35,
        "root_stage_accuracy": overall["root_stage_accuracy"] is not None
        and overall["root_stage_accuracy"] >= 0.65,
        "evidence_sufficiency": overall["evidence_sufficiency"] is not None
        and overall["evidence_sufficiency"] >= 0.80,
        "cross_scene_drop": cross_scene_drop is not None and abs(cross_scene_drop) <= 0.20,
        "actionable_checker_concentration": concentration_pass,
    }
    if not criteria["system_gates"]:
        verdict = "STOP"
    elif overall["priority_actionable_p_at_20"] is not None and overall["priority_actionable_p_at_20"] < 0.50:
        verdict = "STOP_OR_REDESIGN"
    elif all(criteria.values()):
        verdict = "GO"
    else:
        verdict = "CONDITIONAL_GO"
    return {"verdict": verdict, "criteria": criteria, "room0_minus_office0_actionable_precision": cross_scene_drop}


def markdown_decision(metrics: dict[str, Any], result: dict[str, Any]) -> str:
    overall = metrics["overall"]
    lines = [
        "# Audit Validity Gate v1 决策",
        "",
        f"结论：**{result['verdict']}**",
        "",
        "这份决策由完成并裁决后的人工标签计算得出；校准随机队列与诊断优先队列始终分开统计。",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
    ]
    shown = (
        ("校准加权 finding precision", overall["weighted_finding_precision"]),
        ("校准加权 actionable precision", overall["weighted_actionable_precision"]),
        ("优先队列 actionable P@20", overall["priority_actionable_p_at_20"]),
        ("根因阶段准确率", overall["root_stage_accuracy"]),
        ("证据充分率", overall["evidence_sufficiency"]),
        ("每小时可行动真错误", overall["actionable_error_yield_per_hour"]),
    )
    for label, value in shown:
        rendered = "N/A" if value is None else f"{value:.4f}"
        lines.append(f"| {label} | {rendered} |")
    lines.extend(["", "## GO 条件", ""])
    for key, passed in result["criteria"].items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{key}`")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- Finding 数量不是错误数量；只有人工确认后的标签进入本页。",
            "- `calibration_random` 用抽样权重估计总体 precision。",
            "- `diagnostic_priority` 只报告 P@K，不用于估计总体发生率。",
            "- 本轮不执行自动删除、拆分、合并或回滚。",
            "",
        ]
    )
    return "\n".join(lines)


def compute(root: Path) -> dict[str, Any]:
    labels_dir = root / "labels"
    worklist = read_jsonl(labels_dir / "r1_worklist.jsonl")
    r1 = read_jsonl(labels_dir / "labels_r1.jsonl")
    validate_completed_labels(r1, worklist)
    adjudicated = read_jsonl(labels_dir / "labels_adjudicated.jsonl", required=False)
    r2 = read_jsonl(labels_dir / "labels_r2_subset.jsonl", required=False)
    r2_manifest = read_jsonl(labels_dir / "r2_subset_manifest.jsonl", required=False)
    rows = merge_metadata_and_labels(worklist, r1, adjudicated)
    calibration = [row for row in rows if row.get("cohort") == "calibration_random"]
    priority = priority_rows(rows)
    finding_yes = [row for row in rows if row.get("finding_correct") == "YES"]
    total_review_hours = sum(float(row.get("review_seconds", 0.0)) for row in rows) / 3600.0
    overall = {
        "labeled_count": len(rows),
        "calibration_count": len(calibration),
        "priority_count": len(priority),
        "weighted_finding_precision": weighted_rate(calibration, lambda row: row.get("finding_correct") == "YES"),
        "weighted_actionable_precision": weighted_rate(calibration, is_actionable),
        "priority_finding_p_at_10": p_at_k(priority, 10, lambda row: row.get("finding_correct") == "YES"),
        "priority_finding_p_at_20": p_at_k(priority, 20, lambda row: row.get("finding_correct") == "YES"),
        "priority_actionable_p_at_10": p_at_k(priority, 10, is_actionable),
        "priority_actionable_p_at_20": p_at_k(priority, 20, is_actionable),
        "root_stage_accuracy": rate(finding_yes, lambda row: row.get("root_stage_correct") == "YES"),
        "evidence_sufficiency": rate(rows, lambda row: row.get("evidence_sufficient") == "YES"),
        "actionable_count": sum(1 for row in rows if is_actionable(row)),
        "review_hours": total_review_hours,
        "actionable_error_yield_per_hour": safe_ratio(sum(1 for row in rows if is_actionable(row)), total_review_hours),
    }
    return {
        "schema_version": "1.0.0",
        "system_gates": system_gates(root),
        "overall": overall,
        "by_checker": grouped(rows, "checker_id"),
        "by_stage": grouped(rows, "stage"),
        "by_scene": grouped(rows, "scene_id"),
        "reviewer_agreement": agreement(r1, r2, r2_manifest),
        "repair_action_counts": dict(Counter(str(row.get("repair_action")) for row in rows if is_actionable(row))),
        "downstream_harm_counts": dict(Counter(str(row.get("downstream_harm")) for row in rows if is_actionable(row))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.validation_root.resolve()
    try:
        metrics = compute(root)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "NOT_READY", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    result = decision(metrics)
    metrics["decision"] = result
    output_dir = root / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "metrics_by_checker.csv", metrics["by_checker"])
    write_csv(output_dir / "metrics_by_stage.csv", metrics["by_stage"])
    write_csv(output_dir / "metrics_by_scene.csv", metrics["by_scene"])
    (root / "decision.md").write_text(markdown_decision(metrics, result), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "verdict": result["verdict"], "outputs": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
