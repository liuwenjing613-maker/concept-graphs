#!/usr/bin/env python3
"""Compute endpoint-label repeatability between frozen R1 and blinded R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


FINAL_STATES = {"CORRECT", "WRONG", "UNCLEAR"}
ERROR_TYPES = {
    "NOT_APPLICABLE",
    "FALSE_MERGE",
    "FALSE_SPLIT",
    "SPURIOUS_OBJECT",
    "MISSING_OBJECT",
    "WRONG_MEMBERSHIP",
    "GEOMETRY_CORRUPTION",
    "SEMANTIC_IDENTITY_ERROR",
    "OTHER",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    value = str(row.get("scene_id") or ""), str(
        row.get("incident_uid") or row.get("case_uid") or ""
    )
    if not all(value):
        raise ValueError("row needs scene_id and incident_uid")
    return value


def index_unique(rows: Iterable[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        key = case_key(row)
        if key in result:
            raise ValueError(f"duplicate {name} key: {key}")
        result[key] = row
    return result


def validate_label(row: dict[str, Any], reviewer_id: str, key: tuple[str, str]) -> None:
    if row.get("reviewer_id") != reviewer_id:
        raise ValueError(f"expected {reviewer_id} label at {key}")
    evidence = row.get("evidence_sufficient")
    state = row.get("final_state")
    error_type = row.get("final_error_type")
    if evidence not in {"YES", "NO"} or state not in FINAL_STATES or error_type not in ERROR_TYPES:
        raise ValueError(f"invalid endpoint label enum at {key}")
    if evidence == "NO" and state != "UNCLEAR":
        raise ValueError(f"evidence NO requires UNCLEAR at {key}")
    if evidence == "YES" and state == "UNCLEAR":
        raise ValueError(f"evidence YES cannot be UNCLEAR at {key}")
    if (state == "WRONG") == (error_type == "NOT_APPLICABLE"):
        raise ValueError(f"final state and error type conflict at {key}")
    if error_type == "OTHER" and not str(row.get("notes") or "").strip():
        raise ValueError(f"OTHER requires notes at {key}")
    try:
        seconds = float(row.get("review_seconds"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid review_seconds at {key}") from exc
    if seconds < 0:
        raise ValueError(f"negative review_seconds at {key}")


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def confusion(first: list[str], second: list[str]) -> dict[str, dict[str, int]]:
    categories = sorted(set(first) | set(second))
    matrix = {left: {right: 0 for right in categories} for left in categories}
    for left, right in zip(first, second, strict=True):
        matrix[left][right] += 1
    return matrix


def cohen_kappa(first: list[str], second: list[str]) -> float | None:
    if not first or len(first) != len(second):
        return None
    total = len(first)
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / total
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(first_counts[value] * second_counts[value] for value in set(first) | set(second)) / (
        total * total
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return round((observed - expected) / (1 - expected), 6)


def agreement(first: list[str], second: list[str], *, include_kappa: bool = True) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("agreement inputs have different lengths")
    agreed = sum(left == right for left, right in zip(first, second, strict=True))
    total = len(first)
    return {
        "case_count": total,
        "agreed_count": agreed,
        "raw_agreement": None if total == 0 else round(agreed / total, 6),
        "raw_agreement_wilson_95": wilson(agreed, total),
        "cohen_kappa": cohen_kappa(first, second) if include_kappa else None,
        "r1_distribution": dict(Counter(first)),
        "r2_distribution": dict(Counter(second)),
        "confusion_r1_rows_r2_columns": confusion(first, second),
    }


def endpoint_class(row: dict[str, Any]) -> str:
    if row.get("evidence_sufficient") == "NO":
        return "UNADJUDICABLE"
    return "CONFIRMED_ERROR" if row.get("final_state") == "WRONG" else "CONFIRMED_CORRECT"


def compute(
    root: Path,
    r1_labels_path: Path,
    *,
    relationship: str = "same-reviewer",
) -> dict[str, Any]:
    worklist_path = root / "labels" / "r2_worklist.jsonl"
    r2_labels_path = root / "labels" / "labels_r2.jsonl"
    worklist = read_jsonl(worklist_path)
    r1_index = index_unique(read_jsonl(r1_labels_path), "R1 labels")
    r2_index = index_unique(read_jsonl(r2_labels_path), "R2 labels")
    expected = index_unique(worklist, "R2 worklist")
    if set(r2_index) != set(expected):
        missing = sorted(set(expected) - set(r2_index))
        extra = sorted(set(r2_index) - set(expected))
        raise ValueError(
            f"R2 incomplete or contains unknown cases: missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )
    if not set(expected) <= set(r1_index):
        raise ValueError("frozen R1 labels do not cover the R2 subset")

    pairs = []
    for row in worklist:
        key = case_key(row)
        r1 = r1_index[key]
        r2 = r2_index[key]
        validate_label(r1, "R1", key)
        validate_label(r2, "R2", key)
        pairs.append((key, r1, r2))

    def field_values(getter: Callable[[dict[str, Any]], str]) -> tuple[list[str], list[str]]:
        return [getter(r1) for _, r1, _ in pairs], [getter(r2) for _, _, r2 in pairs]

    evidence = field_values(lambda row: str(row["evidence_sufficient"]))
    final_state = field_values(lambda row: str(row["final_state"]))
    endpoint = field_values(endpoint_class)
    full_error_type = field_values(lambda row: str(row["final_error_type"]))
    exact = field_values(
        lambda row: "|".join(
            (
                str(row["evidence_sufficient"]),
                str(row["final_state"]),
                str(row["final_error_type"]),
            )
        )
    )
    union_wrong = [pair for pair in pairs if pair[1]["final_state"] == "WRONG" or pair[2]["final_state"] == "WRONG"]
    both_wrong = [pair for pair in pairs if pair[1]["final_state"] == "WRONG" and pair[2]["final_state"] == "WRONG"]

    disagreements = []
    for key, r1, r2 in pairs:
        fields = {
            field: {"R1": r1[field], "R2": r2[field]}
            for field in ("evidence_sufficient", "final_state", "final_error_type")
            if r1[field] != r2[field]
        }
        if fields:
            disagreements.append(
                {
                    "scene_id": key[0],
                    "incident_uid": key[1],
                    "case_dir": expected[key].get("case_dir"),
                    "adjudication_status": "PENDING_HUMAN_ADJUDICATION",
                    "changed_fields": fields,
                    "r1_label": {
                        field: r1.get(field)
                        for field in ("evidence_sufficient", "final_state", "final_error_type", "notes")
                    },
                    "r2_label": {
                        field: r2.get(field)
                        for field in ("evidence_sufficient", "final_state", "final_error_type", "notes")
                    },
                }
            )

    by_scene = []
    for scene in sorted({key[0] for key, _, _ in pairs}):
        scene_pairs = [pair for pair in pairs if pair[0][0] == scene]
        left = [str(r1["final_state"]) for _, r1, _ in scene_pairs]
        right = [str(r2["final_state"]) for _, _, r2 in scene_pairs]
        record = agreement(left, right)
        record["scene_id"] = scene
        by_scene.append(record)

    r2_seconds = [float(r2["review_seconds"]) for _, _, r2 in pairs]
    state_transitions = Counter(
        f"{r1['final_state']} -> {r2['final_state']}" for _, r1, r2 in pairs
    )
    state_stability = {}
    for state in sorted(FINAL_STATES):
        selected = [(r1, r2) for _, r1, r2 in pairs if r1["final_state"] == state]
        if selected:
            state_stability[state] = {
                "r1_case_count": len(selected),
                "same_state_count": sum(r2["final_state"] == state for _, r2 in selected),
                "same_state_rate": round(
                    sum(r2["final_state"] == state for _, r2 in selected) / len(selected), 6
                ),
            }
    same_reviewer = relationship == "same-reviewer"
    metrics = {
        "schema_version": "1.0.0",
        "status": "COMPLETE",
        "agreement_type": "intra_rater_test_retest" if same_reviewer else "inter_rater",
        "relationship": relationship,
        "case_count": len(pairs),
        "source_integrity": {
            "r2_worklist_sha256": sha256_file(worklist_path),
            "frozen_r1_labels_sha256": sha256_file(r1_labels_path),
            "r2_labels_sha256": sha256_file(r2_labels_path),
        },
        "evidence_sufficiency": agreement(*evidence),
        "final_state": agreement(*final_state),
        "endpoint_class": agreement(*endpoint),
        "exact_three_field_label": agreement(*exact),
        "error_type_all_cases": agreement(*full_error_type),
        "error_type_when_either_round_wrong": agreement(
            [str(r1["final_error_type"]) if r1["final_state"] == "WRONG" else "NON_WRONG" for _, r1, _ in union_wrong],
            [str(r2["final_error_type"]) if r2["final_state"] == "WRONG" else "NON_WRONG" for _, _, r2 in union_wrong],
        ),
        "error_type_when_both_rounds_wrong": agreement(
            [str(r1["final_error_type"]) for _, r1, _ in both_wrong],
            [str(r2["final_error_type"]) for _, _, r2 in both_wrong],
        ),
        "by_scene_final_state": by_scene,
        "final_state_transitions": dict(state_transitions),
        "r1_state_stability_in_stratified_subset": state_stability,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "r2_review_seconds": {
            "total": round(sum(r2_seconds), 1),
            "mean": round(sum(r2_seconds) / len(r2_seconds), 3),
            "median": round(statistics.median(r2_seconds), 3),
            "minimum": min(r2_seconds),
            "maximum": max(r2_seconds),
        },
        "interpretation_limits": [
            (
                "The same person completed R1 and R2, so these are test-retest/intra-rater measurements, "
                "not independent-reviewer reliability."
                if same_reviewer
                else "R1 and R2 are treated as different people; identity must be documented outside the label round IDs."
            ),
            "The 24-case subset is enriched for endpoint states and error types; it is not a new population error-rate sample.",
            "A short interval between rounds can inflate agreement through memory, so the result is a lower-strength sensitivity check.",
            "Kappa is descriptive for this stratified subset and should be read alongside raw agreement and confusion matrices.",
        ],
    }
    return metrics


def markdown(metrics: dict[str, Any]) -> str:
    final_state = metrics["final_state"]
    exact = metrics["exact_three_field_label"]
    evidence = metrics["evidence_sufficiency"]
    error_type = metrics["error_type_when_both_rounds_wrong"]
    kind = "同一复核者重复稳定性（intra-rater）" if metrics["relationship"] == "same-reviewer" else "独立复核者一致性"
    return "\n".join(
        [
            "# R2 endpoint 复核一致性",
            "",
            f"口径：**{kind}**",
            "",
            f"- R2 案例数：{metrics['case_count']}",
            f"- 证据充分性一致：{evidence['agreed_count']}/{evidence['case_count']} = {evidence['raw_agreement']}",
            f"- 最终状态一致：{final_state['agreed_count']}/{final_state['case_count']} = {final_state['raw_agreement']}",
            f"- 最终状态 Cohen's kappa：{final_state['cohen_kappa']}",
            f"- 三字段完全一致：{exact['agreed_count']}/{exact['case_count']} = {exact['raw_agreement']}",
            f"- 两轮都判 WRONG 时错误类型一致：{error_type['agreed_count']}/{error_type['case_count']} = {error_type['raw_agreement']}",
            f"- 有任一字段变化的案例：{metrics['disagreement_count']}",
            f"- R1 WRONG 保持 WRONG：{metrics['r1_state_stability_in_stratified_subset']['WRONG']['same_state_count']}/{metrics['r1_state_stability_in_stratified_subset']['WRONG']['r1_case_count']}",
            "",
            "## 应如何解释",
            "",
            "R2 只检验同一证据、同一简化字段在第二次判断时是否稳定。它不重新估计 97 个 endpoint 的错误率，也不验证阶段根因或修复动作。",
            "",
            "若 R1、R2 都由同一人完成，本结果不能写成 inter-rater reliability。两轮间隔很短时，还应明确说明记忆可能使一致率偏高。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--r1-labels", required=True, type=Path)
    parser.add_argument("--relationship", choices=("same-reviewer", "independent-reviewer"), default="same-reviewer")
    args = parser.parse_args()
    root = args.validation_root.resolve()
    try:
        metrics = compute(root, args.r1_labels.resolve(), relationship=args.relationship)
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "NOT_READY", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output = metrics_dir / "r2_repeatability.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "r2_repeatability.md").write_text(markdown(metrics), encoding="utf-8")
    disagreement_queue = root / "expert" / "r2_disagreement_queue.jsonl"
    disagreement_queue.parent.mkdir(parents=True, exist_ok=True)
    disagreement_queue.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in metrics["disagreements"]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "agreement_type": metrics["agreement_type"],
                "case_count": metrics["case_count"],
                "evidence_agreement": metrics["evidence_sufficiency"]["raw_agreement"],
                "final_state_agreement": metrics["final_state"]["raw_agreement"],
                "final_state_kappa": metrics["final_state"]["cohen_kappa"],
                "exact_three_field_agreement": metrics["exact_three_field_label"]["raw_agreement"],
                "output": str(output),
                "disagreement_queue": str(disagreement_queue),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
