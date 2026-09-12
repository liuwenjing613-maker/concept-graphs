from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


CONFUSION_CELLS = ("TP", "FN", "FP", "TN")


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def confusion_cell(is_error: bool, gate_triggered: bool) -> str:
    if is_error:
        return "TP" if gate_triggered else "FN"
    return "FP" if gate_triggered else "TN"


def summarize_detection(
    rows: Iterable[Mapping[str, Any]],
    *,
    evaluable_field: str,
    error_field: str,
) -> dict[str, Any]:
    rows = list(rows)
    evaluable = [row for row in rows if bool(row.get(evaluable_field))]
    cells = Counter(
        confusion_cell(bool(row[error_field]), bool(row.get("gate_triggered")))
        for row in evaluable
    )
    tp, fn, fp, tn = (int(cells[name]) for name in CONFUSION_CELLS)
    positives = tp + fn
    negatives = fp + tn
    triggered = tp + fp
    passed = fn + tn
    precision = safe_div(tp, triggered)
    recall = safe_div(tp, positives)
    specificity = safe_div(tn, negatives)
    return {
        "all_proposal_count": len(rows),
        "evaluable_count": len(evaluable),
        "unevaluable_count": len(rows) - len(evaluable),
        "evaluation_coverage": safe_div(len(evaluable), len(rows)),
        "positive_error_count": positives,
        "negative_correct_count": negatives,
        "confusion_matrix": {name: int(cells[name]) for name in CONFUSION_CELLS},
        "error_recall": recall,
        "missed_error_rate": safe_div(fn, positives),
        "gate_precision": precision,
        "specificity": specificity,
        "false_intervention_rate": safe_div(fp, negatives),
        "residual_error_rate_among_passed": safe_div(fn, passed),
        "gate_intervention_rate": safe_div(triggered, len(evaluable)),
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall > 0.0
            else None
        ),
        "balanced_accuracy": (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        ),
    }


def summarize_post_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    evaluable_field: str,
    error_field: str,
) -> dict[str, Any]:
    evaluable = [row for row in rows if bool(row.get(evaluable_field))]
    caught_errors = [
        row
        for row in evaluable
        if bool(row[error_field]) and bool(row.get("gate_triggered"))
    ]
    false_interventions = [
        row
        for row in evaluable
        if not bool(row[error_field]) and bool(row.get("gate_triggered"))
    ]
    corrected = sum(row.get("final_identity_correct") is True for row in caught_errors)
    still_wrong = sum(row.get("final_identity_correct") is False for row in caught_errors)
    correction_unknown = len(caught_errors) - corrected - still_wrong
    preserved = sum(
        row.get("final_identity_correct") is True for row in false_interventions
    )
    harmed = sum(
        row.get("final_identity_correct") is False for row in false_interventions
    )
    preservation_unknown = len(false_interventions) - preserved - harmed
    return {
        "caught_error_count": len(caught_errors),
        "corrected_error_count": int(corrected),
        "still_wrong_after_intervention_count": int(still_wrong),
        "post_intervention_unknown_error_count": int(correction_unknown),
        "correction_success_rate_over_caught_errors": safe_div(
            corrected, len(caught_errors)
        ),
        "false_intervention_count": len(false_interventions),
        "correct_association_preserved_count": int(preserved),
        "harmed_correct_association_count": int(harmed),
        "post_intervention_unknown_correct_count": int(preservation_unknown),
        "conditional_harm_rate_over_false_interventions": safe_div(
            harmed, len(false_interventions)
        ),
        "net_corrected_minus_harmed": int(corrected - harmed),
        "net_gain_over_evaluable_proposals": safe_div(
            corrected - harmed, len(evaluable)
        ),
    }


def summarize_error_subtypes(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not row.get("wrong_fusion_evaluable") or not row.get("wrong_fusion_error"):
            continue
        subtype = str(row.get("wrong_fusion_subtype") or "UNKNOWN")
        grouped.setdefault(subtype, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for subtype, group in sorted(grouped.items()):
        caught = sum(bool(row.get("gate_triggered")) for row in group)
        total = len(group)
        result[subtype] = {
            "error_count": total,
            "caught_count": caught,
            "missed_count": total - caught,
            "error_recall": safe_div(caught, total),
            "missed_error_rate": safe_div(total - caught, total),
        }
    return result
