#!/usr/bin/env python3
"""Summarize annotation validity and preliminary Experiment 0 labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(raw) for raw in handle if raw.strip()]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def agreement_value(label: dict[str, Any]) -> tuple[Any, ...]:
    return (
        label["blind"]["observation_quality"],
        tuple(sorted(label["blind"]["matching_candidate_codes"])),
        label["final"]["target_state"],
        label["final"]["outside_candidate_status"],
        label["derived"]["derived_action"],
        label["derived"]["eligible_main"],
    )


def main() -> int:
    args = parse_args()
    root = args.packet_root.resolve()
    manifest = read_json(root / "manifest.json")
    worklist = read_jsonl(root / "worklist.jsonl")
    labels = {str(row["case_uid"]): row for row in read_jsonl(root / "labels/event_labels.jsonl")}
    work_by_case = {str(row["case_uid"]): row for row in worklist}
    base_cases = [row for row in worklist if not row.get("repeat_of")]
    base_labels = [labels[str(row["case_uid"])] for row in base_cases if str(row["case_uid"]) in labels]

    derived_counts = Counter(row["derived"]["derived_status"] for row in base_labels)
    action_counts = Counter(row["derived"]["derived_action"] for row in base_labels)
    observation_counts = Counter(row["blind"]["observation_quality"] for row in base_labels)
    evidence_counts = Counter(row["final"]["evidence_sufficient"] for row in base_labels)
    eligible = [row for row in base_labels if row["derived"]["eligible_main"]]
    main_set = [row for row in base_labels if row["derived"]["main_set"]]
    sensitivity_set = [row for row in base_labels if row["derived"]["sensitivity_set"]]
    main_roots = sum(bool(row["derived"]["is_root_false_attach"]) for row in main_set)
    sensitivity_roots = sum(bool(row["derived"]["is_root_false_attach"]) for row in sensitivity_set)

    repeat_pairs = []
    for row in worklist:
        repeat_of = row.get("repeat_of")
        if not repeat_of:
            continue
        repeat_uid = str(row["case_uid"])
        original_uid = str(repeat_of)
        if repeat_uid not in labels or original_uid not in labels:
            continue
        left = labels[original_uid]
        right = labels[repeat_uid]
        repeat_pairs.append({
            "original": original_uid,
            "repeat": repeat_uid,
            "exact_agreement": agreement_value(left) == agreement_value(right),
            "eligible_agreement": left["derived"]["eligible_main"] == right["derived"]["eligible_main"],
            "action_agreement": left["derived"]["derived_action"] == right["derived"]["derived_action"],
        })

    def ratio(key: str) -> float | None:
        if not repeat_pairs:
            return None
        return sum(bool(row[key]) for row in repeat_pairs) / len(repeat_pairs)

    blind_times = [float(row["blind"].get("blind_review_seconds") or 0) for row in base_labels]
    final_times = [float(row["final"].get("final_review_seconds") or 0) for row in base_labels]
    evidence_sufficient = evidence_counts.get("YES", 0)
    validity = {
        "evidence_sufficient_rate": evidence_sufficient / len(base_labels) if base_labels else None,
        "hidden_repeat_pairs_completed": len(repeat_pairs),
        "hidden_repeat_exact_agreement": ratio("exact_agreement"),
        "hidden_repeat_eligible_agreement": ratio("eligible_agreement"),
        "hidden_repeat_action_agreement": ratio("action_agreement"),
        "calibration_ready": bool(
            len(base_labels) == len(base_cases)
            and len(repeat_pairs) >= 4
            and evidence_sufficient / max(1, len(base_labels)) >= 0.80
            and ratio("eligible_agreement") is not None
            and ratio("eligible_agreement") >= 0.90
            and ratio("action_agreement") is not None
            and ratio("action_agreement") >= 0.90
        ),
    }

    result = {
        "schema_version": "experiment0-annotation-summary/1.0",
        "status": "COMPLETE" if len(labels) == len(worklist) else "ANNOTATION_INCOMPLETE",
        "packet_status": manifest.get("status"),
        "case_counts": {
            "worklist_total": len(worklist),
            "base_total": len(base_cases),
            "labels_total": len(labels),
            "base_labels": len(base_labels),
            "eligible_main_or_sensitivity": len(eligible),
        },
        "validity": validity,
        "label_counts": {
            "observation_quality": dict(sorted(observation_counts.items())),
            "evidence_sufficient": dict(sorted(evidence_counts.items())),
            "derived_status": dict(sorted(derived_counts.items())),
            "derived_action": dict(sorted(action_counts.items())),
        },
        "preliminary_rates": {
            "main_clean_root_errors": main_roots,
            "main_clean_denominator": len(main_set),
            "main_clean_root_error_rate": main_roots / len(main_set) if main_set else None,
            "main_clean_wilson95": wilson(main_roots, len(main_set)),
            "clean_plus_borderline_root_errors": sensitivity_roots,
            "clean_plus_borderline_denominator": len(sensitivity_set),
            "clean_plus_borderline_root_error_rate": sensitivity_roots / len(sensitivity_set) if sensitivity_set else None,
            "clean_plus_borderline_wilson95": wilson(sensitivity_roots, len(sensitivity_set)),
            "warning": "Calibration-balanced or case-harvest queues must not be used as natural prevalence estimates.",
        },
        "timing_seconds": {
            "blind_total": round(sum(blind_times), 1),
            "final_total": round(sum(final_times), 1),
            "combined_total": round(sum(blind_times) + sum(final_times), 1),
            "mean_per_base_case": round((sum(blind_times) + sum(final_times)) / len(base_labels), 1) if base_labels else None,
        },
        "repeat_pairs": repeat_pairs,
        "limitations": [
            "Root/cascade closure across post-process object merges requires the later episode compiler.",
            "REASSIGN outside displayed top-K remains DEFER unless the reviewer checks the full t^- map.",
            "Only probability-sampled queues support a natural occurrence-rate estimate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": result["status"],
        "calibration_ready": validity["calibration_ready"],
        "labels": len(labels),
        "worklist": len(worklist),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

