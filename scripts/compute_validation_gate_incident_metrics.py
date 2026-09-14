#!/usr/bin/env python3
"""Compute endpoint-first metrics over unique, incident-level R1 labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = "2.1.0"
THRESHOLDS = {
    "minimum_evidence_coverage": 0.80,
    "minimum_calibration_weighted_endpoint_error_precision": 0.50,
    "minimum_priority_confirmed_error_yield_at_20": 0.50,
}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    scene_id = str(row.get("scene_id") or "")
    incident_uid = str(row.get("incident_uid") or row.get("case_uid") or "")
    if not scene_id or not incident_uid:
        raise ValueError("row needs scene_id and incident_uid")
    return scene_id, incident_uid


def index_unique(rows: Iterable[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        key = case_key(row)
        if key in result:
            raise ValueError(f"duplicate {name} key: {key[0]}/{key[1]}")
        result[key] = row
    return result


def validate_label(row: dict[str, Any], key: tuple[str, str]) -> None:
    required = ("reviewer_id", "evidence_sufficient", "final_state", "final_error_type", "review_seconds")
    missing = [field for field in required if row.get(field) is None or row.get(field) == ""]
    if missing:
        raise ValueError(f"incomplete labels for {key[0]}/{key[1]}: {missing}")
    if row.get("reviewer_id") != "R1":
        raise ValueError(f"reviewer_id must be R1 for {key[0]}/{key[1]}")
    evidence = row.get("evidence_sufficient")
    state = row.get("final_state")
    error_type = row.get("final_error_type")
    if evidence not in {"YES", "NO"} or state not in FINAL_STATES or error_type not in ERROR_TYPES:
        raise ValueError(f"invalid endpoint label enum for {key[0]}/{key[1]}")
    if evidence == "NO" and state != "UNCLEAR":
        raise ValueError(f"evidence NO requires UNCLEAR for {key[0]}/{key[1]}")
    if evidence == "YES" and state == "UNCLEAR":
        raise ValueError(f"evidence YES cannot be UNCLEAR for {key[0]}/{key[1]}")
    if state == "WRONG" and error_type == "NOT_APPLICABLE":
        raise ValueError(f"WRONG requires an endpoint error type for {key[0]}/{key[1]}")
    if state != "WRONG" and error_type != "NOT_APPLICABLE":
        raise ValueError(f"non-WRONG requires NOT_APPLICABLE for {key[0]}/{key[1]}")
    if error_type == "OTHER" and not str(row.get("notes") or "").strip():
        raise ValueError(f"OTHER requires notes for {key[0]}/{key[1]}")
    try:
        seconds = float(row.get("review_seconds"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid review_seconds for {key[0]}/{key[1]}") from exc
    if seconds < 0:
        raise ValueError(f"negative review_seconds for {key[0]}/{key[1]}")


def merge_labels(worklist: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = index_unique(worklist, "worklist")
    actual = index_unique(labels, "labels_r1")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(
            f"incomplete labels: missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )
    rows = []
    for meta in worklist:
        key = case_key(meta)
        label = actual[key]
        validate_label(label, key)
        merged = dict(meta)
        for field in (
            "reviewer_id",
            "evidence_sufficient",
            "final_state",
            "final_error_type",
            "review_seconds",
            "notes",
        ):
            merged[field] = label.get(field)
        rows.append(merged)
    return rows


def is_adjudicable(row: dict[str, Any]) -> bool:
    return row.get("evidence_sufficient") == "YES"


def is_endpoint_error(row: dict[str, Any]) -> bool:
    return is_adjudicable(row) and row.get("final_state") == "WRONG"


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def rate(rows: Iterable[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    values = list(rows)
    return safe_ratio(sum(bool(predicate(row)) for row in values), len(values))


def weighted_rate(rows: Iterable[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    weighted = [
        (row, float(row["sampling_weight"]))
        for row in rows
        if row.get("sampling_weight") is not None
    ]
    return safe_ratio(
        sum(weight for row, weight in weighted if predicate(row)),
        sum(weight for _, weight in weighted),
    )


def calibration_bounds(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    weighted = [
        (row, float(row["sampling_weight"]))
        for row in rows
        if row.get("sampling_weight") is not None
    ]
    denominator = sum(weight for _, weight in weighted)
    lower = sum(weight for row, weight in weighted if is_endpoint_error(row))
    upper = sum(
        weight
        for row, weight in weighted
        if is_endpoint_error(row) or not is_adjudicable(row)
    )
    return safe_ratio(lower, denominator), safe_ratio(upper, denominator)


def priority_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row.get("cohort") == "diagnostic_priority"),
        key=lambda row: (
            -float(row.get("review_score") or 0),
            str(row.get("scene_id")),
            str(row.get("incident_uid")),
        ),
    )


def p_at_k(rows: list[dict[str, Any]], k: int, predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    selected = rows[:k]
    return rate(selected, predicate)


def conditional_p_at_k(
    rows: list[dict[str, Any]], k: int, predicate: Callable[[dict[str, Any]], bool]
) -> float | None:
    selected = [row for row in rows[:k] if is_adjudicable(row)]
    return rate(selected, predicate)


def group_record(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    adjudicable = [row for row in rows if is_adjudicable(row)]
    calibration = [row for row in rows if row.get("cohort") == "calibration_random"]
    calibration_adjudicable = [row for row in calibration if is_adjudicable(row)]
    lower, upper = calibration_bounds(calibration)
    return {
        "group": name,
        "incident_count": len(rows),
        "evidence_sufficient_count": sum(is_adjudicable(row) for row in rows),
        "evidence_sufficiency": rate(rows, is_adjudicable),
        "endpoint_error_count": sum(is_endpoint_error(row) for row in rows),
        "endpoint_error_rate_conditional": rate(adjudicable, is_endpoint_error),
        "confirmed_endpoint_error_yield": rate(rows, is_endpoint_error),
        "endpoint_error_rate_lower_bound": safe_ratio(
            sum(is_endpoint_error(row) for row in rows), len(rows)
        ),
        "endpoint_error_rate_upper_bound": safe_ratio(
            sum(is_endpoint_error(row) or not is_adjudicable(row) for row in rows),
            len(rows),
        ),
        "endpoint_correct_count": sum(row.get("final_state") == "CORRECT" for row in rows),
        "unclear_count": sum(row.get("final_state") == "UNCLEAR" for row in rows),
        "calibration_incident_count": len(calibration),
        "calibration_weighted_evidence_sufficiency": weighted_rate(calibration, is_adjudicable),
        "calibration_weighted_endpoint_error_precision": weighted_rate(
            calibration_adjudicable, is_endpoint_error
        ),
        "calibration_endpoint_error_rate_lower_bound": lower,
        "calibration_endpoint_error_rate_upper_bound": upper,
        "endpoint_error_type_counts": dict(
            Counter(str(row.get("final_error_type")) for row in rows if is_endpoint_error(row))
        ),
    }


def grouped(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    values = sorted({str(row.get(field)) for row in rows})
    return [group_record(value, [row for row in rows if str(row.get(field)) == value]) for value in values]


def grouped_membership(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    values = sorted({str(value) for row in rows for value in (row.get(field) or [])})
    records = []
    total_errors = sum(is_endpoint_error(row) for row in rows)
    for value in values:
        selected = [row for row in rows if value in {str(item) for item in (row.get(field) or [])}]
        record = group_record(value, selected)
        record["confirmed_error_coverage"] = safe_ratio(
            sum(is_endpoint_error(row) for row in selected), total_errors
        )
        records.append(record)
    return records


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    result = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(result, 3)


def timing_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("review_seconds") or 0) for row in rows]
    return {
        "case_count": len(values),
        "total_seconds": round(sum(values), 3),
        "mean_seconds": None if not values else round(statistics.mean(values), 3),
        "median_seconds": percentile(values, 0.5),
        "p25_seconds": percentile(values, 0.25),
        "p75_seconds": percentile(values, 0.75),
        "minimum_seconds": None if not values else min(values),
        "maximum_seconds": None if not values else max(values),
        "under_5_seconds": sum(value < 5 for value in values),
        "under_10_seconds": sum(value < 10 for value in values),
    }


def ranking_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [
        row
        for row in rows
        if is_adjudicable(row) and row.get("review_score") is not None
    ]
    ranked.sort(
        key=lambda row: (
            -float(row["review_score"]),
            str(row.get("scene_id")),
            str(row.get("incident_uid")),
        )
    )
    positive = [float(row["review_score"]) for row in ranked if is_endpoint_error(row)]
    negative = [float(row["review_score"]) for row in ranked if not is_endpoint_error(row)]
    auc = None
    if positive and negative:
        wins = sum(left > right for left in positive for right in negative)
        ties = sum(left == right for left in positive for right in negative)
        auc = round((wins + 0.5 * ties) / (len(positive) * len(negative)), 6)
    average_precision = None
    if positive:
        precisions = []
        errors_seen = 0
        for rank, row in enumerate(ranked, 1):
            if is_endpoint_error(row):
                errors_seen += 1
                precisions.append(errors_seen / rank)
        average_precision = round(sum(precisions) / len(positive), 6)
    prevalence = safe_ratio(len(positive), len(ranked))
    top_k = []
    for requested in (5, 10, 20, 40):
        selected = ranked[:requested]
        if not selected:
            continue
        precision = rate(selected, is_endpoint_error)
        top_k.append(
            {
                "requested_k": requested,
                "evaluated_k": len(selected),
                "confirmed_error_count": sum(is_endpoint_error(row) for row in selected),
                "confirmed_error_precision": precision,
                "lift_over_adjudicable_prevalence": (
                    None
                    if precision is None or prevalence in (None, 0)
                    else round(precision / prevalence, 6)
                ),
            }
        )
    return {
        "score_direction": "higher review_score was intended to mean higher diagnostic priority",
        "adjudicable_scored_endpoint_count": len(ranked),
        "confirmed_error_count": len(positive),
        "confirmed_correct_count": len(negative),
        "confirmed_error_prevalence": prevalence,
        "roc_auc": auc,
        "average_precision": average_precision,
        "wrong_score_mean": None if not positive else round(statistics.mean(positive), 6),
        "correct_score_mean": None if not negative else round(statistics.mean(negative), 6),
        "wrong_score_median": percentile(positive, 0.5),
        "correct_score_median": percentile(negative, 0.5),
        "top_k": top_k,
    }


def endpoint_error_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = [row for row in rows if is_endpoint_error(row)]
    values = sorted({str(row["final_error_type"]) for row in errors})
    return [
        {
            "endpoint_error_type": value,
            "count": sum(row["final_error_type"] == value for row in errors),
            "share_of_confirmed_errors": safe_ratio(
                sum(row["final_error_type"] == value for row in errors), len(errors)
            ),
            "by_scene": dict(
                Counter(
                    str(row["scene_id"])
                    for row in errors
                    if row["final_error_type"] == value
                )
            ),
        }
        for value in values
    ]


def verify_review_projection(root: Path, worklist: list[dict[str, Any]]) -> dict[str, Any]:
    worklist_path = root / "labels" / "r1_worklist.jsonl"
    manifest = read_json(root / "review_evidence_manifest.json")
    work_index = index_unique(worklist, "worklist")
    manifest_index = {}
    for item in manifest.get("cases") or []:
        key = (str(item.get("scene_id") or ""), str(item.get("incident_uid") or item.get("case_uid") or ""))
        if key in manifest_index:
            raise ValueError(f"duplicate review manifest incident: {key}")
        manifest_index[key] = item
    case_keys_match = set(work_index) == set(manifest_index)
    review_hashes_match = True
    source_case_hashes_match = True
    displayed_asset_hashes_match = True
    checked_assets = 0
    for key, row in work_index.items():
        item = manifest_index.get(key)
        if item is None:
            continue
        case_dir = Path(str(row["case_dir"])).resolve()
        review_path = case_dir / "review_evidence.json"
        case_path = case_dir / "case.json"
        if not review_path.is_file() or item.get("review_evidence_sha256") != sha256_file(review_path):
            review_hashes_match = False
            continue
        review = read_json(review_path)
        if review.get("scene_id") != key[0] or review.get("case_uid") != key[1]:
            review_hashes_match = False
        if review.get("source_case_json_sha256") != sha256_file(case_path):
            source_case_hashes_match = False
        for name, expected_sha in (review.get("displayed_asset_sha256") or {}).items():
            asset = (case_dir / str(name)).resolve()
            if case_dir != asset and case_dir not in asset.parents:
                displayed_asset_hashes_match = False
                continue
            checked_assets += 1
            if not asset.is_file() or sha256_file(asset) != expected_sha:
                displayed_asset_hashes_match = False
    structural_gate = all(
        (
            manifest.get("schema_version") == SCHEMA_VERSION,
            manifest.get("status") in {"READY", "READY_WITH_DECLARED_LIMITATIONS"},
            manifest.get("worklist_sha256") == sha256_file(worklist_path),
            int(manifest.get("case_count", -1)) == len(worklist),
            case_keys_match,
            review_hashes_match,
            source_case_hashes_match,
            displayed_asset_hashes_match,
            manifest.get("all_artifact_hashes_match") is True,
            manifest.get("all_available_final_objects_link_exactly") is True,
        )
    )
    return {
        "structural_gate": "PASS" if structural_gate else "FAIL",
        "schema_version": manifest.get("schema_version"),
        "status": manifest.get("status"),
        "case_keys_match": case_keys_match,
        "review_hashes_match": review_hashes_match,
        "source_case_hashes_match": source_case_hashes_match,
        "displayed_asset_hashes_match": displayed_asset_hashes_match,
        "checked_displayed_asset_count": checked_assets,
        "critical_gap_cases_by_checker": manifest.get("critical_gap_cases_by_checker") or {},
    }


def system_gates(root: Path, worklist: list[dict[str, Any]]) -> dict[str, Any]:
    assembly = read_json(root / "incident_worklist_manifest.json")
    parity_path = root / "parity" / "parity_report.json"
    parity = read_json(parity_path) if parity_path.is_file() else {"status": "NOT_COPIED"}
    scenes = []
    for scene in assembly.get("scenes") or []:
        experiment = Path(str(scene["experiment_dir"])).resolve()
        audit_dir = Path(str(scene["audit_dir"])).resolve()
        evidence_validation = read_json(experiment / "audit" / "validation.json")
        audit_summary = read_json(audit_dir / "audit_summary.json")
        selection = read_json(audit_dir / "case_selection.json")
        passed = all(
            (
                evidence_validation.get("gate_status") == "PASS",
                audit_summary.get("validation_gate_status") == "PASS",
                audit_summary.get("population_censored") is False,
                audit_summary.get("weighted_precision_allowed") is True,
                selection.get("annotation_unit") == "incident",
                selection.get("weighted_precision_allowed") is True,
            )
        )
        scenes.append(
            {
                "scene_id": scene.get("scene_id"),
                "status": "PASS" if passed else "FAIL",
                "evidence_gate": evidence_validation.get("gate_status"),
                "audit_gate": audit_summary.get("validation_gate_status"),
                "population_censored": audit_summary.get("population_censored"),
                "reviewable_incidents": (selection.get("deduplication") or {}).get("reviewable_incident_count"),
                "blocked_incidents": (selection.get("deduplication") or {}).get("blocked_incident_count"),
            }
        )
    projection = verify_review_projection(root, worklist)
    parity_pass = parity.get("status") == "PASS"
    all_pass = parity_pass and all(scene["status"] == "PASS" for scene in scenes) and projection["structural_gate"] == "PASS"
    before_block = sum(
        int(((scene.get("deduplication") or {}).get("incident_count_before_evidence_block") or 0))
        for scene in assembly.get("scenes") or []
    )
    blocked = sum(
        int(((scene.get("deduplication") or {}).get("blocked_incident_count") or 0))
        for scene in assembly.get("scenes") or []
    )
    return {
        "all_system_gates_pass": all_pass,
        "selection_mode": assembly.get("selection_mode"),
        "full_endpoint_census": assembly.get("full_endpoint_census") is True,
        "parity": {"status": parity.get("status"), "accepted": parity_pass},
        "scenes": scenes,
        "endpoint_census_denominators": {
            "before_machine_evidence_block": before_block,
            "machine_evidence_blocked": blocked,
            "human_reviewable": len(worklist),
        },
        "human_system_evidence_projection": projection,
    }


def decision(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    census = metrics["system_gates"].get("full_endpoint_census") is True
    if census:
        coverage_values = [
            overall.get("evidence_sufficiency"),
            *(record.get("evidence_sufficiency") for record in metrics["by_scene"]),
        ]
    else:
        coverage_values = [
            overall.get("evidence_sufficiency"),
            overall.get("calibration_weighted_evidence_sufficiency"),
            overall.get("priority_evidence_coverage_at_20"),
            *(record.get("calibration_weighted_evidence_sufficiency") for record in metrics["by_scene"]),
        ]
    coverage_pass = all(
        value is not None and value >= THRESHOLDS["minimum_evidence_coverage"]
        for value in coverage_values
    )
    precision = overall.get("calibration_weighted_endpoint_error_precision")
    priority_yield = overall.get("priority_confirmed_error_yield_at_20")
    precision_pass = None if census else (
        precision is not None
        and precision >= THRESHOLDS["minimum_calibration_weighted_endpoint_error_precision"]
    )
    priority_pass = None if census else (
        priority_yield is not None
        and priority_yield >= THRESHOLDS["minimum_priority_confirmed_error_yield_at_20"]
    )
    if not metrics["system_gates"]["all_system_gates_pass"]:
        verdict = "STOP_SYSTEM_INTEGRITY"
    elif not coverage_pass:
        verdict = "STOP_OR_REDESIGN_EVIDENCE"
    elif census and int(overall.get("endpoint_error_count") or 0) == 0:
        verdict = "NO_CONFIRMED_ENDPOINT_ERRORS_REVISE_SCREENERS_OR_EXPAND_EVALUATION"
    elif census:
        verdict = "PROCEED_TO_EXPERT_TRACE"
    elif not precision_pass or not priority_pass:
        verdict = "REVISE_SCREENERS"
    else:
        verdict = "PROCEED_TO_EXPERT_TRACE"
    return {
        "verdict": verdict,
        "r1_scope": "unique incident endpoint validity only",
        "selection_mode": "endpoint_census" if census else "dual_cohort_sample",
        "confirmed_endpoint_error_count": overall.get("endpoint_error_count"),
        "confirmed_endpoint_error_yield": overall.get("confirmed_endpoint_error_yield"),
        "evidence_coverage_pass": coverage_pass,
        "calibration_endpoint_precision_pass": precision_pass,
        "priority_endpoint_yield_pass": priority_pass,
        "repair_gate_status": "PENDING_EXPERT_TRACE_AND_REPLAY",
        "thresholds": THRESHOLDS,
        "limitations": [
            "R1 has one human reviewer and does not establish inter-rater reliability.",
            "R1 does not validate root stage or repair action; those require expert trace plus intervention/replay.",
        ],
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row if not isinstance(row.get(key), (dict, list))})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})


def markdown(metrics: dict[str, Any], result: dict[str, Any]) -> str:
    overall = metrics["overall"]
    timing = metrics["review_timing"]["overall"]
    ranking = metrics["ranking_diagnostics"]
    return "\n".join(
        [
            "# Incident-level validation gate",
            "",
            f"Status: `{result['verdict']}`",
            "",
            "## R1 endpoint results",
            "",
            f"- Selection mode: {result['selection_mode']}",
            f"- Unique incidents: {overall['incident_count']}",
            f"- Evidence sufficiency: {overall['evidence_sufficiency']}",
            f"- Confirmed endpoint errors: {overall['endpoint_error_count']}",
            f"- Confirmed endpoint correct: {overall['endpoint_correct_count']}",
            f"- Human-unclear endpoints: {overall['unclear_count']}",
            f"- Endpoint-error rate among sufficient cases: {overall['endpoint_error_rate_conditional']}",
            f"- Full endpoint bounds: [{overall['endpoint_error_rate_lower_bound']}, {overall['endpoint_error_rate_upper_bound']}]",
            f"- Bounds including the machine-blocked endpoint: [{overall['all_flagged_endpoint_error_lower_bound']}, {overall['all_flagged_endpoint_error_upper_bound']}]",
            f"- Total / median review time: {timing['total_seconds']} s / {timing['median_seconds']} s",
            "",
            "## Screener ranking diagnostic",
            "",
            f"- review_score ROC AUC: {ranking['roc_auc']}",
            f"- review_score average precision: {ranking['average_precision']}",
            f"- adjudicable endpoint-error prevalence: {ranking['confirmed_error_prevalence']}",
            "",
            "Higher review_score was intended to rank risk. AUC and top-k lift below the prevalence baseline mean the score should be revised rather than used as a probability.",
            "",
            "## Interpretation",
            "",
            "R1 answers only whether an error remains visible in the exact final map. Checker stage and repair are not human-label fields.",
            "Even after an R1 pass, repair remains pending until confirmed endpoint errors receive expert causal traces and actual intervention/replay verification.",
            "",
            "## Known limitation",
            "",
            "This run has one human R1 reviewer; it does not claim inter-rater reliability.",
            "",
        ]
    )


def compute(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    labels_dir = root / "labels"
    worklist = read_jsonl(labels_dir / "r1_worklist.jsonl")
    labels = read_jsonl(labels_dir / "labels_r1.jsonl")
    rows = merge_labels(worklist, labels)
    calibration = [row for row in rows if row.get("cohort") == "calibration_random"]
    calibration_adjudicable = [row for row in calibration if is_adjudicable(row)]
    priority = priority_rows(rows)
    overall = group_record("overall", rows)
    overall.update(
        {
            "calibration_weighted_endpoint_error_precision": weighted_rate(
                calibration_adjudicable, is_endpoint_error
            ),
            "priority_endpoint_precision_at_10_conditional": conditional_p_at_k(priority, 10, is_endpoint_error),
            "priority_endpoint_precision_at_20_conditional": conditional_p_at_k(priority, 20, is_endpoint_error),
            "priority_evidence_coverage_at_10": p_at_k(priority, 10, is_adjudicable),
            "priority_evidence_coverage_at_20": p_at_k(priority, 20, is_adjudicable),
            "priority_confirmed_error_yield_at_10": p_at_k(priority, 10, is_endpoint_error),
            "priority_confirmed_error_yield_at_20": p_at_k(priority, 20, is_endpoint_error),
            "mean_review_seconds": safe_ratio(
                sum(float(row.get("review_seconds") or 0) for row in rows), len(rows)
            ),
        }
    )
    gates = system_gates(root, worklist)
    denominators = gates["endpoint_census_denominators"]
    all_flagged = int(denominators["before_machine_evidence_block"])
    blocked = int(denominators["machine_evidence_blocked"])
    overall.update(
        {
            "all_flagged_endpoint_error_lower_bound": safe_ratio(
                overall["endpoint_error_count"], all_flagged
            ),
            "all_flagged_endpoint_error_upper_bound": safe_ratio(
                overall["endpoint_error_count"] + overall["unclear_count"] + blocked,
                all_flagged,
            ),
        }
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "annotation_unit": "incident",
        "system_gates": gates,
        "overall": overall,
        "by_scene": grouped(rows, "scene_id"),
        "by_cohort": grouped(rows, "cohort"),
        "by_representative_checker": grouped(rows, "checker_id"),
        "by_representative_stage": grouped(rows, "stage"),
        "by_linked_checker": grouped_membership(rows, "checker_ids"),
        "by_linked_stage": grouped_membership(rows, "stages"),
        "endpoint_error_types": endpoint_error_types(rows),
        "ranking_diagnostics": ranking_diagnostics(rows),
        "review_timing": {
            "overall": timing_record(rows),
            "by_final_state": {
                state: timing_record([row for row in rows if row["final_state"] == state])
                for state in sorted(FINAL_STATES)
            },
        },
        "label_quality": {
            "semantic_validation": "PASS",
            "exact_worklist_coverage": True,
            "labels_r1_sha256": sha256_file(labels_dir / "labels_r1.jsonl"),
            "nonempty_notes_count": sum(bool(str(row.get("notes") or "").strip()) for row in rows),
        },
        "linked_checker_count_distribution": dict(
            Counter(len(row.get("checker_ids") or []) for row in rows)
        ),
    }
    result = decision(metrics)
    return metrics, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.validation_root.resolve()
    try:
        metrics, result = compute(root)
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "NOT_READY", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "incident_endpoint_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(metrics_dir / "metrics_by_scene.csv", metrics["by_scene"])
    write_csv(metrics_dir / "metrics_by_representative_checker.csv", metrics["by_representative_checker"])
    write_csv(metrics_dir / "metrics_by_representative_stage.csv", metrics["by_representative_stage"])
    write_csv(metrics_dir / "metrics_by_linked_checker.csv", metrics["by_linked_checker"])
    write_csv(metrics_dir / "metrics_by_linked_stage.csv", metrics["by_linked_stage"])
    write_csv(metrics_dir / "endpoint_error_types.csv", metrics["endpoint_error_types"])
    (root / "decision.md").write_text(markdown(metrics, result), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "decision": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
