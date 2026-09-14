#!/usr/bin/env python3
"""Compute Audit Validity Gate v1 metrics from completed human labels.

This script deliberately refuses incomplete, contradictory, or placeholder
labels.  It keeps the calibration and diagnostic cohorts separate, applies
adjudicated labels as the final authority, and writes the four metric files plus
a human-readable decision.

Accuracy is conditioned on evidence_sufficient=YES.  PARTIAL/NO cases are
coverage failures, not negative findings.  The report therefore includes both
conditional accuracy and conservative lower/upper bounds over the full sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
REVIEW_EVIDENCE_SCHEMA = "1.0.0"
REVIEW_EVIDENCE_MANIFEST = "review_evidence_manifest.json"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(value: Any, *, label: str) -> Path:
    path = Path(str(value or ""))
    if not str(value or "") or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def verify_review_projection_files(
    root: Path,
    review_manifest: dict[str, Any],
    worklist: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recheck the exact review JSON and images at final decision time."""

    expected_keys = set(index_unique(worklist, "r1_worklist"))
    manifest_cases = review_manifest.get("cases") or []
    manifest_index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest_cases:
        key = case_key(item)
        if key in manifest_index:
            raise ValueError(f"duplicate case in review evidence manifest: {key[0]}/{key[1]}")
        manifest_index[key] = item
    case_keys_match = set(manifest_index) == expected_keys
    review_json_hashes_match = True
    case_json_bindings_match = True
    displayed_asset_hashes_match = True
    checked_asset_count = 0
    for key, item in manifest_index.items():
        relative = safe_relative_path(item.get("review_evidence_path"), label="review_evidence_path")
        review_path = root / relative
        if not review_path.is_file() or sha256_file(review_path) != item.get("review_evidence_sha256"):
            review_json_hashes_match = False
            continue
        review = read_json(review_path)
        if review.get("scene_id") != key[0] or review.get("case_uid") != key[1]:
            case_json_bindings_match = False
        case_path = review_path.parent / "case.json"
        if (
            not case_path.is_file()
            or review.get("source_case_json_sha256") != sha256_file(case_path)
        ):
            case_json_bindings_match = False
        declared_assets = review.get("displayed_asset_sha256")
        if not isinstance(declared_assets, dict) or len(declared_assets) != item.get("displayed_asset_count"):
            displayed_asset_hashes_match = False
            continue
        case_dir = review_path.parent.resolve()
        for name, expected_sha in declared_assets.items():
            asset_relative = safe_relative_path(name, label="displayed asset path")
            asset_path = (case_dir / asset_relative).resolve()
            if case_dir != asset_path and case_dir not in asset_path.parents:
                raise ValueError(f"review asset escaped case directory: {name}")
            checked_asset_count += 1
            if not asset_path.is_file() or sha256_file(asset_path) != expected_sha:
                displayed_asset_hashes_match = False
    return {
        "case_keys_match": case_keys_match,
        "review_json_hashes_match": review_json_hashes_match,
        "case_json_bindings_match": case_json_bindings_match,
        "displayed_asset_hashes_match": displayed_asset_hashes_match,
        "checked_review_json_count": len(manifest_index),
        "checked_displayed_asset_count": checked_asset_count,
        "all_projection_files_match": (
            case_keys_match
            and review_json_hashes_match
            and case_json_bindings_match
            and displayed_asset_hashes_match
        ),
    }


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
    evidence = row["evidence_sufficient"]
    finding = row["finding_correct"]
    root = row["root_stage_correct"]
    harm = row["downstream_harm"]
    repair = row["repair_action"]
    locality = row["repair_locality"]
    if evidence == "YES" and (
        finding == "UNCERTAIN"
        or root == "UNCERTAIN"
        or harm == "UNKNOWN"
        or repair in {"UNKNOWN", "NEED_MORE_VIEW"}
    ):
        raise ValueError(
            f"evidence_sufficient=YES cannot contain unresolved conclusions for {key[0]}/{key[1]}"
        )
    if evidence == "PARTIAL" and not str(row.get("notes") or "").strip():
        raise ValueError(f"evidence_sufficient=PARTIAL needs a missing-evidence note for {key[0]}/{key[1]}")
    if finding == "UNCERTAIN" and (
        root != "UNCERTAIN"
        or harm != "UNKNOWN"
        or repair != "NEED_MORE_VIEW"
        or locality != "NOT_APPLICABLE"
    ):
        raise ValueError(
            f"finding_correct=UNCERTAIN cannot carry guessed root/harm/repair for {key[0]}/{key[1]}"
        )
    if evidence == "NO" and finding != "UNCERTAIN":
        raise ValueError(f"evidence_sufficient=NO requires finding_correct=UNCERTAIN for {key[0]}/{key[1]}")
    if finding == "NO" and (
        root != "NOT_APPLICABLE"
        or harm != "NONE"
        or repair != "NONE"
        or locality != "NOT_APPLICABLE"
    ):
        raise ValueError(f"finding_correct=NO has contradictory downstream labels for {key[0]}/{key[1]}")
    if finding == "YES" and root == "NOT_APPLICABLE":
        raise ValueError(f"finding_correct=YES requires an applicable root-stage judgment for {key[0]}/{key[1]}")
    if harm == "NONE" and repair != "NONE":
        raise ValueError(f"downstream_harm=NONE requires repair_action=NONE for {key[0]}/{key[1]}")
    if repair == "NONE" and harm != "NONE":
        raise ValueError(f"repair_action=NONE requires downstream_harm=NONE for {key[0]}/{key[1]}")
    if repair in {"NONE", "NEED_MORE_VIEW"} and locality != "NOT_APPLICABLE":
        raise ValueError(f"repair_action={repair} requires NOT_APPLICABLE locality for {key[0]}/{key[1]}")
    if repair in {"REASSIGN_OBSERVATION", "MERGE_OBJECTS", "SPLIT_OBJECT"} and locality != "MULTI_OBJECT":
        raise ValueError(f"repair_action={repair} requires MULTI_OBJECT locality for {key[0]}/{key[1]}")


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


def is_adjudicable(row: dict[str, Any]) -> bool:
    """A label can enter accuracy denominators only when its evidence is complete."""

    return row.get("evidence_sufficient") == "YES"


def is_confirmed_finding(row: dict[str, Any]) -> bool:
    return is_adjudicable(row) and row.get("finding_correct") == "YES"


def is_actionable(row: dict[str, Any]) -> bool:
    return (
        is_confirmed_finding(row)
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


def weighted_bounds(
    rows: Iterable[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, float | None]:
    """Bound a full-sample rate without treating missing evidence as negative.

    The lower bound counts only confirmed positives.  The upper bound assumes
    every PARTIAL/NO case could have been positive.  Both use the frozen
    calibration sampling weights.
    """

    total = 0.0
    confirmed = 0.0
    unresolved = 0.0
    for row in rows:
        weight = float(row.get("sampling_weight", 0.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"invalid sampling_weight for {case_key(row)}: {weight}")
        total += weight
        if is_adjudicable(row):
            if predicate(row):
                confirmed += weight
        else:
            unresolved += weight
    return {
        "lower": safe_ratio(confirmed, total),
        "upper": safe_ratio(confirmed + unresolved, total),
    }


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


def conditional_p_at_k(
    rows: list[dict[str, Any]], k: int, predicate: Callable[[dict[str, Any]], bool]
) -> float | None:
    top_adjudicable = [row for row in rows[:k] if is_adjudicable(row)]
    return rate(top_adjudicable, predicate)


def agreement(
    r1: list[dict[str, Any]],
    r2: list[dict[str, Any]],
    expected_subset: list[dict[str, Any]],
    adjudicated: list[dict[str, Any]],
) -> dict[str, Any]:
    if not r2:
        return {
            "status": "PENDING",
            "expected_cases": len(expected_subset),
            "completed_cases": 0,
            "disagreement_cases": None,
            "unadjudicated_disagreement_cases": None,
        }
    r1_index = index_unique(r1, "labels_r1")
    r2_index = index_unique(r2, "labels_r2_subset")
    for key, row in r2_index.items():
        validate_label_values(row, key, "labels_r2_subset")
    expected_keys = set(index_unique(expected_subset, "r2_subset_manifest"))
    extra_keys = set(r2_index) - expected_keys
    if extra_keys:
        raise ValueError(
            f"R2 contains {len(extra_keys)} cases outside the frozen subset: "
            + ",".join(f"{scene}/{uid}" for scene, uid in sorted(extra_keys))
        )
    missing_keys = expected_keys - set(r2_index)
    if missing_keys:
        return {
            "status": "PENDING",
            "expected_cases": len(expected_keys),
            "completed_cases": len(r2_index),
            "missing_cases": len(missing_keys),
            "missing_case_keys": [f"{scene}/{uid}" for scene, uid in sorted(missing_keys)],
            "disagreement_cases": None,
            "unadjudicated_disagreement_cases": None,
        }
    fields = ("finding_correct", "downstream_harm", "repair_action")
    disagreement_keys = {
        key
        for key in expected_keys
        if any(r1_index[key].get(field) != r2_index[key].get(field) for field in fields)
    }
    adjudicated_keys = set(index_unique(adjudicated, "labels_adjudicated"))
    missing_adjudication = disagreement_keys - adjudicated_keys
    result: dict[str, Any] = {
        "status": "NEEDS_ADJUDICATION" if missing_adjudication else "COMPLETE",
        "expected_cases": len(expected_keys),
        "completed_cases": len(r2_index),
        "disagreement_cases": len(disagreement_keys),
        "unadjudicated_disagreement_cases": len(missing_adjudication),
        "unadjudicated_case_keys": [f"{scene}/{uid}" for scene, uid in sorted(missing_adjudication)],
    }
    for field in fields:
        same = sum(1 for key in expected_keys if r1_index[key].get(field) == r2_index[key].get(field))
        result[f"{field}_raw_agreement"] = safe_ratio(same, len(expected_keys))
    return result


def group_record(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = [row for row in rows if row.get("cohort") == "calibration_random"]
    calibration_adjudicable = [row for row in calibration if is_adjudicable(row)]
    priority = priority_rows(rows)
    priority_adjudicable = [row for row in priority if is_adjudicable(row)]
    review_seconds = sum(float(row.get("review_seconds", 0.0)) for row in rows)
    finding_bounds = weighted_bounds(calibration, is_confirmed_finding) if calibration else {"lower": None, "upper": None}
    actionable_bounds = weighted_bounds(calibration, is_actionable) if calibration else {"lower": None, "upper": None}
    return {
        "group": name,
        "labeled_count": len(rows),
        "adjudicable_count": len([row for row in rows if is_adjudicable(row)]),
        "partial_evidence_count": len([row for row in rows if row.get("evidence_sufficient") == "PARTIAL"]),
        "no_evidence_count": len([row for row in rows if row.get("evidence_sufficient") == "NO"]),
        "calibration_count": len(calibration),
        "calibration_adjudicable_count": len(calibration_adjudicable),
        "weighted_evidence_sufficiency": weighted_rate(calibration, is_adjudicable) if calibration else None,
        "weighted_finding_precision": weighted_rate(calibration_adjudicable, is_confirmed_finding) if calibration_adjudicable else None,
        "weighted_actionable_precision": weighted_rate(calibration_adjudicable, is_actionable) if calibration_adjudicable else None,
        "weighted_finding_rate_lower_bound": finding_bounds["lower"],
        "weighted_finding_rate_upper_bound": finding_bounds["upper"],
        "weighted_actionable_rate_lower_bound": actionable_bounds["lower"],
        "weighted_actionable_rate_upper_bound": actionable_bounds["upper"],
        "priority_count": len(priority),
        "priority_adjudicable_count": len(priority_adjudicable),
        "priority_evidence_coverage": rate(priority, is_adjudicable),
        "priority_finding_precision": rate(priority_adjudicable, is_confirmed_finding),
        "priority_actionable_precision": rate(priority_adjudicable, is_actionable),
        "priority_confirmed_finding_yield": rate(priority, is_confirmed_finding),
        "priority_confirmed_actionable_yield": rate(priority, is_actionable),
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
    review_manifest = read_json(root / REVIEW_EVIDENCE_MANIFEST)
    worklist_path = root / "labels" / "r1_worklist.jsonl"
    worklist = read_jsonl(worklist_path)
    worklist_count = len(worklist)
    projection_file_checks = verify_review_projection_files(root, review_manifest, worklist)
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
    review_projection_pass = (
        review_manifest.get("schema_version") == REVIEW_EVIDENCE_SCHEMA
        and review_manifest.get("status") in {"READY", "READY_WITH_DECLARED_LIMITATIONS"}
        and review_manifest.get("worklist_sha256") == sha256_file(worklist_path)
        and review_manifest.get("case_count") == worklist_count
        and len(review_manifest.get("cases") or []) == worklist_count
        and review_manifest.get("all_artifact_hashes_match") is True
        and review_manifest.get("all_available_final_objects_link_exactly") is True
        and projection_file_checks["all_projection_files_match"]
    )
    return {
        "parity_non_interference": "PASS" if parity_pass else "FAIL",
        "formal_runs": run_checks,
        "human_system_evidence_projection": {
            "status": review_manifest.get("status"),
            "schema_version": review_manifest.get("schema_version"),
            "case_count": review_manifest.get("case_count"),
            "worklist_sha256_match": review_manifest.get("worklist_sha256") == sha256_file(worklist_path),
            "all_artifact_hashes_match": review_manifest.get("all_artifact_hashes_match"),
            "all_available_final_objects_link_exactly": review_manifest.get("all_available_final_objects_link_exactly"),
            "critical_gap_cases_by_checker": review_manifest.get("critical_gap_cases_by_checker") or {},
            **projection_file_checks,
            "structural_gate": "PASS" if review_projection_pass else "FAIL",
        },
        "all_system_gates_pass": parity_pass and runs_pass and review_projection_pass,
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
        "independent_r2_and_adjudication_complete": metrics["reviewer_agreement"]["status"] == "COMPLETE",
        "priority_actionable_p_at_20": overall["priority_actionable_p_at_20"] is not None
        and overall["priority_actionable_p_at_20"] >= 0.70,
        "weighted_finding_precision": overall["weighted_finding_precision"] is not None
        and overall["weighted_finding_precision"] >= 0.50,
        "weighted_actionable_precision": overall["weighted_actionable_precision"] is not None
        and overall["weighted_actionable_precision"] >= 0.35,
        "root_stage_accuracy": overall["root_stage_accuracy"] is not None
        and overall["root_stage_accuracy"] >= 0.65,
        "overall_evidence_sufficiency": overall["evidence_sufficiency"] is not None
        and overall["evidence_sufficiency"] >= 0.80,
        "weighted_calibration_evidence_coverage": overall["weighted_evidence_sufficiency"] is not None
        and overall["weighted_evidence_sufficiency"] >= 0.80,
        "priority_evidence_coverage_at_20": overall["priority_evidence_coverage_at_20"] is not None
        and overall["priority_evidence_coverage_at_20"] >= 0.80,
        "per_scene_weighted_calibration_evidence_coverage": all(
            by_scene.get(scene, {}).get("weighted_evidence_sufficiency") is not None
            and by_scene[scene]["weighted_evidence_sufficiency"] >= 0.80
            for scene in ("room0", "office0")
        ),
        "cross_scene_drop": cross_scene_drop is not None and abs(cross_scene_drop) <= 0.20,
        "actionable_checker_concentration": concentration_pass,
    }
    evidence_coverage_pass = all(
        criteria[name]
        for name in (
            "overall_evidence_sufficiency",
            "weighted_calibration_evidence_coverage",
            "priority_evidence_coverage_at_20",
            "per_scene_weighted_calibration_evidence_coverage",
        )
    )
    if not criteria["system_gates"]:
        verdict = "STOP"
    elif not evidence_coverage_pass:
        verdict = "STOP_OR_REDESIGN_EVIDENCE"
    elif metrics["reviewer_agreement"]["status"] == "PENDING":
        verdict = "PENDING_INDEPENDENT_R2"
    elif metrics["reviewer_agreement"]["status"] == "NEEDS_ADJUDICATION":
        verdict = "PENDING_ADJUDICATION"
    elif overall["priority_actionable_p_at_20"] is not None and overall["priority_actionable_p_at_20"] < 0.50:
        verdict = "STOP_OR_REDESIGN"
    elif all(criteria.values()):
        verdict = "GO"
    else:
        verdict = "CONDITIONAL_GO"
    return {
        "verdict": verdict,
        "criteria": criteria,
        "evidence_coverage_pass": evidence_coverage_pass,
        "room0_minus_office0_actionable_precision": cross_scene_drop,
    }


def markdown_decision(metrics: dict[str, Any], result: dict[str, Any]) -> str:
    overall = metrics["overall"]
    lines = [
        "# Audit Validity Gate v1 决策",
        "",
        f"结论：**{result['verdict']}**",
        "",
        "这份决策由完成并裁决后的人工标签计算得出；校准随机队列与诊断优先队列始终分开统计。",
        "PARTIAL/NO 只表示人类证据覆盖不足，不会被当成 finding=NO。准确率仅在 evidence_sufficient=YES 的案例上计算。",
        "32 例独立 R2 未完成，或 R1/R2 分歧尚未逐例裁决时，本工具只能输出 PENDING 状态，不能给出最终 GO。",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
    ]
    shown = (
        ("校准加权 finding precision（仅证据充分）", overall["weighted_finding_precision"]),
        ("校准加权 finding 全样本下界", overall["weighted_finding_rate_lower_bound"]),
        ("校准加权 finding 全样本上界", overall["weighted_finding_rate_upper_bound"]),
        ("校准加权 actionable precision（仅证据充分）", overall["weighted_actionable_precision"]),
        ("优先队列 actionable P@20（仅证据充分）", overall["priority_actionable_p_at_20"]),
        ("优先队列 confirmed actionable yield@20（全 20 例）", overall["priority_confirmed_actionable_yield_at_20"]),
        ("根因阶段准确率（仅证据充分且 finding=YES）", overall["root_stage_accuracy"]),
        ("全部案例证据充分率", overall["evidence_sufficiency"]),
        ("校准队列加权证据充分率", overall["weighted_evidence_sufficiency"]),
        ("优先队列证据覆盖率@20", overall["priority_evidence_coverage_at_20"]),
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
            "- `evidence_sufficient=YES` 才进入 accuracy/precision 分母；PARTIAL/NO 进入覆盖率与上下界。",
            "- `calibration_random` 用抽样权重估计条件 precision，并报告未决证据造成的全样本上下界。",
            "- `diagnostic_priority` 的条件 P@K 与全 K 个案例上的 confirmed yield 分开报告，不用于估计总体发生率。",
            "- 任一核心证据覆盖率低于 80%，结论固定为 `STOP_OR_REDESIGN_EVIDENCE`；不能用低覆盖下的高准确率放行。",
            "- 独立 R2 必须覆盖冻结的 32 例；R1/R2 在 finding、harm、repair 任一字段有分歧时，必须存在 adjudicated label。",
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
    calibration_adjudicable = [row for row in calibration if is_adjudicable(row)]
    priority = priority_rows(rows)
    adjudicable_finding_yes = [row for row in rows if is_confirmed_finding(row)]
    finding_bounds = weighted_bounds(calibration, is_confirmed_finding)
    actionable_bounds = weighted_bounds(calibration, is_actionable)
    total_review_hours = sum(float(row.get("review_seconds", 0.0)) for row in rows) / 3600.0
    overall = {
        "labeled_count": len(rows),
        "adjudicable_count": sum(1 for row in rows if is_adjudicable(row)),
        "partial_evidence_count": sum(1 for row in rows if row.get("evidence_sufficient") == "PARTIAL"),
        "no_evidence_count": sum(1 for row in rows if row.get("evidence_sufficient") == "NO"),
        "calibration_count": len(calibration),
        "calibration_adjudicable_count": len(calibration_adjudicable),
        "priority_count": len(priority),
        "weighted_evidence_sufficiency": weighted_rate(calibration, is_adjudicable),
        "weighted_finding_precision": weighted_rate(calibration_adjudicable, is_confirmed_finding),
        "weighted_actionable_precision": weighted_rate(calibration_adjudicable, is_actionable),
        "weighted_finding_rate_lower_bound": finding_bounds["lower"],
        "weighted_finding_rate_upper_bound": finding_bounds["upper"],
        "weighted_actionable_rate_lower_bound": actionable_bounds["lower"],
        "weighted_actionable_rate_upper_bound": actionable_bounds["upper"],
        "priority_evidence_coverage_at_10": p_at_k(priority, 10, is_adjudicable),
        "priority_evidence_coverage_at_20": p_at_k(priority, 20, is_adjudicable),
        "priority_finding_p_at_10": conditional_p_at_k(priority, 10, is_confirmed_finding),
        "priority_finding_p_at_20": conditional_p_at_k(priority, 20, is_confirmed_finding),
        "priority_actionable_p_at_10": conditional_p_at_k(priority, 10, is_actionable),
        "priority_actionable_p_at_20": conditional_p_at_k(priority, 20, is_actionable),
        "priority_confirmed_finding_yield_at_10": p_at_k(priority, 10, is_confirmed_finding),
        "priority_confirmed_finding_yield_at_20": p_at_k(priority, 20, is_confirmed_finding),
        "priority_confirmed_actionable_yield_at_10": p_at_k(priority, 10, is_actionable),
        "priority_confirmed_actionable_yield_at_20": p_at_k(priority, 20, is_actionable),
        "root_stage_accuracy": rate(adjudicable_finding_yes, lambda row: row.get("root_stage_correct") == "YES"),
        "evidence_sufficiency": rate(rows, lambda row: row.get("evidence_sufficient") == "YES"),
        "actionable_count": sum(1 for row in rows if is_actionable(row)),
        "review_hours": total_review_hours,
        "actionable_error_yield_per_hour": safe_ratio(sum(1 for row in rows if is_actionable(row)), total_review_hours),
    }
    return {
        "schema_version": "2.0.0",
        "system_gates": system_gates(root),
        "overall": overall,
        "by_checker": grouped(rows, "checker_id"),
        "by_stage": grouped(rows, "stage"),
        "by_scene": grouped(rows, "scene_id"),
        "reviewer_agreement": agreement(r1, r2, r2_manifest, adjudicated),
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
