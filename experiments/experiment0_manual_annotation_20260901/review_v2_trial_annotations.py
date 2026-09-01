#!/usr/bin/env python3
"""Audit completed Experiment 0 v2 trial annotations against packet bindings.

The private auto-routing strata are treated as an audit reference, not as a
replacement for human adjudication.  Raw labels are never modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from label_logic_v2 import (
    derive_routing_label,
    validate_blind_label,
    validate_final_label,
)


ROUTING_LABELS = {
    "CORRECT_ATTACH",
    "WRONG_ATTACH_EXISTING",
    "SHOULD_HAVE_BEEN_NEW",
    "CORRECT_NEW",
    "WRONG_NEW_FALSE_SPLIT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def by_case(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_uid = row["case_uid"]
        if case_uid in result:
            raise ValueError(f"duplicate case_uid: {case_uid}")
        result[case_uid] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "sum": 0.0, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "sum": round(sum(values), 1),
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def normalized_human_targets(matches: list[str]) -> set[str] | None:
    value = set(matches)
    if value == {"NONE_SHOWN"}:
        return set()
    if value == {"UNCERTAIN"}:
        return None
    return value


def main() -> None:
    args = parse_args()
    root = args.packet_root.resolve()
    output = (args.output or root / "annotation_review_metrics.json").resolve()

    worklist = by_case(read_jsonl(root / "worklist.jsonl"))
    private = by_case(read_jsonl(root / "private_v2_trial_worklist.jsonl"))
    drafts = by_case(read_jsonl(root / "labels" / "blind_drafts.jsonl"))
    labels = by_case(read_jsonl(root / "labels" / "event_labels.jsonl"))
    case_uids = sorted(worklist)
    errors: list[str] = []

    case_sets = {
        "worklist": set(worklist),
        "private_worklist": set(private),
        "blind_drafts": set(drafts),
        "event_labels": set(labels),
    }
    expected_cases = case_sets["worklist"]
    for name, values in case_sets.items():
        if values != expected_cases:
            errors.append(
                f"{name} case set differs: missing={sorted(expected_cases-values)}, "
                f"extra={sorted(values-expected_cases)}"
            )

    rows: list[dict[str, Any]] = []
    quality_neutralized_agreement = 0
    private_identity_eligible = 0
    private_identity_exact = 0
    blind_seconds: list[float] = []
    final_seconds: list[float] = []
    total_seconds: list[float] = []

    for case_uid in case_uids:
        label = labels[case_uid]
        draft = drafts[case_uid]
        private_row = private[case_uid]
        public_path = root / "cases" / case_uid / "case_public.json"
        private_path = root / "cases" / case_uid / "case_private.json"
        public_case = json.loads(public_path.read_text(encoding="utf-8"))
        private_case = json.loads(private_path.read_text(encoding="utf-8"))
        candidate_codes = {candidate["code"] for candidate in public_case["candidates"]}

        for source_name, source in (
            ("worklist", worklist[case_uid]),
            ("private_worklist", private_row),
            ("blind_draft", draft),
            ("event_label", label),
            ("case_public", public_case),
            ("case_private", private_case),
        ):
            if source.get("case_uid") != case_uid:
                errors.append(f"{case_uid}: {source_name} case_uid mismatch")
            if source.get("event_uid") != label["event_uid"]:
                errors.append(f"{case_uid}: {source_name} event_uid mismatch")

        expected_snapshot = public_case["tminus_snapshot_sha256"]
        for source_name, source in (
            ("worklist", worklist[case_uid]),
            ("private_worklist", private_row),
            ("event_label", label),
        ):
            recorded_snapshot = source.get("tminus_snapshot_sha256")
            if recorded_snapshot is not None and recorded_snapshot != expected_snapshot:
                errors.append(f"{case_uid}: {source_name} snapshot hash mismatch")

        if private_case.get("source_public_sha256") != sha256(public_path):
            errors.append(f"{case_uid}: case_private is not bound to case_public")
        for asset_name, expected_hash in public_case["displayed_asset_sha256"].items():
            asset_path = public_path.parent / asset_name
            if not asset_path.is_file():
                errors.append(f"{case_uid}: missing displayed asset {asset_name}")
            elif sha256(asset_path) != expected_hash:
                errors.append(f"{case_uid}: displayed asset hash mismatch: {asset_name}")

        if private_case["original_action_type"] != label["reveal"]["original_action_type"]:
            errors.append(f"{case_uid}: revealed action differs from private packet")
        if private_case.get("original_target_code") != label["reveal"]["original_target_code"]:
            errors.append(f"{case_uid}: revealed target differs from private packet")

        if draft["blind"] != label["blind"]:
            errors.append(f"{case_uid}: final label changed mapper-blind answer")

        try:
            clean_blind = validate_blind_label(label["blind"], candidate_codes)
            clean_final = validate_final_label(
                label["final"], clean_blind, label["reveal"]["original_action_type"]
            )
            recomputed = derive_routing_label(
                clean_blind,
                clean_final,
                label["reveal"]["original_action_type"],
                label["reveal"]["original_target_code"],
            )
            if recomputed != label["derived"]:
                errors.append(f"{case_uid}: stored derived fields differ from recomputation")
        except ValueError as exc:
            errors.append(f"{case_uid}: schema validation failed: {exc}")
            recomputed = label["derived"]

        neutralized_blind = copy.deepcopy(label["blind"])
        if neutralized_blind["observation_quality"] == "GRANULARITY_AMBIGUOUS":
            neutralized_blind["observation_quality"] = "CLEAN_SINGLE_INSTANCE"
        neutralized = derive_routing_label(
            neutralized_blind,
            label["final"],
            label["reveal"]["original_action_type"],
            label["reveal"]["original_target_code"],
        )
        private_route = private_row["private_auto_routing_label"]
        quality_neutralized_agreement += int(neutralized["routing_label"] == private_route)

        private_targets = set(
            private_case["private_full_map_gt_audit"]["legal_candidate_codes"]
        )
        human_targets = normalized_human_targets(label["blind"]["matching_candidate_codes"])
        evaluate_identity = private_route in ROUTING_LABELS
        if evaluate_identity:
            private_identity_eligible += 1
            private_identity_exact += int(human_targets == private_targets)

        blind_time = float(label["blind"]["blind_review_seconds"])
        final_time = float(label["final"]["final_review_seconds"])
        blind_seconds.append(blind_time)
        final_seconds.append(final_time)
        total_seconds.append(blind_time + final_time)

        rows.append(
            {
                "case_uid": case_uid,
                "event_uid": label["event_uid"],
                "sample_kind": private_row["sample_kind"],
                "repeat_of": private_row.get("repeat_of"),
                "frame": label["event_frame_idx"],
                "original_action_type": label["reveal"]["original_action_type"],
                "original_target_code": label["reveal"]["original_target_code"],
                "observation_quality": label["blind"]["observation_quality"],
                "identity_evidence_status": label["blind"]["identity_evidence_status"],
                "human_target_codes": label["blind"]["matching_candidate_codes"],
                "private_target_codes": sorted(private_targets),
                "identity_target_exact": human_targets == private_targets
                if evaluate_identity
                else None,
                "human_routing_label": label["derived"]["routing_label"],
                "quality_neutralized_routing_label": neutralized["routing_label"],
                "private_auto_routing_label": private_route,
                "raw_routing_agreement": label["derived"]["routing_label"] == private_route,
                "quality_neutralized_routing_agreement": neutralized["routing_label"]
                == private_route,
                "target_pre_state": label["final"]["target_pre_state"],
                "full_map_status": label["final"]["full_map_status"],
                "confidence": label["final"]["confidence"],
                "blind_review_seconds": blind_time,
                "final_review_seconds": final_time,
                "physical_instance_note": label["blind"].get("physical_instance_note"),
                "causal_note": label["final"].get("causal_note"),
                "notes": label["final"].get("notes"),
            }
        )

    repeat_rows: list[dict[str, Any]] = []
    repeat_fields = (
        ("observation_quality", lambda x: x["blind"]["observation_quality"]),
        ("matching_candidate_codes", lambda x: sorted(x["blind"]["matching_candidate_codes"])),
        ("identity_evidence_status", lambda x: x["blind"]["identity_evidence_status"]),
        ("target_pre_state", lambda x: x["final"]["target_pre_state"]),
        ("full_map_status", lambda x: x["final"]["full_map_status"]),
        ("outside_matching_node_uids", lambda x: sorted(x["final"]["outside_matching_node_uids"])),
        ("routing_label", lambda x: x["derived"]["routing_label"]),
    )
    repeat_field_counts = Counter()
    for case_uid in case_uids:
        original_uid = private[case_uid].get("repeat_of")
        if not original_uid:
            continue
        agreements = {
            name: getter(labels[case_uid]) == getter(labels[original_uid])
            for name, getter in repeat_fields
        }
        repeat_field_counts.update({name: int(value) for name, value in agreements.items()})
        repeat_rows.append(
            {
                "repeat_case_uid": case_uid,
                "original_case_uid": original_uid,
                "field_agreement": agreements,
                "all_core_fields_exact": all(agreements.values()),
                "identity_core_exact": all(
                    agreements[name]
                    for name in (
                        "matching_candidate_codes",
                        "identity_evidence_status",
                        "target_pre_state",
                        "full_map_status",
                        "outside_matching_node_uids",
                    )
                ),
            }
        )

    status_counts = Counter(label["derived"]["annotation_status"] for label in labels.values())
    quality_counts = Counter(label["blind"]["observation_quality"] for label in labels.values())
    evidence_counts = Counter(label["blind"]["identity_evidence_status"] for label in labels.values())
    route_counts = Counter(label["derived"]["routing_label"] for label in labels.values())
    private_route_counts = Counter(row["private_auto_routing_label"] for row in private.values())
    raw_route_agreement = sum(
        labels[uid]["derived"]["routing_label"]
        == private[uid]["private_auto_routing_label"]
        for uid in case_uids
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "experiment0-v2-annotation-review/1.0",
        "purpose": "SCHEMA_AND_ANNOTATION_VALIDITY_ONLY_NOT_PREVALENCE",
        "packet_root": str(root),
        "packet_manifest_sha256": sha256(root / "manifest.json"),
        "case_count": len(case_uids),
        "base_case_count": sum(not private[uid].get("repeat_of") for uid in case_uids),
        "hidden_repeat_count": len(repeat_rows),
        "mapper_complete": manifest["mapper_complete"],
        "mapper_latest_frame": manifest["mapper_latest_frame"],
        "binding_and_schema": {
            "status": "PASS" if not errors else "FAIL",
            "error_count": len(errors),
            "errors": errors,
        },
        "completion": {
            "blind_draft_count": len(drafts),
            "final_label_count": len(labels),
            "unique_case_count": len(set(labels)),
        },
        "counts": {
            "annotation_status": dict(sorted(status_counts.items())),
            "observation_quality": dict(sorted(quality_counts.items())),
            "identity_evidence_status": dict(sorted(evidence_counts.items())),
            "human_routing_label": dict(sorted(route_counts.items())),
            "private_auto_routing_label": dict(sorted(private_route_counts.items())),
        },
        "private_reference_comparison": {
            "warning": "Private auto strata are audit references and require visual adjudication.",
            "raw_route_exact": raw_route_agreement,
            "raw_route_total": len(case_uids),
            "quality_neutralized_route_exact": quality_neutralized_agreement,
            "quality_neutralized_route_total": len(case_uids),
            "identity_target_exact": private_identity_exact,
            "identity_target_total": private_identity_eligible,
        },
        "hidden_repeat_consistency": {
            "pair_count": len(repeat_rows),
            "all_core_fields_exact": sum(row["all_core_fields_exact"] for row in repeat_rows),
            "identity_core_exact": sum(row["identity_core_exact"] for row in repeat_rows),
            "field_exact_counts": dict(sorted(repeat_field_counts.items())),
            "pairs": repeat_rows,
        },
        "timing_seconds": {
            "blind": summarize_numbers(blind_seconds),
            "final": summarize_numbers(final_seconds),
            "total_per_case": summarize_numbers(total_seconds),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
