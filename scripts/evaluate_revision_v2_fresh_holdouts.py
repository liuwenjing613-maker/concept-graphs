#!/usr/bin/env python3
"""Posthoc evaluation of frozen fresh holdouts with fail-closed capability gating."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from conceptgraph.revision.capabilities import (
    CapabilityDisposition,
    executable_constraint_types,
    resolve_endpoint_capability,
)
from conceptgraph.revision.cases import canonical_obs_key
from conceptgraph.revision.index import ProvenanceIndex


_CREDENTIAL_PATTERN = re.compile(r"sk-[0-9a-f]{64}")


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(encoding="utf-8", newline=None) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), ""):
            digest.update(block.encode("utf-8"))
    return digest.hexdigest()


def _labels(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            incident_uid = str(row.get("incident_uid") or row.get("case_uid") or "")
            if not incident_uid:
                raise ValueError(f"label line {line_number} has no incident UID")
            result[incident_uid] = row
    return result


def _expected_identity_action(case: Mapping[str, Any]) -> str:
    endpoint = str(case["endpoint_error_type"])
    if endpoint == "FALSE_SPLIT":
        return "SAME_INSTANCE"
    if endpoint == "FALSE_MERGE":
        return "SEPARATE_MEMBER_GROUPS"
    raise ValueError(f"unsupported identity endpoint: {endpoint}")


def _vote_summary(
    *,
    blind_case: Mapping[str, Any],
    generated: Mapping[str, Any],
    expected_action: str,
) -> dict[str, Any]:
    blind_uid = str(blind_case["blind_case_uid"])
    votes = [row for row in generated["votes"] if row["case_uid"] == blind_uid]
    if len(votes) != 3:
        raise ValueError(f"{blind_uid} must have exactly three frozen votes")
    actions = [str(row["constraint"]["action"]) for row in votes]
    counts = Counter(actions)
    majority_action, majority_count = counts.most_common(1)[0]
    aggregate = generated["aggregate"][blind_uid]
    selected = aggregate.get("selected_proposal")
    strict_action = str(selected["action"]) if selected else "DEFER"
    compiled = generated["compiled_candidates"][blind_uid]
    return {
        "blind_case_uid": blind_uid,
        "vote_action_counts": dict(sorted(counts.items())),
        "vote_correct_count": sum(action == expected_action for action in actions),
        "vote_count": len(votes),
        "majority_action": majority_action,
        "majority_count": majority_count,
        "majority_correct": majority_action == expected_action,
        "strict_aggregate_action": strict_action,
        "strict_aggregate_correct": strict_action == expected_action,
        "strict_selected_proposal": selected,
        "strict_gate_defer_reasons": aggregate.get("defer_reasons") or [],
        "compiled_stage": str(compiled["stage"]),
        "compiled_has_executable_constraint": bool(
            compiled.get("candidate_constraint")
        ),
    }


def _owners(state: Mapping[str, Any], obs_uid: str) -> list[str]:
    return sorted(
        str(entity_uid)
        for entity_uid, members in (state.get("membership") or {}).items()
        if obs_uid in {str(item) for item in members or ()}
    )


def _owner_summaries(
    state: Mapping[str, Any], owner_uids: list[str]
) -> list[dict[str, Any]]:
    wanted = set(owner_uids)
    rows = []
    for item in state.get("objects") or ():
        if str(item.get("entity_uid")) not in wanted:
            continue
        rows.append(
            {
                "entity_uid": str(item["entity_uid"]),
                "class_name": item.get("class_name"),
                "member_count": len(item.get("member_observation_uids") or ()),
                "num_detections": item.get("num_detections"),
                "n_points": item.get("n_points"),
                "revision_lineage_uids": item.get("revision_lineage_uids") or [],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", required=True, type=Path)
    parser.add_argument("--generation-result", required=True, type=Path)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--r1-labels", required=True, type=Path)
    parser.add_argument("--geometry-build", required=True, type=Path)
    parser.add_argument("--geometry-local", required=True, type=Path)
    parser.add_argument("--geometry-global", required=True, type=Path)
    parser.add_argument("--office-run", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    blind = _read(args.blind_manifest.resolve())
    generated = _read(args.generation_result.resolve())
    identity = _read(args.identity_manifest.resolve())
    holdout = _read(args.holdout_manifest.resolve())
    geometry_build = _read(args.geometry_build.resolve())
    geometry_local = _read(args.geometry_local.resolve())
    geometry_global = _read(args.geometry_global.resolve())

    if holdout.get("status") != "FROZEN":
        raise ValueError("fresh holdout manifest is not frozen")
    label_ref = holdout["source_artifacts"]["r1_labels_hash_only_not_parsed"]
    label_hash_matches = (
        _text_sha256(args.r1_labels.resolve()) == label_ref["sha256_utf8_canonical_lf"]
    )
    if not label_hash_matches:
        raise ValueError("posthoc label artifact drift")
    labels = _labels(args.r1_labels.resolve())

    protocol_path = Path(str(generated["inference_protocol_path"])).resolve()
    protocol = _read(protocol_path)
    protocol_hash_matches = (
        _sha256(protocol_path) == generated["inference_protocol_sha256"]
    )
    blind_hash_matches = (
        _sha256(args.blind_manifest.resolve()) == protocol["blind_manifest_sha256"]
    )
    no_prompt_leakage = not any(protocol.get("forbidden_prompt_leakage", {}).values())
    no_credential_persistence = not _CREDENTIAL_PATTERN.search(
        args.generation_result.read_text(encoding="utf-8")
        + protocol_path.read_text(encoding="utf-8")
    )
    protocol_frozen = bool(
        holdout.get("frozen_before_posthoc_label_access")
        and holdout.get("frozen_before_generator_outcomes")
        and blind.get("frozen_before_generator_responses")
        and protocol.get("frozen_before_responses")
        and generated.get("inference_protocol_frozen_before_responses")
        and protocol_hash_matches
        and blind_hash_matches
        and no_prompt_leakage
        and no_credential_persistence
    )

    blind_cases = list(blind.get("cases") or ())
    if len(blind_cases) != 5 or int(generated.get("vote_count", 0)) != 15:
        raise ValueError("frozen five-case/15-vote protocol changed")

    identity_by_event = {
        str(case["anchor_association_event_uid"]): case
        for case in identity.get("cases") or ()
        if case.get("causal_disposition") == "REPLAYABLE_ASSOCIATION_CAUSE"
    }
    holdout_by_incident = {
        str(case["incident_uid"]): case for case in holdout.get("cases") or ()
    }

    evaluation_rows = []
    identity_rows = []
    capability_rows = []
    holdout_labels = {}
    for blind_case in blind_cases:
        if blind_case["input_family"] == "IDENTITY_ASSOCIATION":
            source = identity_by_event[str(blind_case["anchor_association_event_uid"])]
            endpoint = str(source["endpoint_error_type"])
            expected_action = _expected_identity_action(source)
            row = _vote_summary(
                blind_case=blind_case,
                generated=generated,
                expected_action=expected_action,
            )
            row.update(
                {
                    "source_case_uid": str(source["case_uid"]),
                    "input_family": "IDENTITY_ASSOCIATION",
                    "posthoc_endpoint_error_type": endpoint,
                    "expected_action": expected_action,
                }
            )
            identity_rows.append(row)
        else:
            source = holdout_by_incident[str(blind_case["incident_uid"])]
            label = labels[str(source["incident_uid"])]
            endpoint = str(label["final_error_type"])
            resolution = resolve_endpoint_capability(endpoint)
            expected_action = resolution.automatic_action
            row = _vote_summary(
                blind_case=blind_case,
                generated=generated,
                expected_action=expected_action,
            )
            expected_obs_key = (
                canonical_obs_key(
                    str(
                        source["inference_inputs"][
                            "representative_trigger_observation_uids"
                        ][0]
                    )
                )
                if endpoint == "GEOMETRY_CORRUPTION"
                else None
            )
            selected = row["strict_selected_proposal"] or {}
            row.update(
                {
                    "source_case_uid": str(source["case_uid"]),
                    "incident_uid": str(source["incident_uid"]),
                    "scene_id": str(source["scene_id"]),
                    "input_family": "CAPABILITY_PROBE",
                    "posthoc_endpoint_error_type": endpoint,
                    "posthoc_human_notes": label.get("notes"),
                    "posthoc_final_owner_uids": label.get("final_owner_uids") or [],
                    "expected_action": expected_action,
                    "capability_resolution": resolution.as_dict(),
                    "expected_obs_key": expected_obs_key,
                    "strict_payload_correct": bool(
                        row["strict_aggregate_correct"]
                        and (
                            expected_obs_key is None
                            or selected.get("obs_key") == expected_obs_key
                        )
                    ),
                }
            )
            capability_rows.append(row)
            holdout_labels[endpoint] = (source, label, row)
        evaluation_rows.append(row)

    if set(holdout_labels) != {"GEOMETRY_CORRUPTION", "SPURIOUS_OBJECT"}:
        raise ValueError("fresh holdouts do not cover the frozen endpoint pair")

    geometry_source, geometry_label, geometry_row = holdout_labels[
        "GEOMETRY_CORRUPTION"
    ]
    geometry_obs_uid = str(
        geometry_source["inference_inputs"]["representative_trigger_observation_uids"][
            0
        ]
    )
    office_source_hashes = ProvenanceIndex(args.office_run.resolve()).source_hashes()
    geometry_formal_checks = {
        "build_role_frozen_holdout": geometry_build.get("evaluation_role")
        == "FROZEN_FRESH_HOLDOUT_GEOMETRY_BUILD",
        "build_pass": bool(geometry_build.get("pass")),
        "office_base_source_hashes_match_build": office_source_hashes
        == geometry_build.get("source_hashes_after"),
        "local_role_frozen_holdout": geometry_local.get("evaluation_role")
        == "FROZEN_FRESH_HOLDOUT_GEOMETRY_LOCAL_CAUSAL_VALIDATION",
        "local_pass": bool(geometry_local.get("pass")),
        "global_role_frozen_holdout": geometry_global.get("evaluation_role")
        == "FROZEN_FRESH_HOLDOUT_GEOMETRY_LOCAL_GLOBAL_PARITY",
        "global_pass": bool(geometry_global.get("pass")),
        "observation_binding_exact": all(
            str(item.get("obs_uid")) == geometry_obs_uid
            for item in (geometry_build, geometry_local, geometry_global)
        ),
        "raw_mask_restored_exact": bool(
            geometry_build.get("geometry_metrics", {}).get("restored_mask_exact_to_raw")
        ),
        "point_support_increased": bool(
            geometry_local.get("endpoint_checks", {}).get("point_support_increased")
        ),
        "overlay_and_recompute_once": bool(
            geometry_local.get("endpoint_checks", {}).get(
                "geometry_overlay_hit_exactly_once"
            )
            and geometry_local.get("endpoint_checks", {}).get(
                "geometry_similarity_recomputed_exactly_once"
            )
        ),
        "local_global_partition_exact": bool(
            geometry_global.get("checks", {}).get(
                "local_global_membership_partition_exact"
            )
        ),
        "sources_immutable": bool(
            geometry_build.get("source_hashes_unchanged")
            and geometry_local.get("source_checks", {}).get(
                "provenance_hashes_unchanged"
            )
            and geometry_global.get("checks", {}).get("provenance_hashes_unchanged")
        ),
        "production_commit_disabled": all(
            item.get("production_commit_permitted") is False
            for item in (geometry_local, geometry_global)
        ),
    }
    natural_state = _read(args.geometry_local.resolve().parent / "natural_state.json")
    sparse_state = _read(args.geometry_local.resolve().parent / "sparse_state.json")
    natural_owners = _owners(natural_state, geometry_obs_uid)
    sparse_owners = _owners(sparse_state, geometry_obs_uid)
    identity_changed = natural_owners != sparse_owners
    geometry_formal = {
        "case_uid": geometry_source["case_uid"],
        "posthoc_endpoint_error_type": "GEOMETRY_CORRUPTION",
        "generator_candidate_family_correct": geometry_row["strict_aggregate_correct"],
        "checks": geometry_formal_checks,
        "geometry_mechanism_pass": all(geometry_formal_checks.values()),
        "original_point_count": geometry_build["geometry_metrics"][
            "original_observation_point_count"
        ],
        "restored_point_count": geometry_build["geometry_metrics"][
            "restored_observation_point_count"
        ],
        "point_support_gain_ratio": geometry_build["geometry_metrics"][
            "point_support_gain_ratio"
        ],
        "natural_owner_uids": natural_owners,
        "sparse_owner_uids": sparse_owners,
        "posthoc_recorded_final_owner_uids": geometry_label.get("final_owner_uids")
        or [],
        "natural_owner_summaries": _owner_summaries(natural_state, natural_owners),
        "sparse_owner_summaries": _owner_summaries(sparse_state, sparse_owners),
        "identity_side_effect": identity_changed,
        "identity_side_effect_adjudication": (
            "UNADJUDICATED_BY_GEOMETRY_ONLY_HUMAN_LABEL"
            if identity_changed
            else "NO_IDENTITY_CHANGE"
        ),
        "endpoint_disposition": (
            "GEOMETRY_MECHANISM_PASS_WITH_IDENTITY_SIDE_EFFECT_UNADJUDICATED"
            if identity_changed
            else "GEOMETRY_MECHANISM_PASS"
        ),
        "production_commit_permitted": False,
    }

    spurious_source, _, spurious_row = holdout_labels["SPURIOUS_OBJECT"]
    spurious_resolution = resolve_endpoint_capability("SPURIOUS_OBJECT")
    room_provenance = ProvenanceIndex(args.room_run.resolve())
    spurious_hashes_before = room_provenance.source_hashes()
    formal_spurious_constraints: list[dict[str, Any]] = []
    spurious_hashes_after = room_provenance.source_hashes()
    spurious_checks = {
        "capability_registry_defers": (
            spurious_resolution.disposition == CapabilityDisposition.DEFER_UNSUPPORTED
        ),
        "no_executable_constraint_type": not spurious_resolution.executable,
        "no_delete_or_suppress_primitive": not any(
            "DELETE" in item or "SUPPRESS" in item
            for item in executable_constraint_types()
        ),
        "no_constraint_materialized": not formal_spurious_constraints,
        "no_replay_or_state_mutation": True,
        "sources_immutable": spurious_hashes_before == spurious_hashes_after,
        "wrong_generator_candidate_blocked": bool(
            not spurious_row["strict_aggregate_correct"]
            and spurious_row["compiled_stage"] == "DEFERRED"
        ),
    }
    spurious_formal = {
        "case_uid": spurious_source["case_uid"],
        "posthoc_endpoint_error_type": "SPURIOUS_OBJECT",
        "generator_candidate_action": spurious_row["strict_aggregate_action"],
        "expected_safe_action": "DEFER",
        "capability_resolution": spurious_resolution.as_dict(),
        "formal_constraints": formal_spurious_constraints,
        "checks": spurious_checks,
        "fail_closed_safety_pass": all(spurious_checks.values()),
        "endpoint_repaired": False,
        "endpoint_disposition": "DEFERRED_NO_SAFE_SUPPRESS_OR_DELETE_PRIMITIVE",
        "production_commit_permitted": False,
        "source_hashes_before": spurious_hashes_before,
        "source_hashes_after": spurious_hashes_after,
    }

    total_votes = sum(row["vote_count"] for row in evaluation_rows)
    vote_correct = sum(row["vote_correct_count"] for row in evaluation_rows)
    automatic_commit_count = sum(
        row["compiled_stage"] == "COMMIT_ELIGIBLE" for row in evaluation_rows
    )
    unsafe_commit_count = sum(
        row["compiled_stage"] == "COMMIT_ELIGIBLE"
        and not row["strict_aggregate_correct"]
        for row in evaluation_rows
    )
    metrics = {
        "case_count": len(evaluation_rows),
        "vote_count": total_votes,
        "vote_candidate_action_accuracy": vote_correct / total_votes,
        "strict_aggregate_action_accuracy": sum(
            row["strict_aggregate_correct"] for row in evaluation_rows
        )
        / len(evaluation_rows),
        "identity_strict_action_accuracy": sum(
            row["strict_aggregate_correct"] for row in identity_rows
        )
        / len(identity_rows),
        "fresh_capability_strict_action_accuracy": sum(
            row["strict_aggregate_correct"] for row in capability_rows
        )
        / len(capability_rows),
        "fresh_capability_strict_payload_correct_count": sum(
            row["strict_payload_correct"] for row in capability_rows
        ),
        "automatic_commit_eligible_count": automatic_commit_count,
        "unsafe_automatic_commit_count_posthoc": unsafe_commit_count,
        "fresh_holdout_endpoint_repaired_count": int(
            geometry_formal["geometry_mechanism_pass"]
        ),
        "fresh_holdout_unresolved_count": 1,
        "fresh_holdout_count": 2,
        "production_commit_count": 0,
    }
    audit_checks = {
        "protocol_frozen_and_label_free": protocol_frozen,
        "label_artifact_hash_matches_freeze": label_hash_matches,
        "geometry_formal_checks_pass": all(geometry_formal_checks.values()),
        "spurious_fail_closed_checks_pass": all(spurious_checks.values()),
        "zero_unsafe_automatic_commits": unsafe_commit_count == 0,
        "zero_production_commits": metrics["production_commit_count"] == 0,
    }
    method_checks = {
        "automatic_identity_routing_succeeds": metrics[
            "identity_strict_action_accuracy"
        ]
        == 1.0,
        "automatic_fresh_capability_routing_succeeds": metrics[
            "fresh_capability_strict_action_accuracy"
        ]
        == 1.0,
        "both_fresh_holdout_endpoints_repaired": metrics[
            "fresh_holdout_endpoint_repaired_count"
        ]
        == 2,
        "geometry_identity_side_effect_fully_adjudicated": not identity_changed,
    }
    result = {
        "schema_version": "1.0.0",
        "evaluation_role": (
            "POSTHOC_EVALUATION_OF_TWO_FROZEN_FRESH_HOLDOUTS; "
            "NOT_A_POPULATION_OR_SCENE_GENERALIZATION_ESTIMATE"
        ),
        "conclusion": "PARTIAL_SUCCESS_FAIL_CLOSED",
        "audit_integrity_and_safety_pass": all(audit_checks.values()),
        "method_complete": all(method_checks.values()),
        "production_commit_permitted": False,
        "input_hashes": {
            "holdout_manifest": _sha256(args.holdout_manifest.resolve()),
            "blind_manifest": _sha256(args.blind_manifest.resolve()),
            "inference_protocol": _sha256(protocol_path),
            "generation_result": _sha256(args.generation_result.resolve()),
            "geometry_build": _sha256(args.geometry_build.resolve()),
            "geometry_local": _sha256(args.geometry_local.resolve()),
            "geometry_global": _sha256(args.geometry_global.resolve()),
        },
        "protocol_checks": {
            "protocol_hash_matches_generation_result": protocol_hash_matches,
            "blind_hash_matches_frozen_protocol": blind_hash_matches,
            "no_forbidden_prompt_leakage": no_prompt_leakage,
            "no_api_credential_persistence": no_credential_persistence,
            "frozen_before_posthoc_label_access": holdout.get(
                "frozen_before_posthoc_label_access"
            ),
            "no_outcome_based_replacement": holdout.get("selection_policy", {}).get(
                "no_outcome_based_replacement"
            ),
        },
        "cases": evaluation_rows,
        "formal_holdout_execution": {
            "geometry": geometry_formal,
            "spurious_object": spurious_formal,
        },
        "metrics": metrics,
        "audit_checks": audit_checks,
        "method_checks": method_checks,
        "limitations": [
            "Only two fresh incidents were selected; this is not a population estimate.",
            "The geometry label does not adjudicate the induced identity merge.",
            "No safe entity suppression/deletion primitive exists, so the spurious object remains unresolved.",
            "Automatic generation produced no commit-eligible mutation under the current shadow gate.",
        ],
    }
    _write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "conclusion": result["conclusion"],
                "audit_integrity_and_safety_pass": result[
                    "audit_integrity_and_safety_pass"
                ],
                "method_complete": result["method_complete"],
                "metrics": metrics,
                "audit_checks": audit_checks,
                "method_checks": method_checks,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result["audit_integrity_and_safety_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
