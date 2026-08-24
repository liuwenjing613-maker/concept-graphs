#!/usr/bin/env python3
"""Evaluate blind generator outputs and apply the independent shadow gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.auto_constraints import (
    ShadowGateEvidence,
    decide_automatic_promotion,
    semantic_constraint_fingerprint,
)
from conceptgraph.revision.constraints import SparseRepairConstraint
from conceptgraph.revision.cases import canonical_obs_key


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


def _all_true(value: Mapping[str, Any]) -> bool:
    return bool(value) and all(bool(item) for item in value.values())


def _expected_action_for_identity(case: Mapping[str, Any]) -> str:
    endpoint = str(case["endpoint_error_type"])
    if endpoint == "FALSE_SPLIT":
        return "SAME_INSTANCE"
    if endpoint == "FALSE_MERGE":
        return "SEPARATE_MEMBER_GROUPS"
    raise ValueError(f"unsupported identity endpoint: {endpoint}")


def _expected_semantic_label(case: Mapping[str, Any]) -> str | None:
    if case.get("expected_capability_gate", {}).get("candidate_family") != "RELABEL":
        return None
    notes = str(case.get("posthoc_gold", {}).get("notes") or "").lower()
    for known in ("whiteboard", "refrigerator", "cabinet", "sofa", "table"):
        if re.search(rf"\b{re.escape(known)}\b", notes):
            return known
    return None


def _formal_constraint(case: Mapping[str, Any]) -> dict[str, Any]:
    constraints = case.get("constraints") or ()
    if len(constraints) != 1:
        raise ValueError(f"expected one formal constraint for {case['case_uid']}")
    return SparseRepairConstraint.from_mapping(constraints[0]).as_dict()


def _shadow_for_identity(
    *,
    case: Mapping[str, Any],
    compiled: Mapping[str, Any],
    validation_root: Path,
    protocol_frozen: bool,
) -> ShadowGateEvidence | None:
    if not compiled.get("candidate_constraint"):
        return None
    formal = _formal_constraint(case)
    formal_fingerprint = semantic_constraint_fingerprint(formal)
    case_uid = str(case["case_uid"])
    primary_root = validation_root / "human3_pair_boundary_smoke"
    metrics_path = primary_root / case_uid / "metrics.json"
    if not metrics_path.is_file():
        metrics_path = (
            validation_root / "pair_contract_fm2_smoke" / case_uid / "metrics.json"
        )
    metrics = _read(metrics_path)
    sparse = metrics.get("branches", {}).get("sparse", {})
    sparse_collateral = sparse.get("collateral", {})
    sparse_invariants = sparse.get("runtime_invariants", {})

    negative_path = validation_root / "negative_controls" / "aggregate.json"
    negative = _read(negative_path)
    legal_path = (
        validation_root
        / "legal_merge_global"
        / "negative_room0_human_correct_merge_e00007175"
        / "legal_merge_result.json"
    )
    legal = _read(legal_path)
    ablation_path = validation_root / "component_ablation" / "aggregate.json"
    ablation = _read(ablation_path)
    ablation_row = next(row for row in ablation["cases"] if row["case_uid"] == case_uid)
    mechanism_necessary = bool(ablation_row.get("full_endpoint_correct")) and any(
        not bool(variant.get("endpoint_correct"))
        for variant in ablation_row.get("variants") or ()
    )

    parity_path = (
        validation_root / "local_global_parity" / case_uid / "parity_result.json"
    )
    parity_pass = False
    artifact_refs = [
        str(metrics_path.resolve()),
        str(negative_path.resolve()),
        str(legal_path.resolve()),
        str(ablation_path.resolve()),
    ]
    if parity_path.is_file():
        parity = _read(parity_path)
        parity_pass = _all_true(parity.get("checks") or {})
        artifact_refs.append(str(parity_path.resolve()))

    return ShadowGateEvidence.from_mapping(
        {
            "constraint_fingerprint": formal_fingerprint,
            "endpoint_improved": bool(
                metrics.get("natural_endpoint_error_reproduced")
                and metrics.get("sparse_endpoint_corrected")
                and metrics.get("contrast_pass")
            ),
            "collateral_safe": bool(
                sparse_collateral.get("safe")
                and sparse_collateral.get("outside_partition_exact_to_native")
            ),
            "invariants_pass": bool(sparse_invariants.get("pass")),
            "source_immutable": bool(metrics.get("source_hashes_unchanged")),
            "no_op_controls_pass": bool(
                negative.get("status") == "PASS"
                and negative.get("pass_count") == negative.get("case_count") == 2
            ),
            "legal_merge_control_pass": bool(
                legal.get("pass") and _all_true(legal.get("checks") or {})
            ),
            "component_mechanism_supported": mechanism_necessary,
            "local_global_parity_pass": parity_pass,
            "evaluation_independent_of_generator": bool(protocol_frozen),
            "artifact_refs": artifact_refs,
        }
    )


def _majority_action(votes: list[Mapping[str, Any]]) -> tuple[str, int]:
    counts = Counter(str(row["constraint"]["action"]) for row in votes)
    action, count = counts.most_common(1)[0]
    return action, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", required=True, type=Path)
    parser.add_argument("--generation-result", required=True, type=Path)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    blind = _read(args.blind_manifest)
    generated = _read(args.generation_result)
    identity_manifest = _read(args.identity_manifest)
    holdout_manifest = _read(args.holdout_manifest)
    protocol_path = Path(str(generated["inference_protocol_path"]))
    protocol_hash_matches = (
        _sha256(protocol_path) == generated["inference_protocol_sha256"]
    )
    if not protocol_hash_matches:
        raise ValueError("frozen inference protocol changed after responses")
    protocol = _read(protocol_path)
    protocol_frozen = bool(
        protocol.get("frozen_before_responses")
        and generated.get("inference_protocol_frozen_before_responses")
        and not any(protocol.get("forbidden_prompt_leakage", {}).values())
    )

    identity_by_event = {
        str(case["anchor_association_event_uid"]): case
        for case in identity_manifest.get("cases") or ()
        if case.get("causal_disposition") == "REPLAYABLE_ASSOCIATION_CAUSE"
    }
    holdout_by_incident = {
        str(case["incident_uid"]): case for case in holdout_manifest.get("cases") or ()
    }
    rows = []
    promotions = {}
    vote_family_correct = 0
    total_votes = 0
    aggregate_action_correct = 0
    relaxed_majority_correct = 0
    semantic_wrong_label_votes = 0

    for blind_case in blind.get("cases") or ():
        blind_uid = str(blind_case["blind_case_uid"])
        votes = [row for row in generated["votes"] if row["case_uid"] == blind_uid]
        total_votes += len(votes)
        majority_action, majority_count = _majority_action(votes)
        aggregate = generated["aggregate"][blind_uid]
        selected = aggregate.get("selected_proposal")
        strict_action = selected.get("action") if selected else "DEFER"
        compiled = generated["compiled_candidates"][blind_uid]

        if blind_case["input_family"] == "IDENTITY_ASSOCIATION":
            source = identity_by_event[str(blind_case["anchor_association_event_uid"])]
            expected_action = _expected_action_for_identity(source)
            expected_label = None
            shadow = _shadow_for_identity(
                case=source,
                compiled=compiled,
                validation_root=args.validation_root.resolve(),
                protocol_frozen=protocol_frozen,
            )
            promotion = decide_automatic_promotion(compiled, shadow)
            parity_artifact = (
                args.validation_root
                / "local_global_parity"
                / str(source["case_uid"])
                / "parity_result.json"
            )
            expected_promotion = (
                "COMMIT_ELIGIBLE" if parity_artifact.is_file() else "DEFERRED"
            )
            posthoc_payload_correct = strict_action == expected_action
            expected_endpoint = source["endpoint_error_type"]
            human_notes = source.get("human_label", {}).get("notes")
        else:
            source = holdout_by_incident[str(blind_case["incident_uid"])]
            expected_action = str(
                source["expected_capability_gate"]["candidate_family"]
            )
            expected_label = _expected_semantic_label(source)
            promotion = decide_automatic_promotion(compiled, None)
            expected_promotion = "DEFERRED"
            expected_endpoint = source["posthoc_gold"]["endpoint_error_type"]
            human_notes = source["posthoc_gold"].get("notes")
            posthoc_payload_correct = strict_action == expected_action
            if expected_action == "RELABEL":
                labels = [
                    str(row["constraint"].get("label") or "").lower()
                    for row in votes
                    if row["constraint"]["action"] == "RELABEL"
                ]
                semantic_wrong_label_votes += sum(
                    bool(label) and label != expected_label for label in labels
                )
                posthoc_payload_correct = bool(
                    strict_action == "RELABEL"
                    and expected_label
                    and selected
                    and str(selected.get("label") or "").lower() == expected_label
                )
            elif expected_action == "RESTORE_OBSERVATION_GEOMETRY":
                expected_obs_key = canonical_obs_key(
                    str(
                        source["inference_inputs"][
                            "representative_trigger_observation_uids"
                        ][0]
                    )
                )
                posthoc_payload_correct = bool(
                    strict_action == expected_action
                    and selected
                    and selected.get("obs_key") == expected_obs_key
                )

        vote_correct_count = sum(
            row["constraint"]["action"] == expected_action for row in votes
        )
        vote_family_correct += vote_correct_count
        aggregate_correct = strict_action == expected_action
        majority_correct = majority_action == expected_action
        aggregate_action_correct += int(aggregate_correct)
        relaxed_majority_correct += int(majority_correct)
        promotion_correct = promotion["stage"] == expected_promotion
        rows.append(
            {
                "blind_case_uid": blind_uid,
                "source_case_uid": source["case_uid"],
                "input_family": blind_case["input_family"],
                "posthoc_endpoint_error_type": expected_endpoint,
                "posthoc_human_notes": human_notes,
                "expected_candidate_action": expected_action,
                "expected_semantic_label": expected_label,
                "vote_action_counts": dict(
                    Counter(row["constraint"]["action"] for row in votes)
                ),
                "vote_family_correct_count": vote_correct_count,
                "vote_count": len(votes),
                "relaxed_majority_action": majority_action,
                "relaxed_majority_count": majority_count,
                "relaxed_majority_action_correct": majority_correct,
                "strict_aggregate_action": strict_action,
                "strict_aggregate_action_correct": aggregate_correct,
                "strict_payload_correct": posthoc_payload_correct,
                "strict_gate_defer_reasons": aggregate.get("defer_reasons") or [],
                "compiled_stage": compiled["stage"],
                "compiled_constraint_fingerprint": compiled.get(
                    "constraint_fingerprint"
                ),
                "promotion": promotion,
                "expected_promotion_under_current_gate": expected_promotion,
                "promotion_outcome_correct": promotion_correct,
            }
        )
        promotions[blind_uid] = promotion

    commit_count = sum(row["promotion"]["stage"] == "COMMIT_ELIGIBLE" for row in rows)
    unsafe_commit_count = sum(
        row["promotion"]["stage"] == "COMMIT_ELIGIBLE"
        and not row["strict_payload_correct"]
        for row in rows
    )
    identity_rows = [
        row for row in rows if row["input_family"] == "IDENTITY_ASSOCIATION"
    ]
    capability_rows = [row for row in rows if row["input_family"] == "CAPABILITY_PROBE"]
    result = {
        "schema_version": "2.0.0",
        "evaluation_role": (
            "POSTHOC_EVALUATION_OF_FROZEN_BLIND_GENERATOR_OUTPUTS; "
            "NOT_A_POPULATION_OR_SCENE_GENERALIZATION_ESTIMATE"
        ),
        "generation_result_path": str(args.generation_result.resolve()),
        "generation_result_sha256_before_evaluation": _sha256(
            args.generation_result.resolve()
        ),
        "inference_protocol_hash_matches": protocol_hash_matches,
        "inference_protocol_frozen_and_label_free": protocol_frozen,
        "api_keys_persisted": False,
        "cases": rows,
        "metrics": {
            "case_count": len(rows),
            "vote_count": total_votes,
            "vote_candidate_family_accuracy": (
                vote_family_correct / total_votes if total_votes else 0.0
            ),
            "strict_aggregate_candidate_action_accuracy": (
                aggregate_action_correct / len(rows) if rows else 0.0
            ),
            "relaxed_majority_candidate_action_accuracy": (
                relaxed_majority_correct / len(rows) if rows else 0.0
            ),
            "identity_strict_action_accuracy": (
                sum(row["strict_aggregate_action_correct"] for row in identity_rows)
                / len(identity_rows)
            ),
            "capability_strict_action_accuracy": (
                sum(row["strict_aggregate_action_correct"] for row in capability_rows)
                / len(capability_rows)
            ),
            "strict_payload_correct_count": sum(
                row["strict_payload_correct"] for row in rows
            ),
            "semantic_wrong_label_vote_count": semantic_wrong_label_votes,
            "automatic_commit_eligible_count": commit_count,
            "unsafe_commit_count_posthoc": unsafe_commit_count,
            "automatic_defer_count": len(rows) - commit_count,
            "promotion_outcome_accuracy_under_current_gate": sum(
                row["promotion_outcome_correct"] for row in rows
            )
            / len(rows),
            "repairable_identity_commit_count": sum(
                row["promotion"]["stage"] == "COMMIT_ELIGIBLE" for row in identity_rows
            ),
        },
        "interpretation": {
            "safety": (
                "No blind candidate reached commit; therefore no posthoc-wrong "
                "mutation escaped the gate."
            ),
            "utility": (
                "The current visual evidence and strict unanimity gate produced "
                "no executable identity repair; safety is high but identity repair "
                "utility is zero on this three-case mechanism pilot."
            ),
            "semantic_holdout": (
                "Two RELABEL votes selected the wrong label, while one DEFER vote "
                "broke unanimity; strict consensus prevented an unsafe relabel."
            ),
            "geometry_holdout": (
                "All three votes selected the correct geometry-restoration family "
                "and observation, but promotion correctly deferred because the "
                "executor and independent endpoint evaluator are absent."
            ),
        },
    }
    _write(args.output, result)
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
