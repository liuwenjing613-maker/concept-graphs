#!/usr/bin/env python3
"""Evaluate finite identity hypotheses with an independent development shadow replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from conceptgraph.revision.auto_constraints import (
    GeneratorStage,
    IncidentBinding,
    ShadowGateEvidence,
    canonicalize_vote,
    decide_automatic_promotion,
    enumerate_identity_hypotheses,
    semantic_constraint_fingerprint,
)
from conceptgraph.revision.benchmark.human_error_pilot import (
    HumanSceneContext,
    _affected_native_observations,
    _mechanism_trace,
    _resolve_groups,
    _validate_causal_case,
    evaluate_collateral,
    evaluate_endpoint_groups,
)
from conceptgraph.revision.constraints import ReplayMode, SparseRepairConstraint
from conceptgraph.revision.runtime_verify import InvariantVerifier
from conceptgraph.revision.snapshot import AnchorStateBuilder


EVALUATION_ROLE = "DEVELOPMENT_HUMAN_ENDPOINT_SHADOW_NOT_PRODUCTION_PROMOTION"


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain one JSON object")
    return value


def _write(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_case(manifest: Mapping[str, Any], case_uid: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in manifest.get("cases") or ()
        if str(row.get("case_uid")) == str(case_uid)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one human case {case_uid}, found {len(matches)}")
    return matches[0]


def _valid_parity(path: Path, case_uid: str) -> tuple[bool, dict[str, Any]]:
    value = _read(path)
    checks = value.get("checks") or {}
    passed = bool(
        value.get("case_uid") == case_uid
        and value.get("pass")
        and len(checks) == 9
        and all(bool(item) for item in checks.values())
    )
    return passed, value


def _component_supported(
    aggregate: Mapping[str, Any], case_uid: str, endpoint_error_type: str
) -> bool:
    rows = [
        row
        for row in aggregate.get("cases") or ()
        if str(row.get("case_uid")) == case_uid
    ]
    if len(rows) != 1:
        return False
    row = rows[0]
    variants = {
        str(item.get("variant")): bool(item.get("endpoint_correct"))
        for item in row.get("variants") or ()
    }
    if endpoint_error_type == "FALSE_SPLIT":
        causal_ablation = variants.get("redirect_off") is False
    else:
        causal_ablation = (
            variants.get("postprocess_off") is False
            and variants.get("both_boundaries_off") is False
        )
    return bool(row.get("full_endpoint_correct") and causal_ablation)


def _vote_counts(
    generation: Mapping[str, Any], blind_case_uid: str
) -> tuple[dict[str, int], list[str]]:
    counts: Counter[str] = Counter()
    errors = []
    for index, row in enumerate(generation.get("votes") or ()):
        if str(row.get("case_uid")) != blind_case_uid:
            continue
        try:
            proposal = canonicalize_vote(row.get("constraint", row))
            counts[str(proposal["action"])] += 1
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"vote_{index}:{type(exc).__name__}:{exc}")
    return dict(sorted(counts.items())), errors


def _state_audit(
    state: Mapping[str, Any], primitive: SparseRepairConstraint
) -> dict[str, Any]:
    anchor_decisions = [
        row
        for row in state.get("decision_trace") or ()
        if str(row.get("obs_uid")) == primitive.obs_uid
    ]
    relevant_postprocess = [
        row
        for row in state.get("postprocess_decision_trace") or ()
        if "persistent_create_instance_boundary"
        in set(str(item) for item in row.get("reject_reasons") or ())
    ]
    return {
        "active_object_count": len(state.get("membership") or {}),
        "runtime_ms": state.get("runtime_ms"),
        "timing": state.get("timing"),
        "constraint_hit_count": state.get("constraint_hit_count"),
        "persistent_create_instance_merge_veto_count": state.get(
            "persistent_create_instance_merge_veto_count", 0
        ),
        "persistent_create_instance_association_veto_count": state.get(
            "persistent_create_instance_association_veto_count", 0
        ),
        "persistent_lineage_redirect_override_count": state.get(
            "persistent_lineage_redirect_override_count", 0
        ),
        "identity_boundaries": state.get("identity_boundaries") or [],
        "anchor_decision_trace": anchor_decisions,
        "relevant_postprocess_decision_trace": relevant_postprocess,
    }


def _evaluate_case(
    *,
    context: HumanSceneContext,
    case: Mapping[str, Any],
    binding: IncidentBinding,
    vote_counts: Mapping[str, int],
    no_op_pass: bool,
    legal_merge_pass: bool,
    component_aggregate: Mapping[str, Any],
    parity_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    case_uid = str(case["case_uid"])
    provenance = context.provenance
    causal = _validate_causal_case(provenance, case)
    if not causal.get("pass"):
        raise ValueError(f"causal validation failed for {case_uid}")

    gold = SparseRepairConstraint.from_mapping(case["constraints"][0])
    gold_fingerprint = semantic_constraint_fingerprint(gold.as_dict())
    hypotheses = enumerate_identity_hypotheses(
        binding, candidate_aliases=["CANDIDATE_1_CONTEXT"]
    )
    if len(hypotheses) != 2:
        raise AssertionError(f"{case_uid} did not produce two finite hypotheses")

    groups = _resolve_groups(provenance, case["evaluation"]["groups"])
    affected = _affected_native_observations(provenance, case)
    seeds = [str(item) for item in case.get("snapshot_seed_version_uids") or ()]
    closure = context.dependency_graph.forward_closure(
        anchor_event_uid=str(case["anchor_association_event_uid"]),
        seed_version_uids=seeds,
    )
    prefix_state, prefix_objects = context.prefix_cache.prefix_before(
        int(case["frame_idx"])
    )
    snapshot = AnchorStateBuilder(provenance, context.engine).build_pre_anchor_state(
        str(case["anchor_association_event_uid"]),
        seeds,
        strict=True,
        prefix_state=prefix_state,
        prefix_objects=prefix_objects,
    )
    native = copy.deepcopy(context.native_state)
    probes = [str(item) for item in case["evaluation"].get("probe_obs_uids") or ()]
    desired = str(case["evaluation"]["desired_owner_relation"])
    native_endpoint = evaluate_endpoint_groups(
        native["membership"], groups, desired, probes=probes
    )
    parity_pass, parity = _valid_parity(parity_path, case_uid)
    component_pass = _component_supported(
        component_aggregate, case_uid, str(case["endpoint_error_type"])
    )

    candidate_rows = []
    verifier = InvariantVerifier()
    for index, compiled in enumerate(hypotheses):
        primitive = SparseRepairConstraint.from_mapping(
            compiled["candidate_constraint"]
        )
        started = time.perf_counter()
        state = context.engine.replay_local_from_snapshot(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            snapshot_objects=snapshot.objects,
            snapshot_runtime_ms=snapshot.state["runtime_ms"],
            snapshot_timing=snapshot.state.get("timing"),
            anchor_frame=snapshot.anchor_frame,
            snapshot_watermark_event_sequence=snapshot.watermark_event_sequence,
            closure=closure,
            constraints=[primitive],
            current_state=native,
        )
        replay_wall_ms = (time.perf_counter() - started) * 1000.0
        endpoint = evaluate_endpoint_groups(
            state["membership"], groups, desired, probes=probes
        )
        collateral = evaluate_collateral(
            native["membership"], state["membership"], affected
        )
        verification = verifier.verify(
            state=state,
            constraints=[primitive],
            source_hashes_before=context.source_hashes_before,
            source_hashes_after=provenance.source_hashes(),
            known_observation_uids=provenance.observations,
        )
        mechanism = _mechanism_trace(state, primitive)
        source_immutable = context.source_hashes_before == provenance.source_hashes()
        exact_gold_semantics = compiled["constraint_fingerprint"] == gold_fingerprint
        same_constraint_parity = bool(exact_gold_semantics and parity_pass)
        endpoint_improved = bool(not native_endpoint["correct"] and endpoint["correct"])
        invariant_gate = bool(
            verification["pass"]
            and snapshot.validation["pass"]
            and mechanism["verified"]
        )
        artifact_refs = [
            str(parity_path),
            str(output_root / case_uid / f"hypothesis_{index}.json"),
        ]
        evidence = ShadowGateEvidence(
            constraint_fingerprint=str(compiled["constraint_fingerprint"]),
            endpoint_improved=endpoint_improved,
            collateral_safe=bool(collateral["safe"]),
            invariants_pass=invariant_gate,
            source_immutable=source_immutable,
            no_op_controls_pass=no_op_pass,
            legal_merge_control_pass=legal_merge_pass,
            component_mechanism_supported=bool(
                component_pass and mechanism["verified"]
            ),
            local_global_parity_pass=same_constraint_parity,
            evaluation_independent_of_generator=True,
            artifact_refs=tuple(artifact_refs),
        )
        promotion = decide_automatic_promotion(compiled, evidence)
        row = {
            "hypothesis_index": index,
            "hypothesis_action": compiled["hypothesis_action"],
            "hypothesis_target_alias": compiled["hypothesis_target_alias"],
            "model_vote_support": int(
                vote_counts.get(str(compiled["hypothesis_action"]), 0)
            ),
            "constraint_fingerprint": compiled["constraint_fingerprint"],
            "gold_constraint_fingerprint": gold_fingerprint,
            "exact_gold_semantics": exact_gold_semantics,
            "candidate_constraint": compiled["candidate_constraint"],
            "native_endpoint": native_endpoint,
            "candidate_endpoint": endpoint,
            "endpoint_improved": endpoint_improved,
            "collateral": collateral,
            "runtime_invariants": verification,
            "snapshot_validation": snapshot.validation,
            "mechanism": mechanism,
            "source_immutable": source_immutable,
            "component_ablation_supported": component_pass,
            "same_constraint_prior_parity_reused": same_constraint_parity,
            "prior_parity_checks": parity.get("checks"),
            "shadow_gate_evidence": evidence.as_dict(),
            "development_shadow_decision": promotion,
            "local_replay_wall_ms": replay_wall_ms,
            "state_audit": _state_audit(state, primitive),
        }
        candidate_rows.append(row)
        _write(artifact_refs[1], row)

    eligible = [
        row
        for row in candidate_rows
        if row["development_shadow_decision"]["stage"]
        == GeneratorStage.COMMIT_ELIGIBLE.value
    ]
    selected = eligible[0] if len(eligible) == 1 else None
    result = {
        "case_uid": case_uid,
        "blind_case_uid": binding.case_uid,
        "scene_id": case["scene_id"],
        "endpoint_error_type": case["endpoint_error_type"],
        "evaluation_role": EVALUATION_ROLE,
        "human_endpoint_gold_used_only_by_shadow_evaluator": True,
        "production_commit_permitted": False,
        "candidate_target_alias_recall": any(
            row["exact_gold_semantics"] for row in candidate_rows
        ),
        "model_vote_counts": dict(vote_counts),
        "native_endpoint_correct": bool(native_endpoint["correct"]),
        "hypothesis_count": len(candidate_rows),
        "commit_eligible_hypothesis_count": len(eligible),
        "unique_development_shadow_pass": len(eligible) == 1,
        "selected_action": selected["hypothesis_action"] if selected else None,
        "selected_matches_gold": bool(selected and selected["exact_gold_semantics"]),
        "all_wrong_hypotheses_rejected": all(
            row["development_shadow_decision"]["stage"]
            != GeneratorStage.COMMIT_ELIGIBLE.value
            for row in candidate_rows
            if not row["exact_gold_semantics"]
        ),
        "candidate_results": [
            {
                key: value
                for key, value in row.items()
                if key not in {"state_audit", "mechanism"}
            }
            for row in candidate_rows
        ],
        "artifact_paths": [
            str(output_root / case_uid / f"hypothesis_{index}.json")
            for index in range(len(candidate_rows))
        ],
    }
    _write(output_root / case_uid / "case_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-manifest", required=True, type=Path)
    parser.add_argument("--evidence-manifest", required=True, type=Path)
    parser.add_argument("--generation-result", required=True, type=Path)
    parser.add_argument("--office-base-run", required=True, type=Path)
    parser.add_argument("--room-base-run", required=True, type=Path)
    parser.add_argument("--negative-controls", required=True, type=Path)
    parser.add_argument("--legal-merge-result", required=True, type=Path)
    parser.add_argument("--component-ablation", required=True, type=Path)
    parser.add_argument("--office-parity", required=True, type=Path)
    parser.add_argument("--room-outlet-parity", required=True, type=Path)
    parser.add_argument("--room-table-parity", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    result_path = output_root / "identity_shadow_search_result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")

    human_manifest = _read(args.human_manifest)
    evidence_manifest = _read(args.evidence_manifest)
    generation = _read(args.generation_result)
    no_op = _read(args.negative_controls)
    legal_merge = _read(args.legal_merge_result)
    component = _read(args.component_ablation)
    no_op_pass = bool(
        no_op.get("status") == "PASS"
        and int(no_op.get("case_count", -1)) == 2
        and int(no_op.get("pass_count", -1)) == 2
    )
    legal_merge_pass = bool(
        legal_merge.get("pass")
        and all(bool(value) for value in (legal_merge.get("checks") or {}).values())
    )
    if evidence_manifest.get("role") != "DEVELOPMENT_NOT_HOLDOUT":
        raise ValueError("identity evidence must be labeled development")
    if generation.get("gold_loaded_by_generator") is not False:
        raise ValueError("generator isolation audit failed")

    cases = list(evidence_manifest.get("cases") or ())
    if len(cases) != 3:
        raise ValueError("development shadow requires exactly three frozen cases")
    parity_paths = {
        "human6_office0_false_split_51aaf9ba": args.office_parity.resolve(),
        "human6_room0_false_merge_06525b4b": args.room_outlet_parity.resolve(),
        "human6_room0_false_merge_9727f850": args.room_table_parity.resolve(),
    }
    base_runs = {
        "office0": args.office_base_run.resolve(),
        "room0": args.room_base_run.resolve(),
    }
    contexts = {
        scene_id: HumanSceneContext.build(scene_id, base_run)
        for scene_id, base_run in base_runs.items()
    }

    protocol = {
        "schema_version": "1.0.0",
        "evaluation_role": EVALUATION_ROLE,
        "frozen_input_hashes": {
            "human_manifest": _sha256(args.human_manifest),
            "evidence_manifest": _sha256(args.evidence_manifest),
            "generation_result": _sha256(args.generation_result),
            "negative_controls": _sha256(args.negative_controls),
            "legal_merge_result": _sha256(args.legal_merge_result),
            "component_ablation": _sha256(args.component_ablation),
            **{
                f"parity_{case_uid}": _sha256(path)
                for case_uid, path in parity_paths.items()
            },
        },
        "model_role": "FINITE_CANDIDATE_RANKER_ONLY",
        "same_and_separate_hypotheses_enumerated_before_endpoint_evaluation": True,
        "human_endpoint_gold_loaded_by_generator": False,
        "human_endpoint_gold_loaded_by_shadow_evaluator": True,
        "production_commit_permitted": False,
        "selection_rule": "EXACTLY_ONE_ALL_GATES_PASS_ELSE_DEFER",
        "case_count": 3,
    }
    _write(output_root / "shadow_protocol.frozen.json", protocol)

    results = []
    vote_parse_errors = {}
    for evidence_row in cases:
        case_uid = str(evidence_row["development_source_case_uid"])
        case = _find_case(human_manifest, case_uid)
        binding_path = Path(evidence_row["binding_private_path"]).resolve()
        if _sha256(binding_path) != str(evidence_row["binding_private_sha256"]):
            raise ValueError(f"binding drift: {binding_path}")
        binding = IncidentBinding.from_mapping(_read(binding_path))
        counts, errors = _vote_counts(generation, str(evidence_row["blind_case_uid"]))
        vote_parse_errors[str(evidence_row["blind_case_uid"])] = errors
        results.append(
            _evaluate_case(
                context=contexts[str(case["scene_id"])],
                case=case,
                binding=binding,
                vote_counts=counts,
                no_op_pass=no_op_pass,
                legal_merge_pass=legal_merge_pass,
                component_aggregate=component,
                parity_path=parity_paths[case_uid],
                output_root=output_root,
            )
        )

    unanimous_wrong_rejections = []
    for result in results:
        unanimous = [
            action
            for action, count in result["model_vote_counts"].items()
            if int(count) == 5
        ]
        if not unanimous:
            continue
        action = unanimous[0]
        matching = [
            row
            for row in result["candidate_results"]
            if row["hypothesis_action"] == action
        ]
        unanimous_wrong_rejections.append(
            {
                "case_uid": result["case_uid"],
                "action": action,
                "was_wrong": bool(matching and not matching[0]["exact_gold_semantics"]),
                "rejected": bool(
                    matching
                    and matching[0]["development_shadow_decision"]["stage"]
                    != GeneratorStage.COMMIT_ELIGIBLE.value
                ),
                "defer_reasons": (
                    matching[0]["development_shadow_decision"]["defer_reasons"]
                    if matching
                    else ["unmatched_action"]
                ),
            }
        )

    aggregate = {
        "schema_version": "1.0.0",
        "evaluation_role": EVALUATION_ROLE,
        "protocol_path": str(output_root / "shadow_protocol.frozen.json"),
        "production_commit_permitted": False,
        "development_case_count": len(results),
        "candidate_target_recall_count": sum(
            bool(row["candidate_target_alias_recall"]) for row in results
        ),
        "unique_development_shadow_pass_count": sum(
            bool(row["unique_development_shadow_pass"]) for row in results
        ),
        "selected_matches_gold_count": sum(
            bool(row["selected_matches_gold"]) for row in results
        ),
        "all_wrong_hypotheses_rejected_count": sum(
            bool(row["all_wrong_hypotheses_rejected"]) for row in results
        ),
        "unsafe_selection_count": sum(
            bool(
                row["unique_development_shadow_pass"]
                and not row["selected_matches_gold"]
            )
            for row in results
        ),
        "no_op_controls_pass": no_op_pass,
        "legal_merge_control_pass": legal_merge_pass,
        "vote_parse_errors": vote_parse_errors,
        "unanimous_model_hypothesis_checks": unanimous_wrong_rejections,
        "wrong_unanimous_model_hypothesis_rejected": bool(
            unanimous_wrong_rejections
            and all(
                row["was_wrong"] and row["rejected"]
                for row in unanimous_wrong_rejections
            )
        ),
        "cases": results,
    }
    aggregate["pass"] = bool(
        len(results) == 3
        and aggregate["candidate_target_recall_count"] == 3
        and aggregate["unique_development_shadow_pass_count"] == 3
        and aggregate["selected_matches_gold_count"] == 3
        and aggregate["all_wrong_hypotheses_rejected_count"] == 3
        and aggregate["unsafe_selection_count"] == 0
        and aggregate["no_op_controls_pass"]
        and aggregate["legal_merge_control_pass"]
        and aggregate["wrong_unanimous_model_hypothesis_rejected"]
        and not any(vote_parse_errors.values())
    )
    _write(result_path, aggregate)

    audit = {key: value for key, value in aggregate.items() if key not in {"cases"}}
    audit["case_summary"] = [
        {
            key: row[key]
            for key in (
                "case_uid",
                "blind_case_uid",
                "endpoint_error_type",
                "model_vote_counts",
                "candidate_target_alias_recall",
                "unique_development_shadow_pass",
                "selected_action",
                "selected_matches_gold",
                "all_wrong_hypotheses_rejected",
            )
        }
        for row in results
    ]
    if args.audit_output is not None:
        _write(args.audit_output.resolve(), audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if aggregate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
