from __future__ import annotations

import pytest

from conceptgraph.revision.auto_constraints import (
    GeneratorStage,
    IncidentBinding,
    ShadowGateEvidence,
    aggregate_candidate_votes,
    canonicalize_vote,
    enumerate_identity_hypotheses,
    compile_blind_candidate,
    decide_automatic_promotion,
    semantic_constraint_fingerprint,
)


def _binding() -> IncidentBinding:
    return IncidentBinding.from_mapping(
        {
            "case_uid": "case_1",
            "obs_uid": "scene_run_f000010_r0002",
            "obs_key": "f000010_r0002",
            "event_uid": "event_10",
            "event_sequence": 10,
            "observed_current_decision": "CREATE",
            "created_entity_uid": "native-created",
            "created_identity_uid": "origin_scene_run_f000010_r0002",
            "evidence_refs": ["incident_blind", "I01", "I02"],
            "aliases": {
                "CANDIDATE_1_CONTEXT": {
                    "entity_uid": "target-object",
                    "lineage_uid": "origin_target",
                    "origin_obs_uid": "scene_run_f000001_r0001",
                    "identity_uids": ["origin_target"],
                    "provenance_lineage_uids": ["origin_target"],
                    "complete": True,
                }
            },
        }
    )


def _votes(action: str, **fields):
    return [
        {
            "constraint": {
                "action": action,
                "confidence": confidence,
                "evidence_image_ids": ["I01", "I02"],
                **fields,
            }
        }
        for confidence in (0.91, 0.93, 0.95)
    ]


def _aggregate(action: str, **fields):
    return aggregate_candidate_votes(
        _votes(action, **fields),
        allowed_evidence_ids={"I01", "I02"},
    )


def _shadow(fingerprint: str, **overrides) -> ShadowGateEvidence:
    value = {
        "constraint_fingerprint": fingerprint,
        "endpoint_improved": True,
        "collateral_safe": True,
        "invariants_pass": True,
        "source_immutable": True,
        "no_op_controls_pass": True,
        "legal_merge_control_pass": True,
        "component_mechanism_supported": True,
        "local_global_parity_pass": True,
        "evaluation_independent_of_generator": True,
        "artifact_refs": ["shadow.json", "negative.json"],
    }
    value.update(overrides)
    return ShadowGateEvidence.from_mapping(value)


def test_oracle_like_inference_field_is_rejected():
    with pytest.raises(ValueError, match="oracle-like"):
        canonicalize_vote(
            {
                "action": "DEFER",
                "confidence": 1.0,
                "posthoc_gold": {"expected_action": "SAME_INSTANCE"},
            }
        )


def test_vote_gate_requires_exact_structural_unanimity_and_valid_citations():
    aggregate = _aggregate(
        "SAME_INSTANCE",
        entities=["ANCHOR", "CANDIDATE_1_CONTEXT"],
    )
    assert aggregate["ready_for_binding"] is True
    assert aggregate["unanimous_structural_signature"] is True

    disagreeing = _votes(
        "SAME_INSTANCE",
        entities=["ANCHOR", "CANDIDATE_1_CONTEXT"],
    )
    disagreeing[-1]["constraint"]["entities"] = ["ANCHOR", "CANDIDATE_2_CONTEXT"]
    blocked = aggregate_candidate_votes(
        disagreeing, allowed_evidence_ids={"I01", "I02"}
    )
    assert blocked["ready_for_binding"] is False
    assert "structural_vote_disagreement" in blocked["defer_reasons"]

    bad_evidence = _votes(
        "SAME_INSTANCE",
        entities=["ANCHOR", "CANDIDATE_1_CONTEXT"],
    )
    bad_evidence[0]["constraint"]["evidence_image_ids"] = ["I99"]
    blocked = aggregate_candidate_votes(
        bad_evidence, allowed_evidence_ids={"I01", "I02"}
    )
    assert "evidence_citation_invalid" in blocked["defer_reasons"]


def test_same_instance_compiles_to_exact_identity_bound_assignment():
    aggregate = _aggregate(
        "SAME_INSTANCE",
        entities=["CANDIDATE_1_CONTEXT", "ANCHOR"],
    )
    compiled = compile_blind_candidate(aggregate, _binding())
    constraint = compiled["candidate_constraint"]
    assert compiled["stage"] == GeneratorStage.BOUND_PENDING_SHADOW.value
    assert constraint["type"] == "ASSIGN_OBSERVATION"
    assert constraint["target_entity_uid"] == "target-object"
    assert constraint["target_lineage_uid"] == "origin_target"
    assert constraint["target_origin_obs_uid"] == "scene_run_f000001_r0001"
    assert constraint["applies_at_event_uid"] == "event_10"


def test_separation_compiles_to_pair_specific_create_boundary():
    aggregate = _aggregate(
        "SEPARATE_MEMBER_GROUPS",
        groups=[["ANCHOR"], ["CANDIDATE_1_CONTEXT"]],
    )
    compiled = compile_blind_candidate(aggregate, _binding())
    constraint = compiled["candidate_constraint"]
    assert compiled["stage"] == GeneratorStage.BOUND_PENDING_SHADOW.value
    assert constraint["type"] == "CREATE_INSTANCE"
    assert constraint["created_entity_uid"] == "native-created"
    assert constraint["created_identity_uid"] == "origin_scene_run_f000010_r0002"
    assert constraint["separate_from_identity_uids"] == ["origin_target"]


def test_associate_separation_compiles_to_deterministic_new_identity():
    value = _binding().as_dict()
    value["observed_current_decision"] = "ASSOCIATE"
    value["created_entity_uid"] = None
    value["created_identity_uid"] = None
    binding = IncidentBinding.from_mapping(value)
    aggregate = _aggregate(
        "SEPARATE_MEMBER_GROUPS",
        groups=[["ANCHOR"], ["CANDIDATE_1_CONTEXT"]],
    )

    compiled = compile_blind_candidate(aggregate, binding)

    constraint = compiled["candidate_constraint"]
    assert compiled["stage"] == GeneratorStage.BOUND_PENDING_SHADOW.value
    assert constraint["type"] == "CREATE_INSTANCE"
    assert constraint["created_entity_uid"] is None
    assert constraint["created_identity_uid"] == (
        "revision-lineage:scene_run_f000010_r0002"
    )
    assert constraint["separate_from_identity_uids"] == ["origin_target"]


def test_identity_ambiguity_and_unsupported_families_fail_closed():
    value = _binding().as_dict()
    value["aliases"]["CANDIDATE_1_CONTEXT"]["identity_uids"] = [
        "origin_a",
        "origin_b",
    ]
    ambiguous = IncidentBinding.from_mapping(value)
    aggregate = _aggregate(
        "SAME_INSTANCE",
        entities=["ANCHOR", "CANDIDATE_1_CONTEXT"],
    )
    compiled = compile_blind_candidate(aggregate, ambiguous)
    assert compiled["stage"] == GeneratorStage.DEFERRED.value
    assert any(
        "non_unique_effective_identity" in reason
        for reason in compiled["defer_reasons"]
    )

    for action, fields in (
        ("RELABEL", {"entity_alias": "ANCHOR", "label": "whiteboard"}),
        (
            "RESTORE_OBSERVATION_GEOMETRY",
            {"obs_key": "f000010_r0002"},
        ),
        ("PARTITION_OBSERVATION", {"obs_key": "f000010_r0002"}),
    ):
        compiled = compile_blind_candidate(_aggregate(action, **fields), _binding())
        assert compiled["stage"] == GeneratorStage.DEFERRED.value
        assert compiled["candidate_constraint"] is None


def test_identity_hypothesis_enumeration_is_finite_complete_and_bound():
    hypotheses = enumerate_identity_hypotheses(
        _binding(), candidate_aliases=["CANDIDATE_1_CONTEXT"]
    )
    assert [item["hypothesis_action"] for item in hypotheses] == [
        "SAME_INSTANCE",
        "SEPARATE_MEMBER_GROUPS",
    ]
    assert all(
        item["stage"] == GeneratorStage.BOUND_PENDING_SHADOW.value
        for item in hypotheses
    )
    assert all(
        item["hypothesis_target_alias"] == "CANDIDATE_1_CONTEXT" for item in hypotheses
    )
    assert len({item["constraint_fingerprint"] for item in hypotheses}) == len(
        hypotheses
    )


def test_associate_hypothesis_enumeration_contains_distinct_opposite_action():
    value = _binding().as_dict()
    value["observed_current_decision"] = "ASSOCIATE"
    value["created_entity_uid"] = None
    value["created_identity_uid"] = None
    hypotheses = enumerate_identity_hypotheses(
        IncidentBinding.from_mapping(value),
        candidate_aliases=["CANDIDATE_1_CONTEXT"],
    )
    assert [item["hypothesis_action"] for item in hypotheses] == [
        "SAME_INSTANCE",
        "SEPARATE_MEMBER_GROUPS",
    ]
    assert all(item["candidate_constraint"] for item in hypotheses)


def test_identity_hypothesis_enumeration_rejects_unknown_alias():
    with pytest.raises(ValueError, match="unknown or incomplete identity aliases"):
        enumerate_identity_hypotheses(
            _binding(), candidate_aliases=["CANDIDATE_99_CONTEXT"]
        )


def test_only_exact_all_pass_shadow_evidence_exposes_commit_constraint():
    aggregate = _aggregate(
        "SAME_INSTANCE",
        entities=["ANCHOR", "CANDIDATE_1_CONTEXT"],
    )
    compiled = compile_blind_candidate(aggregate, _binding())
    fingerprint = compiled["constraint_fingerprint"]

    promoted = decide_automatic_promotion(compiled, _shadow(fingerprint))
    assert promoted["stage"] == GeneratorStage.COMMIT_ELIGIBLE.value
    assert promoted["commit_constraint"] == compiled["candidate_constraint"]
    assert promoted["defer_reasons"] == []

    blocked = decide_automatic_promotion(
        compiled,
        _shadow(fingerprint, local_global_parity_pass=False),
    )
    assert blocked["stage"] == GeneratorStage.DEFERRED.value
    assert blocked["commit_constraint"] is None
    assert "shadow_gate_failed:local_global_parity_pass" in blocked["defer_reasons"]

    mismatch = decide_automatic_promotion(
        compiled, _shadow("constraint_semantics_wrong")
    )
    assert mismatch["stage"] == GeneratorStage.DEFERRED.value
    assert "shadow_constraint_fingerprint_mismatch" in mismatch["defer_reasons"]


def test_semantic_fingerprint_ignores_audit_metadata_but_not_behavior():
    constraint = compile_blind_candidate(
        _aggregate(
            "SAME_INSTANCE",
            entities=["ANCHOR", "CANDIDATE_1_CONTEXT"],
        ),
        _binding(),
    )["candidate_constraint"]
    same = dict(constraint)
    same["source"] = "different_source"
    same["evidence_refs"] = ["different_evidence"]
    same["constraint_uid"] = "different_uid"
    assert semantic_constraint_fingerprint(same) == semantic_constraint_fingerprint(
        constraint
    )

    changed = dict(constraint)
    changed["target_entity_uid"] = "different-target"
    assert semantic_constraint_fingerprint(changed) != semantic_constraint_fingerprint(
        constraint
    )
