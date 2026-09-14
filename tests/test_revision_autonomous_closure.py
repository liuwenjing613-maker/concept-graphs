import json

import pytest

from conceptgraph.revision.autonomous_identity import (
    distinct_candidate_partitions,
    heldout_assignments_distinguishable,
)
from conceptgraph.revision.candidate_verifier import (
    CandidateEvidenceScore,
    CandidateVerifier,
    critic_preference_value,
)
from conceptgraph.revision.capabilities import enumerate_feasible_actions
from conceptgraph.revision.evidence_split import (
    EvidenceReference,
    EvidenceSplitManifest,
)
from conceptgraph.revision.selective_commit import (
    CalibrationArtifact,
    SelectiveCandidate,
    decide_selective_commit,
)
from conceptgraph.revision.shadow_critic import parse_shadow_critic_response


def _ref(obs_uid, frame, digest_char):
    return EvidenceReference.build(
        obs_uid=obs_uid,
        frame_index=frame,
        sha256=digest_char * 64,
        source_role="TEST_CROP",
    )


def _split():
    return EvidenceSplitManifest.build(
        incident_uid="incident_test",
        anchor_obs_uid="anchor",
        anchor_frame=1,
        proposal=[_ref("anchor", 1, "a")],
        verification=[_ref("future", 3, "b")],
        minimum_frame_gap=2,
    )


def _state(selected_score, competing_score):
    return {
        "decision_trace": [
            {
                "obs_uid": "future",
                "natural_match": 0,
                "applied_match": 0,
                "constraint_overrode_natural": False,
                "threshold_semantics": {"sim_threshold": 1.2},
                "natural_candidates": [
                    {"index": 0, "score": selected_score},
                    {"index": 1, "score": competing_score},
                ],
            }
        ]
    }


def _calibration(*, features, coefficients, intercept=0.0, threshold=0.8, ready=True):
    return CalibrationArtifact.from_mapping(
        {
            "capability": "IDENTITY",
            "feature_names": features,
            "coefficients": coefficients,
            "intercept": intercept,
            "commit_threshold": threshold,
            "ready_for_automatic_commit": ready,
            "fit_case_count": 20,
            "fit_positive_count": 5,
            "fit_negative_count": 15,
            "target_harm_rate": 0.05,
            "source_hashes": {"frozen_fit_rows": "c" * 64},
        }
    )


def test_evidence_split_rejects_observation_and_hash_overlap():
    proposal = _ref("anchor", 1, "a")
    with pytest.raises(ValueError, match="overlap"):
        EvidenceSplitManifest.build(
            incident_uid="incident_test",
            anchor_obs_uid="anchor",
            anchor_frame=1,
            proposal=[proposal],
            verification=[_ref("anchor", 3, "b")],
        )
    with pytest.raises(ValueError, match="overlap"):
        EvidenceSplitManifest.build(
            incident_uid="incident_test",
            anchor_obs_uid="anchor",
            anchor_frame=1,
            proposal=[proposal],
            verification=[_ref("future", 3, "a")],
        )


def test_evidence_split_rejects_early_and_oracle_evidence():
    with pytest.raises(ValueError, match="temporal embargo"):
        EvidenceSplitManifest.build(
            incident_uid="incident_test",
            anchor_obs_uid="anchor",
            anchor_frame=10,
            proposal=[_ref("anchor", 10, "a")],
            verification=[_ref("future", 10, "b")],
        )
    with pytest.raises(ValueError, match="oracle-like"):
        EvidenceReference.from_mapping(
            {
                "obs_uid": "future",
                "frame_index": 12,
                "sha256": "b" * 64,
                "source_role": "TEST",
                "human_label": "same",
            }
        )


def test_mapper_self_likelihood_adversary_favors_noop_and_primary_only_defers():
    verifier = CandidateVerifier()
    score = verifier.score_identity(
        incident_uid="incident_test",
        candidate_uid="repair_same",
        candidate_state=_state(0.8, 1.5),
        noop_state=_state(1.5, 0.8),
        split=_split(),
        runtime_valid=True,
    )
    assert score.score_advantage_over_noop < 0.0
    calibration = _calibration(
        features=["primary_advantage"], coefficients=[2.0], threshold=0.6
    )
    decision = decide_selective_commit(
        incident_uid="incident_test",
        candidates=[
            SelectiveCandidate(score=score, candidate_constraint={"type": "ASSIGN"})
        ],
        calibration=calibration,
    )
    assert decision["decision"] == "DEFER"
    assert decision["semantic_commit_threshold_count"] == 1


def test_calibrated_critic_feature_can_select_without_using_self_confidence():
    score = CandidateEvidenceScore.build(
        incident_uid="incident_test",
        candidate_uid="repair_same",
        capability="IDENTITY",
        primary_statistic="TEST",
        primary_score=-0.2,
        noop_primary_score=-0.1,
        valid=True,
        verification_observation_count=4,
        diagnostics={},
        vlm_pairwise_preference=1.0,
    )
    calibration = _calibration(
        features=["primary_advantage", "vlm_pairwise_preference"],
        coefficients=[0.0, 4.0],
        threshold=0.9,
    )
    decision = decide_selective_commit(
        incident_uid="incident_test",
        candidates=[
            SelectiveCandidate(score=score, candidate_constraint={"type": "ASSIGN"})
        ],
        calibration=calibration,
    )
    assert decision["decision"] == "COMMIT"
    assert decision["selected_candidate_uid"] == "repair_same"


def test_unready_calibration_and_exact_tie_are_fail_closed():
    first = CandidateEvidenceScore.build(
        incident_uid="incident_test",
        candidate_uid="first",
        capability="IDENTITY",
        primary_statistic="TEST",
        primary_score=1.0,
        noop_primary_score=0.0,
        valid=True,
        verification_observation_count=2,
        diagnostics={},
    )
    second = CandidateEvidenceScore.build(
        incident_uid="incident_test",
        candidate_uid="second",
        capability="IDENTITY",
        primary_statistic="TEST",
        primary_score=1.0,
        noop_primary_score=0.0,
        valid=True,
        verification_observation_count=2,
        diagnostics={},
    )
    unready = _calibration(
        features=["primary_advantage"], coefficients=[3.0], ready=False
    )
    unready_decision = decide_selective_commit(
        incident_uid="incident_test",
        candidates=[SelectiveCandidate(first, {"type": "A"})],
        calibration=unready,
    )
    assert unready_decision["decision"] == "DEFER"
    assert (
        "calibration_not_ready_for_automatic_commit"
        in unready_decision["defer_reasons"]
    )
    ready = _calibration(features=["primary_advantage"], coefficients=[3.0])
    tie = decide_selective_commit(
        incident_uid="incident_test",
        candidates=[
            SelectiveCandidate(first, {"type": "A"}),
            SelectiveCandidate(second, {"type": "B"}),
        ],
        calibration=ready,
    )
    assert tie["decision"] == "DEFER"
    assert "non_unique_best_candidate" in tie["defer_reasons"]


def test_feasible_actions_depend_on_payload_not_endpoint_label():
    assert enumerate_feasible_actions() == ("NO_OP",)
    assert enumerate_feasible_actions(
        identity_candidate_count=1,
        observed_current_decision="CREATE",
        created_identity_binding_complete=True,
    ) == ("NO_OP", "SAME_INSTANCE", "SEPARATE_MEMBER_GROUPS")
    assert enumerate_feasible_actions(
        identity_candidate_count=1,
        observed_current_decision="ASSOCIATE",
    ) == ("NO_OP", "SAME_INSTANCE", "SEPARATE_MEMBER_GROUPS")
    assert enumerate_feasible_actions(
        geometry_contract={"payload_uid": "g"},
        partition_contract={"payload_uid": "p"},
    ) == ("NO_OP", "RESTORE_OBSERVATION_GEOMETRY")


def test_shadow_critic_parser_keeps_confidence_diagnostic_only():
    parsed = parse_shadow_critic_response(
        json.dumps(
            {
                "preferred_state": "STATE_B",
                "confidence": 0.97,
                "reason": "B groups recurring views more coherently",
                "counterevidence": [],
                "needed_evidence": [],
                "cited_evidence_ids": ["V01", "V02"],
            }
        ),
        allowed_state_ids=["STATE_A", "STATE_B"],
        allowed_evidence_ids=["V01", "V02"],
    )
    assert parsed["preferred_state"] == "STATE_B"
    assert parsed["confidence_is_not_calibrated_probability"] is True
    assert (
        critic_preference_value(
            preferred_state="STATE_B", candidate_state="STATE_B", noop_state="STATE_A"
        )
        == 1.0
    )


def test_shadow_critic_parser_normalizes_qualitative_diagnostic_confidence():
    parsed = parse_shadow_critic_response(
        json.dumps(
            {
                "preferred_state": "DEFER",
                "confidence": "moderate",
                "reason": "views are ambiguous",
                "counterevidence": "ambiguous seam",
                "needed_evidence": "another viewpoint",
                "cited_evidence_ids": ["V01"],
            }
        ),
        allowed_state_ids=["STATE_A", "STATE_B"],
        allowed_evidence_ids=["V01"],
    )

    assert parsed["confidence_diagnostic_only"] == 0.5
    assert parsed["confidence_raw_diagnostic"] == "moderate"
    assert parsed["confidence_parse_status"] == "QUALITATIVE_NORMALIZED"
    assert parsed["counterevidence"] == ["ambiguous seam"]
    assert parsed["needed_evidence"] == ["another viewpoint"]


def test_wide_context_is_not_scheduled_for_observationally_equal_partitions():
    same_assignment_a = {
        "groups": [
            {"evidence_ids": ["V01", "V02"], "n_points": 100},
        ]
    }
    same_assignment_b = {
        "groups": [
            {"evidence_ids": ["V02", "V01"], "n_points": 999},
        ]
    }
    distinguishable = {
        "groups": [
            {"evidence_ids": ["V01"], "n_points": 60},
            {"evidence_ids": ["V02"], "n_points": 40},
        ]
    }
    assert not heldout_assignments_distinguishable(
        {"STATE_A": same_assignment_a, "STATE_B": same_assignment_b}
    )
    assert heldout_assignments_distinguishable(
        {"STATE_A": same_assignment_a, "STATE_B": distinguishable}
    )


def test_behaviorally_equivalent_candidates_become_auditable_no_repair():
    noop = "partition_noop"
    assert distinct_candidate_partitions(noop, [noop, noop]) == ()
    assert distinct_candidate_partitions(
        noop, [noop, "partition_b", "partition_a", "partition_b"]
    ) == ("partition_a", "partition_b")
