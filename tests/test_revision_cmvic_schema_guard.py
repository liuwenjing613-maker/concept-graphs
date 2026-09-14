from conceptgraph.revision.candidate_verifier import CandidateEvidenceScore
from conceptgraph.revision.counterfactual_projection import CMVIC_STATISTIC_NAME
from conceptgraph.revision.selective_commit import (
    CalibrationArtifact,
    SelectiveCandidate,
    decide_selective_commit,
)


def _score():
    return CandidateEvidenceScore.build(
        incident_uid="incident",
        candidate_uid="candidate",
        capability="IDENTITY",
        primary_statistic=CMVIC_STATISTIC_NAME,
        primary_score=0.2,
        noop_primary_score=0.1,
        valid=True,
        verification_observation_count=4,
        diagnostics={"counterfactual_observable": True},
        evidence_policy_uid="cmvic_policy_current",
    )


def _calibration(**overrides):
    value = {
        "capability": "IDENTITY",
        "feature_names": ["primary_advantage"],
        "coefficients": [10.0],
        "intercept": 0.0,
        "commit_threshold": 0.5,
        "ready_for_automatic_commit": True,
        "fit_case_count": 10,
        "fit_positive_count": 5,
        "fit_negative_count": 5,
        "target_harm_rate": 0.05,
        "source_hashes": {"fit": "a" * 64},
    }
    value.update(overrides)
    return CalibrationArtifact.from_mapping(value)


def _decide(calibration):
    return decide_selective_commit(
        incident_uid="incident",
        candidates=[SelectiveCandidate(_score(), {"type": "CREATE_INSTANCE"})],
        calibration=calibration,
    )


def test_legacy_calibration_cannot_commit_cmvic_score():
    decision = _decide(_calibration())
    assert decision["decision"] == "DEFER"
    assert "calibration_evidence_schema_undeclared" in decision["defer_reasons"]


def test_declared_mismatched_evidence_policy_defers():
    decision = _decide(
        _calibration(
            primary_statistic=CMVIC_STATISTIC_NAME,
            evidence_policy_uid="cmvic_policy_other",
        )
    )
    assert decision["decision"] == "DEFER"
    assert "calibration_evidence_policy_mismatch" in decision["defer_reasons"]


def test_declared_matching_schema_can_be_evaluated():
    decision = _decide(
        _calibration(
            primary_statistic=CMVIC_STATISTIC_NAME,
            evidence_policy_uid="cmvic_policy_current",
        )
    )
    assert decision["decision"] == "COMMIT"
