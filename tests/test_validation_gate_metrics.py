from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL_STAGING_PATH = Path(__file__).with_name("compute_validation_gate_metrics.py")
REPOSITORY_PATH = Path(__file__).parents[1] / "scripts" / "compute_validation_gate_metrics.py"
MODULE_PATH = LOCAL_STAGING_PATH if LOCAL_STAGING_PATH.exists() else REPOSITORY_PATH
SPEC = spec_from_file_location("compute_validation_gate_metrics", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(uid, cohort="calibration_random", finding="YES", harm="NONE", repair="NONE", weight=1.0):
    return {
        "scene_id": "room0",
        "case_uid": uid,
        "finding_uid": uid,
        "checker_id": "DET-001",
        "stage": "detection",
        "cohort": cohort,
        "sampling_weight": weight,
        "review_score": 1.0,
        "reviewer_id": "R1",
        "evidence_sufficient": "YES",
        "finding_correct": finding,
        "root_stage_correct": "YES",
        "physical_interpretation": "test interpretation",
        "downstream_harm": harm,
        "harm_confidence": 5,
        "repair_action": repair,
        "repair_locality": "LOCAL" if repair != "NONE" else "NOT_APPLICABLE",
        "repair_confidence": 5,
        "review_seconds": 60,
    }


def test_actionable_requires_true_harm_and_specific_action():
    assert MODULE.is_actionable(row("a", harm="WRONG_OBSERVATION_MEMBERSHIP", repair="REASSIGN_OBSERVATION"))
    assert not MODULE.is_actionable(row("b", finding="NO", harm="WRONG_OBSERVATION_MEMBERSHIP", repair="REASSIGN_OBSERVATION"))
    assert not MODULE.is_actionable(row("c", harm="NONE", repair="REASSIGN_OBSERVATION"))
    assert not MODULE.is_actionable(row("d", harm="WRONG_OBSERVATION_MEMBERSHIP", repair="UNKNOWN"))


def test_weighted_rate_uses_sampling_weight():
    rows = [row("a", finding="YES", weight=1), row("b", finding="NO", weight=3)]
    assert MODULE.weighted_rate(rows, lambda item: item["finding_correct"] == "YES") == 0.25


def test_incomplete_r1_is_rejected():
    worklist = [row("a")]
    incomplete = [dict(row("a"), repair_action=None)]
    try:
        MODULE.validate_completed_labels(incomplete, worklist)
    except ValueError as exc:
        assert "incomplete labels" in str(exc)
    else:
        raise AssertionError("incomplete labels must be rejected")


def test_priority_sort_and_p_at_k():
    rows = [
        dict(row("low", cohort="diagnostic_priority", finding="NO"), review_score=1),
        dict(row("high", cohort="diagnostic_priority", finding="YES"), review_score=9),
    ]
    ordered = MODULE.priority_rows(rows)
    assert [item["case_uid"] for item in ordered] == ["high", "low"]
    assert MODULE.p_at_k(ordered, 1, lambda item: item["finding_correct"] == "YES") == 1.0
