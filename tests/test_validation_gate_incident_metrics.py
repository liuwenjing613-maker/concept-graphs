from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL = Path(__file__).with_name("compute_validation_gate_incident_metrics.py")
REPOSITORY = Path(__file__).parents[1] / "scripts" / "compute_validation_gate_incident_metrics.py"
PATH = LOCAL if LOCAL.exists() else REPOSITORY
SPEC = spec_from_file_location("compute_validation_gate_incident_metrics", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(uid, *, state="WRONG", evidence="YES", weight=1.0, cohort="calibration_random"):
    return {
        "schema_version": "2.1.0",
        "annotation_unit": "incident",
        "scene_id": "room0",
        "case_uid": uid,
        "incident_uid": uid,
        "checker_id": "DET-001",
        "stage": "detection",
        "checker_ids": ["DET-001", "SEG-004"],
        "cohort": cohort,
        "sampling_weight": weight,
        "review_score": 5.0,
        "reviewer_id": "R1",
        "evidence_sufficient": evidence,
        "final_state": state,
        "final_error_type": "FALSE_SPLIT" if state == "WRONG" else "NOT_APPLICABLE",
        "review_seconds": 10,
        "notes": None,
    }


def test_endpoint_contract_does_not_require_root_stage_or_repair_guess():
    item = row("incident-a")
    MODULE.validate_label(item, ("room0", "incident-a"))
    assert "root_stage_correct" not in item
    assert "repair_action" not in item


def test_weighted_incident_precision_and_missing_evidence_bounds_are_separate():
    rows = [
        row("wrong", state="WRONG", weight=1),
        row("correct", state="CORRECT", weight=3),
        row("unclear", state="UNCLEAR", evidence="NO", weight=4),
    ]
    record = MODULE.group_record("test", rows)
    assert record["calibration_weighted_evidence_sufficiency"] == 0.5
    assert record["calibration_weighted_endpoint_error_precision"] == 0.25
    assert record["calibration_endpoint_error_rate_lower_bound"] == 0.125
    assert record["calibration_endpoint_error_rate_upper_bound"] == 0.625
    assert record["endpoint_error_rate_conditional"] == 0.5
    assert record["confirmed_endpoint_error_yield"] == 0.333333
    assert record["endpoint_error_rate_lower_bound"] == 0.333333
    assert record["endpoint_error_rate_upper_bound"] == 0.666667


def test_priority_reports_conditional_precision_and_full_queue_yield():
    rows = [
        dict(row("unclear", state="UNCLEAR", evidence="NO", cohort="diagnostic_priority"), review_score=9),
        dict(row("wrong", state="WRONG", cohort="diagnostic_priority"), review_score=8),
    ]
    ordered = MODULE.priority_rows(rows)
    assert MODULE.conditional_p_at_k(ordered, 2, MODULE.is_endpoint_error) == 1.0
    assert MODULE.p_at_k(ordered, 2, MODULE.is_adjudicable) == 0.5
    assert MODULE.p_at_k(ordered, 2, MODULE.is_endpoint_error) == 0.5


def test_ranking_diagnostics_detects_reversed_priority_score():
    rows = [
        dict(row("wrong-low", state="WRONG"), review_score=1),
        dict(row("wrong-mid", state="WRONG"), review_score=2),
        dict(row("correct-mid", state="CORRECT"), review_score=8),
        dict(row("correct-high", state="CORRECT"), review_score=9),
    ]
    result = MODULE.ranking_diagnostics(rows)
    assert result["roc_auc"] == 0.0
    assert result["confirmed_error_prevalence"] == 0.5
    assert result["top_k"][0]["confirmed_error_precision"] == 0.5


def test_linked_checker_groups_use_all_incident_memberships():
    rows = [row("one", state="WRONG"), row("two", state="CORRECT")]
    records = {record["group"]: record for record in MODULE.grouped_membership(rows, "checker_ids")}
    assert records["DET-001"]["incident_count"] == 2
    assert records["SEG-004"]["endpoint_error_count"] == 1
    assert records["SEG-004"]["confirmed_error_coverage"] == 1.0


def test_label_merge_requires_exact_incident_coverage():
    worklist = [row("one"), row("two")]
    try:
        MODULE.merge_labels(worklist, [row("one")])
    except ValueError as exc:
        assert "incomplete labels" in str(exc)
    else:
        raise AssertionError("partial incident labels must not produce final metrics")


def decision_fixture(*, coverage=0.8, precision=0.5, priority=0.5, errors=1, census=True):
    return {
        "system_gates": {
            "all_system_gates_pass": True,
            "full_endpoint_census": census,
        },
        "overall": {
            "evidence_sufficiency": coverage,
            "endpoint_error_count": errors,
            "confirmed_endpoint_error_yield": 0.1 if errors else 0.0,
            "calibration_weighted_evidence_sufficiency": coverage,
            "priority_evidence_coverage_at_20": coverage,
            "calibration_weighted_endpoint_error_precision": precision,
            "priority_confirmed_error_yield_at_20": priority,
        },
        "by_scene": [
            {
                "evidence_sufficiency": coverage,
                "calibration_weighted_evidence_sufficiency": coverage,
            },
            {
                "evidence_sufficiency": coverage,
                "calibration_weighted_evidence_sufficiency": coverage,
            },
        ],
    }


def test_decision_only_advances_to_expert_trace_not_direct_repair():
    result = MODULE.decision(decision_fixture())
    assert result["verdict"] == "PROCEED_TO_EXPERT_TRACE"
    assert result["repair_gate_status"] == "PENDING_EXPERT_TRACE_AND_REPLAY"
    assert MODULE.decision(decision_fixture(coverage=0.79))["verdict"] == "STOP_OR_REDESIGN_EVIDENCE"
    assert MODULE.decision(decision_fixture(errors=0))["verdict"].startswith(
        "NO_CONFIRMED_ENDPOINT_ERRORS"
    )
    assert MODULE.decision(decision_fixture(precision=0.49, census=False))["verdict"] == "REVISE_SCREENERS"
