import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL_STAGING_PATH = Path(__file__).with_name("compute_validation_gate_metrics.py")
REPOSITORY_PATH = Path(__file__).parents[1] / "scripts" / "compute_validation_gate_metrics.py"
MODULE_PATH = LOCAL_STAGING_PATH if LOCAL_STAGING_PATH.exists() else REPOSITORY_PATH
SPEC = spec_from_file_location("compute_validation_gate_metrics", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(
    uid,
    cohort="calibration_random",
    finding="YES",
    harm="NONE",
    repair="NONE",
    weight=1.0,
    evidence="YES",
    root=None,
    notes=None,
    locality=None,
):
    if root is None:
        root = "NOT_APPLICABLE" if finding == "NO" else ("UNCERTAIN" if finding == "UNCERTAIN" else "YES")
    if locality is None:
        locality = (
            "MULTI_OBJECT"
            if repair in {"REASSIGN_OBSERVATION", "MERGE_OBJECTS", "SPLIT_OBJECT"}
            else ("NOT_APPLICABLE" if repair in {"NONE", "NEED_MORE_VIEW"} else "LOCAL")
        )
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
        "evidence_sufficient": evidence,
        "finding_correct": finding,
        "root_stage_correct": root,
        "physical_interpretation": "test interpretation",
        "downstream_harm": harm,
        "harm_confidence": 5,
        "repair_action": repair,
        "repair_locality": locality,
        "repair_confidence": 5,
        "review_seconds": 60,
        "notes": notes,
    }


def test_actionable_requires_true_harm_and_specific_action():
    assert MODULE.is_actionable(row("a", harm="WRONG_OBSERVATION_MEMBERSHIP", repair="REASSIGN_OBSERVATION"))
    assert not MODULE.is_actionable(row("b", finding="NO", harm="WRONG_OBSERVATION_MEMBERSHIP", repair="REASSIGN_OBSERVATION"))
    assert not MODULE.is_actionable(row("c", harm="NONE", repair="REASSIGN_OBSERVATION"))
    assert not MODULE.is_actionable(row("d", harm="WRONG_OBSERVATION_MEMBERSHIP", repair="UNKNOWN"))
    assert not MODULE.is_actionable(
        row(
            "e",
            evidence="PARTIAL",
            harm="WRONG_OBSERVATION_MEMBERSHIP",
            repair="REASSIGN_OBSERVATION",
            notes="缺少最终对象视图",
        )
    )


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


def test_missing_evidence_is_coverage_gap_not_false_positive():
    rows = [
        row("yes", finding="YES", weight=1),
        row("no", finding="NO", weight=3),
        row("missing", finding="YES", weight=4, evidence="PARTIAL", notes="缺少历史 PCD"),
    ]
    record = MODULE.group_record("test", rows)
    assert record["weighted_evidence_sufficiency"] == 0.5
    assert record["weighted_finding_precision"] == 0.25
    assert record["weighted_finding_rate_lower_bound"] == 0.125
    assert record["weighted_finding_rate_upper_bound"] == 0.625


def test_priority_reports_conditional_precision_coverage_and_confirmed_yield_separately():
    rows = [
        dict(
            row(
                "missing",
                cohort="diagnostic_priority",
                evidence="PARTIAL",
                harm="WRONG_OBSERVATION_MEMBERSHIP",
                repair="REASSIGN_OBSERVATION",
                notes="缺少最终对象视图",
            ),
            review_score=9,
        ),
        dict(
            row(
                "confirmed",
                cohort="diagnostic_priority",
                harm="WRONG_OBSERVATION_MEMBERSHIP",
                repair="REASSIGN_OBSERVATION",
            ),
            review_score=8,
        ),
    ]
    ordered = MODULE.priority_rows(rows)
    assert MODULE.conditional_p_at_k(ordered, 2, MODULE.is_actionable) == 1.0
    assert MODULE.p_at_k(ordered, 2, MODULE.is_adjudicable) == 0.5
    assert MODULE.p_at_k(ordered, 2, MODULE.is_actionable) == 0.5


def test_logical_label_constraints_match_browser_service():
    key = ("room0", "case")
    valid_partial = row(
        "case",
        evidence="PARTIAL",
        finding="YES",
        harm="UNKNOWN",
        repair="NEED_MORE_VIEW",
        notes="缺少融合前点云",
    )
    MODULE.validate_label_values(valid_partial, key, "test")
    invalid = dict(valid_partial, notes=None)
    try:
        MODULE.validate_label_values(invalid, key, "test")
    except ValueError as exc:
        assert "missing-evidence note" in str(exc)
    else:
        raise AssertionError("PARTIAL without a note must be rejected")


def decision_fixture(evidence_coverage):
    return {
        "system_gates": {"all_system_gates_pass": True},
        "reviewer_agreement": {"status": "COMPLETE"},
        "overall": {
            "priority_actionable_p_at_20": 0.8,
            "weighted_finding_precision": 0.7,
            "weighted_actionable_precision": 0.5,
            "root_stage_accuracy": 0.8,
            "evidence_sufficiency": evidence_coverage,
            "weighted_evidence_sufficiency": evidence_coverage,
            "priority_evidence_coverage_at_20": evidence_coverage,
        },
        "by_scene": [
            {
                "group": "room0",
                "weighted_actionable_precision": 0.5,
                "weighted_evidence_sufficiency": evidence_coverage,
            },
            {
                "group": "office0",
                "weighted_actionable_precision": 0.5,
                "weighted_evidence_sufficiency": evidence_coverage,
            },
        ],
        "by_checker": [{"group": "DET-001", "actionable_count": 20}],
    }


def test_decision_stops_when_evidence_coverage_is_too_low():
    result = MODULE.decision(decision_fixture(0.79))
    assert result["verdict"] == "STOP_OR_REDESIGN_EVIDENCE"
    assert result["evidence_coverage_pass"] is False


def test_decision_can_go_only_after_evidence_coverage_passes():
    result = MODULE.decision(decision_fixture(0.8))
    assert result["verdict"] == "GO"
    assert result["evidence_coverage_pass"] is True


def test_decision_cannot_go_before_independent_r2_and_adjudication():
    metrics = decision_fixture(0.8)
    metrics["reviewer_agreement"] = {"status": "PENDING"}
    assert MODULE.decision(metrics)["verdict"] == "PENDING_INDEPENDENT_R2"
    metrics["reviewer_agreement"] = {"status": "NEEDS_ADJUDICATION"}
    assert MODULE.decision(metrics)["verdict"] == "PENDING_ADJUDICATION"


def test_r2_disagreements_require_adjudicated_labels():
    r1 = [row("agree"), row("disagree")]
    r2 = [row("agree"), row("disagree", finding="NO")]
    expected = [
        {"scene_id": "room0", "case_uid": "agree"},
        {"scene_id": "room0", "case_uid": "disagree"},
    ]
    pending = MODULE.agreement(r1, r2, expected, [])
    assert pending["status"] == "NEEDS_ADJUDICATION"
    assert pending["disagreement_cases"] == 1
    assert pending["unadjudicated_disagreement_cases"] == 1
    complete = MODULE.agreement(r1, r2, expected, [row("disagree", finding="NO")])
    assert complete["status"] == "COMPLETE"
    assert complete["unadjudicated_disagreement_cases"] == 0


def test_partial_r2_is_pending_instead_of_a_fake_final_decision():
    r1 = [row("done"), row("missing")]
    expected = [
        {"scene_id": "room0", "case_uid": "done"},
        {"scene_id": "room0", "case_uid": "missing"},
    ]
    result = MODULE.agreement(r1, [row("done")], expected, [])
    assert result["status"] == "PENDING"
    assert result["completed_cases"] == 1
    assert result["missing_case_keys"] == ["room0/missing"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_system_gate_requires_hash_locked_human_system_projection(tmp_path):
    write_json(
        tmp_path / "parity" / "parity_report.json",
        {"status": "PASS", "checks": {"map_unchanged": True}},
    )
    for scene in ("room0", "office0"):
        formal = tmp_path / "runs" / scene / "formal"
        write_json(formal / "audit" / "validation.json", {"gate_status": "PASS"})
        write_json(
            formal / "audit_validity_gate_v1" / "audit_summary.json",
            {
                "validation_gate_status": "PASS",
                "population_censored": False,
                "weighted_precision_allowed": True,
            },
        )
    worklist = tmp_path / "labels" / "r1_worklist.jsonl"
    worklist.parent.mkdir(parents=True)
    worklist.write_text('{"scene_id":"room0","case_uid":"case_1"}\n', encoding="utf-8")
    case_dir = tmp_path / "cases" / "room0" / "case_1"
    case_json = case_dir / "case.json"
    write_json(case_json, {"finding_uid": "case_1"})
    asset = case_dir / "review_final_objects_relative.png"
    asset.write_bytes(b"locked-review-image")
    review_path = case_dir / "review_evidence.json"
    write_json(
        review_path,
        {
            "scene_id": "room0",
            "case_uid": "case_1",
            "source_case_json_sha256": hashlib.sha256(case_json.read_bytes()).hexdigest(),
            "displayed_asset_sha256": {
                asset.name: hashlib.sha256(asset.read_bytes()).hexdigest()
            },
        },
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "READY_WITH_DECLARED_LIMITATIONS",
        "worklist_sha256": hashlib.sha256(worklist.read_bytes()).hexdigest(),
        "case_count": 1,
        "cases": [
            {
                "scene_id": "room0",
                "case_uid": "case_1",
                "review_evidence_path": "cases/room0/case_1/review_evidence.json",
                "review_evidence_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                "displayed_asset_count": 1,
            }
        ],
        "all_artifact_hashes_match": True,
        "all_available_final_objects_link_exactly": True,
    }
    write_json(tmp_path / "review_evidence_manifest.json", manifest)
    result = MODULE.system_gates(tmp_path)
    assert result["human_system_evidence_projection"]["structural_gate"] == "PASS"
    assert result["human_system_evidence_projection"]["checked_displayed_asset_count"] == 1
    assert result["all_system_gates_pass"] is True
    asset.write_bytes(b"tampered")
    result = MODULE.system_gates(tmp_path)
    assert result["human_system_evidence_projection"]["displayed_asset_hashes_match"] is False
    assert result["all_system_gates_pass"] is False
    asset.write_bytes(b"locked-review-image")
    manifest["worklist_sha256"] = "tampered"
    write_json(tmp_path / "review_evidence_manifest.json", manifest)
    result = MODULE.system_gates(tmp_path)
    assert result["human_system_evidence_projection"]["structural_gate"] == "FAIL"
    assert result["all_system_gates_pass"] is False
