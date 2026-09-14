import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL = Path(__file__).with_name("assemble_validation_gate_incidents.py")
REPOSITORY = Path(__file__).parents[1] / "scripts" / "assemble_validation_gate_incidents.py"
PATH = LOCAL if LOCAL.exists() else REPOSITORY
SPEC = spec_from_file_location("assemble_validation_gate_incidents", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_worklist_uses_incident_identity_but_preserves_representative_finding(tmp_path):
    experiment = tmp_path / "experiment"
    audit = experiment / "audit_validity_gate_endpoint_v2_1"
    case_dir = audit / "cases" / "finding_7"
    case_dir.mkdir(parents=True)
    (case_dir / "case.json").write_text('{"finding_uid":"finding_7"}', encoding="utf-8")
    selection = {
        "annotation_unit": "incident",
        "strategy": "incident_deduplicated_dual_cohort_endpoint_review",
        "weighted_precision_allowed": True,
        "deduplication": {"eligible_finding_count": 3, "reviewable_incident_count": 1},
        "selected_cohort_counts": {"calibration_random": 1},
        "selected": [
            {
                "incident_uid": "incident_abc",
                "representative_finding_uid": "finding_7",
                "checker_id": "ASSOC-004",
                "stage": "association",
                "subtype": "SAME_TARGET",
                "checker_ids": ["DET-001", "SEG-004", "ASSOC-004"],
                "stages": ["detection", "segmentation", "association"],
                "subtypes": ["DUPLICATE", "OVERSEGMENTATION", "SAME_TARGET"],
                "member_finding_uids": ["finding_1", "finding_2", "finding_7"],
                "trigger_observation_uids": ["obs-a", "obs-b"],
                "representative_trigger_observation_uids": ["obs-a", "obs-b"],
                "all_trigger_observation_uids": ["obs-a", "obs-b", "obs-c"],
                "final_owner_uids": ["object-final"],
                "machine_resolution_status": "TRIGGERS_CONVERGED_TO_ONE_FINAL_OBJECT",
                "identity_kind": "final_endpoint_set",
                "cohort": "calibration_random",
                "case_rank": 1,
                "review_score": 8.0,
                "review_priority": 2,
                "selection_probability": 0.25,
                "sampling_weight": 4.0,
            }
        ],
    }
    (audit / "case_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    validation_root = tmp_path / "validation"
    rows, record = MODULE.worklist_rows(
        "room0", experiment, validation_root, "audit_validity_gate_endpoint_v2_1"
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["case_uid"] == "incident_abc"
    assert row["incident_uid"] == "incident_abc"
    assert row["representative_finding_uid"] == "finding_7"
    assert row["checker_ids"] == ["DET-001", "SEG-004", "ASSOC-004"]
    assert row["representative_trigger_observation_uids"] == ["obs-a", "obs-b"]
    assert row["all_trigger_observation_uids"] == ["obs-a", "obs-b", "obs-c"]
    assert Path(row["case_dir"]).name == "finding_7"
    assert row["final_state"] is None
    assert record["selected_incident_count"] == 1
    assert record["selection_mode"] == "endpoint_census"
