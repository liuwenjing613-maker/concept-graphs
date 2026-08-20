from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL = Path(__file__).with_name("generate_validation_gate_expert_queue.py")
REPOSITORY = Path(__file__).parents[1] / "scripts" / "generate_validation_gate_expert_queue.py"
PATH = LOCAL if LOCAL.exists() else REPOSITORY
SPEC = spec_from_file_location("generate_validation_gate_expert_queue", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def meta(uid):
    return {
        "scene_id": "room0",
        "case_uid": uid,
        "incident_uid": uid,
        "representative_finding_uid": "finding-1",
        "member_finding_uids": ["finding-1", "finding-2"],
        "checker_ids": ["DET-001", "SEG-004"],
        "stages": ["detection", "segmentation"],
        "trigger_observation_uids": ["obs-1", "obs-2"],
        "representative_trigger_observation_uids": ["obs-1", "obs-2"],
        "all_trigger_observation_uids": ["obs-1", "obs-2", "obs-3"],
        "final_owner_uids": ["object-1"],
        "case_dir": "/validation/cases/room0/finding-1",
    }


def label(uid, state, evidence="YES"):
    return {
        "scene_id": "room0",
        "incident_uid": uid,
        "reviewer_id": "R1",
        "evidence_sufficient": evidence,
        "final_state": state,
        "final_error_type": "FALSE_SPLIT" if state == "WRONG" else "NOT_APPLICABLE",
        "review_seconds": 12.5,
        "notes": "",
    }


def test_only_confirmed_endpoint_errors_enter_causal_trace_queue():
    worklist = [meta("wrong"), meta("correct"), meta("unclear")]
    labels = [label("wrong", "WRONG"), label("correct", "CORRECT"), label("unclear", "UNCLEAR", "NO")]
    result = MODULE.generate(worklist, labels)
    assert len(result) == 1
    assert result[0]["incident_uid"] == "wrong"
    assert result[0]["earliest_causal_stage"] is None
    assert result[0]["replay_status"] == "NOT_RUN"
    assert result[0]["repair_verified"] is None
    assert result[0]["representative_trigger_observation_uids"] == ["obs-1", "obs-2"]
    assert result[0]["trigger_observation_uids"] == ["obs-1", "obs-2", "obs-3"]


def test_incomplete_r1_does_not_create_expert_queue():
    try:
        MODULE.generate([meta("one"), meta("two")], [label("one", "WRONG")])
    except ValueError as exc:
        assert "complete" in str(exc)
    else:
        raise AssertionError("expert queue must wait for complete R1")


def test_invalid_r1_combination_does_not_create_expert_queue():
    invalid = label("one", "WRONG")
    invalid["evidence_sufficient"] = "NO"
    try:
        MODULE.generate([meta("one")], [invalid])
    except ValueError as exc:
        assert "requires UNCLEAR" in str(exc)
    else:
        raise AssertionError("expert queue must validate R1 semantics")
