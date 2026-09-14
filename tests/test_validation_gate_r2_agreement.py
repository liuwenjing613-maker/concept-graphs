import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL = Path(__file__).with_name("compute_validation_gate_r2_agreement.py")
REPOSITORY = Path(__file__).parents[1] / "scripts" / "compute_validation_gate_r2_agreement.py"
PATH = LOCAL if LOCAL.exists() else REPOSITORY
SPEC = spec_from_file_location("compute_validation_gate_r2_agreement", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def label(scene, uid, reviewer, evidence, state, error_type):
    return {
        "scene_id": scene,
        "incident_uid": uid,
        "reviewer_id": reviewer,
        "evidence_sufficient": evidence,
        "final_state": state,
        "final_error_type": error_type,
        "review_seconds": 10,
        "notes": None,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_repeatability_metrics_distinguish_state_and_error_type_changes(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    keys = [("room0", f"case-{index}") for index in range(4)]
    write_jsonl(
        labels_dir / "r2_worklist.jsonl",
        [{"scene_id": scene, "incident_uid": uid} for scene, uid in keys],
    )
    r1 = [
        label(*keys[0], "R1", "YES", "CORRECT", "NOT_APPLICABLE"),
        label(*keys[1], "R1", "YES", "WRONG", "FALSE_MERGE"),
        label(*keys[2], "R1", "NO", "UNCLEAR", "NOT_APPLICABLE"),
        label(*keys[3], "R1", "YES", "WRONG", "GEOMETRY_CORRUPTION"),
    ]
    r2 = [
        label(*keys[0], "R2", "YES", "CORRECT", "NOT_APPLICABLE"),
        label(*keys[1], "R2", "YES", "WRONG", "FALSE_MERGE"),
        label(*keys[2], "R2", "YES", "CORRECT", "NOT_APPLICABLE"),
        label(*keys[3], "R2", "YES", "WRONG", "SEMANTIC_IDENTITY_ERROR"),
    ]
    r1_path = labels_dir / "labels_r1_frozen.jsonl"
    write_jsonl(r1_path, r1)
    write_jsonl(labels_dir / "labels_r2.jsonl", r2)
    metrics = MODULE.compute(tmp_path, r1_path, relationship="same-reviewer")
    assert metrics["agreement_type"] == "intra_rater_test_retest"
    assert metrics["evidence_sufficiency"]["raw_agreement"] == 0.75
    assert metrics["final_state"]["raw_agreement"] == 0.75
    assert metrics["exact_three_field_label"]["raw_agreement"] == 0.5
    assert metrics["error_type_when_both_rounds_wrong"]["raw_agreement"] == 0.5
    assert metrics["disagreement_count"] == 2
    assert metrics["final_state_transitions"] == {
        "CORRECT -> CORRECT": 1,
        "UNCLEAR -> CORRECT": 1,
        "WRONG -> WRONG": 2,
    }
    assert metrics["r1_state_stability_in_stratified_subset"]["WRONG"]["same_state_rate"] == 1.0
    assert metrics["r2_review_seconds"]["median"] == 10.0


def test_kappa_perfect_constant_labels_is_one():
    assert MODULE.cohen_kappa(["YES", "YES"], ["YES", "YES"]) == 1.0
