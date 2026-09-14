import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL = Path(__file__).with_name("generate_validation_gate_endpoint_r2.py")
REPOSITORY = Path(__file__).parents[1] / "scripts" / "generate_validation_gate_endpoint_r2.py"
PATH = LOCAL if LOCAL.exists() else REPOSITORY
SPEC = spec_from_file_location("generate_validation_gate_endpoint_r2", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixture():
    worklist = []
    labels = []

    def add(scene, state, error_type, number):
        uid = f"{scene}-{state}-{error_type}-{number}"
        worklist.append(
            {
                "annotation_unit": "incident",
                "scene_id": scene,
                "incident_uid": uid,
                "case_uid": uid,
                "case_dir": f"/tmp/cases/{scene}/{uid}",
            }
        )
        labels.append(
            {
                "scene_id": scene,
                "incident_uid": uid,
                "reviewer_id": "R1",
                "evidence_sufficient": "NO" if state == "UNCLEAR" else "YES",
                "final_state": state,
                "final_error_type": error_type,
                "review_seconds": 10,
                "notes": None,
            }
        )

    for scene, count in (("room0", 15), ("office0", 8)):
        for number in range(count):
            add(scene, "CORRECT", "NOT_APPLICABLE", number)
    error_types = (
        "FALSE_MERGE",
        "FALSE_SPLIT",
        "SPURIOUS_OBJECT",
        "GEOMETRY_CORRUPTION",
        "SEMANTIC_IDENTITY_ERROR",
    )
    for scene in ("room0", "office0"):
        for error_type in error_types:
            for number in range(2):
                add(scene, "WRONG", error_type, number)
    for scene, count in (("room0", 2), ("office0", 1)):
        for number in range(count):
            add(scene, "UNCLEAR", "NOT_APPLICABLE", number)
    return worklist, labels


def test_r2_subset_is_deterministic_blind_and_covers_error_types():
    worklist, labels = fixture()
    selected, design = MODULE.select_subset(worklist, labels, size=24, seed=7)
    reversed_selected, reversed_design = MODULE.select_subset(
        list(reversed(worklist)), list(reversed(labels)), size=24, seed=7
    )
    assert len(selected) == 24
    assert [MODULE.case_key(row) for row in selected] == [
        MODULE.case_key(row) for row in reversed_selected
    ]
    assert design == reversed_design
    assert all(not (set(row) & MODULE.LABEL_FIELDS) for row in selected)
    assert all(row["r2_r1_answers_exposed_to_page"] is False for row in selected)

    label_index = {MODULE.case_key(row): row for row in labels}
    selected_labels = [label_index[MODULE.case_key(row)] for row in selected]
    assert set(row["final_error_type"] for row in selected_labels if row["final_state"] == "WRONG") == {
        "FALSE_MERGE",
        "FALSE_SPLIT",
        "SPURIOUS_OBJECT",
        "GEOMETRY_CORRUPTION",
        "SEMANTIC_IDENTITY_ERROR",
    }
    assert dict(MODULE.Counter(row["scene_id"] for row in selected)) == design["scene_targets"]


def test_r2_requires_complete_frozen_r1_census():
    worklist, labels = fixture()
    try:
        MODULE.select_subset(worklist, labels[:-1], size=24, seed=7)
    except ValueError as exc:
        assert "exactly cover" in str(exc)
    else:
        raise AssertionError("incomplete R1 freeze must be rejected")
