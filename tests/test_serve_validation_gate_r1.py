from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL = Path(__file__).with_name("serve_validation_gate_r1.py")
REPOSITORY = Path(__file__).parents[1] / "scripts" / "serve_validation_gate_r1.py"
PATH = LOCAL if LOCAL.exists() else REPOSITORY
SPEC = spec_from_file_location("serve_validation_gate_r1", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_payload():
    return {
        "reviewer_id": "someone-else",
        "evidence_sufficient": "YES",
        "finding_correct": "YES",
        "root_stage_correct": "YES",
        "physical_interpretation": "同一物体的重复 proposal",
        "downstream_harm": "LOCAL_WEIGHTING_BIAS",
        "harm_confidence": "4",
        "repair_action": "DROP_OBSERVATION",
        "repair_locality": "LOCAL",
        "repair_confidence": 5,
        "alternative_explanation": "",
        "review_seconds": 42.25,
        "notes": "",
    }


def test_validation_forces_r1_and_normalizes_values():
    result = MODULE.validate_label(valid_payload())
    assert result["reviewer_id"] == "R1"
    assert result["harm_confidence"] == 4
    assert result["repair_confidence"] == 5
    assert result["alternative_explanation"] is None
    assert result["notes"] is None


def test_validation_rejects_incomplete_or_invalid_labels():
    for field, value in (
        ("finding_correct", ""),
        ("harm_confidence", 7),
        ("physical_interpretation", "  "),
    ):
        payload = valid_payload()
        payload[field] = value
        try:
            MODULE.validate_label(payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {field} must be rejected")
