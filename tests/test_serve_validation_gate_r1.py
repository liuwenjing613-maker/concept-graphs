import hashlib
import json
import tempfile
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


def assert_rejected(**changes):
    payload = valid_payload()
    payload.update(changes)
    try:
        MODULE.validate_label(payload)
    except ValueError:
        return
    raise AssertionError(f"contradictory label must be rejected: {changes}")


def test_validation_rejects_guessing_when_evidence_is_missing():
    assert_rejected(evidence_sufficient="NO")
    result = MODULE.validate_label(
        dict(
            valid_payload(),
            evidence_sufficient="NO",
            finding_correct="UNCERTAIN",
            root_stage_correct="UNCERTAIN",
            downstream_harm="UNKNOWN",
            repair_action="NEED_MORE_VIEW",
            repair_locality="NOT_APPLICABLE",
        )
    )
    assert result["finding_correct"] == "UNCERTAIN"


def test_validation_requires_partial_evidence_note():
    assert_rejected(evidence_sufficient="PARTIAL", notes="")
    result = MODULE.validate_label(
        dict(valid_payload(), evidence_sufficient="PARTIAL", notes="缺少融合前对象点云快照")
    )
    assert result["notes"] == "缺少融合前对象点云快照"


def test_validation_rejects_unresolved_fields_under_sufficient_evidence():
    assert_rejected(finding_correct="UNCERTAIN")
    assert_rejected(root_stage_correct="UNCERTAIN")
    assert_rejected(downstream_harm="UNKNOWN")
    assert_rejected(repair_action="NEED_MORE_VIEW", repair_locality="NOT_APPLICABLE")


def test_validation_rejects_downstream_contradictions():
    assert_rejected(
        finding_correct="NO",
        root_stage_correct="YES",
        downstream_harm="NONE",
        repair_action="NONE",
        repair_locality="NOT_APPLICABLE",
    )
    assert_rejected(downstream_harm="GEOMETRY_CORRUPTION", repair_action="NONE")
    assert_rejected(
        downstream_harm="WRONG_OBSERVATION_MEMBERSHIP",
        repair_action="REASSIGN_OBSERVATION",
        repair_locality="LOCAL",
    )


def test_review_store_verifies_manifest_review_json_and_display_asset_hashes():
    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        labels = root / "labels"
        case_dir = root / "cases" / "room0" / "finding_1"
        labels.mkdir(parents=True)
        case_dir.mkdir(parents=True)
        case_json = {"finding_uid": "finding_1", "checker_id": "DET-002", "stage": "detection"}
        case_path = case_dir / "case.json"
        case_path.write_text(json.dumps(case_json), encoding="utf-8")
        image_path = case_dir / "review_final_objects_relative.png"
        image_path.write_bytes(b"stable-image")
        review = {
            "schema_version": "1.0.0",
            "scene_id": "room0",
            "case_uid": "finding_1",
            "source_case_json_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "displayed_asset_sha256": {
                image_path.name: hashlib.sha256(image_path.read_bytes()).hexdigest()
            },
        }
        review_path = case_dir / "review_evidence.json"
        review_path.write_text(json.dumps(review), encoding="utf-8")
        worklist_path = labels / "r1_worklist.jsonl"
        worklist_path.write_text(
            json.dumps(
                {"scene_id": "room0", "case_uid": "finding_1", "case_dir": str(case_dir)}
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "1.0.0",
            "status": "READY",
            "worklist_sha256": hashlib.sha256(worklist_path.read_bytes()).hexdigest(),
            "case_count": 1,
            "cases": [
                {
                    "scene_id": "room0",
                    "case_uid": "finding_1",
                    "review_evidence_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                }
            ],
        }
        (root / "review_evidence_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        store = MODULE.ReviewStore(root)
        assert store.case_payload(0)["assets"] == [image_path.name]
        image_path.write_bytes(b"tampered")
        try:
            store.case_payload(0)
        except ValueError as exc:
            assert "图片在证据生成后发生变化" in str(exc)
        else:
            raise AssertionError("tampered reviewer asset must be rejected")
