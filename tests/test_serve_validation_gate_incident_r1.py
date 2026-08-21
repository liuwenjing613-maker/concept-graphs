import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL = Path(__file__).with_name("serve_validation_gate_incident_r1.py")
REPOSITORY = Path(__file__).parents[1] / "scripts" / "serve_validation_gate_incident_r1.py"
PATH = LOCAL if LOCAL.exists() else REPOSITORY
SPEC = spec_from_file_location("serve_validation_gate_incident_r1", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_payload():
    return {
        "reviewer_id": "ignored",
        "evidence_sufficient": "YES",
        "final_state": "WRONG",
        "final_error_type": "FALSE_MERGE",
        "review_seconds": 12.34,
        "notes": "",
    }


def assert_rejected(**changes):
    payload = valid_payload()
    payload.update(changes)
    try:
        MODULE.validate_label(payload)
    except ValueError:
        return
    raise AssertionError(f"invalid endpoint label accepted: {changes}")


def test_endpoint_label_contract_is_small_and_conditionally_consistent():
    result = MODULE.validate_label(valid_payload())
    assert result["reviewer_id"] == "R1"
    assert result["review_seconds"] == 12.3
    assert result["notes"] is None
    assert_rejected(evidence_sufficient="NO")
    assert_rejected(final_state="CORRECT", final_error_type="FALSE_MERGE")
    assert_rejected(final_state="WRONG", final_error_type="NOT_APPLICABLE")
    assert_rejected(final_error_type="OTHER", notes="")
    unclear = MODULE.validate_label(
        dict(
            valid_payload(),
            evidence_sufficient="NO",
            final_state="UNCLEAR",
            final_error_type="NOT_APPLICABLE",
        )
    )
    assert unclear["final_state"] == "UNCLEAR"


def test_same_contract_can_be_saved_as_blinded_r2_round():
    result = MODULE.validate_label(valid_payload(), reviewer_id="R2")
    assert result["reviewer_id"] == "R2"


def test_store_binds_incident_to_representative_packet_and_hides_checker_metadata(tmp_path):
    labels = tmp_path / "labels"
    case_dir = tmp_path / "cases" / "room0" / "finding_1"
    labels.mkdir(parents=True)
    case_dir.mkdir(parents=True)
    case_path = case_dir / "case.json"
    case_path.write_text('{"finding_uid":"finding_1","checker_id":"DET-001"}', encoding="utf-8")
    image_path = case_dir / "review_final_objects_relative.png"
    image_path.write_bytes(b"stable-image")
    review = {
        "schema_version": "2.1.0",
        "scene_id": "room0",
        "case_uid": "incident_1",
        "finding_uid": "finding_1",
        "checker_id": "DET-001",
        "stage": "detection",
        "subtype": "DUPLICATE",
        "incident": {
            "incident_uid": "incident_1",
            "representative_finding_uid": "finding_1",
            "member_finding_uids": ["finding_1", "finding_2"],
            "checker_ids": ["DET-001", "SEG-004"],
            "stages": ["detection", "segmentation"],
        },
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
            {
                "annotation_unit": "incident",
                "scene_id": "room0",
                "case_uid": "incident_1",
                "incident_uid": "incident_1",
                "representative_finding_uid": "finding_1",
                "case_dir": str(case_dir),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "2.1.0",
        "status": "READY",
        "worklist_sha256": hashlib.sha256(worklist_path.read_bytes()).hexdigest(),
        "case_count": 1,
        "cases": [
            {
                "scene_id": "room0",
                "case_uid": "incident_1",
                "incident_uid": "incident_1",
                "review_evidence_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "review_evidence_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    store = MODULE.ReviewStore(tmp_path)
    payload = store.case_payload(0)
    assert payload["case_uid"] == "incident_1"
    assert payload["assets"] == [image_path.name]
    assert "checker_id" not in payload["review_evidence"]
    assert "checker_ids" not in payload["review_evidence"]["incident"]
    image_path.write_bytes(b"tampered")
    try:
        store.case_payload(0)
    except ValueError as exc:
        assert "图片在证据生成后发生变化" in str(exc)
    else:
        raise AssertionError("tampered reviewer asset must be rejected")
