from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conceptgraph.vlm_repair.evidence import EndpointEvidence, EvidenceError
from conceptgraph.vlm_repair.overlay import apply_repairs_to_bundle
from conceptgraph.vlm_repair.policy import (
    assess_execution,
    validate_against_evidence,
    validate_audit_against_evidence,
)
from conceptgraph.vlm_repair.schemas import (
    Diagnosis,
    EvidenceAudit,
    SchemaError,
    Verification,
    extract_json_object,
)
from conceptgraph.vlm_repair.runner import _complete_validated


def _audit(**overrides):
    payload = {
        "schema_version": "1.0.0",
        "target_alias": "O1",
        "best_physical_identity": "projector",
        "checks": {
            "REAL_USABLE_OBJECT": {"probability": 0.95, "evidence": "solid housing"},
            "SAVED_LABEL_MATCH": {"probability": 0.05, "evidence": "visible lens"},
            "SINGLE_PHYSICAL_OBJECT": {"probability": 0.95, "evidence": "one housing"},
            "GEOMETRY_COMPLETE_AND_WELL_PLACED": {
                "probability": 0.9,
                "evidence": "compact complete cloud",
            },
            "MEMBERSHIP_COHERENT": {"probability": 0.9, "evidence": "views agree"},
            "NOT_BACKGROUND_OR_NOISE": {"probability": 0.98, "evidence": "3D object"},
        },
        "context_relations": [
            {
                "other_alias": "O2",
                "same_physical_object_probability": 0.01,
                "relationship": "DIFFERENT_OBJECT",
                "evidence": "different identity and location",
            }
        ],
        "hypothesis_probabilities": {
            "CORRECT": 0.03,
            "FALSE_MERGE": 0.02,
            "FALSE_SPLIT": 0.01,
            "SPURIOUS_OBJECT": 0.01,
            "WRONG_MEMBERSHIP": 0.02,
            "GEOMETRY_CORRUPTION": 0.03,
            "SEMANTIC_IDENTITY_ERROR": 0.96,
            "OTHER": 0.01,
        },
        "leading_hypothesis": "SEMANTIC_IDENTITY_ERROR",
        "evidence_for_wrong": ["lens contradicts speaker"],
        "evidence_for_correct": ["saved observations agree"],
        "evidence_gaps": [],
        "audit_summary": "Coherent projector geometry with a wrong saved label.",
    }
    payload.update(overrides)
    return EvidenceAudit.from_mapping(payload)


def _diagnosis(**overrides):
    payload = {
        "schema_version": "1.0.0",
        "target_alias": "O1",
        "evidence_sufficient": True,
        "final_state": "WRONG",
        "error_type": "SEMANTIC_IDENTITY_ERROR",
        "confidence": 0.91,
        "physical_identity": "projector",
        "diagnosis": "The lens and housing are a projector, not a speaker.",
        "repair": {
            "action": "RELABEL",
            "target_alias": "O1",
            "new_label": "projector",
            "other_alias": None,
            "member_view_groups": {},
            "rationale": "Geometry is coherent and only the identity is wrong.",
        },
    }
    payload.update(overrides)
    return Diagnosis.from_mapping(payload)


def _packet(tmp_path: Path, *, include_human_label: bool = False) -> EndpointEvidence:
    image_names = [
        "detail.png",
        "relative.png",
        "context.png",
        "mask.png",
        "panel.png",
        "timeline.jpg",
    ]
    hashes = {}
    for index, name in enumerate(image_names):
        content = b"not-a-real-png-but-hash-bound-" + bytes([index])
        (tmp_path / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    payload = {
        "schema_version": "2.1.0",
        "scene_id": "room0",
        "case_uid": "incident_test",
        "incident": {"final_owner_uids": ["uid-1"]},
        "evidence_contract": {
            "fidelity_status": "TRACEABLE",
            "artifact_hashes_match": True,
            "exact_final_map_linkage": True,
            "critical_gaps": [],
        },
        "final_objects": [
            {
                "object_uid": "uid-1",
                "object_alias": "O1",
                "endpoint_role": "INCIDENT_FINAL_OWNER",
                "class_name": "speaker",
                "member_count": 2,
                "unique_frame_count": 2,
                "n_points": 6,
                "bbox_extent": [1, 1, 1],
                "observed_class_histogram": {"speaker": 2},
            },
            {
                "object_uid": "uid-2",
                "object_alias": "O2",
                "endpoint_role": "CONTEXT_CANDIDATE_FINAL_OBJECT",
                "class_name": "tv",
                "member_count": 3,
                "unique_frame_count": 3,
                "n_points": 6,
                "bbox_extent": [1, 1, 1],
                "observed_class_histogram": {"tv": 3},
            },
        ],
        "representative_views": [
            {
                "obs_uid": "obs-1",
                "object_uids": ["uid-1"],
                "class_name": "speaker",
                "selection_reasons": ["earliest_creation_view"],
                "assets": {"context_crop": "context.png", "masked_crop": "mask.png"},
            }
        ],
        "assets": {
            "final_object_geometry": ["relative.png", "detail.png"],
            "trigger_observation_panels": ["panel.png"],
            "timeline": "timeline.jpg",
        },
        "displayed_asset_sha256": hashes,
    }
    if include_human_label:
        payload["final_state"] = "WRONG"
    (tmp_path / "review_evidence.json").write_text(json.dumps(payload), encoding="utf-8")
    return EndpointEvidence.load(tmp_path)


def test_json_parser_accepts_json_and_one_fence_only():
    assert extract_json_object('{"x": 1}') == {"x": 1}
    assert extract_json_object('```json\n{"x": 1}\n```') == {"x": 1}
    with pytest.raises(SchemaError):
        extract_json_object('{"x": 1}\nextra')


def test_schema_failure_gets_one_vlm_format_retry():
    class FakeClient:
        def __init__(self):
            self.responses = ["not-json", '{"x": 1}']

        def complete(self, **_kwargs):
            text = self.responses.pop(0)
            return SimpleNamespace(
                text=text,
                model="fake",
                response_id=None,
                usage={},
                elapsed_seconds=0.0,
            )

    parsed, attempts = _complete_validated(
        client=FakeClient(),
        system_prompt="system",
        user_prompt="user",
        images=[],
        parse_and_validate=lambda text: extract_json_object(text),
    )
    assert parsed == {"x": 1}
    assert len(attempts) == 2


def test_cross_field_schema_blocks_unsafe_answers():
    diagnosis = _diagnosis()
    assert diagnosis.repair.action == "RELABEL"
    with pytest.raises(SchemaError):
        _diagnosis(final_state="CORRECT")
    with pytest.raises(SchemaError):
        _diagnosis(evidence_sufficient=False)


def test_forced_audit_requires_all_checks_and_exact_context(tmp_path: Path):
    evidence = _packet(tmp_path)
    audit = _audit()
    validate_audit_against_evidence(evidence, audit)
    incomplete = audit.as_dict()
    del incomplete["checks"]["SAVED_LABEL_MATCH"]
    with pytest.raises(SchemaError):
        EvidenceAudit.from_mapping(incomplete)
    with pytest.raises(SchemaError):
        validate_audit_against_evidence(
            evidence, _audit(context_relations=[])
        )


def test_evidence_is_hash_bound_and_label_blind(tmp_path: Path):
    evidence = _packet(tmp_path)
    images = evidence.select_images(max_images=4)
    assert evidence.target_alias == "O1"
    assert len(images) == 4
    assert [image.role for image in images] == [
        "final_geometry_1",
        "final_geometry_2",
        "trigger_panel_Q1",
        "representative_timeline",
    ]
    summary = evidence.summary_text(images)
    assert "exact_saved_label_support=" in summary
    assert "Deterministic endpoint-to-context geometry" in summary
    (tmp_path / "detail.png").write_bytes(b"tampered")
    with pytest.raises(EvidenceError):
        evidence.select_images(max_images=4)


def test_evidence_rejects_human_label_fields(tmp_path: Path):
    with pytest.raises(EvidenceError):
        _packet(tmp_path, include_human_label=True)


def test_verified_semantic_repair_is_executable(tmp_path: Path):
    evidence = _packet(tmp_path)
    diagnosis = _diagnosis()
    validate_against_evidence(evidence, diagnosis)
    verification = Verification.from_mapping(
        {
            "approve": True,
            "confidence": 0.9,
            "diagnosis_supported": True,
            "action_supported": True,
            "reason": "The lens and housing directly support projector relabeling.",
        }
    )
    decision = assess_execution(evidence, diagnosis, verification)
    assert decision.executable is True
    assert decision.status == "APPROVED_FOR_DERIVED_MAP"


def _object(label: str, offset: float, obs_uid: str):
    points = np.asarray([[offset, 0, 0], [offset + 1, 1, 1]], dtype=np.float64)
    return {
        "id": int(offset),
        "class_name": label,
        "num_detections": 1,
        "image_idx": [int(offset)],
        "mask_idx": [0],
        "class_id": [1],
        "conf": [0.9],
        "obs_uids": [obs_uid],
        "clip_ft": np.ones((1, 2), dtype=np.float32),
        "pcd_np": points,
        "pcd_color_np": np.ones_like(points),
        "bbox_np": np.zeros((8, 3)),
        "n_points": len(points),
    }


def _member(uid: str, index: int, obs_uid: str):
    return {
        "object_uid": uid,
        "current_object_index": index,
        "class_name": "old",
        "member_observation_uids": [obs_uid],
        "parent_or_merged_from_object_uids": [],
        "num_detections": 1,
        "n_points": 2,
    }


def test_overlay_relabels_without_mutating_source():
    bundle = {"objects": [_object("speaker", 0, "obs-1")], "edges": {}}
    membership = [_member("uid-1", 0, "obs-1")]
    derived, derived_membership, reports = apply_repairs_to_bundle(
        bundle,
        membership,
        [{"case_uid": "c1", "action": "RELABEL", "target_uid": "uid-1", "new_label": "projector"}],
    )
    assert bundle["objects"][0]["class_name"] == "speaker"
    assert derived["objects"][0]["class_name"] == "projector"
    assert derived_membership[0]["class_name"] == "projector"
    assert reports[0]["apply_status"] == "APPLIED"


def test_overlay_merges_and_refreshes_membership():
    bundle = {
        "objects": [_object("chair", 0, "obs-1"), _object("chair", 2, "obs-2")],
        "edges": {},
    }
    membership = [_member("uid-1", 0, "obs-1"), _member("uid-2", 1, "obs-2")]
    derived, derived_membership, reports = apply_repairs_to_bundle(
        bundle,
        membership,
        [
            {
                "case_uid": "c1",
                "action": "MERGE_WITH",
                "target_uid": "uid-1",
                "other_uid": "uid-2",
                "new_label": None,
            }
        ],
    )
    assert len(derived["objects"]) == 1
    assert derived["objects"][0]["num_detections"] == 2
    assert set(derived_membership[0]["member_observation_uids"]) == {"obs-1", "obs-2"}
    assert reports[0]["apply_status"] == "APPLIED"
