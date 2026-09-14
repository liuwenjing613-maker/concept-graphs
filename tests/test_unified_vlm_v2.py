from __future__ import annotations

import hashlib
import numpy as np
import json
import pytest

from conceptgraph.revision.online_mvp import LiveEvidenceLedger, OnlineEvidencePacket
from scripts.run_ali_my_new_online import (
    _bind_completion_timeline,
    _candidate_pool_passes,
    _ticket_state_from_vlm_response,
)
from scripts.validate_unified_vlm_v2 import (
    ALIAS_COLORS,
    SYSTEM_PROMPT,
    _candidate_event_context,
    _expanded_bbox,
    _h_snapshot_payload,
    _pool_route_text,
    _review_question,
    _semantic_label_text,
    _shared_event_frame,
    _validate_h_snapshot,
    PreflightDefer,
    output_schema,
    validate_output,
    write_root_html,
)


IDENTITY = ("E0", "E1", "SEPARATE", "UNRESOLVED")
SEMANTIC = ("L0", "L1", "UNRESOLVED", "NOT_APPLICABLE")


def _output(identity: str, semantic: str, missing: str = "NONE") -> dict:
    return {
        "identity_target": identity,
        "semantic_target": semantic,
        "evidence_ids": ["I1", "I2"],
        "reason": "Visible boundaries support this declarative current target state.",
        "missing_evidence": missing,
    }


def test_schema_is_declarative_and_has_no_action_or_confidence_fields() -> None:
    schema = output_schema(IDENTITY, SEMANTIC)
    properties = schema["properties"]
    assert set(properties) == {
        "identity_target",
        "semantic_target",
        "evidence_ids",
        "reason",
        "missing_evidence",
    }
    assert "selected_candidate" not in properties
    assert "confidence" not in properties


def test_cross_field_validator_accepts_identity_and_semantic_targets() -> None:
    assert validate_output(_output("E0", "L1"), IDENTITY, SEMANTIC) == []
    assert validate_output(
        _output("E1", "NOT_APPLICABLE"), IDENTITY, SEMANTIC
    ) == []


def test_cross_field_validator_fails_closed_on_compound_output() -> None:
    errors = validate_output(_output("E1", "L1"), IDENTITY, SEMANTIC)
    assert "non-E0 identity requires semantic_target NOT_APPLICABLE" in errors


def test_unresolved_and_missing_evidence_are_biconditional() -> None:
    assert validate_output(
        _output("UNRESOLVED", "NOT_APPLICABLE", "WIDER_CONTEXT_NEEDED"),
        IDENTITY,
        SEMANTIC,
    ) == []
    errors = validate_output(
        _output("E0", "L0", "CURRENT_OBJECT_UNCLEAR"), IDENTITY, SEMANTIC
    )
    assert "missing_evidence must be non-NONE iff one target is UNRESOLVED" in errors


def test_reason_must_be_a_complete_sentence() -> None:
    value = _output("E0", "L0")
    value["reason"] = "A belongs to"
    errors = validate_output(value, IDENTITY, SEMANTIC)
    assert "reason must be a complete sentence ending in punctuation" in errors


def test_unavailable_alias_is_rejected() -> None:
    errors = validate_output(_output("E2", "NOT_APPLICABLE"), IDENTITY, SEMANTIC)
    assert "identity_target is unavailable in this frozen case" in errors


def test_i1_prefers_exact_s_then_nearest_causal_shared_frame() -> None:
    anchors = [
        {"obs_uid": "scene_f000004_r0019"},
        {"obs_uid": "scene_f000000_r0021"},
    ]
    event_core = [
        {"obs_uid": "scene_f000000_r0020"},
        {"obs_uid": "scene_f000004_r0020"},
    ]
    assert _shared_event_frame(anchors, event_core, s_frame=4, d_frame=4) == 4


def test_i1_never_uses_a_shared_frame_after_d() -> None:
    anchors = [
        {"obs_uid": "scene_f000000_r0019"},
        {"obs_uid": "scene_f000006_r0019"},
    ]
    event_core = [
        {"obs_uid": "scene_f000000_r0020"},
        {"obs_uid": "scene_f000006_r0020"},
    ]
    assert _shared_event_frame(anchors, event_core, s_frame=4, d_frame=5) == 0


def test_i1_can_fall_back_when_a_and_e0_have_no_shared_frame() -> None:
    anchors = [{"obs_uid": "scene_f000004_r0019"}]
    event_core = [{"obs_uid": "scene_f000000_r0020"}]
    assert _shared_event_frame(anchors, event_core, s_frame=4, d_frame=4) is None


def test_h_snapshot_hash_is_verified_and_tampering_fails_closed() -> None:
    snapshot = {
        "schema_version": "ali_my_h_snapshot/1.0",
        "cutoff_frame": 8,
        "cutoff_sequence": 12,
        "active_object_version_uids": {"object-a": "object-a@v3"},
        "merge_redirects": {},
    }
    digest = hashlib.sha256(
        json.dumps(
            _h_snapshot_payload(snapshot),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    snapshot.update(
        {"snapshot_uid": "hsnap_" + digest[:16], "snapshot_sha256": digest}
    )
    assert _validate_h_snapshot({"h_snapshot": snapshot})["snapshot_uid"].startswith(
        "hsnap_"
    )
    snapshot["active_object_version_uids"]["object-a"] = "object-a@v4"
    with pytest.raises(PreflightDefer, match="H snapshot digest is invalid"):
        _validate_h_snapshot({"h_snapshot": snapshot})


def test_completion_c_binds_the_same_h_snapshot(tmp_path) -> None:
    h_snapshot = {
        "snapshot_uid": "hsnap_demo",
        "snapshot_sha256": "a" * 64,
        "cutoff_frame": 8,
        "cutoff_sequence": 12,
        "watermark_source": "ledger_committed",
    }
    packet = OnlineEvidencePacket(
        ticket_uid="ticket-demo",
        issue_uid="issue-demo",
        freeze_frame=8,
        freeze_sequence=12,
        evidence=None,
        association={},
        alias_version_uids={"E0": "object-a@v3"},
        allowed_image_ids=("I1", "I2", "I3"),
        packet_manifest={
            "timeline": {
                "s_frame": 2,
                "d_frame": 4,
                "h_frame": 8,
                "h_snapshot_uid": "hsnap_demo",
                "h_snapshot_sha256": "a" * 64,
            },
            "h_snapshot": h_snapshot,
        },
    )
    response = _bind_completion_timeline(
        packet=packet,
        response={"ticket_uid": "ticket-demo", "status": "VALID"},
        ledger=LiveEvidenceLedger(tmp_path),
        latest_committed=11,
        output_root=tmp_path / "online",
    )
    assert response["timeline"]["c_frame"] == 11
    assert response["timeline"]["frame_order_valid"] is True
    assert response["source_h_snapshot"]["snapshot_uid"] == "hsnap_demo"
    completion = json.loads(
        (tmp_path / "online" / "vlm" / "ticket-demo" / "completion_manifest.json").read_text()
    )
    assert completion["source_h_snapshot"]["snapshot_sha256"] == "a" * 64


def test_alias_colors_are_fixed_red_blue_green() -> None:
    assert ALIAS_COLORS["A"][0] > max(ALIAS_COLORS["A"][1:])
    assert ALIAS_COLORS["E0"][2] > max(ALIAS_COLORS["E0"][:2])
    assert ALIAS_COLORS["E1"][1] > max(ALIAS_COLORS["E1"][0], ALIAS_COLORS["E1"][2])


def test_extreme_aspect_mask_keeps_scene_context() -> None:
    mask = np.zeros((100, 400), dtype=bool)
    mask[48:52, 80:320] = True
    x0, y0, x1, y1 = _expanded_bbox(mask, 0.25)
    assert x1 - x0 >= 168
    assert y1 - y0 >= 42
    assert 0 <= x0 < x1 <= 400
    assert 0 <= y0 < y1 <= 100


def test_prompt_has_no_action_or_quality_gate_instruction() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "action" not in prompt
    assert "confidence" not in prompt
    assert "96" not in prompt
    assert "same saved frame" not in prompt
    assert "visible fragment or subregion" in prompt
    assert "two distinct physical instances" in prompt
    assert "duplicate representations" in prompt
    assert "not supporting evidence" in prompt
    assert "one complete sentence" in prompt


def test_join_candidate_question_explains_suspicion_without_declaring_answer() -> None:
    text = _review_question(
        "NEAR_THRESHOLD_CREATE",
        {"raw_signals": {"decision": "CREATE_OBJECT"}},
        {"repair_predicate": "JOIN_CANDIDATE"},
        has_e1=True,
    )
    assert "created/kept E0" in text
    assert "E1 was a recorded existing candidate" in text
    assert "only a suspicion, not ground truth" in text
    assert "choose the pre-existing E1" in text


def test_semantic_question_requires_identity_before_label() -> None:
    text = _review_question(
        "SEMANTIC_DRIFT",
        {},
        {"repair_predicate": "ADOPT_LABEL"},
        has_e1=False,
    )
    assert "First verify that A belongs to E0" in text


def test_prepare_only_is_not_recorded_as_aborted() -> None:
    assert _ticket_state_from_vlm_response({"status": "PREPARED_ONLY"}) == "WAIT_EVIDENCE"
    assert _ticket_state_from_vlm_response({"status": "API_OR_PARSE_ERROR"}) == "ABORTED"
    assert _ticket_state_from_vlm_response(
        {
            "status": "VALID",
            "output": {"identity_target": "E0", "semantic_target": "L0"},
        }
    ) == "NO_ACTION"
    assert _ticket_state_from_vlm_response(
        {
            "status": "VALID",
            "output": {"identity_target": "SEPARATE", "semantic_target": "NOT_APPLICABLE"},
        }
    ) == "DIAGNOSED"


def test_semantic_labels_are_explicit_for_review_without_image_overlay() -> None:
    text = _semantic_label_text(
        {"label_candidates": {"L0": "cabinet", "L1": "desk"}}
    )
    assert text == "L0=cabinet（当前 E0 标签） · L1=desk"


def test_pool_display_distinguishes_current_and_shadow_destination() -> None:
    manifest = {
        "routing": {"pool_location": "MAIN_POOL", "destination": "AUDIT_POOL"}
    }
    assert _pool_route_text(manifest) == "MAIN_POOL → AUDIT_POOL"


def test_root_html_distinguishes_real_vlm_results_from_prepare_only(tmp_path) -> None:
    case_dir = tmp_path / "ticket_demo"
    case_dir.mkdir()
    (case_dir / "case_manifest.json").write_text(
        json.dumps(
            {
                "issue_family": "SEMANTIC_DRIFT",
                "images": {
                    key: {"file": f"{key}.jpg", "layout": "same_frame"}
                    for key in ("I1", "I2", "I3")
                },
                "routing": {"pool_location": "MAIN_POOL"},
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "input_summary.json").write_text(
        json.dumps(
            {
                "current_assignment": "E0",
                "degraded_images": [],
                "label_candidates": {"L0": "stool", "L1": "ottoman"},
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "validation.json").write_text(
        json.dumps({"status": "VALID"}), encoding="utf-8"
    )
    (case_dir / "vlm_output.json").write_text(
        json.dumps({"identity_target": "E0", "semantic_target": "L1"}),
        encoding="utf-8",
    )

    write_root_html(tmp_path)

    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "1 例真实审核结果" in page
    assert "真实 VLM 调用" in page
    assert "尚未调用 API" not in page


def test_drain_all_candidates_falls_back_to_the_other_pool() -> None:
    assert _candidate_pool_passes(0, drain_all_candidates=False) == (False,)
    assert _candidate_pool_passes(9, drain_all_candidates=False) == (True,)
    assert _candidate_pool_passes(0, drain_all_candidates=True) == (False, True)
    assert _candidate_pool_passes(9, drain_all_candidates=True) == (True, False)


def test_root_html_respects_recorded_candidate_order(tmp_path) -> None:
    for ticket_uid in ("ticket_a", "ticket_z"):
        case_dir = tmp_path / ticket_uid
        case_dir.mkdir()
        (case_dir / "validation.json").write_text(
            json.dumps({"status": "NO_EVIDENCE_PACKET"}),
            encoding="utf-8",
        )

    write_root_html(tmp_path, ordered_ticket_uids=["ticket_z", "ticket_a"])

    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert page.index("ticket_z") < page.index("ticket_a")
    assert "#1" in page
    assert "#2" in page
    assert "案例按候选池真实派发顺序编号" in page


def test_i1_never_invents_e1_from_unbound_raw_candidate_order() -> None:
    packet = {
        "alias_version_uids": {"E0": "current@v2"},
        "candidate_alias_observation_uids": {},
    }
    observations = {
        "candidate": {
            "obs_uid": "candidate",
            "status": "kept",
            "processed_mask_ref": {"path": "mask.npz"},
        }
    }
    assert _candidate_event_context(packet, observations) == []


def test_i1_uses_only_explicit_resolver_bound_candidate_alias() -> None:
    packet = {
        "alias_version_uids": {"E0": "current@v2", "E1": "alternative@v1"},
        "candidate_alias_observation_uids": {"E1": ["candidate"]},
    }
    row = {
        "obs_uid": "candidate",
        "status": "kept",
        "processed_mask_ref": {"path": "mask.npz"},
    }
    assert _candidate_event_context(packet, {"candidate": row}) == [("E1", row)]
