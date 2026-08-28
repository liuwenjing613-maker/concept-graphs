from __future__ import annotations

from scripts.validate_unified_vlm_v2 import (
    _candidate_event_context,
    _shared_event_frame,
    output_schema,
    validate_output,
)


IDENTITY = ("E0", "E1", "SEPARATE", "UNRESOLVED")
SEMANTIC = ("L0", "L1", "UNRESOLVED", "NOT_EVALUATED")


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
        _output("E1", "NOT_EVALUATED"), IDENTITY, SEMANTIC
    ) == []


def test_cross_field_validator_fails_closed_on_compound_output() -> None:
    errors = validate_output(_output("E1", "L1"), IDENTITY, SEMANTIC)
    assert "non-E0 identity requires semantic_target NOT_EVALUATED" in errors


def test_unresolved_and_missing_evidence_are_biconditional() -> None:
    assert validate_output(
        _output("UNRESOLVED", "NOT_EVALUATED", "WIDER_CONTEXT_NEEDED"),
        IDENTITY,
        SEMANTIC,
    ) == []
    errors = validate_output(
        _output("E0", "L0", "CURRENT_OBJECT_UNCLEAR"), IDENTITY, SEMANTIC
    )
    assert "missing_evidence must be non-NONE iff one target is UNRESOLVED" in errors


def test_unavailable_alias_is_rejected() -> None:
    errors = validate_output(_output("E2", "NOT_EVALUATED"), IDENTITY, SEMANTIC)
    assert "identity_target is unavailable in this frozen case" in errors


def test_i1_prefers_nearest_frame_where_a_and_e0_are_both_visible() -> None:
    anchors = [
        {"obs_uid": "scene_f000004_r0019"},
        {"obs_uid": "scene_f000000_r0021"},
    ]
    event_core = [{"obs_uid": "scene_f000000_r0020"}]
    assert _shared_event_frame(anchors, event_core, trigger_frame=4) == 0


def test_i1_defer_condition_when_a_and_e0_have_no_shared_frame() -> None:
    anchors = [{"obs_uid": "scene_f000004_r0019"}]
    event_core = [{"obs_uid": "scene_f000000_r0020"}]
    assert _shared_event_frame(anchors, event_core, trigger_frame=4) is None


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
