from __future__ import annotations

from scripts.evaluate_revision_identity_shadow_dev import _constraint_compatibility


def _create_reference():
    return {
        "type": "CREATE_INSTANCE",
        "obs_uid": "obs-1",
        "applies_at_event_uid": "event-1",
        "active_from_sequence": 7,
        "active_until_sequence": 7,
        "created_entity_uid": "entity-created",
        "created_lineage_uid": "lineage-created",
    }


def test_create_instance_safety_strengthening_is_execution_compatible():
    reference = _create_reference()
    candidate = {
        **reference,
        "created_identity_uid": "lineage-created",
        "separate_from_identity_uids": ["other-identity"],
    }

    result = _constraint_compatibility(candidate, reference)

    assert result["compatible"] is True
    assert result["relation"] == "COMPATIBLE_SAFETY_STRENGTHENING"
    assert result["candidate_extra_separation_binding"] is True
    assert result["candidate_created_identity_bound"] is True


def test_create_instance_different_anchor_is_not_compatible():
    reference = _create_reference()
    candidate = {**reference, "obs_uid": "obs-other"}

    result = _constraint_compatibility(candidate, reference)

    assert result["compatible"] is False
    assert result["mismatched_fields"] == ["obs_uid"]


def test_assign_observation_target_must_match():
    reference = {
        "type": "ASSIGN_OBSERVATION",
        "obs_uid": "obs-1",
        "applies_at_event_uid": "event-1",
        "active_from_sequence": 3,
        "active_until_sequence": 3,
        "target_entity_uid": "entity-a",
        "target_lineage_uid": "lineage-a",
        "target_origin_obs_uid": "origin-a",
    }
    exact = _constraint_compatibility(dict(reference), reference)
    wrong = _constraint_compatibility(
        {**reference, "target_lineage_uid": "lineage-b"}, reference
    )

    assert exact["compatible"] is True
    assert exact["relation"] == "EXACT_EXECUTION_SEMANTICS"
    assert wrong["compatible"] is False
    assert wrong["mismatched_fields"] == ["target_lineage_uid"]
