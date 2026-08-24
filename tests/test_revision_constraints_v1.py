import pytest

from conceptgraph.revision.sparse_replay import (
    resolve_persistent_instance_boundary_match,
)

from conceptgraph.revision.constraints import (
    CandidateTarget,
    ConstraintAction,
    ConstraintConflictError,
    ConstraintEngine,
    SparseRepairConstraint,
)
from conceptgraph.revision.sparse_replay import _resolved_constraint_match
from conceptgraph.revision.sparse_replay import persistent_instance_boundary_reason


def _candidate(index=0, *, lineage="lineage_a", origin="obs_origin", entity="entity_a"):
    return CandidateTarget.build(
        index=index,
        entity_uid=entity,
        lineage_uids=[lineage],
        member_obs_uids=[origin],
        score=2.0 - index,
    )


def test_conflicting_must_and_cannot_link_is_a_hard_failure():
    shared = {
        "obs_uid": "obs_anchor",
        "target_lineage_uid": "lineage_a",
        "applies_at_event_uid": "event_anchor",
    }
    with pytest.raises(ConstraintConflictError, match="required and forbidden"):
        ConstraintEngine(
            [
                {"type": "MUST_LINK", **shared},
                {"type": "CANNOT_LINK", **shared},
            ]
        )


def test_multiple_heterogeneous_positive_targets_fail_as_constraint_conflict():
    with pytest.raises(ConstraintConflictError, match="multiple positive targets"):
        ConstraintEngine(
            [
                {
                    "type": "MUST_LINK",
                    "obs_uid": "obs_anchor",
                    "target_lineage_uid": "lineage_a",
                    "applies_at_event_uid": "event_anchor",
                },
                {
                    "type": "ASSIGN_OBSERVATION",
                    "obs_uid": "obs_anchor",
                    "target_origin_obs_uid": "obs_origin_b",
                    "applies_at_event_uid": "event_anchor",
                },
            ]
        )


def test_assign_resolves_active_lineage_without_final_trajectory():
    engine = ConstraintEngine(
        [
            {
                "type": "ASSIGN_OBSERVATION",
                "obs_uid": "obs_anchor",
                "target_lineage_uid": "lineage_a",
                "applies_at_event_uid": "event_anchor",
            }
        ]
    )
    result = engine.resolve_for_observation(
        obs_uid="obs_anchor",
        event_uid="event_anchor",
        event_sequence=7,
        natural_match=1,
        natural_candidates=[_candidate(), _candidate(1, lineage="lineage_b")],
        anchor_only=True,
    )
    assert result.action == ConstraintAction.FORCE_TARGET
    assert result.target_index == 0
    assert result.constrained


def test_unresolved_positive_target_defers_instead_of_falling_back():
    engine = ConstraintEngine(
        [
            SparseRepairConstraint.from_mapping(
                {
                    "type": "ASSIGN_OBSERVATION",
                    "obs_uid": "obs_anchor",
                    "target_lineage_uid": "missing_lineage",
                    "applies_at_event_uid": "event_anchor",
                }
            )
        ]
    )
    result = engine.resolve_for_observation(
        obs_uid="obs_anchor",
        event_uid="event_anchor",
        event_sequence=7,
        natural_match=0,
        natural_candidates=[_candidate()],
    )
    assert result.action == ConstraintAction.DEFER
    assert result.reason == "target_not_active"


def test_cannot_link_forces_create_when_only_natural_target_is_forbidden():
    engine = ConstraintEngine(
        [
            {
                "type": "CANNOT_LINK",
                "obs_uid": "obs_anchor",
                "target_origin_obs_uid": "obs_origin",
                "applies_at_event_uid": "event_anchor",
                "created_lineage_uid": "new_lineage",
                "created_entity_uid": "new_entity",
            }
        ]
    )
    result = engine.resolve_for_observation(
        obs_uid="obs_anchor",
        event_uid="event_anchor",
        event_sequence=7,
        natural_match=0,
        natural_candidates=[_candidate()],
    )
    assert result.action == ConstraintAction.FORCE_CREATE
    assert result.created_lineage_uid == "new_lineage"
    assert result.created_entity_uid == "new_entity"


def test_create_instance_forces_new_object_even_with_eligible_alternatives():
    engine = ConstraintEngine(
        [
            {
                "type": "CREATE_INSTANCE",
                "obs_uid": "obs_anchor",
                "applies_at_event_uid": "event_anchor",
                "created_lineage_uid": "new_lineage",
                "created_entity_uid": "new_entity",
            }
        ]
    )
    candidates = [
        _candidate(0),
        _candidate(1, lineage="lineage_b", origin="obs_b", entity="entity_b"),
        _candidate(2, lineage="lineage_c", origin="obs_c", entity="entity_c"),
    ]

    result = engine.resolve_for_observation(
        obs_uid="obs_anchor",
        event_uid="event_anchor",
        event_sequence=7,
        natural_match=0,
        natural_candidates=candidates,
    )

    assert result.action == ConstraintAction.FORCE_CREATE
    assert result.target_index is None
    assert result.created_lineage_uid == "new_lineage"
    assert result.created_entity_uid == "new_entity"
    assert _resolved_constraint_match(
        result,
        native_match=0,
        historical_default_match=1,
    ) is None


def test_create_instance_conflicts_with_positive_target_at_same_scope():
    with pytest.raises(ConstraintConflictError, match="existing target and a new instance"):
        ConstraintEngine(
            [
                {
                    "type": "CREATE_INSTANCE",
                    "obs_uid": "obs_anchor",
                    "applies_at_event_uid": "event_anchor",
                },
                {
                    "type": "ASSIGN_OBSERVATION",
                    "obs_uid": "obs_anchor",
                    "target_lineage_uid": "lineage_a",
                    "applies_at_event_uid": "event_anchor",
                },
            ]
        )


def test_create_instance_rejects_target_reference():
    with pytest.raises(ValueError, match="must not specify a target"):
        SparseRepairConstraint.from_mapping(
            {
                "type": "CREATE_INSTANCE",
                "obs_uid": "obs_anchor",
                "target_lineage_uid": "lineage_a",
            }
        )


def test_create_instance_boundary_only_rejects_cross_boundary_postprocess_merge():
    protected = {"lineage_created"}

    assert persistent_instance_boundary_reason(
        ["lineage_created"], ["lineage_existing"], protected
    ) == "persistent_create_instance_boundary"
    assert persistent_instance_boundary_reason(
        ["lineage_existing"], ["lineage_other"], protected
    ) is None
    assert persistent_instance_boundary_reason(
        ["lineage_created"], ["lineage_created", "lineage_other"], protected
    ) is None


def test_keep_natural_uses_native_create_not_injected_historical_target():
    engine = ConstraintEngine(
        [
            {
                "type": "CANNOT_LINK",
                "obs_uid": "obs_anchor",
                "target_origin_obs_uid": "obs_wrong",
                "applies_at_event_uid": "event_anchor",
            }
        ]
    )
    wrong = _candidate(origin="obs_wrong")
    alternative = _candidate(1, lineage="lineage_b", origin="obs_other")
    decision = engine.resolve_for_observation(
        obs_uid="obs_anchor",
        event_uid="event_anchor",
        event_sequence=7,
        natural_match=None,
        natural_candidates=[wrong, alternative],
    )

    assert decision.action == ConstraintAction.KEEP_NATURAL
    assert _resolved_constraint_match(
        decision,
        native_match=None,
        historical_default_match=wrong.index,
    ) is None


def test_no_constraint_preserves_injected_historical_default():
    decision = ConstraintEngine([]).resolve_for_observation(
        obs_uid="obs_anchor",
        event_uid="event_anchor",
        event_sequence=7,
        natural_match=None,
        natural_candidates=[],
    )
    assert decision.action == ConstraintAction.NO_CONSTRAINT
    assert _resolved_constraint_match(
        decision,
        native_match=None,
        historical_default_match=4,
    ) == 4


def test_constraint_is_inactive_outside_its_event_scope():
    engine = ConstraintEngine(
        [
            {
                "type": "MUST_LINK",
                "obs_uid": "obs_anchor",
                "target_lineage_uid": "lineage_a",
                "applies_at_event_uid": "event_anchor",
            }
        ]
    )
    result = engine.resolve_for_observation(
        obs_uid="obs_anchor",
        event_uid="event_later",
        event_sequence=8,
        natural_match=None,
        natural_candidates=[_candidate()],
    )
    assert result.action == ConstraintAction.NO_CONSTRAINT


def test_create_instance_boundary_repairs_association_with_strict_alternative():
    crossing = CandidateTarget.build(
        index=0,
        entity_uid="existing",
        lineage_uids=["lineage_existing"],
        member_obs_uids=["obs_existing"],
        score=2.0,
        eligible=True,
    )
    same_side = CandidateTarget.build(
        index=1,
        entity_uid="created",
        lineage_uids=["lineage_created"],
        member_obs_uids=["obs_created"],
        score=1.5,
        eligible=True,
    )

    resolved, forbidden, changed = resolve_persistent_instance_boundary_match(
        0,
        [crossing, same_side],
        ["lineage_created"],
        ["lineage_created"],
    )
    assert (resolved, forbidden, changed) == (1, (0,), True)

    resolved, forbidden, changed = resolve_persistent_instance_boundary_match(
        0, [crossing], ["lineage_created"], ["lineage_created"]
    )
    assert (resolved, forbidden, changed) == (None, (0,), True)

    assert resolve_persistent_instance_boundary_match(
        0, [crossing], [], ["lineage_created"]
    ) == (0, (), False)
