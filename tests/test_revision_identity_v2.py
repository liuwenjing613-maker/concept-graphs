from __future__ import annotations

import pytest

from conceptgraph.revision.constraints import (
    CandidateTarget,
    ConstraintAction,
    ConstraintEngine,
    SparseRepairConstraint,
)
from conceptgraph.revision.identity import (
    BoundaryDisposition,
    IdentityBoundary,
    IdentityRecord,
    IdentityRelation,
    assess_identity_boundaries,
    merge_identity_metadata,
    record_for_object,
    write_identity_record,
)
from conceptgraph.revision.sparse_replay import (
    ReplayComponentPolicy,
    persistent_instance_boundary_reason,
    resolve_persistent_instance_boundary_match_detailed,
)


def _record(*identities: str, complete: bool = True) -> IdentityRecord:
    return IdentityRecord.build(
        provenance_lineage_uids=identities,
        effective_identity_uids=identities,
        complete=complete,
        source="test",
    )


def test_effective_redirect_does_not_rewrite_immutable_provenance() -> None:
    obj = {"obs_uids": ["obs_a"]}
    redirected = IdentityRecord.build(
        provenance_lineage_uids=("provenance_a",),
        effective_identity_uids=("identity_target",),
        evidence_observation_uids=("obs_a",),
        complete=True,
        source="test_redirect",
    )
    write_identity_record(obj, redirected)

    recovered = record_for_object(
        obj,
        lambda obs_uid: ("provenance_a",) if obs_uid == "obs_a" else (),
    )

    assert recovered.provenance_lineage_uids == ("provenance_a",)
    assert recovered.effective_identity_uids == ("identity_target",)
    assert obj["revision_lineage_uids"] == ["identity_target"]


def test_pair_boundary_vetoes_only_the_asserted_different_pair() -> None:
    boundary = IdentityBoundary.build(
        left_identity_uids=("created_outlet",),
        right_identity_uids=("existing_outlet",),
        evidence_refs=("constraint_1",),
    )

    crossed = assess_identity_boundaries(
        _record("created_outlet"),
        _record("existing_outlet"),
        (boundary,),
    )
    unrelated = assess_identity_boundaries(
        _record("chair_a"),
        _record("chair_b"),
        (boundary,),
    )

    assert crossed.disposition == BoundaryDisposition.VETO
    assert crossed.identity_relation == IdentityRelation.DIFFERENT
    assert crossed.crossed_boundary_uids == (boundary.boundary_uid,)
    assert unrelated.disposition == BoundaryDisposition.ALLOW


def test_missing_identity_is_unknown_and_never_a_negative_veto() -> None:
    boundary = IdentityBoundary.build(
        left_identity_uids=("created",),
        right_identity_uids=("target",),
    )
    missing = IdentityRecord.build(
        provenance_lineage_uids=(),
        effective_identity_uids=(),
        complete=False,
    )

    assessment = assess_identity_boundaries(
        missing,
        _record("target"),
        (boundary,),
    )

    assert assessment.disposition == BoundaryDisposition.UNKNOWN
    assert not assessment.veto
    assert persistent_instance_boundary_reason((), ("target",), ("created",)) is None


def test_boundary_resolver_uses_known_safe_alternative() -> None:
    boundary = IdentityBoundary.build(
        left_identity_uids=("created",),
        right_identity_uids=("wrong_target",),
    )
    candidates = [
        CandidateTarget.build(
            index=0,
            entity_uid="wrong",
            lineage_uids=("wrong_target",),
            provenance_lineage_uids=("wrong_provenance",),
            member_obs_uids=("obs_wrong",),
            score=0.95,
        ),
        CandidateTarget.build(
            index=1,
            entity_uid="safe",
            lineage_uids=("safe_identity",),
            provenance_lineage_uids=("safe_provenance",),
            member_obs_uids=("obs_safe",),
            score=0.90,
        ),
    ]

    resolution = resolve_persistent_instance_boundary_match_detailed(
        0,
        candidates,
        _record("created"),
        (boundary,),
    )

    assert resolution.resolved_match == 1
    assert resolution.forbidden_indices == (0,)
    assert resolution.unknown_indices == ()
    assert resolution.overrode_match


def test_boundary_resolver_preserves_unknown_default() -> None:
    boundary = IdentityBoundary.build(
        left_identity_uids=("created",),
        right_identity_uids=("wrong_target",),
    )
    candidate = CandidateTarget.build(
        index=0,
        entity_uid="unknown",
        lineage_uids=(),
        provenance_lineage_uids=(),
        member_obs_uids=("obs_unknown",),
        score=0.95,
        identity_complete=False,
    )

    resolution = resolve_persistent_instance_boundary_match_detailed(
        0,
        (candidate,),
        _record("created"),
        (boundary,),
    )

    assert resolution.resolved_match == 0
    assert resolution.forbidden_indices == ()
    assert resolution.unknown_indices == (0,)
    assert not resolution.overrode_match


def test_legal_merge_unions_provenance_and_effective_aliases() -> None:
    target = {"obs_uids": ["obs_a"]}
    source = {"obs_uids": ["obs_b"]}
    write_identity_record(
        target,
        IdentityRecord.build(
            provenance_lineage_uids=("provenance_a",),
            effective_identity_uids=("identity_a",),
        ),
    )
    write_identity_record(
        source,
        IdentityRecord.build(
            provenance_lineage_uids=("provenance_b",),
            effective_identity_uids=("identity_b",),
        ),
    )

    merged = merge_identity_metadata(target, source)

    assert merged.provenance_lineage_uids == (
        "provenance_a",
        "provenance_b",
    )
    assert merged.effective_identity_uids == ("identity_a", "identity_b")
    assert target["revision_provenance_lineage_uids"] == [
        "provenance_a",
        "provenance_b",
    ]


def test_create_instance_carries_an_explicit_pair_identity_contract() -> None:
    constraint = SparseRepairConstraint.from_mapping(
        {
            "type": "CREATE_INSTANCE",
            "obs_uid": "obs_new",
            "applies_at_event_uid": "event_anchor",
            "created_lineage_uid": "created_identity",
            "created_identity_uid": "created_identity",
            "created_entity_uid": "created_entity",
            "separate_from_identity_uids": ["existing_identity"],
        }
    )
    decision = ConstraintEngine((constraint,)).resolve_for_observation(
        obs_uid="obs_new",
        event_uid="event_anchor",
        event_sequence=7,
        natural_match=None,
        natural_candidates=(),
    )

    assert decision.action == ConstraintAction.FORCE_CREATE
    assert decision.created_identity_uid == "created_identity"
    assert decision.separate_from_identity_uids == ("existing_identity",)
    assert constraint.effective_created_identity_uid == "created_identity"
    assert constraint.as_dict()["separate_from_identity_uids"] == ["existing_identity"]


def test_component_policy_is_explicit_and_rejects_unknown_keys() -> None:
    policy = ReplayComponentPolicy.from_value(
        {
            "positive_lineage_redirect": False,
            "create_association_boundary": True,
            "create_postprocess_boundary": False,
        }
    )
    assert policy.as_dict() == {
        "positive_lineage_redirect": False,
        "create_association_boundary": True,
        "create_postprocess_boundary": False,
    }
    with pytest.raises(ValueError, match="unknown replay component policy"):
        ReplayComponentPolicy.from_value({"magic": True})
