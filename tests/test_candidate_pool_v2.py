from __future__ import annotations

from pathlib import Path

from conceptgraph.revision.online_mvp import (
    ActiveStateResolver,
    EvidenceRouter,
    LiveDependencyTracker,
    LiveEvidenceLedger,
    ObjectTicket,
    ReviewContext,
    SubIssue,
    TaskContext,
    TicketStore,
)


def _frame(index: int) -> str:
    return f"run_f{index:06d}"


def _obs(index: int, detection: int = 0) -> str:
    return f"run_f{index:06d}_r{detection:04d}"


def _version(
    object_uid: str,
    version: int,
    frame: int,
    trigger: str,
    members: list[str],
    *,
    status: str = "active",
    lineage: str | None = None,
) -> dict:
    return {
        "object_version_uid": f"{object_uid}@v{version}",
        "object_uid": object_uid,
        "version": version,
        "frame_uid": _frame(frame),
        "trigger_event_uid": trigger,
        "status": status,
        "lineage_uid": lineage or f"lineage-{object_uid}",
        "member_observation_uids": members,
        "class_name": "chair",
    }


def _ledger(tmp_path: Path) -> LiveEvidenceLedger:
    ledger = LiveEvidenceLedger(tmp_path)
    ledger.frames = {
        index: {
            "frame_uid": _frame(index),
            "frame_idx": index,
            "rgb_path": f"rgb/{index}.jpg",
        }
        for index in range(6)
    }
    return ledger


def _add_version(ledger: LiveEvidenceLedger, row: dict) -> None:
    ledger.object_versions[row["object_version_uid"]] = row
    ledger.versions_for_object.setdefault(row["object_uid"], []).append(row)
    ledger.versions_for_lineage.setdefault(row["lineage_uid"], []).append(row)


def _state_case(tmp_path: Path):
    ledger = _ledger(tmp_path)
    anchor, candidate_ref = _obs(1), _obs(0)
    association_uid, create_uid = "run_e00000001", "run_e00000002"
    ledger.observations = {
        anchor: {
            "obs_uid": anchor,
            "frame_uid": _frame(1),
            "status": "kept",
            "processed_mask_ref": {"path": "mask.npz"},
        },
        candidate_ref: {
            "obs_uid": candidate_ref,
            "frame_uid": _frame(0),
            "status": "kept",
            "processed_mask_ref": {"path": "mask.npz"},
        },
    }
    ledger.associations[association_uid] = {
        "event_uid": association_uid,
        "event_sequence": 1,
        "frame_uid": _frame(1),
        "obs_uid": anchor,
        "decision": "CREATE_OBJECT",
        "target_object_uid": "created",
        "target_object_version_after": "created@v1",
        "mapping_event_uid": create_uid,
        "object_uids_before": ["candidate"],
        "candidate_object_version_uids": ["candidate@v1"],
        "top_candidates": [{"object_uid": "candidate", "aggregate_score": 1.19}],
    }
    ledger.mapping_events[create_uid] = {
        "event_uid": create_uid,
        "event_sequence": 2,
        "frame_uid": _frame(1),
        "event_type": "OBJECT_CREATE",
        "object_uid": "created",
    }
    _add_version(
        ledger,
        _version("candidate", 1, 0, "run_e00000000", [candidate_ref]),
    )
    _add_version(ledger, _version("created", 1, 1, create_uid, [anchor]))
    issue = SubIssue.build(
        family="NEAR_THRESHOLD_CREATE",
        anchor_event_uid=association_uid,
        anchor_obs_uid=anchor,
        detected_frame=1,
        detected_sequence=1,
        object_uids=("created", "candidate"),
        lineage_uids=("lineage-created", "lineage-candidate"),
        raw_signals={"top1_score": 1.19, "sim_threshold": 1.2},
    )
    store = TicketStore()
    ticket = store.upsert(issue)
    return ledger, store, ticket, issue, anchor, candidate_ref


def _refresh(
    ledger: LiveEvidenceLedger,
    store: TicketStore,
    *,
    sequence: int,
    frame: int,
    mode: str = "active",
) -> None:
    store.refresh(
        ledger=ledger,
        tracker=LiveDependencyTracker(),
        task_context=TaskContext(),
        stop_sequence=sequence,
        cutoff_frame=frame,
        routing_mode=mode,
    )


def _merge_into_candidate(
    ledger: LiveEvidenceLedger,
    anchor: str,
    candidate_ref: str,
    *,
    sequence: int = 3,
    frame: int = 2,
) -> None:
    event_uid = f"run_e{sequence:08d}"
    ledger.mapping_events[event_uid] = {
        "event_uid": event_uid,
        "event_sequence": sequence,
        "frame_uid": _frame(frame),
        "event_type": "OBJECT_MERGE",
        "source_object_uid": "created",
        "target_object_uid": "candidate",
    }
    _add_version(
        ledger,
        _version("created", 2, frame, event_uid, [anchor], status="merged"),
    )
    _add_version(
        ledger,
        _version(
            "candidate",
            2,
            frame,
            event_uid,
            [anchor, candidate_ref],
        ),
    )


def _update_candidate(
    ledger: LiveEvidenceLedger,
    anchor: str,
    candidate_ref: str,
    *,
    sequence: int = 4,
    frame: int = 3,
) -> None:
    event_uid = f"run_e{sequence:08d}"
    ledger.mapping_events[event_uid] = {
        "event_uid": event_uid,
        "event_sequence": sequence,
        "frame_uid": _frame(frame),
        "event_type": "OBJECT_UPDATE",
        "object_uid": "candidate",
    }
    _add_version(
        ledger,
        _version(
            "candidate",
            3,
            frame,
            event_uid,
            [anchor, candidate_ref],
        ),
    )


def test_active_state_resolver_follows_native_merge_redirect(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    source_obs, target_obs = _obs(0), _obs(0, 1)
    merge_uid = "run_e00000010"
    ledger.mapping_events[merge_uid] = {
        "event_uid": merge_uid,
        "event_sequence": 10,
        "frame_uid": _frame(2),
        "event_type": "OBJECT_MERGE",
        "source_object_uid": "source",
        "target_object_uid": "target",
    }
    _add_version(ledger, _version("source", 2, 2, merge_uid, [source_obs], status="merged"))
    _add_version(
        ledger,
        _version("target", 2, 2, merge_uid, [source_obs, target_obs], status="active"),
    )
    resolver = ActiveStateResolver(ledger, cutoff_frame=2, cutoff_sequence=10)
    assert resolver.active_object_uid("source") == "target"
    assert resolver.owner_for_observation(source_obs) == "target"
    assert resolver.active_version_for_object("source")["object_version_uid"] == "target@v2"


def test_no_post_event_version_enters_pool_as_open_uncertain(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    anchor, candidate_ref = _obs(1), _obs(0)
    assoc_uid, create_uid = "run_e00000001", "run_e00000002"
    ledger.observations = {
        anchor: {
            "obs_uid": anchor,
            "frame_uid": _frame(1),
            "status": "kept",
            "processed_mask_ref": {"path": "mask.npz"},
        },
        candidate_ref: {
            "obs_uid": candidate_ref,
            "frame_uid": _frame(0),
            "status": "kept",
            "processed_mask_ref": {"path": "mask.npz"},
        },
    }
    ledger.associations[assoc_uid] = {
        "event_uid": assoc_uid,
        "event_sequence": 1,
        "frame_uid": _frame(1),
        "obs_uid": anchor,
        "decision": "CREATE_OBJECT",
        "target_object_uid": "created",
        "target_object_version_after": "created@v1",
        "mapping_event_uid": create_uid,
        "object_uids_before": ["candidate"],
        "candidate_object_version_uids": ["candidate@v1"],
        "top_candidates": [{"object_uid": "candidate", "aggregate_score": 1.19}],
    }
    ledger.mapping_events[create_uid] = {
        "event_uid": create_uid,
        "event_sequence": 2,
        "frame_uid": _frame(1),
        "event_type": "OBJECT_CREATE",
        "object_uid": "created",
    }
    _add_version(ledger, _version("candidate", 1, 0, "run_e00000000", [candidate_ref]))
    _add_version(ledger, _version("created", 1, 1, create_uid, [anchor]))
    issue = SubIssue.build(
        family="NEAR_THRESHOLD_CREATE",
        anchor_event_uid=assoc_uid,
        anchor_obs_uid=anchor,
        detected_frame=1,
        detected_sequence=1,
        object_uids=("created", "candidate"),
        lineage_uids=("lineage-created", "lineage-candidate"),
        raw_signals={"top1_score": 1.19, "sim_threshold": 1.2},
    )
    store = TicketStore()
    ticket = store.upsert(issue)
    store.refresh(
        ledger=ledger,
        tracker=LiveDependencyTracker(),
        task_context=TaskContext(),
        stop_sequence=2,
        cutoff_frame=1,
    )
    assert ticket.resolution_state == "OPEN_UNCERTAIN"
    assert ticket.has_post_event_update is False
    assert ticket.routing_state == "NO_UPDATE"
    assert ticket.pool_location == "MAIN_POOL"
    assert ticket.pool_since_frame == 1
    assert store.ordered(current_frame=1) == [ticket]


def test_changed_once_stays_in_main_pool_after_native_update(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    anchor, candidate_ref = _obs(1), _obs(0)
    issue = SubIssue.build(
        family="NEAR_THRESHOLD_CREATE",
        anchor_event_uid="run_e00000001",
        anchor_obs_uid=anchor,
        detected_frame=1,
        detected_sequence=1,
        object_uids=("created", "candidate"),
        lineage_uids=("lineage-created", "lineage-candidate"),
        raw_signals={"top1_score": 1.19, "sim_threshold": 1.2},
    )
    ledger.observations = {
        uid: {
            "obs_uid": uid,
            "frame_uid": _frame(1 if uid == anchor else 0),
            "status": "kept",
            "processed_mask_ref": {"path": "mask.npz"},
        }
        for uid in (anchor, candidate_ref)
    }
    ledger.associations[issue.anchor_event_uid] = {
        "event_uid": issue.anchor_event_uid,
        "event_sequence": 1,
        "frame_uid": _frame(1),
        "obs_uid": anchor,
        "decision": "CREATE_OBJECT",
        "target_object_uid": "created",
        "target_object_version_after": "created@v1",
        "mapping_event_uid": "run_e00000002",
        "object_uids_before": ["candidate"],
        "candidate_object_version_uids": ["candidate@v1"],
        "top_candidates": [{"object_uid": "candidate"}],
    }
    ledger.mapping_events["run_e00000002"] = {
        "event_uid": "run_e00000002",
        "event_sequence": 2,
        "frame_uid": _frame(1),
        "event_type": "OBJECT_CREATE",
    }
    ledger.mapping_events["run_e00000003"] = {
        "event_uid": "run_e00000003",
        "event_sequence": 3,
        "frame_uid": _frame(2),
        "event_type": "OBJECT_MERGE",
        "source_object_uid": "created",
        "target_object_uid": "candidate",
    }
    _add_version(ledger, _version("candidate", 1, 0, "run_e00000000", [candidate_ref]))
    _add_version(ledger, _version("created", 1, 1, "run_e00000002", [anchor]))
    _add_version(ledger, _version("created", 2, 2, "run_e00000003", [anchor], status="merged"))
    _add_version(ledger, _version("candidate", 2, 2, "run_e00000003", [anchor, candidate_ref]))
    store = TicketStore()
    ticket = store.upsert(issue)
    store.refresh(
        ledger=ledger,
        tracker=LiveDependencyTracker(),
        task_context=TaskContext(),
        stop_sequence=3,
        cutoff_frame=2,
    )
    assert ticket.resolution_state == "OPEN"
    assert ticket.routing_state == "CHANGED_UNSTABLE"
    assert ticket.resolved_by is None
    assert ticket.pool_location == "MAIN_POOL"
    assert store.ordered(current_frame=2) == [ticket]


def test_relevant_update_with_same_signature_is_persistent(tmp_path: Path) -> None:
    ledger, store, ticket, _, anchor, _ = _state_case(tmp_path)
    _refresh(ledger, store, sequence=2, frame=1)
    update_uid = "run_e00000003"
    ledger.mapping_events[update_uid] = {
        "event_uid": update_uid,
        "event_sequence": 3,
        "frame_uid": _frame(2),
        "event_type": "OBJECT_UPDATE",
        "object_uid": "created",
    }
    _add_version(ledger, _version("created", 2, 2, update_uid, [anchor]))
    _refresh(ledger, store, sequence=3, frame=2)
    assert ticket.routing_state == "PERSISTENT"
    assert ticket.latest_reconfirmed is True
    assert ticket.pool_location == "MAIN_POOL"


def test_two_stable_changed_updates_enter_audit_pool_in_active_mode(
    tmp_path: Path,
) -> None:
    ledger, store, ticket, _, anchor, candidate_ref = _state_case(tmp_path)
    _refresh(ledger, store, sequence=2, frame=1)
    _merge_into_candidate(ledger, anchor, candidate_ref)
    _refresh(ledger, store, sequence=3, frame=2)
    assert ticket.routing_state == "CHANGED_UNSTABLE"
    _update_candidate(ledger, anchor, candidate_ref)
    _refresh(ledger, store, sequence=4, frame=3)
    assert ticket.routing_state == "LIKELY_RESOLVED"
    assert ticket.routing_destination == "AUDIT_POOL"
    assert ticket.pool_location == "AUDIT_POOL"
    assert store.ordered(current_frame=3) == []
    assert store.ordered(current_frame=3, audit_slot=True) == [ticket]


def test_shadow_mode_records_audit_destination_but_keeps_validation_sample(
    tmp_path: Path,
) -> None:
    ledger, store, ticket, _, anchor, candidate_ref = _state_case(tmp_path)
    _refresh(ledger, store, sequence=2, frame=1, mode="shadow")
    _merge_into_candidate(ledger, anchor, candidate_ref)
    _refresh(ledger, store, sequence=3, frame=2, mode="shadow")
    _update_candidate(ledger, anchor, candidate_ref)
    _refresh(ledger, store, sequence=4, frame=3, mode="shadow")
    assert ticket.routing_state == "LIKELY_RESOLVED"
    assert ticket.routing_destination == "AUDIT_POOL"
    assert ticket.pool_location == "MAIN_POOL"
    assert store.ordered(current_frame=3) == [ticket]


def test_vlm_packet_keeps_event_primary_as_e0_and_current_owner_as_e1(
    tmp_path: Path,
) -> None:
    ledger, store, ticket, _, anchor, _ = _state_case(tmp_path)
    _refresh(ledger, store, sequence=2, frame=1)

    created_context = _obs(2, 1)
    ledger.observations[created_context] = {
        "obs_uid": created_context,
        "frame_uid": _frame(2),
        "status": "kept",
        "processed_mask_ref": {"path": "mask.npz"},
    }
    ledger.mapping_events["run_e00000003"] = {
        "event_uid": "run_e00000003",
        "event_sequence": 3,
        "frame_uid": _frame(2),
        "event_type": "OBJECT_UPDATE",
        "object_uid": "created",
    }
    ledger.mapping_events["run_e00000004"] = {
        "event_uid": "run_e00000004",
        "event_sequence": 4,
        "frame_uid": _frame(3),
        "event_type": "OBJECT_CREATE",
        "object_uid": "other",
    }
    _add_version(
        ledger,
        _version("created", 2, 2, "run_e00000003", [created_context]),
    )
    _add_version(
        ledger,
        _version("other", 1, 3, "run_e00000004", [anchor]),
    )
    _refresh(ledger, store, sequence=4, frame=3)

    packet = EvidenceRouter(tmp_path).build_v2(
        ticket=ticket,
        ledger=ledger,
        freeze_frame=3,
        freeze_sequence=4,
        output_dir=tmp_path / "packet",
    )
    assert packet is not None
    manifest = packet.packet_manifest
    assert manifest["alias_owner_uids"] == {"E0": "created", "E1": "other"}
    assert manifest["alias_version_uids"] == {"E0": "created@v2", "E1": "other@v1"}
    assert manifest["current_assignment"] == "E1"
    assert manifest["newer_state_available"] is True


def test_likely_resolved_retrigger_returns_to_main_pool(tmp_path: Path) -> None:
    ledger, store, ticket, issue, anchor, candidate_ref = _state_case(tmp_path)
    _refresh(ledger, store, sequence=2, frame=1)
    _merge_into_candidate(ledger, anchor, candidate_ref)
    _refresh(ledger, store, sequence=3, frame=2)
    _update_candidate(ledger, anchor, candidate_ref)
    _refresh(ledger, store, sequence=4, frame=3)
    assert ticket.routing_state == "LIKELY_RESOLVED"
    store.upsert(
        SubIssue(
            **{
                **issue.as_dict(),
                "detected_frame": 3,
                "detected_sequence": 5,
            }
        )
    )
    _refresh(ledger, store, sequence=4, frame=3)
    assert ticket.routing_state == "CHANGED_UNSTABLE"
    assert ticket.routing_reason == "CHANGED_SIGNATURE_RETRIGGERED"
    assert ticket.pool_location == "MAIN_POOL"


def test_duplicate_refresh_does_not_increase_stability_count(tmp_path: Path) -> None:
    ledger, store, ticket, _, anchor, candidate_ref = _state_case(tmp_path)
    _refresh(ledger, store, sequence=2, frame=1)
    _merge_into_candidate(ledger, anchor, candidate_ref)
    _refresh(ledger, store, sequence=3, frame=2)
    assert ticket.relevant_update_count == 1
    _refresh(ledger, store, sequence=3, frame=2)
    assert ticket.relevant_update_count == 1
    assert ticket.stable_changed_count == 1


def test_group_without_strict_majority_routes_unknown() -> None:
    resolver = object.__new__(ActiveStateResolver)
    resolver.observation_owners = {
        "a": ("owner-a",),
        "r0-1": ("owner-b",),
        "r0-2": ("owner-c",),
    }
    resolver.active_versions = {
        "owner-a": {"object_version_uid": "owner-a@v1", "class_name": "chair"},
        "owner-b": {"object_version_uid": "owner-b@v1", "class_name": "chair"},
        "owner-c": {"object_version_uid": "owner-c@v1", "class_name": "chair"},
    }
    snapshot = TicketStore.build_state_snapshot(
        ReviewContext(
            anchor_obs_uid="a",
            primary_core_obs_uids=("r0-1", "r0-2", "r0-missing"),
            alternative_core_obs_uids=(),
            event_frame_id=1,
            event_sequence=1,
        ),
        resolver,
        frame_id=2,
        event_signature={"relations": {"A_R0": "DIFFERENT"}},
        event_update_token=(("A", "old", "old@v1"),),
    )
    ticket = ObjectTicket(
        ticket_uid="ticket",
        primary_lineage_uids=(),
        primary_object_uids=(),
        first_seen_frame=1,
        last_seen_frame=1,
        event_snapshot={"update_observable": True, "comparable": True},
        event_signature={"relations": {"A_R0": "DIFFERENT"}},
    )
    state, _, _ = TicketStore._classify_routing_state(ticket, snapshot)
    assert state == "UNKNOWN"


def test_review_issue_change_resets_state_history(tmp_path: Path) -> None:
    ledger, store, ticket, issue, anchor, _ = _state_case(tmp_path)
    _refresh(ledger, store, sequence=2, frame=1)
    update_uid = "run_e00000003"
    ledger.mapping_events[update_uid] = {
        "event_uid": update_uid,
        "event_sequence": 3,
        "frame_uid": _frame(2),
        "event_type": "OBJECT_UPDATE",
        "object_uid": "created",
    }
    _add_version(ledger, _version("created", 2, 2, update_uid, [anchor]))
    _refresh(ledger, store, sequence=3, frame=2)
    assert ticket.state_history
    ledger.associations[update_uid] = {
        **ledger.associations[issue.anchor_event_uid],
        "event_uid": update_uid,
        "event_sequence": 3,
        "frame_uid": _frame(2),
        "mapping_event_uid": update_uid,
    }
    stronger = SubIssue(
        **{
            **issue.as_dict(),
            "issue_uid": "issue-stronger",
            "anchor_event_uid": update_uid,
            "detected_frame": 2,
            "detected_sequence": 3,
            "strength": 1.0,
        }
    )
    store.upsert(stronger)
    _refresh(ledger, store, sequence=3, frame=2)
    assert ticket.review_issue_uid == "issue-stronger"
    assert ticket.state_history == []
    assert any(
        event["type"] == "REVIEW_ANCHOR_CHANGED"
        for event in ticket.routing_events
    )


def test_error_tier_precedes_task_raised_impact_and_wait() -> None:
    store = TicketStore()
    high = store.upsert(
        SubIssue.build(
            family="NEAR_THRESHOLD_CREATE",
            anchor_event_uid="e-high",
            anchor_obs_uid="o-high",
            detected_frame=9,
            detected_sequence=9,
            object_uids=("high",),
            lineage_uids=("lineage-high",),
            raw_signals={"top1_score": 1.2, "sim_threshold": 1.2},
        )
    )
    low = store.upsert(
        SubIssue.build(
            family="POSTPROCESS_MERGE_CONFLICT",
            anchor_event_uid="e-low",
            anchor_obs_uid="o-low",
            detected_frame=0,
            detected_sequence=0,
            object_uids=("low",),
            lineage_uids=("lineage-low",),
            raw_signals={},
        )
    )
    for ticket in (high, low):
        ticket.pool_location = "MAIN_POOL"
        ticket.pool_since_frame = ticket.first_seen_frame
    high.error_tier, high.impact_tier = 3, 1
    low.error_tier, low.impact_tier, low.task_blocking = 1, 3, True
    assert store.ordered(current_frame=100)[0] is high
