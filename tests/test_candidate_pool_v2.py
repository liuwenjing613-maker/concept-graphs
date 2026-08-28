from __future__ import annotations

from pathlib import Path

from conceptgraph.revision.online_mvp import (
    ActiveStateResolver,
    LiveDependencyTracker,
    LiveEvidenceLedger,
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
    ledger.frames = {index: {"frame_uid": _frame(index), "frame_idx": index} for index in range(5)}
    return ledger


def _add_version(ledger: LiveEvidenceLedger, row: dict) -> None:
    ledger.object_versions[row["object_version_uid"]] = row
    ledger.versions_for_object.setdefault(row["object_uid"], []).append(row)
    ledger.versions_for_lineage.setdefault(row["lineage_uid"], []).append(row)


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
    assert ticket.pool_since_frame == 1
    assert store.ordered(current_frame=1) == [ticket]


def test_exact_join_candidate_auto_resolves_after_native_update(tmp_path: Path) -> None:
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
    assert ticket.resolution_state == "AUTO_RESOLVED"
    assert ticket.resolved_by == "JOIN_CANDIDATE"
    assert store.ordered(current_frame=2) == []


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
        ticket.resolution_state = "OPEN"
        ticket.pool_since_frame = ticket.first_seen_frame
    high.error_tier, high.impact_tier = 3, 1
    low.error_tier, low.impact_tier, low.task_blocking = 1, 3, True
    assert store.ordered(current_frame=100)[0] is high
