from __future__ import annotations

import json
from pathlib import Path

from conceptgraph.revision.online_mvp import (
    JsonlTail,
    LiveDependencyTracker,
    LiveEvidenceLedger,
    OnlineScanner,
    SubIssue,
    TaskContext,
    TicketStore,
    final_scene_metric_gate,
    scene_health_metrics,
)


def _append(path: Path, value: dict, *, newline: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value))
        if newline:
            handle.write("\n")


def _frame(index: int) -> str:
    return f"run_f{index:06d}"


def _obs(index: int, detection: int = 0) -> str:
    return f"run_f{index:06d}_r{detection:04d}"


def test_jsonl_tail_waits_for_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _append(path, {"a": 1}, newline=False)
    tail = JsonlTail(path)
    assert tail.poll() == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert tail.poll() == [{"a": 1}]
    assert tail.poll() == []


def test_ledger_commits_with_one_frame_delay(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    for name in (
        "frames.jsonl",
        "observations.jsonl",
        "associations.jsonl",
        "mapping_events.jsonl",
        "object_versions.jsonl",
        "object_pair_decisions.jsonl",
    ):
        (evidence / name).parent.mkdir(parents=True, exist_ok=True)
        (evidence / name).touch()
    ledger = LiveEvidenceLedger(tmp_path)
    _append(evidence / "frames.jsonl", {"frame_uid": _frame(0), "frame_idx": 0})
    assert ledger.poll() == []
    _append(evidence / "frames.jsonl", {"frame_uid": _frame(1), "frame_idx": 1})
    assert ledger.poll() == [0]
    assert ledger.poll(mapping_done=True) == [1]


def test_scanner_builds_object_group_ticket_and_task_priority(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    for name in (
        "frames.jsonl",
        "observations.jsonl",
        "associations.jsonl",
        "mapping_events.jsonl",
        "object_versions.jsonl",
        "object_pair_decisions.jsonl",
    ):
        (evidence / name).parent.mkdir(parents=True, exist_ok=True)
        (evidence / name).touch()
    version = {
        "object_version_uid": "object-a@v1",
        "object_uid": "object-a",
        "lineage_uid": "lineage-a",
        "version": 1,
        "frame_uid": _frame(0),
        "trigger_event_uid": "event-map-0",
        "status": "active",
        "member_observation_uids": [_obs(0)],
        "bbox_volume": 1.0,
        "dominant_class_ratio": 1.0,
        "class_name": "chair",
    }
    observation = {
        "obs_uid": _obs(1),
        "frame_uid": _frame(1),
        "status": "kept",
        "confidence": 0.9,
        "class_name": "chair",
        "bbox_2d": [0, 0, 10, 10],
    }
    association = {
        "event_uid": "event-assoc-1",
        "event_sequence": 3,
        "frame_uid": _frame(1),
        "obs_uid": _obs(1),
        "decision": "CREATE_OBJECT",
        "top1_score": 1.15,
        "top2_score": 1.0,
        "margin": 0.15,
        "sim_threshold": 1.2,
        "target_object_uid": "object-new",
        "target_object_version_before": None,
        "target_object_version_after": "object-new@v1",
        "object_uids_before": ["object-a"],
        "candidate_object_version_uids": ["object-a@v1"],
        "top_candidates": [{"object_uid": "object-a", "aggregate_score": 1.15}],
    }
    mapping = {
        "event_uid": "event-map-1",
        "event_sequence": 4,
        "frame_uid": _frame(1),
        "event_type": "OBJECT_CREATE",
        "object_uid": "object-new",
        "obs_uid": _obs(1),
        "input_object_version_uids": [],
        "output_object_version_uids": ["object-new@v1"],
    }
    new_version = {
        **version,
        "object_version_uid": "object-new@v1",
        "object_uid": "object-new",
        "lineage_uid": "lineage-new",
        "frame_uid": _frame(1),
        "trigger_event_uid": "event-map-1",
        "member_observation_uids": [_obs(1)],
    }
    for index in range(3):
        _append(evidence / "frames.jsonl", {"frame_uid": _frame(index), "frame_idx": index})
    _append(evidence / "object_versions.jsonl", version)
    _append(evidence / "observations.jsonl", observation)
    _append(evidence / "associations.jsonl", association)
    _append(evidence / "mapping_events.jsonl", mapping)
    _append(evidence / "object_versions.jsonl", new_version)
    ledger = LiveEvidenceLedger(tmp_path)
    assert ledger.poll() == [0, 1]
    issues = OnlineScanner().scan_frame(1, ledger)
    assert any(issue.family == "NEAR_THRESHOLD_CREATE" for issue in issues)
    store = TicketStore()
    for issue in issues:
        store.upsert(issue)
    store.refresh(
        ledger=ledger,
        tracker=LiveDependencyTracker(),
        task_context=TaskContext.from_mapping(
            {
                "task_id": "pick-chair",
                "active": True,
                "required_lineage_uids": ["lineage-a"],
            }
        ),
        stop_sequence=ledger.max_sequence,
    )
    ordered = store.ordered(current_frame=2)
    assert ordered
    assert ordered[0].task_blocking
    assert ordered[0].affected_event_count >= 1


def test_priority_is_lexicographic_not_weighted() -> None:
    store = TicketStore()
    blocking = SubIssue.build(
        family="A",
        anchor_event_uid="e1",
        anchor_obs_uid="o1",
        detected_frame=10,
        detected_sequence=1,
        object_uids=("a",),
        lineage_uids=("la",),
        raw_signals={},
    )
    broad = SubIssue.build(
        family="B",
        anchor_event_uid="e2",
        anchor_obs_uid="o2",
        detected_frame=0,
        detected_sequence=2,
        object_uids=("b",),
        lineage_uids=("lb",),
        raw_signals={},
    )
    first = store.upsert(blocking)
    second = store.upsert(broad)
    first.task_blocking = True
    first.affected_lineage_uids = ("la",)
    first.affected_event_count = 1
    second.affected_lineage_uids = tuple(f"l{index}" for index in range(100))
    second.affected_event_count = 1000
    assert store.ordered(current_frame=100)[0].ticket_uid == first.ticket_uid


def test_scene_health_metrics_are_label_free() -> None:
    state = {
        "state_hash": "abc",
        "membership": {"a": [_obs(0), _obs(1)]},
        "objects": [
            {
                "entity_uid": "a",
                "member_observation_uids": [_obs(0), _obs(1)],
                "class_histogram": {"chair": 2},
                "n_points": 100,
                "bbox_center": [0, 0, 0],
                "bbox_extent": [1, 1, 1],
            }
        ],
    }
    metrics = scene_health_metrics(state)
    assert metrics["object_count"] == 1
    assert metrics["weighted_semantic_purity"] == 1.0
    assert metrics["duplicate_ownership_count"] == 0
    assert not any("gold" in key or "label" in key for key in metrics)


def test_final_scene_metric_gate_rejects_loss_and_collapse() -> None:
    baseline = {
        "state_hash": "a",
        "observation_count": 100,
        "object_count": 20,
        "weighted_semantic_purity": 0.8,
        "singleton_object_rate": 0.2,
        "low_purity_object_rate": 0.1,
        "duplicate_ownership_count": 0,
        "invalid_geometry_object_count": 0,
    }
    candidate = {
        **baseline,
        "state_hash": "b",
        "observation_count": 99,
        "object_count": 15,
    }
    gate = final_scene_metric_gate(baseline, candidate)
    assert gate["partition_changed"]
    assert not gate["observation_count_conserved"]
    assert not gate["object_count_not_collapsed"]
