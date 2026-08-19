from pathlib import Path

import numpy as np

from conceptgraph.audit.evidence_audit import (
    _Findings,
    _audit_mapping_invariants,
)


def _summary(uid, members):
    return {
        "object_uid": uid,
        "member_observation_uids": members,
        "num_detections": len(members),
    }


def test_mapping_audit_detects_reused_merge_source(tmp_path: Path):
    events = [
        {"event_uid": "e1", "event_sequence": 1, "event_type": "OBJECT_CREATE", "object_uid": "a", "obs_uid": "oa"},
        {"event_uid": "e2", "event_sequence": 2, "event_type": "OBJECT_CREATE", "object_uid": "b", "obs_uid": "ob"},
        {"event_uid": "e3", "event_sequence": 3, "event_type": "OBJECT_CREATE", "object_uid": "c", "obs_uid": "oc"},
        {
            "event_uid": "e4",
            "event_sequence": 4,
            "event_type": "OBJECT_MERGE",
            "merge_transaction_uid": "tx",
            "source_object_uid": "a",
            "target_object_uid": "b",
            "source_before": _summary("a", ["oa"]),
            "target_before": _summary("b", ["ob"]),
            "target_after": _summary("b", ["ob", "oa"]),
        },
        {
            "event_uid": "e5",
            "event_sequence": 5,
            "event_type": "OBJECT_MERGE",
            "merge_transaction_uid": "tx",
            "source_object_uid": "a",
            "target_object_uid": "c",
            "source_before": _summary("a", ["oa"]),
            "target_before": _summary("c", ["oc"]),
            "target_after": _summary("c", ["oc", "oa"]),
        },
    ]
    pair_decisions = [
        {"merge_transaction_uid": "tx", "source_object_uid": "a", "target_object_uid": "b", "decision": "ACCEPT"},
        {"merge_transaction_uid": "tx", "source_object_uid": "a", "target_object_uid": "c", "decision": "ACCEPT"},
    ]
    findings = _Findings("run")
    _audit_mapping_invariants(
        {
            "evidence_dir": tmp_path,
            "events": events,
            "pair_decisions": pair_decisions,
            "final_membership": [],
            "associations": [],
        },
        findings,
    )
    rules = [rule for item in findings.items for rule in item["rule_ids"]]
    assert "MAP-005" in rules
    assert "MAP-007" in rules


def test_mapping_audit_accepts_valid_single_merge(tmp_path: Path):
    events = [
        {"event_uid": "e1", "event_sequence": 1, "event_type": "OBJECT_CREATE", "object_uid": "a", "obs_uid": "oa"},
        {"event_uid": "e2", "event_sequence": 2, "event_type": "OBJECT_CREATE", "object_uid": "b", "obs_uid": "ob"},
        {
            "event_uid": "e3",
            "event_sequence": 3,
            "event_type": "OBJECT_MERGE",
            "merge_transaction_uid": "tx",
            "source_object_uid": "a",
            "target_object_uid": "b",
            "source_before": _summary("a", ["oa"]),
            "target_before": _summary("b", ["ob"]),
            "target_after": _summary("b", ["ob", "oa"]),
        },
    ]
    findings = _Findings("run")
    _audit_mapping_invariants(
        {
            "evidence_dir": tmp_path,
            "events": events,
            "pair_decisions": [
                {"merge_transaction_uid": "tx", "source_object_uid": "a", "target_object_uid": "b", "decision": "ACCEPT"}
            ],
            "final_membership": [],
            "associations": [],
        },
        findings,
    )
    assert not any("MAP-005" in item["rule_ids"] for item in findings.items)


def test_top_k_audit_accepts_different_order_for_equal_scores(tmp_path: Path):
    similarity_dir = tmp_path / "similarities"
    similarity_dir.mkdir()
    np.savez_compressed(
        similarity_dir / "frame_000000.npz",
        observation_uids=np.asarray(["obs-a"]),
        object_uids=np.asarray(["obj-a", "obj-b", "obj-c"]),
        aggregate_sim=np.asarray([[1.5, 1.5, 0.2]], dtype=np.float32),
    )
    findings = _Findings("run")
    _audit_mapping_invariants(
        {
            "evidence_dir": tmp_path,
            "events": [],
            "pair_decisions": [],
            "final_membership": [],
            "associations": [
                {
                    "event_uid": "e1",
                    "frame_uid": "run_f000000",
                    "obs_uid": "obs-a",
                    "decision": "MERGE_TO_OBJECT",
                    "target_object_uid": "obj-a",
                    "sim_threshold": 1.2,
                    "margin": 0.0,
                    "top_candidates": [
                        {"object_uid": "obj-b"},
                        {"object_uid": "obj-a"},
                        {"object_uid": "obj-c"},
                    ],
                }
            ],
        },
        findings,
    )
    assert not any("MAP-002" in item["rule_ids"] for item in findings.items)
