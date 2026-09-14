import hashlib
import json

import pytest

from conceptgraph.revision.evaluate import edge_metrics
from conceptgraph.revision.relations import load_edge_stream, remap_frame_records


def _write_stream(root, rows):
    root.mkdir()
    frames = root / "frames.jsonl"
    frames.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(frames.read_bytes()).hexdigest()
    observations = sum(len(row.get("edges") or ()) for row in rows)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "frame_count": len(rows),
                "input_edge_observations": observations,
                "frames_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    return frames


def test_edge_stream_is_hash_bound_and_label_bound(tmp_path):
    root = tmp_path / "stream"
    frames = _write_stream(
        root,
        [
            {
                "source_frame_id": "frame000000",
                "input_labels": ["0: chair", "1: table"],
                "edges": [["0", "on top of", "1"]],
            }
        ],
    )
    manifest, rows = load_edge_stream(root)
    assert manifest["status"] == "PASS"
    assert rows["frame000000"]["edges"] == [["0", "on top of", "1"]]

    frames.write_text(frames.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_edge_stream(root)


def test_edge_metrics_distinguish_corruption_and_exact_recovery():
    clean = {
        "membership": {"A": ["a"], "B": ["b"]},
        "edges": [
            {
                "source_entity_uid": "A",
                "relation": "on top of",
                "target_entity_uid": "B",
            }
        ],
    }
    corrupted = {
        "membership": {"A": ["a"], "C": ["b"]},
        "edges": [
            {
                "source_entity_uid": "A",
                "relation": "on top of",
                "target_entity_uid": "C",
            }
        ],
    }
    degraded = edge_metrics(clean, corrupted)
    support_changed = edge_metrics(
        {
            **clean,
            "edges": [{**clean["edges"][0], "num_detections": 3}],
        },
        {
            **clean,
            "edges": [{**clean["edges"][0], "num_detections": 2}],
        },
    )
    recovered = edge_metrics(clean, clean)
    assert degraded["edge_set_f1_to_clean"] == 0.0
    assert degraded["false_positive_edge_count"] == 1
    assert degraded["false_negative_edge_count"] == 1
    assert support_changed["edge_set_f1_to_clean"] == 1.0
    assert support_changed["edge_relation_match"] is True
    assert support_changed["edge_state_match"] is False
    assert support_changed["support_mismatch_edge_count"] == 1
    assert support_changed["support_absolute_error"] == 1
    assert recovered["edge_set_f1_to_clean"] == 1.0
    assert recovered["edge_relation_match"] is True
    assert recovered["edge_state_match"] is True


def test_frame_records_can_be_remapped_to_branch_membership():
    source = [
        {
            "frame_idx": 0,
            "detection_class_labels": ["chair 0", "table 1"],
            "edges": [["0", "on top of", "1"]],
            "observation_uids": ["obs-a", "obs-b"],
            "match_indices": [0, 1],
        }
    ]
    objects, records = remap_frame_records(
        source,
        {"entity-z": ["obs-a", "obs-b"]},
    )
    assert [item["entity_uid"] for item in objects] == ["entity-z"]
    assert records[0]["match_indices"] == [0, 0]
    assert source[0]["match_indices"] == [0, 1]
