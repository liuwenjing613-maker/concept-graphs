import uuid

import pytest


pytest.importorskip("open3d")

from conceptgraph.revision.relations import AliDevBaselineRelationBackend


def test_nonempty_relation_rebuild_uses_unchanged_ali_dev_semantics():
    objects = [
        {"entity_uid": "A", "id": uuid.uuid4(), "curr_obj_num": 0},
        {"entity_uid": "B", "id": uuid.uuid4(), "curr_obj_num": 1},
    ]
    frames = [
        {
            "frame_idx": 0,
            "detection_class_labels": ["chair 0", "table 1"],
            "edges": [("0", "on top of", "1")],
            "match_indices": [0, 1],
        },
        {
            "frame_idx": 1,
            "detection_class_labels": ["chair 0", "table 1"],
            "edges": [("0", "on top of", "1")],
            "match_indices": [0, 1],
        },
    ]
    backend = AliDevBaselineRelationBackend()
    result = backend.rebuild(objects=objects, frame_records=frames)
    assert result["used_process_edges"] is True
    assert result["validation"]["pass"] is True
    assert result["input_relation_types"] == ["on top of"]
    assert result["output_edges"] == [
        {
            "source_entity_uid": "A",
            "relation": "on top of",
            "target_entity_uid": "B",
            "num_detections": 2,
        }
    ]
