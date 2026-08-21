from __future__ import annotations

import copy
import json
from uuid import uuid4

import numpy as np
import pytest

from conceptgraph.vlm_repair.overlay import (
    OverlayError,
    _object_summary,
    apply_repairs_to_bundle,
)


def test_object_summary_serializes_uuid_and_numpy_ids():
    uuid_id = uuid4()
    objects = [
        {
            "id": uuid_id,
            "class_name": "chair",
            "pcd_np": np.asarray([[0, 0, 0], [1, 2, 3]], dtype=np.float64),
        },
        {
            "id": np.int64(7),
            "class_name": "table",
            "pcd_np": np.asarray([[1, 1, 1], [2, 3, 4]], dtype=np.float64),
        },
    ]

    summary = _object_summary(objects)

    assert summary["object_1"]["id"] == str(uuid_id)
    assert summary["object_2"]["id"] == 7
    json.dumps(summary)


def _map_object(object_id, label: str, offset: float):
    points = np.asarray([[offset, 0, 0], [offset + 1, 1, 1]], dtype=np.float64)
    return {
        "id": object_id,
        "class_name": label,
        "num_detections": 1,
        "pcd_np": points,
        "pcd_color_np": np.ones_like(points),
    }


def _membership(uid: str, index: int, label: str):
    return {
        "object_uid": uid,
        "current_object_index": index,
        "class_name": label,
        "member_observation_uids": [],
        "parent_or_merged_from_object_uids": [],
        "num_detections": 1,
        "n_points": 2,
    }


def test_overlay_keeps_edge_object_snapshot_aligned_after_merge():
    objects = [_map_object(uuid4(), "chair", 0), _map_object(uuid4(), "chair", 2)]
    bundle = {
        "objects": objects,
        "edges": {"edges": [], "objects": copy.deepcopy(objects)},
    }
    membership = [_membership("uid-1", 0, "chair"), _membership("uid-2", 1, "chair")]

    derived, derived_membership, reports = apply_repairs_to_bundle(
        bundle,
        membership,
        [
            {
                "case_uid": "case-1",
                "action": "MERGE_WITH",
                "target_uid": "uid-1",
                "other_uid": "uid-2",
                "new_label": None,
            }
        ],
    )

    assert reports[0]["apply_status"] == "APPLIED"
    assert len(derived["objects"]) == len(derived_membership) == 1
    assert len(derived["edges"]["objects"]) == 1
    assert derived["edges"]["objects"][0]["id"] == derived["objects"][0]["id"]


def test_overlay_rejects_structural_repair_with_nonempty_edges():
    objects = [_map_object(uuid4(), "chair", 0), _map_object(uuid4(), "chair", 2)]
    bundle = {
        "objects": objects,
        "edges": {"edges": [[0, 1]], "objects": copy.deepcopy(objects)},
    }
    membership = [_membership("uid-1", 0, "chair"), _membership("uid-2", 1, "chair")]

    with pytest.raises(OverlayError, match="graph edges are non-empty"):
        apply_repairs_to_bundle(
            bundle,
            membership,
            [
                {
                    "case_uid": "case-1",
                    "action": "MERGE_WITH",
                    "target_uid": "uid-1",
                    "other_uid": "uid-2",
                    "new_label": None,
                }
            ],
        )
