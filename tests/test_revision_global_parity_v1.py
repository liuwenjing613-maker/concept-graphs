from copy import deepcopy

import numpy as np

from conceptgraph.revision.benchmark.experiment_v1 import aligned_relation_metrics
from conceptgraph.revision.evaluate import geometry_metrics
from scripts.validate_revision_v1_global_clean_parity import (
    _bbox_corner_hausdorff,
    _object_parity,
    _point_digest,
)


def _state():
    return {
        "objects": [
            {
                "entity_uid": "entity_a",
                "member_observation_uids": ["obs_1", "obs_2"],
                "n_points": 50,
                "point_digest": "points",
                "clip_feature_digest": "clip",
                "class_histogram": {"3": 2},
                "class_name": "chair",
                "bbox_center": [0.0, 1.0, 2.0],
                "bbox_extent": [1.0, 1.0, 1.0],
            }
        ]
    }


def test_global_payload_parity_aligns_by_membership_not_runtime_uuid():
    reference = _state()
    replayed = deepcopy(reference)
    replayed["objects"][0]["entity_uid"] = "new_runtime_uuid"

    result = _object_parity(reference, replayed)

    assert result["pass"] is True


def test_global_payload_parity_rejects_clip_or_geometry_drift():
    reference = _state()
    replayed = deepcopy(reference)
    replayed["objects"][0]["clip_feature_digest"] = "different"
    replayed["objects"][0]["bbox_center"][0] = 0.01

    result = _object_parity(reference, replayed)

    assert result["pass"] is False
    assert result["mismatch_count"] == 1
    checks = result["mismatches"][0]["checks"]
    assert checks["clip_feature_digest_exact"] is False
    assert checks["bbox_center_within_tolerance"] is False


def test_quantized_point_digest_is_order_independent_and_tolerates_sub_micro_drift():
    first = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    second = np.asarray(
        [[4.0, 5.0, 6.0], [1.0 + 1e-8, 2.0, 3.0]], dtype=np.float32
    )

    assert _point_digest(first, quantized=True) == _point_digest(
        second, quantized=True
    )


def test_bbox_corner_comparison_is_permutation_invariant_but_detects_drift() -> None:
    corners = np.asarray(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 2.0) for z in (0.0, 3.0)]
    )
    assert _bbox_corner_hausdorff(corners, corners[::-1]) == 0.0
    assert _bbox_corner_hausdorff(corners, corners + 0.01) > 0.002


def test_relation_alignment_requires_exact_member_partition() -> None:
    reference = {
        "membership": {"clean_a": ["obs_1", "obs_2"], "clean_b": ["obs_3"]},
        "edges": [
            {
                "source_entity_uid": "clean_a",
                "relation": "left of",
                "target_entity_uid": "clean_b",
                "num_detections": 2,
            }
        ],
    }
    uuid_only_difference = {
        "membership": {"runtime_a": ["obs_1", "obs_2"], "runtime_b": ["obs_3"]},
        "edges": [
            {
                "source_entity_uid": "runtime_a",
                "relation": "left of",
                "target_entity_uid": "runtime_b",
                "num_detections": 2,
            }
        ],
    }
    exact = aligned_relation_metrics(reference, uuid_only_difference)
    assert exact["edge_state_match"] is True
    assert exact["alignment_basis"] == "EXACT_MEMBER_PARTITION"

    changed_partition = {
        "membership": {"clean_a": ["obs_1"], "runtime_b": ["obs_2", "obs_3"]},
        "edges": [
            {
                "source_entity_uid": "clean_a",
                "relation": "left of",
                "target_entity_uid": "runtime_b",
                "num_detections": 2,
            }
        ],
    }
    changed = aligned_relation_metrics(reference, changed_partition)
    assert changed["edge_state_match"] is False
    assert set(changed["unaligned_candidate_entities"]) == {"clean_a", "runtime_b"}

    duplicate_edge = deepcopy(uuid_only_difference)
    duplicate_edge["edges"].append(deepcopy(duplicate_edge["edges"][0]))
    duplicate = aligned_relation_metrics(reference, duplicate_edge)
    assert duplicate["candidate_duplicate_edge_count"] == 1
    assert duplicate["edge_state_match"] is False


def test_geometry_errors_use_the_same_aabb_representation_as_iou() -> None:
    reference = {
        "objects": [
            {
                "entity_uid": "a",
                "member_observation_uids": ["obs"],
                "aabb_min": [0.0, 0.0, 0.0],
                "aabb_max": [1.0, 2.0, 3.0],
                "bbox_center": [0.5, 1.0, 1.5],
                "bbox_extent": [4.0, 5.0, 6.0],
                "n_points": 10,
            }
        ]
    }
    candidate = {
        "objects": [
            {
                **reference["objects"][0],
                "entity_uid": "runtime_a",
                "bbox_extent": [1.0, 2.0, 3.0],
            }
        ]
    }
    result = geometry_metrics(reference, candidate, observation_scope=["obs"])
    assert result["bbox_iou_to_clean"] == 1.0
    assert result["center_error_to_clean"] == 0.0
    assert result["extent_error_to_clean"] == 0.0
