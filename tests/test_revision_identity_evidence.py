from __future__ import annotations

import json
import math

import pytest

from conceptgraph.revision.auto_constraints import (
    GeneratorStage,
    aggregate_candidate_votes,
    compile_blind_candidate,
    forbidden_inference_paths,
)
from conceptgraph.revision.identity_evidence import (
    IdentityEvidenceBundleBuilder,
    bbox_pair_metrics,
    sanitize_machine_review,
)


class _FakeProvenance:
    def __init__(self):
        self.events = {
            "event_10": {
                "event_uid": "event_10",
                "obs_uid": "scene_run_f000010_r0002",
                "decision": "CREATE_OBJECT",
                "sim_threshold": 1.2,
                "top1_score": 1.1,
                "top_candidates": [
                    {
                        "object_uid": "object-a",
                        "spatial_score": 0.4,
                        "visual_score": 0.7,
                        "aggregate_score": 1.1,
                    },
                    {
                        "object_uid": "object-b",
                        "spatial_score": 0.0,
                        "visual_score": 0.6,
                        "aggregate_score": 0.6,
                    },
                ],
                "object_uids_before": ["object-a", "object-b"],
                "candidate_object_version_uids": ["object-a@v1", "object-b@v1"],
                "mapping_event_uid": "event_11",
                "event_sequence": 10,
            },
            "event_11": {
                "event_uid": "event_11",
                "event_type": "OBJECT_CREATE",
                "object_uid": "created-object",
                "output_object_version_uids": ["created-object@v1"],
                "event_sequence": 11,
            },
        }
        self.observations = {
            "scene_run_f000010_r0002": {
                "obs_uid": "scene_run_f000010_r0002",
                "class_name": "chair",
                "confidence": 0.8,
                "bbox_3d_center": [1.0, 0.0, 0.0],
                "bbox_3d_extent": [1.0, 1.0, 1.0],
                "n_points": 100,
                "raw_mask_area": 1000,
                "processed_mask_area": 900,
                "removed_pixel_count": 100,
                "valid_depth_ratio": 1.0,
                "pre_dbscan": {
                    "cluster_count": 1,
                    "largest_cluster_ratio": 1.0,
                },
            },
            "scene_run_f000001_r0001": {
                "obs_uid": "scene_run_f000001_r0001",
                "class_name": "chair",
            },
            "scene_run_f000009_r0001": {
                "obs_uid": "scene_run_f000009_r0001",
                "class_name": "armchair",
            },
            "scene_run_f000002_r0001": {
                "obs_uid": "scene_run_f000002_r0001",
                "class_name": "table",
            },
            "scene_run_f000008_r0001": {
                "obs_uid": "scene_run_f000008_r0001",
                "class_name": "table",
            },
        }
        self.versions = {
            "object-a@v1": {
                "object_uid": "object-a",
                "lineage_uid": "origin-a",
                "origin_observation_uid": "scene_run_f000001_r0001",
                "bbox_center": [1.25, 0.0, 0.0],
                "bbox_extent": [1.0, 1.0, 1.0],
                "n_points": 200,
                "class_name": "chair",
                "unique_frame_count": 2,
            },
            "object-b@v1": {
                "object_uid": "object-b",
                "lineage_uid": "origin-b",
                "origin_observation_uid": "scene_run_f000002_r0001",
                "bbox_center": [4.0, 0.0, 0.0],
                "bbox_extent": [1.0, 1.0, 1.0],
                "n_points": 180,
                "class_name": "table",
                "unique_frame_count": 2,
            },
            "created-object@v1": {
                "object_uid": "created-object",
                "lineage_uid": "origin-created",
                "origin_observation_uid": "scene_run_f000010_r0002",
            },
        }
        self.members = {
            "object-a@v1": (
                "scene_run_f000001_r0001",
                "scene_run_f000009_r0001",
            ),
            "object-b@v1": (
                "scene_run_f000002_r0001",
                "scene_run_f000008_r0001",
            ),
            "created-object@v1": ("scene_run_f000010_r0002",),
        }

    def get_event(self, uid):
        return self.events[uid]

    def get_observation(self, uid):
        return self.observations[uid]

    def get_object_version(self, uid):
        return self.versions[uid]

    def get_member_observations(self, uid):
        return self.members[uid]

    def sequence(self, row):
        return int(row["event_sequence"])


def _review():
    return {
        "schema_version": "2.1.0",
        "checker_id": "DET-001",
        "stage": "detection",
        "subtype": "DUPLICATE_PROPOSAL",
        "human_label": {"expected_action": "forbidden but ignored"},
        "incident": {
            "checker_ids": ["DET-001", "ASSOC-003"],
            "stages": ["detection", "association"],
            "subtypes": ["DUPLICATE_PROPOSAL"],
        },
        "evidence_contract": {
            "fidelity_status": "TRACEABLE",
            "artifact_hashes_match": True,
            "exact_final_map_linkage": True,
            "critical_gaps": [],
        },
        "trigger_observations": [
            {
                "observation_alias": "Q1",
                "obs_uid": "scene_run_f000010_r0002",
                "class_name": "chair",
                "final_owner_uids": ["oracle-owner"],
                "bbox_3d_center": [1.0, 0.0, 0.0],
                "bbox_3d_extent": [1.0, 1.0, 1.0],
            }
        ],
        "association_decisions": [
            {
                "obs_uid": "scene_run_f000010_r0002",
                "decision": "CREATE_OBJECT",
                "target_object_uid": "must-not-leak",
                "sim_threshold": 1.2,
                "top1_score": 1.1,
                "candidates": [
                    {
                        "rank": 1,
                        "object_alias": "O1",
                        "object_uid": "must-not-leak",
                        "aggregate_score": 1.1,
                    }
                ],
            }
        ],
        "final_outcome": {
            "machine_resolution_status": "TWO_CURRENT_OBJECTS",
            "final_membership": {"oracle": ["must-not-leak"]},
        },
        "final_objects": [
            {
                "object_alias": "O1",
                "object_uid": "must-not-leak",
                "class_name": "chair",
                "member_count": 2,
                "unique_frame_count": 2,
                "bbox_center": [1.0, 0.0, 0.0],
                "bbox_extent": [1.0, 1.0, 1.0],
                "parent_or_merged_from_object_uids": ["opaque-parent"],
            }
        ],
    }


def _votes(action, **fields):
    return [
        {
            "constraint": {
                "action": action,
                "confidence": confidence,
                "evidence_image_ids": ["I01", "I02"],
                **fields,
            }
        }
        for confidence in (0.91, 0.93, 0.95)
    ]


def test_bbox_pair_metrics_has_exact_overlap_and_surface_gap_semantics():
    same = bbox_pair_metrics([0, 0, 0], [2, 2, 2], [0, 0, 0], [2, 2, 2])
    assert same["aabb_iou"] == pytest.approx(1.0)
    assert same["surface_gap"] == pytest.approx(0.0)
    assert same["center_distance"] == pytest.approx(0.0)

    apart = bbox_pair_metrics([0, 0, 0], [2, 2, 2], [5, 0, 0], [2, 2, 2])
    assert apart["aabb_iou"] == pytest.approx(0.0)
    assert apart["surface_gap"] == pytest.approx(3.0)
    assert apart["center_distance"] == pytest.approx(5.0)

    missing = bbox_pair_metrics(None, None, [0, 0, 0], [1, 1, 1])
    assert all(value is None for value in missing.values())


def test_machine_review_sanitizer_excludes_raw_uids_and_oracle_fields():
    sanitized = sanitize_machine_review(_review())
    encoded = json.dumps(sanitized, sort_keys=True)
    assert "must-not-leak" not in encoded
    assert "oracle-owner" not in encoded
    assert "final_membership" not in encoded
    assert forbidden_inference_paths(sanitized) == ()
    assert sanitized["current_map_objects"][0]["merged_from_count"] == 1


def test_bundle_is_deterministic_finite_alias_evidence_and_binding_is_private():
    provenance = _FakeProvenance()
    builder = IdentityEvidenceBundleBuilder(provenance)
    first = builder.build(
        case_uid="dev_1",
        association_event_uid="event_10",
        machine_review=_review(),
    )
    second = builder.build(
        case_uid="dev_1",
        association_event_uid="event_10",
        machine_review=_review(),
    )
    bundle = first.inference_bundle
    assert bundle == second.inference_bundle
    assert bundle["bundle_uid"] == second.inference_bundle["bundle_uid"]
    assert bundle["threshold_semantics"]["top1_exceeds_threshold"] is False
    assert [row["alias"] for row in bundle["candidate_aliases"]] == [
        "CANDIDATE_1_CONTEXT",
        "CANDIDATE_2_CONTEXT",
    ]
    assert (
        bundle["candidate_aliases"][0]["current_object_summary"][
            "frames_since_last_observation"
        ]
        == 1
    )
    assert bundle["candidate_aliases"][0]["anchor_candidate_geometry"]["aabb_iou"] > 0
    assert "object-a" not in json.dumps(bundle, sort_keys=True)
    assert first.binding.aliases["CANDIDATE_1_CONTEXT"].entity_uid == "object-a"
    assert first.binding.created_identity_uid == "origin-created"


@pytest.mark.parametrize(
    ("action", "fields", "constraint_type"),
    [
        (
            "SAME_INSTANCE",
            {"entities": ["ANCHOR", "CANDIDATE_1_CONTEXT"]},
            "ASSIGN_OBSERVATION",
        ),
        (
            "SEPARATE_MEMBER_GROUPS",
            {"groups": [["ANCHOR"], ["CANDIDATE_1_CONTEXT"]]},
            "CREATE_INSTANCE",
        ),
    ],
)
def test_finite_alias_vote_compiles_to_exact_private_binding(
    action, fields, constraint_type
):
    built = IdentityEvidenceBundleBuilder(_FakeProvenance()).build(
        case_uid="dev_1",
        association_event_uid="event_10",
        machine_review=_review(),
    )
    aggregate = aggregate_candidate_votes(
        _votes(action, **fields),
        allowed_evidence_ids={"I01", "I02"},
    )
    compiled = compile_blind_candidate(aggregate, built.binding)
    assert compiled["stage"] == GeneratorStage.BOUND_PENDING_SHADOW.value
    assert compiled["target_alias"] == "CANDIDATE_1_CONTEXT"
    assert compiled["candidate_constraint"]["type"] == constraint_type
