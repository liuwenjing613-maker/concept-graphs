from pathlib import Path

import numpy as np

from conceptgraph.revision.counterfactual_projection import (
    CounterfactualProjectionVerifier,
    InstanceGeometry,
    ProjectionFrameEvidence,
)


def _frame():
    return ProjectionFrameEvidence(
        frame_uid="scene_f000011",
        frame_index=11,
        rgb_ref={},
        depth_ref={},
        pose=np.eye(4),
        intrinsics=np.array(
            [
                [10.0, 0.0, 10.0, 0.0],
                [0.0, 10.0, 10.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        processed_mask_refs=(),
        evidence_hash="frame_hash",
        depth_scale_to_meters=1.0,
        rgb_path=Path("unused.png"),
        depth_path=Path("unused.png"),
        depth_m=np.full((20, 20), 2.0),
        observed_masks=(),
        observed_mask_uids=(),
    )


def test_frozen_depth_rejects_occluded_points_without_semantic_threshold():
    instances = (
        InstanceGeometry.build(
            member_obs_uids=("visible",),
            points=np.array([[0.0, 0.0, 2.0]]),
            source_state_hash="state",
        ),
        InstanceGeometry.build(
            member_obs_uids=("behind",),
            points=np.array([[0.0, 0.0, 3.0]]),
            source_state_hash="state",
        ),
    )
    projected = CounterfactualProjectionVerifier(voxel_size=0.01).project_state(
        state_uid="state", instances=instances, frame=_frame()
    )
    by_member = {item.member_obs_uids[0]: item for item in projected}

    assert by_member["visible"].mask.any()
    assert not by_member["behind"].mask.any()
    assert by_member["behind"].visible_point_count == 0
