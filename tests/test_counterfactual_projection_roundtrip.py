from pathlib import Path

import numpy as np

from conceptgraph.revision.counterfactual_projection import (
    CounterfactualProjectionVerifier,
    InstanceGeometry,
    ProjectionFrameEvidence,
)


def _frame(depth):
    return ProjectionFrameEvidence(
        frame_uid="scene_f000010",
        frame_index=10,
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
        depth_m=np.asarray(depth, dtype=float),
        observed_masks=(),
        observed_mask_uids=(),
    )


def test_world_camera_projection_roundtrip_uses_inverse_pose():
    depth = np.full((20, 20), 2.0)
    frame = _frame(depth)
    geometry = InstanceGeometry.build(
        member_obs_uids=("obs-a",),
        points=np.array([[0.0, 0.0, 2.0]]),
        source_state_hash="state",
    )
    verifier = CounterfactualProjectionVerifier(voxel_size=0.01)
    projected = verifier.project_state(
        state_uid="state-a", instances=(geometry,), frame=frame
    )

    assert projected[0].visible_point_count == 1
    assert projected[0].total_projected_point_count == 1
    assert projected[0].mask[10, 10]
    assert int(projected[0].mask.sum()) == 5
