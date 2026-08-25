from pathlib import Path

import numpy as np

from conceptgraph.revision.counterfactual_projection import (
    CounterfactualProjectionVerifier,
    ProjectedInstance,
    ProjectionFrameEvidence,
)


def _projected(uid, mask):
    return ProjectedInstance(
        state_uid="state",
        canonical_partition_uid=uid,
        member_obs_uids=(uid,),
        mask=mask,
        visible_point_count=1,
        total_projected_point_count=1,
        depth_compatible_pixel_count=int(mask.sum()),
        source_state_hash="state",
    )


def test_hungarian_iou_penalizes_unmatched_observed_masks():
    first = np.zeros((8, 8), dtype=bool)
    second = np.zeros((8, 8), dtype=bool)
    third = np.zeros((8, 8), dtype=bool)
    first[1:3, 1:3] = True
    second[4:6, 4:6] = True
    third[1:3, 5:7] = True
    frame = ProjectionFrameEvidence(
        frame_uid="scene_f000012",
        frame_index=12,
        rgb_ref={},
        depth_ref={},
        pose=np.eye(4),
        intrinsics=np.eye(4),
        processed_mask_refs=(),
        evidence_hash="frame",
        depth_scale_to_meters=1.0,
        rgb_path=Path("unused"),
        depth_path=Path("unused"),
        depth_m=np.ones((8, 8)),
        observed_masks=(second, first, third),
        observed_mask_uids=("m2", "m1", "m3"),
    )
    result = CounterfactualProjectionVerifier(voxel_size=0.01).score_frame(
        frame=frame,
        state_uid="state",
        projected=(_projected("p1", first), _projected("p2", second)),
        observed_masks=frame.observed_masks,
        observed_mask_uids=frame.observed_mask_uids,
    )

    assert result.score == 2.0 / 3.0
    assert result.diagnostics["unmatched_observed_mask_count"] == 1
