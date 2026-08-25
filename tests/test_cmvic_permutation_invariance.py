from pathlib import Path

import numpy as np

from conceptgraph.revision.counterfactual_projection import (
    CounterfactualProjectionVerifier,
    ProjectedInstance,
    ProjectionFrameEvidence,
)


def _projected(state, uid, mask):
    return ProjectedInstance(
        state_uid=state,
        canonical_partition_uid=uid,
        member_obs_uids=(uid,),
        mask=mask,
        visible_point_count=1,
        total_projected_point_count=1,
        depth_compatible_pixel_count=int(mask.sum()),
        source_state_hash=state,
    )


def test_cmvic_is_partition_order_and_ab_position_invariant():
    left = np.zeros((8, 8), dtype=bool)
    right = np.zeros((8, 8), dtype=bool)
    left[1:4, 1:3] = True
    right[1:4, 5:7] = True
    union = left | right
    frame = ProjectionFrameEvidence(
        frame_uid="scene_f000013",
        frame_index=13,
        rgb_ref={},
        depth_ref={},
        pose=np.eye(4),
        intrinsics=np.eye(4),
        processed_mask_refs=(),
        evidence_hash="frame_hash",
        depth_scale_to_meters=1.0,
        rgb_path=Path("unused"),
        depth_path=Path("unused"),
        depth_m=np.ones((8, 8)),
        observed_masks=(left, right),
        observed_mask_uids=("left", "right"),
    )
    verifier = CounterfactualProjectionVerifier(voxel_size=0.01)
    merged = (_projected("merged", "group-union", union),)
    split = (
        _projected("split", "group-right", right),
        _projected("split", "group-left", left),
    )
    original = verifier.compare(
        noop_state_uid="merged",
        candidate_state_uid="split",
        frames=(frame,),
        projected_by_frame={frame.frame_uid: {"merged": merged, "split": split}},
    )
    swapped = verifier.compare(
        noop_state_uid="split",
        candidate_state_uid="merged",
        frames=(frame,),
        projected_by_frame={
            frame.frame_uid: {"merged": merged, "split": tuple(reversed(split))}
        },
    )

    assert original.observable
    assert original.noop.projected_difference_pixel_count > 0
    assert original.noop.score == swapped.candidate.score
    assert original.candidate.score == swapped.noop.score
    assert original.advantage_over_noop == -swapped.advantage_over_noop
    assert original.evidence_selection_audit

    later_frame = ProjectionFrameEvidence(
        frame_uid="scene_f000099",
        frame_index=99,
        rgb_ref={},
        depth_ref={},
        pose=np.eye(4),
        intrinsics=np.eye(4),
        processed_mask_refs=(),
        evidence_hash="different_frame_hash",
        depth_scale_to_meters=1.0,
        rgb_path=Path("unused"),
        depth_path=Path("unused"),
        depth_m=np.ones((8, 8)),
        observed_masks=(left, right),
        observed_mask_uids=("later-left", "later-right"),
    )
    later = verifier.compare(
        noop_state_uid="merged",
        candidate_state_uid="split",
        frames=(later_frame,),
        projected_by_frame={later_frame.frame_uid: {"merged": merged, "split": split}},
    )
    assert original.noop.evidence_policy_uid == later.noop.evidence_policy_uid
    assert original.noop.score_uid != later.noop.score_uid
    assert original.evidence_selection_audit != later.evidence_selection_audit
