from conceptgraph.revision.corruption import ControlledCorruptionController
from conceptgraph.revision.models import CorruptionPlan


def test_cross_run_observation_and_target_origin_selectors_are_stable():
    plan = CorruptionPlan(
        case_uid="cross_run",
        frame_idx=12,
        obs_uid="old_run_f000012_r0003",
        corruption_type="FORCE_ASSOCIATE",
        target_object_uid="old-random-object-uuid",
        target_origin_obs_uid="old_run_f000004_r0001",
    )
    controller = ControlledCorruptionController(plan)
    matches = controller.apply(
        frame_idx=12,
        detection_list=[{"obs_uids": ["new_run_f000012_r0003"]}],
        objects=[
            {"id": "new-random-object-uuid", "obs_uids": ["new_run_f000004_r0001"]},
            {"id": "other", "obs_uids": ["new_run_f000002_r0000"]},
        ],
        original_match_indices=[1],
    )
    assert matches == [0]
    assert controller.records[0]["obs_uid"] == "new_run_f000012_r0003"
    assert controller.records[0]["corrupted_decision"]["target_object_uid"] == (
        "new-random-object-uuid"
    )
    controller.finalize()


def test_cross_run_source_guard_uses_origin_without_disabling_drift_checks():
    plan = CorruptionPlan(
        case_uid="cross_run_source",
        frame_idx=9,
        obs_uid="old_run_f000009_r0002",
        corruption_type="FORCE_CREATE",
        source_object_uid="old-random-source-uuid",
        source_origin_obs_uid="old_run_f000001_r0007",
    )
    controller = ControlledCorruptionController(plan)
    matches = controller.apply(
        frame_idx=9,
        detection_list=[{"obs_uids": ["new_run_f000009_r0002"]}],
        objects=[
            {"id": "new-random-source-uuid", "obs_uids": ["new_run_f000001_r0007"]}
        ],
        original_match_indices=[0],
    )
    assert matches == [None]
    controller.finalize()
