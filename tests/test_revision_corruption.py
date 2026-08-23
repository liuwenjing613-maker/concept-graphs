import uuid

import pytest

from conceptgraph.revision.corruption import ControlledCorruptionController
from conceptgraph.revision.models import CorruptionPlan


def _det(obs_uid):
    return {"obs_uids": [obs_uid], "id": uuid.uuid4()}


def _obj(uid):
    return {"id": uuid.UUID(uid)}


def test_force_create_changes_exactly_one_named_decision():
    source = "00000000-0000-0000-0000-000000000001"
    plan = CorruptionPlan(
        case_uid="case_split",
        frame_idx=4,
        obs_uid="obs_1",
        corruption_type="FORCE_CREATE",
        source_object_uid=source,
    )
    controller = ControlledCorruptionController(plan)
    assert controller.apply(
        frame_idx=3,
        detection_list=[_det("obs_1")],
        objects=[_obj(source)],
        original_match_indices=[0],
    ) == [0]
    changed = controller.apply(
        frame_idx=4,
        detection_list=[_det("obs_0"), _det("obs_1")],
        objects=[_obj(source)],
        original_match_indices=[0, 0],
    )
    assert changed == [0, None]
    controller.finalize()
    assert controller.records[0]["original_decision"]["target_object_uid"] == source


def test_force_associate_resolves_target_uid_not_unstable_index():
    source = "00000000-0000-0000-0000-000000000001"
    target = "00000000-0000-0000-0000-000000000002"
    controller = ControlledCorruptionController(
        CorruptionPlan(
            case_uid="case_wrong",
            frame_idx=8,
            obs_uid="obs_1",
            corruption_type="FORCE_ASSOCIATE",
            source_object_uid=source,
            target_object_uid=target,
        )
    )
    changed = controller.apply(
        frame_idx=8,
        detection_list=[_det("obs_1")],
        objects=[_obj(target), _obj(source)],
        original_match_indices=[1],
    )
    assert changed == [0]
    controller.finalize()


def test_missing_case_is_a_hard_failure():
    controller = ControlledCorruptionController(
        CorruptionPlan("case", 2, "missing", "FORCE_CREATE")
    )
    with pytest.raises(RuntimeError, match="applied 0 times"):
        controller.finalize()
