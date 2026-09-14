from types import MethodType

from conceptgraph.revision.sparse_replay import SparseCounterfactualReplayEngine


def test_state_only_and_raw_object_apis_share_exact_internal_execution():
    engine = SparseCounterfactualReplayEngine.__new__(SparseCounterfactualReplayEngine)
    state = {
        "state_hash": "hash",
        "membership": {"entity": ["obs"]},
        "objects": [{"entity_uid": "entity", "member_observation_uids": ["obs"]}],
    }
    objects = [{"id": "entity", "obs_uids": ["obs"], "pcd": object()}]

    call_kwargs = {
        "mode": "NATURAL_REPLAY",
        "snapshot_objects": (),
        "snapshot_runtime_ms": 0.0,
        "anchor_frame": 0,
        "snapshot_watermark_event_sequence": 0,
        "closure": object(),
        "current_state": {},
        "constraints": (),
        "corruption_plan": None,
        "historical_anchor_plan": None,
        "snapshot_timing": None,
        "component_policy": None,
    }

    def impl(self, **kwargs):
        assert kwargs == call_kwargs
        return state, objects

    engine._replay_suffix_from_snapshot_impl = MethodType(impl, engine)

    old = engine.replay_suffix_from_snapshot(**call_kwargs)
    new_state, new_objects = engine.replay_local_from_snapshot_with_objects(
        **call_kwargs
    )
    assert old is state
    assert new_state is state
    assert old["state_hash"] == new_state["state_hash"]
    assert old["membership"] == new_state["membership"]
    assert old["objects"] == new_state["objects"]
    assert new_objects[0]["obs_uids"] == ["obs"]
