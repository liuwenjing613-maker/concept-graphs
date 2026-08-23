from scripts.compare_revision_live_simulator_v1 import (
    _decision_fidelity,
    _membership_observation_scope,
    _strict_symmetric_membership_metrics,
    observation_key,
)


class _LiveLedger:
    object_versions = {
        "version_a": {
            "origin_observation_uid": "live_run_f000001_r0004",
            "member_observation_uids": ["live_run_f000001_r0004"],
        }
    }
    association_rows = [
        {
            "obs_uid": "live_run_f000003_r0001",
            "event_uid": "live_event",
            "decision": "MERGE_TO_OBJECT",
            "target_object_version_before": "version_a",
        },
        {
            "obs_uid": "live_run_f000003_r0002",
            "event_uid": "live_event_2",
            "decision": "CREATE_OBJECT",
        },
    ]

    def get_object_version(self, uid):
        return self.object_versions[uid]


def test_observation_key_ignores_independent_run_prefix():
    assert observation_key("first_run_f000003_r0001") == observation_key(
        "second_run_f000003_r0001"
    )


def test_live_simulator_decision_comparison_detects_create_merge_drift():
    simulated = [
        {
            "obs_uid": "base_run_f000003_r0001",
            "event_uid": "base_event",
            "applied_match": 4,
            "natural_match": 4,
            "applied_target_origin_obs_uid": "base_run_f000001_r0004",
        },
        {
            "obs_uid": "base_run_f000003_r0002",
            "event_uid": "base_event_2",
            "applied_match": 2,
            "natural_match": 2,
        },
    ]

    result = _decision_fidelity(_LiveLedger(), simulated)

    assert result["pass"] is False
    assert result["decision_kind_mismatch_count"] == 1
    assert result["first_mismatches"][0]["observation_key"] == "_f000003_r0002"


def test_live_membership_scope_is_union_not_live_only():
    live = {"membership": {"live": ["obs_a"]}}
    simulator = {"membership": {"sim": ["obs_a", "obs_extra"]}}

    assert _membership_observation_scope(live, simulator) == {
        "obs_a",
        "obs_extra",
    }

    result = _strict_symmetric_membership_metrics(
        live["membership"], simulator["membership"]
    )
    assert result["comparison_scope"] == "UNION_OF_LIVE_AND_SIMULATOR_OBSERVATIONS"
    assert result["observation_count"] == 2
    assert result["missing_in_live"] == ["obs_extra"]
    assert result["member_f1"] < 1.0
    assert result["partition_exact"] is False


def test_live_membership_exact_is_uuid_independent_and_rejects_duplicates():
    live = {"left_a": ["obs_a", "obs_b"], "left_b": ["obs_c"]}
    simulator = {"right_x": ["obs_c"], "right_y": ["obs_b", "obs_a"]}

    exact = _strict_symmetric_membership_metrics(live, simulator)
    assert exact["member_f1"] == 1.0
    assert exact["partition_exact"] is True

    simulator["right_x"].append("obs_a")
    duplicate = _strict_symmetric_membership_metrics(live, simulator)
    assert duplicate["simulator_duplicate_observations"] == ["obs_a"]
    assert duplicate["partition_exact"] is False

    duplicate_within_one_entity = {
        "right_x": ["obs_c"],
        "right_y": ["obs_a", "obs_a", "obs_b"],
    }
    duplicate = _strict_symmetric_membership_metrics(
        live, duplicate_within_one_entity
    )
    assert duplicate["simulator_duplicate_observations"] == ["obs_a"]
    assert duplicate["partition_exact"] is False
