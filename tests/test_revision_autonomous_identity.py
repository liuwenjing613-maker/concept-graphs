from __future__ import annotations

import pytest

from conceptgraph.revision.autonomous_identity import (
    anonymous_state_summary,
    balanced_partition_sample,
    build_pairwise_critic_prompt,
    evenly_spaced,
    frame_index,
    partition_hash,
    relevant_future_observations,
    signed_pairwise_preference,
)


def _obs(frame: int, class_name: str = "chair"):
    return {"frame_uid": f"run_f{frame:06d}", "class_name": class_name}


def test_partition_hash_ignores_entity_ids_but_not_grouping():
    first = {"entity-a": ["o1", "o2"], "entity-b": ["o3"]}
    renamed = {"random-x": ["o3"], "random-y": ["o2", "o1"]}
    changed = {"entity-a": ["o1"], "entity-b": ["o2", "o3"]}

    assert partition_hash(first) == partition_hash(renamed)
    assert partition_hash(first) != partition_hash(changed)


def test_temporal_helpers_are_deterministic():
    assert frame_index(_obs(12)) == 12
    assert evenly_spaced(("a", "b", "c", "d", "e"), 3) == ("a", "c", "e")
    with pytest.raises(ValueError, match="positive"):
        evenly_spaced(("a",), 0)


def test_relevant_future_observations_uses_union_of_all_states():
    observations = {
        "root-a": _obs(2),
        "root-b": _obs(0),
        "late-a": _obs(5),
        "late-b": _obs(7),
        "unrelated": _obs(9),
    }
    states = [
        {
            "membership": {
                "x": ["root-a", "late-a"],
                "y": ["root-b", "late-b"],
                "z": ["unrelated"],
            }
        },
        {
            "membership": {
                "q": ["root-a", "root-b", "late-a", "late-b"],
                "z": ["unrelated"],
            }
        },
    ]
    selected = relevant_future_observations(
        states=states,
        root_obs_uids=("root-a", "root-b"),
        observation_rows=observations,
        minimum_frame=5,
    )
    assert selected == ("late-a", "late-b")
    assert "unrelated" not in selected


def test_anonymous_summary_and_prompt_hide_action_names():
    observations = {"o1": _obs(5), "o2": _obs(8)}
    state = {
        "membership": {"private-entity": ["o1", "o2"]},
        "objects": [
            {
                "entity_uid": "private-entity",
                "member_observation_uids": ["o1", "o2"],
                "n_points": 42,
                "bbox_center": [1.0, 2.0, 3.0],
                "bbox_extent": [0.4, 0.5, 0.6],
            }
        ],
    }
    summary = anonymous_state_summary(
        state=state,
        evidence_id_by_obs={"o1": "V01", "o2": "V02"},
        observation_rows=observations,
    )
    prompt = build_pairwise_critic_prompt(
        incident_uid="blind-1",
        evidence_rows=[
            {"evidence_id": "V01", "frame_index": 5, "class_name": "chair"},
            {"evidence_id": "V02", "frame_index": 8, "class_name": "chair"},
        ],
        state_summaries={"STATE_A": summary, "STATE_B": summary},
    )

    assert summary["groups"][0]["evidence_ids"] == ["V01", "V02"]
    assert "private-entity" not in prompt
    assert "SAME_INSTANCE" not in prompt
    assert "SEPARATE_MEMBER_GROUPS" not in prompt


def test_signed_order_swapped_preference_uses_defer_as_zero():
    assert (
        signed_pairwise_preference(
            preferred_partition_hashes=("repair", "repair"),
            candidate_partition_hash="repair",
            noop_partition_hash="noop",
        )
        == 1.0
    )
    assert (
        signed_pairwise_preference(
            preferred_partition_hashes=("repair", "noop"),
            candidate_partition_hash="repair",
            noop_partition_hash="noop",
        )
        == 0.0
    )
    assert (
        signed_pairwise_preference(
            preferred_partition_hashes=(None, "repair"),
            candidate_partition_hash="repair",
            noop_partition_hash="noop",
        )
        == 0.5
    )
    with pytest.raises(ValueError, match="unknown partition"):
        signed_pairwise_preference(
            preferred_partition_hashes=("third",),
            candidate_partition_hash="repair",
            noop_partition_hash="noop",
        )


def test_balanced_partition_sample_does_not_let_long_group_dominate():
    observations = {
        **{f"a{index}": _obs(index) for index in range(1, 7)},
        "b1": _obs(2),
        "b2": _obs(6),
    }
    long_group = [f"a{index}" for index in range(1, 7)]
    short_group = ["b1", "b2"]
    states = [
        {"membership": {"long": long_group, "short": short_group}},
        {"membership": {"merged": long_group + short_group}},
    ]
    selected = balanced_partition_sample(
        values=tuple(long_group + short_group),
        states=states,
        observation_rows=observations,
        limit=4,
    )

    assert len(set(selected) & set(long_group)) == 2
    assert len(set(selected) & set(short_group)) == 2
