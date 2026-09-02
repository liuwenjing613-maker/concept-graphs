from analyze_mixed_root_temporal_chain import (
    contiguous_ranges,
    is_pair_mixed,
    is_pure_reliable,
)
from run_mixed_interval_clean_create_replay import (
    create_constraint,
    quarantine_observations,
    select_old_preferred_boundary_near_ties,
)


def test_temporal_categories_are_disjoint_for_strict_rows():
    mixed = {
        "gt_assignment_eligible": True,
        "gt_top_id": 19,
        "gt_second_id": 15,
        "gt_purity": 0.55,
        "mask_mixed": True,
        "mask_two_foreground": True,
    }
    pure = {
        "gt_assignment_eligible": True,
        "gt_top_id": 19,
        "gt_second_id": 17,
        "gt_purity": 0.99,
        "mask_mixed": False,
        "mask_two_foreground": False,
    }
    assert is_pair_mixed(mixed, {15, 19})
    assert not is_pure_reliable(mixed, {15, 19})
    assert is_pure_reliable(pure, {15, 19})
    assert not is_pair_mixed(pure, {15, 19})


def test_quarantine_removes_only_requested_memberships():
    state = {
        "membership": {"entity-a": ["a", "b"], "entity-b": ["c"]},
        "objects": [
            {"member_observation_uids": ["a", "b"]},
            {"obs_uids": ["c"]},
        ],
    }
    filtered, edit = quarantine_observations(state, ["b", "c"])
    assert filtered["membership"] == {"entity-a": ["a"]}
    assert edit["removed_membership_count"] == 2
    assert edit["still_owned_observation_uids"] == []
    assert state["membership"]["entity-a"] == ["a", "b"]


def test_constraint_and_ranges_are_deterministic():
    constraint = create_constraint(
        trigger_obs="observation",
        trigger_event_uid="event",
        trigger_sequence=7,
        target_lineage_uid="old-lineage",
    ).as_dict()
    assert constraint["type"] == "CREATE_INSTANCE"
    assert constraint["active_from_sequence"] == 7
    assert constraint["active_until_sequence"] == 7
    assert constraint["separate_from_identity_uids"] == ["old-lineage"]
    assert contiguous_ranges([1, 2, 4, 7, 6]) == [
        {"start_frame": 1, "end_frame": 2, "frame_count": 2},
        {"start_frame": 4, "end_frame": 4, "frame_count": 1},
        {"start_frame": 6, "end_frame": 7, "frame_count": 2},
    ]


def test_margin_selection_uses_scores_and_boundary_entities_only():
    state = {
        "decision_trace": [
            {
                "obs_uid": "near",
                "event_uid": "event-near",
                "frame_idx": 11,
                "applied_match": 1,
                "natural_candidates": [
                    {"entity_uid": "old", "index": 1, "score": 1.90, "eligible": True},
                    {"entity_uid": "new", "index": 2, "score": 1.88, "eligible": True},
                ],
            },
            {
                "obs_uid": "far",
                "event_uid": "event-far",
                "frame_idx": 12,
                "applied_match": 1,
                "natural_candidates": [
                    {"entity_uid": "old", "index": 1, "score": 1.90, "eligible": True},
                    {"entity_uid": "new", "index": 2, "score": 1.20, "eligible": True},
                ],
            },
        ]
    }
    result = select_old_preferred_boundary_near_ties(
        state,
        old_entity_uid="old",
        new_entity_uid="new",
        after_frame=10,
        margin_delta=0.03,
    )
    assert result["selected_count"] == 1
    assert result["selected"][0]["obs_uid"] == "near"
    assert round(result["minimum_unselected_old_preferred_margin"], 6) == 0.7
