import copy

from make_v2_r2_adjudication_overlay import DECISIONS, adjudicate_label


def source_label(case_uid, quality, matches, action, target):
    return {
        "case_uid": case_uid,
        "blind": {
            "observation_quality": quality,
            "matching_candidate_codes": matches,
            "identity_evidence_status": "SUFFICIENT_FOR_IDENTITY",
            "physical_instance_note": None,
            "blind_review_seconds": 1,
        },
        "final": {
            "target_pre_state": "NOT_APPLICABLE"
            if action == "NEW"
            else "CLEAN_SINGLE_INSTANCE",
            "full_map_status": "NOT_NEEDED_MATCH_SHOWN",
            "outside_matching_node_uids": [],
            "confidence": 5,
            "causal_note": None,
            "notes": None,
            "final_review_seconds": 1,
        },
        "reveal": {
            "original_action_type": action,
            "original_target_code": target,
        },
    }


def test_partial_visibility_overlays_restore_routing_cells():
    label = source_label(
        "v2_r2_007", "BACKGROUND_OR_FRAGMENT", ["B"], "NEW", None
    )
    result = adjudicate_label(label, DECISIONS["v2_r2_007"], {"A", "B"})
    assert result["blind"]["observation_quality"] == "BORDERLINE_SINGLE_INSTANCE"
    assert result["derived"]["routing_label"] == "WRONG_NEW_FALSE_SPLIT"

    label = source_label(
        "v2_r2_008", "BACKGROUND_OR_FRAGMENT", ["B"], "ATTACH_EXISTING", "B"
    )
    result = adjudicate_label(label, DECISIONS["v2_r2_008"], {"A", "B"})
    assert result["derived"]["routing_label"] == "CORRECT_ATTACH"


def test_same_category_overlays_require_new_without_mutating_source():
    label = source_label(
        "v2_r2_011", "CLEAN_SINGLE_INSTANCE", ["A", "D"], "ATTACH_EXISTING", "D"
    )
    original = copy.deepcopy(label)
    result = adjudicate_label(label, DECISIONS["v2_r2_011"], {"A", "B", "C", "D"})
    assert label == original
    assert result["blind"]["matching_candidate_codes"] == ["NONE_SHOWN"]
    assert result["final"]["full_map_status"] == "NO_MATCHING_NODE_EXISTS"
    assert result["derived"]["routing_label"] == "SHOULD_HAVE_BEEN_NEW"
