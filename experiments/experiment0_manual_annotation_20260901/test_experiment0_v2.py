import pytest
import threading

from label_logic_v2 import (
    derive_routing_label,
    validate_blind_label,
    validate_final_label,
)


def blind(matches, quality="CLEAN_SINGLE_INSTANCE", evidence="SUFFICIENT_FOR_IDENTITY"):
    return validate_blind_label(
        {
            "observation_quality": quality,
            "matching_candidate_codes": matches,
            "identity_evidence_status": evidence,
            "physical_instance_note": "same physical object",
            "blind_review_seconds": 8,
        },
        {"A", "B", "C"},
    )


def final(
    blind_row,
    action="ATTACH_EXISTING",
    target_state="CLEAN_SINGLE_INSTANCE",
    full_map="NOT_NEEDED_MATCH_SHOWN",
    outside=None,
):
    if action == "NEW":
        target_state = "NOT_APPLICABLE"
    return validate_final_label(
        {
            "target_pre_state": target_state,
            "full_map_status": full_map,
            "outside_matching_node_uids": outside or [],
            "confidence": 5,
            "causal_note": "",
            "notes": "",
            "final_review_seconds": 4,
        },
        blind_row,
        action,
    )


def test_five_identity_routing_cells_are_deterministic():
    b = blind(["B"])
    assert derive_routing_label(b, final(b), "ATTACH_EXISTING", "B")["routing_label"] == "CORRECT_ATTACH"

    b = blind(["A"])
    assert derive_routing_label(b, final(b), "ATTACH_EXISTING", "B")["routing_label"] == "WRONG_ATTACH_EXISTING"

    b = blind(["NONE_SHOWN"])
    f = final(b, full_map="NO_MATCHING_NODE_EXISTS")
    assert derive_routing_label(b, f, "ATTACH_EXISTING", "B")["routing_label"] == "SHOULD_HAVE_BEEN_NEW"

    b = blind(["NONE_SHOWN"])
    f = final(b, action="NEW", full_map="NO_MATCHING_NODE_EXISTS")
    assert derive_routing_label(b, f, "NEW", None)["routing_label"] == "CORRECT_NEW"

    b = blind(["A"])
    f = final(b, action="NEW")
    assert derive_routing_label(b, f, "NEW", None)["routing_label"] == "WRONG_NEW_FALSE_SPLIT"


def test_outside_full_map_target_is_existing_action():
    b = blind(["NONE_SHOWN"])
    f = final(
        b,
        full_map="MATCH_EXISTS_OUTSIDE",
        outside=["object-at-tminus"],
    )
    result = derive_routing_label(b, f, "ATTACH_EXISTING", "B")
    assert result["routing_label"] == "WRONG_ATTACH_EXISTING"
    assert result["legal_target_uids_outside"] == ["object-at-tminus"]


def test_precontamination_does_not_erase_routing_truth():
    b = blind(["A"])
    f = final(b, target_state="ALREADY_CONTAMINATED")
    result = derive_routing_label(b, f, "ATTACH_EXISTING", "B")
    assert result["routing_label"] == "WRONG_ATTACH_EXISTING"
    assert result["episode_review"] == "PRECONTAMINATED_REQUIRES_CAUSAL_REVIEW"


def test_invalid_observation_is_excluded_before_routing():
    b = blind(["A"], quality="BACKGROUND_OR_FRAGMENT")
    result = derive_routing_label(b, final(b), "ATTACH_EXISTING", "B")
    assert result["annotation_status"] == "EXCLUDED"
    assert result["routing_label"] == "OUT_OF_SCOPE_BACKGROUND_OR_FRAGMENT"


def test_uncertain_cannot_claim_sufficient_evidence():
    with pytest.raises(ValueError):
        blind(["UNCERTAIN"], evidence="SUFFICIENT_FOR_IDENTITY")


def test_partial_visibility_is_not_granularity_by_definition():
    b = blind(["A"], quality="CLEAN_SINGLE_INSTANCE")
    f = final(b)
    result = derive_routing_label(b, f, "ATTACH_EXISTING", "A")
    assert result["annotation_status"] == "COMPLETED"
    assert result["routing_label"] == "CORRECT_ATTACH"


def test_granularity_requires_explicit_part_whole_note():
    payload = {
        "observation_quality": "GRANULARITY_AMBIGUOUS",
        "matching_candidate_codes": ["UNCERTAIN"],
        "identity_evidence_status": "PARTIAL",
        "physical_instance_note": "",
        "blind_review_seconds": 2,
    }
    with pytest.raises(ValueError, match="part-whole"):
        validate_blind_label(payload, {"A", "B", "C"})
    payload["physical_instance_note"] = "无法判断坐垫是独立实例还是沙发本体的一部分"
    assert validate_blind_label(payload, {"A", "B", "C"})[
        "observation_quality"
    ] == "GRANULARITY_AMBIGUOUS"


def test_new_requires_not_applicable_target_state():
    b = blind(["A"])
    with pytest.raises(ValueError):
        validate_final_label(
            {
                "target_pre_state": "CLEAN_SINGLE_INSTANCE",
                "full_map_status": "NOT_NEEDED_MATCH_SHOWN",
                "outside_matching_node_uids": [],
                "confidence": 5,
                "final_review_seconds": 1,
            },
            b,
            "NEW",
        )


def test_v2_html_patch_loads_and_contains_required_fields():
    from serve_event_labels_v2 import HTML

    assert "identityEvidence" in HTML
    assert "outsideUids" in HTML
    assert "original_action_type" in HTML
    assert "evidenceStatus" not in HTML
    assert "局部可见不等于粒度歧义" in HTML
    assert "同类别、同材质" in HTML
    assert "window.confirm(summary)" in HTML


def test_blind_and_final_submissions_are_immutable_after_save():
    from serve_event_labels_v2 import AnnotationStoreV2

    store = object.__new__(AnnotationStoreV2)
    store.lock = threading.RLock()
    store.reload = lambda: None
    store.by_case = {"case-1": {}}
    store.drafts = {"case-1": {"blind": {}}}
    store.labels = {}
    with pytest.raises(ValueError, match="盲标已经锁定"):
        store.save_blind({"case_uid": "case-1"})

    store.labels = {"case-1": {}}
    with pytest.raises(ValueError, match="最终标签已经保存"):
        store.save_final({"case_uid": "case-1"})
