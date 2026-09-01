from label_logic import derive_event_label, validate_blind_label, validate_final_label


def blind(matches, quality="CLEAN_SINGLE_INSTANCE"):
    return validate_blind_label(
        {
            "observation_quality": quality,
            "matching_candidate_codes": matches,
            "physical_instance_note": "same chair",
            "blind_review_seconds": 10,
        },
        {"A", "B", "C"},
    )


def final(blind_row, target="CLEAN_SINGLE_INSTANCE", outside="NOT_NEEDED", evidence="YES"):
    return validate_final_label(
        {
            "target_state": target,
            "outside_candidate_status": outside,
            "evidence_sufficient": evidence,
            "confidence": 4,
            "final_review_seconds": 5,
            "notes": "enough" if evidence != "YES" else "",
        },
        blind_row,
    )


def test_keep_is_derived_when_selected_candidate_matches():
    b = blind(["A", "B"])
    result = derive_event_label(b, final(b), "B")
    assert result["derived_action"] == "KEEP"
    assert result["is_root_false_attach"] is False


def test_reassign_is_root_when_another_candidate_matches():
    b = blind(["A"])
    result = derive_event_label(b, final(b), "B")
    assert result["derived_action"] == "REASSIGN"
    assert result["is_root_false_attach"] is True


def test_new_is_root_when_no_matching_node_exists():
    b = blind(["NONE_SHOWN"])
    f = final(b, outside="NO_MATCHING_NODE_EXISTS")
    result = derive_event_label(b, f, "B")
    assert result["derived_action"] == "NEW"
    assert result["is_root_false_attach"] is True


def test_precontaminated_target_is_not_another_root():
    b = blind(["A"])
    f = final(b, target="ALREADY_CONTAMINATED")
    result = derive_event_label(b, f, "B")
    assert result["derived_status"] == "CASCADE_OR_PRECONTAMINATED"
    assert result["is_root_false_attach"] is False


def test_mixed_mask_is_excluded():
    b = blind(["A"], quality="MIXED_MULTIPLE_INSTANCES")
    result = derive_event_label(b, final(b), "B")
    assert result["eligible_main"] is False
    assert result["derived_action"] == "EXCLUDE"


def test_special_match_cannot_mix_with_candidate():
    try:
        blind(["A", "NONE_SHOWN"])
    except ValueError:
        pass
    else:
        raise AssertionError("mixed special/candidate selection should fail")

