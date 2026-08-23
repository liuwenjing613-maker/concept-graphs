import pytest

from scripts.prepare_revision_v1_live_fidelity import (
    parse_counts,
    select,
    validate_predeclared_source,
)


def _cases():
    return [
        {"case_uid": f"{failure_type.lower()}_{index}", "failure_type": failure_type}
        for failure_type in ("FALSE_MERGE", "FALSE_SPLIT", "WRONG_MEMBERSHIP")
        for index in range(5)
    ]


def test_staged_live_selection_is_deterministic_and_respects_declared_counts():
    counts = {"FALSE_MERGE": 4, "FALSE_SPLIT": 1, "WRONG_MEMBERSHIP": 1}

    first = select(_cases(), seed=20260823, counts=counts)
    second = select(_cases(), seed=20260823, counts=counts)

    assert first == second
    assert len(first) == 6
    assert sum(row["failure_type"] == "FALSE_MERGE" for row in first) == 4
    assert sum(row["failure_type"] == "FALSE_SPLIT" for row in first) == 1
    assert sum(row["failure_type"] == "WRONG_MEMBERSHIP" for row in first) == 1


def test_staged_live_count_parser_rejects_unknown_or_duplicate_types():
    assert parse_counts(["false_split=1", "wrong_membership=2"]) == {
        "FALSE_SPLIT": 1,
        "WRONG_MEMBERSHIP": 2,
    }
    with pytest.raises(ValueError):
        parse_counts(["UNKNOWN=1"])
    with pytest.raises(ValueError):
        parse_counts(["FALSE_SPLIT=1", "FALSE_SPLIT=2"])


def test_staged_subset_must_be_predeclared_by_valid_pre_live_manifest():
    selected = [{"case_uid": "case_a"}, {"case_uid": "case_b"}]
    source = {
        "selected_case_uids": ["case_a", "case_b", "case_c"],
        "outcome_screened": False,
        "frozen_before_new_live_outcomes": True,
    }

    result = validate_predeclared_source(selected, source)
    assert result["all_selected_cases_predeclared"] is True
    assert result["source_case_count"] == 3

    with pytest.raises(ValueError):
        validate_predeclared_source([{"case_uid": "not_predeclared"}], source)
