from conceptgraph.revision.cases import (
    apply_controlled_membership_corruption,
    canonical_membership,
    stable_entity_uid,
)
from conceptgraph.revision.evaluate import membership_metrics


def test_three_controlled_corruptions_are_deterministic_and_measurable():
    clean = {
        "A": ["run_f000001_r0000", "run_f000002_r0000", "run_f000003_r0000"],
        "B": ["run_f000001_r0001", "run_f000002_r0001", "run_f000003_r0001"],
    }
    cases = [
        {
            "failure_type": "FALSE_SPLIT",
            "obs_uid": "run_f000002_r0000",
            "source_identity_uid": "A",
            "target_identity_uid": None,
        },
        {
            "failure_type": "WRONG_MEMBERSHIP",
            "obs_uid": "run_f000002_r0000",
            "source_identity_uid": "A",
            "target_identity_uid": "B",
        },
        {
            "failure_type": "FALSE_MERGE",
            "obs_uid": "run_f000001_r0000",
            "source_identity_uid": "A",
            "target_identity_uid": "B",
        },
    ]
    for case in cases:
        first = apply_controlled_membership_corruption(clean, case)
        second = apply_controlled_membership_corruption(clean, case)
        assert first == second
        assert membership_metrics(clean, first)["member_f1"] < 1.0


def test_final_member_assignment_is_canonical_and_stable():
    first = canonical_membership({"b": ["run_f000002_r0000"], "a": ["run_f000001_r0000"]})
    second = canonical_membership({"a": ["run_f000001_r0000"], "b": ["run_f000002_r0000"]})
    assert first == second
    assert stable_entity_uid(first["a"]) == stable_entity_uid(second["a"])
