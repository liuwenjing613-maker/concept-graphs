import json

from scripts.evaluate_revision_cmvic_pilot_v0 import (
    _critic_preferences,
    _risk_coverage,
)


def _row(uid: str, label: int, score: float) -> dict:
    return {
        "row_uid": uid,
        "case_uid": f"case_{uid}",
        "candidate_uid": f"candidate_{uid}",
        "beneficial_label": label,
        "score": score,
    }


def test_risk_coverage_marks_one_class_as_noncomparative() -> None:
    summary, curve = _risk_coverage(
        [
            _row("a", 0, 0.7),
            _row("b", 0, 0.2),
        ],
        method="CMVIC",
        score_field="score",
    )
    assert summary["status"] == "ONE_CLASS_ONLY"
    assert summary["aurc"] is None
    assert summary["positive_count"] == 0
    assert len(curve) == 2


def test_risk_coverage_requires_both_classes_for_exploratory_status() -> None:
    summary, _ = _risk_coverage(
        [
            _row("a", 1, 0.7),
            _row("b", 0, 0.2),
        ],
        method="CMVIC",
        score_field="score",
    )
    assert summary["status"] == "EXPLORATORY"


def test_order_inconsistent_critic_preferences_fail_closed(tmp_path) -> None:
    result_path = tmp_path / "critic.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "status": "PASS",
                        "request_uid": "r0",
                        "response": {"critic": {"preferred_state": "STATE_A"}},
                    },
                    {
                        "status": "PASS",
                        "request_uid": "r1",
                        "response": {"critic": {"preferred_state": "DEFER"}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    mappings = {
        "r0": {
            "candidate_uid": "candidate",
            "candidate_state_uid": "CANDIDATE",
            "noop_state_uid": "NOOP",
            "label_to_state_uid": {"STATE_A": "NOOP", "STATE_B": "CANDIDATE"},
        },
        "r1": {
            "candidate_uid": "candidate",
            "candidate_state_uid": "CANDIDATE",
            "noop_state_uid": "NOOP",
            "label_to_state_uid": {"STATE_A": "CANDIDATE", "STATE_B": "NOOP"},
        },
    }
    preferences, audits = _critic_preferences(
        critic_results=[result_path],
        execution_by_case={"case": {"critic_state_mappings": mappings}},
    )
    key = ("case", "candidate")
    assert key not in preferences
    assert audits[key]["mapped_preferences"] == [-1.0, 0.0]
    assert audits[key]["order_consistent"] is False
