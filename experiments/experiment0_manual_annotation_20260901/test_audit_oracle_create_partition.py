from audit_oracle_create_partition import gt_recovery_metrics


def _row(gt_id: int, eligible: bool = True) -> dict:
    return {
        "gt_top_id": gt_id,
        "gt_assignment_eligible": eligible,
    }


def test_gt_recovery_detects_high_precision_partial_recall() -> None:
    observation_gt = {
        "new_kept": _row(19),
        "new_residual": _row(19),
        "old_target": _row(15),
        "unknown": _row(19, eligible=False),
    }
    result = gt_recovery_metrics(
        predicted_members=["new_kept"],
        affected_native_members=observation_gt,
        target_members=["new_residual", "old_target"],
        observation_gt=observation_gt,
        expected_gt_id=19,
    )
    assert result is not None
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["false_negative_observation_count"] == 1
    assert result["residual_expected_gt_in_target_fraction"] == 0.5


def test_gt_recovery_is_unavailable_without_distinct_gt_identity() -> None:
    assert (
        gt_recovery_metrics(
            predicted_members=[],
            affected_native_members=[],
            target_members=[],
            observation_gt={},
            expected_gt_id=None,
        )
        is None
    )
