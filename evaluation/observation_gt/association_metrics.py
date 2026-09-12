from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .instance_identity import IDENTITY_RELIABLE


ACTION_ATTACH = "ATTACH"
ACTION_NEW = "NEW"
ACTION_DISCARD = "DISCARD"
ACTION_UNRESOLVED = "UNRESOLVED"

CORRECT_CANONICAL = "CORRECT_CANONICAL"
CORRECT_FRAGMENT = "CORRECT_FRAGMENT"
CORRECT_NEW = "CORRECT_NEW"
CORRECT_MERGE_REVIEW = "CORRECT_MERGE_REVIEW"
WRONG_FUSION = "WRONG_FUSION"
WRONG_MERGE_REVIEW = "WRONG_MERGE_REVIEW"
DUPLICATE_CREATION = "DUPLICATE_CREATION"
FALSE_DISCARD = "FALSE_DISCARD"
AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
AMBIGUOUS_NEW = "AMBIGUOUS_NEW"
AMBIGUOUS_MERGE_REVIEW = "AMBIGUOUS_MERGE_REVIEW"
CORRECT_DISCARD = "CORRECT_DISCARD"
MIXED_MASK_ATTACHMENT = "MIXED_MASK_ATTACHMENT"
MIXED_MASK_NEW = "MIXED_MASK_NEW"
UNSCORABLE_OBSERVATION = "UNSCORABLE_OBSERVATION"
UNRESOLVED_ACTION = "UNRESOLVED_ACTION"

CORRECT_IDENTITY_RESULTS = {
    CORRECT_CANONICAL,
    CORRECT_FRAGMENT,
    CORRECT_NEW,
    CORRECT_MERGE_REVIEW,
}
DETERMINATE_CLEAN_ERRORS = {
    WRONG_FUSION,
    WRONG_MERGE_REVIEW,
    DUPLICATE_CREATION,
    FALSE_DISCARD,
}
DETERMINATE_CLEAN_RESULTS = CORRECT_IDENTITY_RESULTS | DETERMINATE_CLEAN_ERRORS

OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_FAILURE = "FAILURE"
OUTCOME_AMBIGUOUS = "AMBIGUOUS"
OUTCOME_UNSCORABLE = "UNSCORABLE"
OUTCOME_UNRESOLVED = "UNRESOLVED"

SUCCESS_RESULTS = CORRECT_IDENTITY_RESULTS | {CORRECT_DISCARD}
FAILURE_RESULTS = DETERMINATE_CLEAN_ERRORS | {
    MIXED_MASK_ATTACHMENT,
    MIXED_MASK_NEW,
}
AMBIGUOUS_RESULTS = {
    AMBIGUOUS_TARGET,
    AMBIGUOUS_NEW,
    AMBIGUOUS_MERGE_REVIEW,
}


def action(kind: str, object_uid: str | None = None) -> dict[str, Any]:
    result = {"kind": str(kind)}
    if object_uid is not None:
        result["object_uid"] = str(object_uid)
    return result


def action_from_index(
    value: Any,
    association_object_uids_before: Iterable[str],
) -> dict[str, Any]:
    if value is None:
        return action(ACTION_NEW)
    index = int(value)
    if index == -1:
        return action(ACTION_DISCARD)
    objects = [str(item) for item in association_object_uids_before]
    if index < 0 or index >= len(objects):
        return action(ACTION_UNRESOLVED)
    return action(ACTION_ATTACH, objects[index])


def action_from_association(association: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(association.get("decision") or "")
    if decision == "CREATE_OBJECT":
        return action(ACTION_NEW)
    if decision == "DISCARD_OBSERVATION":
        return action(ACTION_DISCARD)
    if decision == "MERGE_TO_OBJECT" and association.get("target_object_uid"):
        return action(ACTION_ATTACH, str(association["target_object_uid"]))
    return action(ACTION_UNRESOLVED)


def classify_action(
    observation_label: Mapping[str, Any],
    selected_action: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    canonical_by_gt: Mapping[int, str],
    candidate_object_uids: Iterable[str] | None = None,
) -> dict[str, Any]:
    quality = str(observation_label.get("quality_status") or "")
    kind = str(selected_action.get("kind") or ACTION_UNRESOLVED)
    if kind == ACTION_UNRESOLVED:
        return _result(UNRESOLVED_ACTION, False, None, None)
    if quality == "UNSCORABLE":
        return _result(UNSCORABLE_OBSERVATION, False, None, None)
    if quality == "MIXED":
        if kind == ACTION_DISCARD:
            return _result(CORRECT_DISCARD, True, None, None)
        if kind == ACTION_ATTACH:
            return _result(MIXED_MASK_ATTACHMENT, True, None, None)
        if kind == ACTION_NEW:
            return _result(MIXED_MASK_NEW, True, None, None)
        return _result(UNRESOLVED_ACTION, False, None, None)
    if quality != "CLEAN":
        return _result(UNSCORABLE_OBSERVATION, False, None, None)

    gt_id_value = observation_label.get("assigned_gt_instance")
    if gt_id_value is None:
        return _result(UNSCORABLE_OBSERVATION, False, None, None)
    gt_id = int(gt_id_value)
    reliable_same = {
        uid
        for uid, identity in identities.items()
        if identity.get("identity_status") == IDENTITY_RELIABLE
        and identity.get("gt_instance") is not None
        and int(identity["gt_instance"]) == gt_id
    }
    uncertain_existing = {
        uid
        for uid, identity in identities.items()
        if identity.get("identity_status") != IDENTITY_RELIABLE
    }
    shown_candidate_uids = {
        str(uid) for uid in (candidate_object_uids or ()) if uid is not None
    }
    uncertain_shown_candidates = uncertain_existing & shown_candidate_uids
    reliable_same_shown = reliable_same & shown_candidate_uids
    new_action_diagnostics = {
        "global_uncertain_prediction_count": len(uncertain_existing),
        "shown_uncertain_prediction_count": len(uncertain_shown_candidates),
        "reliable_same_gt_prediction_count": len(reliable_same),
        "shown_reliable_same_gt_prediction_count": len(reliable_same_shown),
    }
    if kind == ACTION_DISCARD:
        return _result(FALSE_DISCARD, True, False, False)
    if kind == ACTION_NEW:
        if reliable_same:
            duplicate_source = (
                "SAME_GT_IN_SHOWN_CANDIDATES"
                if reliable_same_shown
                else "SAME_GT_OUTSIDE_SHOWN_CANDIDATES"
            )
            return _result(
                DUPLICATE_CREATION,
                True,
                False,
                False,
                new_action_context=duplicate_source,
                **new_action_diagnostics,
            )
        if uncertain_shown_candidates:
            return _result(
                AMBIGUOUS_NEW,
                False,
                None,
                None,
                new_action_context="UNCERTAIN_SHOWN_CANDIDATE",
                uncertain_shown_candidate_uids=sorted(uncertain_shown_candidates),
                **new_action_diagnostics,
            )
        return _result(
            CORRECT_NEW,
            True,
            True,
            True,
            new_action_context="NO_RELIABLE_SAME_GT",
            **new_action_diagnostics,
        )
    if kind != ACTION_ATTACH:
        return _result(UNRESOLVED_ACTION, False, None, None)

    target_uid = str(selected_action.get("object_uid") or "")
    target = identities.get(target_uid)
    if target is None or target.get("identity_status") != IDENTITY_RELIABLE:
        return _result(AMBIGUOUS_TARGET, False, None, None)
    target_gt = int(target["gt_instance"])
    if target_gt != gt_id:
        return _result(WRONG_FUSION, True, False, False)
    canonical_uid = canonical_by_gt.get(gt_id)
    if canonical_uid == target_uid:
        return _result(CORRECT_CANONICAL, True, True, True)
    return _result(CORRECT_FRAGMENT, True, True, False)


def classify_vlm_staged_decision(
    observation_label: Mapping[str, Any],
    staged_decision: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    canonical_by_gt: Mapping[int, str],
    candidate_alias_to_object_uid: Mapping[str, str],
    fallback_execution_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Score V7's semantic intent before conservative execution fallback.

    A MERGE_REVIEW decision means the model identified multiple shown candidates
    as the same instance and submitted their pairs to the merge-vote gate.  A
    temporary DISCARD while that certificate is pending is execution behavior,
    not the semantic VLM decision being evaluated here.
    """
    kind = str(staged_decision.get("kind") or "")
    if kind != "MERGE_REVIEW":
        return dict(fallback_execution_result)

    aliases = [str(item) for item in staged_decision.get("same_aliases") or ()]
    object_uids = [candidate_alias_to_object_uid.get(alias) for alias in aliases]
    diagnostics = {
        "vlm_intent": "MERGE_REVIEW",
        "merge_review_same_aliases": aliases,
        "merge_review_candidate_object_uids": object_uids,
        "execution_fallback_result": dict(fallback_execution_result),
        "execution_fallback_action": str(staged_decision.get("choice") or ""),
        "merge_review_reason_code": staged_decision.get("reason_code"),
    }

    # MIXED/UNSCORABLE still use their observation-action category.  The merge
    # proposal remains a separate diagnostic because the current observation
    # cannot supply a unique GT identity for the full SAME claim.
    quality = str(observation_label.get("quality_status") or "")
    if quality != "CLEAN":
        return {
            **dict(fallback_execution_result),
            **diagnostics,
            "merge_review_evaluation": "OBSERVATION_NOT_CLEAN",
        }

    if len(aliases) < 2 or any(uid is None for uid in object_uids):
        return _result(
            AMBIGUOUS_MERGE_REVIEW,
            False,
            None,
            None,
            merge_review_evaluation="INVALID_OR_UNRESOLVED_ALIASES",
            **diagnostics,
        )
    candidate_identities = [identities.get(str(uid)) for uid in object_uids]
    if any(
        identity is None or identity.get("identity_status") != IDENTITY_RELIABLE
        for identity in candidate_identities
    ):
        return _result(
            AMBIGUOUS_MERGE_REVIEW,
            False,
            None,
            None,
            merge_review_evaluation="UNRELIABLE_CANDIDATE_IDENTITY",
            merge_review_candidate_identity_statuses=[
                (identity or {}).get("identity_status")
                for identity in candidate_identities
            ],
            **diagnostics,
        )

    candidate_gt_instances = [
        int(identity["gt_instance"]) for identity in candidate_identities
    ]
    observation_gt = observation_label.get("assigned_gt_instance")
    if observation_gt is None:
        return _result(
            AMBIGUOUS_MERGE_REVIEW,
            False,
            None,
            None,
            merge_review_evaluation="MISSING_OBSERVATION_GT",
            merge_review_candidate_gt_instances=candidate_gt_instances,
            **diagnostics,
        )
    observation_gt = int(observation_gt)
    if len(set(candidate_gt_instances)) == 1 and candidate_gt_instances[0] == observation_gt:
        return _result(
            CORRECT_MERGE_REVIEW,
            True,
            True,
            canonical_by_gt.get(observation_gt) in set(object_uids),
            merge_review_evaluation="ALL_CANDIDATES_MATCH_OBSERVATION_GT",
            merge_review_candidate_gt_instances=candidate_gt_instances,
            **diagnostics,
        )
    return _result(
        WRONG_MERGE_REVIEW,
        True,
        False,
        False,
        merge_review_evaluation="CANDIDATE_OR_OBSERVATION_GT_CONTRADICTION",
        merge_review_candidate_gt_instances=candidate_gt_instances,
        **diagnostics,
    )


def _result(
    category: str,
    evaluable: bool,
    identity_correct: bool | None,
    canonical_correct: bool | None,
    **details: Any,
) -> dict[str, Any]:
    return {
        "category": category,
        "evaluable": bool(evaluable),
        "identity_correct": identity_correct,
        "canonical_correct": canonical_correct,
        **details,
    }


def outcome_for_category(category: str) -> str:
    if category in SUCCESS_RESULTS:
        return OUTCOME_SUCCESS
    if category in FAILURE_RESULTS:
        return OUTCOME_FAILURE
    if category in AMBIGUOUS_RESULTS:
        return OUTCOME_AMBIGUOUS
    if category == UNSCORABLE_OBSERVATION:
        return OUTCOME_UNSCORABLE
    return OUTCOME_UNRESOLVED


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quality_from_category(category: str) -> str:
    if category in CORRECT_IDENTITY_RESULTS | DETERMINATE_CLEAN_ERRORS | AMBIGUOUS_RESULTS:
        return "CLEAN"
    if category in {CORRECT_DISCARD, MIXED_MASK_ATTACHMENT, MIXED_MASK_NEW}:
        return "MIXED"
    if category == UNSCORABLE_OBSERVATION:
        return "UNSCORABLE"
    return "UNKNOWN"


def summarize_decisions(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, Any]:
    rows = list(rows)
    row_categories = [
        str((row.get(field) or {}).get("category") or "MISSING") for row in rows
    ]
    categories = Counter(row_categories)
    outcomes = Counter(outcome_for_category(category) for category in row_categories)
    quality = Counter(
        str(row.get("quality_status") or _quality_from_category(category))
        for row, category in zip(rows, row_categories)
    )
    total = len(rows)
    success_count = outcomes[OUTCOME_SUCCESS]
    failure_count = outcomes[OUTCOME_FAILURE]
    ambiguous_count = outcomes[OUTCOME_AMBIGUOUS]
    unscorable_count = outcomes[OUTCOME_UNSCORABLE]
    unresolved_count = outcomes[OUTCOME_UNRESOLVED]
    scorable_count = quality["CLEAN"] + quality["MIXED"]
    resolved_scorable_count = success_count + failure_count

    determinate = sum(categories[item] for item in DETERMINATE_CLEAN_RESULTS)
    identity_correct = sum(categories[item] for item in CORRECT_IDENTITY_RESULTS)
    canonical_correct = categories[CORRECT_CANONICAL] + categories[CORRECT_NEW]
    clean_error = sum(categories[item] for item in DETERMINATE_CLEAN_ERRORS)
    attach_determinate = (
        categories[CORRECT_CANONICAL]
        + categories[CORRECT_FRAGMENT]
        + categories[WRONG_FUSION]
    )
    same_gt_attach = categories[CORRECT_CANONICAL] + categories[CORRECT_FRAGMENT]
    new_determinate = categories[CORRECT_NEW] + categories[DUPLICATE_CREATION]
    new_contexts = Counter(
        str((row.get(field) or {}).get("new_action_context"))
        for row in rows
        if (row.get(field) or {}).get("new_action_context")
    )
    new_with_global_uncertainty = sum(
        int((row.get(field) or {}).get("global_uncertain_prediction_count") or 0) > 0
        for row in rows
        if str((row.get(field) or {}).get("new_action_context") or "")
    )
    mixed_total = (
        categories[CORRECT_DISCARD]
        + categories[MIXED_MASK_ATTACHMENT]
        + categories[MIXED_MASK_NEW]
    )
    action_field = field.removesuffix("_result") + "_action"
    predicted_discard_count = sum(
        str((row.get(action_field) or {}).get("kind") or "") == ACTION_DISCARD
        for row in rows
    )
    correct_discard = categories[CORRECT_DISCARD]
    decision_false_discard = categories[FALSE_DISCARD]
    executed_false_discard = sum(
        str((row.get(action_field) or {}).get("kind") or "") == ACTION_DISCARD
        and str(row.get("quality_status") or "") == "CLEAN"
        for row in rows
    )
    unscorable_discard = sum(
        str((row.get(action_field) or {}).get("kind") or "") == ACTION_DISCARD
        and category == UNSCORABLE_OBSERVATION
        for row, category in zip(rows, row_categories)
    )
    executed_correct_discard = sum(
        str((row.get(action_field) or {}).get("kind") or "") == ACTION_DISCARD
        and str(row.get("quality_status") or "") == "MIXED"
        for row in rows
    )
    scorable_discard = executed_correct_discard + executed_false_discard
    true_negative_discard = max(0, quality["CLEAN"] - executed_false_discard)
    discard_precision = _ratio(correct_discard, scorable_discard)
    discard_recall = _ratio(correct_discard, mixed_total)
    discard_f1 = (
        2.0 * discard_precision * discard_recall / (discard_precision + discard_recall)
        if discard_precision is not None
        and discard_recall is not None
        and discard_precision + discard_recall
        else None
    )
    return {
        "event_count": total,
        "denominator_policy": {
            "primary": "all_vlm_gate_events",
            "primary_denominator_count": total,
            "subgroup_metrics_use_fixed_ground_truth_quality_cohorts": True,
        },
        "category_counts": dict(sorted(categories.items())),
        "primary_event_metrics": {
            "denominator_count": total,
            "success_count": success_count,
            "failure_count": failure_count,
            "ambiguous_count": ambiguous_count,
            "unscorable_count": unscorable_count,
            "unresolved_count": unresolved_count,
            "success_rate": _ratio(success_count, total),
            "failure_rate": _ratio(failure_count, total),
            "ambiguous_rate": _ratio(ambiguous_count, total),
            "unscorable_rate": _ratio(unscorable_count, total),
            "unresolved_rate": _ratio(unresolved_count, total),
        },
        "scorable_event_metrics": {
            "denominator_count": scorable_count,
            "resolved_count": resolved_scorable_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "ambiguous_count": ambiguous_count,
            "success_rate": _ratio(success_count, scorable_count),
            "failure_rate": _ratio(failure_count, scorable_count),
            "ambiguous_rate": _ratio(ambiguous_count, scorable_count),
            "resolved_coverage": _ratio(resolved_scorable_count, scorable_count),
            "resolved_accuracy": _ratio(success_count, resolved_scorable_count),
        },
        "clean_identity_metrics": {
            "denominator_count": quality["CLEAN"],
            "identity_correct_count": identity_correct,
            "canonical_correct_count": canonical_correct,
            "error_count": clean_error,
            "ambiguous_count": ambiguous_count,
            "identity_success_rate": _ratio(identity_correct, quality["CLEAN"]),
            "canonical_success_rate": _ratio(canonical_correct, quality["CLEAN"]),
            "error_rate": _ratio(clean_error, quality["CLEAN"]),
            "ambiguous_rate": _ratio(ambiguous_count, quality["CLEAN"]),
            "resolved_count": determinate,
            "resolved_identity_accuracy": _ratio(identity_correct, determinate),
        },
        "mixed_discard_metrics": {
            "denominator_count": mixed_total,
            "correct_discard_count": correct_discard,
            "kept_attachment_count": categories[MIXED_MASK_ATTACHMENT],
            "kept_new_count": categories[MIXED_MASK_NEW],
            "discard_recall": _ratio(correct_discard, mixed_total),
            "miss_rate": _ratio(
                categories[MIXED_MASK_ATTACHMENT] + categories[MIXED_MASK_NEW],
                mixed_total,
            ),
        },
        "discard_classifier_metrics": {
            "scorable_denominator_count": quality["CLEAN"] + quality["MIXED"],
            "predicted_discard_count_all_events": predicted_discard_count,
            "true_positive_mixed_discard": executed_correct_discard,
            "false_positive_clean_discard": executed_false_discard,
            "vlm_decision_false_discard_count": decision_false_discard,
            "clean_merge_review_fallback_discard_count": sum(
                category
                in {
                    CORRECT_MERGE_REVIEW,
                    WRONG_MERGE_REVIEW,
                    AMBIGUOUS_MERGE_REVIEW,
                }
                and str((row.get(action_field) or {}).get("kind") or "")
                == ACTION_DISCARD
                for row, category in zip(rows, row_categories)
            ),
            "false_negative_mixed_keep": (
                categories[MIXED_MASK_ATTACHMENT] + categories[MIXED_MASK_NEW]
            ),
            "true_negative_clean_keep": true_negative_discard,
            "unscorable_discard_count": unscorable_discard,
            "precision_scorable": discard_precision,
            "recall_mixed": discard_recall,
            "f1_scorable": discard_f1,
            "clean_false_discard_rate": _ratio(
                executed_false_discard, quality["CLEAN"]
            ),
            "binary_accuracy_scorable": _ratio(
                correct_discard + true_negative_discard,
                quality["CLEAN"] + quality["MIXED"],
            ),
        },
        "association_diagnostics": {
            "determinate_attach_count": attach_determinate,
            "wrong_fusion_rate": _ratio(categories[WRONG_FUSION], attach_determinate),
            "same_gt_attach_count": same_gt_attach,
            "fragment_attachment_rate": _ratio(
                categories[CORRECT_FRAGMENT], same_gt_attach
            ),
            "determinate_new_count": new_determinate,
            "duplicate_creation_rate": _ratio(
                categories[DUPLICATE_CREATION], new_determinate
            ),
        },
        "new_action_diagnostics": {
            "context_counts": dict(sorted(new_contexts.items())),
            "new_events_with_global_uncertainty_count": new_with_global_uncertainty,
            "ambiguity_policy": "only uncertain identities among shown gate candidates",
        },
        "merge_review_diagnostics": {
            "correct_count": categories[CORRECT_MERGE_REVIEW],
            "wrong_count": categories[WRONG_MERGE_REVIEW],
            "ambiguous_count": categories[AMBIGUOUS_MERGE_REVIEW],
            "policy": (
                "score staged MERGE_REVIEW intent; keep temporary DISCARD as "
                "execution-only fallback diagnostics"
            ),
        },
    }


def summarize_transitions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    transitions = Counter()
    outcome_transitions = Counter()
    correction_count = 0
    harm_count = 0
    new_failure_count = 0
    failure_recovery_count = 0
    changed_count = 0
    fragment_improvements = 0
    fragment_regressions = 0
    for row in rows:
        before = str((row.get("baseline_result") or {}).get("category") or "MISSING")
        after = str((row.get("final_result") or {}).get("category") or "MISSING")
        before_outcome = outcome_for_category(before)
        after_outcome = outcome_for_category(after)
        transitions[(before, after)] += 1
        outcome_transitions[(before_outcome, after_outcome)] += 1
        if not row.get("changed"):
            continue
        changed_count += 1
        if before_outcome != OUTCOME_SUCCESS and after_outcome == OUTCOME_SUCCESS:
            correction_count += 1
        if before_outcome == OUTCOME_SUCCESS and after_outcome != OUTCOME_SUCCESS:
            harm_count += 1
        if before_outcome != OUTCOME_FAILURE and after_outcome == OUTCOME_FAILURE:
            new_failure_count += 1
        if before_outcome == OUTCOME_FAILURE and after_outcome != OUTCOME_FAILURE:
            failure_recovery_count += 1
        if before == CORRECT_FRAGMENT and after == CORRECT_CANONICAL:
            fragment_improvements += 1
        if before == CORRECT_CANONICAL and after == CORRECT_FRAGMENT:
            fragment_regressions += 1
    total = len(rows)
    before_success = sum(
        count for (before, _), count in outcome_transitions.items() if before == OUTCOME_SUCCESS
    )
    after_success = sum(
        count for (_, after), count in outcome_transitions.items() if after == OUTCOME_SUCCESS
    )
    return {
        "denominator_policy": "all_vlm_gate_events",
        "denominator_count": total,
        "transition_counts": [
            {"baseline": before, "final": after, "count": count}
            for (before, after), count in sorted(transitions.items())
        ],
        "outcome_transition_counts": [
            {"baseline": before, "final": after, "count": count}
            for (before, after), count in sorted(outcome_transitions.items())
        ],
        "changed_event_count": changed_count,
        "changed_event_rate": _ratio(changed_count, total),
        "baseline_success_count": before_success,
        "baseline_success_rate": _ratio(before_success, total),
        "final_success_count": after_success,
        "final_success_rate": _ratio(after_success, total),
        "success_rate_delta": _ratio(after_success - before_success, total),
        "vlm_correction_count": correction_count,
        "vlm_harm_count": harm_count,
        "vlm_correction_rate_over_all_events": _ratio(correction_count, total),
        "vlm_harm_rate_over_all_events": _ratio(harm_count, total),
        "net_vlm_gain_over_all_events": _ratio(correction_count - harm_count, total),
        "net_vlm_gain": _ratio(correction_count - harm_count, total),
        "vlm_correction_rate_among_changed": _ratio(correction_count, changed_count),
        "vlm_harm_rate_among_changed": _ratio(harm_count, changed_count),
        "new_failure_count": new_failure_count,
        "new_failure_rate_over_all_events": _ratio(new_failure_count, total),
        "failure_recovery_count": failure_recovery_count,
        "failure_recovery_rate_over_all_events": _ratio(failure_recovery_count, total),
        "fragment_to_canonical_count": fragment_improvements,
        "canonical_to_fragment_count": fragment_regressions,
    }
