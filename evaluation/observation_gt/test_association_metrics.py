import unittest

from evaluation.observation_gt.association_metrics import (
    AMBIGUOUS_NEW,
    AMBIGUOUS_MERGE_REVIEW,
    CORRECT_CANONICAL,
    CORRECT_DISCARD,
    CORRECT_FRAGMENT,
    CORRECT_NEW,
    CORRECT_MERGE_REVIEW,
    DUPLICATE_CREATION,
    FALSE_DISCARD,
    MIXED_MASK_ATTACHMENT,
    UNSCORABLE_OBSERVATION,
    WRONG_FUSION,
    WRONG_MERGE_REVIEW,
    action,
    action_from_index,
    classify_action,
    classify_vlm_staged_decision,
    summarize_decisions,
    summarize_transitions,
)
from evaluation.observation_gt.instance_identity import (
    MappingReplayState,
    PredictionIdentityConfig,
    canonical_predictions,
    infer_prediction_identity,
)


def clean_label(gt_id, purity=0.95, support=0.9):
    return {
        "quality_status": "CLEAN",
        "assigned_gt_instance": gt_id,
        "top1_purity": purity,
        "gt_support_ratio": support,
        "matched_gt_points": 100,
    }


def identity(uid, gt_id, score, supports=1):
    return {
        "object_uid": uid,
        "identity_status": "RELIABLE",
        "gt_instance": gt_id,
        "dominant_score": score,
        "dominant_support_count": supports,
        "dominant_matched_gt_points": supports * 100,
    }


class AssociationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.identities = {
            "large": identity("large", 6001, 3.0, 3),
            "small": identity("small", 6001, 1.0, 1),
            "wrong": identity("wrong", 4002, 2.0, 2),
        }
        self.canonical = canonical_predictions(self.identities)
        self.label = clean_label(6001)

    def test_same_gt_noncanonical_is_correct_fragment(self):
        result = classify_action(
            self.label, action("ATTACH", "small"), self.identities, self.canonical
        )
        self.assertEqual(result["category"], CORRECT_FRAGMENT)
        self.assertTrue(result["identity_correct"])
        self.assertFalse(result["canonical_correct"])

    def test_canonical_target(self):
        result = classify_action(
            self.label, action("ATTACH", "large"), self.identities, self.canonical
        )
        self.assertEqual(result["category"], CORRECT_CANONICAL)

    def test_wrong_gt_is_wrong_fusion(self):
        result = classify_action(
            self.label, action("ATTACH", "wrong"), self.identities, self.canonical
        )
        self.assertEqual(result["category"], WRONG_FUSION)

    def test_new_with_same_gt_is_duplicate(self):
        result = classify_action(self.label, action("NEW"), self.identities, self.canonical)
        self.assertEqual(result["category"], DUPLICATE_CREATION)

    def test_unshown_uncertain_identity_does_not_poison_new(self):
        identities = {
            "wrong": identity("wrong", 4002, 2.0, 2),
            "unknown": {"object_uid": "unknown", "identity_status": "UNKNOWN_PREDICTION"},
        }
        result = classify_action(
            self.label,
            action("NEW"),
            identities,
            canonical_predictions(identities),
            candidate_object_uids=["wrong"],
        )
        self.assertEqual(result["category"], CORRECT_NEW)
        self.assertEqual(result["global_uncertain_prediction_count"], 1)
        self.assertEqual(result["shown_uncertain_prediction_count"], 0)

    def test_shown_uncertain_identity_makes_new_ambiguous(self):
        identities = {
            "wrong": identity("wrong", 4002, 2.0, 2),
            "unknown": {"object_uid": "unknown", "identity_status": "UNKNOWN_PREDICTION"},
        }
        result = classify_action(
            self.label,
            action("NEW"),
            identities,
            canonical_predictions(identities),
            candidate_object_uids=["wrong", "unknown"],
        )
        self.assertEqual(result["category"], AMBIGUOUS_NEW)
        self.assertEqual(result["new_action_context"], "UNCERTAIN_SHOWN_CANDIDATE")

    def test_duplicate_source_distinguishes_retrieval_from_selection(self):
        shown = classify_action(
            self.label,
            action("NEW"),
            self.identities,
            self.canonical,
            candidate_object_uids=["large"],
        )
        hidden = classify_action(
            self.label,
            action("NEW"),
            self.identities,
            self.canonical,
            candidate_object_uids=["wrong"],
        )
        self.assertEqual(shown["new_action_context"], "SAME_GT_IN_SHOWN_CANDIDATES")
        self.assertEqual(hidden["new_action_context"], "SAME_GT_OUTSIDE_SHOWN_CANDIDATES")

    def test_mixed_discard(self):
        result = classify_action(
            {"quality_status": "MIXED"}, action("DISCARD"), self.identities, self.canonical
        )
        self.assertEqual(result["category"], CORRECT_DISCARD)

    def test_correct_merge_review_replaces_clean_false_discard(self):
        fallback = classify_action(
            self.label, action("DISCARD"), self.identities, self.canonical
        )
        result = classify_vlm_staged_decision(
            self.label,
            {
                "kind": "MERGE_REVIEW",
                "same_aliases": ["A", "B"],
                "choice": "DISCARD",
                "reason_code": "MULTIPLE_SAME_AWAITING_PAIR_APPROVAL",
            },
            self.identities,
            self.canonical,
            {"A": "large", "B": "small"},
            fallback,
        )
        self.assertEqual(result["category"], CORRECT_MERGE_REVIEW)
        self.assertEqual(
            result["execution_fallback_result"]["category"], FALSE_DISCARD
        )

    def test_wrong_merge_review_when_candidate_gt_contradicts_observation(self):
        result = classify_vlm_staged_decision(
            self.label,
            {"kind": "MERGE_REVIEW", "same_aliases": ["A", "B"]},
            self.identities,
            self.canonical,
            {"A": "large", "B": "wrong"},
            {"category": FALSE_DISCARD},
        )
        self.assertEqual(result["category"], WRONG_MERGE_REVIEW)

    def test_ambiguous_merge_review_for_uncertain_candidate(self):
        identities = {
            **self.identities,
            "unknown": {"identity_status": "UNKNOWN_PREDICTION"},
        }
        result = classify_vlm_staged_decision(
            self.label,
            {"kind": "MERGE_REVIEW", "same_aliases": ["A", "B"]},
            identities,
            self.canonical,
            {"A": "large", "B": "unknown"},
            {"category": FALSE_DISCARD},
        )
        self.assertEqual(result["category"], AMBIGUOUS_MERGE_REVIEW)

    def test_nonclean_merge_review_keeps_observation_action_category(self):
        result = classify_vlm_staged_decision(
            {"quality_status": "MIXED"},
            {"kind": "MERGE_REVIEW", "same_aliases": ["A", "B"]},
            self.identities,
            self.canonical,
            {"A": "large", "B": "small"},
            {"category": CORRECT_DISCARD, "evaluable": True},
        )
        self.assertEqual(result["category"], CORRECT_DISCARD)
        self.assertEqual(result["vlm_intent"], "MERGE_REVIEW")

    def test_action_index(self):
        self.assertEqual(action_from_index(None, ["a"]), {"kind": "NEW"})
        self.assertEqual(action_from_index(-1, ["a"]), {"kind": "DISCARD"})
        self.assertEqual(
            action_from_index(0, ["a"]), {"kind": "ATTACH", "object_uid": "a"}
        )

    def test_transition_correction_and_harm(self):
        rows = [
            {
                "changed": True,
                "baseline_result": {"category": WRONG_FUSION},
                "final_result": {"category": CORRECT_FRAGMENT},
            },
            {
                "changed": True,
                "baseline_result": {"category": CORRECT_CANONICAL},
                "final_result": {"category": WRONG_FUSION},
            },
            {
                "changed": True,
                "baseline_result": {"category": MIXED_MASK_ATTACHMENT},
                "final_result": {"category": CORRECT_DISCARD},
            },
            {
                "changed": True,
                "baseline_result": {"category": AMBIGUOUS_NEW},
                "final_result": {"category": FALSE_DISCARD},
            },
        ]
        summary = summarize_transitions(rows)
        self.assertEqual(summary["denominator_count"], 4)
        self.assertEqual(summary["vlm_correction_count"], 2)
        self.assertEqual(summary["vlm_harm_count"], 1)
        self.assertEqual(summary["new_failure_count"], 2)
        self.assertAlmostEqual(summary["net_vlm_gain"], 0.25)
        self.assertAlmostEqual(summary["success_rate_delta"], 0.25)

    def test_all_event_denominator_and_discard_diagnostics(self):
        rows = [
            {
                "quality_status": "CLEAN",
                "final_action": action("ATTACH", "a"),
                "final_result": {"category": CORRECT_CANONICAL},
            },
            {
                "quality_status": "CLEAN",
                "final_action": action("ATTACH", "b"),
                "final_result": {"category": CORRECT_FRAGMENT},
            },
            {
                "quality_status": "CLEAN",
                "final_action": action("DISCARD"),
                "final_result": {"category": FALSE_DISCARD},
            },
            {
                "quality_status": "CLEAN",
                "final_action": action("NEW"),
                "final_result": {"category": AMBIGUOUS_NEW},
            },
            {
                "quality_status": "MIXED",
                "final_action": action("DISCARD"),
                "final_result": {"category": CORRECT_DISCARD},
            },
            {
                "quality_status": "MIXED",
                "final_action": action("ATTACH", "a"),
                "final_result": {"category": MIXED_MASK_ATTACHMENT},
            },
            {
                "quality_status": "UNSCORABLE",
                "final_action": action("DISCARD"),
                "final_result": {"category": UNSCORABLE_OBSERVATION},
            },
        ]
        summary = summarize_decisions(rows, "final_result")
        primary = summary["primary_event_metrics"]
        self.assertEqual(primary["denominator_count"], 7)
        self.assertEqual(primary["success_count"], 3)
        self.assertEqual(primary["failure_count"], 2)
        self.assertEqual(primary["ambiguous_count"], 1)
        self.assertEqual(primary["unscorable_count"], 1)
        self.assertAlmostEqual(primary["success_rate"], 3 / 7)

        clean = summary["clean_identity_metrics"]
        self.assertEqual(clean["denominator_count"], 4)
        self.assertAlmostEqual(clean["identity_success_rate"], 0.5)

        mixed = summary["mixed_discard_metrics"]
        self.assertEqual(mixed["denominator_count"], 2)
        self.assertAlmostEqual(mixed["discard_recall"], 0.5)

        discard = summary["discard_classifier_metrics"]
        self.assertEqual(discard["predicted_discard_count_all_events"], 3)
        self.assertEqual(discard["unscorable_discard_count"], 1)
        self.assertAlmostEqual(discard["precision_scorable"], 0.5)
        self.assertAlmostEqual(discard["clean_false_discard_rate"], 0.25)
        self.assertAlmostEqual(discard["binary_accuracy_scorable"], 4 / 6)


class IdentityAndReplayTest(unittest.TestCase):
    def test_weighted_identity_vote(self):
        labels = {
            "o1": clean_label(6001),
            "o2": clean_label(6001),
            "o3": clean_label(4002, purity=0.81, support=0.8),
        }
        result = infer_prediction_identity(
            "p", ["o1", "o2", "o3"], labels, PredictionIdentityConfig(0.6)
        )
        self.assertEqual(result["identity_status"], "RELIABLE")
        self.assertEqual(result["gt_instance"], 6001)

    def test_replay_create_associate_merge(self):
        replay = MappingReplayState()
        replay.apply({"event_type": "OBJECT_CREATE", "object_uid": "a", "obs_uid": "o1"})
        replay.apply({"event_type": "OBS_ASSOCIATE", "object_uid": "a", "obs_uid": "o2"})
        replay.apply({"event_type": "OBJECT_CREATE", "object_uid": "b", "obs_uid": "o3"})
        replay.apply(
            {
                "event_type": "OBJECT_MERGE",
                "source_object_uid": "b",
                "target_object_uid": "a",
                "member_union_after": ["o1", "o2", "o3"],
            }
        )
        self.assertEqual(replay.members, {"a": {"o1", "o2", "o3"}})


if __name__ == "__main__":
    unittest.main()
