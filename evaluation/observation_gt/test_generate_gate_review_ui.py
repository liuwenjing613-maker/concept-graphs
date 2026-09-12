from __future__ import annotations

import unittest

try:
    from .generate_gate_review import gt_oracle_choice
except ImportError:
    from generate_gate_review import gt_oracle_choice


class GTOracleChoiceTests(unittest.TestCase):
    def test_prefers_shown_canonical_candidate(self) -> None:
        choice, reason = gt_oracle_choice(
            "CLEAN",
            "uid-b",
            [
                {"alias": "A", "object_uid": "uid-a", "same_gt": False, "canonical": False},
                {"alias": "B", "object_uid": "uid-b", "same_gt": True, "canonical": True},
            ],
        )
        self.assertEqual(choice, "Candidate B")
        self.assertEqual(reason, "same GT canonical prediction")

    def test_mixed_observation_oracle_is_discard(self) -> None:
        self.assertEqual(
            gt_oracle_choice("MIXED", None, []),
            ("DISCARD", "MIXED observation"),
        )

    def test_reports_retrieval_failure_when_canonical_is_not_shown(self) -> None:
        choice, _ = gt_oracle_choice(
            "CLEAN",
            "uid-hidden",
            [{"alias": "A", "same_gt": False, "canonical": False}],
        )
        self.assertEqual(choice, "NOT SHOWN")

    def test_reliably_different_shown_candidates_make_new_the_oracle(self) -> None:
        choice, reason = gt_oracle_choice(
            "CLEAN",
            None,
            [
                {"alias": "A", "identity_status": "RELIABLE", "same_gt": False},
                {"alias": "B", "identity_status": "RELIABLE", "same_gt": False},
            ],
        )
        self.assertEqual(choice, "NEW")
        self.assertIn("reliably assigned", reason)

    def test_uncertain_shown_candidate_keeps_new_ambiguous(self) -> None:
        choice, _ = gt_oracle_choice(
            "CLEAN",
            None,
            [{"alias": "A", "identity_status": "UNKNOWN_PREDICTION", "same_gt": False}],
        )
        self.assertEqual(choice, "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
