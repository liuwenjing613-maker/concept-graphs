#!/usr/bin/env python3
import unittest

from audit_experiment0_core_scope import (
    current_observation_gate,
    human_scope_row,
    target_gate,
    wilson_interval,
)


class CoreScopeAuditTest(unittest.TestCase):
    def test_clean_target_gate(self) -> None:
        result = target_gate(
            {
                "member_observation_count": 4,
                "dominant_gt_fraction": 1.0,
                "projected_gt_non_dominant_pixel_fraction_top2": 0.01,
                "objective_target_pre_state_joint": "CLEAN_SINGLE_INSTANCE",
            }
        )
        self.assertTrue(result["causally_clean"])

    def test_precontaminated_target_gate(self) -> None:
        result = target_gate(
            {
                "member_observation_count": 20,
                "dominant_gt_fraction": 0.98,
                "projected_gt_non_dominant_pixel_fraction_top2": 0.01,
                "objective_target_pre_state_joint": "ALREADY_CONTAMINATED",
            }
        )
        self.assertFalse(result["causally_clean"])
        self.assertIn("TARGET_HAS_PRE_EVENT_CONTAMINATION", result["reasons"])

    def test_mixed_current_observation_is_rejected(self) -> None:
        passed, reasons = current_observation_gate(
            {
                "human_observation_quality": "CLEAN_SINGLE_INSTANCE",
                "human_identity_evidence_status": "SUFFICIENT_FOR_IDENTITY",
                "human_identity_routing_eligible": True,
                "gt_assignment_eligible": True,
                "gt_purity": 0.99,
                "mask_mixed": True,
                "mask_two_foreground": True,
            },
            require_human=True,
        )
        self.assertFalse(passed)
        self.assertIn("CURRENT_MASK_MIXED", reasons)

    def test_false_split_is_out_even_with_clean_evidence(self) -> None:
        row = human_scope_row(
            {
                "case_uid": "example",
                "routing_label": "WRONG_NEW_FALSE_SPLIT",
                "original_action_type": "NEW",
                "current_mask": {
                    "human_observation_quality": "CLEAN_SINGLE_INSTANCE",
                    "human_identity_evidence_status": "SUFFICIENT_FOR_IDENTITY",
                    "human_identity_routing_eligible": True,
                    "gt_assignment_eligible": True,
                    "gt_purity": 1.0,
                    "mask_mixed": False,
                    "mask_two_foreground": False,
                },
            }
        )
        self.assertEqual(row["scope_status"], "OUT_FALSE_SPLIT_OR_NON_ATTACH")

    def test_zero_of_sixty_nine_wilson_interval(self) -> None:
        interval = wilson_interval(0, 69)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval[0], 0.0, places=12)
        self.assertAlmostEqual(interval[1], 0.0527372582, places=9)


if __name__ == "__main__":
    unittest.main()
