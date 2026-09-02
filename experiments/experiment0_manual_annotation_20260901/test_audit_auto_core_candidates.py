#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_auto_core_candidates.py")
SPEC = importlib.util.spec_from_file_location("audit_auto_core_candidates", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def gt(obs_uid: str, gt_id: int, frame: int, *, mixed: bool = False, second: int | None = None):
    return {
        "obs_uid": obs_uid,
        "frame_idx": frame,
        "gt_assignment_eligible": True,
        "gt_top_id": gt_id,
        "gt_top_pixels": 90,
        "gt_purity": 0.95 if not mixed else 0.65,
        "gt_second_id": second,
        "gt_second_pixels": 10 if second is not None else 0,
        "gt_second_fraction": 0.10 if second is not None else 0.0,
        "mask_mixed": mixed,
        "mask_two_foreground": mixed,
    }


class TargetHistoryAuditTest(unittest.TestCase):
    def test_clean_target_passes(self):
        rows = {"a": gt("a", 1, 1), "b": gt("b", 1, 2)}
        result = MODULE.summarize_target(["a", "b"], rows, {}, current_gt_id=2)
        self.assertTrue(result["causally_clean"])
        self.assertEqual(result["joint_state"], "CLEAN_SINGLE_INSTANCE")

    def test_prior_current_identity_fails_even_once(self):
        rows = {
            "a": gt("a", 1, 1),
            "b": gt("b", 1, 2),
            "c": gt("c", 2, 3, mixed=True, second=1),
        }
        result = MODULE.summarize_target(["a", "b", "c"], rows, {}, current_gt_id=2)
        self.assertFalse(result["causally_clean"])
        self.assertEqual(result["prior_current_identity_evidence_count"], 1)
        self.assertIn("CURRENT_IDENTITY_ALREADY_PRESENT_BEFORE_EVENT", result["gate_reasons"])

    def test_repeated_mixed_history_is_precontaminated(self):
        rows = {
            "a": gt("a", 1, 1, mixed=True, second=2),
            "b": gt("b", 1, 2, mixed=True, second=2),
            "c": gt("c", 1, 3),
        }
        result = MODULE.summarize_target(["a", "b", "c"], rows, {}, current_gt_id=3)
        self.assertFalse(result["causally_clean"])
        self.assertEqual(result["joint_state"], "ALREADY_CONTAMINATED")
        self.assertTrue(result["persistent_pixel_contamination_2obs_2frames_5pct"])


if __name__ == "__main__":
    unittest.main()
