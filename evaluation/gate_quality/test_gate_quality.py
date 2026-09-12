import unittest

from evaluation.gate_quality.evaluate_gate_quality import build_event_rows, build_metrics
from evaluation.gate_quality.metrics import confusion_cell, summarize_detection
from evaluation.observation_gt.instance_identity import PredictionIdentityConfig


def clean(obs_uid, gt_id, gt_class):
    return {
        "obs_uid": obs_uid,
        "quality_status": "CLEAN",
        "quality_reason": "test",
        "assigned_gt_instance": gt_id,
        "top1_gt_class": gt_class,
        "top1_purity": 0.95,
        "gt_support_ratio": 0.95,
        "matched_gt_points": 100,
    }


def version(version_uid, object_uid, members):
    return {
        "object_version_uid": version_uid,
        "object_uid": object_uid,
        "member_observation_uids": members,
    }


def association(seq, obs_uid, final_decision, target=None):
    return {
        "event_uid": f"e{seq}",
        "event_sequence": seq,
        "frame_uid": f"run_f{seq:06d}",
        "obs_uid": obs_uid,
        "object_uids_before": ["same", "wrong"],
        "candidate_object_version_uids": ["same@v1", "wrong@v1"],
        "decision": final_decision,
        "target_object_uid": target,
        "top1_score": 1.0,
        "top2_score": 0.9,
        "margin": 0.1,
        "sim_threshold": 1.2,
    }


class GateMetricTest(unittest.TestCase):
    def test_confusion_cells(self):
        self.assertEqual(confusion_cell(True, True), "TP")
        self.assertEqual(confusion_cell(True, False), "FN")
        self.assertEqual(confusion_cell(False, True), "FP")
        self.assertEqual(confusion_cell(False, False), "TN")

    def test_summary_reports_missed_error(self):
        rows = [
            {"ok": True, "error": True, "gate_triggered": True},
            {"ok": True, "error": True, "gate_triggered": False},
            {"ok": True, "error": False, "gate_triggered": True},
            {"ok": True, "error": False, "gate_triggered": False},
            {"ok": False, "error": None, "gate_triggered": False},
        ]
        result = summarize_detection(rows, evaluable_field="ok", error_field="error")
        self.assertEqual(result["confusion_matrix"], {"TP": 1, "FN": 1, "FP": 1, "TN": 1})
        self.assertEqual(result["error_recall"], 0.5)
        self.assertEqual(result["missed_error_rate"], 0.5)
        self.assertEqual(result["residual_error_rate_among_passed"], 0.5)
        self.assertEqual(result["evaluation_coverage"], 0.8)


class EventEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.labels = [
            clean("history_same", 1, "chair"),
            clean("history_wrong", 2, "table"),
            clean("current_missed", 1, "chair"),
            clean("current_caught", 1, "chair"),
            clean("current_clean", 1, "chair"),
        ]
        self.versions = [
            version("same@v1", "same", ["history_same"]),
            version("wrong@v1", "wrong", ["history_wrong"]),
        ]

    def test_all_proposals_include_true_and_false_gate_cases(self):
        associations = [
            association(1, "current_missed", "MERGE_TO_OBJECT", "wrong"),
            association(2, "current_caught", "MERGE_TO_OBJECT", "same"),
            association(3, "current_clean", "MERGE_TO_OBJECT", "same"),
        ]
        gates = [
            {
                "current_observation_uid": "current_caught",
                "event_id": "g2",
                "baseline_match_index": 1,
                "changed": True,
                "decision_source": "test",
                "route_reason": "test_new",
                "trigger": {"kind": "association", "reasons": ["low_margin"]},
            }
        ]
        rows, audit = build_event_rows(
            associations,
            gates,
            self.versions,
            self.labels,
            PredictionIdentityConfig(0.8),
        )
        metrics = build_metrics(rows, audit)
        confusion = metrics["wrong_fusion_detection"]["confusion_matrix"]
        self.assertEqual(confusion, {"TP": 1, "FN": 1, "FP": 0, "TN": 1})
        self.assertEqual(metrics["wrong_fusion_detection"]["error_recall"], 0.5)
        self.assertEqual(metrics["wrong_fusion_post_gate"]["corrected_error_count"], 1)
        missed = next(row for row in rows if row["obs_uid"] == "current_missed")
        self.assertEqual(missed["wrong_fusion_subtype"], "CROSS_CLASS")
        self.assertEqual(missed["wrong_fusion_confusion_cell"], "FN")

    def test_current_observation_is_excluded_from_candidate_identity(self):
        leaking_versions = [
            version("same@v1", "same", ["history_same", "current_missed"]),
            version("wrong@v1", "wrong", ["history_wrong"]),
        ]
        rows, audit = build_event_rows(
            [association(1, "current_missed", "MERGE_TO_OBJECT", "wrong")],
            [],
            leaking_versions,
            self.labels,
            PredictionIdentityConfig(0.8),
        )
        self.assertEqual(audit["current_observation_member_leak_count"], 1)
        self.assertFalse(
            rows[0]["candidate_snapshot_validation"]["candidate_snapshot_valid"]
        )


if __name__ == "__main__":
    unittest.main()
