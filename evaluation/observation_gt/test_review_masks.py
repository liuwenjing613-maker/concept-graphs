import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.observation_gt.review_masks import ReviewDataset, mask_bounds


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ReviewMasksTest(unittest.TestCase):
    def make_dataset(self, root: Path) -> ReviewDataset:
        run_dir = root / "run"
        metrics_dir = run_dir / "observation_gt_metrics"
        masks = run_dir / "evidence/processed_masks"
        frames = run_dir / "frames"
        masks.mkdir(parents=True)
        frames.mkdir(parents=True)
        labels = []
        observations = []
        frame_rows = []
        event_rows = []
        for index, status in enumerate(("MIXED", "UNSCORABLE", "CLEAN", "MIXED")):
            uid = f"scene_f{index:06d}_r0000"
            frame_uid = f"scene_f{index:06d}"
            mask = np.zeros((80, 120), dtype=bool)
            mask[20:60, 30:90] = True
            np.savez_compressed(masks / f"{uid}.npz", mask=mask)
            Image.new("RGB", (120, 80), (35 + index * 20, 50, 70)).save(
                frames / f"{index}.jpg"
            )
            labels.append(
                {
                    "obs_uid": uid,
                    "frame_uid": frame_uid,
                    "frame_index": index,
                    "class_name": "chair",
                    "quality_status": status,
                    "quality_reason": "insufficient_instance_purity",
                    "mask_area": int(mask.sum()),
                    "valid_depth_ratio": 1.0,
                    "matched_gt_points": 80,
                    "gt_support_ratio": 0.9,
                    "top1_purity": 0.6,
                    "top2_purity": 0.4,
                }
            )
            observations.append(
                {
                    "obs_uid": uid,
                    "processed_mask_ref": {
                        "path": f"evidence/processed_masks/{uid}.npz",
                        "key": "mask",
                    },
                }
            )
            frame_rows.append(
                {"frame_uid": frame_uid, "rgb_ref": {"path": f"frames/{index}.jpg"}}
            )
            if status == "MIXED" and index == 0:
                event_rows.append(
                    {
                        "obs_uid": uid,
                        "quality_status": status,
                        "baseline_result": {"category": "MIXED_MASK_NEW"},
                        "final_result": {"category": "CORRECT_DISCARD"},
                        "changed": True,
                    }
                )
            if status == "MIXED" and index == 3:
                event_rows.append(
                    {
                        "obs_uid": uid,
                        "quality_status": status,
                        "baseline_result": {"category": "MIXED_MASK_ATTACHMENT"},
                        "final_result": {"category": "MIXED_MASK_ATTACHMENT"},
                        "changed": False,
                    }
                )
            if status == "CLEAN":
                event_rows.append(
                    {
                        "obs_uid": uid,
                        "quality_status": status,
                        "baseline_result": {"category": "CORRECT_CANONICAL"},
                        "final_result": {"category": "FALSE_DISCARD"},
                        "changed": True,
                    }
                )
        write_jsonl(metrics_dir / "observation_labels.jsonl", labels)
        write_jsonl(run_dir / "evidence/observations.jsonl", observations)
        write_jsonl(run_dir / "evidence/frames.jsonl", frame_rows)
        write_jsonl(metrics_dir / "event_decisions.jsonl", event_rows)
        event_id = "gate_000003"
        write_jsonl(
            run_dir / "blocking_association_gate/events.jsonl",
            [
                {
                    "event_id": event_id,
                    "current_observation_uid": "scene_f000003_r0000",
                    "schema_version": "blocking-association-gate-v7-vlmsplit",
                    "mode": "v7_merge",
                    "trigger": {
                        "kind": "association",
                        "reasons": ["association_margin", "mask_change"],
                        "top1": 1.31,
                        "top2": 1.26,
                        "margin": 0.05,
                    },
                    "audit_scores_hidden_from_vlm": {
                        "candidate_scores": {"A": 1.31, "B": 1.26},
                        "sim_threshold": 1.2,
                        "margin_threshold": 0.2,
                        "threshold_distance": 0.3,
                    },
                    "candidate_iou_prefilter_hidden_from_vlm": {
                        "iou_threshold": 0.85,
                    },
                    "candidate_alias_to_object_index": {"A": 2, "B": 8},
                    "model_output": {"choice": "A"},
                }
            ],
        )
        quality_path = (
            run_dir
            / "blocking_association_gate/events"
            / event_id
            / "quality/result.json"
        )
        quality_path.parent.mkdir(parents=True)
        quality_path.write_text(
            json.dumps(
                {
                    "value": {
                        "status": "USABLE",
                        "reason": "The mask primarily covers one chair.",
                    },
                    "attempts": [{"attempt": 1}],
                    "timeout_count": 0,
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
        return ReviewDataset(run_dir, metrics_dir)

    def test_indexes_quality_and_discard_cohorts_and_renders_rgb_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            self.assertEqual(
                dataset.summary()["scope_counts"],
                {
                    "VLM_USABLE_GT_MIXED": 1,
                    "FALSE_DISCARD_CLEAN": 1,
                    "CORRECT_DISCARD_MIXED": 1,
                    "MASK_QUALITY": 3,
                },
            )
            page = dataset.query(
                scope="FALSE_DISCARD_CLEAN",
                status="CLEAN",
                reason="ALL",
                text="",
                page=1,
                page_size=10,
            )
            self.assertEqual(page["total"], 1)
            self.assertEqual(page["rows"][0]["final_category"], "FALSE_DISCARD")
            payload, size = dataset.render(page["rows"][0]["obs_uid"], view="crop", max_side=720)
            self.assertTrue(payload.startswith(b"RIFF"))
            self.assertLess(size[0], 120)
            rendered = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB")).astype(np.int16)
            vivid_green = (
                (rendered[..., 1] > 190)
                & (rendered[..., 1] > rendered[..., 0] + 60)
                & (rendered[..., 1] > rendered[..., 2] + 40)
            )
            self.assertGreater(int(vivid_green.sum()), 100)

            page = dataset.query(
                scope="VLM_USABLE_GT_MIXED",
                status="MIXED",
                reason="ALL",
                text="",
                page=1,
                page_size=10,
            )
            self.assertEqual(page["total"], 1)
            self.assertEqual(page["rows"][0]["vlm_quality_status"], "USABLE")
            self.assertEqual(
                page["rows"][0]["gate_trigger_reasons"],
                ["association_margin", "mask_change"],
            )
            self.assertEqual(page["rows"][0]["candidate_iou_threshold"], 0.85)
            self.assertEqual(page["rows"][0]["trigger_margin"], 0.05)

    def test_mask_bounds_include_padding(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[40:60, 80:120] = True
        x0, y0, x1, y1 = mask_bounds(mask)
        self.assertLess(x0, 80)
        self.assertLess(y0, 40)
        self.assertGreater(x1, 120)
        self.assertGreater(y1, 60)


if __name__ == "__main__":
    unittest.main()
