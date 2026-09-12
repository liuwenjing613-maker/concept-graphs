import unittest

import numpy as np

from evaluation.observation_gt.observation_labeler import (
    GTReference,
    ObservationGTLabeler,
    ObservationLabelConfig,
    QUALITY_CLEAN,
    QUALITY_MIXED,
    QUALITY_UNSCORABLE,
)


def projected_grid(height=8, width=8):
    rows, cols = np.indices((height, width))
    return np.column_stack((cols.ravel(), rows.ravel(), np.ones(height * width)))


class ObservationLabelerTest(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "min_valid_depth_ratio": 0.5,
            "min_gt_support_ratio": 0.5,
            "min_matched_gt_points": 8,
            "min_core_matched_gt_points": 4,
            "mask_erosion_radius": 1,
            "max_sampled_points_per_mask": 128,
            "query_workers": 1,
        }
        values.update(overrides)
        return ObservationLabelConfig(**values)

    def reference(self, instances, semantics, *, instance_to_semantic=None):
        xyz = projected_grid()
        instances = np.asarray(instances, dtype=np.int64)
        semantics = np.asarray(semantics, dtype=np.int64)
        if instance_to_semantic is None:
            instance_to_semantic = {
                int(instance_id): int(np.unique(semantics[instances == instance_id])[0])
                for instance_id in np.unique(instances[semantics > 0])
            }
        return GTReference(
            xyz=xyz,
            semantic=semantics,
            instance=instances,
            semantic_classes=("wall", "ceiling", "floor", "chair", "other"),
            instance_to_semantic=instance_to_semantic,
        )

    def label(self, reference, config=None, class_name="chair"):
        labeler = ObservationGTLabeler(reference, config or self.config())
        return labeler.label_mask(
            obs_uid="scene_f000001_r0000",
            frame_uid="scene_f000001",
            class_name=class_name,
            mask=np.ones((8, 8), dtype=bool),
            depth_m=np.ones((8, 8), dtype=np.float64),
            intrinsics=np.eye(3),
            pose_camera_to_world=np.eye(4),
        )

    def test_clean_single_instance(self):
        row = self.label(self.reference(np.full(64, 4001), np.full(64, 4)))
        self.assertEqual(row["quality_status"], QUALITY_CLEAN)
        self.assertEqual(row["assigned_gt_instance"], 4001)
        self.assertAlmostEqual(row["top1_purity"], 1.0)

    def test_mixed_two_instances(self):
        instances = np.where(projected_grid()[:, 0] < 4, 4001, 4002)
        row = self.label(self.reference(instances, np.full(64, 4)))
        self.assertEqual(row["quality_status"], QUALITY_MIXED)
        self.assertIsNone(row["assigned_gt_instance"])

    def test_structural_gt_class_participates_in_instance_classification(self):
        row = self.label(self.reference(np.full(64, 1001), np.full(64, 1)))
        self.assertEqual(row["quality_status"], QUALITY_CLEAN)
        self.assertEqual(row["assigned_gt_instance"], 1001)
        self.assertEqual(row["top1_gt_class"], "wall")

    def test_unlabeled_region_is_unscorable(self):
        row = self.label(
            self.reference(
                np.full(64, -1), np.zeros(64), instance_to_semantic={}
            )
        )
        self.assertEqual(row["quality_status"], QUALITY_UNSCORABLE)
        self.assertEqual(row["quality_reason"], "unlabeled_without_instance_identity")
        self.assertEqual(row["unlabeled_gt_points"], 64)
        self.assertEqual(row["unlabeled_gt_ratio"], 1.0)

    def test_unlabeled_points_reduce_instance_purity(self):
        semantics = np.where(projected_grid()[:, 0] < 6, 4, 0)
        instances = np.where(semantics == 4, 4001, -1)
        row = self.label(
            self.reference(instances, semantics, instance_to_semantic={4001: 4})
        )
        self.assertEqual(row["quality_status"], QUALITY_MIXED)
        self.assertAlmostEqual(row["top1_purity"], 0.75)
        self.assertAlmostEqual(row["unlabeled_gt_ratio"], 0.25)

    def test_native_instance_zero_is_valid(self):
        row = self.label(
            self.reference(
                np.zeros(64), np.full(64, 4), instance_to_semantic={0: 4}
            )
        )
        self.assertEqual(row["quality_status"], QUALITY_CLEAN)
        self.assertEqual(row["assigned_gt_instance"], 0)

    def test_official_other_instance_is_clean(self):
        row = self.label(
            self.reference(
                np.full(64, 23), np.full(64, 5), instance_to_semantic={23: 5}
            ),
            class_name="other",
        )
        self.assertEqual(row["quality_status"], QUALITY_CLEAN)
        self.assertEqual(row["assigned_gt_instance"], 23)
        self.assertEqual(row["top1_gt_class"], "other")

    def test_detection_class_does_not_skip_instance_classification(self):
        row = self.label(
            self.reference(np.full(64, 4001), np.full(64, 4)), class_name="other"
        )
        self.assertEqual(row["quality_status"], QUALITY_CLEAN)
        self.assertEqual(row["assigned_gt_instance"], 4001)

    def test_distance_threshold_is_strict(self):
        reference = GTReference(
            xyz=np.asarray([[0.05, 0.0, 1.0]]),
            semantic=np.asarray([4]),
            instance=np.asarray([4001]),
            semantic_classes=("wall", "ceiling", "floor", "chair"),
            instance_to_semantic={4001: 4},
        )
        config = self.config(
            gt_match_distance_m=0.05,
            min_matched_gt_points=1,
            min_core_matched_gt_points=1,
            mask_erosion_radius=0,
        )
        result = ObservationGTLabeler(reference, config)._query(
            np.asarray([[0.0, 0.0, 1.0]])
        )
        self.assertEqual(result["matched_any_points"], 0)


if __name__ == "__main__":
    unittest.main()
