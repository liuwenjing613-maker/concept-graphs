"""Synthetic metric tests, no GT or map files required."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from eval_replica_instance_pq import instance_pq
from replica_eval_common import coverage_rows, nearest, validate_xyz, begin_output


class MetricTests(unittest.TestCase):
    def test_perfect_is_one(self):
        r, _ = instance_pq([7,7,21,21], [0,0,1,1], 2)
        self.assertEqual((r['pq'],r['rq'],r['sq']), (1.,1.,1.))

    def test_same_class_merge_is_not_perfect(self):
        r, _ = instance_pq([7,7,21,21], [0,0,0,0], 1)
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['pq']), (0,1,2,0.))

    def test_exact_half_iou_does_not_match(self):
        r, _ = instance_pq([7,7], [0,-1], 1)
        self.assertEqual(r['tp'], 0)

    def test_split_is_penalized(self):
        r, _ = instance_pq([7]*4, [0,0,1,1], 2)
        self.assertEqual((r['fp'],r['fn'],r['pq']), (2,1,0.))

    def test_abstention_keeps_gt_denominator(self):
        r, _ = instance_pq([7]*4, [0,0,0,-1], 1)
        self.assertAlmostEqual(r['pq'], .75)
        self.assertEqual(r['uncovered_gt_points'], 1)

    def test_empty_prediction_is_zero_with_false_negatives(self):
        r, _ = instance_pq([7,8], [-1,-1], 0)
        self.assertEqual((r['pq'],r['fn'],r['fp']), (0.,2,0))

    def test_empty_gt_is_undefined(self):
        with self.assertRaises(ValueError): instance_pq([-1], [0], 1)

    def test_missing_object_is_false_negative(self):
        r, _ = instance_pq([7,7,8,8], [0,0,-1,-1], 1)
        self.assertEqual((r['tp'],r['fn']), (1,1))
        self.assertAlmostEqual(r['pq'], 2/3)

    def test_zero_support_duplicate_is_false_positive(self):
        r, _ = instance_pq([7,7], [0,0], 2)
        self.assertEqual((r['tp'],r['fp']), (1,1))
        self.assertAlmostEqual(r['pq'], 2/3)

    def test_void_subtracted_from_union(self):
        r, _ = instance_pq([7,7,-1,-1], [0,0,0,0], 1)
        self.assertEqual(r['pq'], 1.)

    def test_unmatched_majority_void_prediction_ignored(self):
        r, _ = instance_pq([7,7,-1,-1], [0,1,1,1], 2)
        self.assertEqual((r['tp'],r['ignored_void_instances'],r['fp'],r['fn']), (0,1,1,1))

    def test_matched_prediction_is_not_ignored(self):
        r, _ = instance_pq([7,-1,-1,-1], [0,0,0,0], 1)
        self.assertEqual((r['pq'],r['ignored_void_instances']), (1.,0))

    def test_id_relabeling_invariant(self):
        a, _ = instance_pq([7,7,8,8,8], [0,0,1,1,-1], 2)
        b, _ = instance_pq([33,33,0,0,0], [1,1,0,0,-1], 2)
        for key in ('pq','rq','sq','tp','fp','fn'): self.assertEqual(a[key],b[key])

    def test_pq_equals_rq_times_sq(self):
        r, _ = instance_pq([7]*3+[8]*4, [0,0,-1,1,1,1,2], 3)
        self.assertAlmostEqual(r['pq'],r['rq']*r['sq'])

    def test_invalid_ids_fail(self):
        with self.assertRaises(ValueError): instance_pq([0], [1], 1)
        with self.assertRaises(ValueError): instance_pq([-2], [0], 1)

    def test_distance_boundary_inclusive_and_monotonic(self):
        rows=coverage_rows(np.array([0.,.025,.05,.10,np.inf]), [.025,.05,.10])
        self.assertEqual([r['covered_points'] for r in rows], [2,3,4])
        self.assertTrue(all(r['total_points']==5 for r in rows))

    def test_coverage_empty_map_is_zero(self):
        d, owner=nearest(np.zeros((2,3)),np.empty((0,3)))
        self.assertTrue(np.all(owner==-1))
        self.assertEqual(coverage_rows(d,[.05])[0]['coverage'],0.)

    def test_nearest_matches_brute_force_across_chunks(self):
        rng=np.random.default_rng(4); q=rng.normal(size=(41,3)); p=rng.normal(size=(17,3))
        d,ix=nearest(q,p,workers=1,chunk_size=7)
        brute=np.linalg.norm(q[:,None,:]-p[None,:,:],axis=2)
        np.testing.assert_allclose(d,brute.min(axis=1),rtol=1e-14)
        np.testing.assert_array_equal(ix,brute.argmin(axis=1))

    def test_nonfinite_cloud_is_rejected(self):
        with self.assertRaises(ValueError): validate_xyz([[np.nan,0,0]],'bad')

    def test_invalid_distance_threshold_rejected(self):
        for t in ([0.],[-.1],[np.inf],[.05,.05]):
            with self.assertRaises(ValueError): coverage_rows(np.array([0.]),t)

    def test_output_overwrite_rejected(self):
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output_dir=Path(tmp))
            begin_output(args)
            with self.assertRaises(FileExistsError): begin_output(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
