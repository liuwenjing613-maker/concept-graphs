"""CPU smoke tests; run on the mapping server, no API/GPU/full-scene run."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from conceptgraph.slam.association_gate import BlockingAssociationGate, HumanInputUnavailableError
from conceptgraph.slam.human_instance_merge import object_state, state_key
from conceptgraph.slam import utils


ROOT = Path(tempfile.mkdtemp(prefix='human_instance_merge_smoke_'))


class MergeReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / self._testMethodName
        self.root.mkdir()
        self.owner = BlockingAssociationGate(
            cfg={'sim_threshold': 1.2, 'association_gate': {'mode': 'human'}},
            output_dir=self.root / 'gate',
        )
        self.reviewer = self.owner.object_merge_reviewer(frame_idx=4, source_frame_id='000020', stage='periodic')
        self.gate = self.owner._instance_merge_gate
        source = self.root / 'source.jpg'
        self.assertTrue(cv2.imwrite(str(source), np.full((80, 120, 3), 120, dtype=np.uint8)))
        self.objects = []
        for k in range(3):
            masks = []
            for t in range(3):
                mask = np.zeros((80, 120), dtype=bool)
                mask[15:60, 15 + k * 8:55 + k * 8 + t] = True
                masks.append(mask)
            self.objects.append({
                'id': f'object-{k}', 'class_name': 'chair', 'num_detections': 3,
                'color_path': [source] * 3, 'mask': masks,
                'xyxy': [[15+k*8, 15, 57+k*8, 60]] * 3,
                'image_idx': [0, 1, 2], 'obs_uids': [f'{k}-{t}' for t in range(3)],
                'clip_ft': torch.tensor([1., 0.]),
                'pcd_np': np.array([[0., 0., 0.], [.1, .2, .4], [.2, .1, .6]]) + .05*k,
            })
        self.matrix = np.array([[0., .95], [.9, 0.]])

    def answer(self, choice):
        paths = sorted((self.gate.root / 'events').glob('*/decision.json'))
        event = json.loads(paths[-1].read_text())
        return event['answer_token'] + ' ' + choice

    def choose(self, choice):
        self.owner._human_input = lambda _: self.answer(choice)

    @staticmethod
    def fake_merge(target, source, *args, **kwargs):
        merged = copy.deepcopy(target)
        for key in ('obs_uids', 'image_idx', 'color_path', 'mask', 'xyxy'):
            merged[key] += source[key]
        merged['num_detections'] += source['num_detections']
        merged['pcd_np'] = np.vstack((target['pcd_np'], source['pcd_np']))
        return merged

    def run_merge(self, objects=None, matrix=None, reviewer='default', guard=None, decisions=None):
        with patch.object(utils, 'merge_obj2_into_obj1', side_effect=self.fake_merge) as merger:
            result, indices = utils.merge_overlap_objects(
                .8, .8, .8, objects if objects is not None else self.objects[:2],
                self.matrix if matrix is None else matrix,
                .02, False, .1, 3, 'overlap', 'cpu',
                merge_review=self.reviewer if reviewer == 'default' else reviewer,
                merge_guard=guard,
                merge_decision_callback=None if decisions is None else lambda *args: decisions.append(args),
            )
            return result, indices, merger.call_count

    def test_no_prevents_mutation_and_reverse_pair_reuses_answer(self):
        self.choose('N')
        before = state_key([object_state(o) for o in self.objects[:2]])
        decisions = []
        result, _, count = self.run_merge(decisions=decisions)
        self.assertEqual((len(result), count), (2, 0))
        self.assertEqual(before, state_key([object_state(o) for o in result]))
        self.assertEqual(self.gate.stats['reviewed'], 1)
        self.assertEqual(self.gate.stats['cached_rejections'], 1)
        self.assertTrue(all(d[5] == 'REJECT' and d[6][0].startswith('human_instance_merge_rejected') for d in decisions))

    def test_yes_merges_and_inactive_reverse_pair_does_not_prompt(self):
        self.choose('Y')
        result, indices, count = self.run_merge()
        self.assertEqual((len(result), count), (1, 1))
        self.assertEqual(result[0]['num_detections'], 6)
        self.assertEqual(indices, [None, 0])
        self.assertEqual(self.gate.stats['reviewed'], 1)

    def test_original_rejected_or_guarded_pairs_never_prompt(self):
        def fail(*args):
            raise AssertionError('must not prompt')
        self.run_merge(matrix=self.matrix * .5, reviewer=fail)
        self.objects[1]['clip_ft'] = torch.tensor([0., 1.])
        self.run_merge(reviewer=fail)
        self.objects[1]['clip_ft'] = torch.tensor([1., 0.])
        self.run_merge(reviewer=fail, guard=lambda *args: 'existing_guard')

    def test_disabled_hook_preserves_original_merge(self):
        result, _, count = self.run_merge(reviewer=None)
        self.assertEqual((len(result), count), (1, 1))
        self.assertFalse(self.gate.events)

    @patch.dict('os.environ', {'GATE_API_KEY': 'test-no-api-call'})
    def test_review_only_enabled_for_human_and_can_be_disabled(self):
        empty_gt = self.root / 'empty_gt.jsonl'
        empty_gt.write_text('')
        for mode in ('off', 'audit', 'oracle', 'vlm', 'human'):
            for enabled in (True, False):
                owner = BlockingAssociationGate(
                    cfg={'sim_threshold': 1.2, 'association_gate': {'mode': mode, 'human_merge_review': enabled, 'oracle_gt_path': str(empty_gt)}},
                    output_dir=self.root / f'{mode}-{enabled}',
                )
                cb = owner.object_merge_reviewer(frame_idx=4, source_frame_id='20', stage='final')
                self.assertEqual(callable(cb), mode == 'human' and enabled)

    def test_changed_geometry_invalidates_no_cache(self):
        self.choose('N')
        self.reviewer(*self.objects[:2], .95, 1., 1.)
        self.objects[0]['pcd_np'][0, 0] += .01
        self.reviewer(*self.objects[:2], .95, 1., 1.)
        self.assertEqual(self.gate.stats['reviewed'], 2)
        self.assertNotEqual(self.gate.events[0]['state_key'], self.gate.events[1]['state_key'])

    def test_chain_review_sees_object_after_previous_merge(self):
        self.choose('Y')
        matrix = np.array([[0., .99, 0.], [0., 0., .98], [0., 0., 0.]])
        result, _, count = self.run_merge(objects=self.objects, matrix=matrix)
        self.assertEqual((len(result), count), (1, 2))
        second = self.gate.events[1]
        self.assertEqual(second['object_A']['num_detections'], 6)
        saved = np.load(self.gate.root / 'events' / second['event_id'] / 'live_pair.npz')
        self.assertEqual(len(saved['object_A']), 6)

    def test_stale_and_unbound_answers_cannot_approve(self):
        answers = iter(['Y', 'M99999-AAAAAAAA Y', 'N'])
        self.owner._human_input = lambda _: next(answers, None) or self.answer('N')
        self.run_merge()
        self.assertEqual(self.gate.stats['invalid_or_stale_answers'], 3)
        self.assertEqual(self.gate.stats['approved'], 0)

    def test_eof_stops_before_map_mutation(self):
        def eof(_):
            raise EOFError()
        self.owner._human_input = eof
        before = state_key([object_state(o) for o in self.objects[:2]])
        with self.assertRaises(HumanInputUnavailableError):
            self.run_merge()
        self.assertEqual(before, state_key([object_state(o) for o in self.objects[:2]]))
        status = json.loads(next((self.gate.root / 'events').glob('*/decision.json')).read_text())
        self.assertEqual(status['status'], 'blocked_or_interrupted')

    def test_future_history_rejected(self):
        self.objects[0]['image_idx'][0] = 5
        with self.assertRaisesRegex(ValueError, 'future'):
            self.run_merge()

    def test_state_change_while_waiting_fails_closed(self):
        def mutate(_):
            answer = self.answer('Y')
            self.objects[0]['pcd_np'][0, 0] += .1
            return answer
        self.owner._human_input = mutate
        with self.assertRaisesRegex(RuntimeError, 'state changed'):
            self.run_merge()
        self.assertFalse(self.gate.events)

    def test_cards_use_three_histories_shared_scale_and_exact_live_cloud(self):
        self.choose('N')
        self.reviewer(*self.objects[:2], .95, 1., 1.)
        event = self.gate.events[0]
        a, b = event['evidence']
        self.assertEqual(len(a['selected_history']), 3)
        self.assertEqual(len(b['selected_history']), 3)
        self.assertEqual(a['point_cloud_event_shared_ranges_uid'], b['point_cloud_event_shared_ranges_uid'])
        directory = self.gate.root / 'events' / event['event_id']
        clouds = np.load(directory / 'live_pair.npz')
        for alias, obj in zip('AB', self.objects):
            self.assertTrue(np.array_equal(clouds[f'object_{alias}'], obj['pcd_np']))
            self.assertEqual(cv2.imread(str(directory / f'candidate_{alias}.jpg')).shape, (1024, 1024, 3))
        self.assertEqual(event['h_snapshot_uid'], event['c_bound_h_snapshot_uid'])
        page = (self.owner.output_dir / 'human_review.html').read_text()
        self.assertIn('data-choice="Y"', page)
        self.assertIn('data-choice="N"', page)
        self.assertIn(event['answer_token'], page)
        self.assertFalse(list(self.gate.root.glob('events/**/*.html')))
        self.assertNotIn('original_scores', page)

    def test_merge_objects_forwards_hook(self):
        self.choose('N')
        with patch.object(utils, 'compute_overlap_matrix_general', return_value=self.matrix):
            result = utils.merge_objects(.8, .8, .8, self.objects[:2], .02, False, .1, 3, 'overlap', 'cpu', merge_review=self.reviewer)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.gate.stats['reviewed'], 1)

    def test_mapper_wires_periodic_and_final_review(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'conceptgraph/slam/rerun_realtime_mapping.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Call) and isinstance(n.func.func, ast.Name)
                 and n.func.func.id == 'measure_time' and n.func.args
                 and isinstance(n.func.args[0], ast.Name) and n.func.args[0].id == 'merge_objects']
        self.assertEqual(len(calls), 1)
        review = next(kw.value for kw in calls[0].keywords if kw.arg == 'merge_review')
        stage = next(kw.value for kw in review.keywords if kw.arg == 'stage')
        self.assertIsInstance(stage, ast.IfExp)
        self.assertEqual((stage.body.value, stage.orelse.value), ('final', 'periodic'))


if __name__ == '__main__':
    print(f'Smoke artifacts (synthetic, not an online experiment): {ROOT}', flush=True)
    unittest.main(verbosity=2)
