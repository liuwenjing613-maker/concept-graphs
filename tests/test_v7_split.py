"""Unchanged parser/evidence invariants; new policy tests are in test_v7_no90_new.py."""
import copy,json,tempfile,unittest

from pathlib import Path

from types import SimpleNamespace

from unittest.mock import patch

import numpy as np

from conceptgraph.slam.v7_runtime import parse_stage,V7Runtime,EvidenceInvariantError

from conceptgraph.slam.v7_merge import V7MergeGate

from conceptgraph.slam.v7_evidence import LiveEvidence

from conceptgraph.slam.human_instance_merge import object_state,state_key

from conceptgraph.slam.association_gate import HumanInputUnavailableError

from conceptgraph.slam.staged_event_decision import decide_event

def obj(uid,offset=0):
    points=np.column_stack((np.arange(10,dtype=float)+offset,np.zeros(10),np.zeros(10)))
    return dict(id=uid,pcd=SimpleNamespace(points=points),clip_ft=np.zeros(3),
        obs_uids=[uid+'-h1'],image_idx=[0],num_detections=1)

class Parsing(unittest.TestCase):
    def test_fenced_and_unique_json(self):
        for text in ['```json\n{"choice":"SAME","confidence":5}\n```','thinking then {"choice":"SAME","confidence":5}']:
            self.assertEqual(parse_stage(text,'pairwise')[0]['choice'],'SAME')
    def test_duplicate_multiple_truncated(self):
        for text in ['{"choice":"SAME","choice":"DIFFERENT","confidence":5}',
            '{"choice":"SAME","confidence":5} {"choice":"DIFFERENT","confidence":5}','{"choice":"SAME",']:
            with self.subTest(text=text),self.assertRaises(ValueError):parse_stage(text,'pairwise')
    def test_merge_numeric_string_only_compatibility(self):
        value,mode,original=parse_stage('{"choice":"SAME","confidence":"5","reason":"同一实例"}','merge')
        self.assertEqual(value['confidence'],5);self.assertEqual(original['confidence'],'5')
        for invalid in ['true','6','"high"']:
            with self.assertRaises(ValueError):parse_stage('{"choice":"SAME","confidence":'+invalid+',"reason":"x"}','merge')
    def test_contradictory_quality_not_usable(self):
        x=dict(views=[dict(view='H1',objects='x',status='MULTIPLE',evidence='x')],choice='CLEAN',reason='x')
        with self.assertRaises(ValueError):parse_stage(json.dumps(x),'node_quality',['H1'])
    def test_quality_failure_blocks_candidates(self):
        self.assertEqual(decide_event(dict(status='INSUFFICIENT',reason='x'),[],['A'])['kind'],'PENDING')
    def test_multiple_same_requires_all_pairs(self):
        p=[dict(alias=a,value=dict(choice='SAME',confidence=0)) for a in 'ABC']
        self.assertEqual(decide_event(dict(status='USABLE',reason='x'),p,list('ABC'))['same_aliases'],list('ABC'))

class Geometry(unittest.TestCase):
    def test_asymmetric_containment(self):
        a,b=obj('a'),obj('b');b['pcd'].points=np.r_[b['pcd'].points,[[30,0,0]]]
        x=LiveEvidence.containment(a,b,.025)
        self.assertEqual(x['a_in_b'],1);self.assertAlmostEqual(x['b_in_a'],10/11);self.assertTrue(x['requires_human'])
    def test_strict_90_percent(self):
        a,b=obj('a'),obj('b');b['pcd'].points[-1]=[99,0,0]
        x=LiveEvidence.containment(a,b,.025)
        self.assertEqual(x['a_in_b'],.9);self.assertEqual(x['b_in_a'],.9);self.assertFalse(x['requires_human'])
    def test_no_mask_projection_or_plot_sampling(self):
        self.assertFalse(LiveEvidence.containment(obj('a'),obj('b',100),.025)['requires_human'])

class IncrementalFeatureAllowlist(unittest.TestCase):
    def test_sync_registers_only_new_logged_refs_and_read_never_scans_observations(self):
        from conceptgraph.slam.vlm_runtime import sha
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);exp=root/'run';(exp/'evidence').mkdir(parents=True)
            log=exp/'evidence/observations.jsonl';log.write_text('')
            feature=root/'feature.npz';np.savez(feature,features=np.array([[1.,2.,3.]]))
            ref=dict(path='../feature.npz',sha256=sha(feature),key='features',index=0)
            reader=LiveEvidence(SimpleNamespace(root=exp/'gate'));reader.sync()
            with self.assertRaises(EvidenceInvariantError):reader.array(ref)
            log.write_text(json.dumps(dict(obs_uid='o1',image_feat_ref=ref))+'\n');reader.sync()
            self.assertEqual(len(reader.allowed_feature_refs),1)
            class NoFullScan(dict):
                def values(self):raise AssertionError('must not rebuild allowlist on feature read')
            reader.observations=NoFullScan(reader.observations)
            np.testing.assert_array_equal(reader.array(ref),[1,2,3])
            reader.sync();self.assertEqual(len(reader.allowed_feature_refs),1)
            # A changed file is still hash checked on EVERY access.
            np.savez(feature,features=np.array([[4.,5.,6.]]))
            with self.assertRaises(EvidenceInvariantError):reader.array(ref)
    def test_unregistered_external_ref_stays_forbidden(self):
        from conceptgraph.slam.vlm_runtime import sha
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);exp=root/'run';(exp/'evidence').mkdir(parents=True)
            (exp/'evidence/observations.jsonl').write_text('')
            p=root/'unregistered.npz';np.savez(p,data=[1])
            reader=LiveEvidence(SimpleNamespace(root=exp/'gate'));reader.sync()
            with self.assertRaises(EvidenceInvariantError):reader.array(dict(path='../unregistered.npz',sha256=sha(p),key='data'))
