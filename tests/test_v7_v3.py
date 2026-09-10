import unittest
import numpy as np
from conceptgraph.slam.v7_merge_policy import containment_route,fragment_context
from conceptgraph.slam.v7_runtime import parse_stage

class Policy(unittest.TestCase):
    def test_boundaries_and_quality_veto(self):
        for q,a,b,want in [(['CLEAN','CLEAN'],.9,.8,'NO_HIGH_CONTAINMENT'),(['CLEAN','CLEAN'],.91,.6,'FRAGMENT_RESOLVER'),
            (['CLEAN','CLEAN'],.91,.61,'MUTUAL_APPROVAL'),(['CONTAMINATED','CLEAN'],1,1,'DEFER_CONTAMINATED'),
            (['INSUFFICIENT','CLEAN'],1,1,'DEFER_QUALITY_UNKNOWN'),([None,'CLEAN'],1,1,'DEFER_QUALITY_UNKNOWN')]:
            self.assertEqual(containment_route(q,dict(a_in_b=a,b_in_a=b)),want)
    def test_fragment_schema(self):
        for c in ['SAME_FRAGMENT','DISTINCT_OBJECT','UNCERTAIN']:
            self.assertEqual(parse_stage('{"choice":"'+c+'","reason":"依据"}','fragment')[0]['choice'],c)
        for x in ['{"choice":"SAME","reason":"x"}','{"choice":"SAME_FRAGMENT","reason":"x","confidence":5}']:
            with self.assertRaises(ValueError):parse_stage(x,'fragment')

class V3Gate(unittest.TestCase):
    def setUp(self):
        from test_v7_split import GateIntegration
        self.f=GateIntegration();self.f.setUp();self.addCleanup(self.f.tearDown)
        self.f.b['pcd'].points=np.r_[self.f.a['pcd'].points,np.ones((20,3))*100]
        self.fragment='UNCERTAIN';base=self.f.owner.vlm_runtime.stage
        def stage(eid,directory,task,*args):
            if task!='fragment':return base(eid,directory,task,*args)
            self.f.calls.append('fragment');return dict(task='fragment',value=None if self.fragment is None else dict(choice=self.fragment,reason='fixture'))
        self.f.owner.vlm_runtime.stage=stage
    def test_same_fragment_approves_once_then_executes(self):
        self.fragment='SAME_FRAGMENT';self.assertIsNone(self.f.review(1))
        event=self.f.gate.events[-1];self.assertEqual(event['decision_source'],'fragment_same')
        self.assertEqual(event['vote_after']['merge_streak'],0)
        self.f.gate.on_merged(self.f.a,self.f.b);self.assertEqual(event['execution'],'MERGED')
    def test_distinct_negative_vote_and_no_geometry_override(self):
        self.fragment='DISTINCT_OBJECT';self.assertIsNotNone(self.f.review(1))
        event=self.f.gate.events[-1];self.assertEqual(event['vote_after']['reject_total'],1)
        self.assertNotIn('decision_source',event)
    def test_uncertain_no_negative_vote_and_no_repeat_same_evidence(self):
        self.assertIsNotNone(self.f.review(1));calls=len(self.f.calls)
        self.assertIsNotNone(self.f.review(2));self.assertEqual(len(self.f.calls),calls)
        self.assertEqual(self.f.gate.events[-1]['vote_after']['reject_total'],0)
        self.f.a['obs_uids'].append('a-h2');self.f.a['num_detections']+=1
        self.f.review(3);self.assertGreater(len(self.f.calls),calls)
    def test_interface_failure_defers(self):
        self.fragment=None;self.f.review(1)
        self.assertEqual(self.f.gate.events[-1]['model_output']['choice'],'DEFER')
    def test_insufficient_skips_identity_and_fragment(self):
        self.f.quality='INSUFFICIENT';self.f.review(1)
        self.assertEqual(self.f.calls,['node_quality','node_quality'])
        self.assertEqual(self.f.gate.events[-1]['vote_after']['reject_total'],0)
    def test_contaminated_never_resolves_or_merges(self):
        self.f.quality='CONTAMINATED';self.fragment='SAME_FRAGMENT';self.f.review(1)
        self.assertNotIn('fragment',self.f.calls)
        self.assertEqual(self.f.gate.events[-1]['model_output']['choice'],'DEFER')

if __name__=='__main__':unittest.main()
