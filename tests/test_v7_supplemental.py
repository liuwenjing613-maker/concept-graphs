"""Supplemental geometry only nominates; real map mutation still requires VLM votes."""
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import torch,open3d as o3d,httpx
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import merge_overlap_objects
from conceptgraph.slam.v7_endpoint_pool import EndpointPool
from conceptgraph.utils.evidence import EvidenceRecorder

class Supplemental(unittest.TestCase):
    def setUp(self):
        from test_v7_split import GateIntegration
        self.f=GateIntegration();self.f.setUp();self.addCleanup(self.f.tearDown)
        self.gate=self.f.gate;self.frame=1;self.native_calls=0
        self.objects=MapObjectList()
        for i in range(2):
            p=o3d.geometry.PointCloud();p.points=o3d.utility.Vector3dVector(np.random.default_rng(0).random((30,3)))
            p.colors=o3d.utility.Vector3dVector(np.ones((30,3))*.5)
            self.objects.append(dict(id=str(i),pcd=p,bbox=p.get_axis_aligned_bounding_box(),clip_ft=torch.ones(4)*(1 if i==0 else -1),
                obs_uids=['obs'+str(i)],image_idx=[0],num_detections=1,class_id=[0],class_name='chair',curr_obj_num=i))
        self.cfg=dict(scene_id='smoke',evidence_mode='strict',sim_threshold=.5,downsample_voxel_size=.01,
            dbscan_remove_noise=False,dbscan_eps=.1,dbscan_min_points=2,spatial_sim_type='overlap',device='cpu',make_edges=False)
        self.evidence=EvidenceRecorder(self.f.root/'map',self.cfg,self.cfg,enabled=True)
        self.addCleanup(lambda:self.evidence.close('component_smoke',objects=self.objects,map_edges=SimpleNamespace(edges_by_index={})))
        empty=np.empty((2,0));self.evidence.record_associations(0,self.objects,[],empty,empty,empty,[None]*2)
        for i,o in enumerate(self.objects):self.evidence.record_association_object_version(0,i,None,None,o)
    def run_pass(self):
        def review(a,b,*args):
            self.native_calls+=1
            return self.gate.review(a,b,frame_idx=self.frame,source_frame_id=str(self.frame),stage='native')
        review.on_merged=self.gate.on_merged
        review.supplemental_candidates=lambda objects,kept,native:self.gate.supplemental_pairs(objects,kept,native,frame_idx=self.frame,stage='test')
        review.supplemental_review=lambda a,b,report:self.gate.review(a,b,frame_idx=self.frame,source_frame_id=str(self.frame),stage='test_supplemental',supplemental=True,trigger_containment=report)
        def merged(a,b,ratio,visual,text):
            self.evidence.record_object_merge(frame_idx=self.frame,source_object=a,target_object=b,overlap_ratio=ratio,visual_similarity=visual,text_similarity=text)
        n=len(self.objects)
        self.objects,_=merge_overlap_objects(.8,.8,.8,self.objects,np.ones((n,n))-np.eye(n),.01,False,.1,2,'overlap','cpu',merge_review=review,merge_event_callback=merged)
    def test_missed_native_positive_needs_two_then_real_merge(self):
        self.f.answer='SAME';self.run_pass();self.assertEqual(len(self.objects),2)
        self.assertTrue(self.gate.events[-1]['supplemental_trigger']);self.assertEqual(self.native_calls,0)
        self.frame=2;self.run_pass();self.assertEqual(len(self.objects),1)
        self.assertEqual(self.gate.events[-1]['execution'],'MERGED');self.assertEqual(self.evidence._errors,[])
        events=[json.loads(s) for s in (self.f.root/'map/evidence/mapping_events.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['event_type'],'OBJECT_MERGE')
    def test_negative_stays_separate_even_after_lock(self):
        for frame in [1,2,3]:self.frame=frame;self.run_pass()
        self.assertEqual(len(self.objects),2);self.assertEqual(self.native_calls,0)
        self.assertTrue(self.gate.votes.state(self.gate.votes.key('0','1'))['locked'])
        self.assertEqual(len(self.f.calls),6)
        self.assertFalse(any(e.get('decision_source')=='containment_gt90' for e in self.gate.events))
    def test_quality_failure_does_not_override(self):
        self.f.quality='CONTAMINATED';self.run_pass()
        self.assertEqual(len(self.objects),2);self.assertEqual(self.f.calls,['node_quality','node_quality'])
    def test_native_candidate_keeps_existing_override_and_is_not_resent(self):
        self.objects[1]['clip_ft']=self.objects[0]['clip_ft'].clone();self.run_pass()
        self.assertEqual(len(self.objects),1);self.assertEqual(self.native_calls,1)
        self.assertFalse(self.gate.events[0]['supplemental_trigger'])
        self.assertEqual(self.gate.events[0]['decision_source'],'containment_gt90')
    def test_inactive_and_spatially_disjoint_are_not_nominated(self):
        self.assertEqual(list(self.gate.supplemental_pairs(self.objects,[True,False],set(),frame_idx=1,stage='test')),[])
        self.objects[1]['pcd'].translate((10,0,0))
        self.assertEqual(list(self.gate.supplemental_pairs(self.objects,[True,True],set(),frame_idx=1,stage='test')),[])
    def test_spatial_cache_refreshes_after_merge_generation_changes(self):
        third=dict(self.objects[0],id='2');third['pcd']=o3d.geometry.PointCloud(self.objects[0]['pcd'])
        objects=MapObjectList([*self.objects,third]);kept=[True,True,True]
        candidates=self.gate.supplemental_pairs(objects,kept,set(),frame_idx=1,stage='cache_test')
        self.assertEqual(next(candidates)[:2],(0,1))
        kept[0]=False;objects[1]['pcd'].translate((10,0,0))
        self.gate.votes.generations['1']=1
        self.assertEqual(list(candidates),[])

    def test_exact_90_is_not_nominated(self):
        points=np.array(self.objects[1]['pcd'].points);points[-3:]+=20
        self.objects[1]['pcd'].points=o3d.utility.Vector3dVector(points)
        self.assertEqual(list(self.gate.supplemental_pairs(self.objects,[True,True],set(),frame_idx=1,stage='test')),[])

class ContainmentEquivalence(unittest.TestCase):
    def test_bounded_query_matches_exact_query_including_distance_boundary(self):
        from scipy.spatial import cKDTree
        from conceptgraph.slam.v7_evidence import LiveEvidence
        a=np.random.default_rng(4).random((1000,3));b=a[:950].copy();b+=.0001
        for distance in [.01,.025,0.0]:
            a[0]=[10,10,10];b[0]=[10+distance,10,10]
            source=dict(pcd=SimpleNamespace(points=a));target=dict(pcd=SimpleNamespace(points=b))
            report=LiveEvidence.containment(source,target,distance)
            self.assertEqual(report['a_in_b_count'],np.count_nonzero(cKDTree(b).query(a)[0]<=distance))
            self.assertEqual(report['b_in_a_count'],np.count_nonzero(cKDTree(a).query(b)[0]<=distance))

class PerEndpointModel(unittest.TestCase):
    def test_model_binding_and_timeout_failover_preserve_input(self):
        pool=EndpointPool(['http://one:1','http://two:2'],1,models=['model-a','model-b'])
        payload=dict(model='default',messages=[dict(content='same frozen H')])
        response=httpx.Response(200,json={'done':True},request=httpx.Request('POST','http://two:2/api/chat'))
        with patch('conceptgraph.slam.v7_endpoint_pool.httpx.Client') as client:
            post=client.return_value.__enter__.return_value.post
            post.side_effect=[httpx.ReadTimeout('slow'),response]
            result,attempts,error=pool.request(payload,lambda *args:None)
            self.assertIsNone(error);self.assertEqual([c.kwargs['json']['model'] for c in post.call_args_list],['model-a','model-b'])
            self.assertEqual([a['model'] for a in attempts],['model-a','model-b'])
            self.assertEqual(payload['model'],'default')
            self.assertEqual(post.call_args_list[0].kwargs['json']['messages'],post.call_args_list[1].kwargs['json']['messages'])
    def test_mismatched_model_count_is_error(self):
        with self.assertRaises(ValueError):EndpointPool(['http://one:1','http://two:2'],1,models=['one'])

if __name__=='__main__':unittest.main()
