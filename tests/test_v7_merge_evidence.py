"""Real point-cloud fusion precedes baseline proposal verification; strict evidence."""
import json,tempfile,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch,open3d as o3d
from conceptgraph.utils.evidence import EvidenceRecorder
from conceptgraph.slam.slam_classes import MapObjectList,DetectionList
from conceptgraph.slam.mapping import merge_obj_matches
from conceptgraph.slam.utils import merge_overlap_objects
import test_v7_no90_new as fixtures

class MergeEvidenceSmoke(unittest.TestCase):
    def test_real_observation_fused_before_baseline_merge_and_recorded(self):
        f=fixtures.Fixture();f.setUp();self.addCleanup(f.tearDown)
        def obj(i):
            cloud=o3d.geometry.PointCloud()
            cloud.points=o3d.utility.Vector3dVector(np.random.default_rng(i).random((30,3)))
            cloud.colors=o3d.utility.Vector3dVector(np.ones((30,3))*.5)
            return dict(id=str(uuid.uuid4()),obs_uids=['obs'+str(i)],pcd=cloud,
                bbox=cloud.get_axis_aligned_bounding_box(),clip_ft=torch.ones(4)/2,
                class_id=[0],class_name='chair',num_detections=1,image_idx=[0],curr_obj_num=i)
        a,b,det=obj(0),obj(1),obj(2);current=det['obs_uids'][0]
        cfg=dict(scene_id='smoke',evidence_mode='strict',sim_threshold=.5,downsample_voxel_size=.01,
            dbscan_remove_noise=False,dbscan_eps=.1,dbscan_min_points=2,spatial_sim_type='overlap',
            device='cpu',make_edges=False)
        evidence=EvidenceRecorder(f.root/'mapping',cfg,cfg,enabled=True)
        objects=MapObjectList([a,b]);detections=DetectionList([det])
        empty=np.empty((2,0));evidence.record_associations(0,objects,[],empty,empty,empty,[None,None])
        for i,o in enumerate(objects):evidence.record_association_object_version(0,i,None,None,o)
        scores=np.array([[.7,.9]])
        evidence.record_associations(2,detections,objects,scores,scores,scores,[1])
        objects=merge_obj_matches(detections,objects,[1],.01,False,.1,2,'overlap','cpu',
            object_update_callback=lambda i,idx,before,after:evidence.record_association_object_version(2,i,idx,before,after))
        self.assertIn(current,objects[1]['obs_uids']);self.assertNotIn(current,objects[0]['obs_uids'])
        count=[]
        def review(source,target,overlap,visual,text):
            count.append(1);self.assertIn(current,target['obs_uids'])
            return f.g.review(source,target,frame_idx=2,source_frame_id='10',stage='periodic',
                              overlap=overlap,visual=visual,text=text)
        review.on_merged=f.g.on_merged
        # Original baseline rejects below-threshold proposal without invoking VLM.
        merge_overlap_objects(.99,.8,.8,objects,np.array([[0,.98],[0,0]]),.01,False,.1,2,'overlap','cpu',merge_review=review)
        self.assertFalse(count)
        result=merge_overlap_objects(.9,.8,.8,objects,np.array([[0,.98],[0,0]]),.01,False,.1,2,'overlap','cpu',
            merge_review=review,merge_event_callback=lambda source,target,o,v,t:evidence.record_object_merge(
                frame_idx=2,source_object=source,target_object=target,overlap_ratio=o,
                visual_similarity=v,text_similarity=t))
        result=result[0] if isinstance(result,tuple) else result
        self.assertEqual(len(result),1);self.assertIn(current,result[0]['obs_uids'])
        self.assertEqual(len(count),1);self.assertEqual(f.g.events[-1]['execution'],'MERGED')
        self.assertEqual(f.g.events[-1]['decision_source'],'VLM_VERIFIER')
        self.assertEqual(evidence._errors,[])
        rows=[json.loads(l) for l in (f.root/'mapping/evidence/mapping_events.jsonl').read_text().splitlines()]
        kinds=[r['event_type'] for r in rows if r['frame_uid'].endswith('f000002')]
        self.assertEqual(kinds,['OBS_ASSOCIATE','OBJECT_MERGE'])
        evidence.close('component_smoke',objects=result,map_edges=SimpleNamespace(edges_by_index={}))
