"""Real cloud merge + strict evidence smoke for the frame-8 crash; no VLM calls."""
import json,tempfile,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import torch
import open3d as o3d
from conceptgraph.utils.evidence import EvidenceRecorder
from conceptgraph.slam.v7_runtime import V7Runtime
from conceptgraph.slam.v7_merge import V7Votes
from conceptgraph.slam.human_instance_merge import object_state,state_key
from conceptgraph.slam.slam_classes import MapObjectList,DetectionList
from conceptgraph.slam.mapping import merge_obj_matches

class MergeEvidenceSmoke(unittest.TestCase):
    def test_real_group_merge_then_observation_versions(self):
        with tempfile.TemporaryDirectory(prefix='v7_merge_evidence_') as temp:
            root=Path(temp)
            def obj(i):
                cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(np.random.default_rng(i).random((30,3)))
                cloud.colors=o3d.utility.Vector3dVector(np.ones((30,3))*.5)
                return dict(id=str(uuid.uuid4()),obs_uids=['obs'+str(i)],pcd=cloud,bbox=cloud.get_axis_aligned_bounding_box(),
                    clip_ft=torch.ones(4)/2,class_id=[0],class_name='chair',num_detections=1,image_idx=[0],curr_obj_num=i)
            cfg=dict(scene_id='smoke',evidence_mode='strict',sim_threshold=.5,downsample_voxel_size=.01,
                dbscan_remove_noise=False,dbscan_eps=.1,dbscan_min_points=2,spatial_sim_type='overlap',device='cpu',make_edges=False)
            evidence=EvidenceRecorder(root,cfg,cfg,enabled=True)
            objects=MapObjectList([obj(i) for i in range(3)]);original_ids=[o['id'] for o in objects]
            empty=np.empty((3,0));evidence.record_associations(0,objects,[],empty,empty,empty,[None]*3)
            for i,o in enumerate(objects):evidence.record_association_object_version(0,i,None,None,o)
            runtime=V7Runtime.__new__(V7Runtime);runtime.root=root/'gate';(runtime.root/'events/parent').mkdir(parents=True)
            runtime.rows={};runtime.publish=lambda:None
            votes=V7Votes();key=votes.key(objects[0]['id'],objects[1]['id'])
            votes.record(key,1,'one','MERGE');votes.record(key,2,'two','MERGE')
            gate=SimpleNamespace(votes=votes,mark_executed=Mock(),_summary=Mock())
            runtime.owner=SimpleNamespace(events=[],_support_history={},_instance_merge_gate=gate)
            members=list(objects[:2]);runtime.forced_groups=[dict(parent_event='parent',objects=members,keys=[key],
                state=state_key([object_state(o) for o in members]),h_snapshot_uid='H')]
            detections=DetectionList([obj(i) for i in range(10,15)])
            scores=np.tile(np.array([.7,.9,.3]),(5,1));snapshot=evidence.association_similarity_snapshot(objects)
            frozen_versions=list(snapshot['object_version_uids'])
            objects,matches=runtime.flush_groups(objects,[1,0,2,None,-1],cfg,evidence,2,None)
            self.assertEqual(matches,[0,0,1,None,-1])
            self.assertEqual(evidence._object_versions[original_ids[0]],2)
            evidence.record_associations(2,detections,objects,scores,scores,scores,matches,similarity_snapshot=snapshot)
            objects=merge_obj_matches(detections,objects,matches,.01,False,.1,2,'overlap','cpu',
                object_update_callback=lambda i,idx,before,after:evidence.record_association_object_version(2,i,idx,before,after))
            rows=[json.loads(line) for line in (root/'evidence/associations.jsonl').read_text().splitlines()][-5:]
            self.assertEqual([r['target_object_uid'] for r in rows],[original_ids[0],original_ids[0],original_ids[2],detections[3]['id'],None])
            self.assertEqual([evidence._object_versions[u] for u in [original_ids[0],original_ids[2],detections[3]['id']]],[4,2,1])
            for r in rows:
                self.assertTrue(r['similarity_evidence_valid']);self.assertEqual(r['object_uids_before'],original_ids)
                self.assertEqual(r['candidate_object_version_uids'],frozen_versions)
                self.assertEqual(r['top_candidates'][0]['object_uid'],original_ids[1])
            events=[json.loads(line) for line in (root/'evidence/mapping_events.jsonl').read_text().splitlines()]
            types=[e['event_type'] for e in events if e['frame_uid'].endswith('f000002')]
            self.assertEqual(types[0],'OBJECT_MERGE');self.assertEqual(types[1:],["OBS_ASSOCIATE"]*3+['OBJECT_CREATE','OBS_DISCARD'])
            self.assertEqual(evidence._errors,[])
            # This component fixture has no RGB/depth frame archive or final scene map.
            evidence.close('component_smoke',objects=objects,map_edges=SimpleNamespace(edges_by_index={}))

if __name__=='__main__':unittest.main()
