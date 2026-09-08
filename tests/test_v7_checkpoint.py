import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import open3d as o3d
from conceptgraph.slam.v7_checkpoint import Checkpoints
from conceptgraph.slam.association_gate import BlockingAssociationGate
from conceptgraph.slam.slam_classes import MapObjectList,MapEdgeMapping
from conceptgraph.utils.evidence import EvidenceRecorder

class CheckpointSmoke(unittest.TestCase):
    def test_geometry_votes_evidence_rollback_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',{'V7_FALLBACK':'auto'}):
            root=Path(temp);cfg=dict(scene_id='smoke',evidence_mode='strict',sim_threshold=1.2,downsample_voxel_size=.01,
                device='cpu',make_edges=False,association_gate=dict(mode='vlm',mask_change_enabled=False))
            ev=EvidenceRecorder(root,cfg,cfg,enabled=True)
            gate=BlockingAssociationGate(cfg=cfg,output_dir=root/'blocking_association_gate')
            cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]));cloud.colors=o3d.utility.Vector3dVector(np.ones((4,3)))
            objects=MapObjectList([dict(id='A',pcd=cloud,bbox=cloud.get_axis_aligned_bounding_box(),clip_ft=torch.ones(4),obs_uids=['o1'])]);edges=MapEdgeMapping(objects)
            gate.object_merge_reviewer(frame_idx=0,source_frame_id='f0',stage='periodic');votes=gate._instance_merge_gate.votes;k=votes.key('A','B');votes.record(k,0,'event0','MERGE')
            ev._event_counter=7;ev._current_object_versions={'A':'version7'};ev._frames={0};gate._support_history={'A':[{'frame':0}]}
            gate.vlm_runtime.projection_frames={0:{'pose_c2w':np.eye(4)}}
            immutable=root/'evidence/immutable.npz';np.savez(immutable,x=np.arange(5))
            history=root/'evidence/test.jsonl';history.write_text('old\n')
            cp=Checkpoints(root,cfg);cp.save(1,objects,edges,ev,gate,SimpleNamespace(total=22),{'counter':1})
            with self.assertRaises(RuntimeError):Checkpoints(root,cfg)
            history.write_text('old\nfuture\n');extra=root/'blocking_association_gate/events/interrupted';extra.mkdir(parents=True);(extra/'request.json').write_text('{}')
            ev._event_counter=99;votes.record(k,1,'future','MERGE')
            cp.load();cp.rollback()
            tracker=SimpleNamespace();a,e,restored,g,local,next_frame=cp.restore(None,tracker)
            self.assertEqual(next_frame,1);self.assertEqual(restored._event_counter,7);self.assertEqual(restored._current_object_versions,{'A':'version7'})
            self.assertEqual(g._instance_merge_gate.votes.state(k)['merge_streak'],1)
            self.assertFalse(g._instance_merge_gate.votes.state(k)['awaiting_execution'])
            np.testing.assert_array_equal(np.asarray(a[0]['pcd'].points),np.asarray(cloud.points))
            np.testing.assert_array_equal(a[0]['bbox'].min_bound,objects[0]['bbox'].min_bound)
            self.assertIs(e.objects,a);self.assertEqual(tracker.total,22)
            self.assertEqual(history.read_text(),'old\n');self.assertFalse(extra.exists());self.assertTrue(list((cp.directory/'interrupted').rglob('request.json')))
            self.assertEqual(g._instance_merge_gate.votes.record(k,1,'replayed','MERGE')[0],True)
            # An incomplete write cannot invalidate the published checkpoint.
            (cp.directory/'frame_broken.pkl.tmp').write_bytes(b'incomplete');cp.load()
            cp.cfg['scene_id']='other'
            with self.assertRaises(ValueError):cp.load()
            cp.cfg['scene_id']='smoke';pointer=json.loads((cp.directory/'latest.json').read_text());blob=cp.directory/pointer['file'];blob.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):cp.load()
            for recorder in [ev,restored]:
                recorder._closed=True
                for f in recorder._files.values():f.close()
            cp.lock.close()

if __name__=='__main__':unittest.main()
