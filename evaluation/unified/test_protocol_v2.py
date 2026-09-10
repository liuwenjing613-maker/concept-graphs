import unittest,tempfile,json,gzip,pickle
from pathlib import Path
import numpy as np
from evaluate_unified import *
class ProtocolV2(unittest.TestCase):
 def test_unassigned_neighbor(self):
  p,d=project(np.array([[.01,0,0],[.02,0,0]]),np.array([-1,17]),np.zeros((1,3)))
  self.assertEqual(p.tolist(),[17]);self.assertAlmostEqual(d[0],.02)
 def test_all_unassigned(self):
  p,d=project(np.zeros((1,3)),np.array([-1]),np.zeros((2,3)))
  self.assertEqual(p.tolist(),[-1,-1]);self.assertTrue(np.isinf(d).all())
 def test_embedding_scale(self):
  text=np.array([[1.,0],[0,1]]);features=np.array([[.8,.6]])
  np.testing.assert_array_equal(cosine_classes(features,text),cosine_classes(features*7,text*np.array([[1],[10]])))
  with self.assertRaises(ValueError):cosine_classes(features,np.zeros((2,2)))
 def test_best_gt_merge(self):
  ids,n,a,i=contingency(np.array([1000]*200+[1001]*200),np.zeros(400,int),1)
  m,_=instance_iou(ids,n,a,i,[1]);one,_=instance_iou_one_to_one(ids,n,a,i,[1])
  self.assertEqual(m['instance_mIoU'],.5);self.assertEqual(one['instance_mIoU'],.25)
  d=detection_diagnostic(ids,n,a,i,[1]);self.assertEqual((d['diagnostic_TP50'],d['diagnostic_FN50']),(1,1))
 def test_ten_point_and_small_prediction(self):
  ids,n,a,i=contingency(np.array([1000]*9+[1001]*10),np.array([-1]*9+[0]*5+[-1]*5),1)
  m,_=instance_iou(ids,n,a,i,[1]);self.assertEqual(m['gt_instances'],1);self.assertEqual(m['instance_mIoU'],.5)
 def test_adapter_parity_and_manifest_cache(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);r=p/'ref'/'room0';r.mkdir(parents=True)
   xyz=np.column_stack([np.arange(200)*.1,np.zeros((200,2))]);np.savez(r/'reference.npz',xyz=xyz,semantic=np.ones(200,int),instance=np.full(200,1000))
   rm={'reference_sha256':sha(r/'reference.npz'),'semantic_classes':['chair','table'],'instance_classes':['chair','table']};dump(r/'manifest.json',rm)
   np.save(p/'text.npy',np.eye(2));cfg={'reference_root':str(p/'ref'),'clip_text':str(p/'text.npy')}
   with gzip.open(p/'cg.pkl.gz','wb') as f:pickle.dump({'objects':[{'pcd_np':xyz,'clip_ft':np.array([1,0])}]},f)
   np.savez(p/'ovi.npz',xyz=xyz,instance=np.zeros(200,int),classes=np.array([1]))
   a=evaluate({'method':'cg','scene':'room0','kind':'cg','map':str(p/'cg.pkl.gz'),'projection_cache':'deliberately_nonexistent'},cfg,p/'out')
   b=evaluate({'method':'ovi','scene':'room0','kind':'npz','map':str(p/'ovi.npz')},cfg,p/'out')
   self.assertEqual(a['metrics'],b['metrics'])
   aggregate([a,b],cfg,p/'out')
   rm['semantic_classes']=['table','chair'];dump(r/'manifest.json',rm)
   with self.assertRaises(RuntimeError):evaluate(a['source'],cfg,p/'out')
if __name__=='__main__':unittest.main()
