import unittest
import numpy as np
from evaluate_unified import *
class Tests(unittest.TestCase):
 def metrics(self,gt,pred,classes):
  ids,n,area,inter=contingency(np.array(gt),np.array(pred),len(classes))
  m,_=instance_iou(ids,n,area,inter,[1,2]);ca=build_matches('s',ids,n,area,inter,np.array(classes),[1,2],['a','b'],True);sa=build_matches('s',ids,n,area,inter,np.array(classes),[1,2],['a','b'],False)
  return m,run_ap({'s':ca},['object'])[0],run_ap({'s':sa},['a','b'])[0]
 def test_perfect(self):
  m,ca,sa=self.metrics([1000]*200+[2000]*200,[0]*200+[1]*200,[1,2]);self.assertEqual(m['instance_mIoU'],1);self.assertEqual(ca,[1,1,1]);self.assertEqual(sa,[1,1,1])
 def test_missing(self):
  m,ca,sa=self.metrics([1000]*200+[2000]*200,[0]*200+[-1]*200,[1]);self.assertEqual(m['instance_mIoU'],.5);self.assertEqual(ca,[.5,.5,.5])
 def test_empty(self):
  m,ca,sa=self.metrics([1000]*200,[-1]*200,[]);self.assertEqual(m['instance_mIoU'],0);self.assertEqual(ca,[0,0,0])
 def test_wrong_class(self):
  m,ca,sa=self.metrics([1000]*200,[0]*200,[2]);self.assertEqual(ca,[1,1,1]);self.assertEqual(sa,[0,0,0])
 def test_merge_and_exact_boundary(self):
  m,ca,_=self.metrics([1000]*200+[1001]*200,[0]*400,[1]);self.assertEqual(m['instance_mIoU'],.25);self.assertEqual(ca[1:], [0,0])
 def test_split(self):
  m,ca,_=self.metrics([1000]*400,[0]*200+[1]*200,[1,1]);self.assertEqual(m['instance_mIoU'],.5);self.assertEqual(ca[1:], [0,0]);self.assertLess(ca[0],1)
 def test_projection_boundary(self):
  pred,dist=project(np.array([[0.,0,0]]),np.array([0]),np.array([[.049,0,0],[.05,0,0],[.051,0,0]]),1);np.testing.assert_array_equal(pred,[0,-1,-1])
 def test_absent_class_false_positive(self):
  conf=np.array([[0,0,0],[0,100,100],[0,0,0]]);m,_=semantic(conf,['a','b']);self.assertEqual(m['semantic_mIoU'],.25);self.assertEqual(m['semantic_mIoU_GT_present_diagnostic'],.5)
 def test_unobserved_false_negative(self):
  conf=np.array([[0,0],[100,100]]);m,_=semantic(conf,['a']);self.assertEqual(m['semantic_mIoU'],.5)
 def test_small_masks(self):
  m,ca,_=self.metrics([1000]*200,[0]*99+[-1]*101,[1]);self.assertEqual(m['instance_mIoU'],0);self.assertEqual(ca,[0,0,0])
if __name__=='__main__':unittest.main(verbosity=2)
