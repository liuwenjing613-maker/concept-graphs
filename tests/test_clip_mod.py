import importlib.util
import unittest
from pathlib import Path
import numpy as np
import supervision as sv
import torch
from conceptgraph.utils.model_utils import compute_clip_features_batched

class PixelEncoder:
    def __init__(self):self.calls=0
    def encode_image(self,batch):
        self.calls+=1
        return torch.stack((batch.mean((1,2,3)),batch.square().mean((1,2,3)),torch.ones(len(batch))),dim=1)

class ClipWeightTests(unittest.TestCase):
    def test_all_weights_match_original_and_skip_unused_encoding(self):
        source=Path(__file__).resolve().parents[2]/'v7_CLIP/conceptgraph/utils/model_utils.py'
        spec=importlib.util.spec_from_file_location('original_clip',source)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        image=np.arange(6*6*3,dtype=np.uint8).reshape(6,6,3)
        mask=np.zeros((1,6,6),dtype=bool);mask[:,2:4,2:4]=True
        det=sv.Detections(xyxy=np.array([[1.,1.,5.,5.]],dtype=np.float32),mask=mask,class_id=np.array([0]))
        kwargs=dict(image=image,detections=det,clip_preprocess=lambda im:torch.from_numpy(np.array(im).copy()).float().permute(2,0,1)/255,clip_tokenizer=lambda x:torch.zeros((len(x),1)),classes=['object'],device='cpu',bbox_padding=0)
        for alpha in [0.,.25,.5,.75,1.]:
            with self.subTest(alpha=alpha):
                a,b=PixelEncoder(),PixelEncoder()
                old_result=old.compute_clip_features_batched(clip_model=a,masked_weight=alpha,**kwargs)
                result=compute_clip_features_batched(clip_model=b,masked_weight=alpha,**kwargs)
                np.testing.assert_allclose(result[1],old_result[1],rtol=1e-6,atol=1e-7)
                self.assertEqual(b.calls,1 if alpha in (0.,1.) else 2)
                np.testing.assert_array_equal(np.asarray(result[0][0]),np.asarray(old_result[0][0]))
                np.testing.assert_allclose(np.linalg.norm(result[1],axis=1),1.,atol=1e-6)

if __name__=='__main__':unittest.main()
