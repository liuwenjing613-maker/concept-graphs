import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.slam import utils
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.two_road import export_readouts
from conceptgraph.slam.v7_image_detection import validate_two_road_features
from conceptgraph.utils.general_utils import save_detection_results, load_saved_detections

class Encoder:
    visual = SimpleNamespace(output_dim=1024)
    def __init__(self): self.calls=0
    def encode_image(self, x):
        self.calls+=1
        y=x.flatten(1)
        return torch.cat((y,y),1)[:,:1024]+.001

def inputs():
    im=np.random.default_rng(4).integers(0,255,(40,40,3),dtype=np.uint8)
    masks=np.zeros((2,40,40),bool);masks[0,5:20,5:20]=True;masks[1,15:35,15:35]=True
    det=SimpleNamespace(xyxy=np.array([[5.,5.,20.,20.],[15.,15.,35.,35.]]),mask=masks,class_id=np.array([0,1]))
    pre=lambda im:torch.from_numpy(np.asarray(im.resize((16,16))).copy()).permute(2,0,1).float()/255
    tok=lambda s:torch.zeros((len(s),2),dtype=torch.long)
    return im,det,pre,tok

def test_encoder_parity_and_no_extra_encoding():
    spec=importlib.util.spec_from_file_location('original_clip',Path(__file__).resolve().parents[2]/'v7_CLIP/conceptgraph/utils/model_utils.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    im,det,pre,tok=inputs()
    for weight in [0,.5,1]:
        model=Encoder();a=old.compute_clip_features_batched(im,det,model,pre,tok,['a','b'],'cpu',masked_weight=weight)
        model=Encoder();b=compute_clip_features_batched(im,det,model,pre,tok,['a','b'],'cpu',masked_weight=weight,return_bbox_features=True)
        assert np.array_equal(a[1],b[1]);assert model.calls==(1 if weight==0 else 2)
        bbox=old.compute_clip_features_batched(im,det,Encoder(),pre,tok,['a','b'],'cpu',masked_weight=0)[1]
        assert np.array_equal(b[3],bbox)
    det.xyxy=np.empty((0,4));r=compute_clip_features_batched(im,det,Encoder(),pre,tok,[],'cpu',return_bbox_features=True)
    assert r[1].shape==r[3].shape==(0,1024)

def obj(seed,n):
    rng=np.random.default_rng(seed)
    pcd=o3d.geometry.PointCloud();pcd.points=o3d.utility.Vector3dVector(rng.normal(size=(20,3)));pcd.colors=o3d.utility.Vector3dVector(rng.uniform(size=(20,3)))
    return dict(id=str(seed),num_detections=n,pcd=pcd,bbox=pcd.get_oriented_bounding_box(),n_points=20,
        clip_ft=F.normalize(torch.tensor(rng.normal(size=1024),dtype=torch.float32),dim=0),
        clip_semantic_ft=F.normalize(torch.tensor(rng.normal(size=1024),dtype=torch.float32),dim=0),obs_uids=[str(seed)])

def test_merge_roundtrip_export(tmp_path,monkeypatch):
    monkeypatch.setattr(utils.tracker,'track_merge',lambda *args:None)
    a,b=obj(1,3),obj(2,7);old_a=copy.deepcopy(a);old_b=copy.deepcopy(b)
    expected=F.normalize(a['clip_semantic_ft']*3+b['clip_semantic_ft']*7,dim=0)
    kwargs=dict(downsample_voxel_size=.01,dbscan_remove_noise=False,dbscan_eps=.1,dbscan_min_points=2,spatial_sim_type='overlap',device='cpu',run_dbscan=False)
    c=utils.merge_obj2_into_obj1(a,b,**kwargs)
    del old_a['clip_semantic_ft'];del old_b['clip_semantic_ft'];d=utils.merge_obj2_into_obj1(old_a,old_b,**kwargs)
    assert torch.equal(c['clip_ft'],d['clip_ft'])
    assert torch.allclose(c['clip_semantic_ft'],expected,atol=1e-7)
    assert np.array_equal(np.asarray(c['pcd'].points),np.asarray(d['pcd'].points))
    maps=MapObjectList([c]);saved=maps.to_serializable();loaded=MapObjectList();loaded.load_serializable(saved)
    assert torch.equal(loaded[0]['clip_semantic_ft'],c['clip_semantic_ft'])
    export_readouts(loaded,tmp_path)
    with np.load(tmp_path/'clip_readouts.npz') as data:
        assert np.array_equal(data['semantic_bbox'][0],c['clip_semantic_ft'].numpy())
    import pytest
    before=copy.deepcopy(c)
    with pytest.raises(ValueError,match='single-road'):utils.merge_obj2_into_obj1(c,d,**kwargs)
    assert torch.equal(c['clip_ft'],before['clip_ft'])

def test_cache_and_filter_alignment(tmp_path):
    import pytest
    im,det,pre,tok=inputs();_,f,_,b=compute_clip_features_batched(im,det,Encoder(),pre,tok,['a','b'],'cpu',return_bbox_features=True)
    g=dict(image_feats=f,bbox_feats=b,mask=det.mask)
    validate_two_road_features(g);save_detection_results(tmp_path,g);r=load_saved_detections(tmp_path)
    assert np.array_equal(r['bbox_feats'],b)
    del r['bbox_feats']
    with pytest.raises(ValueError,match='missing'):validate_two_road_features(r)
    g.update(xyxy=det.xyxy,classes=['a','b'],class_id=det.class_id,confidence=np.array([.1,.9]),labels=[],edges=[],text_feats=[],captions=[],detection_class_labels=['a 0','b 1'])
    result=utils.filter_gobs(g,im,skip_bg=False,BG_CLASSES=[],mask_area_threshold=1,max_bbox_area_ratio=1,mask_conf_threshold=.5)
    assert np.array_equal(result['bbox_feats'],b[1:])
    assert np.array_equal(result['image_feats'],f[1:])


def test_semantic_evaluation_export(tmp_path):
    import gzip,pickle,hashlib
    from scripts.export_two_road_semantic_map import export
    source=tmp_path/'map.pkl.gz';dest=tmp_path/'bbox.pkl.gz'
    values=MapObjectList([obj(4,2)]).to_serializable()
    with gzip.open(source,'wb') as stream:pickle.dump({'objects':values,'edges':[]},stream)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    assert export(source,dest)==1
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    with gzip.open(dest,'rb') as stream: result=pickle.load(stream)
    assert np.array_equal(result['objects'][0]['clip_ft'],values[0]['clip_semantic_ft'])
    for key in ['pcd_np','bbox_np','pcd_color_np']:
        assert np.array_equal(result['objects'][0][key],values[0][key])
    assert result['objects'][0]['obs_uids']==values[0]['obs_uids']
    import pytest
    with pytest.raises(FileExistsError):export(source,dest)
