"""Renderer parity smoke: one observation pair and one selected merge; zero API calls."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib
import numpy as np
from PIL import Image
import yaml
from conceptgraph.slam.v7_evidence import LiveEvidence
from conceptgraph.slam import v7_render as r
from conceptgraph.slam.vlm_runtime import save_json,sha
BASE=Path('/home/chenkejun/beauty/ali-my-v6projection')
EXP=Path('/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/v5_human2000')
OUT=Path('/home/chenkejun/beauty/v7_split_smoke_20260907/renderer');OUT.mkdir(exist_ok=True)
cfg=json.loads((EXP/'config_params.json').read_text());scale=yaml.safe_load(Path(cfg['dataset_config']).read_text())['camera_params']['png_depth_scale']
frames={}
for line in (EXP/'evidence/frames.jsonl').open():
    f=json.loads(line);frames[f['frame_idx']]=dict(frame_idx=f['frame_idx'],rgb_path=str((EXP/f['rgb_ref']['path']).resolve()),
        depth_path=str((EXP/f['depth_ref']['path']).resolve()),pose_c2w=f['pose'],intrinsics=f['intrinsics'],
        image_hw=[cfg['image_height'],cfg['image_width']],depth_scale=scale)
reader=LiveEvidence(SimpleNamespace(root=EXP/'blocking_association_gate',projection_frames=frames));reader.sync()
pair=BASE/'event_chain_v5_focused_full_20260907/events/f000373_d009_association/candidates/B'
m=json.loads((pair/'input_manifest.json').read_text())
cur=reader.visual(m['current_observation_uid']);hist=reader.visual(m['history']['uid'])
with np.load(pair/Path(m['history']['projection_path']).name) as z:proj={k:z[k] for k in z.files}
proj['reliable_uv']=proj['uv'][proj['status']==1]
r.focused_card(cur,'B',hist,proj,OUT/'pair.jpg')
r.cards.audit_card(cur,OUT/'observation_quality.jpg')
quality_source=pair.parent.parent/'quality/input.jpg'
observation_equal=np.array_equal(np.asarray(Image.open(OUT/'observation_quality.jpg')),np.asarray(Image.open(quality_source)))
pair_equal=np.array_equal(np.asarray(Image.open(OUT/'pair.jpg')),np.asarray(Image.open(pair/'input.jpg')))
ptr=json.loads(Path('/home/chenkejun/beauty/v5_merge_test_selection_latest.json').read_text())
assert sha(ptr['active_manifest'])==ptr['sha256'];selection=json.loads(Path(ptr['active_manifest']).read_text())
cid='m00002_f000004_periodic';assert cid in selection['selected_ids']
old=BASE/'merge_full_events_v1_20260907/events'/cid;m=json.loads((old/'input_manifest.json').read_text())
directory=OUT/cid;directory.mkdir(exist_ok=True)
raw=EXP/'blocking_association_gate/human_instance_merge/events'/cid
with np.load(raw/'live_pair.npz') as z:clouds={a:z['object_'+a].copy() for a in 'AB'}
objects={}
for a in 'AB':
    s=m['objects'][a];objects[a]=dict(id=s['object_uid'],obs_uids=s['member_observation_uids'],
        image_idx=s['image_indices'],num_detections=s['num_detections'],pcd=SimpleNamespace(points=clouds[a]),clip_ft=np.zeros(3))
binding=reader.merge(directory,objects['A'],objects['B'],m['source_timeline']['h_frame'])
merge_equal=np.array_equal(np.asarray(Image.open(directory/'merge.png')),
    np.asarray(Image.open(BASE/'merge_rgb_zoom_maskcontent_v2_20260907/events'/cid/'input.png')))
quality_equal={a:np.array_equal(np.asarray(Image.open(directory/f'quality_{a}.jpg')),np.asarray(Image.open(old/f'quality_{a}.jpg'))) for a in 'AB'}
result=dict(observation_quality_pixel_equal=observation_equal,pair_pixel_equal=pair_equal,merge_pixel_equal=merge_equal,node_quality_pixel_equal=quality_equal,api_calls=0,selected_merge=cid,
    selection_pointer_sha256=ptr['sha256'],outputs=str(OUT))
save_json(OUT/'validation.json',result);print(json.dumps(result,ensure_ascii=False))
assert observation_equal and pair_equal and merge_equal and all(quality_equal.values())
