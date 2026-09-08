"""Zero-API performance/parity replay of two frozen events from the stopped v7 run."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,time,hashlib
import numpy as np
from PIL import Image
import yaml
from conceptgraph.slam.v7_evidence import LiveEvidence
from conceptgraph.slam.vlm_runtime import save_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    exp=Path('/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/v7_auto_20260907_235039_1894703')
    cfg=json.loads((exp/'config_params.json').read_text());scale=yaml.safe_load(Path(cfg['dataset_config']).read_text())['camera_params']['png_depth_scale']
    frames={}
    for line in (exp/'evidence/frames.jsonl').open():
        f=json.loads(line);frames[f['frame_idx']]=dict(frame_idx=f['frame_idx'],rgb_path=str((exp/f['rgb_ref']['path']).resolve()),
            depth_path=str((exp/f['depth_ref']['path']).resolve()),pose_c2w=f['pose'],intrinsics=f['intrinsics'],image_hw=[cfg['image_height'],cfg['image_width']],depth_scale=scale)
    reader=LiveEvidence(SimpleNamespace(root=exp/'blocking_association_gate',projection_frames=frames));t=time.perf_counter();reader.sync();sync_seconds=time.perf_counter()-t
    ref=next(o['image_feat_ref'] for o in reader.observations.values() if o.get('image_feat_ref'))
    t=time.perf_counter()
    for _ in range(3):reader.array(ref)
    feature_seconds=(time.perf_counter()-t)/3
    results=[]
    for eid in ['m00434_f000309_periodic','m00440_f000314_periodic']:
        old=exp/'blocking_association_gate/vlm_instance_merge/events'/eid;m=json.loads((old/'input_manifest.json').read_text());event=json.loads((old/'decision.json').read_text())
        with np.load(old/'live_pair.npz') as z:clouds={a:z['object_'+a].copy() for a in 'AB'}
        objects={}
        for a in 'AB':
            s=m['objects'][a];objects[a]=dict(id=s['object_uid'],obs_uids=[r['uid'] for r in m['histories'][a]['all']],image_idx=s['image_indices'],
                num_detections=s['num_detections'],pcd=SimpleNamespace(points=clouds[a]),clip_ft=np.zeros(3))
        directory=out/eid;directory.mkdir(exist_ok=True)
        t=time.perf_counter();selection=reader.prepare_merge(objects['A'],objects['B'],m['source_timeline']['h_frame']);selection_seconds=time.perf_counter()-t
        keys=['histories','selected_anchor','selected_identity_histories','other_appearance_history']
        save_json(directory/'selection.json',selection)
        parity={k:selection[k]==m[k] for k in keys}
        if not all(parity.values()):
            differences=[]
            for a in 'AB':
                for kind in ['all','selected']:
                    for new,prior in zip(selection['histories'][a][kind],m['histories'][a][kind]):
                        for key in new:
                            if new[key]!=prior.get(key):differences.append((a,kind,new['uid'],key,new[key],prior.get(key)))
            print('HISTORY_DIFFERENCES',differences[:10],flush=True)
        assert all(parity.values()),parity
        # A locked decision ends here, before render_merge. Render only to verify exact pixel parity.
        assert not list(directory.glob('*.jpg')) and not list(directory.glob('*.png'))
        t=time.perf_counter();binding=reader.render_merge(directory,objects['A'],objects['B'],m['source_timeline']['h_frame'],selection);render_seconds=time.perf_counter()-t
        images={name:np.array_equal(np.asarray(Image.open(directory/name)),np.asarray(Image.open(old/name))) for name in ['quality_A.jpg','quality_B.jpg','merge.png']};assert all(images.values()),images
        from datetime import datetime
        stamp=lambda x:datetime.fromisoformat(x.replace('Z','+00:00')).timestamp()
        old_seconds=stamp(event['timeline']['c_utc'])-stamp(event['timeline']['h_utc'])
        row=dict(event=eid,histories={a:len(m['histories'][a]['all']) for a in 'AB'},old_locked_event_seconds=old_seconds,
            new_history_check_seconds=selection_seconds,new_render_for_parity_only_seconds=render_seconds,selection_equal=parity,images_equal=images,
            isolated_history_check_speedup=old_seconds/selection_seconds)
        results.append(row);print(json.dumps(row),flush=True)
    result=dict(api_calls=0,full_scene_run=False,initial_sync_seconds=sync_seconds,feature_read_mean_seconds=feature_seconds,events=results,
        limitation='Replayed frozen events only; speed ratios are not a full-scene runtime guarantee.')
    save_json(out/'validation.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
