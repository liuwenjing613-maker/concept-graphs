"""Read only this run's append-only evidence; freeze inputs before any model call."""
import hashlib,json
from pathlib import Path
from functools import lru_cache
import numpy as np
from conceptgraph.slam.v7_errors import EvidenceInvariantError
from conceptgraph.slam.history_projection import load_history,project_points,sha
from conceptgraph.slam.staged_identity_cards import audit_card,unproject
from conceptgraph.slam.human_instance_merge import object_state,state_key
from conceptgraph.slam.vlm_runtime import save_json
from conceptgraph.slam import v7_render as render

class LiveEvidence:
    def __init__(self,runtime):
        self.runtime=runtime
        self.exp=runtime.root.parent
        self.observations={}
        self.offset=0

    def sync(self):
        path=self.exp/'evidence/observations.jsonl'
        with path.open() as f:
            f.seek(self.offset)
            for line in iter(f.readline, ""):
                row=json.loads(line)
                self.observations[row['obs_uid']]=row
            self.offset=f.tell()

    def ref(self,ref):
        p=(self.exp/ref['path']).resolve()
        if not p.is_relative_to(self.exp.resolve()):
            # Frozen per-frame detector features may live in the shared detection cache.
            # Only refs explicitly recorded for observations in THIS online run may be read.
            allowed={(str((self.exp/o['image_feat_ref']['path']).resolve()),o['image_feat_ref']['sha256'])
                     for o in self.observations.values() if o.get('image_feat_ref')}
            if (str(p),ref['sha256']) not in allowed:
                raise EvidenceInvariantError('external evidence is not a registered detector feature')
        if sha(p)!=ref['sha256']:raise EvidenceInvariantError('evidence hash mismatch')
        return p

    def array(self,ref):
        with np.load(self.ref(ref),allow_pickle=False) as z:
            v=z[ref['key']]
            return (v[ref['index']] if ref.get('index') is not None else v).copy()

    @lru_cache(maxsize=12)
    def visual(self,uid):
        o=self.observations[uid]
        fidx=int(o['frame_uid'].rsplit('_f',1)[1])
        f=self.runtime.projection_frames[fidx]
        rgb,mask,depth=load_history(dict(f,mask=self.array(o['processed_mask_ref']).astype(bool)))
        return dict(uid=uid,obs_uid=uid,frame_idx=fidx,rgb=rgb,mask=mask,depth=depth,
                    pose=f['pose_c2w'],K=f['intrinsics'],bbox=o['bbox_2d'])

    def members(self,obj,h):
        uids=list(map(str,obj.get('obs_uids',[])))
        if not uids or len(uids)!=len(set(uids)):raise ValueError('missing/duplicate members')
        for u in uids:
            o=self.observations[u]
            if int(o['frame_uid'].rsplit('_f',1)[1])>h:raise EvidenceInvariantError('future observation')
        return uids

    def observation(self,directory,rgb,detection,candidates,frame):
        self.sync()
        uid=str(detection['obs_uids'][-1]);cur=self.visual(uid)
        if cur['frame_idx']!=frame:raise EvidenceInvariantError('I1 must be from this issue S frame')
        if not np.array_equal(cur['rgb'],rgb):raise EvidenceInvariantError('current RGB mismatch')
        directory.mkdir(parents=True,exist_ok=True)
        audit_card(cur,directory/'quality.jpg')
        states=[object_state(o) for _,_,o in candidates]
        if len({s['object_uid'] for s in states})!=len(states):raise EvidenceInvariantError('duplicate objects')
        binding=dict(current_observation_uid=uid,h_frame=frame,objects=states,candidates=[])
        clouds={'current':np.asarray(detection['pcd'].points).copy()}
        images=[('CURRENT QUALITY',directory/'quality.jpg')]
        for alias,index,obj in candidates:
            members=self.members(obj,frame)
            if uid in members:raise EvidenceInvariantError('CURRENT already fused into candidate')
            center=np.asarray(obj['bbox'].get_center());curdir=np.asarray(cur['pose'])[:3,3]-center
            curdir/=max(np.linalg.norm(curdir),1e-12);ranking=[]
            for u in members:
                o=self.observations[u];idx=int(o['frame_uid'].rsplit('_f',1)[1])
                cam=self.runtime.projection_frames[idx];direction=np.asarray(cam['pose_c2w'])[:3,3]-center
                direction/=max(np.linalg.norm(direction),1e-12)
                points=self.array(o['pcd_ref'])
                proj=project_points(points,cur['pose'],cur['K'],cur['depth'])
                fraction=proj['counts']['reliable']/max(1,len(points))
                quality=float(o['processed_mask_area'])*float(o.get('confidence') or 0)*(1-min(.8,float(o.get('boundary_touch_ratio') or 0)))
                ranking.append(dict(uid=u,frame_idx=idx,quality=quality,
                                    comparability=float(np.sqrt(fraction)*max(0,np.dot(direction,curdir)))))
            first=max(ranking,key=lambda x:(x['comparability'],x['quality'],x['frame_idx']))
            hist=self.visual(first['uid'])
            points,_=unproject(hist['mask'],hist['depth'],hist['pose'],hist['K'])
            projection=project_points(points,cur['pose'],cur['K'],cur['depth'])
            geom=render.focused_card(cur,alias,hist,projection,directory/f'candidate_{alias}.jpg')
            np.savez_compressed(directory/f'projection_{alias}.npz',**{k:v for k,v in projection.items() if isinstance(v,np.ndarray)})
            clouds[alias]=np.asarray(obj['pcd'].points).copy()
            binding['candidates'].append(dict(alias=alias,index=index,object_uid=str(obj['id']),
                object_version=state_key([object_state(obj)]),history=first,ranking=ranking,geometry=geom,
                source_mask_sha256=self.observations[first['uid']]['processed_mask_ref']['sha256'],
                projection_source='selected_historical_mask_unprojected',same_history_rgb_mask_projection=True))
            images.append(('CANDIDATE '+alias,directory/f'candidate_{alias}.jpg'))
        np.savez_compressed(directory/'live_points.npz',**clouds)
        binding['images']=[dict(label=n,path=p.name,sha256=sha(p)) for n,p in images]
        binding['geometry_sha256']=sha(directory/'live_points.npz')
        save_json(directory/'staged_evidence.json',binding)
        return binding,images

    def merge(self,directory,source,target,frame):
        self.sync();objects={'A':source,'B':target}
        states={a:object_state(o) for a,o in objects.items()}
        if states['A']['object_uid']==states['B']['object_uid']:raise EvidenceInvariantError('same object pair')
        clouds={a:np.asarray(o['pcd'].points).copy() for a,o in objects.items()}
        if any(not len(p) for p in clouds.values()):raise ValueError('empty merge cloud')
        histories={};anchors=[]
        for a,obj in objects.items():
            members=self.members(obj,frame)
            rows,chosen=render.history_scores(members,self.observations,self.array)
            histories[a]=dict(all=rows,selected=chosen)
            render.node_card(a,chosen,directory/f'quality_{a}.jpg',self.visual)
            anchors+=render.anchor_scores(a,rows,clouds['B' if a=='A' else 'A'],self.observations,self.visual)
        anchor=max(anchors,key=lambda r:(r['score'],r['target_pixels'],r['uid']))
        other='B' if anchor['alias']=='A' else 'A';v=self.visual(anchor['uid'])
        projection=project_points(clouds[other],v['pose'],v['K'],v['depth'])
        cm=v['mask'];pm=np.zeros_like(cm);uv=projection['reliable_uv'];pm[uv[:,1],uv[:,0]]=True
        np.savez_compressed(directory/'live_pair.npz',**{'object_'+a:p for a,p in clouds.items()})
        np.savez_compressed(directory/'visible_projection.npz',**{k:v for k,v in projection.items() if isinstance(v,np.ndarray)})
        np.savez_compressed(directory/'display_support.npz',anchor_mask=cm,projected_mask=pm)
        snapshot=state_key(list(states.values()))
        binding=dict(objects=states,histories=histories,selected_anchor=anchor,
            other_appearance_history=dict(histories[other]['selected'][0],alias=other),
            source_timeline=dict(s_frame=frame,d_frame=frame,h_frame=frame),
            source_h_snapshot_uid=snapshot,projection_source='entire_frozen_live_node',
            projection_cloud_sha256=states[other]['pcd_sha256'])
        frames={i:dict(pose=f['pose_c2w']) for i,f in self.runtime.projection_frames.items() if i<=frame}
        selected={a:render.select(binding,a,frames) for a in 'AB'}
        im,meta=render.merge_card(binding,selected,self.visual,cm,pm);im.save(directory/'merge.png')
        binding.update(selected_identity_histories=selected,layout=meta,
            images={name:sha(directory/name) for name in ['quality_A.jpg','quality_B.jpg','merge.png']},
            geometry_sha256=sha(directory/'live_pair.npz'),depth_tolerance_m=.03)
        snapshot=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
        binding['h_snapshot_uid']=snapshot
        save_json(directory/'input_manifest.json',binding)
        return binding

    @staticmethod
    def containment(source,target,distance):
        from scipy.spatial import cKDTree
        a=np.asarray(source['pcd'].points);b=np.asarray(target['pcd'].points)
        if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('containment requires nonempty finite full point clouds')
        ab=int(np.count_nonzero(cKDTree(b).query(a,k=1)[0]<=distance))
        ba=int(np.count_nonzero(cKDTree(a).query(b,k=1)[0]<=distance))
        return dict(method='bidirectional_full_cloud_nearest_neighbor',
            distance_m=float(distance),a_points=len(a),b_points=len(b),a_in_b_count=ab,b_in_a_count=ba,
            a_in_b=ab/len(a),b_in_a=ba/len(b),threshold=.9,requires_human=ab/len(a)>.9 or ba/len(b)>.9)
