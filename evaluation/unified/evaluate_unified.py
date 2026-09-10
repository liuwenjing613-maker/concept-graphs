"""Unified full-mesh evaluation. CPU only. See README.md for the frozen protocol."""
import argparse, copy, gzip, hashlib, json, pickle, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import ovimap_ap_pinned as ap_engine
ROOT=Path(__file__).resolve().parent
VERSION='unified_fullmesh_v2_20260910'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def dump(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False))
def read(p):return json.loads(Path(p).read_text())

def project(points,owners,xyz,workers=4):
 points=np.asarray(points);owners=np.asarray(owners)
 assert points.shape==(len(owners),3) and np.isfinite(points).all()
 if not len(points):return np.full(len(xyz),-1,np.int32),np.full(len(xyz),np.inf)
 assert np.issubdtype(owners.dtype,np.integer) and (owners>=-1).all()
 valid=owners>=0;points=points[valid];owners=owners[valid]
 if not len(points):return np.full(len(xyz),-1,np.int32),np.full(len(xyz),np.inf)
 tree=cKDTree(points)
 distance,index=tree.query(xyz,k=1,workers=workers)
 return np.where(distance<.05,owners[index],-1).astype(np.int32),distance

def cosine_classes(features,emb):
 emb=np.asarray(emb,np.float64);ft=np.asarray(features,np.float64)
 if emb.ndim!=2 or not len(emb) or not np.isfinite(emb).all():raise ValueError('Invalid text embeddings')
 en=np.linalg.norm(emb,axis=1,keepdims=True)
 if (en<=0).any():raise ValueError('Zero text embedding')
 ft=ft.reshape(len(ft),emb.shape[1]);fn=np.linalg.norm(ft,axis=1,keepdims=True)
 if not np.isfinite(ft).all() or (fn<=0).any():raise ValueError('Invalid object embeddings')
 return ((ft/fn)@(emb/en).T).argmax(1)+1 if len(ft) else np.empty(0,int)

def load_prediction(entry,xyz,text,workers):
 path=Path(entry['map']);kind=entry['kind']
 if kind=='cg':
  with gzip.open(path,'rb') as f:objects=pickle.load(f)['objects']
  points=[];owners=[];features=[]
  for i,o in enumerate(objects):
   p=np.asarray(o['pcd_np'],np.float32);points.append(p);owners.append(np.full(len(p),i,np.int32));features.append(np.asarray(o['clip_ft'],np.float32).reshape(-1))
  points=np.concatenate(points) if points else np.empty((0,3),np.float32)
  owners=np.concatenate(owners) if owners else np.empty(0,np.int32)
  classes=cosine_classes(features,np.load(text))
 else:
  with np.load(path,allow_pickle=False) as d:points=d['xyz'];owners=d['instance'];classes=d['classes']
 assert np.issubdtype(owners.dtype,np.integer) and np.issubdtype(classes.dtype,np.integer)
 assert (owners>=-1).all() and (owners<len(classes)).all()
 pred,dist=project(points,owners,xyz,workers)
 return pred,dist,classes

def contingency(gt,pred,n):
 ids,inv,counts=np.unique(gt,return_inverse=True,return_counts=True)
 covered=pred>=0
 inter=np.bincount(inv[covered]*n+pred[covered],minlength=len(ids)*n).reshape(len(ids),n) if n else np.zeros((len(ids),0),np.int64)
 area=np.bincount(pred[covered],minlength=n)
 return ids,counts,area,inter

def semantic(conf,names):
 gt=conf.sum(1);pa=conf.sum(0);tp=conf.diagonal();union=gt+pa-tp
 iou=np.divide(tp,union,out=np.full(len(gt),np.nan),where=union>0)
 acc=np.divide(tp,gt,out=np.full(len(gt),np.nan),where=gt>0)
 vals=iou[1:]; recalls=acc[1:]
 return {'semantic_mIoU':float(np.mean(vals[gt[1:]>0])) if (gt[1:]>0).any() else None,'mAcc':float(np.nanmean(recalls)) if np.isfinite(recalls).any() else None,'semantic_mIoU_union_present_diagnostic':float(np.nanmean(vals)) if np.isfinite(vals).any() else None},[{'id':i,'class':name,'gt_vertices':int(gt[i]),'pred_vertices':int(pa[i]),'intersection':int(tp[i]),'IoU':iou[i],'accuracy':acc[i]}for i,name in enumerate(names,1)]

def instance_iou_one_to_one(ids,counts,area,inter,valid_classes):
 mask=(np.isin(ids//1000,valid_classes))&(counts>=100)&(ids>=1000)
 gs=ids[mask];sizes=counts[mask];ints=inter[mask]
 union=sizes[:,None]+area[None,:]-ints
 iou=np.divide(ints,union,out=np.zeros_like(ints,dtype=float),where=union>0)
 iou[:,area<100]=0
 a,b=linear_sum_assignment(iou,maximize=True)
 values=np.zeros(len(gs));values[a]=iou[a,b]
 return {'instance_mIoU':float(values.mean()) if len(gs) else None,'instance_iou_sum':float(values.sum()),'gt_instances':len(gs),'instance_mIoU_matched_diagnostic':float(values[values>0].mean()) if (values>0).any() else 0.,'instance_mIoU_gt_best_diagnostic':float(iou.max(1).mean()) if len(gs) and len(area) else (0. if len(gs) else None)},[{'gt':int(gs[g]),'pred':int(p),'iou':float(iou[g,p])}for g,p in zip(a,b) if iou[g,p]>0]

def instance_iou(ids,counts,area,inter,valid_classes):
 # OVI best-per-GT reduction: GT >=10, no prediction area filter, reuse allowed.
 keep=np.isin(ids//1000,valid_classes)&(ids>=1000)&(counts>=10)
 gs=ids[keep];ints=inter[keep];union=counts[keep,None]+area[None,:]-ints
 iou=np.divide(ints,union,out=np.zeros_like(ints,dtype=float),where=union>0)
 values=iou.max(1) if len(area) else np.zeros(len(gs))
 matches=[{'gt':int(g),'pred':int(iou[i].argmax()),'iou':float(values[i])} for i,g in enumerate(gs) if values[i]>0]
 return {'instance_mIoU':float(values.mean()) if len(gs) else None,'instance_iou_sum':float(values.sum()),'gt_instances':len(gs)},matches

def detection_diagnostic(ids,counts,area,inter,valid_classes):
 keep=np.isin(ids//1000,valid_classes)&(ids>=1000)&(counts>=100)
 ps=area>=100;ints=inter[keep][:,ps];union=counts[keep,None]+area[ps][None,:]-ints
 iou=np.divide(ints,union,out=np.zeros_like(ints,dtype=float),where=union>0)
 # Cardinality is primary; total IoU breaks ties. One prediction cannot explain two GTs.
 good=iou>=.5;weight=good.astype(float)*(min(iou.shape)+1)+np.where(good,iou,0)
 a,b=linear_sum_assignment(weight,maximize=True);tp=int(good[a,b].sum());fp=int(ps.sum())-tp;fn=int(keep.sum())-tp
 return detection_counts(tp,fp,fn)

def detection_counts(tp,fp,fn):
 return {'diagnostic_TP50':tp,'diagnostic_FP50':fp,'diagnostic_FN50':fn,
 'diagnostic_precision50':tp/(tp+fp) if tp+fp else 0.,
 'diagnostic_recall50':tp/(tp+fn) if tp+fn else None,
 'diagnostic_F1_50':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}

def build_matches(scene,ids,counts,area,inter,classes,valid_classes,names,agnostic=False):
 labels=['object'] if agnostic else [names[c-1] for c in valid_classes]
 gt={k:[] for k in labels};pred={k:[] for k in labels}
 valid=np.isin(ids//1000,valid_classes)&(ids>=1000)
 void=inter[~valid].sum(0)
 gt_by_index={}
 for g in np.flatnonzero(valid):
  label='object' if agnostic else names[int(ids[g]//1000)-1]
  obj={'instance_id':int(ids[g]),'label_id':1 if agnostic else int(ids[g]//1000),'vert_count':int(counts[g]),'med_dist':-1,'dist_conf':0.,'matched_pred':[]}
  gt[label].append(obj);gt_by_index[g]=obj
 max_all=max(area,default=0)
 max_cls={int(c):max(area[classes==c],default=0) for c in np.unique(classes)}
 for p in range(len(classes)):
  c=int(classes[p])
  if area[p]<100 or (not agnostic and c not in valid_classes):continue
  label='object' if agnostic else names[c-1]
  denominator=max_all if agnostic else max_cls[c]
  # Six-decimal area score reproduces the existing official-export convention.
  score=float(f'{area[p]/denominator:.6f}')
  obj={'filename':f'{scene}/object_{p}','pred_id':p,'label_id':1 if agnostic else c,'vert_count':int(area[p]),'confidence':score,'void_intersection':int(void[p])}
  matched=[]
  for g in np.flatnonzero((inter[:,p]>0)&valid):
   if not agnostic and ids[g]//1000!=c:continue
   gc={k:v for k,v in gt_by_index[g].items() if k!='matched_pred'};gc['intersection']=int(inter[g,p]);matched.append(gc)
   pc=dict(obj);pc['intersection']=int(inter[g,p]);gt_by_index[g]['matched_pred'].append(pc)
  obj['matched_gt']=matched;pred[label].append(obj)
 return {'gt':gt,'pred':pred}

def run_ap(matches,labels):
 ap_engine.CLASS_LABELS=labels;ap_engine.overlaps=np.array([.25,.50,.75]);ap_engine.min_region_sizes=np.array([100]);ap_engine.dist_threshes=np.array([np.inf]);ap_engine.dist_confs=np.array([-np.inf])
 values=ap_engine.evaluate_matches(matches)[0]
 means=[float(np.nanmean(values[:,i])) if np.isfinite(values[:,i]).any() else None for i in range(3)]
 return means,values

def evaluate(entry,manifest,out_root,workers=4):
 start=time.perf_counter();scene=entry['scene'];method=entry['method'];out=Path(out_root)/method/scene
 out.mkdir(parents=True,exist_ok=True)
 refdir=Path(manifest['reference_root'])/scene;rm=read(refdir/'manifest.json');rp=refdir/'reference.npz'
 assert sha(rp)==rm['reference_sha256']
 fingerprint={'version':VERSION,'map_sha256':sha(entry['map']),'reference_sha256':sha(rp),'reference_manifest_sha256':sha(refdir/'manifest.json'),'evaluator_sha256':sha(__file__),'engine_sha256':sha(ROOT/'ovimap_ap_pinned.py'),'text_sha256':sha(manifest['clip_text'])}
 if (out/'result.json').exists():
  old=read(out/'result.json')
  if old['fingerprint']==fingerprint:return old
  raise RuntimeError(f'Output exists with different inputs: {out}. Choose a fresh --out.')
 ref=np.load(rp);xyz=ref['xyz'];gtsem=ref['semantic'];gtinst=ref['instance'];names=rm['semantic_classes'];nc=len(names)
 valid_classes=[names.index(n)+1 for n in rm['instance_classes']]
 assert ((gtsem>=0)&(gtsem<=nc)).all() and len(xyz)==len(gtinst)
 pred,dist,classes=load_prediction(entry,xyz,manifest['clip_text'],workers);projection_source='fresh exact cKDTree over assigned object points'
 assert len(pred)==len(xyz)==len(dist) and ((classes>=0)&(classes<=nc)).all()
 assert ((pred>=-1)&(pred<len(classes))).all() and np.all(dist[pred>=0]<.05)
 np.savez_compressed(out/'projection.npz',instance=pred,distance_m=dist,object_class=classes)
 sem=np.zeros(len(xyz),np.int64);covered=pred>=0;sem[covered]=classes[pred[covered]]
 valid=gtsem>0;conf=np.bincount(gtsem[valid].astype(np.int64)*(nc+1)+sem[valid],minlength=(nc+1)**2).reshape(nc+1,nc+1)
 np.save(out/'confusion.npy',conf)
 metrics,per_class=semantic(conf,names)
 ids,counts,area,inter=contingency(gtinst,pred,len(classes))
 im,matched=instance_iou(ids,counts,area,inter,valid_classes);metrics.update(im)
 one,one_matches=instance_iou_one_to_one(ids,counts,area,inter,valid_classes)
 metrics.update(instance_mIoU_one_to_one=one['instance_mIoU'],instance_iou_sum_one_to_one=one['instance_iou_sum'],gt_instances_one_to_one=one['gt_instances'])
 metrics.update(detection_diagnostic(ids,counts,area,inter,valid_classes))
 dump(out/'instance_matches_one_to_one.json',one_matches)
 ca=build_matches(scene,ids,counts,area,inter,classes,valid_classes,names,True)
 sa=build_matches(scene,ids,counts,area,inter,classes,valid_classes,names,False)
 cav,caf=run_ap({scene:ca},['object']);sav,saf=run_ap({scene:sa},[names[c-1] for c in valid_classes])
 metrics.update(instance_AP25=cav[0],instance_AP50=cav[1],instance_AP75=cav[2],semantic_AP25=sav[0],semantic_AP50=sav[1])
 instvalid=np.isin(gtsem,valid_classes)
 for label,mask in [('instances',instvalid),('semantic',valid),('all_mesh',np.ones(len(xyz),bool))]:
  metrics['coverage_'+label]=float(covered[mask].mean()) if mask.any() else None
  metrics['covered_'+label]=int(covered[mask].sum());metrics['vertices_'+label]=int(mask.sum())
 metrics['object_surface_coverage_5cm']=metrics.pop('coverage_instances')
 metrics.update(map_instances=len(classes),zero_support_instances=int((area==0).sum()),small_support_instances=int(((area>0)&(area<100)).sum()))
 dump(out/'ca_ap_matches.json',ca);dump(out/'semantic_ap_matches.json',sa);dump(out/'instance_matches.json',matched);dump(out/'per_class.json',per_class);dump(out/'ap_per_class.json',{'classes':[names[c-1]for c in valid_classes],'thresholds':[.25,.5,.75],'ap':saf})
 result={'scene':scene,'method':method,'protocol':VERSION,'fingerprint':fingerprint,'source':entry,'reference':rm,'evaluation_protocol':{'version':VERSION,'projection':'assigned points only; exact k1; distance <0.05m','semantic_average':'GT-present classes; no exclude6','instance_mIoU':'best per GT, GT >=10, no predicted size filter','one_to_one_and_AP_min_points':100,'diagnostic_F1_threshold':'>=0.5','AP_thresholds':'strict >'},'projection_source':projection_source,'metrics':metrics,'cost':entry.get('cost',{}),'evaluation_seconds':time.perf_counter()-start}
 dump(out/'result.json',result);print(method,scene,clean(metrics),flush=True)
 return result

def aggregate(results,manifest,out):
 all_methods={};names=results[0]['reference']['semantic_classes'];valid_classes=[names.index(n)+1 for n in results[0]['reference']['instance_classes']]
 for method in dict.fromkeys(r['method']for r in results):
  group=[r for r in results if r['method']==method];ca={};sa={};conf=np.zeros((len(names)+1,)*2,np.int64)
  for r in group:
   d=Path(out)/method/r['scene'];conf+=np.load(d/'confusion.npy');ca[r['scene']]=read(d/'ca_ap_matches.json');sa[r['scene']]=read(d/'semantic_ap_matches.json')
  cav,_=run_ap(ca,['object']);sav,_=run_ap(sa,[names[c-1]for c in valid_classes]);m,pc=semantic(conf,names)
  ng=sum(r['metrics']['gt_instances']for r in group);si=sum(r['metrics']['instance_iou_sum']for r in group)
  m.update(instance_mIoU=si/ng if ng else None,instance_AP25=cav[0],instance_AP50=cav[1],instance_AP75=cav[2],semantic_AP25=sav[0],semantic_AP50=sav[1],scene_count=len(group),gt_instances=ng)
  n1=sum(r['metrics']['gt_instances_one_to_one']for r in group);s1=sum(r['metrics']['instance_iou_sum_one_to_one']for r in group)
  m.update(instance_mIoU_one_to_one=s1/n1 if n1 else None,gt_instances_one_to_one=n1)
  m.update(detection_counts(*(sum(r['metrics']['diagnostic_'+k+'50']for r in group)for k in ['TP','FP','FN'])))
  for key in ['instances','semantic','all_mesh']:
   denominator=sum(r['metrics']['vertices_'+key]for r in group);numerator=sum(r['metrics']['covered_'+key]for r in group);m['coverage_'+key]=numerator/denominator if denominator else None
  m['object_surface_coverage_5cm']=m.pop('coverage_instances')
  scene_mean={k:float(np.mean([r['metrics'][k]for r in group])) for k in ['instance_mIoU','instance_AP25','instance_AP50','instance_AP75','semantic_mIoU','semantic_AP50','object_surface_coverage_5cm','mAcc','instance_mIoU_one_to_one','semantic_AP25'] if all(r['metrics'][k] is not None for r in group)}
  costs=[r['cost'] for r in group];cost={}
  for k in ['vlm_queries','mapping_runtime_seconds','end_to_end_runtime_seconds']:
   vals=[c.get(k)for c in costs];cost[k]=sum(vals) if all(v is not None for v in vals) else None
  cost['runtime_seconds_per_frame']=cost['end_to_end_runtime_seconds']/sum(c['frames']for c in costs) if cost['end_to_end_runtime_seconds'] is not None else None
  all_methods[method]={'pooled':m,'scene_mean':scene_mean,'cost':cost,'per_class':pc,'scenes':group}
 summary={'protocol':VERSION,'aggregation':'Main aggregate: joint AP across scans; pooled confusion; GT-instance weighted instance mIoU; vertex-weighted coverage. Scene macro means are separate.','methods':all_methods}
 dump(Path(out)/'comparison.json',summary)
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--scenes',nargs='*');p.add_argument('--methods',nargs='*');p.add_argument('--workers',type=int,default=4);a=p.parse_args();m=read(a.manifest)
 entries=[e for e in m['entries']if (not a.scenes or e['scene']in a.scenes)and(not a.methods or e['method']in a.methods)]
 assert entries and len({(e['method'],e['scene'])for e in entries})==len(entries)
 r=[evaluate(e,m,a.out,a.workers)for e in entries];aggregate(r,m,a.out)
if __name__=='__main__':main()
