"""Read-only nomination feasibility on an existing map; never initializes online mapping."""
import argparse,gzip,pickle,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import compute_overlap_matrix
from conceptgraph.slam.v7_merge import V7MergeGate
from conceptgraph.slam.v7_evidence import LiveEvidence

def main():
    parser=argparse.ArgumentParser();parser.add_argument('map');parser.add_argument('output');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    data=pickle.load(gzip.open(args.map));objects=MapObjectList();objects.load_serializable(data['objects']);cfg=data['cfg']
    overlap=compute_overlap_matrix(objects,cfg['downsample_voxel_size']);native=set();similarities={}
    for i in range(len(objects)):
        for j in range(i+1,len(objects)):
            key=tuple(sorted((str(objects[i]['id']),str(objects[j]['id']))))
            visual=float(torch.nn.functional.cosine_similarity(objects[i]['clip_ft'],objects[j]['clip_ft'],dim=0));similarities[key]=visual
            if max(overlap[i,j],overlap[j,i])>cfg['merge_overlap_thresh'] and visual>max(cfg['merge_visual_sim_thresh'],cfg['merge_text_sim_thresh']):native.add(key)
    runtime=SimpleNamespace(fallback='auto',evidence=SimpleNamespace(containment=LiveEvidence.containment),containment_distance=cfg['downsample_voxel_size'])
    gate=V7MergeGate(SimpleNamespace(output_dir=out,vlm_runtime=runtime));pairs=[]
    for i,j,report in gate.supplemental_pairs(objects,[True]*len(objects),native,frame_idx=-1,stage='offline_diagnostic'):
        key=tuple(sorted((str(objects[i]['id']),str(objects[j]['id']))))
        pairs.append(dict(pair=key,containment=report,native_overlap=float(max(overlap[i,j],overlap[j,i])),visual=similarities[key]))
    result=dict(kind='offline_nomination_diagnostic_not_accuracy_or_online_replay',source_map=str(Path(args.map).resolve()),objects=len(objects),
        possible_pairs=len(objects)*(len(objects)-1)//2,native_candidates=len(native),supplemental_candidates=len(pairs),pairs=pairs,seconds=time.perf_counter()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
