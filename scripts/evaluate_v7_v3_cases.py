"""Isolated frozen-event counterfactual; never loads or mutates an online map."""
import argparse,copy,json,os,shutil,time,hashlib
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from conceptgraph.slam.v7_runtime import V7Runtime,PROMPTS
from conceptgraph.slam.v7_endpoint_pool import EndpointPool
from conceptgraph.slam.v7_merge import V7MergeGate
from conceptgraph.slam.v7_merge_policy import containment_route
from conceptgraph.slam.vlm_runtime import save_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--urls',nargs='+',default=['http://127.0.0.1:11437']);p.add_argument('--models',nargs='+',default=['qwen3.6:35b-a3b-mtp-q4_K_M'])
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(Path(args.inventory).read_text());rows=[]
    for row in inventory['cases']:
        row=copy.deepcopy(row);src=Path(row['source_dir']);dst=root/'events'/row['event_id'];dst.mkdir(parents=True,exist_ok=True)
        assert sha(src/'decision.json')==row['decision_sha256']
        original=json.loads((src/'decision.json').read_text())
        for name in ['quality_A.jpg','quality_B.jpg','merge.png','input_manifest.json']:
            shutil.copy2(src/name,dst/name);assert sha(src/name)==sha(dst/name)
        row['image_hashes']={n:sha(dst/n) for n in ['quality_A.jpg','quality_B.jpg','merge.png']}
        row['route']=containment_route(row['quality'],row['containment']);row['status']='pending';rows.append(row)
    state=dict(kind='fixed_H_event_counterfactual',cases=rows,status='prepared',scope='all 20 high-containment v7_merge events; no full scene execution',
        reference=inventory['gt_proxy_protocol'],endpoints=dict(zip(args.urls,args.models)))
    save_json(root/'data.json',state)
    if args.prepare_only:return
    runtime=V7Runtime.__new__(V7Runtime);runtime.root=root;runtime.rows={};runtime.fallback='auto'
    runtime.owner=SimpleNamespace(model=args.models[0],base_url=args.urls[0],timeout_seconds=120)
    runtime.endpoint_pool=EndpointPool(args.urls,120,models=args.models)
    runtime.templates=json.loads((PROMPTS/'request_templates.json').read_text())
    runtime.stage_prompts={f.stem:f.read_text() for f in PROMPTS.glob('*.txt')};runtime.publish=lambda:save_json(root/'data.json',state)
    gate=V7MergeGate.__new__(V7MergeGate);gate.runtime=runtime
    state['status']='running';runtime.publish();start=time.perf_counter()
    for row in rows:
        dst=root/'events'/row['event_id'];src=Path(row['source_dir']);event=copy.deepcopy(json.loads((src/'decision.json').read_text()))
        for key in ['decision_source','executed_at','target_uid','containment_human_choice','vote_after']:
            event.pop(key,None)
        event['execution']='NOT_APPLIED_OFFLINE';event['stages']=copy.deepcopy(event['stages'])
        runtime.rows[row['event_id']]={}
        row['status']='running';runtime.publish();t=time.perf_counter()
        choice=gate.arbitrate_negative(event,dst)
        row.update(new_action=choice,route=event['override_route'],fragment_output=event.get('fragment_output'),defer_reason=event.get('defer_reason'),status='complete',seconds=time.perf_counter()-t)
        row['interface_failure']=event.get('defer_reason')=='FRAGMENT_INTERFACE_FAILURE'
        row['changed']=choice!=row['old_action']
        row['new_disagreement']=row['gt_proxy']=='SAME' and choice!='MERGE' or row['gt_proxy']=='DIFFERENT' and choice=='MERGE'
        save_json(dst/'counterfactual.json',event);runtime.publish()
    eligible=[x for x in rows if x['gt_proxy']!='AMBIGUOUS']
    def metrics(action):
        tp=sum(x['gt_proxy']=='SAME' and x[action]=='MERGE' for x in eligible)
        fp=sum(x['gt_proxy']=='DIFFERENT' and x[action]=='MERGE' for x in eligible)
        fn=sum(x['gt_proxy']=='SAME' and x[action]!='MERGE' for x in eligible)
        return dict(correct_merges=tp,incorrect_merges=fp,missed_same_event_merges=fn,merge_precision=tp/(tp+fp) if tp+fp else None,merge_recall=tp/(tp+fn) if tp+fn else None)
    state.update(status='complete',seconds=time.perf_counter()-start,summary=dict(total=len(rows),routes=dict(Counter(x['route'] for x in rows)),actions=dict(Counter(x['new_action'] for x in rows)),
        gt_proxy_counts=dict(Counter(x['gt_proxy'] for x in rows)),old=metrics('old_action'),v3=metrics('new_action'),interface_failures=sum(x['interface_failure'] for x in rows)))
    runtime.publish();print(json.dumps(state['summary'],ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
