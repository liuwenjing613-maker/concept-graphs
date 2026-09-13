"""Single VLM verifier for existing baseline merge proposals; no votes/rescue."""
from collections import Counter
import hashlib,json,time
import numpy as np
from conceptgraph.slam.human_instance_merge import object_state,state_key
from conceptgraph.slam.vlm_runtime import save_json,sha
from conceptgraph.slam.association_gate import _utc_now,_jsonl_append
from conceptgraph.slam.v7_request_cache import digest

class V7MergeGate:
    def __init__(self,owner):
        self.owner=owner;self.runtime=owner.vlm_runtime
        self.root=owner.output_dir/'vlm_instance_merge';self.root.mkdir(exist_ok=True)
        self.events=[];self.stats=Counter();self.generations={}
        self.pending_execution={};self._summary('ready')

    def _summary(self,status):
        save_json(self.root/'summary.json',dict(status=status,counts=dict(self.stats),
            policy=dict(merge_verifications_per_evidence=1,votes=False,locks=False,generations=self.generations),
            failure_policy='DEFER_PRESERVES_BASELINE_MERGE',
            containment='disabled; no geometry rescue',
            proposal_policy='original baseline conditions only, after observation fusion',
            cache_policy='node quality: generation + selected history; identity: complete decision evidence'))

    def close(self,status='completed'):self._summary(status)

    def pair_key(self,source,target):
        return digest(sorted((str(o['id']),self.generations.get(str(o['id']),0)) for o in (source,target)))

    def selected_evidence(self,rows):
        result=[]
        for row in rows:
            uid=row['uid'];v=self.runtime.evidence.visual(uid)
            def ah(a):
                a=np.ascontiguousarray(a)
                return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
            result.append(dict(obs_uid=uid,rgb=ah(v['rgb']),mask=ah(v['mask'])))
        return result

    def cache_contexts(self,binding,source,target):
        objects=dict(zip('AB',(source,target)))
        node={}
        renderer_sha=sha(__import__('pathlib').Path(__file__).with_name('v7_render.py'))
        for a,o in objects.items():
            node[a]=dict(layer='node_quality',object_uid=str(o['id']),
                generation=self.generations.get(str(o['id']),0),
                selected_history=self.selected_evidence(binding['histories'][a]['selected']),
                renderer_sha256=renderer_sha)
        anchor=binding['selected_anchor'];v=self.runtime.evidence.visual(anchor['uid'])
        identity=dict(layer='merge_identity',pair=[
            dict(role=a,object_uid=str(o['id']),generation=self.generations.get(str(o['id']),0))
            for a,o in objects.items()],
            selected_history={a:self.selected_evidence(binding['selected_identity_histories'][a]) for a in 'AB'},
            node_quality_evidence=node,
            anchor=dict(alias=anchor['alias'],obs_uid=anchor['uid'],
                pose=np.asarray(v['pose']).tolist(),intrinsics=np.asarray(v['K']).tolist(),
                depth_sha256=hashlib.sha256(np.ascontiguousarray(v['depth']).tobytes()).hexdigest()),
            geometry={a:binding['objects'][a]['pcd_sha256'] for a in 'AB'},
            projection_source=binding['projection_source'],depth_tolerance_m=binding['depth_tolerance_m'],
            renderer_sha256=renderer_sha)
        return node,identity

    def review(self,source,target,*,frame_idx,source_frame_id,stage,overlap=0,visual=0,text=0,parent_event=None):
        if source is target or str(source['id'])==str(target['id']):
            raise ValueError('distinct objects required')
        if stage not in {'periodic','final'} or parent_event is not None:
            raise self.runtime.invariant_error('only original baseline merge proposals may be verified')
        key=self.pair_key(source,target);states=state_key([object_state(source),object_state(target)])
        self.pending_execution.pop(key,None)
        event_id=f'm{len(self.events)+1:05d}_f{frame_idx:06d}_{stage}'
        directory=self.root/'events'/event_id;directory.mkdir(parents=True,exist_ok=False)
        event=dict(event_id=event_id,task='merge',frame_idx=frame_idx,source_frame_id=source_frame_id,
            pair_key=key,parent_event=None,object_A=object_state(source),object_B=object_state(target),
            state_key=states,baseline_choice='MERGE',baseline_scores=dict(overlap=float(overlap),visual=float(visual),text=float(text)),
            timeline=dict(s_frame=frame_idx,d_frame=frame_idx,h_frame=frame_idx,h_utc=_utc_now(),
                          online_main_graph_latest_frame_at_h=frame_idx),
            h_snapshot_uid=states,status='preparing',execution='NOT_APPLIED',stages=[])
        self.events.append(event);self.stats['original_merge_proposals']+=1
        save_json(directory/'decision.json',event)
        choice='DEFER';reason='INPUT_FAILURE';images=[]
        try:
            started=time.perf_counter()
            binding=self.runtime.evidence.prepare_merge(source,target,frame_idx)
            event['history_check_seconds']=time.perf_counter()-started
            started=time.perf_counter()
            binding=self.runtime.evidence.render_merge(directory,source,target,frame_idx,binding)
            event['render_seconds']=time.perf_counter()-started
            snapshot=binding['h_snapshot_uid'];event['h_snapshot_uid']=snapshot
            images=[('QUALITY A',directory/'quality_A.jpg'),('QUALITY B',directory/'quality_B.jpg'),
                    ('MERGE',directory/'merge.png')]
            self.runtime.pending(event_id,directory,'merge',images,['MERGE','KEEP_SEPARATE','DEFER'])
            self.runtime.rows[event_id].update(h_snapshot_uid=snapshot,stages=event['stages'])
            node_context,identity_context=self.cache_contexts(binding,source,target)
            specs=[]
            for a in 'AB':
                labels=['H'+str(i+1) for i in range(len(binding['histories'][a]['selected']))]
                specs.append((directory/('quality_'+a),'node_quality',
                    [('NODE '+a,directory/f'quality_{a}.jpg')],labels,node_context[a]))
            results=self.runtime.stage_many(event_id,specs,snapshot)
            event['stages'].extend(results);qualities=[x['value'] for x in results]
            if any(q is None for q in qualities):reason='NODE_QUALITY_INTERFACE_FAILURE'
            elif any(q['choice']=='INSUFFICIENT' for q in qualities):reason='NODE_QUALITY_INSUFFICIENT'
            elif any(q['choice']=='CONTAMINATED' for q in qualities):
                # Keep the original explicit contaminated-node veto.
                choice='KEEP_SEPARATE';reason='NODE_CONTAMINATED'
            else:
                result=self.runtime.stage(event_id,directory/'identity','merge',
                    [('MERGE',directory/'merge.png')],snapshot,cache_context=identity_context)
                event['stages'].append(result);value=result['value'];event['identity_output']=value
                choice={'SAME':'MERGE','DIFFERENT':'KEEP_SEPARATE','UNCERTAIN':'DEFER'}[value['choice']] if value else 'DEFER'
                reason='IDENTITY_'+value['choice'] if value else 'IDENTITY_INTERFACE_FAILURE'
        except Exception as exc:
            if isinstance(exc,self.runtime.invariant_error):raise
            event['input_error']=type(exc).__name__+': '+str(exc)
            choice='DEFER';reason='INPUT_FAILURE'
        if states!=state_key([object_state(source),object_state(target)]):
            raise self.runtime.invariant_error('objects changed during blocking verifier')
        # SAME approves; DIFFERENT vetoes; DEFER follows the baseline MERGE.
        approved=choice in {'MERGE','DEFER'}
        event.update(model_output=dict(choice=choice,confidence=0),reason_code=reason,status='complete',
            decision_source='BASELINE_DEFER' if choice=='DEFER' else 'VLM_VERIFIER',
            final_choice='MERGE' if approved else 'KEEP_SEPARATE',
            execution='APPROVED_AWAITING_EXECUTION' if approved else 'VETOED_THIS_EVIDENCE',
            c_bound_h_snapshot_uid=event['h_snapshot_uid'])
        event['timeline'].update(c_frame=frame_idx,c_utc=_utc_now(),online_main_graph_latest_frame_at_c=frame_idx,
                                 ordering_valid=True)
        if approved:self.pending_execution[key]=dict(event=event,state=states)
        save_json(directory/'decision.json',event);_jsonl_append(self.root/'events.jsonl',event)
        self.runtime.rows.setdefault(event_id,dict(event_id=event_id,images=[])).update(event)
        self.stats['decision_'+choice]+=1
        self.stats['cache_hits']+=sum(x.get('cache_hit',False) for x in event['stages'])
        self.stats['actual_calls']+=sum(x.get('actual_call_count',0) for x in event['stages'])
        self.runtime.publish();self._summary('ready')
        print('[v7-merge]',event_id,reason,'->',event['final_choice'],event['decision_source'],flush=True)
        return None if approved else 'v7_verifier_veto'

    def on_merged(self,source,target):
        key=self.pair_key(source,target)
        certificate=self.pending_execution.pop(key,None)
        if certificate is None:raise self.runtime.invariant_error('merge without current proposal approval')
        event=certificate['event']
        event.update(execution='MERGED',executed_at=_utc_now(),target_uid=str(target['id']))
        for o in (source,target):
            uid=str(o['id']);self.generations[uid]=self.generations.get(uid,0)+1
            self.owner._support_history.pop(uid,None)
        event['target_generation_after']=self.generations[str(target['id'])]
        save_json(self.root/'events'/event['event_id']/'decision.json',event)
        _jsonl_append(self.root/'executions.jsonl',event)
        self.runtime.rows[event['event_id']].update(event);self.stats['executed']+=1
        self.runtime.publish();self._summary('ready')
