"""V7 pair voting and live merge review."""
from collections import Counter
import hashlib,json,time
from itertools import combinations
from conceptgraph.slam.vlm_instance_merge import MergeVotes
from conceptgraph.slam.human_instance_merge import object_state,state_key
from conceptgraph.slam.vlm_runtime import save_json,sha
from conceptgraph.slam.association_gate import _utc_now,_jsonl_append

class V7Votes(MergeVotes):
    def __init__(self):super().__init__(merge_required=2,reject_required=2)
    def skip_reason(self,key,frame):
        why=super().skip_reason(key,frame)
        return 'locked_after_two_rejections' if why=='locked_after_three_rejections' else why
    def refresh_history(self,key,signature):
        row=self.state(key)
        if row['locked'] and row.get('history_signature')!=signature:
            row.update(merge_streak=0,reject_total=0,locked=False,awaiting_execution=False)
            row['history_unlocks']=row.get('history_unlocks',0)+1
        row['history_signature']=signature

class V7MergeGate:
    def __init__(self,owner):
        self.owner=owner;self.runtime=owner.vlm_runtime;self.root=owner.output_dir/'vlm_instance_merge'
        self.root.mkdir(exist_ok=True);self.votes=V7Votes();self.events=[];self.stats=Counter()
        self.certificates={};self.last_event={};self._summary('ready')
    def _summary(self,status):
        save_json(self.root/'summary.json',dict(status=status,policy=self.votes.snapshot(),counts=dict(self.stats),
            failure_policy=self.runtime.fallback,containment='KEEP_SEPARATE only; >90%: human reviews, auto records conflict and keeps separate',
            unlock='changed selected history observation UIDs or masks; geometry-only update does not unlock'))
    def close(self,status='completed'):self._summary(status)
    def containment_check(self,source,target):
        report=self.runtime.evidence.containment(source,target,self.runtime.containment_distance)
        report['exceeds_threshold']=report.pop('requires_human')
        report['requires_human']=report['exceeds_threshold'] and self.runtime.fallback=='human'
        report['resolution_policy']=('HUMAN_REVIEW' if report['requires_human'] else
            'AUTO_KEEP_SEPARATE' if report['exceeds_threshold'] else 'KEEP_SEPARATE')
        return report
    def review(self,source,target,*,frame_idx,source_frame_id,stage,overlap=0,visual=0,text=0,parent_event=None):
        if source is target or str(source['id'])==str(target['id']):raise ValueError('distinct objects required')
        key=self.votes.key(source['id'],target['id']);states=state_key([object_state(source),object_state(target)])
        row=self.votes.state(key)
        # One decision per identity pair and frame; previously approved certificate is reusable only unchanged.
        if row['last_frame'] is not None and frame_idx<=row['last_frame']:
            return None if row['awaiting_execution'] and self.certificates.get(key)==states else 'v7_pair_already_reviewed_frame'
        if row['awaiting_execution']:
            row['awaiting_execution']=False
        event_id=f'm{len(self.events)+1:05d}_f{frame_idx:06d}_{stage}'
        directory=self.root/'events'/event_id;directory.mkdir(parents=True,exist_ok=False)
        event=dict(event_id=event_id,task='merge',frame_idx=frame_idx,pair_key=key,parent_event=parent_event,
            object_A=object_state(source),object_B=object_state(target),state_key=states,
            timeline=dict(s_frame=frame_idx,d_frame=frame_idx,h_frame=frame_idx,h_utc=_utc_now(),
                online_main_graph_latest_frame_at_h=frame_idx),status='preparing',execution='NOT_APPLIED',stages=[])
        self.events.append(event)
        save_json(directory/"decision.json",event)
        images=[];choice=None
        try:
            prepare_started=time.perf_counter()
            binding=self.runtime.evidence.prepare_merge(source,target,frame_idx)
            event['history_check_seconds']=time.perf_counter()-prepare_started
            snapshot=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
            event['h_snapshot_uid']=snapshot
            fingerprint=hashlib.sha256(json.dumps({
                str((source if a=='A' else target)['id']):sorted(set((r['uid'],self.runtime.evidence.observations[r['uid']]['processed_mask_ref']['sha256'])
                    for r in binding['histories'][a]['selected']+binding['selected_identity_histories'][a]))
                for a in 'AB'},sort_keys=True).encode()).hexdigest()
            before_unlock=row.get('history_unlocks',0);self.votes.refresh_history(key,fingerprint)
            if row['locked']:
                save_json(directory/'history_check.json',dict(h_snapshot_uid=snapshot,selection=binding,
                    history_signature=fingerprint,vlm_images_rendered=False))
                self.runtime.pending(event_id,directory,'merge',[],['KEEP_SEPARATE'])
                event.update(status='locked',execution='LOCKED_KEEP_SEPARATE',vote_after=dict(row))
                event['timeline'].update(c_frame=frame_idx,c_utc=_utc_now(),online_main_graph_latest_frame_at_c=frame_idx,ordering_valid=True)
                event['c_bound_h_snapshot_uid']=snapshot
                save_json(directory/'decision.json',event)
                self.runtime.rows[event_id].update(event);self.runtime.publish();self.stats['locked_skips']+=1
                return 'v7_locked_after_two_rejections'
            render_started=time.perf_counter()
            binding=self.runtime.evidence.render_merge(directory,source,target,frame_idx,binding)
            event['render_seconds']=time.perf_counter()-render_started
            snapshot=binding['h_snapshot_uid'];event['h_snapshot_uid']=snapshot
            images=[('QUALITY A',directory/'quality_A.jpg'),('QUALITY B',directory/'quality_B.jpg'),('MERGE',directory/'merge.png')]
            self.runtime.pending(event_id,directory,'merge',images,['MERGE','KEEP_SEPARATE'])
            self.runtime.rows[event_id].update(parent_event=parent_event,h_snapshot_uid=snapshot,stages=event['stages'])
            qualities=[]
            for a in 'AB':
                labels=['H'+str(i+1) for i in range(len(binding['histories'][a]['selected']))]
                result=self.runtime.stage(event_id,directory/('quality_'+a),'node_quality',
                    [('NODE '+a,directory/f'quality_{a}.jpg')],snapshot,labels=labels)
                event['stages'].append(result)
                value=result['value']
                if self.runtime.fallback=='human' and (value is None or value['choice']!='CLEAN'):
                    selected=self.runtime.human_choice(event_id,directory/('quality_'+a),
                        ['CLEAN','CONTAMINATED','INSUFFICIENT'],[('NODE '+a,directory/f'quality_{a}.jpg')],
                        '节点 '+a+' 的历史质量复核：单一实例／混入多个实例／证据不足；质量通过后继续合并身份 VLM。',
                        snapshot,question_type='node_quality_'+a)
                    value=dict(choice=selected,reason='人工节点质量复核')
                    event.setdefault('human_node_quality',{})[a]=value
                qualities.append(value)
            if any(q is None for q in qualities):reason='NODE_QUALITY_INTERFACE_FAILURE'
            elif any(q['choice']=='CONTAMINATED' for q in qualities):reason='NODE_CONTAMINATED'
            elif any(q['choice']=='INSUFFICIENT' for q in qualities):reason='NODE_INSUFFICIENT'
            else:
                result=self.runtime.stage(event_id,directory/'identity','merge',[('MERGE',directory/'merge.png')],snapshot)
                event['stages'].append(result)
                value=result['value']
                choice={'SAME':'MERGE','DIFFERENT':'KEEP_SEPARATE','UNCERTAIN':None}.get(value['choice']) if value else None
                event['identity_output']=value
                reason='IDENTITY_UNCERTAIN' if value else 'IDENTITY_INTERFACE_FAILURE'
            if choice is None and self.runtime.fallback=='human' and reason in {'NODE_CONTAMINATED','NODE_INSUFFICIENT'}:
                choice='KEEP_SEPARATE'
                event['fallback_reason']=reason
            if choice is None:
                choice=self.runtime.fallback_choice(event_id,directory,'merge',images,reason,[],snapshot)
                event['fallback_reason']=reason
            # Test only a negative decision. Never turn containment directly into a merge.
            if choice=='KEEP_SEPARATE':
                containment=self.containment_check(source,target)
                event['containment']=containment;save_json(directory/'containment.json',containment)
                if containment['requires_human']:
                    choice=self.runtime.human_choice(event_id,directory,['MERGE','KEEP_SEPARATE'],images,
                        '不合并与点云包含率冲突：'+json.dumps(containment,ensure_ascii=False),snapshot)
                    event['containment_human_choice']=choice
        except self.runtime.input_unavailable:
            event['status']='waiting_for_human';save_json(directory/'decision.json',event);self._summary('waiting_for_human');raise
        except Exception as exc:
            # Input problems are explicit unresolved events, never baseline merges.
            if isinstance(exc,self.runtime.invariant_error):raise
            event['input_error']=type(exc).__name__+': '+str(exc)
            save_json(directory/'decision.json',event)
            print('[v7-input-error]',event_id,event['input_error'],flush=True)
            snapshot=event.setdefault('h_snapshot_uid',states)
            if not images:
                images=[(p.stem,p) for p in directory.glob('*.jpg')]
            choice=self.runtime.fallback_choice(event_id,directory,'merge',images,'INPUT_FAILURE',[],snapshot)
            # Without valid geometric inputs a negative decision cannot pass the required check.
            if choice=='KEEP_SEPARATE':
                containment=self.containment_check(source,target)
                event['containment']=containment;save_json(directory/'containment.json',containment)
                if containment['requires_human']:
                    choice=self.runtime.human_choice(event_id,directory,['MERGE','KEEP_SEPARATE'],images,
                        '不合并与点云包含率冲突：'+json.dumps(containment,ensure_ascii=False),snapshot)
        if states!=state_key([object_state(source),object_state(target)]):raise self.runtime.invariant_error('objects changed during blocking review')
        approved,after=self.votes.record(key,frame_idx,event_id,choice)
        if approved:self.certificates[key]=states
        self.last_event[key]=event
        event.update(model_output=dict(choice=choice,confidence=0),vote_after=after,status='complete',
            execution='APPROVED_AWAITING_EXECUTION' if approved else 'LOCKED_KEEP_SEPARATE' if after['locked'] else 'DEFERRED',
            c_bound_h_snapshot_uid=event['h_snapshot_uid'])
        event['timeline'].update(c_frame=frame_idx,c_utc=_utc_now(),online_main_graph_latest_frame_at_c=frame_idx,ordering_valid=True)
        save_json(directory/'decision.json',event);_jsonl_append(self.root/'events.jsonl',event)
        self.runtime.rows.setdefault(event_id,dict(event_id=event_id,images=[])).update(event)
        self.runtime.publish();self.stats['reviewed']+=1;self._summary('ready')
        print(f'[v7-merge] {event_id} {choice} merge={after["merge_streak"]}/2 reject={after["reject_total"]}/2 {event["execution"]}',flush=True)
        return None if approved else 'v7_merge_deferred'

    def mark_executed(self,key,target_uid):
        event=self.last_event[key];event.update(execution='MERGED',executed_at=_utc_now(),target_uid=str(target_uid))
        save_json(self.root/'events'/event['event_id']/'decision.json',event)
        _jsonl_append(self.root/'executions.jsonl',event)
        self.runtime.rows[event['event_id']].update(execution='MERGED')
        self.stats['executed']+=1
    def on_merged(self,source,target):
        key=self.votes.key(source['id'],target['id'])
        if not self.votes.state(key)['awaiting_execution']:raise RuntimeError('unapproved map merge')
        self.mark_executed(key,target['id']);self.votes.merged(source['id'],target['id'])
        for o in (source,target):self.owner._support_history.pop(str(o['id']),None)
        self.runtime.publish();self._summary('ready')
