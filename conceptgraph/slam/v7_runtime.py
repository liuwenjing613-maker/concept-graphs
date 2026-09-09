"""Staged online VLM decisions and explicit human/auto fallback."""
import base64,copy,hashlib,html,json,os,re,time
from pathlib import Path
from itertools import combinations
import httpx
from concurrent.futures import ThreadPoolExecutor
from conceptgraph.slam.v7_endpoint_pool import EndpointPool
import numpy as np
from conceptgraph.slam.v7_errors import EvidenceInvariantError
from conceptgraph.slam.vlm_runtime import VLMRuntime,save_json,sha
from conceptgraph.slam.staged_event_decision import DECODER,DuplicateKey,schema_valid,decide_event
from conceptgraph.slam.human_instance_merge import object_state,state_key
from conceptgraph.slam.association_gate import HumanInputUnavailableError,_utc_now,_jsonl_append
from conceptgraph.slam.v7_evidence import LiveEvidence

PROMPTS=Path(__file__).with_name('prompts')/'v7'

def parse_stage(content,task,labels=None):
    def decode(t):
        o,end=DECODER.raw_decode(t.lstrip())
        if t.lstrip()[end:].strip():raise ValueError('trailing text')
        return o
    mode='STRICT_JSON'
    try:value=decode(content)
    except DuplicateKey:raise
    except ValueError:
        fence=re.fullmatch(r'\s*`{3}(?:json)?\s*(.*?)\s*`{3}\s*',content,re.S)
        if fence:value=decode(fence.group(1));mode='FENCED_JSON'
        else:
            objects=[];end=0
            for m in re.finditer(r'[\{\[]',content):
                if m.start()<end:continue
                try:o,n=DECODER.raw_decode(content[m.start():]);end=m.start()+n;objects.append(o)
                except DuplicateKey:raise
                except ValueError:continue
            if len(objects)!=1:raise ValueError('not exactly one JSON value')
            value=objects[0];mode='EXTRACTED_UNIQUE_JSON'
    if not isinstance(value,dict):raise ValueError('expected object')
    original=copy.deepcopy(value)
    if task=='merge' and isinstance(value.get('confidence'),str) and re.fullmatch('[0-5](?:\\.0)?',value['confidence']):
        value['confidence']=int(float(value['confidence']));mode+='_NUMERIC_STRING'
    if task in ('observation_quality','pairwise'):
        if not schema_valid(value,'quality' if task=='observation_quality' else 'pairwise'):
            raise ValueError('invalid stage schema')
    elif task=='merge':
        if set(value)!={'choice','confidence','reason'} or value['choice'] not in {'SAME','DIFFERENT','UNCERTAIN'}:
            raise ValueError('invalid merge fields')
        if type(value['confidence']) is not int or not 0<=value['confidence']<=5 or not isinstance(value['reason'],str) or not 1<=len(value['reason'])<=240:
            raise ValueError('invalid merge values')
    elif task=='node_quality':
        if set(value)!={'views','choice','reason'} or value['choice'] not in {'CLEAN','CONTAMINATED','INSUFFICIENT'}:
            raise ValueError('invalid node quality fields')
        if not isinstance(value['views'],list) or [v.get('view') for v in value['views']]!=labels:
            raise ValueError('missing or reordered history labels')
        for v in value['views']:
            if set(v)!={'view','objects','status','evidence'} or v['status'] not in {'SINGLE','MULTIPLE','UNCERTAIN'}:
                raise ValueError('invalid history view')
            if any(not isinstance(v[k],str) or not 1<=len(v[k])<=240 for k in ('objects','evidence')):
                raise ValueError('invalid history text')
        if not isinstance(value['reason'],str) or not 1<=len(value['reason'])<=240:raise ValueError('invalid node reason')
        if value['choice']=='CLEAN' and any(v['status']!='SINGLE' for v in value['views']):
            raise ValueError('CLEAN contradicts history status')
        if any(v['status']=='MULTIPLE' for v in value['views']) and value['choice']!='CONTAMINATED':
            raise ValueError('MULTIPLE contradicts node status')
    else:raise ValueError('unknown stage')
    return value,mode,original

class V7Runtime(VLMRuntime):
    input_unavailable=HumanInputUnavailableError
    invariant_error=EvidenceInvariantError
    def __init__(self,owner):
        import PIL,cv2
        from PIL import features
        if PIL.__version__!='11.0.0' or not features.check('raqm') or cv2.__version__!='4.12.0':
            raise RuntimeError('Validated renderers require Pillow 11 + RAQM and OpenCV 4.12; run setup_v7.sh')
        self.fallback=os.environ.get('V7_FALLBACK','auto')
        if self.fallback not in {'human','auto'}:raise ValueError('V7_FALLBACK must be human or auto')
        self.containment_distance=owner.v7_containment_distance
        self.templates=json.loads((PROMPTS/'request_templates.json').read_text())
        self.stage_prompts={p.stem:p.read_text() for p in PROMPTS.glob('*.txt')}
        self.forced_groups=[];self.staged_bindings={}
        self.endpoint_pool=EndpointPool(json.loads(os.environ.get('V7_VLM_URLS', json.dumps([owner.base_url]))), owner.timeout_seconds)
        self.owner=owner;self.root=owner.output_dir;self.rows={};self.projection_frames={}
        self.status='running';self.prompts={}
        self.root.joinpath('review').mkdir(exist_ok=True)
        self.evidence=LiveEvidence(self)
        self.versions=dict(version='v7_merge',model=owner.model,fallback=self.fallback,
            execution_revision='20260909_containment_direct_merge',
            endpoints=self.endpoint_pool.urls,max_parallel=len(self.endpoint_pool.urls),timeout_retries=3,
            timeout_failure_after=4,
            prompt_sha256={p.stem:sha(p) for p in PROMPTS.glob('*.txt')},templates_sha256=sha(PROMPTS/'request_templates.json'),
            renderer_dependencies=dict(pillow=PIL.__version__,opencv=cv2.__version__,raqm=True),
            renderer='focused-fivepanel + full-RGB-node-audit + target-RGB-history/RGB-projection/zoom',
            merge_required_consecutive=2,reject_required_total=2,containment_distance_m=self.containment_distance,
            containment_threshold=.9,auto_requires_human=False,
            containment_trigger='KEEP_SEPARATE only; either full-cloud direction >90% directly approves merge, bypassing VLM votes')
        save_json(self.root/'vlm_versions.json',self.versions)
        template=Path(__file__).with_name('v7_dashboard.html')
        for dest in [self.root/'index.html',self.root/'review/index.html']:dest.write_text(template.read_text())
        self.publish()
    def prompts_for(self,kind,aliases):
        return 'V7 staged execution: observation quality, then each frozen candidate identity.', 'Runtime dispatches separate versioned stage requests.'
    def event_evidence(self,event_dir,rgb,detection,candidates):
        frame=int(re.search(r'f(\d+)',event_dir.name).group(1))
        try:
            binding,images=self.evidence.observation(event_dir,rgb,detection,candidates,frame)
        except EvidenceInvariantError:raise
        except Exception as exc:
            images=[(p.stem,p) for p in sorted(event_dir.glob('*.jpg'))]
            np.savez_compressed(event_dir/'live_points.npz',current=np.asarray(detection['pcd'].points),
                **{a:np.asarray(o['pcd'].points) for a,_,o in candidates})
            binding=dict(objects=[object_state(o) for _,_,o in candidates],h_frame=frame,
                input_error=type(exc).__name__+': '+str(exc),
                images=[dict(label=n,path=p.name,sha256=sha(p)) for n,p in images])
            save_json(event_dir/'staged_evidence.json',binding)
        self.staged_bindings[event_dir.name]=binding
        return binding['images'],images
    def bind_projection_snapshot(self,event_dir,snapshot_uid,frame_idx):
        binding=self.staged_bindings[event_dir.name]
        binding.update(h_snapshot_uid=snapshot_uid,h_frame=frame_idx,online_main_graph_latest_frame_at_h=frame_idx)
        save_json(event_dir/'staged_evidence.json',binding)
    def _pool(self):
        if not hasattr(self,'endpoint_pool'):
            self.endpoint_pool=EndpointPool([self.owner.base_url],self.owner.timeout_seconds)
        return self.endpoint_pool

    def stage(self,event_id,directory,task,images,snapshot,labels=None):
        return self.stage_many(event_id,[(directory,task,images,labels)],snapshot)[0]

    def stage_many(self,event_id,specs,snapshot):
        # Prepare and publish on the mapping thread. Workers only touch their own files.
        if not specs:return []
        prepared=[self._prepare_stage(event_id,*spec[:3],snapshot,spec[3]) for spec in specs]
        row=self.rows[event_id]
        row.update(status='waiting_for_vlm',current_stage=','.join(spec[1] for spec in specs))
        self.publish()
        with ThreadPoolExecutor(max_workers=min(3,len(self._pool().urls),len(specs))) as executor:
            futures=[executor.submit(self._invoke_stage,item) for item in prepared]
            # Commit once all calls/retries finish, in candidate order, never completion order.
            results=[future.result() for future in futures]
        for result in results:
            save_json(self.root/result['directory']/'result.json',result)
            row.setdefault('stage_results',[]).append(result)
            print('[v7-stage]',event_id,result['task'],'DONE',result['value'] or result['error'],flush=True)
        row['status']='stage_complete';self.publish()
        return results

    def _prepare_stage(self,event_id,directory,task,images,snapshot,labels=None):
        directory.mkdir(parents=True,exist_ok=False)
        payload=copy.deepcopy(self.templates[task]);payload['model']=self.owner.model
        payload['messages'][0]['content']=self.stage_prompts[task]
        if task=='pairwise':
            alias=images[0][0]
            if alias not in {'A','B','C'}:raise ValueError('pairwise request needs frozen candidate alias')
            payload['messages'][1]['content']=re.sub(r'CANDIDATE [ABC]', 'CANDIDATE '+alias,
                payload['messages'][1]['content'])
        payload['messages'][1]['images']=[base64.b64encode(p.read_bytes()).decode() for _,p in images]
        request=copy.deepcopy(payload)
        request['messages'][1]['images']=[dict(label=n,path=str(p.relative_to(self.root)),sha256=sha(p)) for n,p in images]
        save_json(directory/'request.json',request)
        save_json(directory/'input_manifest.json',dict(h_snapshot_uid=snapshot,task=task,labels=labels,
            images=request['messages'][1]['images'],prompt_sha256=hashlib.sha256(self.stage_prompts[task].encode()).hexdigest()))
        print('[v7-stage]',event_id,task,'START',directory.name,flush=True)
        return dict(directory=directory,task=task,snapshot=snapshot,labels=labels,payload=payload,request=request)

    def _invoke_stage(self,item):
        directory,task,snapshot=item['directory'],item['task'],item['snapshot']
        started=time.perf_counter()
        result=dict(task=task,value=None,format_mode=None,error=None,h_snapshot_uid=snapshot,
                    directory=str(directory.relative_to(self.root)),images=item['request']['messages'][1]['images'])
        def record(attempt,response):
            attempt_dir=directory/'attempts'/f"{attempt['number']:02d}"
            attempt_dir.mkdir(parents=True,exist_ok=False)
            save_json(attempt_dir/'attempt.json',dict(attempt,h_snapshot_uid=snapshot))
            if response is not None:
                try:save_json(attempt_dir/'response.json',response.json())
                except (ValueError,TypeError):pass
        try:
            response,attempts,error=self._pool().request(item['payload'],record)
            result['attempts']=attempts
            result['timeout_count']=sum(a['timed_out'] for a in attempts)
            if error:raise RuntimeError(error)
            result['http_status']=response.status_code
            raw=response.json();save_json(directory/'response.json',raw)
            result['raw_output']=raw.get('message',{}).get('content','')
            result['timing']={k:raw.get(k) for k in ['total_duration','prompt_eval_count','eval_count','eval_duration']}
            if raw.get('model')!=self.owner.model or not raw.get('done') or raw.get('done_reason')!='stop':
                raise ValueError('wrong model or incomplete response')
            value,mode,original=parse_stage(result['raw_output'],task,item['labels'])
            result.update(value=value,format_mode=mode,parsed_before_compatibility=original)
        except Exception as exc:
            result['error']=type(exc).__name__+': '+str(exc)
        result.update(seconds=time.perf_counter()-started,c_bound_h_snapshot_uid=snapshot,completed_at=_utc_now())
        return result
    def human_choice(self,event_id,directory,allowed,images,reason,snapshot):
        # Auto must never read stdin, including a containment-conflict call path.
        if self.fallback=='auto':
            task='merge' if 'KEEP_SEPARATE' in allowed else 'observation'
            return self.fallback_choice(event_id,directory,task,images,reason,allowed,snapshot)
        token=(event_id+'-'+snapshot[:10]).upper()
        question=dict(event_id=event_id,token=token,allowed=allowed,reason=reason,h_snapshot_uid=snapshot,
            images=[dict(label=n,path=str(p.relative_to(self.root))) for n,p in images],state='waiting')
        save_json(directory/'human_question.json',question)
        self.rows.setdefault(event_id,dict(event_id=event_id,images=[])).update(status='waiting_for_human',human_question=question)
        self.publish()
        print('[v7-human] 建图暂停；复核页:',self.root/'review/index.html',flush=True)
        print('[v7-human]',reason,'\n输入:',token,'选项:',','.join(allowed),flush=True)
        while True:
            try:answer=self.owner._human_input('[v7-human] TOKEN CHOICE: ').strip().split()
            except EOFError as exc:
                raise HumanInputUnavailableError('v7 human review needs interactive stdin; evidence saved at '+str(directory)) from exc
            if len(answer)==2 and answer[0].upper()==token and answer[1].upper() in allowed:
                choice=answer[1].upper();break
            print('[v7-human] 编号或选项无效；请从当前题复制答案。',flush=True)
        question.update(state='answered',choice=choice,answered_at=_utc_now(),c_bound_h_snapshot_uid=snapshot)
        save_json(directory/'human_answer.json',question);self.rows[event_id]['human_question']=question
        self.publish();return choice
    def fallback_choice(self,event_id,directory,task,images,reason,aliases,snapshot):
        if self.fallback=='human':
            return self.human_choice(event_id,directory,['MERGE','KEEP_SEPARATE'] if task=='merge' else [*aliases,'NEW','DISCARD'],
                images,reason,snapshot)
        choice='KEEP_SEPARATE' if task=='merge' else 'DISCARD'
        save_json(directory/'auto_fallback.json',dict(reason=reason,choice=choice,h_snapshot_uid=snapshot))
        return choice
    def adjudicate(self,event_dir,candidates,snapshot,frame,source_frame):
        started=time.perf_counter();eid=event_dir.name;binding=self.staged_bindings[eid]
        images=[(i['label'],event_dir/i['path']) for i in binding['images']]
        self.pending(eid,event_dir,'observation',images,[a for a,_,_ in candidates]+['NEW','DISCARD'])
        self.rows[eid].update(h_snapshot_uid=snapshot,trigger_kind=eid.rsplit('_',1)[-1])
        quality=(dict(value=None,error=binding['input_error']) if binding.get('input_error') else
            self.stage(eid,event_dir/'quality','observation_quality',[images[0]],snapshot))
        pairs=[]
        if quality['value'] and quality['value']['status']=='USABLE':
            specs=[(event_dir/('candidate_'+a),'pairwise',[(a,event_dir/f'candidate_{a}.jpg')],None) for a,_,_ in candidates]
            stages=self.stage_many(eid,specs,snapshot)
            pairs=[dict(alias=a,value=stage['value']) for (a,_,_),stage in zip(candidates,stages)]
        decision=decide_event(quality['value'],pairs,[a for a,_,_ in candidates],pool_complete=True)
        if binding.get('input_error'):decision['reason_code']='INPUT_FAILURE: '+binding['input_error']
        if decision['kind']=='ASSOCIATE':choice=decision['target_alias']
        elif decision['kind']=='NEW':choice='NEW'
        elif decision['kind']=='MERGE_REVIEW':
            byalias={a:(idx,obj) for a,idx,obj in candidates};same=decision['same_aliases']
            gate=self.merge_gate(frame,source_frame)
            all_approved=True;pair_keys=[]
            for a,b in combinations(same,2):
                first,second=byalias[a][1],byalias[b][1]
                result=gate.review(first,second,frame_idx=frame,source_frame_id=source_frame,stage='multi_same',
                    parent_event=eid)
                all_approved &= result is None;pair_keys.append(gate.votes.key(first['id'],second['id']))
            if all_approved:
                choice=same[0]
                group=dict(parent_event=eid,objects=[byalias[a][1] for a in same],keys=pair_keys,
                    state=state_key([object_state(byalias[a][1]) for a in same]),h_snapshot_uid=snapshot)
                uids={str(o['id']) for o in group['objects']}
                overlaps=[g for g in self.forced_groups if uids & {str(o['id']) for o in g['objects']}]
                if any(uids!={str(o['id']) for o in g['objects']} for g in overlaps):
                    decision['reason_code']='OVERLAPPING_GROUP_REQUIRES_NEW_SNAPSHOT'
                    choice=self.fallback_choice(eid,event_dir,'observation',images,decision['reason_code'],
                        [a for a,_,_ in candidates],snapshot)
                else:
                    if not overlaps:self.forced_groups.append(group)
                    decision['merge_execution']='QUEUED_BEFORE_OBSERVATION_FUSION'
            else:
                decision['reason_code']='MULTIPLE_SAME_AWAITING_PAIR_APPROVAL'
                choice=self.fallback_choice(eid,event_dir,'observation',images,decision['reason_code'],[a for a,_,_ in candidates],snapshot)
        else:
            choice=self.fallback_choice(eid,event_dir,'observation',images,decision['reason_code'],[a for a,_,_ in candidates],snapshot)
        if binding['objects']!=[object_state(o) for _,_,o in candidates]:raise EvidenceInvariantError('candidate changed during blocking event')
        decision.update(choice=choice,h_snapshot_uid=snapshot,c_bound_h_snapshot_uid=snapshot,
            timeline=dict(s_frame=frame,d_frame=frame,h_frame=frame,c_frame=frame,online_main_graph_latest_frame_at_c=frame))
        save_json(event_dir/'staged_decision.json',decision)
        self.rows[eid]['staged_decision']=decision
        return decision,dict(choice=choice,confidence=0),time.perf_counter()-started
    def merge_gate(self,frame,source_frame):
        self.owner.object_merge_reviewer(frame_idx=frame,source_frame_id=source_frame,stage='multi_same')
        return self.owner._instance_merge_gate
    def flush_groups(self,objects,matches,cfg,evidence,frame,map_edges):
        if not self.forced_groups:return objects,matches
        from conceptgraph.slam.utils import merge_obj2_into_obj1
        if cfg.get('make_edges'):raise EvidenceInvariantError('v7 forced merges require make_edges=false')
        groups,self.forced_groups=self.forced_groups,[]
        old_uids=[str(o['id']) for o in objects];redirect={};removed=set()
        gate=self.owner._instance_merge_gate
        for group in groups:
            members=group['objects'];uids=[str(o['id']) for o in members]
            if any(u in removed or u in redirect.values() for u in uids) or group['state']!=state_key([object_state(o) for o in members]):
                raise EvidenceInvariantError('queued merge group changed before execution')
            # Complete pairwise approval is required. No transitive inference from just A-B and B-C.
            if len(group['keys'])!=len(members)*(len(members)-1)//2 or any(not gate.votes.state(k)['awaiting_execution'] for k in group['keys']):
                raise EvidenceInvariantError('incomplete clique merge certificate')
            dest=members[0];target_uid=str(dest['id'])
            for source in members[1:]:
                source_uid=str(source['id'])
                dest=merge_obj2_into_obj1(dest,source,cfg['downsample_voxel_size'],cfg['dbscan_remove_noise'],
                    cfg['dbscan_eps'],cfg['dbscan_min_points'],cfg['spatial_sim_type'],cfg['device'],run_dbscan=True)
                objects[old_uids.index(target_uid)]=dest
                evidence.record_object_merge(frame_idx=frame,source_object=source,target_object=dest,
                    overlap_ratio=0.,visual_similarity=0.,text_similarity=0.)
                removed.add(source_uid);redirect[source_uid]=target_uid
            for key in group['keys']:gate.mark_executed(key,target_uid)
            for uid in uids:
                gate.votes.generations[uid]=gate.votes.generations.get(uid,0)+1
                self.owner._support_history.pop(uid,None)
            save_json(self.root/'events'/group['parent_event']/'merge_execution.json',dict(status='MERGED',
                source_uids=uids,target_uid=target_uid,h_snapshot_uid=group['h_snapshot_uid'],frame_idx=frame,
                policy='every unordered pair approved by two VLM votes or >90% containment before any component mutation'))
        kept=[o for o in objects if str(o['id']) not in removed]
        new_index={str(o['id']):i for i,o in enumerate(kept)}
        def remap(idx):
            if idx is None or idx<0:return idx
            uid=old_uids[idx]
            while uid in redirect:uid=redirect[uid]
            return new_index[uid]
        updated=[remap(i) for i in matches]
        # Launcher disables semantic edges; reject unsupported direct custom invocation explicitly.
        if cfg.get('make_edges') and removed:raise EvidenceInvariantError('v7 forced merges require make_edges=false')
        objects[:]=kept
        for event in self.owner.events:
            if event['timeline']['h_frame']==frame:
                event['pre_merge_match_index']=event['final_match_index']
                event['final_match_index']=remap(event['final_match_index'])
                event['candidate_indices_are_h_snapshot_indices']=True
                save_json(self.root/'events'/event['event_id']/'decision.json',event)
        self.publish();gate._summary('ready')
        return objects,updated
