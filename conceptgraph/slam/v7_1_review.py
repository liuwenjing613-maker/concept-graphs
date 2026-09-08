"""Bounded joint observation review. Never approves or executes a node merge."""
import hashlib,json
from conceptgraph.slam import v7_render as render
from conceptgraph.slam.v7_errors import EvidenceInvariantError
from conceptgraph.slam.human_instance_merge import object_state
from conceptgraph.slam.vlm_runtime import save_json,sha


def validate_joint(value,aliases):
    if not isinstance(value,dict) or set(value)!={'current_status','choice','candidates','reason'}:
        raise ValueError('invalid joint fields')
    if value['current_status'] not in {'USABLE','CORRUPTED','INSUFFICIENT'}:
        raise ValueError('invalid current status')
    if value['choice'] not in set(aliases)|{'UNRESOLVED'}:raise ValueError('invalid joint choice')
    rows=value['candidates']
    if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows):raise ValueError('invalid candidates')
    if [r.get('alias') for r in rows]!=aliases:raise ValueError('missing/reordered/duplicate joint aliases')
    for r in rows:
        if set(r)!={'alias','relation','node_quality','evidence'}:raise ValueError('invalid candidate fields')
        if r['relation'] not in {'SAME','SAME_OBJECT_PART','DIFFERENT','UNCERTAIN'} or r['node_quality'] not in {'CLEAN','CONTAMINATED','INSUFFICIENT'}:
            raise ValueError('invalid relation/quality')
        if not isinstance(r['evidence'],str) or not 1<=len(r['evidence'])<=240:raise ValueError('invalid evidence')
    if not isinstance(value['reason'],str) or not 1<=len(value['reason'])<=240:raise ValueError('invalid reason')
    if value['choice']!='UNRESOLVED':
        chosen=next(r for r in rows if r['alias']==value['choice'])
        if value['current_status']!='USABLE' or chosen['relation'] not in {'SAME','SAME_OBJECT_PART'} or chosen['node_quality']!='CLEAN':
            raise ValueError('joint selected owner is not usable/same/clean')
        if any(r['relation']!='DIFFERENT' for r in rows if r is not chosen):
            raise ValueError('joint owner is not unique')


def unique_owner(value,aliases,original_same,protected_pairs=()):
    if value is None:return None
    validate_joint(value,aliases)
    if any(value['choice'] in pair for pair in protected_pairs):return None
    return value['choice'] if value['choice'] in original_same else None


def joint_review(runtime,eid,event_dir,candidates,binding,parent_snapshot,original_same,protected_pairs=()):
    folder=event_dir/'joint_evidence';folder.mkdir(exist_ok=False)
    audit=dict(parent_h_snapshot_uid=parent_snapshot,h_frame=binding['h_frame'],
        objects=binding['objects'],original_same_aliases=original_same,history_selection={})
    aliases=[a for a,_,_ in candidates]
    audit['protected_positive_pairs']=protected_pairs
    if len(protected_pairs)==len(original_same)*(len(original_same)-1)//2:
        outcome=dict(choice='UNRESOLVED',reason='POSITIVE_PAIR_CONFIRMATIONS_PENDING',parent_h_snapshot_uid=parent_snapshot,protected_positive_pairs=protected_pairs)
        save_json(folder/'outcome.json',outcome)
        return None,outcome
    try:
        if binding['objects']!=[object_state(o) for _,_,o in candidates]:
            raise EvidenceInvariantError('objects changed before joint review')
        images=[]
        for ref in binding['images']:
            path=event_dir/ref['path']
            if sha(path)!=ref['sha256']:raise EvidenceInvariantError('joint parent image hash mismatch')
            images.append((ref['label'],path))
        for alias,_,obj in candidates:
            uids=runtime.evidence.members(obj,binding['h_frame'])
            ranking,chosen=render.history_scores(uids,runtime.evidence.observations,runtime.evidence.array)
            path=folder/f'history_{alias}.jpg'
            render.node_card(alias,chosen,path,runtime.evidence.visual)
            images.append(('CANDIDATE '+alias+' FULL NODE HISTORY AUDIT',path))
            audit['history_selection'][alias]=dict(ranking=ranking,selected=chosen,
                mask_sha256={r['uid']:runtime.evidence.observations[r['uid']]['processed_mask_ref']['sha256'] for r in chosen})
        audit['images']=[dict(label=n,path=str(p.relative_to(runtime.root)),sha256=sha(p)) for n,p in images]
        snapshot=hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest()
        audit['h_snapshot_uid']=snapshot;save_json(folder/'input_manifest.json',audit)
        result=runtime.stage(eid,event_dir/'joint_review','joint_review',images,snapshot,aliases)
        if binding['objects']!=[object_state(o) for _,_,o in candidates]:
            raise EvidenceInvariantError('objects changed during joint review')
        chosen=unique_owner(result['value'],aliases,original_same,protected_pairs)
        outcome=dict(choice=chosen or 'UNRESOLVED',h_snapshot_uid=snapshot,parent_h_snapshot_uid=parent_snapshot,
            c_bound_h_snapshot_uid=snapshot,stage=result,applies_to='observation_only',merge_vote_added=False)
    except EvidenceInvariantError:raise
    except Exception as exc:
        outcome=dict(choice='UNRESOLVED',parent_h_snapshot_uid=parent_snapshot,error=type(exc).__name__+': '+str(exc))
        chosen=None
    save_json(folder/'outcome.json',outcome)
    return chosen,outcome
