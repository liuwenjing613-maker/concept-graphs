from pathlib import Path
import json,sys,collections,hashlib
P=Path(__file__).parent;R=Path(sys.argv[1]);frames=int(sys.argv[2]);out=Path(sys.argv[3]);read=lambda p:json.loads(p.read_text())
e=read(R/'evidence/evidence_summary.json');a=read(R/'audit/audit_summary.json')
assert e['num_frames']==frames and a['gate_status']=='PASS'
assert e['missing_reference_count']==e['duplicate_membership_occurrences']==e['logging_error_count']==0
G=R/'blocking_association_gate';merges=[read(p) for p in sorted((G/'vlm_instance_merge/events').glob('*/decision.json'))];seen=set();execution=collections.Counter();threshold=collections.Counter()
for m in merges:
 t=m['timeline'];assert t['s_frame']<=t['d_frame']<=t['h_frame']<=t['c_frame'];assert m['c_bound_h_snapshot_uid']==m['h_snapshot_uid'];execution[m['execution']]+=1
 if m['status']=='locked':continue
 k=(m['pair_key'],m['frame_idx']);assert k not in seen;seen.add(k)
 q=m.get('quality_outputs',[]);identity=m.get('identity_output') or {};choice=m['model_output']['choice'];req=m.get('merge_required',2);threshold[req]+=1
 if any(v and v['choice']=='CONTAMINATED' for v in q):assert choice=='KEEP_SEPARATE' and not identity
 if m.get('quality_interface_failure'):assert choice in ('DEFER','KEEP_SEPARATE') and m['execution']!='MERGED'
 if identity.get('choice')=='UNCERTAIN':assert choice=='DEFER'
 if choice=='MERGE':
  assert identity['choice']=='SAME' and not m['quality_interface_failure'];assert req==(1 if all(v['choice']=='CLEAN' for v in q) else 2)
 if m['execution'] in ['MERGED','APPROVED_AWAITING_EXECUTION']:assert m['vote_after']['merge_streak']>=req
qualities=[];calls=0;confirmations=0
for p in G.rglob('result.json'):
 r=read(p)
 if 'quality_policy' not in r:continue
 qualities.append(r);assert r['c_bound_h_snapshot_uid']==r['h_snapshot_uid'];v=r['value'];f='status' if r['task']=='observation_quality' else 'choice';neg='CORRUPTED' if f=='status' else 'CONTAMINATED'
 if v[f]==neg:assert r['primary']['value'][f]==neg and r['confirmation']['value'][f]==neg
 if r.get('cache_hit'):assert r['cache_source'];assert not r['attempts']
 if r.get('interface_failure'):assert v[f]=='INSUFFICIENT'
 if r.get('confirmation') and not r.get('cache_hit'):confirmations+=1
# HTTP attempts are exact physical records, independent of duplicated embedded cache provenance.
calls=len(list(G.rglob('attempts/*/attempt.json')))
for p in (G/'events').glob('*/staged_decision.json'):
 d=read(p);t=d['timeline'];assert t['s_frame']<=t['d_frame']<=t['h_frame']<=t['c_frame'];assert d['h_snapshot_uid']==d['c_bound_h_snapshot_uid']
 if d.get('kind')=='DISCARD':assert read(p.parent/'quality/result.json')['value']['status']=='CORRUPTED'
 if d.get('reason_code')=='MULTIPLE_SAME_UNIQUE_ANCHOR':assert d['kind']=='ASSOCIATE' and d['target_alias'] in 'ABC'
x=dict(status='PASS',frames=frames,strict_audit=a['gate_status'],semantic_findings_not_visual_ground_truth=a['finding_count'],merges=dict(execution),vote_threshold_events=dict(threshold),quality_stages=len(qualities),quality_cache_hits=sum(r.get('cache_hit',False) for r in qualities),negative_confirmations=confirmations,http_attempts=calls,zero_evidence_logging_errors=True,run=str(R))
out.write_text(json.dumps(x,ensure_ascii=False,indent=2));print(json.dumps(x,ensure_ascii=False))
