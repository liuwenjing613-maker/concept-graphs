"""Behavioral regression tests for the user-confirmed no90_new policy (no GPU/API)."""
import copy,json,tempfile,unittest,time,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx,numpy as np
from conceptgraph.slam.v7_runtime import V7Runtime,PROMPTS,parse_stage,EvidenceInvariantError
from conceptgraph.slam.v7_request_cache import EvidenceRequestCache,digest
from conceptgraph.slam.v7_endpoint_pool import EndpointPool
from conceptgraph.slam.v7_merge import V7MergeGate
from conceptgraph.slam.human_instance_merge import object_state
from conceptgraph.slam.association_gate import route_choice

def runtime(root):
    r=V7Runtime.__new__(V7Runtime);r.root=root;r.rows={};r.fallback='auto';r.forced_groups=[]
    r.owner=SimpleNamespace(output_dir=root,model='test-model',base_url='http://test',timeout_seconds=1,
                            _support_history={},events=[])
    r.owner.vlm_runtime=r
    r.templates=json.loads((PROMPTS/'request_templates.json').read_text())
    r.stage_prompts={p.stem:p.read_text() for p in PROMPTS.glob('*.txt')}
    r.endpoint_pool=EndpointPool(['http://test'],1,max_timeout_retries=0)
    r.request_cache=EvidenceRequestCache(root/'request_cache')
    r.publish=lambda:None
    r.pending=lambda eid,*args:r.rows.setdefault(eid,dict(event_id=eid))
    return r

def obj(uid):
    return dict(id=uid,pcd=SimpleNamespace(points=np.array([[0.,0,1],[1,1,1]])),
        clip_ft=np.zeros(3),obs_uids=[uid+'-h1'],image_idx=[0],num_detections=1)

class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.r=runtime(self.root);self.a=obj('a');self.b=obj('b')
        self.answer='SAME';self.quality='CLEAN';self.calls=[];self.select_last=False;self.pose=np.eye(4)
        self.fail=False;self.corrupt_during_call=False
        def visual(uid):
            n=sum(map(ord,uid))%255
            return dict(rgb=np.full((12,12,3),n,np.uint8),mask=np.ones((12,12),bool),
                        pose=self.pose,K=np.eye(3),depth=np.ones((12,12),np.float32))
        def prepare(a,b,frame):
            selected={role:[dict(uid=o['obs_uids'][-1 if self.select_last else 0])] for role,o in zip('AB',(a,b))}
            return dict(histories={k:dict(selected=v) for k,v in selected.items()},
                selected_identity_histories=copy.deepcopy(selected),objects={k:object_state(o) for k,o in zip('AB',(a,b))},
                selected_anchor=dict(alias='A',uid=selected['A'][0]['uid']),projection_source='entire_frozen_live_node')
        def render(d,a,b,frame,binding):
            for role in 'AB':
                # Node A/B caption changes when swapping roles; underlying evidence is unchanged.
                (d/f'quality_{role}.jpg').write_bytes(('node'+role+binding['histories'][role]['selected'][0]['uid']).encode())
            (d/'merge.png').write_bytes(digest(dict(geometry={k:v['pcd_sha256'] for k,v in binding['objects'].items()},histories=binding['selected_identity_histories'])).encode())
            binding.update(depth_tolerance_m=.03,h_snapshot_uid=digest(dict(binding,frame=frame)))
            return binding
        self.r.evidence=SimpleNamespace(visual=visual,prepare_merge=prepare,render_merge=render)
        self.g=V7MergeGate(self.r.owner)
        def post(url,json):
            task=next(k for k,v in self.r.templates.items() if v['format']==json['format'])
            self.calls.append(task)
            if self.fail:raise httpx.ReadTimeout('intentional test')
            if self.corrupt_during_call:self.a['pcd'].points[0,0]+=1
            if task=='node_quality':
                q=self.quality
                value=dict(choice=q,reason='fixture',views=[dict(view='H1',objects='item',
                    status={'CLEAN':'SINGLE','CONTAMINATED':'MULTIPLE','INSUFFICIENT':'UNCERTAIN'}[q],evidence='fixture')])
            else:value=dict(choice=self.answer,confidence=4,reason='fixture')
            return httpx.Response(200,request=httpx.Request('POST',url),json=dict(model='test-model',done=True,
                done_reason='stop',message=dict(content=__import__('json').dumps(value))))
        self.http=patch('httpx.Client');self.client=self.http.start()
        self.client.return_value.__enter__.return_value.post.side_effect=post

    def tearDown(self):self.http.stop();self.tmp.cleanup()
    def review(self,frame=1,a=None,b=None):
        return self.g.review(a or self.a,b or self.b,frame_idx=frame,source_frame_id=str(frame),stage='periodic')
    def event(self):return self.g.events[-1]

    def test_same_approves_first_proposal_and_records_actual_execution_generation(self):
        self.assertIsNone(self.review());self.assertEqual(self.calls,['node_quality','node_quality','merge'])
        self.assertEqual(self.event()['decision_source'],'VLM_VERIFIER')
        self.assertFalse(hasattr(self.g,'votes'))
        before=self.g.pair_key(self.a,self.b);self.g.on_merged(self.a,self.b)
        self.assertNotEqual(before,self.g.pair_key(self.a,self.b));self.assertEqual(self.event()['execution'],'MERGED')
        with self.assertRaises(EvidenceInvariantError):self.g.on_merged(self.a,self.b)

    def test_different_veto_and_identical_evidence_never_requeries_even_other_frame(self):
        self.answer='DIFFERENT';self.assertIsNotNone(self.review())
        self.assertIsNotNone(self.review(2));self.assertEqual(len(self.calls),3)
        self.assertTrue(all(x['cache_hit'] for x in self.event()['stages']))
        self.assertEqual(self.event()['execution'],'VETOED_THIS_EVIDENCE')

    def test_uncertain_preserves_baseline_merge_and_reuses_abstention(self):
        self.answer='UNCERTAIN';self.assertIsNone(self.review());self.assertIsNone(self.review(2))
        self.assertEqual(len(self.calls),3);self.assertEqual(self.event()['decision_source'],'BASELINE_DEFER')

    def test_timeout_one_attempt_per_evidence_cached_failure_never_becomes_veto(self):
        self.fail=True;self.assertIsNone(self.review());self.assertEqual(len(self.calls),2)
        self.assertIsNone(self.review(2));self.assertEqual(len(self.calls),2)
        self.assertEqual(self.event()['reason_code'],'NODE_QUALITY_INTERFACE_FAILURE')
        self.assertEqual(self.event()['decision_source'],'BASELINE_DEFER')

    def test_insufficient_skips_identity_and_preserves_baseline(self):
        self.quality='INSUFFICIENT';self.assertIsNone(self.review())
        self.assertEqual(self.calls,['node_quality','node_quality'])
        self.assertEqual(self.event()['model_output']['choice'],'DEFER')

    def test_contaminated_keeps_original_veto_without_identity(self):
        self.quality='CONTAMINATED';self.assertIsNotNone(self.review())
        self.assertEqual(self.calls,['node_quality','node_quality'])

    def test_geometry_change_reuses_node_quality_but_invalidates_identity(self):
        self.answer='DIFFERENT';self.review();self.a['pcd'].points[0,0]+=2
        self.answer='SAME';self.assertIsNone(self.review(2))
        self.assertEqual(self.calls,['node_quality','node_quality','merge','merge'])
        self.assertEqual([s['cache_hit'] for s in self.event()['stages']],[True,True,False])

    def test_unselected_new_observation_without_changed_geometry_causes_zero_calls(self):
        self.answer='DIFFERENT';self.review()
        self.a['obs_uids'].append('a-new');self.a['num_detections']+=1;self.a['image_idx'].append(1)
        self.review(2);self.assertEqual(len(self.calls),3)

    def test_selected_history_change_invalidates_only_affected_quality_and_identity(self):
        self.answer='DIFFERENT';self.review();self.a['obs_uids'].append('a-new');self.select_last=True
        self.review(2);self.assertEqual(len(self.calls),5)
        self.assertEqual([s['cache_hit'] for s in self.event()['stages']],[False,True,False])

    def test_anchor_camera_change_invalidates_identity_not_quality(self):
        self.review();self.pose[0,3]=2;self.review(2)
        self.assertEqual(self.calls,['node_quality','node_quality','merge','merge'])

    def test_role_swap_reuses_node_quality(self):
        self.answer='DIFFERENT';self.review();self.review(2,self.b,self.a)
        self.assertEqual(self.calls,['node_quality','node_quality','merge','merge'])

    def test_generation_change_invalidates_quality_even_identical_selected_history(self):
        self.review();self.g.generations['a']=1;self.review(2)
        self.assertEqual([s['cache_hit'] for s in self.event()['stages']],[False,True,False])

    def test_input_failure_preserves_baseline_but_invariant_failure_stops(self):
        self.r.evidence.prepare_merge=lambda *a:(_ for _ in ()).throw(ValueError('missing'))
        self.assertIsNone(self.review());self.assertEqual(self.event()['model_output']['choice'],'DEFER')
        self.r.evidence.prepare_merge=lambda *a:(_ for _ in ()).throw(EvidenceInvariantError('future evidence'))
        with self.assertRaises(EvidenceInvariantError):self.review(2)

    def test_mutation_during_blocking_verifier_is_rejected(self):
        self.corrupt_during_call=True
        with self.assertRaises(EvidenceInvariantError):self.review()
        self.assertFalse(self.g.pending_execution)

    def test_nonbaseline_multi_same_proposal_forbidden(self):
        with self.assertRaises(EvidenceInvariantError):
            self.g.review(self.a,self.b,frame_idx=1,source_frame_id='1',stage='multi_same')
        self.assertFalse(self.calls)

class Observation(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.r=runtime(self.root)
        self.candidates=[(a,i,obj(a)) for i,a in enumerate('ABC')]
        self.q='USABLE';self.answers=['SAME','SAME','SAME']
        self.r.stage=lambda *a,**k:dict(value=None if self.q is None else dict(status=self.q,reason='fixture'))
        self.r.stage_many=lambda e,s,h:[dict(value=dict(choice=c,confidence=5-i)) for i,c in enumerate(self.answers)]
    def tearDown(self):self.tmp.cleanup()
    def adjudicate(self):
        d=self.root/'e';d.mkdir(exist_ok=True)
        self.r.staged_bindings={'e':dict(objects=[object_state(o) for _,_,o in self.candidates],
            images=[dict(label='quality',path='quality.png')])}
        return self.r.adjudicate(d,self.candidates,'H',2,'10',association_scores=dict(A=.2,B=.9,C=.4))
    def test_multi_same_highest_mapper_score_associates_without_forced_merge(self):
        decision,out,_=self.adjudicate()
        self.assertEqual(out['choice'],'B');self.assertFalse(self.r.forced_groups)
        self.assertEqual(decision['merge_execution'],'BASELINE_PROPOSALS_ONLY_AFTER_OBSERVATION_FUSION')
        self.assertEqual(route_choice(out['choice'],dict(A=0,B=1,C=2),0)[0],1)
    def test_one_known_same_and_other_uncertain_still_associates(self):
        self.answers=['SAME','UNCERTAIN','DIFFERENT']
        self.assertEqual(self.adjudicate()[1]['choice'],'A')
    def test_all_uncertain_preserves_baseline_new_and_existing(self):
        self.answers=['UNCERTAIN']*3;out=self.adjudicate()[1]
        self.assertEqual(out['choice'],'DEFER')
        for baseline in [None,0,2]:
            self.assertEqual(route_choice(out['choice'],dict(A=0,B=1,C=2),baseline)[0],baseline)
    def test_insufficient_and_interface_failure_defer(self):
        for q in ['INSUFFICIENT',None]:
            self.q=q;self.assertEqual(self.adjudicate()[1]['choice'],'DEFER')
    def test_invalid_quality_still_discards(self):
        self.q='CORRUPTED';self.assertEqual(self.adjudicate()[1]['choice'],'DISCARD')

class CacheAndProtocol(unittest.TestCase):
    def test_singleflight_concurrent_and_durable_restart(self):
        with tempfile.TemporaryDirectory() as t:
            c=EvidenceRequestCache(t);calls=[]
            def invoke():calls.append(1);time.sleep(.02);return dict(value='SAME')
            with ThreadPoolExecutor(max_workers=8) as pool:
                result=list(pool.map(lambda _:c.execute('same',invoke),range(8)))
            self.assertEqual(len(calls),1);self.assertEqual(sum(not h for _,h in result),1)
            self.assertTrue(EvidenceRequestCache(t).execute('same',invoke)[1]);self.assertEqual(len(calls),1)
    def test_interrupted_request_is_defer_without_reissue(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'k.json').write_text('{"state":"IN_FLIGHT"}')
            r,hit=EvidenceRequestCache(t).execute('k',lambda:(_ for _ in ()).throw(AssertionError('reissue')))
            self.assertTrue(hit);self.assertIsNone(r['value'])
    def test_payload_variations_and_snapshot_binding(self):
        with tempfile.TemporaryDirectory() as t:
            r=runtime(Path(t));r.rows['e']={};im=Path(t)/'im.png';im.write_bytes(b'im')
            calls=[]
            def invoke(item):
                calls.append(1);return dict(value=dict(choice='SAME',confidence=5),error=None,
                    directory=str(item['directory'].relative_to(r.root)),h_snapshot_uid=item['snapshot'],
                    attempts=[dict(number=1)],seconds=2,timing=dict(eval_count=5))
            r._invoke_uncached=invoke
            def call(n,snap='H'):
                return r.stage('e',Path(t)/str(n),'pairwise',[('A',im)],snap)
            self.assertFalse(call(0)['cache_hit']);hit=call(1,'NEW-H')
            self.assertTrue(hit['cache_hit']);self.assertEqual(hit['c_bound_h_snapshot_uid'],'NEW-H')
            self.assertEqual(hit['cache_origin']['h_snapshot_uid'],'H');self.assertEqual(hit['actual_call_count'],0)
            self.assertEqual(hit['timing'],{});self.assertEqual(len(calls),1)
            r.stage_prompts['pairwise']+=' changed';self.assertFalse(call(2)['cache_hit'])
            r.owner.model='other';self.assertFalse(call(3)['cache_hit'])
            r.templates['pairwise']['options']['temperature']=.1;self.assertFalse(call(4)['cache_hit'])
            r.templates['pairwise']['format']['description']='version 2';self.assertFalse(call(5)['cache_hit'])
            im.write_bytes(b'changed image');self.assertFalse(call(6)['cache_hit'])
    def test_confirmed_png_reaches_annotation_packet_without_reencoding(self):
        from conceptgraph.slam.association_gate import _image_data_url,_image_media_descriptor
        from PIL import Image
        import base64,hashlib
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'quality.png';Image.new('RGB',(1298,544),(4,5,6)).save(p)
            raw=p.read_bytes();url=_image_data_url(p);meta=_image_media_descriptor(p)
            self.assertTrue(url.startswith('data:image/png;base64,'))
            self.assertEqual(base64.b64decode(url.split(',')[1]),raw)
            self.assertEqual(meta['mime_type'],'image/png')
            self.assertEqual(meta['payload_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertIsNone(meta['jpeg_magic_hex'])
            p.write_bytes(b'<svg></svg>')
            with self.assertRaises(ValueError):_image_data_url(p)

    def test_quality_exact_new_schema_raw_preserved_and_length_enforced(self):
        x=dict(quality='INVALID',confidence=4,reason='multiple instances',error_type='mixed_objects')
        value,_,raw=parse_stage(json.dumps(x),'observation_quality')
        self.assertEqual(value['status'],'CORRUPTED');self.assertEqual(raw,x)
        x['reason']='\u4e2d'*21
        with self.assertRaises(ValueError):parse_stage(json.dumps(x),'observation_quality')

if __name__=='__main__':unittest.main()
