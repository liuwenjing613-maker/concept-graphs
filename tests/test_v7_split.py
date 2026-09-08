"""Small server smoke for v7 decisions, voting and mutation boundaries."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from conceptgraph.slam.v7_runtime import parse_stage,V7Runtime,EvidenceInvariantError
from conceptgraph.slam.v7_merge import V7Votes,V7MergeGate
from conceptgraph.slam.v7_evidence import LiveEvidence
from conceptgraph.slam.human_instance_merge import object_state,state_key
from conceptgraph.slam.association_gate import HumanInputUnavailableError
from conceptgraph.slam.staged_event_decision import decide_event

def obj(uid,offset=0):
    points=np.column_stack((np.arange(10,dtype=float)+offset,np.zeros(10),np.zeros(10)))
    return dict(id=uid,pcd=SimpleNamespace(points=points),clip_ft=np.zeros(3),
        obs_uids=[uid+'-h1'],image_idx=[0],num_detections=1)

class Voting(unittest.TestCase):
    def test_two_consecutive(self):
        v=V7Votes();k=v.key('a','b')
        self.assertFalse(v.record(k,1,'e1','MERGE')[0]);self.assertTrue(v.record(k,2,'e2','MERGE')[0])
    def test_reject_total_not_streak(self):
        v=V7Votes();k=v.key('a','b')
        for f,c in enumerate(['KEEP_SEPARATE','MERGE','KEEP_SEPARATE'],1):v.record(k,f,str(f),c)
        self.assertTrue(v.state(k)['locked']);self.assertEqual(v.state(k)['merge_streak'],0)
    def test_same_frame_once(self):
        v=V7Votes();k=v.key('a','b');self.assertEqual(k,v.key('b','a'));v.record(k,1,'e','MERGE')
        with self.assertRaises(ValueError):v.record(k,1,'e2','MERGE')
    def test_history_unlock_and_geometry_does_not(self):
        v=V7Votes();k=v.key('a','b');v.refresh_history(k,'h1')
        v.record(k,1,'e','KEEP_SEPARATE');v.record(k,2,'e2','KEEP_SEPARATE')
        v.refresh_history(k,'h1');self.assertTrue(v.state(k)['locked'])
        v.refresh_history(k,'h2');self.assertFalse(v.state(k)['locked']);self.assertEqual(v.state(k)['reject_total'],0)
    def test_real_third_party_merge_new_identity(self):
        v=V7Votes();k=v.key('a','b');v.record(k,1,'e','KEEP_SEPARATE');v.record(k,2,'e2','KEEP_SEPARATE')
        v.merged('a','c');self.assertNotEqual(k,v.key('a','b'));self.assertFalse(v.state(v.key('a','b'))['locked'])

class Parsing(unittest.TestCase):
    def test_fenced_and_unique_json(self):
        for text in ['```json\n{"choice":"SAME","confidence":5}\n```','thinking then {"choice":"SAME","confidence":5}']:
            self.assertEqual(parse_stage(text,'pairwise')[0]['choice'],'SAME')
    def test_duplicate_multiple_truncated(self):
        for text in ['{"choice":"SAME","choice":"DIFFERENT","confidence":5}',
            '{"choice":"SAME","confidence":5} {"choice":"DIFFERENT","confidence":5}','{"choice":"SAME",']:
            with self.subTest(text=text),self.assertRaises(ValueError):parse_stage(text,'pairwise')
    def test_merge_numeric_string_only_compatibility(self):
        value,mode,original=parse_stage('{"choice":"SAME","confidence":"5","reason":"同一实例"}','merge')
        self.assertEqual(value['confidence'],5);self.assertEqual(original['confidence'],'5')
        for invalid in ['true','6','"high"']:
            with self.assertRaises(ValueError):parse_stage('{"choice":"SAME","confidence":'+invalid+',"reason":"x"}','merge')
    def test_contradictory_quality_not_usable(self):
        x=dict(views=[dict(view='H1',objects='x',status='MULTIPLE',evidence='x')],choice='CLEAN',reason='x')
        with self.assertRaises(ValueError):parse_stage(json.dumps(x),'node_quality',['H1'])
    def test_quality_failure_blocks_candidates(self):
        self.assertEqual(decide_event(dict(status='INSUFFICIENT',reason='x'),[],['A'])['kind'],'PENDING')
    def test_multiple_same_requires_all_pairs(self):
        p=[dict(alias=a,value=dict(choice='SAME',confidence=0)) for a in 'ABC']
        self.assertEqual(decide_event(dict(status='USABLE',reason='x'),p,list('ABC'))['same_aliases'],list('ABC'))

class Geometry(unittest.TestCase):
    def test_asymmetric_containment(self):
        a,b=obj('a'),obj('b');b['pcd'].points=np.r_[b['pcd'].points,[[30,0,0]]]
        x=LiveEvidence.containment(a,b,.025)
        self.assertEqual(x['a_in_b'],1);self.assertAlmostEqual(x['b_in_a'],10/11);self.assertTrue(x['requires_human'])
    def test_strict_90_percent(self):
        a,b=obj('a'),obj('b');b['pcd'].points[-1]=[99,0,0]
        x=LiveEvidence.containment(a,b,.025)
        self.assertEqual(x['a_in_b'],.9);self.assertEqual(x['b_in_a'],.9);self.assertFalse(x['requires_human'])
    def test_no_mask_projection_or_plot_sampling(self):
        self.assertFalse(LiveEvidence.containment(obj('a'),obj('b',100),.025)['requires_human'])

class GateIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.humans=[];self.calls=[]
        self.owner=SimpleNamespace(output_dir=self.root,_support_history={},events=[])
        runtime=SimpleNamespace(root=self.root,rows={},fallback='auto',input_unavailable=HumanInputUnavailableError,
            invariant_error=EvidenceInvariantError,containment_distance=.025,publish=lambda:None)
        self.answer='DIFFERENT';self.quality='CLEAN';self.human_answer='KEEP_SEPARATE'
        self.a,self.b=obj('a'),obj('b',100)
        def prepare(a,b,frame):
            uids={alias:o['obs_uids'][-1] for alias,o in zip('AB',(a,b))}
            runtime.evidence.observations={u:dict(processed_mask_ref=dict(sha256=u)) for u in uids.values()}
            return dict(h_snapshot_uid='h'+str(frame),histories={a:dict(selected=[dict(uid=uids[a])]) for a in 'AB'},
                selected_identity_histories={a:[dict(uid=uids[a])] for a in 'AB'})
        def render(directory,a,b,frame,binding):
            for name in ['quality_A.jpg','quality_B.jpg','merge.png']:(directory/name).write_bytes(b'image')
            return binding
        runtime.evidence=SimpleNamespace(prepare_merge=prepare,render_merge=render,observations={},containment=LiveEvidence.containment)
        def stage(event,directory,task,images,snapshot,labels=None):
            self.calls.append(task)
            return dict(value=dict(choice=self.quality) if task=='node_quality' else None if self.answer is None else dict(choice=self.answer,confidence=5,reason='x'))
        runtime.stage=stage
        runtime.stage_many=lambda eid,specs,snapshot:[stage(eid,d,t,i,snapshot,l) for d,t,i,l in specs]
        runtime.pending=lambda eid,directory,task,images,allowed:runtime.rows.setdefault(eid,dict(event_id=eid))
        runtime.fallback_choice=lambda *args:'KEEP_SEPARATE'
        def human(*args):self.humans.append(args);return self.human_answer
        runtime.human_choice=human;self.owner.vlm_runtime=runtime;self.gate=V7MergeGate(self.owner)
    def tearDown(self):self.tmp.cleanup()
    def review(self,frame):return self.gate.review(self.a,self.b,frame_idx=frame,source_frame_id=str(frame),stage='smoke')
    def test_failed_vlm_auto_is_unresolved(self):
        self.answer=None;self.review(1);self.review(2)
        row=self.gate.votes.state(self.gate.votes.key('a','b'))
        self.assertFalse(row['locked']);self.assertEqual(row['reject_total'],0)
    def test_quality_mixed_no_identity_call(self):
        self.quality='CONTAMINATED';self.review(1)
        self.assertEqual(self.calls,['node_quality','node_quality'])
    def test_negative_containment_human_mode_asks(self):
        self.owner.vlm_runtime.fallback="human"
        self.b['pcd'].points=self.a['pcd'].points.copy();self.review(1)
        self.assertEqual(len(self.humans),1);self.assertEqual(self.gate.events[0]['containment_human_choice'],'KEEP_SEPARATE')
    def test_negative_containment_auto_never_asks(self):
        self.b['pcd'].points=self.a['pcd'].points.copy()
        self.review(1);self.review(2)
        self.assertEqual(self.humans,[])
        report=self.gate.events[0]['containment']
        self.assertTrue(report['exceeds_threshold']);self.assertFalse(report['requires_human'])
        self.assertEqual(report['resolution_policy'],'AUTO_KEEP_SEPARATE')
        self.assertFalse(self.gate.votes.state(self.gate.votes.key('a','b'))['locked'])
        self.assertEqual(self.gate.votes.state(self.gate.votes.key('a','b'))['reject_total'],0)
    def test_input_failure_with_containment_auto_never_asks(self):
        self.b['pcd'].points=self.a['pcd'].points.copy()
        self.owner.vlm_runtime.evidence.prepare_merge=lambda *args: (_ for _ in ()).throw(ValueError('missing history'))
        self.review(1);self.assertEqual(self.humans,[])
        self.assertEqual(self.gate.events[0]['model_output']['choice'],'KEEP_SEPARATE')
        self.assertFalse(self.gate.events[0]['containment']['requires_human'])
    def test_positive_does_not_containment_check(self):
        self.b['pcd'].points=self.a['pcd'].points.copy();self.answer='SAME'
        self.assertIsNotNone(self.review(1));self.assertIsNone(self.review(2));self.assertFalse(self.humans)
    def test_same_frame_no_second_model_call(self):
        self.answer='SAME';self.review(1);count=len(self.calls);self.review(1)
        self.assertEqual(len(self.calls),count);self.assertEqual(self.gate.votes.state(self.gate.votes.key('a','b'))['merge_streak'],1)
    def test_locked_then_new_selected_history(self):
        self.review(1);self.review(2);count=len(self.calls);self.review(3);self.assertEqual(count,len(self.calls))
        self.a['obs_uids'].append('a-h2');self.a['image_idx'].append(3);self.a['num_detections']=2
        self.answer='SAME';self.review(4)
        row=self.gate.votes.state(self.gate.votes.key('a','b'))
        self.assertFalse(row['locked']);self.assertEqual(row['reject_total'],0);self.assertEqual(row['merge_streak'],1)
    def test_locked_pair_checks_history_without_rendering_images(self):
        self.review(1);self.review(2)
        self.owner.vlm_runtime.evidence.render_merge=lambda *args: (_ for _ in ()).throw(AssertionError('locked render'))
        self.review(3)
        directory=self.gate.root/'events'/self.gate.events[-1]['event_id']
        self.assertTrue((directory/'history_check.json').exists());self.assertFalse((directory/'merge.png').exists())
        self.assertEqual(self.gate.events[-1]['status'],'locked')
    def test_reversed_pair_does_not_unlock_same_history(self):
        self.review(1);self.review(2);count=len(self.calls)
        self.gate.review(self.b,self.a,frame_idx=3,source_frame_id='3',stage='reverse')
        self.assertEqual(len(self.calls),count);self.assertTrue(self.gate.votes.state(self.gate.votes.key('a','b'))['locked'])
    def test_actual_callback_changes_identity(self):
        self.answer='SAME';self.review(1);self.review(2);before=self.gate.votes.key('a','b')
        self.gate.on_merged(self.a,self.b)
        self.assertNotEqual(before,self.gate.votes.key('a','b'));self.assertEqual(self.gate.events[-1]['execution'],'MERGED')

class MutationAndFallback(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.r=V7Runtime.__new__(V7Runtime);self.r.root=self.root;self.r.rows={};self.r.fallback='auto'
        self.r.owner=SimpleNamespace(events=[],_support_history={});self.r.publish=lambda:None
    def tearDown(self):self.tmp.cleanup()
    def test_auto_fallback_discards_observation_and_keeps_merge(self):
        for task,expected in [('observation','DISCARD'),('merge','KEEP_SEPARATE')]:
            self.assertEqual(self.r.fallback_choice('e',self.root,task,[],'failure',['A'],'snap'),expected)
    def test_human_choice_rejects_wrong_snapshot_token(self):
        self.r.fallback="human"
        answers=iter(['OLD MERGE','E-SNAPSHOT MERGE'])
        self.r.owner._human_input=lambda prompt:next(answers)
        self.assertEqual(self.r.human_choice('e',self.root,['MERGE','KEEP_SEPARATE'],[],'smoke','snapshot'),'MERGE')
        self.assertEqual(json.loads((self.root/'human_answer.json').read_text())['c_bound_h_snapshot_uid'],'snapshot')
    def test_auto_guard_never_reads_stdin_or_creates_human_question(self):
        self.r.owner._human_input=lambda prompt: (_ for _ in ()).throw(AssertionError('auto read stdin'))
        for allowed,expected in [(['MERGE','KEEP_SEPARATE'],'KEEP_SEPARATE'),(['A','NEW','DISCARD'],'DISCARD')]:
            self.assertEqual(self.r.human_choice('e',self.root,allowed,[],'conflict','H'),expected)
        self.assertFalse((self.root/'human_question.json').exists())
    def test_human_fallback_uses_explicit_choice(self):
        self.r.fallback='human';self.r.owner._human_input=lambda prompt:'E-SNAPSHOT DISCARD'
        self.assertEqual(self.r.fallback_choice('e',self.root,'observation',[],'failure',['A'],'snapshot'),'DISCARD')
    def test_pairwise_request_binds_actual_candidate_alias(self):
        from conceptgraph.slam.v7_runtime import PROMPTS
        self.r.templates=json.loads((PROMPTS/'request_templates.json').read_text())
        self.r.stage_prompts={'pairwise':(PROMPTS/'pairwise.txt').read_text()}
        self.r.owner.model='smoke-model';self.r.owner.base_url='http://127.0.0.1:1';self.r.owner.timeout_seconds=1
        image=self.root/'input.jpg';image.write_bytes(b'smoke fixture')
        response=SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:dict(model='smoke-model',done=True,done_reason='stop',message=dict(content='{"choice":"SAME","confidence":5}')))
        for alias in 'ABC':
            self.r.rows[alias]={}
            with patch('conceptgraph.slam.v7_runtime.httpx.Client') as client:
                client.return_value.__enter__.return_value.post.return_value=response
                result=self.r.stage(alias,self.root/alias,'pairwise',[(alias,image)],'H')
                payload=client.return_value.__enter__.return_value.post.call_args.kwargs['json']
                self.assertIn('CANDIDATE '+alias,payload['messages'][1]['content'])
                self.assertEqual(result['value']['choice'],'SAME')
    def test_multi_same_routes_every_unordered_pair(self):
        candidates=[(a,i,obj(a,i*100)) for i,a in enumerate('ABC')]
        directory=self.root/'e';directory.mkdir()
        self.r.staged_bindings={'e':dict(objects=[object_state(o) for _,_,o in candidates],images=[dict(label='quality',path='quality.jpg')])}
        self.r.pending=lambda *args:self.r.rows.setdefault('e',{})
        self.r.stage=lambda eid,d,task,*args:dict(value=dict(status='USABLE',reason='x') if task=='observation_quality' else dict(choice='SAME',confidence=5))
        self.r.stage_many=lambda eid,specs,snapshot:[self.r.stage(eid,d,t,i,snapshot,l) for d,t,i,l in specs]
        calls=[];votes=V7Votes()
        def review(a,b,**kwargs):calls.append((a['id'],b['id']));return None
        self.r.merge_gate=lambda *args:SimpleNamespace(review=review,votes=votes)
        self.r.forced_groups=[]
        decision,output,_=self.r.adjudicate(directory,candidates,'H',2,'10')
        self.assertEqual(calls,[('A','B'),('A','C'),('B','C')]);self.assertEqual(output['choice'],'A')
        self.assertEqual(len(self.r.forced_groups),1);self.assertEqual(len(self.r.forced_groups[0]['keys']),3)
        # Same-frame repeated identical clique shares the queued execution.
        self.r.adjudicate(directory,candidates,'H',2,'10');self.assertEqual(len(self.r.forced_groups),1)
        # A partially overlapping group cannot silently associate after stale merging.
        self.r.forced_groups[0]['objects']=self.r.forced_groups[0]['objects'][:2]
        _,output,_=self.r.adjudicate(directory,candidates,'H',2,'10');self.assertEqual(output['choice'],'DISCARD')
    def test_merge_clique_execution_remaps_every_observation(self):
        from itertools import combinations
        from unittest.mock import Mock
        import sys
        objects=[obj(x,i*100) for i,x in enumerate('abcd')];members=objects[:3]
        votes=V7Votes();keys=[]
        for a,b in combinations(members,2):
            k=votes.key(a['id'],b['id']);keys.append(k);votes.record(k,1,'one','MERGE');votes.record(k,2,'two','MERGE')
        gate=SimpleNamespace(votes=votes,mark_executed=Mock(),_summary=Mock())
        self.r.owner._instance_merge_gate=gate
        self.r.forced_groups=[dict(parent_event='parent',objects=members,keys=keys,state=state_key([object_state(o) for o in members]),h_snapshot_uid='H')]
        (self.root/'events/parent').mkdir(parents=True)
        evidence=SimpleNamespace(record_object_merge=Mock())
        def merge(a,b,*args,**kw):
            a['obs_uids']+=b['obs_uids'];a['num_detections']+=b['num_detections'];return a
        cfg=dict(downsample_voxel_size=.01,dbscan_remove_noise=False,dbscan_eps=.1,dbscan_min_points=2,spatial_sim_type='overlap',device='cpu',make_edges=False)
        with patch.dict(sys.modules,{'conceptgraph.slam.utils':SimpleNamespace(merge_obj2_into_obj1=merge)}):
            result,matches=self.r.flush_groups(objects,[0,1,2,3,None,-1],cfg,evidence,2,None)
        self.assertEqual([o['id'] for o in result],['a','d']);self.assertEqual(matches,[0,0,0,1,None,-1])
        self.assertEqual(evidence.record_object_merge.call_count,2);self.assertEqual(gate.mark_executed.call_count,3)
        self.assertEqual(votes.generations,dict(a=1,b=1,c=1))
    def test_observation_input_failure_no_vlm_and_auto_discard(self):
        self.r.staged_bindings={'e':dict(objects=[],images=[],input_error='missing historical mask')}
        self.r.pending=lambda *args:self.r.rows.setdefault('e',{})
        self.r.stage=lambda *args: (_ for _ in ()).throw(AssertionError('must not call VLM'))
        decision,output,_=self.r.adjudicate(self.root/'e',[],'H',2,'10')
        self.assertEqual(output['choice'],'DISCARD');self.assertTrue(decision['reason_code'].startswith('INPUT_FAILURE'))

class IncrementalFeatureAllowlist(unittest.TestCase):
    def test_sync_registers_only_new_logged_refs_and_read_never_scans_observations(self):
        from conceptgraph.slam.vlm_runtime import sha
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);exp=root/'run';(exp/'evidence').mkdir(parents=True)
            log=exp/'evidence/observations.jsonl';log.write_text('')
            feature=root/'feature.npz';np.savez(feature,features=np.array([[1.,2.,3.]]))
            ref=dict(path='../feature.npz',sha256=sha(feature),key='features',index=0)
            reader=LiveEvidence(SimpleNamespace(root=exp/'gate'));reader.sync()
            with self.assertRaises(EvidenceInvariantError):reader.array(ref)
            log.write_text(json.dumps(dict(obs_uid='o1',image_feat_ref=ref))+'\n');reader.sync()
            self.assertEqual(len(reader.allowed_feature_refs),1)
            class NoFullScan(dict):
                def values(self):raise AssertionError('must not rebuild allowlist on feature read')
            reader.observations=NoFullScan(reader.observations)
            np.testing.assert_array_equal(reader.array(ref),[1,2,3])
            reader.sync();self.assertEqual(len(reader.allowed_feature_refs),1)
            # A changed file is still hash checked on EVERY access.
            np.savez(feature,features=np.array([[4.,5.,6.]]))
            with self.assertRaises(EvidenceInvariantError):reader.array(ref)
    def test_unregistered_external_ref_stays_forbidden(self):
        from conceptgraph.slam.vlm_runtime import sha
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);exp=root/'run';(exp/'evidence').mkdir(parents=True)
            (exp/'evidence/observations.jsonl').write_text('')
            p=root/'unregistered.npz';np.savez(p,data=[1])
            reader=LiveEvidence(SimpleNamespace(root=exp/'gate'));reader.sync()
            with self.assertRaises(EvidenceInvariantError):reader.array(dict(path='../unregistered.npz',sha256=sha(p),key='data'))

if __name__=='__main__':unittest.main()
