import copy,json,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
from conceptgraph.slam.v7_quality import resolve,parse_quality,payload,QualityService
from conceptgraph.slam.v7_runtime import V7Runtime,PROMPTS
from conceptgraph.slam.v7_merge import V7Votes
from conceptgraph.slam.v7_endpoint_pool import EndpointPool
from conceptgraph.slam.v7_quality_render import select_quality
import test_v7_split

class QualityPolicy(unittest.TestCase):
 def test_exhaustive_resolution(self):
  for task,f,pos,neg in [('observation_quality','status','USABLE','CORRUPTED'),('node_quality','choice','CLEAN','CONTAMINATED')]:
   for p in [pos,neg,'INSUFFICIENT',None]:
    for c in [pos,neg,'INSUFFICIENT',None]:
     make=lambda v:dict(value=None if v is None else {f:v,'reason':'测试'})
     v,why,failure=resolve(task,make(p),make(c))
     expected=pos if p==pos else neg if p==neg and c==neg else 'INSUFFICIENT'
     self.assertEqual(v[f],expected)
     self.assertEqual(failure,p is None or (p==neg and c is None))
 def test_strict_parser_rejects_salvage_and_extra_fields(self):
  for text in ['x {"choice":"CLEAN","reason":"x"}', '{"choice":"CLEAN","choice":"CONTAMINATED","reason":"x"}', '{"views":[],"choice":"CLEAN","reason":"x"}', '{"choice":"CLEAN","reason":"'+'x'*121+'"}']:
   with self.assertRaises(ValueError):parse_quality(text,'node_quality')
  self.assertEqual(parse_quality('```json\n{"choice":"CLEAN","reason":"x"}\n```','node_quality')[1],'FENCED_JSON')
 def test_quality_history_filter_preserves_strong_outlier(self):
  rows=[dict(uid=str(i),quality=q,mask_pixels=10,visual_distance=v,geometry_score=g) for i,(q,v,g) in enumerate([(100,0,0),(1,1,99),(30,.8,2),(40,.2,9)])]
  self.assertEqual([x['uid'] for x in select_quality(rows)],['0','2','3'])

class DynamicVotes(unittest.TestCase):
 def test_clean_one_uncertain_two_failure_resets(self):
  v=V7Votes();k=v.key('a','b');self.assertTrue(v.record(k,1,'e','MERGE',1)[0])
  v=V7Votes();k=v.key('a','b');self.assertFalse(v.record(k,1,'e','MERGE',2)[0]);self.assertTrue(v.record(k,2,'f','MERGE',2)[0])
  v=V7Votes();k=v.key('a','b');v.record(k,1,'e','MERGE',2);v.record(k,2,'f','DEFER',2)
  self.assertFalse(v.record(k,3,'g','MERGE',2)[0]);self.assertEqual(v.state(k)['reject_total'],0)
 def test_duplicate_frame_and_generation(self):
  v=V7Votes();k=v.key('a','b');v.record(k,1,'e','MERGE',2)
  with self.assertRaises(ValueError):v.record(k,1,'f','MERGE',2)
  v.merged('a','c');self.assertNotEqual(k,v.key('a','b'))

class MergeFailureSafety(unittest.TestCase):
 def test_quality_interface_failure_runs_identity_but_never_approves(self):
  f=test_v7_split.GateIntegration();f.setUp();self.addCleanup(f.tearDown);f.answer='SAME'
  f.owner.vlm_runtime.stage_many=lambda *a:[dict(value=dict(choice='INSUFFICIENT'),interface_failure=True),dict(value=dict(choice='CLEAN'))]
  for i in (1,2,3):self.assertIsNotNone(f.review(i))
  self.assertEqual(f.calls,['merge']*3)
  self.assertEqual(f.gate.events[-1]['vote_after']['merge_streak'],0)
  self.assertEqual(f.gate.events[-1]['vote_after']['reject_total'],0)
 def test_confirmed_mixed_overrides_other_failure(self):
  f=test_v7_split.GateIntegration();f.setUp();self.addCleanup(f.tearDown);f.answer='SAME'
  f.owner.vlm_runtime.stage_many=lambda *a:[dict(value=dict(choice='CONTAMINATED')),dict(value=None,interface_failure=True)]
  self.assertIsNotNone(f.review(1));self.assertEqual(f.calls,[])
  self.assertEqual(f.gate.events[-1]['model_output']['choice'],'KEEP_SEPARATE')

class RuntimeQuality(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  r=V7Runtime.__new__(V7Runtime);self.r=r;r.root=self.root;r.rows={};r.publish=lambda:None
  r.owner=SimpleNamespace(model='test',timeout_seconds=1);r.templates=json.loads((PROMPTS/'request_templates.json').read_text());r.stage_prompts={p.stem:p.read_text() for p in PROMPTS.glob('*.txt')}
  self.image=self.root/'input.png';self.image.write_bytes(b'fixed evidence')
  r.evidence=SimpleNamespace(quality_contexts={str(self.image):dict(identity=dict(object_uid='a',generation=0),histories=[dict(uid='h1',mask_sha='m')])})
  r.quality_model_identity={'test':dict(model='test',digest='test',backend='test')};r.endpoint_pool=EndpointPool(['http://test:1'],1);r.quality_service=QualityService(r)
 def invoke(self,eid):
  self.r.rows[eid]={};return self.r.stage(eid,self.root/eid,'node_quality',[('NODE A',self.image)],eid,['H1'])
 def test_real_dispatch_confirmation_cache_and_fresh_snapshot(self):
  calls=[]
  def post(url,json):
   calls.append(copy.deepcopy(json));answer='CONTAMINATED' if len(calls)==1 else 'CLEAN'
   return httpx.Response(200,request=httpx.Request('POST',url),json=dict(model='test',done=True,done_reason='stop',message=dict(content='{"choice":"'+answer+'","reason":"x"}')))
  with patch('httpx.Client') as client:
   client.return_value.__enter__.return_value.post.side_effect=post
   first=self.invoke('h1');second=self.invoke('h2')
  self.assertEqual(len(calls),2);self.assertEqual(first['value']['choice'],'INSUFFICIENT');self.assertTrue(second['cache_hit'])
  self.assertEqual(second['c_bound_h_snapshot_uid'],'h2');self.assertEqual(second['cache_source']['h_snapshot_uid'],'h1')
  self.assertEqual(calls[0]['messages'][1],calls[1]['messages'][1]);self.assertEqual(calls[1]['messages'][1]['content'],'')
  self.assertNotEqual(calls[0]['messages'][0],calls[1]['messages'][0])
 def test_failure_not_cached_and_single_positive_call(self):
  calls=[]
  def post(url,json):
   calls.append(json);content='bad' if len(calls)==1 else '{"choice":"CLEAN","reason":"x"}'
   return httpx.Response(200,request=httpx.Request('POST',url),json=dict(model='test',done=True,done_reason='stop',message=dict(content=content)))
  with patch('httpx.Client') as client:
   client.return_value.__enter__.return_value.post.side_effect=post
   a=self.invoke('h1');b=self.invoke('h2');c=self.invoke('h3')
  self.assertTrue(a['interface_failure']);self.assertFalse(b['cache_hit']);self.assertTrue(c['cache_hit']);self.assertEqual(len(calls),2)
 def test_cache_changes_with_evidence_generation_and_model(self):
  def item():return self.r._prepare_stage('x',self.root/str(len(list(self.root.iterdir()))),'node_quality',[('NODE A',self.image)],'h',['H1'])
  key=lambda:self.r.quality_service.key(item())[0]
  a=key();self.r.evidence.quality_contexts[str(self.image)]['identity']['generation']=1;b=key();self.assertNotEqual(a,b)
  self.image.write_bytes(b'new evidence');c=key();self.assertNotEqual(b,c)
  self.r.owner.model='new';self.assertNotEqual(c,key())

class MediaPacket(unittest.TestCase):
 def test_png_and_jpeg_annotation_descriptor(self):
  from PIL import Image
  from conceptgraph.slam.association_gate import _image_media_descriptor
  with tempfile.TemporaryDirectory() as td:
   for ext,mime in [('png','image/png'),('jpg','image/jpeg')]:
    p=Path(td)/('quality.'+ext);Image.new('RGB',(1944,704)).save(p)
    d=_image_media_descriptor(p);self.assertEqual(d['mime_type'],mime);self.assertEqual(d['width'],1944)
   p=Path(td)/'bad.png';p.write_text('<svg/>')
   with self.assertRaises(ValueError):_image_media_descriptor(p)

if __name__=='__main__':unittest.main()
