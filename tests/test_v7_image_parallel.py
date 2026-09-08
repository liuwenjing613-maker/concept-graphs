import copy
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import numpy as np
from conceptgraph.slam.v7_endpoint_pool import EndpointPool,validate_urls
from conceptgraph.slam.v7_runtime import V7Runtime,PROMPTS
from conceptgraph.slam.v7_image_detection import DetectionCache,validate_coordinates

URLS=['http://one:1','http://two:2','http://three:3']
def response(content='{"choice":"SAME","confidence":5}',status=200,reason='stop'):
    return httpx.Response(status,request=httpx.Request('POST','http://test/api/chat'),
        json=dict(model='smoke-model',done=True,done_reason=reason,message=dict(content=content)))

class Transport(unittest.TestCase):
    def run_pool(self,effects):
        pool=EndpointPool(URLS,1);record=[]
        with patch('httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=effects
            result=pool.request({'x':1},lambda a,r:record.append(a.copy()))
        return result,record
    def test_three_timeouts_then_success_rotate_and_do_not_drop(self):
        result,record=self.run_pool([httpx.ReadTimeout('slow')]*3+[response()])
        self.assertIsNone(result[2]);self.assertEqual(len(record),4)
        self.assertEqual(sum(a['timed_out'] for a in record),3)
        self.assertEqual([a['endpoint'] for a in record[:3]],URLS)
        self.assertNotEqual(record[2]['endpoint'],record[3]['endpoint'])
    def test_four_timeouts_exhaust_without_fifth_call(self):
        result,record=self.run_pool([httpx.ReadTimeout('slow')]*4)
        self.assertIsNone(result[0]);self.assertIn('TIMEOUT_EXHAUSTED',result[2])
        self.assertEqual(len(record),4)
    def test_gateway_timeout_retries_but_other_http_failure_does_not(self):
        result,record=self.run_pool([response(status=504),response()]);self.assertIsNone(result[2]);self.assertEqual(len(record),2)
        result,record=self.run_pool([response(status=500)]);self.assertIsNotNone(result[2]);self.assertEqual(len(record),1)
    def test_three_slots_even_with_six_threads(self):
        pool=EndpointPool(URLS,1);barrier=threading.Barrier(3);lock=threading.Lock();active=set();peak=[0];seen=[]
        def post(url,json):
            with lock:
                self.assertNotIn(url,active);active.add(url);peak[0]=max(peak[0],len(active));seen.append(url)
            barrier.wait(timeout=3)
            time.sleep(.02)
            with lock:active.remove(url)
            return response()
        with patch('httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=post
            with ThreadPoolExecutor(max_workers=6) as executor:
                results=list(executor.map(lambda _:pool.request({},lambda a,r:None),range(6)))
        self.assertEqual(peak[0],3);self.assertEqual(len(seen),6);self.assertTrue(all(r[2] is None for r in results))
    def test_invalid_endpoint_count(self):
        for urls in [[],URLS+['http://four'],['http://user:secret@server']]:
            with self.assertRaises(ValueError):validate_urls(urls)

class Stages(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.r=V7Runtime.__new__(V7Runtime);self.r.root=self.root;self.r.rows={'e':{}};self.r.fallback='auto'
        self.r.owner=SimpleNamespace(model='smoke-model',base_url=URLS[0],timeout_seconds=1)
        self.r.templates=json.loads((PROMPTS/'request_templates.json').read_text())
        self.r.stage_prompts={p.stem:p.read_text() for p in PROMPTS.glob('*.txt')}
        self.r.endpoint_pool=EndpointPool(URLS,1);self.thread=threading.get_ident();self.publish_count=0
        def publish():
            self.assertEqual(threading.get_ident(),self.thread);self.publish_count+=1
        self.r.publish=publish
        self.image=self.root/'input.jpg';self.image.write_bytes(b'image fixture')
    def tearDown(self):self.tmp.cleanup()
    def test_parallel_results_bound_to_alias_and_snapshot_not_arrival_order(self):
        barrier=threading.Barrier(3);completed=[]
        def post(url,json):
            alias=next(a for a in 'ABC' if 'CANDIDATE '+a in json['messages'][1]['content'])
            barrier.wait(timeout=3);time.sleep({'A':.06,'B':.03,'C':0}[alias]);completed.append(alias)
            return response('{"choice":"'+('SAME' if alias=='B' else 'DIFFERENT')+'","confidence":5}')
        with patch('httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=post
            rows=self.r.stage_many('e',[(self.root/a,'pairwise',[(a,self.image)],None) for a in 'ABC'],'FROZEN-H')
        self.assertEqual(completed,['C','B','A'])
        self.assertEqual([r['value']['choice'] for r in rows],['DIFFERENT','SAME','DIFFERENT'])
        self.assertEqual([r['directory'] for r in self.r.rows['e']['stage_results']],list('ABC'))
        self.assertTrue(all(r['c_bound_h_snapshot_uid']=='FROZEN-H' for r in rows))
        self.assertEqual(self.publish_count,2)
    def test_timeout_retries_same_payload_once_stage_result(self):
        payloads=[]
        def post(url,json):
            payloads.append(copy.deepcopy(json))
            if len(payloads)<=3:raise httpx.ReadTimeout('slow')
            return response()
        with patch('httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=post
            r=self.r.stage('e',self.root/'A','pairwise',[('A',self.image)],'H')
        self.assertEqual(r['value']['choice'],'SAME');self.assertEqual(r['timeout_count'],3)
        self.assertTrue(all(p==payloads[0] for p in payloads));self.assertEqual(len(self.r.rows['e']['stage_results']),1)
        self.assertEqual(len(list((self.root/'A/attempts').glob('*/attempt.json'))),4)
    def test_truncation_exhausts_one_completion_retry(self):
        with patch('httpx.Client') as client:
            client.return_value.__enter__.return_value.post.return_value=response(reason='length')
            r=self.r.stage('e',self.root/'A','pairwise',[('A',self.image)],'H')
        self.assertIsNone(r['value']);self.assertEqual(len(r['attempts']),2)
    def test_exhaustion_routes_auto_and_human_after_four_attempts(self):
        with patch('httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=httpx.ReadTimeout('slow')
            r=self.r.stage('e',self.root/'A','pairwise',[('A',self.image)],'H')
        self.assertEqual(len(r['attempts']),4);self.assertIsNone(r['value'])
        self.assertEqual(self.r.fallback_choice('e',self.root,'observation',[],'timeout',[],'H'),'DISCARD')
        self.assertEqual(self.r.fallback_choice('e',self.root,'merge',[],'timeout',[],'H'),'KEEP_SEPARATE')
        self.r.fallback='human';calls=[];self.r.human_choice=lambda *a:calls.append(a) or 'NEW'
        self.assertEqual(self.r.fallback_choice('e',self.root,'observation',[],'timeout',[],'H'),'NEW');self.assertEqual(len(calls),1)

class Cache(unittest.TestCase):
    def test_old_cache_rejected_and_new_cache_checks_coverage_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'frame000000.jpg';source.write_bytes(b'RGB')
            c=root/'cache';c.mkdir()
            with self.assertRaisesRegex(ValueError,'unversioned'):DetectionCache(c,{'imgsz':1200},[source],False)
            spec={'yolo_imgsz':1200};cache=DetectionCache(c,spec,[source],True)
            with self.assertRaisesRegex(ValueError,'incomplete'):DetectionCache(c,spec,[source],False)
            folder=c/'detections'/source.stem;folder.mkdir(parents=True)
            for name in ['xyxy','mask','class_id','confidence','image_feats']:(folder/(name+'.npz')).touch()
            cache.record(source,[704,1216],[680,1200])
            reused=DetectionCache(c,spec,[source],False);self.assertEqual(reused.check_source(source)['actual_yolo_input_hw'],[704,1216])
            with self.assertRaisesRegex(ValueError,'mismatch'):DetectionCache(c,{'yolo_imgsz':640},[source],False)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):reused.check_source(source)
    def test_coordinate_mismatch_rejected(self):
        g=dict(xyxy=np.array([[0,0,20,10]]),mask=np.ones((1,10,20)),class_id=np.array([0]),classes=['x'])
        validate_coordinates(g,[10,20],['x'])
        with self.assertRaises(ValueError):validate_coordinates(g,[20,40],['x'])
        with self.assertRaises(ValueError):validate_coordinates(g,[10,20],['y'])

if __name__=='__main__':unittest.main()
