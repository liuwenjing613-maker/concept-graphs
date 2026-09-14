import unittest,base64,copy,os,json,tempfile,hashlib
from pathlib import Path
from unittest.mock import patch,MagicMock
from scripts.v7_h800_adapter import translate
from conceptgraph.slam.v7_quality import payload,verify_model,QualityService
from types import SimpleNamespace
class Transport(unittest.TestCase):
 def test_image_and_parameters_preserved(self):
  p=payload('node_quality','fp8');imgs=[base64.b64encode(x).decode() for x in [b'\x89PNG\r\n\x1a\nabc',b'\xff\xd8\xffabc']];p['messages'][1]['images']=imgs
  original=copy.deepcopy(p);r=translate(p,'fp8');self.assertEqual(original,p)
  self.assertEqual(r['max_tokens'],1024);self.assertFalse(r['chat_template_kwargs']['enable_thinking']);self.assertEqual(r['response_format']['json_schema']['schema'],p['format'])
  for i,mime in enumerate(['png','jpeg']):self.assertEqual(r['messages'][1]['content'][i+1]['image_url']['url'],'data:image/'+mime+';base64,'+imgs[i])
  with self.assertRaises(ValueError):translate(p,'other')
  p['options']['seed']=1
  with self.assertRaises(ValueError):translate(p,'fp8')
 def test_verified_manifest_and_endpoint(self):
  files={'config.json':{'sha256':'abc','size':3}};d=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'identity.json';p.write_text(json.dumps(dict(model='fp8',digest=d,backend='vllm',files=files)))
   with patch.dict(os.environ,{'V7_MODEL_IDENTITY_FILE':str(p)}),patch('httpx.Client') as client:
    response=client.return_value.__enter__.return_value.get.return_value
    response.json.return_value={'models':[dict(name='fp8',digest=d,backend='vllm')]}
    self.assertEqual(verify_model('fp8',['url'])['url']['digest'],d)
    response.json.return_value['models'][0]['digest']='wrong'
    with self.assertRaises(ValueError):verify_model('fp8',['url'])
 def test_cache_identity_isolation(self):
  with tempfile.TemporaryDirectory() as td:
   r=SimpleNamespace(root=Path(td),owner=SimpleNamespace(model='fp8'),quality_model_identity={'url':dict(model='fp8',digest='a',backend='vllm')})
   s=QualityService(r);item=dict(task='node_quality',quality_context={'uid':'1'},request={'messages':[{},dict(images=[{'sha256':'im'}])]})
   a=s.key(item)[0];r.quality_model_identity['url']['digest']='b';self.assertNotEqual(a,s.key(item)[0])
if __name__=='__main__':unittest.main()
