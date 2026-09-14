"""Frozen quality prompts, selective negative confirmation and run-local evidence cache."""
import copy,hashlib,json,threading,time,os
from pathlib import Path
from conceptgraph.slam.vlm_runtime import save_json
PROMPTS=Path(__file__).with_name('prompts')/'v7_must_v2'
TASKS={'observation_quality','node_quality'}
MODEL_DIGEST='c7bd058dd9774cae7dae32ef8cf3822aaacddd21e635e3cb6ec38effca0dc57f'
POLICY='v2_quality_selective_negative_confirmation_20260914'
def verify_model(model,urls):
 import httpx
 identity_path=os.environ.get('V7_MODEL_IDENTITY_FILE')
 expected=json.loads(Path(identity_path).read_text()) if identity_path else dict(model='qwen3.6:35b-a3b-mtp-q4_K_M',digest=MODEL_DIGEST,backend='ollama')
 if model!=expected['model']:raise ValueError('Configured model does not match frozen model identity')
 if identity_path:
  digest=hashlib.sha256(json.dumps(expected['files'],sort_keys=True).encode()).hexdigest()
  if digest!=expected['digest']:raise ValueError('Model file manifest digest mismatch')
 verified={}
 for url in urls:
  with httpx.Client(timeout=30,trust_env=False) as c:
   response=c.get(url+'/api/tags');response.raise_for_status()
  matches=[m for m in response.json()['models'] if m['name']==model]
  if len(matches)!=1 or matches[0]['digest']!=expected['digest']:raise ValueError('Endpoint does not match frozen model digest')
  if identity_path and matches[0].get('backend')!=expected['backend']:raise ValueError('Backend mismatch')
  verified[url]=dict(model=model,digest=expected['digest'],backend=expected['backend'])
 if not verified:raise ValueError('No verified model endpoint')
 return verified

def style(task,confirmation=False):
 first='boundary' if task=='observation_quality' else 'concise'
 return ('concise' if first=='boundary' else 'boundary') if confirmation else first

def payload(task,model,confirmation=False):
 field='status' if task=='observation_quality' else 'choice'
 labels=['USABLE','CORRUPTED','INSUFFICIENT'] if field=='status' else ['CLEAN','CONTAMINATED','INSUFFICIENT']
 name=style(task,confirmation)
 props={field:dict(type='string',enum=labels),'reason':dict(type='string',minLength=1,maxLength=120)}
 if name=='concise':props={'reason':props['reason'],field:props[field]}
 return dict(model=model,stream=False,think=False,keep_alive=-1,
     messages=[dict(role='system',content=(PROMPTS/(task+'_'+name+'.txt')).read_text()),dict(role='user',content='')],
     format=dict(type='object',properties=props,required=list(props),additionalProperties=False),
     options=dict(num_ctx=32768,num_predict=1024,temperature=0))

def parse_quality(text,task):
 text=text.strip();mode='STRICT_JSON'
 if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4].strip();mode='FENCED_JSON'
 elif text.startswith('```\n') and text.endswith('\n```'):text=text[4:-4].strip();mode='FENCED_JSON'
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ValueError('duplicate JSON key')
   d[k]=v
  return d
 v=json.loads(text,object_pairs_hook=unique);f='status' if task=='observation_quality' else 'choice'
 labels={'USABLE','CORRUPTED','INSUFFICIENT'} if f=='status' else {'CLEAN','CONTAMINATED','INSUFFICIENT'}
 if not isinstance(v,dict) or set(v)!={f,'reason'} or v[f] not in labels or not isinstance(v['reason'],str) or not 1<=len(v['reason'])<=120:raise ValueError('invalid quality schema')
 return v,mode,copy.deepcopy(v)

def resolve(task,primary,confirmation=None):
 f='status' if task=='observation_quality' else 'choice';neg='CORRUPTED' if f=='status' else 'CONTAMINATED'
 pos='USABLE' if f=='status' else 'CLEAN';v=primary.get('value')
 if not v:return {f:'INSUFFICIENT','reason':'质量初判接口或格式失败，证据未获得有效判断。'},'primary_failure',True
 if v[f]==pos:return v,'primary_positive',False
 if v[f]=='INSUFFICIENT':return v,'primary_uncertain',False
 c=(confirmation or {}).get('value')
 if c and c[f]==neg:return v,'negative_confirmed',False
 if not c:return {f:'INSUFFICIENT','reason':'负面质量初判未获得有效复核。'},'confirmation_failure',True
 return {f:'INSUFFICIENT','reason':'两套提示词对相同证据的质量结论不一致，目标归属尚不能确认。'},'negative_not_confirmed',False

class QualityService:
 def __init__(self,runtime):
  self.runtime=runtime;self.root=runtime.root/'quality_cache';self.root.mkdir(exist_ok=True)
  self.lock=threading.Lock();self.locks={}
 def key(self,item):
  if not item.get('quality_context'):raise ValueError('missing frozen quality context')
  signature=dict(policy=POLICY,task=item['task'],context=item['quality_context'],
      image_hashes=[x['sha256'] for x in item['request']['messages'][1]['images']],model_identity=next(iter(self.runtime.quality_model_identity.values())),
      primary=payload(item['task'],self.runtime.owner.model),confirmation=payload(item['task'],self.runtime.owner.model,True))
  return hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),signature
 def invoke(self,item):
  key,signature=self.key(item)
  with self.lock:lock=self.locks.setdefault(key,threading.Lock())
  with lock:
   path=self.root/(key+'.json');start=time.perf_counter()
   if path.exists():
    old=json.loads(path.read_text());r=copy.deepcopy(old['result'])
    # Cached evidence does not carry an execution certificate into the current event.
    r.update(h_snapshot_uid=item['snapshot'],c_bound_h_snapshot_uid=item['snapshot'],directory=str(item['directory'].relative_to(self.runtime.root)),
       images=item['request']['messages'][1]['images'],cache_hit=True,cache_key=key,cache_source=old['source'],seconds=time.perf_counter()-start)
    r['attempts']=[];r['timeout_count']=0
    return r
   primary=self.runtime._invoke_single_stage(item);confirmation=None
   f='status' if item['task']=='observation_quality' else 'choice';negative='CORRUPTED' if f=='status' else 'CONTAMINATED'
   if primary.get('value',{} ) and primary['value'][f]==negative:
    follow=self.runtime._prepare_stage(item['event_id'],item['directory']/'confirmation',item['task'],item['original_images'],item['snapshot'],item['labels'],confirmation=True)
    confirmation=self.runtime._invoke_single_stage(follow);save_json(follow['directory']/'result.json',confirmation)
   value,policy_reason,failure=resolve(item['task'],primary,confirmation)
   r=copy.deepcopy(primary);r.update(value=value,quality_policy=POLICY,policy_reason=policy_reason,interface_failure=failure,
      primary=primary,confirmation=confirmation,cache_hit=False,cache_key=key,seconds=time.perf_counter()-start)
   # Operational failures are never cached as visual evidence conclusions.
   if not failure:save_json(path,dict(signature=signature,source=dict(directory=r['directory'],h_snapshot_uid=item['snapshot']),result=r))
   return r
