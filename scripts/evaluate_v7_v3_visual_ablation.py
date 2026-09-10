"""Frozen visual-only fragment ablation. No online policy is changed."""
import copy,json,time,hashlib,shutil
from pathlib import Path
from types import SimpleNamespace
from conceptgraph.slam.v7_runtime import V7Runtime,PROMPTS
from conceptgraph.slam.v7_endpoint_pool import EndpointPool
from conceptgraph.slam.vlm_runtime import save_json,sha

ROOT=Path('/home/chenkejun/beauty/v7_merge_v3_validation_20260909')
OUT=ROOT/'visual_only_ablation'
source=json.loads((ROOT/'offline_cases/data.json').read_text())
human=json.loads((ROOT/'human_review_20260909.json').read_text())['cases']
class VisualRuntime(V7Runtime):
 def _prepare_stage(self,*args,**kw):
  item=super()._prepare_stage(*args,**kw)
  user=item['payload']['messages'][1]
  user['content']='请结合三张图，判断A、B实际选中的区域属于同一具体物理物体还是不同物体。'
  item['request']['messages'][1]['content']=user['content']
  assert not any(x in '\n'.join(m['content'] for m in item['payload']['messages']) for x in ['a_in_b','b_in_a','distance_m','被包含节点','包含节点','本次冻结的几何信息'])
  save_json(item['directory']/'request.json',item['request'])
  return item

def runtime(cls,prompt):
 r=cls.__new__(cls);r.root=OUT;r.rows={};r.fallback='auto'
 r.owner=SimpleNamespace(model='qwen3.6:35b-a3b-mtp-q4_K_M',base_url='http://127.0.0.1:11437',timeout_seconds=120)
 r.endpoint_pool=EndpointPool([r.owner.base_url],120,models=[r.owner.model])
 r.templates=json.loads((PROMPTS/'request_templates.json').read_text())
 r.stage_prompts={f.stem:f.read_text() for f in PROMPTS.glob('*.txt')};r.stage_prompts['fragment']=prompt;r.publish=lambda:None
 return r

def main():
 import sys
 OUT.mkdir(exist_ok=True)
 p=(PROMPTS/'fragment.txt').read_text()
 removed='系统只确认某一方向的点云高度接近：较高方向的源节点称为“被包含节点”，另一个称为“包含节点”。它不是按名称、类别、点数推断的物体身份。\n'
 question='你的任务：被包含节点是否只是包含节点所指同一具体物理物体的局部或重复重建，还是独立物体恰好接触、放置在其上、在其内部或被遮挡？'
 assert removed in p and question in p
 visual=p.replace(removed,'').replace(question,'你的任务：A、B实际选中的区域是否属于同一具体物理物体的局部或重复重建，还是两个独立物体？')
 (OUT/'control_prompt.txt').write_text(p);(OUT/'visual_only_prompt.txt').write_text(visual)
 rows=[]
 for c in source['cases']:
  if c['route']!='FRAGMENT_RESOLVER' or human[c['event_id']]['choice']=='UNCERTAIN':continue
  row=copy.deepcopy(c);dst=OUT/'events'/c['event_id'];dst.mkdir(parents=True,exist_ok=True)
  for n in ['quality_A.jpg','quality_B.jpg','merge.png','input_manifest.json']:
   src=ROOT/'offline_cases/events'/c['event_id']/n;shutil.copy2(src,dst/n)
   if n in c['image_hashes']:assert sha(dst/n)==c['image_hashes'][n]
  row['reference']=human[c['event_id']]['choice'];row['historical_fragment']=row['fragment_output'];row['runs']={};rows.append(row)
 assert len(rows)==5
 state=dict(kind='paired_frozen_fragment_ablation',status='prepared',cases=rows,model='qwen3.6:35b-a3b-mtp-q4_K_M',endpoint='http://127.0.0.1:11437',excluded_uncertain=3)
 save_json(OUT/'data.json',state)
 if '--prepare-only' in sys.argv:return
 from conceptgraph.slam.v7_merge_policy import fragment_context
 old=runtime(V7Runtime,p);new=runtime(VisualRuntime,visual);t0=time.perf_counter();state['status']='running'
 for i,row in enumerate(rows):
  d=OUT/'events'/row['event_id'];images=[('QUALITY A',d/'quality_A.jpg'),('QUALITY B',d/'quality_B.jpg'),('MERGE',d/'merge.png')]
  event=json.loads((Path(row['source_dir'])/'decision.json').read_text())
  for mode,r in ([('control',old),('visual_only',new)] if i%2==0 else [('visual_only',new),('control',old)]):
   save_json(OUT/'data.json',state)
   r.rows[row['event_id']]={}
   result=r.stage(row['event_id'],d/mode,'fragment',images,event['h_snapshot_uid'],fragment_context(row['containment']) if mode=='control' else None)
   row['runs'][mode]=result;save_json(OUT/'data.json',state)
 state['status']='complete';state['seconds']=time.perf_counter()-t0
 def metrics(mode):
  vals=[(x['reference'],x['runs'][mode]['value']) for x in rows]
  return dict(correct_merge=sum(h=='SAME' and v and v['choice']=='SAME_FRAGMENT' for h,v in vals),
   wrong_merge=sum(h=='DIFFERENT' and v and v['choice']=='SAME_FRAGMENT' for h,v in vals),
   correct_separate=sum(h=='DIFFERENT' and v and v['choice']=='DISTINCT_OBJECT' for h,v in vals),
   missed_same=sum(h=='SAME' and (not v or v['choice']!='SAME_FRAGMENT') for h,v in vals),
   uncertain=sum(bool(v and v['choice']=='UNCERTAIN') for h,v in vals),interface_failures=sum(v is None for h,v in vals))
 state['summary']={m:metrics(m) for m in ['control','visual_only']};save_json(OUT/'data.json',state);print(json.dumps(state['summary']),flush=True)
if __name__=='__main__':main()

