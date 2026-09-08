"""Fresh room0 mapping, strict audit, three fixed-protocol evaluations and comparison."""
import argparse,datetime,hashlib,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/home/chenkejun/beauty/conceptgraphs')
EXPS=PROJECT/'results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps'
PYTHON=PROJECT/'envs/cg-ali/bin/python'

def read(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def collect(exp):
 d=EXPS/exp;out=dict(experiment=exp,files={},values={},protocols={})
 patterns=dict(semantic='point_semantic_ali_dev_nexclude6_*/semseg_results.json',instance='instance_pq_fixed_support_*/results.json',coverage='geometry_coverage_fixed_support_*/results.json')
 for kind,pattern in patterns.items():
  files=sorted(d.glob(pattern))
  if not files:raise FileNotFoundError(str(d/pattern))
  p=files[-1];x=read(p);out['files'][kind]=dict(path=str(p),sha256=digest(p));out[kind]=x
  if kind=='semantic':
   out['protocols'][kind]=x['protocol'];row=next(r for r in x['summary_rows_percent'] if r['scope']=='room0')
   for k in ['miou','mrecall','mprecision','mf1score','fmiou','point_accuracy']:out['values'][k]=row[k]
  elif kind=='instance':
   out['protocols'][kind]=x['protocol']
   for k in ['pq','rq','sq']:out['values'][k]=100*x['metrics'][k]
   for k in ['tp','fp','fn','predicted_map_instances']:out['values'][k]=x['metrics'][k]
  else:
   out['protocols'][kind]=x['definition']
   for row in x['coverage']:out['values'][f"coverage_{row['distance_m']}m"]=100*row['coverage']
 out['map_sha256']=digest(d/f'pcd_{exp}.pkl.gz')
 return out

def compare(run_dir,exp):
 baseline=read(run_dir/'baseline.json');current=collect(exp)
 for kind in baseline['protocols']:
  if baseline['protocols'][kind]!=current['protocols'][kind]:raise ValueError('evaluation protocol changed: '+kind)
 if baseline['semantic']['summary_rows_percent'][0]['point_count']!=current['semantic']['summary_rows_percent'][0]['point_count']:
  raise ValueError('semantic reference support changed')
 rows=[dict(metric=k,baseline=v,current=current['values'][k],delta=current['values'][k]-v) for k,v in baseline['values'].items()]
 live=read(EXPS/exp/'blocking_association_gate/live_vlm.json');events=live['rows']
 if isinstance(events,dict):events=list(events.values())
 stages=[s for e in events for s in e.get('stage_results',[])]
 diagnostics=dict(events=len(events),logical_stages=len(stages),http_calls=sum(len(s.get('attempts',[])) for s in stages),
  stage_failures=sum(bool(s.get('error')) for s in stages),timeouts=sum(s.get('timeout_count',0) for s in stages),
  length_recoveries=sum(bool(s.get('length_recovery')) for s in stages),
  joint_reviews=sum(s.get('task')=='joint_review' for s in stages),
  joint_rescued_observations=sum(e.get('staged_decision',{}).get('reason_code')=='JOINT_REVIEW_UNIQUE_OWNER' for e in events),
  unresolved_holds=sum(e.get('status')=='unresolved_hold' for e in events))
 write(run_dir/'comparison.json',dict(baseline=baseline,current=current,rows=rows,diagnostics=diagnostics))
 text=['# v7_1 room0 完整流程结果','',f'生成时间：{utc()}。对照：v7_auto_image；实验：{exp}。',
  '相同原图检测缓存、从空地图在线处理400帧；结果是单场景探索，不是人工准确率证明。',
  '','| 指标 | 此前最优 | v7_1 | 差值 |','|---|---:|---:|---:|']
 for r in rows:text.append(f"| {r['metric']} | {r['baseline']:.4f} | {r['current']:.4f} | {r['delta']:+.4f} |")
 text+=['','比例指标已转为百分数，差值为百分点；TP/FP/FN/节点数保持计数。实例PQ是既定固定支撑诊断口径，不是官方Replica PQ。',
  '','## 流程诊断','```json',json.dumps(diagnostics,ensure_ascii=False,indent=2),'```',
  '','## 局限与后续分析','新旧调用接口数量不同，本次复用检测而旧版生成检测，不能直接把总耗时差全部归因于新策略。',
  '需要结合逐类IoU与PQ匹配变化报告成功、退化与失败；改判数、证据审计通过不等于改判正确。',
  '原始指标、参考路径、哈希和逐项差值见 comparison.json；执行与各阶段耗时见 state.json。']
 (run_dir/'comparison.md').write_text('\n'.join(text)+'\n')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--exp-suffix',required=True);args=parser.parse_args()
 if not re.fullmatch(r'v7_1_[A-Za-z0-9_.-]+',args.exp_suffix):raise ValueError('invalid unique experiment name')
 run_dir=args.run_dir.resolve();run_dir.mkdir(parents=True,exist_ok=False)
 if (EXPS/args.exp_suffix).exists():raise FileExistsError('refuse existing map')
 state=dict(status='preparing',pid=os.getpid(),experiment=args.exp_suffix,run_dir=str(run_dir),map_dir=str(EXPS/args.exp_suffix),started_at=utc(),stages=[],code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
 write(run_dir/'state.json',state)
 def run(name,command,env=None):
  stage=dict(name=name,command=command,started_at=utc(),status='running');state['stages'].append(stage);state['status']=name;write(run_dir/'state.json',state);start=time.monotonic()
  with (run_dir/(name+'.log')).open('w') as f:
   p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);stage['pid']=p.pid;write(run_dir/'state.json',state);rc=p.wait()
  stage.update(status='completed' if rc==0 else 'failed',return_code=rc,seconds=time.monotonic()-start,completed_at=utc());write(run_dir/'state.json',state)
  if rc:raise RuntimeError(f'{name} failed with exit {rc}; see stage log')
 try:
  write(run_dir/'baseline.json',collect('v7_auto_image'))
  wrappers={}
  for name in ['evaluate','evaluate_instance_pq','evaluate_geometry_coverage']:
   src=PROJECT/'scripts'/f'{name}.sh';dst=run_dir/src.name;shutil.copy2(src,dst);wrappers[name]=str(dst)
  write(run_dir/'evaluation_scripts.json',{name:dict(path=p,sha256=digest(Path(p))) for name,p in wrappers.items()})
  mapping_env=dict(os.environ,V7_1_JOINT_REVIEW='0')
  run('mapping',['bash',str(ROOT/'run_v7_1.sh'),'auto','--exp-suffix',args.exp_suffix,'--scene','room0','--end','2000','--stride','5','--gpu','1','--detections-exp-suffix','v7_auto_image_detections','--no-web-link'],mapping_env)
  manifest=read(EXPS/args.exp_suffix/'evidence/manifest.json')
  if manifest['status']!='MAP_COMPLETED_EVIDENCE_VALID':raise ValueError('mapping evidence incomplete/invalid')
  frame_lines=(EXPS/args.exp_suffix/'evidence/frames.jsonl').read_text().splitlines()
  if len(frame_lines)!=400:raise ValueError('incomplete frame range')
  env=dict(os.environ,PYTHONPATH=str(ROOT/'.runtime-deps')+':'+str(ROOT))
  run('audit',[str(PYTHON),str(ROOT/'scripts/audit_evidence.py'),str(EXPS/args.exp_suffix/'evidence'),'--strict'],env)
  for name in wrappers:run(name,['bash',wrappers[name],args.exp_suffix,'room0'])
  compare(run_dir,args.exp_suffix)
  state.update(status='completed',completed_at=utc());write(run_dir/'state.json',state)
 except Exception as exc:
  state.update(status='failed',error=str(exc),completed_at=utc());write(run_dir/'state.json',state)
  (run_dir/'failure.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
