"""Three real concurrent calls on one previously frozen event; no map or accuracy evaluation."""
import argparse,json,shutil,time
from pathlib import Path
from types import SimpleNamespace
from conceptgraph.slam.v7_runtime import V7Runtime,PROMPTS
from conceptgraph.slam.v7_endpoint_pool import EndpointPool
from conceptgraph.slam.vlm_runtime import save_json,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-event',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--urls',nargs='+',default=['http://127.0.0.1:11464','http://127.0.0.1:11463','http://127.0.0.1:11435'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    binding=json.loads((a.source_event/'staged_evidence.json').read_text());snapshot=binding['h_snapshot_uid']
    r=V7Runtime.__new__(V7Runtime);r.root=a.output;r.rows={'smoke':{}};r.fallback='auto'
    r.owner=SimpleNamespace(model='qwen3.6:35b-a3b-mtp-q4_K_M',base_url=a.urls[0],timeout_seconds=60)
    r.templates=json.loads((PROMPTS/'request_templates.json').read_text());r.stage_prompts={p.stem:p.read_text() for p in PROMPTS.glob('*.txt')}
    r.endpoint_pool=EndpointPool(a.urls,60);r.publish=lambda:save_json(a.output/'live_vlm.json',r.rows)
    specs=[]
    for alias in 'ABC':
        source=a.source_event/f'candidate_{alias}.jpg'
        expected=next(i['sha256'] for i in binding['images'] if i['path']==source.name)
        assert sha(source)==expected
        image=a.output/source.name;shutil.copyfile(source,image)
        specs.append((a.output/alias,'pairwise',[(alias,image)],None))
    start=time.perf_counter();results=r.stage_many('smoke',specs,snapshot);wall=time.perf_counter()-start
    ok=all(row['error'] is None and row['c_bound_h_snapshot_uid']==snapshot for row in results)
    report=dict(passed=ok,source=str(a.source_event),h_snapshot_uid=snapshot,wall_seconds=wall,
                summed_attempt_seconds=sum(t['seconds'] for row in results for t in row.get('attempts',[])),
                endpoints=[t['endpoint'] for row in results for t in row.get('attempts',[])],
                decisions=[row['value'] for row in results],scope='transport/concurrency smoke; not accuracy evaluation')
    save_json(a.output/'validation.json',report);print(json.dumps(report,ensure_ascii=False,indent=2));assert ok
if __name__=='__main__':main()
