"""Fixed-event threshold replay against confirmed human labels; no model or map execution."""
from pathlib import Path
import json,hashlib
from collections import Counter
from conceptgraph.slam.v7_merge import V7MergeGate

def main():
    source=Path('/home/chenkejun/beauty/v7_merge_v3_validation_20260909')
    out=Path('/home/chenkejun/beauty/v7_merge_70_analysis_20260909');out.mkdir(exist_ok=True)
    labels_path=source/'human_review_20260909.json';inventory_path=source/'offline_cases/data.json'
    labels=json.loads(labels_path.read_text())['cases'];inventory=json.loads(inventory_path.read_text())['cases'];rows=[]
    for item in inventory:
        p=Path(item['source_dir'])/'decision.json';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['decision_sha256']
        d=json.loads(raw);c=d['containment'];a,b=c['a_in_b'],c['b_in_a']
        assert a==c['a_in_b_count']/c['a_points'] and b==c['b_in_a_count']/c['b_points']
        assert d['execution']=='MERGED' and max(a,b)>.9
        if d.get('containment_geometry'):
            g=d['containment_geometry'];assert hashlib.sha256(Path(g['path']).read_bytes()).hexdigest()==g['sha256']
        rows.append(dict(event_id=d['event_id'],label=labels[d['event_id']]['choice'],a_in_b=a,b_in_a=b,
            new_merge=V7MergeGate.qualifies_containment(a,b),identity=d.get('identity_output'),fallback=d.get('fallback_reason'),source=str(p),source_sha256=item['decision_sha256']))
    event_root=Path(inventory[0]['source_dir']).parent
    actual={d['event_id'] for p in event_root.glob('*/decision.json') for d in [json.loads(p.read_text())]
            if d.get('decision_source')=='containment_gt90' and d.get('execution')=='MERGED'}
    assert actual=={r['event_id'] for r in rows}
    sweep=[]
    for low in [0,.05,.1,.2,.3,.4,.5,.6,.7,.8]:
        counts={label:dict(merge=0,keep=0) for label in ['SAME','DIFFERENT','UNCERTAIN']}
        for r in rows:counts[r['label']]['merge' if min(r['a_in_b'],r['b_in_a'])>low else 'keep']+=1
        sweep.append(dict(reverse_threshold=low,counts=counts))
    result=dict(kind='fixed_event_counterfactual_not_online_metrics',primary=.9,secondary=.6,
        labels_source=str(labels_path),labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        excluded_from_accuracy='UNCERTAIN',source_complete_executed_override_events=True,cases=rows,sweep=sweep)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=['| 事件 | 人工标签 | A→B | B→A | 新规则 |','|---|---|---:|---:|---|']
    for r in rows:lines.append(f"| {r['event_id']} | {r['label']} | {r['a_in_b']:.2%} | {r['b_in_a']:.2%} | {'合并' if r['new_merge'] else '保持不合并'} |")
    (out/'cases.md').write_text('\n'.join(lines)+'\n')
    print('Verified all',len(rows),'executed override events; human labels',dict(Counter(r['label'] for r in rows)))
    print('90/60',next(x for x in sweep if x['reverse_threshold']==.6)['counts'])
if __name__=='__main__':main()
