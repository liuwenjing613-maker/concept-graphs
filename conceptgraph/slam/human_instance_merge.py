"""Blocking human approval for already-proposed map-object merges. No VLM calls."""
from collections import Counter
import hashlib
import html
import json
import time

import numpy as np

from conceptgraph.slam.association_gate import (
    HumanInputUnavailableError, _as_numpy, _json_dump, _jsonl_append,
    _point_array, _sample_points, _shared_projection_ranges, _utc_now,
)


def object_state(obj):
    """Fingerprint live geometry, identity, membership, and feature state."""
    points = np.ascontiguousarray(_point_array(obj), dtype='<f8')
    feature = np.ascontiguousarray(_as_numpy(obj.get('clip_ft', [])), dtype='<f4')
    return {
        'object_uid': str(obj['id']),
        'member_observation_uids': sorted(map(str, obj.get('obs_uids', []))),
        'image_indices': [int(frame) for frame in obj.get('image_idx', [])],
        'num_detections': int(obj.get('num_detections', 0)),
        'pcd_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
        'n_points': len(points),
        'clip_sha256': hashlib.sha256(feature.tobytes()).hexdigest(),
    }


def state_key(states):
    ordered = sorted(states, key=lambda item: item['object_uid'])
    return hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest()


class HumanInstanceMergeGate:
    def __init__(self, owner):
        self.owner = owner
        self.root = owner.output_dir / 'human_instance_merge'
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / 'events.jsonl'
        self.rejected_states = {}
        self.events = []
        self.stats = Counter()
        self._summary('ready')

    def _summary(self, status):
        _json_dump(self.root / 'summary.json', {
            'schema_version': 'human-instance-merge-v1', 'status': status,
            'counts': dict(self.stats), 'updated_at': _utc_now(),
            'actions': ['MERGE', 'KEEP_SEPARATE'],
            'trigger': 'original object merge rules and merge_guard accept, immediately before mutation',
            'cache_policy': 'reuse KEEP_SEPARATE only for the exact unordered pair of object states',
            'approval_is_not_execution': 'actual merges are recorded by evidence/mapping_events.jsonl',
            'vlm_calls': 0,
        })

    def close(self, *, status='completed'):
        self._summary(status)

    def _write_page(self, event, event_dir):
        token = event['answer_token']
        prefix = 'human_instance_merge/events/' + event['event_id']
        document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>实例合并人工审查</title>
<style>body{margin:20px;font:16px system-ui;background:#111827;color:#f8fafc}
.notice{padding:14px;background:#1e293b;border-radius:8px;line-height:1.7}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}
figure{margin:12px 0}img{width:100%;background:#020617}button{padding:12px 24px;margin:10px 12px 10px 0;font:600 18px system-ui;cursor:pointer}
input{font:18px monospace;width:min(95%,600px);padding:8px}code{color:#7dd3fc}</style></head><body>
<h1>实例合并：OBJECT A 与 OBJECT B 是否是同一个物理实例？</h1>
<div class="notice">当前题目 <code>__EVENT__</code> 已暂停建图。上半红色mask分别表示各自对象，背景仅供参考；每个对象最多3张历史观测。
下半是两对象<strong>此刻地图中的点云</strong>，不是历史点云并集：洋红色=OBJECT A，青色=OBJECT B；两张卡共用XY/XZ/YZ坐标尺度。
同类别、相邻或接触不等于同实例。证据不足以确认同实例时选“不合并”。<br>
选择按钮会复制本题答案；回终端粘贴并回车。答案含本题编号，旧页答案会被拒绝。</div>
<button disabled data-choice="Y">合并</button><button disabled data-choice="N">不合并</button><span id="status">证据图片加载中…</span><br>
<input id="answer" readonly aria-label="本题答案（也可手动复制）">
<div class="images"><figure><img src="__PREFIX__/candidate_A.jpg" alt="OBJECT A"><figcaption>OBJECT A</figcaption></figure>
<figure><img src="__PREFIX__/candidate_B.jpg" alt="OBJECT B"><figcaption>OBJECT B</figcaption></figure></div>
<script>
// Poll for a changed question without reloading an unchanged evidence card.
setInterval(async()=>{try{const r=await fetch(location.href,{cache:'no-store'});if(!r.ok)return;
const page=await r.text();if(!page.includes(__TOKEN_JSON__))location.reload();
}catch(e){if(!document.hasFocus())location.reload();}},2000);
// file:// without fetch support: refresh after returning from the terminal.
window.addEventListener('focus',()=>{if(location.protocol==='file:')location.reload()});
const token=__TOKEN_JSON__,imgs=[...document.images],buttons=[...document.querySelectorAll('button')];
function ready(){const ok=imgs.every(i=>i.complete&&i.naturalWidth>0);buttons.forEach(b=>b.disabled=!ok);document.getElementById('status').textContent=ok?'请选择；编号已自动绑定。':'图片未全部加载，请等待或刷新。'}
imgs.forEach(i=>{i.onload=ready;i.onerror=ready});ready();
buttons.forEach(b=>b.onclick=async()=>{const a=document.getElementById('answer');a.value=token+' '+b.dataset.choice;a.select();
try{if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(a.value);else if(!document.execCommand('copy'))throw Error('copy');document.getElementById('status').textContent='已复制，请回终端粘贴并回车。'}catch(e){document.getElementById('status').textContent='请手动复制上方答案，回终端粘贴。'}});
</script></body></html>'''
        document = document.replace('__EVENT__', html.escape(event['event_id'])).replace('__PREFIX__', prefix).replace('__TOKEN_JSON__', json.dumps(token))
        live = self.owner.output_dir / 'human_review.html'
        temporary = live.with_name('.human_review.html.tmp')
        temporary.write_text(document, encoding='utf-8')
        temporary.replace(live)
        return live

    def _choose(self, event, event_dir):
        live = self._write_page(event, event_dir)
        token = event['answer_token']
        print(f"\n[human-merge] Mapping paused. Review page: {live}", flush=True)
        print(f"[human-merge] MERGE: {token} Y  |  KEEP SEPARATE: {token} N", flush=True)
        while True:
            try:
                answer = self.owner._human_input('[human-merge] Paste this page answer: ').strip().upper().split()
            except EOFError as exc:
                raise HumanInputUnavailableError(f'human merge review needs interactive stdin; evidence saved at {live}') from exc
            if len(answer) == 2 and answer[0] == token and answer[1] in {'Y', 'N'}:
                return 'MERGE' if answer[1] == 'Y' else 'KEEP_SEPARATE'
            self.stats['invalid_or_stale_answers'] += 1
            print('[human-merge] Wrong event or invalid answer. Copy the answer from the CURRENT page.', flush=True)

    def review(self, source, target, *, frame_idx, source_frame_id, stage, overlap, visual, text):
        if source is target or str(source['id']) == str(target['id']):
            raise ValueError('instance merge requires two distinct live objects')
        states = [object_state(source), object_state(target)]
        if any(any(f > frame_idx for f in state['image_indices']) for state in states):
            raise ValueError('instance merge evidence contains a future observation')
        key = state_key(states)
        self.stats['original_merge_proposals'] += 1
        if key in self.rejected_states:
            self.stats['cached_rejections'] += 1
            _jsonl_append(self.events_path, {
                'event_type': 'cached_rejection', 'frame_idx': frame_idx,
                'source_frame_id': source_frame_id, 'stage': stage,
                'state_key': key, 'review_event_id': self.rejected_states[key],
                'source_uid': str(source['id']), 'target_uid': str(target['id']),
                'choice': 'KEEP_SEPARATE',
            })
            self._summary('ready')
            return 'human_instance_merge_rejected_cached'
        event_id = f"m{len(self.events) + 1:05d}_f{frame_idx:06d}_{stage}"
        directory = self.root / 'events' / event_id
        directory.mkdir(parents=True, exist_ok=False)
        points = [_point_array(obj) for obj in (source, target)]
        if any(not len(p) for p in points):
            raise ValueError('instance merge review requires nonempty live point clouds')
        # Shared robust ranges use full live clouds; only plotting density is sampled.
        ranges = _shared_projection_ranges(points)
        sampled = [_sample_points(p, 5000, str(obj['id'])) for p, obj in zip(points, (source, target))]
        h_utc = _utc_now()
        event = {
            'schema_version': 'human-instance-merge-v1', 'event_type': 'human_review',
            'event_id': event_id, 'state_key': key, 'frame_idx': frame_idx,
            'source_frame_id': source_frame_id, 'stage': stage,
            'object_A': states[0], 'object_B': states[1],
            'timeline': {'s_frame': frame_idx, 'd_frame': frame_idx, 'h_frame': frame_idx,
                         'h_utc': h_utc, 'online_main_graph_latest_frame_at_h': frame_idx},
            'baseline': 'MERGE_A_INTO_B',
            'original_scores_hidden_from_reviewer': {'overlap': float(overlap), 'visual': float(visual), 'text': float(text)},
            'overlap_score_note': 'original merge-pass score; evidence uses the latest objects immediately before this merge',
            'actions': ['MERGE', 'KEEP_SEPARATE'], 'status': 'preparing_evidence',
        }
        _json_dump(directory / 'decision.json', event)
        started = time.perf_counter()
        try:
            evidence = []
            for alias, obj in zip(('A', 'B'), (source, target)):
                card = self.owner._save_candidate_image(
                    directory, alias, obj, sampled[0], sampled[1], ranges,
                    pair_labels=('OBJECT A', 'OBJECT B'),
                )
                card.update({
                    'role': f'object_{alias}',
                    'point_cloud_sources': {'magenta': "object_A['pcd']", 'cyan': "object_B['pcd']"},
                    'current_point_color': 'magenta_object_A', 'candidate_point_color': 'cyan_object_B',
                })
                evidence.append(card)
            # Retain the exact live geometry used for the decision, not only pictures.
            np.savez_compressed(directory / 'live_pair.npz', object_A=points[0], object_B=points[1])
            snapshot = hashlib.sha256(json.dumps({'run_output_dir': str(self.owner.output_dir.resolve()), 'event': event_id, 'state': states, 'images': [c['image_sha256'] for c in evidence]}, sort_keys=True).encode()).hexdigest()
            event.update({'evidence': evidence, 'h_snapshot_uid': snapshot,
                          'answer_token': f'M{len(self.events)+1:05d}-{snapshot[:8]}'.upper(),
                          'status': 'waiting_for_human'})
            _json_dump(directory / 'input_manifest.json', {k: v for k, v in event.items() if k not in {'original_scores_hidden_from_reviewer', 'baseline'}})
            _json_dump(directory / 'decision.json', event)
            self._summary('waiting_for_human')
            choice = self._choose(event, directory)
            if key != state_key([object_state(source), object_state(target)]):
                raise RuntimeError('object state changed while human review was blocked')
            event.update({'choice': choice, 'status': 'review_complete',
                          'latency_seconds': time.perf_counter() - started,
                          'c_bound_h_snapshot_uid': snapshot})
            event['timeline'].update({'c_frame': frame_idx, 'c_utc': _utc_now(), 'ordering_valid': True})
        except BaseException as exc:
            event.update({'status': 'blocked_or_interrupted', 'error_type': type(exc).__name__})
            _json_dump(directory / 'decision.json', event)
            self._summary('blocked_or_interrupted')
            raise  # Fail closed: never silently merge when evidence/input fails.
        self.events.append(event)
        self.stats['reviewed'] += 1
        self.stats['approved' if choice == 'MERGE' else 'rejected'] += 1
        if choice == 'KEEP_SEPARATE':
            self.rejected_states[key] = event_id
        _json_dump(directory / 'decision.json', event)
        _jsonl_append(self.events_path, event)
        self._summary('ready')
        rows = ''.join(
            f'<tr><td>{html.escape(e["event_id"])}</td><td>{e["choice"]}</td>'
            f'<td><a href="events/{e["event_id"]}/candidate_A.jpg">OBJECT A</a> · '
            f'<a href="events/{e["event_id"]}/candidate_B.jpg">OBJECT B</a> · '
            f'<a href="events/{e["event_id"]}/decision.json">记录</a></td></tr>'
            for e in self.events
        )
        (self.root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>实例合并审查记录</title><h1>实例合并审查记录</h1><table>' + rows + '</table>', encoding='utf-8')
        print(f'[human-merge] {event_id}: {choice}; review evidence saved at {directory}', flush=True)
        return None if choice == 'MERGE' else 'human_instance_merge_rejected'
