"""Pair-identity voting, immediately before the original map merge mutation."""
from collections import Counter
import hashlib
import json
import time

import numpy as np

from conceptgraph.slam.human_instance_merge import object_state, state_key
from conceptgraph.slam.association_gate import _point_array, _sample_points, _shared_projection_ranges, _utc_now, _jsonl_append
from conceptgraph.slam.vlm_runtime import save_json, sha


class MergeVotes:
    """Observation updates retain identity; actual third-party merges change it."""
    def __init__(self, merge_required=3, reject_required=3):
        if merge_required < 1 or reject_required < 1:
            raise ValueError('vote thresholds must be positive')
        self.merge_required = merge_required
        self.reject_required = reject_required
        self.generations = {}
        self.pairs = {}

    def key(self, a, b):
        a, b = str(a), str(b)
        if a == b:
            raise ValueError('two distinct object identities required')
        return json.dumps(sorted([(a, self.generations.get(a, 0)), (b, self.generations.get(b, 0))]))

    def state(self, key):
        return self.pairs.setdefault(key, {'merge_streak': 0, 'reject_total': 0, 'locked': False,
                                           'last_frame': None, 'last_event': None, 'awaiting_execution': False})

    def skip_reason(self, key, frame):
        row = self.state(key)
        if row['locked']:
            return 'locked_after_three_rejections'
        if row['last_frame'] is not None and frame <= row['last_frame']:
            return 'already_reviewed_this_pair_frame'
        if row['awaiting_execution']:
            return 'awaiting_approved_merge_execution'
        return None

    def record(self, key, frame, event_id, choice):
        if self.skip_reason(key, frame):
            raise ValueError('duplicate or locked pair vote')
        row = self.state(key)
        row.update(last_frame=frame, last_event=event_id)
        if choice == 'MERGE':
            row['merge_streak'] += 1
        else:
            # Failed/invalid request breaks a consecutive run, but is not a negative vote.
            row['merge_streak'] = 0
            if choice == 'KEEP_SEPARATE':
                row['reject_total'] += 1
        row['locked'] = row['reject_total'] >= self.reject_required
        approved = not row['locked'] and row['merge_streak'] >= self.merge_required
        row['awaiting_execution'] = approved
        return approved, dict(row)

    def merged(self, source_uid, target_uid):
        # Target UID is retained by ConceptGraphs, hence explicit generation is essential.
        for uid in (str(source_uid), str(target_uid)):
            self.generations[uid] = self.generations.get(uid, 0)+1

    def snapshot(self):
        return {'merge_required_consecutive': self.merge_required, 'reject_required_total': self.reject_required,
                'generations': self.generations, 'pairs': self.pairs}


class VLMInstanceMergeGate:
    def __init__(self, owner):
        self.owner = owner
        self.runtime = owner.vlm_runtime
        self.root = owner.output_dir/'vlm_instance_merge'
        self.root.mkdir(parents=True, exist_ok=True)
        self.votes = MergeVotes()
        self.events = []
        self.stats = Counter()
        self._summary('ready')

    def _summary(self, status):
        save_json(self.root/'summary.json', {'schema_version': 'vlm-instance-merge-3streak-3reject-v1',
                  'status': status, 'updated_at': _utc_now(), 'counts': dict(self.stats),
                  'policy': self.votes.snapshot(), 'failure_policy': 'defer; no reject vote; reset merge streak',
                  'identity_policy': 'unordered object UID + actual object-merge generation; observation fusion does not reset',
                  'duplicate_policy': 'one model decision per unordered pair per frame',
                  'approval_is_not_execution': 'execution is recorded by actual merge callback'})

    def close(self, *, status='completed'):
        self._summary(status)

    def on_merged(self, source, target):
        key = self.votes.key(source['id'], target['id'])
        row = self.votes.state(key)
        if not row['awaiting_execution']:
            raise RuntimeError('map attempted unapproved object merge')
        event_id = row['last_event']
        self.votes.merged(source['id'], target['id'])
        self.owner._support_history.pop(str(source['id']), None)
        self.owner._support_history.pop(str(target['id']), None)
        self.stats['executed'] += 1
        event = next(e for e in reversed(self.events) if e['event_id'] == event_id)
        event['execution'] = 'MERGED'
        event['executed_at'] = _utc_now()
        event['target_generation_after'] = self.votes.generations[str(target['id'])]
        save_json(self.root/'events'/event_id/'decision.json', event)
        _jsonl_append(self.root/'executions.jsonl', {'event_id': event_id, 'pair_key': key,
                       'source_uid': str(source['id']), 'target_uid': str(target['id']), 'executed_at': event['executed_at']})
        self.runtime.rows[event_id]['execution'] = 'MERGED'
        self.runtime.publish()
        self._summary('ready')

    def review(self, source, target, *, frame_idx, source_frame_id, stage, overlap, visual, text):
        if source is target:
            raise ValueError('merge requires distinct objects')
        key = self.votes.key(source['id'], target['id'])
        self.stats['original_merge_proposals'] += 1
        reason = self.votes.skip_reason(key, int(frame_idx))
        if reason:
            self.stats[reason] += 1
            _jsonl_append(self.root/'skips.jsonl', {'frame_idx': frame_idx, 'stage': stage, 'pair_key': key,
                           'reason': reason, 'vote_state': dict(self.votes.state(key))})
            self._summary('ready')
            return 'vlm_merge_'+reason
        states = [object_state(source), object_state(target)]
        if any(any(f > frame_idx for f in s['image_indices']) for s in states):
            raise ValueError('future merge evidence')
        frozen_key = state_key(states)
        event_id = f'm{len(self.events)+1:05d}_f{frame_idx:06d}_{stage}'
        directory = self.root/'events'/event_id
        directory.mkdir(parents=True, exist_ok=False)
        points = [_point_array(o) for o in (source, target)]
        if any(not len(p) for p in points):
            raise ValueError('empty live merge cloud')
        ranges = _shared_projection_ranges(points)
        plotted = [_sample_points(p, 5000, str(o['id'])) for p, o in zip(points, (source, target))]
        start = time.perf_counter()
        event = {'event_id': event_id, 'task': 'merge', 'pair_key': key, 'state_key': frozen_key,
                 'frame_idx': frame_idx, 'source_frame_id': source_frame_id, 'stage': stage,
                 'object_A': states[0], 'object_B': states[1], 'status': 'preparing_evidence',
                 'timeline': {'s_frame': frame_idx, 'd_frame': frame_idx, 'h_frame': frame_idx,
                              'h_utc': _utc_now(), 'online_main_graph_latest_frame_at_h': frame_idx},
                 'vote_before': dict(self.votes.state(key)),
                 'original_scores_hidden_from_vlm': {'overlap': float(overlap), 'visual': float(visual), 'text': float(text)}}
        save_json(directory/'decision.json', event)
        # Evidence errors are fatal, not a silent permission to merge.
        evidence = [self.owner._save_candidate_image(directory, alias, obj, plotted[0], plotted[1], ranges,
                    pair_labels=('OBJECT A', 'OBJECT B')) for alias, obj in zip(('A', 'B'), (source, target))]
        np.savez_compressed(directory/'live_pair.npz', object_A=points[0], object_B=points[1])
        snapshot = hashlib.sha256(json.dumps({'state': states, 'images': [e['image_sha256'] for e in evidence],
                  'geometry': sha(directory/'live_pair.npz')}, sort_keys=True).encode()).hexdigest()
        event.update(evidence=evidence, h_snapshot_uid=snapshot)
        images = [('OBJECT '+a, directory/f'candidate_{a}.jpg') for a in ('A', 'B')]
        allowed = ['MERGE', 'KEEP_SEPARATE']
        system, user = self.runtime.prompts_for('merge', [])
        (directory/'system_prompt.txt').write_text(system+'\n', encoding='utf-8')
        (directory/'user_prompt.txt').write_text(user+'\n', encoding='utf-8')
        payload = self.runtime.payload(system, user, images, allowed)
        save_json(directory/'actual_request_redacted.json', self.runtime.redact(payload, images))
        save_json(directory/'input_manifest.json', {'h_snapshot_uid': snapshot, 'images': evidence,
                  'live_pair_sha256': sha(directory/'live_pair.npz'), 'versions': self.runtime.versions})
        self.runtime.pending(event_id, directory, 'merge', images, allowed)
        output, error, raw = None, None, None
        self.stats['vlm_calls'] += 1
        try:
            raw, output, latency = self.runtime.call(payload, directory/'vlm_error.json')
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
            self.stats['failures'] += 1
        if frozen_key != state_key([object_state(source), object_state(target)]):
            raise RuntimeError('objects changed while blocking on VLM')
        approved, vote_after = self.votes.record(key, int(frame_idx), event_id, output['choice'] if output else None)
        execution = 'APPROVED_AWAITING_EXECUTION' if approved else ('LOCKED_KEEP_SEPARATE' if vote_after['locked'] else 'DEFERRED')
        event.update(model_output=output, error=error, vote_after=vote_after, execution=execution,
                     status='failure' if error else 'complete', latency_seconds=time.perf_counter()-start,
                     c_bound_h_snapshot_uid=snapshot)
        event['timeline'].update(c_frame=frame_idx, c_utc=_utc_now(), ordering_valid=True)
        self.events.append(event)
        self.stats['reviewed'] += 1
        self.stats['approved' if approved else 'deferred'] += 1
        save_json(directory/'vlm_output.json', output)
        save_json(directory/'decision.json', event)
        _jsonl_append(self.root/'events.jsonl', event)
        self.runtime.rows[event_id].update({k: event[k] for k in
                    ('model_output', 'error', 'execution', 'status', 'latency_seconds', 'vote_after')})
        self.runtime.publish()
        self._summary('ready')
        print(f'[vlm-merge] {event_id} choice={output} streak={vote_after["merge_streak"]}/3 '
              f'rejections={vote_after["reject_total"]}/3 action={execution}', flush=True)
        return None if approved else 'vlm_merge_deferred'
