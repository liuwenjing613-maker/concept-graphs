"""Native Ollama decisions and live evidence; no offline labels or final maps."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import cv2
import httpx
import numpy as np

PROMPT_ROOT = Path(__file__).with_name('prompts') / 'v6_hatched'
VERSION = 'observation-p2-hatched-projection-v4__merge-v3__online-20260906'
MODEL = 'qwen3.6:35b-a3b-mtp-q4_K_M'
COLORS = [(220, 62, 108), (20, 162, 184), (75, 50, 110)]


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(value, allowed):
    if not isinstance(value, dict) or set(value) != {'choice', 'confidence'}:
        raise ValueError('expected exactly choice and confidence')
    if value['choice'] not in allowed:
        raise ValueError('choice outside current actions')
    if type(value['confidence']) is not int or not 0 <= value['confidence'] <= 5:
        raise ValueError('confidence must be integer 0..5 (not bool/string/float)')
    return value


def parse_output(text, allowed):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    clean = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', clean, re.IGNORECASE)
    if fenced:
        clean = fenced.group(1)
    return validate(json.loads(clean, object_pairs_hook=pairs), allowed), not bool(fenced)


def fit(im, width, height):
    out = np.full((height, width, 3), (28, 31, 36), np.uint8)
    ratio = min(width / im.shape[1], height / im.shape[0])
    w, h = max(1, round(im.shape[1] * ratio)), max(1, round(im.shape[0] * ratio))
    small = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
    x, y = (width-w)//2, (height-h)//2
    out[y:y+h, x:x+w] = small
    return out


def label(im, text, x, y, scale=.5, color=(245, 245, 245)):
    cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def outlined(im, mask):
    out = im.copy()
    edge = mask & ~cv2.erode(mask.astype('uint8'), np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = (255, 40, 40)
    return out


def crop_views(im, mask):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('empty target mask')
    x1, x2, y1, y2 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    px, py = round((x2-x1)*.18), round((y2-y1)*.18)
    x1, x2 = max(0, x1-px), min(im.shape[1], x2+px)
    y1, y2 = max(0, y1-py), min(im.shape[0], y2+py)
    target = np.full_like(im, (90, 90, 90))
    target[mask] = im[mask]
    return outlined(im, mask)[y1:y2, x1:x2], outlined(target, mask)[y1:y2, x1:x2]


def write_jpg(path, im):
    ok, data = cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError('JPEG encoding failed')
    Path(path).write_bytes(data.tobytes())


def occupancy(points, a, b, limits, size=288):
    ca, cb, span = limits
    if not np.isfinite([ca, cb, span]).all() or span <= 0:
        raise ValueError('bad projection range')
    q = np.asarray(points, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError('bad points')
    xy = np.rint(np.column_stack(((q[:, a]-ca)/span+.5, .5-(q[:, b]-cb)/span))*(size-1)).astype(int)
    xy = xy[np.all((xy >= 0) & (xy < size), axis=1)]
    out = np.zeros((size, size), np.uint8)
    out[xy[:, 1], xy[:, 0]] = 1
    return cv2.dilate(out, np.ones((3, 3), np.uint8)).astype(bool)


def cloud_panel(points_a, points_b, ranges, names):
    canvas = np.full((456, 1024, 3), (248, 249, 252), np.uint8)
    label(canvas, '3D WORLD: equal point size; no drawing-order occlusion', 14, 26, .55, (30, 38, 57))
    for x, name, color in zip([20, 290, 580], [*names, 'BOTH (projection only)'], COLORS):
        cv2.rectangle(canvas, (x, 42), (x+16, 58), color, -1)
        label(canvas, name, x+24, 56, .46, (30, 38, 57))
    for i, (a, b, name) in enumerate([(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]):
        left = 20+i*338
        label(canvas, name, left+8, 88, .58, (30, 38, 57))
        pa, pb = occupancy(points_a, a, b, ranges[i]), occupancy(points_b, a, b, ranges[i])
        tile = np.full((288, 288, 3), (248, 249, 252), np.uint8)
        tile[pa & ~pb] = COLORS[0]
        tile[pb & ~pa] = COLORS[1]
        tile[pa & pb] = COLORS[2]
        canvas[105:393, left+12:left+300] = tile
        cv2.rectangle(canvas, (left+11, 104), (left+300, 393), (194, 203, 218), 1)
        label(canvas, 'same frozen world axes / scale', left+3, 423, .39, (70, 78, 96))
    return canvas


class VLMRuntime:
    def __init__(self, owner):
        self.owner = owner
        self.root = owner.output_dir
        self.rows = {}
        self.projection_frames = {}
        self.status = 'running'
        self.prompts = {kind: (PROMPT_ROOT / (kind+'.txt')).read_text(encoding='utf-8').strip()
                        for kind in ('association', 'create', 'merge')}
        self.versions = {'version': VERSION, 'model': owner.model,
                         'prompt_sha256': {k: sha(PROMPT_ROOT/(k+'.txt')) for k in self.prompts},
                         'output': 'choice + integer confidence 0..5; all confidence accepted',
                         'renderer': 'integrated_candidate_rgb_projection_hatch_v4',
                         'yellow_locator_box': False}
        save_json(self.root/'vlm_versions.json', self.versions)
        self.root.joinpath('review').mkdir(exist_ok=True)
        template = Path(__file__).with_name('vlm_dashboard.html')
        shutil.copyfile(template, self.root/'index.html')
        shutil.copyfile(template, self.root/'review/index.html')
        self.publish()

    def prompts_for(self, kind, aliases):
        allowed = ['MERGE', 'KEEP_SEPARATE'] if kind == 'merge' else list(aliases)+['NEW', 'UNCERTAIN', 'DISCARD']
        order = 'OBJECT A → OBJECT B' if kind == 'merge' else 'CURRENT CONTEXT → CURRENT CROP → '+' → '.join('CANDIDATE '+a for a in aliases)
        user = '合法选项：'+', '.join(allowed)+'。图片顺序：'+order+'。请独立判断，只返回 choice、confidence 两字段的 JSON。'
        return self.prompts[kind], user

    def payload(self, system, user, images, allowed):
        encoded = []
        for _, path in images:
            raw = Path(path).read_bytes()
            im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if not raw.startswith(b'\xff\xd8') or im is None or min(im.shape[:2]) < 512:
                raise ValueError('VLM input must be real readable JPEG, both sides >=512')
            encoded.append(base64.b64encode(raw).decode('ascii'))
        return {'model': self.owner.model,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user, 'images': encoded}],
                'stream': False, 'think': False,
                'format': {'type': 'object', 'properties': {'choice': {'type': 'string', 'enum': list(allowed)},
                           'confidence': {'type': 'integer', 'minimum': 0, 'maximum': 5}},
                           'required': ['choice', 'confidence'], 'additionalProperties': False},
                'options': {'num_ctx': 32768, 'num_predict': 8192, 'temperature': 0}, 'keep_alive': -1}

    @staticmethod
    def redact(payload, images):
        result = json.loads(json.dumps(payload))
        result['messages'][1]['images'] = [{'label': label_, 'path': path.name, 'sha256': sha(path),
                                            'bytes': Path(path).stat().st_size} for label_, path in images]
        return result

    def call(self, payload, failure_log_path):
        path = Path(failure_log_path)
        started = time.perf_counter()
        endpoint = self.owner.base_url.rstrip('/')+'/api/chat'
        save_json(path.parent/'request_transport.json', {
            'client': 'native-ollama-httpx', 'endpoint': endpoint, 'trust_env': False,
            'max_attempts': self.owner.max_retries+1,
            'request_body_utf8_bytes': len(json.dumps(payload, ensure_ascii=False).encode())})
        attempts = []
        with httpx.Client(timeout=self.owner.timeout_seconds, trust_env=False) as client:
            for attempt in range(self.owner.max_retries+1):
                try:
                    response = client.post(endpoint, json=payload)
                    response.raise_for_status()
                    raw = response.json()
                    save_json(path.parent/'vlm_raw_response.json', raw)
                    if raw.get('model') != self.owner.model:
                        raise ValueError('server returned a different model')
                    if raw.get('done') is not True or raw.get('done_reason') == 'length':
                        raise ValueError('incomplete model response')
                    output, strict = parse_output(raw.get('message', {}).get('content', ''),
                                                  payload['format']['properties']['choice']['enum'])
                    raw['_format_strict'] = strict
                    raw['_attempts'] = attempt+1
                    save_json(path.parent/'vlm_raw_response.json', raw)
                    return raw, output, time.perf_counter()-started
                except Exception as exc:
                    # Do not put credentials, raw HTTP headers, or input Base64 in logs.
                    attempts.append({'attempt': attempt+1, 'error_type': type(exc).__name__,
                                     'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)})
                    save_json(path, {'attempts': attempts, 'latency_seconds': time.perf_counter()-started})
                    if isinstance(exc, (ValueError, TypeError, KeyError)):
                        break  # invalid model answer is a failure, not extra votes/re-sampling
                    if attempt < self.owner.max_retries:
                        time.sleep(1)
        raise RuntimeError('VLM request/validation failed; see vlm_error.json')

    def save_candidate(self, event_dir, alias, obj, current_points, candidate_points, ranges, pair_labels=None):
        from conceptgraph.slam.association_gate import _mask_array
        card = np.full((1024, 1024, 3), (28, 31, 36), np.uint8)
        title = ('OBJECT ' if pair_labels else 'CANDIDATE ')+alias
        label(card, title+' | SAME HISTORY; ORIGINAL COLORS', 14, 34, .62)
        histories = []
        frame_limit = int(re.search(r'_?f(\d+)', event_dir.name).group(1))
        if any(int(f) > frame_limit for f in obj.get('image_idx', [])):
            raise ValueError('candidate object contains future observations')
        for col, (reason, idx) in enumerate(self.owner._representative_members(obj, limit=3)):
            frame = int(obj['image_idx'][idx])
            if frame > frame_limit:
                raise ValueError('candidate history contains future data')
            source = Path(str(obj['color_path'][idx]))
            bgr = cv2.imread(str(source))
            if bgr is None:
                raise ValueError('unreadable history image')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = _mask_array(obj['mask'][idx], rgb.shape[:2])
            raw, target = crop_views(rgb, mask)
            left = 12+col*336
            label(card, f'H{col+1} RGB + OUTLINE', left+3, 67, .46)
            card[78:284, left:left+328] = fit(raw, 328, 206)
            label(card, f'H{col+1} TARGET ONLY', left+3, 310, .46)
            card[322:550, left:left+328] = fit(target, 328, 228)
            histories.append({'display_role': 'H'+str(col+1), 'member_index': idx,
                              'selection_reason': reason, 'frame_idx': frame,
                              'source_rgb_path': str(source), 'source_rgb_sha256': sha(source),
                              'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
                              'mask_area': int(mask.sum()),
                              'obs_uid': str(obj.get('obs_uids', [None]*len(obj['image_idx']))[idx])})
        if not histories:
            raise ValueError('no readable candidate history')
        for col in range(len(histories), 3):
            label(card, f'H{col+1} UNAVAILABLE', 15+col*336, 67, .46)
        names = pair_labels or ('CURRENT', 'CANDIDATE '+alias)
        card[568:] = cloud_panel(current_points, candidate_points, ranges, names)
        path = event_dir/('candidate_'+alias+'.jpg')
        write_jpg(path, card)
        return {'alias': alias, 'object_uid': str(obj['id']), 'selected_history': histories,
                'image_path': path.name, 'image_sha256': sha(path), 'image_layout': 'RGB+OUTLINE / TARGET ONLY / shared XYZ',
                'point_cloud_event_shared_ranges': ranges, 'point_cloud_sources':
                {'magenta': "object_A['pcd']" if pair_labels else "detection['pcd']",
                 'cyan': "object_B['pcd']" if pair_labels else "obj['pcd']"}}

    def event_evidence(self, event_dir, rgb, detection, candidates):
        from conceptgraph.slam.association_gate import _mask_array, _point_array, _sample_points, _shared_projection_ranges
        mask = _mask_array(detection['mask'][-1], rgb.shape[:2])
        context = np.full((1024, 1024, 3), (28, 31, 36), np.uint8)
        label(context, 'CURRENT CONTEXT | RED OUTLINE = TARGET', 18, 34, .66)
        context[60:] = fit(outlined(rgb, mask), 1024, 964)
        crop = np.full_like(context, (28, 31, 36))
        label(crop, 'CURRENT CROP | RED OUTLINE = TARGET', 18, 32, .68)
        label(crop, 'RGB + OUTLINE', 20, 76, .6)
        label(crop, 'TARGET ONLY', 530, 76, .6)
        raw, target = crop_views(rgb, mask)
        crop[95:1004, 16:504] = fit(raw, 488, 909)
        crop[95:1004, 520:1008] = fit(target, 488, 909)
        images = [('CURRENT CONTEXT', event_dir/'current_context.jpg'), ('CURRENT CROP', event_dir/'current_crop.jpg')]
        write_jpg(images[0][1], context)
        write_jpg(images[1][1], crop)
        full = [_point_array(detection)]+[_point_array(obj) for _, _, obj in candidates]
        if any(not len(p) for p in full):
            raise ValueError('empty live point cloud')
        ranges = _shared_projection_ranges(full)
        plotted = [_sample_points(p, 5000, event_dir.name+':'+str(i)) for i, p in enumerate(full)]
        np.savez_compressed(event_dir/'live_points.npz', current=full[0], **{a: p for (a, _, _), p in zip(candidates, full[1:])})
        manifest = [{'role': name, 'path': p.name, 'sha256': sha(p)} for name, p in images]
        for (alias, _, obj), points in zip(candidates, plotted[1:]):
            candidate_manifest = self.save_candidate(event_dir, alias, obj, plotted[0], points, ranges)
            projection = self.save_projection(event_dir, alias, obj, full[0], candidate_manifest)
            candidate_manifest.update(
                original_image_path=candidate_manifest['image_path'],
                original_image_sha256=candidate_manifest['image_sha256'],
                image_path=projection['path'], image_sha256=projection['sha256'],
                image_layout='RGB + CURRENT PROJECTION / TARGET ONLY / shared XYZ',
                projection_metadata_path='projection_'+alias+'.json',
                projection_metadata_sha256=sha(event_dir/('projection_'+alias+'.json')))
            manifest.append(candidate_manifest)
            images.append(('CANDIDATE '+alias, event_dir/projection['path']))
        save_json(event_dir/'evidence_manifest.json', {'images': manifest, 'live_points_sha256': sha(event_dir/'live_points.npz')})
        return manifest, images

    def register_projection_frame(self, frame_idx, rgb_path, depth_path,
                                  pose_c2w, intrinsics, image_hw, depth_scale):
        record = dict(frame_idx=int(frame_idx), rgb_path=str(rgb_path),
                      depth_path=str(depth_path), pose_c2w=np.asarray(pose_c2w).tolist(),
                      intrinsics=np.asarray(intrinsics).tolist(), image_hw=list(image_hw),
                      depth_scale=float(depth_scale))
        if int(frame_idx) in self.projection_frames:
            raise ValueError('camera frame registered twice')
        self.projection_frames[int(frame_idx)] = record
        with (self.root/'projection_frames.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')

    def save_projection(self, event_dir, alias, obj, full_current, candidate_manifest):
        from conceptgraph.slam.history_projection import build_integrated_candidate_card
        from conceptgraph.slam.association_gate import _mask_array
        histories = []
        h_frame = int(re.search(r'f(\d+)', event_dir.name).group(1))
        for member in candidate_manifest['selected_history']:
            frame = member['frame_idx']
            camera = self.projection_frames.get(frame)
            if camera is None:
                # Keep the original mask, but never infer missing camera/depth data.
                raw_hw = cv2.imread(member['source_rgb_path']).shape[:2]
                histories.append(dict(frame_idx=frame, rgb_path=member['source_rgb_path'],
                                      depth_path='', image_hw=list(raw_hw),
                                      mask=_mask_array(obj['mask'][member['member_index']], raw_hw),
                                      depth_scale=1, pose_c2w=np.eye(4).tolist(),
                                      intrinsics=np.eye(3).tolist(), obs_uid=member['obs_uid']))
                continue
            h = dict(camera)
            h.update(obs_uid=member['obs_uid'], member_index=member['member_index'],
                     selection_reason=member['selection_reason'],
                     mask=_mask_array(obj['mask'][member['member_index']], tuple(camera['image_hw'])))
            if Path(h['rgb_path']).resolve() != Path(member['source_rgb_path']).resolve():
                raise ValueError('history RGB does not match registered frame')
            histories.append(h)
        projection = build_integrated_candidate_card(
            alias, event_dir/candidate_manifest['image_path'], full_current, histories, event_dir, h_frame, candidate_hatch=True)
        projection['object_uid'] = str(obj['id'])
        projection['object_num_detections'] = int(obj.get('num_detections', 0))
        save_json(event_dir/('projection_'+alias+'.json'), projection)
        return projection

    def bind_projection_snapshot(self, event_dir, snapshot_uid, frame_idx):
        paths = sorted(event_dir.glob('candidate_*_projection.jpg'))
        save_json(event_dir/'projection_snapshot_binding.json', dict(
            h_snapshot_uid=snapshot_uid, h_frame=int(frame_idx),
            online_main_graph_latest_frame_at_h=int(frame_idx),
            full_current_points_sha256=sha(event_dir/'live_points.npz'),
            projection_images={p.name: sha(p) for p in paths},
            note='Images and full geometry are included in the frozen H snapshot payload.'))

    def pending(self, event_id, directory, kind, images, allowed):
        self.rows[event_id] = {'event_id': event_id, 'task': kind, 'status': 'waiting_for_vlm',
                              'images': [{'label': name, 'path': str(p.relative_to(self.root))} for name, p in images],
                              'directory': str(directory.relative_to(self.root)), 'allowed': allowed,
                              'model_output': None, 'error': None}
        self.publish()

    def publish(self):
        for event in self.owner.events:
            row = self.rows.get(event['event_id'])
            if row:
                row.update({'status': 'failure' if event['error'] else 'complete', 'model_output': event['model_output'],
                            'error': event['error'], 'latency_seconds': event['latency_seconds'],
                            'execution': event['route_reason'], 'final_match_index': event['final_match_index']})
        save_json(self.root/'live_vlm.json', {'status': self.status, 'version': self.versions['version'], 'model': self.owner.model,
                                            'rows': list(self.rows.values()), 'stats': dict(self.owner.stats)})
