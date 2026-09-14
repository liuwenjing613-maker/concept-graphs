"""Shared CPU fixed-support evaluation; never imports mapper or CLIP models."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import platform
import runpy
import time
from pathlib import Path

import h5py
import numpy as np
import scipy
from scipy.spatial import cKDTree


EXCLUSIONS = {1: ['other'], 4: ['other', 'floor', 'wall', 'ceiling'],
              6: ['other', 'floor', 'wall', 'ceiling', 'door', 'window']}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(f'{value.dtype}:{value.shape}'.encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--replica-semantic-root', type=Path, required=True)
    p.add_argument('--slam-root', type=Path, required=True)
    p.add_argument('--scene', nargs=3, metavar=('SCENE', 'SEMANTIC_SCENE', 'MAP'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--n-exclude', type=int, choices=sorted(EXCLUSIONS), default=6)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk-size', type=int, default=250000)
    p.add_argument('--reference-start', type=int, default=0)
    p.add_argument('--reference-end', type=int, default=2000)
    p.add_argument('--reference-stride', type=int, default=5)
    return p


def begin_output(args):
    out = args.output_dir.resolve()
    # Wrappers create the directory and evaluation.log first. Never overwrite results.
    if out.exists() and any(p.name != 'evaluation.log' for p in out.iterdir()):
        raise FileExistsError(f'Output contains existing artifacts: {out}')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'status.json', {'status': 'running', 'started_unix': time.time()})
    return out


def read_h5(path, key):
    with h5py.File(path, 'r') as handle:
        if key not in handle:
            raise ValueError(f'Missing {key} in {path}')
        return np.asarray(handle[key])


def validate_xyz(value, name):
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f'{name}: expected finite Nx3 XYZ; do not silently drop invalid points')
    return points


def nearest(query, points, workers=8, chunk_size=250000):
    if chunk_size <= 0 or workers == 0 or workers < -1:
        raise ValueError('invalid chunk size / workers')
    if not len(points):
        return np.full(len(query), np.inf), np.full(len(query), -1, dtype=np.int64)
    tree = cKDTree(points)
    distances = np.empty(len(query), dtype=np.float64)
    indices = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        distances[start:stop], indices[start:stop] = tree.query(query[start:stop], k=1, eps=0, workers=workers)
    return distances, indices


def distance_summary(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {
        'point_count': len(values), 'infinite_count': int(np.isinf(values).sum()),
        'mean_m': float(np.mean(finite)) if len(finite) == len(values) and len(values) else None,
        'finite_quantiles_m': dict(zip(['min', 'p50', 'p90', 'p95', 'p99', 'max'],
                                    map(float, np.quantile(finite, [0, .5, .9, .95, .99, 1])))) if len(finite) else {},
    }


def coverage_rows(distances, thresholds):
    if not len(distances):
        raise ValueError('empty evaluation denominator')
    thresholds = list(map(float, thresholds))
    if not thresholds or any(not np.isfinite(t) or t <= 0 for t in thresholds) or len(set(thresholds)) != len(thresholds):
        raise ValueError('thresholds must be unique finite positive metre distances')
    return [{'distance_m': t, 'covered_points': int(np.count_nonzero(distances <= t)),
             'total_points': len(distances), 'coverage': float(np.mean(distances <= t))}
            for t in sorted(thresholds)]


def load_support(args):
    """Same SLAM support and GT semantics/pose as the referenced ali-dev evaluator."""
    started = time.perf_counter()
    scene, semantic_scene, _ = args.scene
    constants_path = args.project_root / 'conceptgraph/dataset/replica_constants.py'
    constants = runpy.run_path(str(constants_path))
    names = constants['REPLICA_CLASSES']
    existing = np.asarray(constants['REPLICA_EXISTING_CLASSES'], dtype=np.int64)
    gt_root = args.replica_semantic_root / semantic_scene / 'Sequence_1'
    gt_path = gt_root / 'saved-maps-gt/pointclouds/pc_points.h5'
    embedding_path = gt_root / 'saved-maps-gt/pointclouds/pc_embeddings.h5'
    pose_path = gt_root / 'traj_w_c.txt'
    slam_path = args.slam_root / scene / 'rgb_cloud/pointclouds/pc_points.h5'
    gt_xyz = validate_xyz(read_h5(gt_path, 'pc_points'), 'semantic GT')
    pose = np.loadtxt(pose_path).reshape(-1, 4, 4)[0]
    if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError('invalid semantic GT first pose')
    gt_xyz = gt_xyz @ pose[:3, :3].T + pose[:3, 3]
    # Stream semantic embeddings instead of loading the entire 102-way array.
    with h5py.File(embedding_path, 'r') as handle:
        ds = handle['pc_embeddings']
        if ds.shape != (len(gt_xyz), len(names)):
            raise ValueError('GT embedding/point/ontology shape mismatch')
        gt_class = np.empty(len(gt_xyz), dtype=np.int32)
        for start in range(0, len(gt_xyz), args.chunk_size):
            values = np.asarray(ds[start:start + args.chunk_size])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite GT embeddings')
            gt_class[start:start + len(values)] = values.argmax(axis=1)
    if not np.isin(gt_class, existing).all():
        raise ValueError('GT class outside ali-dev existing ontology')
    xyz = validate_xyz(read_h5(slam_path, 'pc_points'), 'SLAM reference')
    if not len(xyz):
        raise ValueError('empty SLAM reference')
    gt_distance, ix = nearest(xyz, gt_xyz, args.workers, args.chunk_size)
    labels = gt_class[ix]
    excluded_ids = [names.index(name) for name in EXCLUSIONS[args.n_exclude]]
    foreground = ~np.isin(labels, excluded_ids)
    if not foreground.any():
        raise ValueError('no foreground reference points')
    meta = {
        'schema': 'replica_fixed_slam_support_v1', 'scene': scene, 'semantic_scene': semantic_scene,
        'reference_start': args.reference_start, 'reference_end': args.reference_end,
        'reference_stride': args.reference_stride,
        'reference_scope_note': 'Declared frame scope must match the supplied fixed SLAM cloud; its XYZ hash freezes support across methods.',
        'n_exclude': args.n_exclude, 'excluded_classes': EXCLUSIONS[args.n_exclude],
        'point_count_all': len(xyz), 'point_count_foreground': int(foreground.sum()),
        'support_xyz_sha256': array_sha(xyz), 'foreground_mask_sha256': array_sha(foreground),
        'semantic_transfer': 'exact k=1 SLAM to transformed semantic GT, no distance truncation; same as ali-dev',
        'coordinate_transform': 'semantic_xyz @ first_gt_pose.R.T + first_gt_pose.t; SLAM and map unchanged',
        'slam_to_semantic_gt_distance': distance_summary(gt_distance),
        'inputs': {str(p.resolve()): sha256(p) for p in [constants_path, gt_path, embedding_path, pose_path, slam_path]},
        'support_seconds': time.perf_counter() - started,
    }
    print(f"Fixed support: {len(xyz)} total, {int(foreground.sum())} foreground points", flush=True)
    return xyz, labels, foreground, names, meta


def load_prediction(args):
    path = Path(args.scene[2]).resolve()
    with gzip.open(path, 'rb') as handle:
        payload = pickle.load(handle)  # trusted local mapping artifact only
    if not isinstance(payload, dict) or 'objects' not in payload:
        raise ValueError('expected serialized mapping payload with objects')
    cfg = payload.get('cfg')
    if cfg is None:
        raise ValueError('map cfg required to verify frame scope')
    actual = {key: cfg.get(key) for key in ['start', 'end', 'stride', 'scene_id']}
    expected = {'start': args.reference_start, 'end': args.reference_end}
    if actual['stride'] is None or int(actual['stride']) < 1:
        raise ValueError('map must record a valid observation stride')
    for key, value in expected.items():
        # end=-1 is not silently assumed to mean 2000. Use resolved frame evidence.
        if key == 'end' and actual[key] == -1:
            trajectory = Path(str(cfg.get('dataset_root'))) / args.scene[0] / 'traj.txt'
            if not trajectory.is_file() or len(np.loadtxt(trajectory).reshape(-1, 4, 4)) != value:
                raise ValueError('end=-1 cannot be verified against reference end')
        elif actual[key] is None or int(actual[key]) != value:
            raise ValueError(f'map {key}={actual[key]} differs from fixed reference {value}; do not compare partial/full runs')
    if actual['scene_id'] not in (None, args.scene[0]):
        raise ValueError('map scene differs from reference scene')
    points, owners, records = [], [], []
    for index, obj in enumerate(payload['objects']):
        xyz = validate_xyz(obj['pcd_np'], f'map object {index}')
        points.append(xyz)
        owners.append(np.full(len(xyz), index, dtype=np.int32))
        records.append({'index': index, 'uid': str(obj.get('id', index)), 'point_count': len(xyz)})
    xyz = np.concatenate(points) if points else np.empty((0, 3), dtype=np.float32)
    owner = np.concatenate(owners) if owners else np.empty(0, dtype=np.int32)
    meta = {'map_path': str(path), 'map_sha256': sha256(path), 'frame_scope': actual,
            'map_stride_differs_from_reference': int(actual['stride']) != args.reference_stride,
            'stride_note': 'Reference support is fixed independently of map sampling. Different map strides are reported, not silently resampled; compare equal-budget methods for causal conclusions.',
            'objects': records, 'object_count': len(records), 'point_count': len(xyz)}
    return xyz, owner, meta


def prediction_on_support(args, support):
    xyz, owner, meta = load_prediction(args)
    d, ix = nearest(support, xyz, args.workers, args.chunk_size)
    labels = owner[ix] if len(xyz) else np.full(len(support), -1, dtype=np.int32)
    return d, labels, meta


def software():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'h5py': h5py.__version__, 'device': 'CPU', 'api_calls': 0}


def finish(out, result, started):
    result.update({'wall_seconds': time.perf_counter() - started, 'software': software()})
    write_json(out / 'results.json', result)
    write_json(out / 'status.json', {'status': 'completed', 'wall_seconds': result['wall_seconds']})


def fail(out, exc):
    write_json(out / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
