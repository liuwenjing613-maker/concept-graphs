#!/usr/bin/env python3
"""Class-agnostic foreground PQ on fixed SLAM support; diagnostic, not official benchmark PQ."""
import json
from pathlib import Path
import time

import numpy as np
from plyfile import PlyData

from replica_eval_common import (parser, begin_output, load_support, prediction_on_support, nearest,
                                 validate_xyz, sha256, array_sha, distance_summary, write_csv, finish, fail)


def instance_pq(gt, pred, pred_count):
    """Disjoint point partitions; strict IoU>.5 gives unique matches without Hungarian.

    -1 is void in GT, abstention/uncovered in prediction. GT void is subtracted
    from unions. An unmatched prediction >50% on GT void is ignored (PQ convention).
    Zero-support map objects are counted as FP, explicitly extending point PQ to
    penalize completely invisible/duplicate map instances instead of hiding them.
    """
    gt, pred = np.asarray(gt, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    if gt.shape != pred.shape or gt.ndim != 1 or np.any(gt < -1) or np.any(pred < -1):
        raise ValueError('invalid partition arrays')
    if pred_count < 0 or np.any(pred >= pred_count):
        raise ValueError('prediction IDs outside map index range')
    valid = gt >= 0
    gt_ids, counts = np.unique(gt[valid], return_counts=True)
    if not len(gt_ids):
        raise ValueError('no valid GT instances; PQ is undefined')
    gindex = np.full(len(gt), -1, dtype=np.int64)
    gindex[valid] = np.searchsorted(gt_ids, gt[valid])
    has_pred = pred >= 0
    p_all = np.bincount(pred[has_pred], minlength=pred_count)
    p_void = np.bincount(pred[has_pred & ~valid], minlength=pred_count)
    p_valid = p_all - p_void
    both = valid & has_pred
    matrix = np.bincount(gindex[both] * pred_count + pred[both],
                         minlength=len(gt_ids)*pred_count).reshape(len(gt_ids), pred_count)
    union = counts[:, None] + p_valid[None, :] - matrix
    iou = np.divide(matrix, union, out=np.zeros(matrix.shape, dtype=float), where=union > 0)
    gg, pp = np.nonzero(iou > .5)
    if len(np.unique(gg)) != len(gg) or len(np.unique(pp)) != len(pp):
        raise AssertionError('IoU>.5 must be one-to-one for disjoint partitions')
    ignored = (p_all > 0) & (p_void > .5 * p_all)
    ignored[pp] = False
    tp = len(gg)
    fp = pred_count - tp - int(ignored.sum())
    fn = len(gt_ids) - tp
    sum_iou = float(iou[gg, pp].sum())
    denominator = tp + .5 * fp + .5 * fn
    result = {'pq': sum_iou / denominator, 'rq': tp / denominator,
              'sq': sum_iou / tp if tp else 0., 'tp': tp, 'fp': fp, 'fn': fn,
              'sum_matched_iou': sum_iou, 'gt_instances': len(gt_ids),
              'predicted_map_instances': pred_count, 'ignored_void_instances': int(ignored.sum()),
              'zero_support_instances_counted_as_fp': int(np.count_nonzero(p_all == 0)),
              'gt_point_count': int(valid.sum()), 'uncovered_gt_points': int(np.count_nonzero(valid & ~has_pred)),
              'matched': [{'gt_id': int(gt_ids[g]), 'pred_index': int(p), 'iou': float(iou[g,p])}
                          for g,p in zip(gg,pp)]}
    tables = {'intersection': matrix, 'iou': iou, 'gt_ids': gt_ids, 'gt_area': counts,
              'pred_area_all': p_all, 'pred_area_valid': p_valid, 'pred_void_area': p_void,
              'ignored_void_predictions': ignored}
    return result, tables


def main():
    p = parser(__doc__)
    p.add_argument('--replica-ssg-root', type=Path, required=True)
    p.add_argument('--gt-transfer-max-distance-m', type=float, default=.03)
    p.add_argument('--prediction-max-distance-m', type=float, default=.05)
    args = p.parse_args()
    started = time.perf_counter()
    out = begin_output(args)
    try:
        if not all(np.isfinite(t) and t > 0 for t in [args.gt_transfer_max_distance_m, args.prediction_max_distance_m]):
            raise ValueError('distance thresholds must be finite and positive')
        xyz, classes, foreground, names, support = load_support(args)
        mesh = args.replica_ssg_root / 'Replica/data' / args.scene[1] / 'labels.instances.annotated.v2.ply'
        labels_path = args.replica_ssg_root / 'files/objects.json'
        vertices = PlyData.read(str(mesh))['vertex'].data
        if not {'x','y','z','objectId'}.issubset(vertices.dtype.names):
            raise ValueError('instance mesh needs x,y,z,objectId vertex properties')
        gt_xyz = validate_xyz(np.column_stack([vertices[k] for k in 'xyz']), 'instance GT')
        raw_ids = np.asarray(vertices['objectId'])
        if raw_ids.dtype.kind not in 'iu' or np.any(raw_ids < 0):
            raise ValueError('GT objectId must be nonnegative integer, not semantic class')
        scans = [s for s in json.loads(labels_path.read_text())['scans'] if s['scan'] == args.scene[1]]
        if len(scans) != 1:
            raise ValueError('missing or ambiguous GT scene metadata')
        label_dict = {int(o['id']): o['label'] for o in scans[0]['objects']}
        if not set(map(int, np.unique(raw_ids))).issubset(label_dict):
            raise ValueError('instance GT IDs absent from metadata')
        dgt, ix = nearest(xyz, gt_xyz, args.workers, args.chunk_size)
        transferred = raw_ids[ix].astype(np.int64)
        known_ids = [key for key, label in label_dict.items() if str(label).strip().lower() not in {'unknown','unlabeled','unlabelled','void'}]
        valid = foreground & (dgt <= args.gt_transfer_max_distance_m) & np.isin(transferred, known_ids)
        validity_ratio = float(valid.sum() / foreground.sum())
        # Do not publish apparently strong scores on a badly aligned tiny subset.
        if validity_ratio < .95:
            raise ValueError(f'Only {validity_ratio:.2%} foreground has valid instance GT: inspect alignment/IDs before reporting PQ')
        gt = np.where(valid, transferred, -1)
        dpred, owner, prediction = prediction_on_support(args, xyz)
        pred = np.where(dpred <= args.prediction_max_distance_m, owner, -1)
        metrics, tables = instance_pq(gt, pred, prediction['object_count'])
        np.savez_compressed(out / 'overlap_tables.npz', **tables)
        np.savez_compressed(out / 'point_assignments.npz', gt_instance=gt, pred_instance=pred,
                            prediction_distance_m=dpred, instance_gt_distance_m=dgt,
                            native_foreground=foreground)
        write_csv(out / 'instance_pq.csv', [{k: metrics[k] for k in ['pq','rq','sq','tp','fp','fn','gt_instances','predicted_map_instances']}],
                  ['pq','rq','sq','tp','fp','fn','gt_instances','predicted_map_instances'])
        write_csv(out / 'matches.csv', metrics['matched'], ['gt_id','pred_index','iou'])
        gt_rows = [{'gt_id': int(i), 'gt_label': label_dict[int(i)], 'reference_points': int(n)} for i,n in zip(tables['gt_ids'], tables['gt_area'])]
        write_csv(out / 'gt_instances.csv', gt_rows, ['gt_id','gt_label','reference_points'])
        result = {'metric': 'class_agnostic_foreground_fixed_support_PQ_v1', 'metrics': metrics,
                  'support': support, 'prediction': prediction,
                  'instance_reference': {'mesh': str(mesh.resolve()), 'mesh_sha256': sha256(mesh),
                      'objects_json_sha256': sha256(labels_path), 'gt_partition_sha256': array_sha(gt),
                      'valid_foreground_ratio': validity_ratio, 'unassigned_foreground_points': int(foreground.sum()-valid.sum()),
                      'transfer_max_distance_m': args.gt_transfer_max_distance_m,
                      'distance_summary': distance_summary(dgt[foreground])},
                  'protocol': {'matching': 'strict IoU > 0.5, unique one-to-one; no predicted class comparison',
                      'prediction_projection_max_distance_m': args.prediction_max_distance_m,
                      'uncovered_gt': 'retained in GT area and FN; never remove based on prediction coverage',
                      'void': 'native excluded classes / unavailable instance GT; subtract from union; unmatched pred with >50% assigned support on void ignored',
                      'zero_support_map_objects': 'count as FP, including empty and duplicate objects; conservative extension of point-partition PQ',
                      'class_aggregation': 'all foreground instances pooled as one category, not class-macro PQ',
                      'minimum_gt_instance_size': 1, 'units': 'PQ/RQ/SQ fractions [0,1]'},
                  'limitations': ['Diagnostic protocol, not official Replica/ScanNet PQ.',
                      'GT instance IDs transferred from annotated sampled surface by bounded NN; not exact mesh-face intersection.',
                      'Fixed SLAM visibility and sampling density determine evaluation support.',
                      'Coincident predicted surfaces use the exact NN readout; zero-support duplicates remain FP.']}
        finish(out, result, started)
        print(f"CA-PQ={metrics['pq']*100:.6f}% RQ={metrics['rq']*100:.6f}% SQ={metrics['sq']*100:.6f}% TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}", flush=True)
        print(f'Results: {out}', flush=True)
    except BaseException as exc:
        fail(out, exc)
        raise


if __name__ == '__main__':
    main()
