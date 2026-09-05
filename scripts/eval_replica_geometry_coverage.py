#!/usr/bin/env python3
"""Fixed ali-dev foreground support coverage. No map reconstruction or CLIP."""
import time
import numpy as np
from replica_eval_common import (parser, begin_output, load_support, prediction_on_support,
                                 coverage_rows, distance_summary, write_csv, finish, fail)


def main():
    p = parser(__doc__)
    p.add_argument('--distances-m', nargs='+', type=float, default=[.025, .05, .10])
    args = p.parse_args()
    started = time.perf_counter()
    out = begin_output(args)
    try:
        coverage_rows(np.array([0.]), args.distances_m)  # validate before loading data
        xyz, classes, foreground, names, support = load_support(args)
        distances, owners, prediction = prediction_on_support(args, xyz)
        rows = coverage_rows(distances[foreground], args.distances_m)
        by_class = []
        for label in np.unique(classes[foreground]):
            mask = foreground & (classes == label)
            for row in coverage_rows(distances[mask], args.distances_m):
                by_class.append({'class_id': int(label), 'class_name': names[label], **row})
        write_csv(out / 'coverage.csv', rows, ['distance_m', 'covered_points', 'total_points', 'coverage'])
        write_csv(out / 'coverage_by_class.csv', by_class,
                  ['class_id', 'class_name', 'distance_m', 'covered_points', 'total_points', 'coverage'])
        np.savez_compressed(out / 'point_distances.npz', reference_index=np.flatnonzero(foreground),
                            distance_m=distances[foreground], gt_class=classes[foreground])
        result = {'metric': 'foreground_fixed_support_geometric_coverage',
                  'definition': 'fraction of ALL fixed foreground SLAM points with nearest predicted point distance <= d; no uncovered-point removal',
                  'main_distance_m': .05, 'coverage': rows, 'support': support, 'prediction': prediction,
                  'distance_summary': distance_summary(distances[foreground]),
                  'limitations': ['Recall-like surface support, not geometry precision or instance identity.',
                                 'Reference is the native fixed SLAM sample labeled by GT, not uniform GT mesh area.',
                                 'Nearby wrong-object surfaces can contribute; interpret together with instance PQ.']}
        finish(out, result, started)
        for row in rows:
            print(f"Coverage@{row['distance_m']*100:g}cm = {row['coverage']*100:.6f}% ({row['covered_points']}/{row['total_points']})", flush=True)
        print(f'Results: {out}', flush=True)
    except BaseException as exc:
        fail(out, exc)
        raise


if __name__ == '__main__':
    main()
