"""V3 containment arbitration. GT and history evaluation labels never enter this module."""
import math


def containment_route(qualities, report, mutual_threshold=.60):
    if len(qualities)!=2 or any(q not in {'CLEAN','CONTAMINATED','INSUFFICIENT',None} for q in qualities):
        return 'DEFER_QUALITY_UNKNOWN'
    if 'CONTAMINATED' in qualities:return 'DEFER_CONTAMINATED'
    if qualities!=['CLEAN','CLEAN']:return 'DEFER_QUALITY_UNKNOWN'
    a,b=float(report['a_in_b']),float(report['b_in_a'])
    if not all(math.isfinite(v) and 0<=v<=1 for v in [a,b]):raise ValueError('invalid containment')
    if max(a,b)<=.90:return 'NO_HIGH_CONTAINMENT'
    if min(a,b)>mutual_threshold:return 'MUTUAL_APPROVAL'
    return 'FRAGMENT_RESOLVER'


def fragment_context(report):
    source='A' if report['a_in_b']>report['b_in_a'] else 'B'
    return dict(a_in_b=report['a_in_b'],b_in_a=report['b_in_a'],distance_m=report['distance_m'],
        a_points=report['a_points'],b_points=report['b_points'],contained_node=source,containing_node='B' if source=='A' else 'A')
