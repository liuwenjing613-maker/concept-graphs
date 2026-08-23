from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def _cluster_index(
    membership: Mapping[str, Iterable[str]],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, int]]:
    clusters = {
        str(entity): set(str(obs) for obs in members)
        for entity, members in membership.items()
        if list(members)
    }
    owner: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for entity, members in clusters.items():
        for obs in members:
            counts[obs] += 1
            owner.setdefault(obs, entity)
    return clusters, owner, dict(counts)


def membership_metrics(
    clean_membership: Mapping[str, Iterable[str]],
    candidate_membership: Mapping[str, Iterable[str]],
    *,
    observation_scope: Iterable[str] | None = None,
) -> dict[str, Any]:
    clean, clean_owner, _ = _cluster_index(clean_membership)
    candidate, candidate_owner, candidate_counts = _cluster_index(candidate_membership)
    scope = set(str(item) for item in observation_scope) if observation_scope is not None else set(clean_owner)
    scope &= set(clean_owner)
    precisions = []
    recalls = []
    f1s = []
    missing = 0
    for obs in sorted(scope):
        clean_cluster = clean[clean_owner[obs]] & scope
        candidate_entity = candidate_owner.get(obs)
        if candidate_entity is None:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            missing += 1
            continue
        candidate_cluster = candidate[candidate_entity] & scope
        overlap = len(clean_cluster & candidate_cluster)
        precision = overlap / len(candidate_cluster) if candidate_cluster else 0.0
        recall = overlap / len(clean_cluster) if clean_cluster else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    over_merge = 0
    for members in candidate.values():
        identities = {clean_owner[item] for item in members & scope if item in clean_owner}
        if len(identities) > 1:
            over_merge += 1
    splits: defaultdict[str, set[str]] = defaultdict(set)
    for obs in scope:
        if obs in candidate_owner:
            splits[clean_owner[obs]].add(candidate_owner[obs])
    over_split = sum(len(values) > 1 for values in splits.values())
    duplicates = sum(count > 1 for obs, count in candidate_counts.items() if obs in scope)
    return {
        "member_precision": float(np.mean(precisions)) if precisions else 1.0,
        "member_recall": float(np.mean(recalls)) if recalls else 1.0,
        "member_f1": float(np.mean(f1s)) if f1s else 1.0,
        "over_merge_count": int(over_merge),
        "over_split_count": int(over_split),
        "duplicate_count": int(duplicates),
        "missing_observation_count": int(missing),
        "observation_count": len(scope),
    }


def _aabb_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    a_min = np.asarray(first["aabb_min"], dtype=float)
    a_max = np.asarray(first["aabb_max"], dtype=float)
    b_min = np.asarray(second["aabb_min"], dtype=float)
    b_max = np.asarray(second["aabb_max"], dtype=float)
    intersection_extent = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    intersection = float(np.prod(intersection_extent))
    volume_a = float(np.prod(np.maximum(0.0, a_max - a_min)))
    volume_b = float(np.prod(np.maximum(0.0, b_max - b_min)))
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0 else 0.0


def geometry_metrics(
    clean_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    *,
    observation_scope: Iterable[str],
) -> dict[str, Any]:
    scope = set(str(item) for item in observation_scope)
    clean_rows = [
        row
        for row in clean_state.get("objects") or ()
        if set(row.get("member_observation_uids") or ()) & scope
    ]
    candidate_rows = list(candidate_state.get("objects") or ())
    ious = []
    center_errors = []
    extent_errors = []
    point_support = []
    matches = []
    for clean in clean_rows:
        clean_members = set(clean.get("member_observation_uids") or ()) & scope
        ranked = sorted(
            (
                (
                    len(clean_members & (set(row.get("member_observation_uids") or ()) & scope)),
                    row,
                )
                for row in candidate_rows
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] == 0:
            ious.append(0.0)
            center_errors.append(float("inf"))
            extent_errors.append(float("inf"))
            point_support.append(0.0)
            continue
        overlap, candidate = ranked[0]
        iou = _aabb_iou(clean, candidate)
        center_error = float(
            np.linalg.norm(
                np.asarray(clean["bbox_center"], dtype=float)
                - np.asarray(candidate["bbox_center"], dtype=float)
            )
        )
        extent_error = float(
            np.linalg.norm(
                np.asarray(clean["bbox_extent"], dtype=float)
                - np.asarray(candidate["bbox_extent"], dtype=float)
            )
        )
        support = float(candidate["n_points"]) / max(1, int(clean["n_points"]))
        ious.append(iou)
        center_errors.append(center_error)
        extent_errors.append(extent_error)
        point_support.append(support)
        matches.append(
            {
                "clean_entity_uid": clean["entity_uid"],
                "candidate_entity_uid": candidate["entity_uid"],
                "member_overlap": overlap,
                "bbox_iou": iou,
                "center_error": center_error,
                "extent_error": extent_error,
                "point_support": support,
            }
        )
    finite_centers = [item for item in center_errors if math.isfinite(item)]
    finite_extents = [item for item in extent_errors if math.isfinite(item)]
    return {
        "bbox_iou_to_clean": float(np.mean(ious)) if ious else 1.0,
        "center_error_to_clean": float(np.mean(finite_centers)) if finite_centers else float("inf"),
        "extent_error_to_clean": float(np.mean(finite_extents)) if finite_extents else float("inf"),
        "point_support": float(np.mean(point_support)) if point_support else 1.0,
        "matched_object_count": len(matches),
        "matches": matches,
    }


def edge_metrics(clean_state: Mapping[str, Any], candidate_state: Mapping[str, Any]) -> dict[str, Any]:
    def edge_map(state: Mapping[str, Any]) -> dict[tuple[str, str, str], int]:
        return {
            (
                str(edge["source_entity_uid"]),
                str(edge["relation"]),
                str(edge["target_entity_uid"]),
            ): int(edge.get("num_detections", 1))
            for edge in state.get("edges") or ()
        }

    clean_map = edge_map(clean_state)
    candidate_map = edge_map(candidate_state)
    clean = set(clean_map)
    candidate = set(candidate_map)
    overlap = len(clean & candidate)
    false_positives = len(candidate - clean)
    false_negatives = len(clean - candidate)
    precision = overlap / len(candidate) if candidate else (1.0 if not clean else 0.0)
    recall = overlap / len(clean) if clean else (1.0 if not candidate else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support_errors = [
        abs(clean_map[key] - candidate_map[key]) for key in clean & candidate
    ]
    support_mismatches = sum(error != 0 for error in support_errors)
    relation_set_match = precision == 1.0 and recall == 1.0
    support_match = support_mismatches == 0
    relation_state_match = relation_set_match and support_match
    active = set(str(item) for item in candidate_state.get("membership") or {})
    dangling = sum(source not in active or target not in active for source, _, target in candidate)
    return {
        "edge_set_precision_to_clean": precision,
        "edge_set_recall_to_clean": recall,
        "edge_set_f1_to_clean": f1,
        "edge_relation_match": relation_set_match,
        "edge_support_match": support_match,
        "edge_state_match": relation_state_match,
        "support_mismatch_edge_count": support_mismatches,
        "support_absolute_error": sum(support_errors),
        "max_support_absolute_error": max(support_errors, default=0),
        "clean_total_support": sum(clean_map.values()),
        "candidate_total_support": sum(candidate_map.values()),
        "dangling_edge_count": dangling,
        "clean_edge_count": len(clean),
        "candidate_edge_count": len(candidate),
        "true_positive_edge_count": overlap,
        "false_positive_edge_count": false_positives,
        "false_negative_edge_count": false_negatives,
        "informative": bool(clean or candidate),
    }


def evaluate_state(
    clean_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    *,
    affected_observations: Iterable[str],
) -> dict[str, Any]:
    affected = set(str(item) for item in affected_observations)
    return {
        "membership": membership_metrics(
            clean_state["membership"],
            candidate_state["membership"],
            observation_scope=affected,
        ),
        "membership_global": membership_metrics(
            clean_state["membership"], candidate_state["membership"]
        ),
        "geometry": geometry_metrics(
            clean_state, candidate_state, observation_scope=affected
        ),
        "relation": edge_metrics(clean_state, candidate_state),
        "cost": {
            "runtime_ms": float(candidate_state.get("runtime_ms", 0.0)),
            "num_replayed_observations": int(candidate_state.get("replayed_observations", 0)),
            "num_replayed_events": int(candidate_state.get("replayed_events", 0)),
            "total_events": int(candidate_state.get("total_events", 0)),
            "replay_fraction": (
                float(candidate_state.get("replayed_events", 0))
                / max(1, int(candidate_state.get("total_events", 0)))
            ),
        },
    }


def evaluate_case(
    *,
    case: Mapping[str, Any],
    clean_state: Mapping[str, Any],
    corrupted_state: Mapping[str, Any],
    refusion_state: Mapping[str, Any],
    local_state: Mapping[str, Any],
    global_state: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    affected = {
        str(obs)
        for members in (case.get("affected_clean_groups") or {}).values()
        for obs in members
    }
    methods = {
        "clean": evaluate_state(clean_state, clean_state, affected_observations=affected),
        "corrupted": evaluate_state(clean_state, corrupted_state, affected_observations=affected),
        "final_member_refusion": evaluate_state(
            clean_state, refusion_state, affected_observations=affected
        ),
        "counterfactual_local_replay": evaluate_state(
            clean_state, local_state, affected_observations=affected
        ),
        "global_replay_reference": evaluate_state(
            clean_state, global_state, affected_observations=affected
        ),
    }
    corrupted = methods["corrupted"]
    local = methods["counterfactual_local_replay"]
    global_reference = methods["global_replay_reference"]
    global_reference_executed = str(global_state.get("scope", "")) != "not_executed"
    local_f1 = local["membership"]["member_f1"]
    corrupt_f1 = corrupted["membership"]["member_f1"]
    runtime_ratio = (
        local["cost"]["runtime_ms"]
        / max(1e-9, global_reference["cost"]["runtime_ms"])
        if global_reference_executed
        else None
    )
    event_fraction_ratio = (
        local["cost"]["replay_fraction"]
        / max(1e-9, global_reference["cost"]["replay_fraction"])
        if global_reference_executed
        else None
    )
    acceptance = {
        "corruption_degrades_membership": corrupt_f1 < 1.0,
        "repair_improves_membership": local_f1 > corrupt_f1,
        "oracle_local_membership_f1_ge_0_99": local_f1 >= 0.99,
        "repaired_geometry_closer_or_equal": (
            local["geometry"]["bbox_iou_to_clean"]
            >= corrupted["geometry"]["bbox_iou_to_clean"]
        ),
        "outside_closure_changed_zero": not verification.get(
            "outside_closure_changed_entities"
        ),
        "hard_invariant_failures_zero": not verification.get("hard_invariant_failures"),
        "dangling_edges_zero": local["relation"]["dangling_edge_count"] == 0,
        "relation_recovery_matches_clean": local["relation"]["edge_state_match"],
    }
    return {
        "case_uid": case["case_uid"],
        "failure_type": case["failure_type"],
        "affected_observation_count": len(affected),
        "methods": methods,
        "local_vs_global": {
            "member_f1_difference": (
                local_f1 - global_reference["membership"]["member_f1"]
                if global_reference_executed
                else None
            ),
            "bbox_iou_difference": (
                local["geometry"]["bbox_iou_to_clean"]
                - global_reference["geometry"]["bbox_iou_to_clean"]
                if global_reference_executed
                else None
            ),
            "runtime_ratio": runtime_ratio,
            "event_fraction_ratio": event_fraction_ratio,
        },
        "safety": {
            "outside_closure_changed_entities": verification.get(
                "outside_closure_changed_entities", []
            ),
            "collateral_damage_count": len(
                verification.get("outside_closure_changed_entities", [])
            ),
            "hard_invariant_failures": verification.get("hard_invariant_failures", []),
        },
        "relation_diagnostics": {
            "informative": local["relation"]["informative"],
            "corruption_changes_relation": not corrupted["relation"]["edge_state_match"],
            "corruption_changes_edge_set": not corrupted["relation"][
                "edge_relation_match"
            ],
            "corruption_changes_support": not corrupted["relation"]["edge_support_match"],
            "local_matches_clean": local["relation"]["edge_state_match"],
            "global_reference_executed": global_reference_executed,
            "global_matches_clean": (
                global_reference["relation"]["edge_state_match"]
                if global_reference_executed
                else None
            ),
        },
        "acceptance": acceptance,
        "pass": all(acceptance.values()),
    }
