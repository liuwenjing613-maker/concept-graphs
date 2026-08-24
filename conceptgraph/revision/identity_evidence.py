"""Deterministic, oracle-free evidence contracts for identity repair proposals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .auto_constraints import IncidentBinding, forbidden_inference_paths
from .cases import canonical_obs_key
from .index import ProvenanceIndex


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_uid(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return prefix + digest[:20]


def _finite_vector(value: Any, length: int = 3) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != length:
        return None
    result = tuple(float(item) for item in value)
    return result if all(math.isfinite(item) for item in result) else None


def _frame_number(obs_uid: str) -> int | None:
    match = re.search(r"_f(\d{6})(?:_|$)", str(obs_uid))
    return int(match.group(1)) if match else None


def _representatives(values: Iterable[str], limit: int = 3) -> tuple[str, ...]:
    ordered = tuple(dict.fromkeys(str(value) for value in values))
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return (ordered[len(ordered) // 2],)
    indices = [
        round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)
    ]
    return tuple(ordered[index] for index in indices)


def bbox_pair_metrics(
    first_center: Sequence[float] | None,
    first_extent: Sequence[float] | None,
    second_center: Sequence[float] | None,
    second_extent: Sequence[float] | None,
) -> dict[str, float | None]:
    """Return stable AABB separation and overlap features for one identity pair."""

    center_a = _finite_vector(first_center)
    extent_a = _finite_vector(first_extent)
    center_b = _finite_vector(second_center)
    extent_b = _finite_vector(second_extent)
    if (
        center_a is None
        or center_b is None
        or extent_a is None
        or extent_b is None
        or any(item <= 0.0 for item in (*extent_a, *extent_b))
    ):
        return {
            "center_distance": None,
            "surface_gap": None,
            "aabb_iou": None,
            "anchor_containment_fraction": None,
            "candidate_containment_fraction": None,
            "normalized_center_distance": None,
        }

    delta = tuple(abs(left - right) for left, right in zip(center_a, center_b))
    center_distance = math.sqrt(sum(item * item for item in delta))
    intersection_axes = tuple(
        max(0.0, (left_extent + right_extent) / 2.0 - distance)
        for distance, left_extent, right_extent in zip(delta, extent_a, extent_b)
    )
    gap_axes = tuple(
        max(0.0, distance - (left_extent + right_extent) / 2.0)
        for distance, left_extent, right_extent in zip(delta, extent_a, extent_b)
    )
    surface_gap = math.sqrt(sum(item * item for item in gap_axes))
    intersection = math.prod(intersection_axes)
    volume_a = math.prod(extent_a)
    volume_b = math.prod(extent_b)
    union = volume_a + volume_b - intersection
    scale = math.sqrt(
        sum(((left + right) / 2.0) ** 2 for left, right in zip(extent_a, extent_b))
    )
    return {
        "center_distance": center_distance,
        "surface_gap": surface_gap,
        "aabb_iou": intersection / union if union > 0.0 else 0.0,
        "anchor_containment_fraction": intersection / volume_a,
        "candidate_containment_fraction": intersection / volume_b,
        "normalized_center_distance": center_distance / max(scale, 1e-12),
    }


def _class_history(
    provenance: ProvenanceIndex, members: Iterable[str], limit: int = 8
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for obs_uid in members:
        try:
            row = provenance.get_observation(str(obs_uid))
        except KeyError:
            continue
        label = str(row.get("class_name") or "UNKNOWN").strip().lower()
        counts[label or "unknown"] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _observation_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    raw_area = int(row.get("raw_mask_area") or 0)
    removed = int(row.get("removed_pixel_count") or 0)
    return {
        "obs_key": canonical_obs_key(str(row["obs_uid"])),
        "frame_number": _frame_number(str(row["obs_uid"])),
        "class_name": row.get("class_name"),
        "confidence": row.get("confidence"),
        "bbox_3d_center": row.get("bbox_3d_center"),
        "bbox_3d_extent": row.get("bbox_3d_extent"),
        "point_count": row.get("n_points"),
        "valid_depth_ratio": row.get("valid_depth_ratio"),
        "raw_mask_area": row.get("raw_mask_area"),
        "processed_mask_area": row.get("processed_mask_area"),
        "removed_pixel_count": row.get("removed_pixel_count"),
        "mask_loss_fraction": removed / raw_area if raw_area > 0 else None,
        "pre_dbscan_cluster_count": (row.get("pre_dbscan") or {}).get("cluster_count"),
        "pre_dbscan_largest_cluster_ratio": (row.get("pre_dbscan") or {}).get(
            "largest_cluster_ratio"
        ),
    }


def sanitize_machine_review(review: Mapping[str, Any]) -> dict[str, Any]:
    """Expose current-map detector evidence while excluding human/gold ownership."""

    incident = review.get("incident") or {}
    contract = review.get("evidence_contract") or {}
    current_outcome = review.get("final_outcome") or {}
    triggers = []
    for row in review.get("trigger_observations") or ():
        triggers.append(
            {
                "observation_alias": row.get("observation_alias"),
                "obs_key": canonical_obs_key(str(row["obs_uid"])),
                "frame_number": row.get("frame_number"),
                "status": row.get("status"),
                "class_name": row.get("class_name"),
                "confidence": row.get("confidence"),
                "raw_mask_area": row.get("raw_mask_area"),
                "processed_mask_area": row.get("processed_mask_area"),
                "removed_pixel_count": row.get("removed_pixel_count"),
                "valid_depth_ratio": row.get("valid_depth_ratio"),
                "point_count": row.get("n_points"),
                "pre_dbscan_cluster_count": (row.get("pre_dbscan") or {}).get(
                    "cluster_count"
                ),
                "pre_dbscan_largest_cluster_ratio": (row.get("pre_dbscan") or {}).get(
                    "largest_cluster_ratio"
                ),
                "bbox_3d_center": row.get("bbox_3d_center"),
                "bbox_3d_extent": row.get("bbox_3d_extent"),
            }
        )

    decisions = []
    for row in review.get("association_decisions") or ():
        candidates = []
        for candidate in row.get("candidates") or ():
            candidates.append(
                {
                    "rank": candidate.get("rank"),
                    "machine_object_alias": candidate.get("object_alias"),
                    "spatial_score": candidate.get("spatial_score"),
                    "visual_score": candidate.get("visual_score"),
                    "aggregate_score": candidate.get("aggregate_score"),
                }
            )
        decisions.append(
            {
                "obs_key": canonical_obs_key(str(row["obs_uid"])),
                "decision": row.get("decision"),
                "machine_target_alias": row.get("target_object_alias"),
                "top1_score": row.get("top1_score"),
                "top2_score": row.get("top2_score"),
                "margin": row.get("margin"),
                "similarity_threshold": row.get("sim_threshold"),
                "similarity_evidence_valid": row.get("similarity_evidence_valid"),
                "candidates": candidates[:3],
            }
        )

    current_objects = []
    for row in review.get("final_objects") or ():
        current_objects.append(
            {
                "machine_object_alias": row.get("object_alias"),
                "endpoint_role": row.get("endpoint_role"),
                "status": row.get("status"),
                "class_name": row.get("class_name"),
                "observed_class_histogram": row.get("observed_class_histogram"),
                "member_count": row.get("member_count"),
                "unique_frame_count": row.get("unique_frame_count"),
                "first_frame": row.get("first_frame"),
                "last_frame": row.get("last_frame"),
                "point_count": row.get("n_points"),
                "bbox_center": row.get("bbox_center"),
                "bbox_extent": row.get("bbox_extent"),
                "merged_from_count": len(
                    row.get("parent_or_merged_from_object_uids") or ()
                ),
            }
        )

    sanitized = {
        "packet_schema_version": review.get("schema_version"),
        "packet_checker": {
            "checker_id": review.get("checker_id"),
            "stage": review.get("stage"),
            "subtype": review.get("subtype"),
        },
        "incident_detector_summary": {
            "checker_ids": sorted(
                set(str(item) for item in incident.get("checker_ids") or ())
            ),
            "stages": sorted(set(str(item) for item in incident.get("stages") or ())),
            "subtypes": sorted(
                set(str(item) for item in incident.get("subtypes") or ())
            ),
        },
        "evidence_integrity": {
            "fidelity_status": contract.get("fidelity_status"),
            "artifact_hashes_match": contract.get("artifact_hashes_match"),
            "exact_map_linkage": contract.get("exact_final_map_linkage"),
            "critical_gap_count": len(contract.get("critical_gaps") or ()),
        },
        "machine_current_map_resolution": current_outcome.get(
            "machine_resolution_status"
        ),
        "trigger_observations": triggers,
        "association_decisions": decisions,
        "current_map_objects": current_objects,
    }
    forbidden = forbidden_inference_paths(sanitized)
    if forbidden:
        raise ValueError(
            "machine review leaked oracle-like fields: " + ", ".join(forbidden)
        )
    return sanitized


@dataclass(frozen=True)
class BuiltIdentityEvidence:
    inference_bundle: dict[str, Any]
    binding: IncidentBinding


class IdentityEvidenceBundleBuilder:
    """Construct finite candidate aliases and evidence without human answers."""

    def __init__(self, provenance: ProvenanceIndex, candidate_limit: int = 2) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.provenance = provenance
        self.candidate_limit = candidate_limit

    def _version_by_object(self, association: Mapping[str, Any]) -> dict[str, str]:
        versions = list(association.get("candidate_object_version_uids") or ())
        objects = list(association.get("object_uids_before") or ())
        return {
            str(object_uid): str(versions[index])
            for index, object_uid in enumerate(objects)
            if index < len(versions)
        }

    def _alias_binding(
        self, *, alias: str, object_uid: str, version_uid: str
    ) -> dict[str, Any]:
        version = self.provenance.get_object_version(version_uid)
        members = self.provenance.get_member_observations(version_uid)
        lineage_uid = version.get("lineage_uid")
        origin_obs_uid = version.get("origin_observation_uid") or (
            members[0] if members else None
        )
        return {
            "entity_uid": object_uid,
            "lineage_uid": lineage_uid,
            "origin_obs_uid": origin_obs_uid,
            "identity_uids": [lineage_uid] if lineage_uid else [],
            "provenance_lineage_uids": [lineage_uid] if lineage_uid else [],
            "complete": bool(
                version.get("object_uid") == object_uid
                and lineage_uid
                and origin_obs_uid
                and members
            ),
        }

    def build(
        self,
        *,
        case_uid: str,
        association_event_uid: str,
        machine_review: Mapping[str, Any] | None = None,
    ) -> BuiltIdentityEvidence:
        association = self.provenance.get_event(association_event_uid)
        anchor = self.provenance.get_observation(str(association["obs_uid"]))
        anchor_frame = _frame_number(str(anchor["obs_uid"]))
        version_by_object = self._version_by_object(association)
        aliases: dict[str, dict[str, Any]] = {}
        candidates = []

        for rank, score_row in enumerate(association.get("top_candidates") or (), 1):
            if rank > self.candidate_limit:
                break
            object_uid = str(score_row.get("object_uid") or "")
            version_uid = version_by_object.get(object_uid)
            if not object_uid or version_uid is None:
                continue
            alias = f"CANDIDATE_{rank}_CONTEXT"
            version = self.provenance.get_object_version(version_uid)
            members = self.provenance.get_member_observations(version_uid)
            member_frames = [
                value
                for value in (_frame_number(obs_uid) for obs_uid in members)
                if value is not None
            ]
            pair = bbox_pair_metrics(
                anchor.get("bbox_3d_center"),
                anchor.get("bbox_3d_extent"),
                version.get("bbox_center"),
                version.get("bbox_extent"),
            )
            candidates.append(
                {
                    "alias": alias,
                    "association_rank": rank,
                    "association_scores": {
                        "spatial": score_row.get("spatial_score"),
                        "visual": score_row.get("visual_score"),
                        "aggregate": score_row.get("aggregate_score"),
                        "threshold": association.get("sim_threshold"),
                        "aggregate_minus_threshold": (
                            float(score_row["aggregate_score"])
                            - float(association["sim_threshold"])
                            if score_row.get("aggregate_score") is not None
                            and association.get("sim_threshold") is not None
                            else None
                        ),
                    },
                    "current_object_summary": {
                        "class_name": version.get("class_name"),
                        "class_history": _class_history(self.provenance, members),
                        "member_count": len(members),
                        "unique_frame_count": version.get("unique_frame_count"),
                        "first_frame": min(member_frames) if member_frames else None,
                        "last_frame": max(member_frames) if member_frames else None,
                        "frames_since_last_observation": (
                            anchor_frame - max(member_frames)
                            if anchor_frame is not None and member_frames
                            else None
                        ),
                        "co_visible_in_anchor_frame": (
                            anchor_frame in member_frames
                            if anchor_frame is not None
                            else None
                        ),
                        "point_count": version.get("n_points"),
                        "bbox_center": version.get("bbox_center"),
                        "bbox_extent": version.get("bbox_extent"),
                        "representative_obs_keys": [
                            canonical_obs_key(obs_uid)
                            for obs_uid in _representatives(members)
                        ],
                    },
                    "anchor_candidate_geometry": pair,
                }
            )
            aliases[alias] = self._alias_binding(
                alias=alias, object_uid=object_uid, version_uid=version_uid
            )

        mapping_event = self.provenance.get_event(str(association["mapping_event_uid"]))
        observed = (
            "CREATE"
            if str(association.get("decision", "")).upper() == "CREATE_OBJECT"
            else "ASSOCIATE"
        )
        created_entity_uid = None
        created_identity_uid = None
        if mapping_event.get("event_type") == "OBJECT_CREATE":
            created_entity_uid = mapping_event.get("object_uid")
            outputs = list(mapping_event.get("output_object_version_uids") or ())
            if len(outputs) == 1:
                created_version = self.provenance.get_object_version(str(outputs[0]))
                created_identity_uid = created_version.get("lineage_uid")
                aliases["ANCHOR"] = self._alias_binding(
                    alias="ANCHOR",
                    object_uid=str(created_entity_uid),
                    version_uid=str(outputs[0]),
                )

        payload = {
            "schema_version": "1.0.0",
            "incident_alias": _sha256_uid(
                "identity_incident_", str(association_event_uid)
            ),
            "observed_current_decision": observed,
            "threshold_semantics": {
                "comparator": "STRICT_GREATER_THAN",
                "threshold": association.get("sim_threshold"),
                "top1_score": association.get("top1_score"),
                "top1_exceeds_threshold": (
                    float(association["top1_score"])
                    > float(association["sim_threshold"])
                    if association.get("top1_score") is not None
                    and association.get("sim_threshold") is not None
                    else None
                ),
            },
            "anchor": _observation_summary(anchor),
            "candidate_aliases": candidates,
            "allowed_identity_decisions": [
                "SAME_INSTANCE",
                "SEPARATE_MEMBER_GROUPS",
                "DEFER",
            ],
            "machine_detector_evidence": (
                sanitize_machine_review(machine_review)
                if machine_review is not None
                else None
            ),
            "contract": {
                "identifiers_are_finite_aliases": True,
                "raw_entity_uids_excluded_from_inference": True,
                "human_answers_excluded": True,
                "current_corrupted_map_is_observable": True,
                "candidate_count": len(candidates),
            },
        }
        forbidden = forbidden_inference_paths(payload)
        if forbidden:
            raise ValueError(
                "identity bundle leaked oracle-like fields: " + ", ".join(forbidden)
            )
        payload["bundle_uid"] = _sha256_uid("identity_evidence_", payload)

        binding = IncidentBinding.from_mapping(
            {
                "case_uid": case_uid,
                "obs_uid": str(association["obs_uid"]),
                "obs_key": canonical_obs_key(str(association["obs_uid"])),
                "event_uid": str(association_event_uid),
                "event_sequence": self.provenance.sequence(association),
                "observed_current_decision": observed,
                "aliases": aliases,
                "created_entity_uid": created_entity_uid,
                "created_identity_uid": created_identity_uid,
                "evidence_refs": [
                    payload["bundle_uid"],
                    payload["incident_alias"],
                ],
            }
        )
        return BuiltIdentityEvidence(inference_bundle=payload, binding=binding)
