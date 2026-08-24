from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constraints import (
    CandidateTarget,
    ConstraintAction,
    ConstraintDecision,
    ConstraintEngine,
    ConstraintType,
    ReplayMode,
    SparseRepairConstraint,
)
from .corruption import ControlledCorruptionController
from .index import ProvenanceIndex
from .identity import (
    BoundaryAssessment,
    BoundaryDisposition,
    IdentityBoundary,
    IdentityRecord,
    assess_identity_boundaries,
    assess_protected_boundary,
    attach_observation_identity,
    record_for_object,
    write_identity_record,
)
from .materialize import ObservationMaterializer
from .models import CorruptionPlan, DependencyClosure


class SparseReplayError(RuntimeError):
    pass


class SparseReplayDeferred(SparseReplayError):
    def __init__(self, *, obs_uid: str, reason: str) -> None:
        super().__init__(f"constraint deferred at {obs_uid}: {reason}")
        self.obs_uid = obs_uid
        self.reason = reason


@dataclass(frozen=True)
class ReplayComponentPolicy:
    positive_lineage_redirect: bool = True
    create_association_boundary: bool = True
    create_postprocess_boundary: bool = True

    @classmethod
    def from_value(
        cls, value: "ReplayComponentPolicy | Mapping[str, Any] | None"
    ) -> "ReplayComponentPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "component_policy must be a mapping or ReplayComponentPolicy"
            )
        known = {
            "positive_lineage_redirect",
            "create_association_boundary",
            "create_postprocess_boundary",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown replay component policy keys: {sorted(unknown)}")
        return cls(**{key: bool(item) for key, item in value.items()})

    def as_dict(self) -> dict[str, bool]:
        return {
            "positive_lineage_redirect": self.positive_lineage_redirect,
            "create_association_boundary": self.create_association_boundary,
            "create_postprocess_boundary": self.create_postprocess_boundary,
        }


def _frame_index(frame_uid: str) -> int:
    return int(str(frame_uid).rsplit("_f", 1)[-1])


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _uuid(value: str | uuid.UUID | None, fallback: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return uuid.uuid5(uuid.NAMESPACE_URL, value)
    return uuid.uuid5(uuid.NAMESPACE_URL, fallback)


def _object_summary(
    obj: Mapping[str, Any], entity_uid: str | None = None
) -> dict[str, Any]:
    points = np.asarray(obj["pcd"].points, dtype=np.float64)
    minimum = points.min(axis=0) if len(points) else np.full(3, np.nan)
    maximum = points.max(axis=0) if len(points) else np.full(3, np.nan)
    bbox = obj["bbox"]
    members = tuple(sorted(set(str(item) for item in obj.get("obs_uids", ()))))
    uid = str(entity_uid or obj.get("id") or "entity_" + _json_hash(members)[:16])
    point_digest = hashlib.sha256(
        np.ascontiguousarray(points, dtype=np.float32).tobytes()
    ).hexdigest()
    clip_value = obj.get("clip_ft")
    if hasattr(clip_value, "detach"):
        clip_value = clip_value.detach().cpu().numpy()
    clip = (
        np.asarray(clip_value, dtype=np.float32).reshape(-1)
        if clip_value is not None
        else np.asarray([], dtype=np.float32)
    )
    class_histogram: dict[str, int] = {}
    for value in obj.get("class_id", ()):
        key = str(int(value))
        class_histogram[key] = class_histogram.get(key, 0) + 1
    return {
        "entity_uid": uid,
        "member_observation_uids": list(members),
        "num_detections": int(obj.get("num_detections", len(members))),
        "n_points": int(len(points)),
        "bbox_center": np.asarray(bbox.get_center(), dtype=np.float64).tolist(),
        "bbox_extent": np.asarray(bbox.extent, dtype=np.float64).tolist(),
        "aabb_min": minimum.tolist(),
        "aabb_max": maximum.tolist(),
        "class_name": str(obj.get("class_name", "")),
        "class_histogram": class_histogram,
        "clip_feature_digest": hashlib.sha256(
            np.ascontiguousarray(clip).tobytes()
        ).hexdigest(),
        "clip_feature_norm": float(np.linalg.norm(clip)),
        "point_digest": point_digest,
        "revision_lineage_uids": sorted(
            set(str(item) for item in obj.get("revision_lineage_uids", ()))
        ),
        "revision_provenance_lineage_uids": sorted(
            set(str(item) for item in obj.get("revision_provenance_lineage_uids", ()))
        ),
        "revision_identity_complete": bool(
            obj.get("revision_identity_complete", False)
        ),
    }


def _expand_observation_scope(
    scoped: set[str],
    membership: Mapping[str, Iterable[str]],
    *,
    entity_uids: Iterable[str] = (),
) -> tuple[int, set[str]]:
    """Expand a write scope to whole current ownership units."""

    requested = set(str(item) for item in entity_uids)
    added = 0
    expanded_entities: set[str] = set()
    for entity_uid, members_value in membership.items():
        members = set(str(item) for item in members_value)
        touches = str(entity_uid) in requested if requested else bool(members & scoped)
        if not touches or members <= scoped:
            continue
        before = len(scoped)
        scoped.update(members)
        added += len(scoped) - before
        expanded_entities.add(str(entity_uid))
    return added, expanded_entities


def _target_origin_observation(
    objects: Sequence[Mapping[str, Any]], index: int | None
) -> str | None:
    if index is None or index < 0 or index >= len(objects):
        return None
    members = [str(item) for item in objects[index].get("obs_uids", ())]
    return members[0] if members else None


def _threshold_semantics_trace(
    score_row: Sequence[float], threshold: float, native_match: int | None
) -> dict[str, Any]:
    from conceptgraph.slam.mapping import (
        SIMILARITY_THRESHOLD_COMPARATOR,
        similarity_exceeds_threshold,
    )

    finite = [float(value) for value in score_row if np.isfinite(float(value))]
    top1 = max(finite) if finite else None
    return {
        "comparator": SIMILARITY_THRESHOLD_COMPARATOR,
        "sim_threshold": float(threshold),
        "top1_score": top1,
        "top1_minus_threshold": (
            float(top1) - float(threshold) if top1 is not None else None
        ),
        "top1_strictly_exceeds_threshold": (
            similarity_exceeds_threshold(top1, threshold) if top1 is not None else False
        ),
        "equality_decision": "CREATE_OBJECT",
        "native_decision": (
            "MERGE_TO_OBJECT" if native_match is not None else "CREATE_OBJECT"
        ),
    }


def persistent_instance_boundary_reason(
    source_lineages: Iterable[str],
    target_lineages: Iterable[str],
    protected_lineages: Iterable[str],
) -> str | None:
    """Compatibility wrapper for callers with complete effective identities."""

    source = tuple(sorted(set(str(item) for item in source_lineages)))
    target = tuple(sorted(set(str(item) for item in target_lineages)))
    assessment = assess_protected_boundary(
        IdentityRecord.build(
            provenance_lineage_uids=source,
            effective_identity_uids=source,
            complete=bool(source),
            source="compatibility_source",
        ),
        IdentityRecord.build(
            provenance_lineage_uids=target,
            effective_identity_uids=target,
            complete=bool(target),
            source="compatibility_target",
        ),
        protected_lineages,
    )
    if assessment.disposition == BoundaryDisposition.VETO:
        return "persistent_create_instance_boundary"
    return None


@dataclass(frozen=True)
class BoundaryMatchResolution:
    resolved_match: int | None
    forbidden_indices: tuple[int, ...]
    unknown_indices: tuple[int, ...]
    overrode_match: bool
    candidate_assessments: tuple[tuple[int, BoundaryAssessment], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolved_match": self.resolved_match,
            "forbidden_indices": list(self.forbidden_indices),
            "unknown_indices": list(self.unknown_indices),
            "overrode_match": self.overrode_match,
            "candidate_assessments": [
                {"index": index, **assessment.as_dict()}
                for index, assessment in self.candidate_assessments
            ],
        }


def resolve_persistent_instance_boundary_match_detailed(
    default_match: int | None,
    candidates: Sequence[CandidateTarget],
    observation_identity: IdentityRecord,
    identity_boundaries: Iterable[IdentityBoundary],
) -> BoundaryMatchResolution:
    """Resolve only known DIFFERENT crossings; UNKNOWN never becomes a veto."""

    boundaries = tuple(identity_boundaries)
    if not boundaries:
        return BoundaryMatchResolution(default_match, (), (), False)
    assessments: list[tuple[int, BoundaryAssessment]] = []
    forbidden: list[int] = []
    unknown: list[int] = []
    for candidate in candidates:
        candidate_identity = IdentityRecord.build(
            provenance_lineage_uids=(
                candidate.provenance_lineage_uids or candidate.lineage_uids
            ),
            effective_identity_uids=candidate.lineage_uids,
            evidence_observation_uids=candidate.member_obs_uids,
            complete=candidate.identity_complete,
            source="active_candidate",
        )
        assessment = assess_identity_boundaries(
            observation_identity, candidate_identity, boundaries
        )
        assessments.append((candidate.index, assessment))
        if assessment.disposition == BoundaryDisposition.VETO:
            forbidden.append(candidate.index)
        elif assessment.disposition == BoundaryDisposition.UNKNOWN:
            unknown.append(candidate.index)
    forbidden_tuple = tuple(sorted(set(forbidden)))
    unknown_tuple = tuple(sorted(set(unknown)))
    if default_match not in set(forbidden_tuple):
        return BoundaryMatchResolution(
            default_match,
            forbidden_tuple,
            unknown_tuple,
            False,
            tuple(assessments),
        )
    dispositions = {index: assessment.disposition for index, assessment in assessments}
    alternative = next(
        (
            candidate.index
            for candidate in candidates
            if candidate.eligible
            and dispositions.get(candidate.index) == BoundaryDisposition.ALLOW
        ),
        None,
    )
    return BoundaryMatchResolution(
        alternative,
        forbidden_tuple,
        unknown_tuple,
        True,
        tuple(assessments),
    )


def resolve_persistent_instance_boundary_match(
    default_match: int | None,
    candidates: Sequence[CandidateTarget],
    observation_lineages: Iterable[str],
    protected_lineages: Iterable[str],
) -> tuple[int | None, tuple[int, ...], bool]:
    observation = tuple(sorted(set(str(item) for item in observation_lineages)))
    protected = tuple(sorted(set(str(item) for item in protected_lineages)))
    if not observation or not protected:
        return default_match, (), False
    forbidden = tuple(
        sorted(
            candidate.index
            for candidate in candidates
            if persistent_instance_boundary_reason(
                observation, candidate.lineage_uids, protected
            )
            is not None
        )
    )
    forbidden_set = set(forbidden)
    if default_match not in forbidden_set:
        return default_match, forbidden, False
    alternative = next(
        (
            candidate.index
            for candidate in candidates
            if candidate.index not in forbidden_set and candidate.eligible
        ),
        None,
    )
    return alternative, forbidden, True


def _resolved_constraint_match(
    decision: ConstraintDecision,
    *,
    native_match: int | None,
    historical_default_match: int | None,
) -> int | None:
    """Apply a constraint without confusing native and injected history.

    `NO_CONSTRAINT` must preserve the recorded/injected historical branch.  Once an
    active negative primitive says to keep the natural decision, however, the result
    must be the native matcher output, not the historical corruption that the
    primitive was introduced to override.
    """

    if decision.action == ConstraintAction.FORCE_TARGET:
        return decision.target_index
    if decision.action == ConstraintAction.FORCE_CREATE:
        return None
    if decision.action == ConstraintAction.KEEP_NATURAL:
        return native_match
    if decision.action == ConstraintAction.NO_CONSTRAINT:
        return historical_default_match
    raise SparseReplayError(f"constraint action is not executable: {decision.action}")


class SparseCounterfactualReplayEngine:
    """Replay native mapper decisions with optional sparse intervention primitives."""

    def __init__(self, provenance: ProvenanceIndex) -> None:
        self.provenance = provenance
        config_path = provenance.experiment_root / "config_params.json"
        with config_path.open(encoding="utf-8") as handle:
            self.cfg = json.load(handle)
        try:
            import torch

            self.cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.cfg["device"] = "cpu"
        self.materializer = ObservationMaterializer(provenance, self.cfg)
        self._obs_lineages: dict[str, tuple[str, ...]] = {}
        self._similarity_cache: dict[tuple[str, str], np.ndarray] = {}
        self._all_rows = [
            row
            for row in provenance.observation_rows
            if row.get("status") == "kept"
            and str(row["obs_uid"]) in provenance.association_for_obs
        ]
        self._all_rows.sort(
            key=lambda row: (
                _frame_index(str(row["frame_uid"])),
                int(row.get("filtered_det_idx", 0)),
                str(row["obs_uid"]),
            )
        )
        self.final_frame = max(
            (_frame_index(str(row["frame_uid"])) for row in self._all_rows), default=-1
        )

    def _lineages_for_observation(self, obs_uid: str) -> tuple[str, ...]:
        if obs_uid in self._obs_lineages:
            return self._obs_lineages[obs_uid]
        lineages: set[str] = set()
        association = self.provenance.association_for_obs.get(obs_uid)
        if association:
            for field in (
                "target_object_version_before",
                "target_object_version_after",
            ):
                version_uid = association.get(field)
                if version_uid in self.provenance.object_versions:
                    version = self.provenance.get_object_version(str(version_uid))
                    lineage = version.get("lineage_uid")
                    if lineage:
                        lineages.add(str(lineage))
        result = tuple(sorted(lineages))
        self._obs_lineages[obs_uid] = result
        return result

    def _identity_for_observation(self, obs_uid: str) -> IdentityRecord:
        lineages = self._lineages_for_observation(obs_uid)
        return IdentityRecord.build(
            provenance_lineage_uids=lineages,
            effective_identity_uids=lineages,
            evidence_observation_uids=(obs_uid,),
            complete=bool(lineages),
            source="immutable_observation_provenance",
        )

    def _identity_for_object(self, obj: Mapping[str, Any]) -> IdentityRecord:
        return record_for_object(obj, self._lineages_for_observation)

    def _lineages_for_object(self, obj: Mapping[str, Any]) -> tuple[str, ...]:
        return self._identity_for_object(obj).effective_identity_uids

    def _initialize_identity_metadata(self, objects: Any) -> None:
        for obj in objects:
            write_identity_record(obj, self._identity_for_object(obj))

    def _preferred_entity(self, obs_uid: str) -> str | None:
        association = self.provenance.get_association_for_obs(obs_uid)
        if str(association.get("decision", "")) == "CREATE_OBJECT":
            value = association.get("target_object_uid")
            return str(value) if value else None
        return None

    def _recorded_match_index(
        self,
        association: Mapping[str, Any],
        objects: Any,
        *,
        obs_uid: str,
    ) -> int | None:
        """Resolve one immutable frame-start association against the active state.

        ConceptGraphs computes every association in a frame against the object list
        that existed at frame start, then applies those decisions sequentially.  A
        mid-frame snapshot therefore has to carry the recorded decision across the
        boundary; recomputing it against objects created earlier in the same frame
        changes the mapper semantics.
        """

        recorded = str(association.get("decision", "")).upper()
        if recorded == "CREATE_OBJECT":
            return None
        if recorded != "MERGE_TO_OBJECT":
            raise SparseReplayError(
                f"unsupported recorded decision {recorded!r} for {obs_uid}"
            )

        target_uid = str(association.get("target_object_uid") or "")
        exact = [
            index
            for index, obj in enumerate(objects)
            if str(obj.get("id")) == target_uid
        ]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise SparseReplayError(
                f"recorded target UID is ambiguous for {obs_uid}: {target_uid}"
            )

        version_uid = association.get("target_object_version_before")
        origin = None
        lineage = None
        if version_uid in self.provenance.object_versions:
            version = self.provenance.get_object_version(str(version_uid))
            members = list(version.get("member_observation_uids") or ())
            origin = version.get("origin_observation_uid") or (
                members[0] if members else None
            )
            lineage = version.get("lineage_uid")

        origin_matches = [
            index
            for index, obj in enumerate(objects)
            if origin
            and str(origin) in set(str(item) for item in obj.get("obs_uids", ()))
        ]
        if len(origin_matches) == 1:
            return origin_matches[0]
        if len(origin_matches) > 1:
            raise SparseReplayError(
                f"recorded target origin is ambiguous for {obs_uid}: {origin}"
            )

        lineage_matches = [
            index
            for index, obj in enumerate(objects)
            if lineage and str(lineage) in set(self._lineages_for_object(obj))
        ]
        if len(lineage_matches) == 1:
            return lineage_matches[0]
        raise SparseReplayError(
            f"recorded frame-start target is not uniquely active for {obs_uid}"
        )

    def _rows_strictly_after_watermark(
        self,
        rows: Iterable[Mapping[str, Any]],
        watermark_event_sequence: int,
    ) -> list[Mapping[str, Any]]:
        """Exclude observations already materialized in the pre-anchor snapshot."""

        return [
            row
            for row in rows
            if self.provenance.sequence(
                self.provenance.get_association_for_obs(str(row["obs_uid"]))
            )
            > int(watermark_event_sequence)
        ]

    def _load_frozen_similarity(self, reference: Mapping[str, Any]) -> np.ndarray:
        relative = reference.get("path")
        key = str(reference.get("key") or "aggregate_sim")
        if not relative:
            raise SparseReplayError("frozen frame similarity reference is missing")
        root = self.provenance.experiment_root.resolve()
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SparseReplayError(
                f"frozen similarity path escapes the experiment root: {path}"
            ) from exc
        cache_key = (str(path), key)
        if cache_key not in self._similarity_cache:
            if not path.exists():
                raise SparseReplayError(f"frozen similarity file is missing: {path}")
            with np.load(path, allow_pickle=False) as payload:
                if key not in payload:
                    raise SparseReplayError(
                        f"frozen similarity key {key!r} is missing from {path}"
                    )
                matrix = np.asarray(payload[key], dtype=float)
            if matrix.ndim != 2 or not np.isfinite(matrix).all():
                raise SparseReplayError(
                    f"frozen similarity matrix is invalid: {path}:{key} {matrix.shape}"
                )
            self._similarity_cache[cache_key] = matrix
        return self._similarity_cache[cache_key]

    def _frozen_recorded_frame_details(
        self,
        frame_rows: Sequence[Mapping[str, Any]],
        objects: Any,
    ) -> tuple[list[int | None], np.ndarray]:
        """Rehydrate the native frame-start decisions and score rows exactly.

        Earlier same-frame creates are present in ``objects`` but were absent from
        the mapper's frozen matrix.  They intentionally receive no candidate score.
        Missing evidence is a hard failure; the runtime never substitutes a newly
        recomputed matrix at this causal boundary.
        """

        scores = np.full((len(frame_rows), len(objects)), -np.inf, dtype=float)
        natural: list[int | None] = []
        for row_index, row in enumerate(frame_rows):
            obs_uid = str(row["obs_uid"])
            association = self.provenance.get_association_for_obs(obs_uid)
            frozen = self._load_frozen_similarity(
                association.get("aggregate_sim_ref") or {}
            )
            detection_index = int(row.get("filtered_det_idx", -1))
            object_uids = list(association.get("object_uids_before") or ())
            version_uids = list(association.get("candidate_object_version_uids") or ())
            if not (0 <= detection_index < frozen.shape[0]):
                raise SparseReplayError(
                    f"frozen similarity row is out of range for {obs_uid}: "
                    f"{detection_index} vs {frozen.shape}"
                )
            if len(object_uids) != frozen.shape[1] or len(version_uids) != len(
                object_uids
            ):
                raise SparseReplayError(
                    f"frozen candidate columns disagree for {obs_uid}: "
                    f"uids={len(object_uids)} versions={len(version_uids)} "
                    f"matrix={frozen.shape[1]}"
                )
            for column, (object_uid, version_uid) in enumerate(
                zip(object_uids, version_uids)
            ):
                active_index = self._recorded_match_index(
                    {
                        "decision": "MERGE_TO_OBJECT",
                        "target_object_uid": object_uid,
                        "target_object_version_before": version_uid,
                    },
                    objects,
                    obs_uid=f"{obs_uid}:candidate:{column}",
                )
                value = float(frozen[detection_index, column])
                scores[row_index, active_index] = max(
                    scores[row_index, active_index], value
                )
            natural.append(
                self._recorded_match_index(association, objects, obs_uid=obs_uid)
            )
        return natural, scores

    def _materialize(
        self,
        obs_uid: str,
        *,
        geometry_contract: Mapping[str, Any] | None = None,
        geometry_audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        preferred = self._preferred_entity(obs_uid)
        detection = self.materializer.materialize(
            obs_uid,
            preferred_uid=preferred,
            geometry_contract=geometry_contract,
            geometry_audit=geometry_audit,
        )
        observation_identity = self._identity_for_observation(obs_uid)
        attach_observation_identity(
            detection,
            obs_uid=obs_uid,
            provenance_lineage_uids=observation_identity.provenance_lineage_uids,
        )
        return detection

    def _natural_details(
        self, detections: Any, objects: Any
    ) -> tuple[list[int | None], Any]:
        if not objects:
            return [None] * len(detections), np.empty((len(detections), 0), dtype=float)
        from conceptgraph.slam.mapping import (
            aggregate_similarities,
            compute_spatial_similarities,
            compute_visual_similarities,
            match_detections_to_objects,
        )

        spatial = compute_spatial_similarities(
            str(self.cfg["spatial_sim_type"]),
            detections,
            objects,
            float(self.cfg["downsample_voxel_size"]),
        )
        visual = compute_visual_similarities(detections, objects)
        aggregate = aggregate_similarities(
            str(self.cfg["match_method"]), float(self.cfg["phys_bias"]), spatial, visual
        )
        matches = match_detections_to_objects(
            aggregate, float(self.cfg["sim_threshold"])
        )
        if hasattr(aggregate, "detach"):
            matrix = aggregate.detach().cpu().numpy()
        else:
            matrix = np.asarray(aggregate)
        return list(matches), np.asarray(matrix, dtype=float)

    def _candidate_targets(
        self, objects: Any, score_row: Sequence[float]
    ) -> list[CandidateTarget]:
        from conceptgraph.slam.mapping import similarity_exceeds_threshold

        threshold = float(self.cfg["sim_threshold"])
        identities = [self._identity_for_object(obj) for obj in objects]
        candidates = [
            CandidateTarget.build(
                index=index,
                entity_uid=str(obj.get("id", index)),
                lineage_uids=identities[index].effective_identity_uids,
                provenance_lineage_uids=(identities[index].provenance_lineage_uids),
                identity_complete=identities[index].complete,
                member_obs_uids=(str(item) for item in obj.get("obs_uids", ())),
                score=float(score_row[index]),
                eligible=similarity_exceeds_threshold(score_row[index], threshold),
            )
            for index, obj in enumerate(objects)
            if np.isfinite(float(score_row[index]))
        ]
        candidates.sort(key=lambda item: (item.score, -item.index), reverse=True)
        return candidates

    def _merge(self, detections: Any, objects: Any, matches: list[int | None]) -> Any:
        from conceptgraph.slam.mapping import merge_obj_matches

        return merge_obj_matches(
            detection_list=detections,
            objects=objects,
            match_indices=matches,
            downsample_voxel_size=float(self.cfg["downsample_voxel_size"]),
            dbscan_remove_noise=bool(self.cfg["dbscan_remove_noise"]),
            dbscan_eps=float(self.cfg["dbscan_eps"]),
            dbscan_min_points=int(self.cfg["dbscan_min_points"]),
            spatial_sim_type=str(self.cfg["spatial_sim_type"]),
            device=str(self.cfg["device"]),
        )

    def _postprocess(
        self,
        *,
        objects: Any,
        map_edges: Any,
        frame: int,
        is_final: bool,
        counts: dict[str, int],
        merge_guard: Any | None = None,
        postprocess_trace: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, Any]:
        from conceptgraph.slam.utils import (
            denoise_objects,
            filter_objects,
            merge_objects,
            processing_needed,
        )

        if processing_needed(
            int(self.cfg.get("denoise_interval", 0)),
            bool(self.cfg.get("run_denoise_final_frame", False)),
            frame,
            is_final,
        ):
            objects = denoise_objects(
                downsample_voxel_size=float(self.cfg["downsample_voxel_size"]),
                dbscan_remove_noise=bool(self.cfg["dbscan_remove_noise"]),
                dbscan_eps=float(self.cfg["dbscan_eps"]),
                dbscan_min_points=int(self.cfg["dbscan_min_points"]),
                spatial_sim_type=str(self.cfg["spatial_sim_type"]),
                device=str(self.cfg["device"]),
                objects=objects,
            )
            map_edges.update_objects_list(objects)
            counts["denoise"] += 1
        if processing_needed(
            int(self.cfg.get("filter_interval", 0)),
            bool(self.cfg.get("run_filter_final_frame", False)),
            frame,
            is_final,
        ):
            objects = filter_objects(
                obj_min_points=int(self.cfg["obj_min_points"]),
                obj_min_detections=int(self.cfg["obj_min_detections"]),
                objects=objects,
                map_edges=map_edges,
            )
            map_edges.update_objects_list(objects)
            counts["filter"] += 1
        if processing_needed(
            int(self.cfg.get("merge_interval", 0)),
            bool(self.cfg.get("run_merge_final_frame", False)),
            frame,
            is_final,
        ):

            def record_merge_decision(
                source: Mapping[str, Any],
                target: Mapping[str, Any],
                overlap_ratio: Any,
                visual_similarity: Any,
                text_similarity: Any,
                decision: str,
                reject_reasons: Sequence[str],
                source_active: bool,
                target_active: bool,
                candidate_rank: int,
            ) -> None:
                if postprocess_trace is None:
                    return
                postprocess_trace.append(
                    {
                        "frame_idx": int(frame),
                        "operation": "OBJECT_MERGE_CANDIDATE",
                        "source_entity_uid": str(source.get("id")),
                        "target_entity_uid": str(target.get("id")),
                        "source_lineage_uids": list(self._lineages_for_object(source)),
                        "source_identity": self._identity_for_object(source).as_dict(),
                        "target_lineage_uids": list(self._lineages_for_object(target)),
                        "target_identity": self._identity_for_object(target).as_dict(),
                        "overlap_ratio": float(overlap_ratio),
                        "visual_similarity": float(visual_similarity),
                        "text_similarity": float(text_similarity),
                        "decision": str(decision),
                        "reject_reasons": [str(item) for item in reject_reasons],
                        "source_active": bool(source_active),
                        "target_active": bool(target_active),
                        "candidate_rank": int(candidate_rank),
                    }
                )

            objects = merge_objects(
                merge_overlap_thresh=float(self.cfg["merge_overlap_thresh"]),
                merge_visual_sim_thresh=float(self.cfg["merge_visual_sim_thresh"]),
                merge_text_sim_thresh=float(self.cfg["merge_text_sim_thresh"]),
                objects=objects,
                downsample_voxel_size=float(self.cfg["downsample_voxel_size"]),
                dbscan_remove_noise=bool(self.cfg["dbscan_remove_noise"]),
                dbscan_eps=float(self.cfg["dbscan_eps"]),
                dbscan_min_points=int(self.cfg["dbscan_min_points"]),
                spatial_sim_type=str(self.cfg["spatial_sim_type"]),
                device=str(self.cfg["device"]),
                do_edges=False,
                map_edges=map_edges,
                merge_guard=merge_guard,
                merge_decision_callback=record_merge_decision,
            )
            map_edges.update_objects_list(objects)
            counts["merge"] += 1
        return objects, map_edges

    def _state(
        self,
        *,
        objects: Any,
        mode: ReplayMode,
        scope: str,
        runtime_ms: float,
        replayed_observations: int,
        decisions: list[dict[str, Any]],
        postprocess_counts: Mapping[str, int],
        constraint_count: int,
        intervention_count: int,
        historical_anchor_count: int = 0,
        snapshot_runtime_ms: float = 0.0,
        postprocess_decisions: Sequence[Mapping[str, Any]] = (),
        component_policy: ReplayComponentPolicy | None = None,
        identity_boundaries: Sequence[IdentityBoundary] = (),
    ) -> dict[str, Any]:
        rows = [_object_summary(obj) for obj in objects]
        membership = {row["entity_uid"]: row["member_observation_uids"] for row in rows}
        hits = sum(bool(item.get("constraint_hit")) for item in decisions)
        overrides = sum(
            bool(item.get("constraint_changed_default_decision")) for item in decisions
        )
        native_overrides = sum(
            bool(item.get("constraint_overrode_native_natural")) for item in decisions
        )
        historical_overrides = sum(
            bool(item.get("constraint_overrode_historical")) for item in decisions
        )
        state = {
            "schema_version": "1.0.0",
            "identity_semantics_version": "2.0.0",
            "component_policy": (
                component_policy.as_dict() if component_policy is not None else None
            ),
            "mode": mode.value,
            "branch": mode.value.lower(),
            "scope": scope,
            "status": "COMPLETED",
            "membership": membership,
            "objects": rows,
            "edges": [],
            "runtime_ms": float(runtime_ms),
            "snapshot_runtime_ms": float(snapshot_runtime_ms),
            "replayed_observations": int(replayed_observations),
            "replayed_events": 2 * int(replayed_observations),
            "total_events": len(self.provenance.events),
            "decision_trace": decisions,
            "identity_boundaries": [item.as_dict() for item in identity_boundaries],
            "postprocess_counts": dict(postprocess_counts),
            "postprocess_decision_trace": [
                dict(item) for item in postprocess_decisions
            ],
            "persistent_create_instance_merge_veto_count": sum(
                "persistent_create_instance_boundary"
                in (item.get("reject_reasons") or ())
                for item in postprocess_decisions
            ),
            "persistent_create_instance_association_veto_count": sum(
                bool(
                    (item.get("persistent_create_instance_boundary") or {}).get(
                        "overrode_match"
                    )
                )
                for item in decisions
            ),
            "persistent_create_instance_association_unknown_count": sum(
                len(
                    (item.get("persistent_create_instance_boundary") or {}).get(
                        "unknown_indices", ()
                    )
                )
                for item in decisions
            ),
            "persistent_lineage_redirect_count": sum(
                (item.get("constraint") or {}).get("reason")
                == "persistent_lineage_redirect"
                for item in decisions
            ),
            "persistent_lineage_redirect_override_count": sum(
                (item.get("constraint") or {}).get("reason")
                == "persistent_lineage_redirect"
                and bool(item.get("constraint_changed_default_decision"))
                for item in decisions
            ),
            "geometry_restoration_hit_count": sum(
                bool((item.get("geometry_restoration") or {}).get("applied"))
                for item in decisions
            ),
            "geometry_similarity_recompute_count": sum(
                item.get("native_default_source") == "RECOMPUTED_AFTER_GEOMETRY_OVERLAY"
                for item in decisions
            ),
            "constraint_parsed_count": int(constraint_count),
            "constraint_hit_count": int(hits),
            "constraint_override_count": int(overrides),
            "constraint_native_override_count": int(native_overrides),
            "constraint_historical_override_count": int(historical_overrides),
            "intervention_count": int(intervention_count),
            "historical_anchor_replay_count": int(historical_anchor_count),
            "source_hashes": self.provenance.source_hashes(),
        }
        state["state_hash"] = _json_hash(membership)
        return state

    def _execute(
        self,
        *,
        mode: ReplayMode,
        rows: Sequence[Mapping[str, Any]],
        frame_start: int,
        frame_end: int,
        initial_objects: Any | None = None,
        constraints: Iterable[SparseRepairConstraint | Mapping[str, Any]] = (),
        corruption_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        historical_anchor_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        final_scene_frame: int | None = None,
        frozen_recorded_frame: int | None = None,
        scope: str,
        snapshot_runtime_ms: float = 0.0,
        component_policy: (ReplayComponentPolicy | Mapping[str, Any] | None) = None,
    ) -> tuple[dict[str, Any], Any]:
        from conceptgraph.slam.slam_classes import (
            DetectionList,
            MapEdgeMapping,
            MapObjectList,
        )

        mode = ReplayMode(mode)
        policy = ReplayComponentPolicy.from_value(component_policy)
        primitives = tuple(
            item
            if isinstance(item, SparseRepairConstraint)
            else SparseRepairConstraint.from_mapping(item)
            for item in constraints
        )
        engine = ConstraintEngine(primitives)
        identity_boundaries: list[IdentityBoundary] = []
        persistent_redirects: list[dict[str, Any]] = []
        if (
            mode == ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY
            and policy.positive_lineage_redirect
        ):
            for item in primitives:
                if item.constraint_type not in {
                    ConstraintType.ASSIGN_OBSERVATION,
                    ConstraintType.MUST_LINK,
                }:
                    continue
                association = self.provenance.get_association_for_obs(str(item.obs_uid))
                if str(association.get("decision")) != "CREATE_OBJECT":
                    continue
                source_lineages = self._lineages_for_observation(str(item.obs_uid))
                if not source_lineages:
                    continue
                persistent_redirects.append(
                    {
                        "primitive": item,
                        "source_lineages": source_lineages,
                        "anchor_sequence": self.provenance.sequence(association),
                    }
                )

        def target_indices_for_constraint(
            item: SparseRepairConstraint, active_objects: Sequence[Mapping[str, Any]]
        ) -> tuple[int, ...]:
            matches = []
            for index, obj in enumerate(active_objects):
                lineages = set(self._lineages_for_object(obj))
                members = set(str(value) for value in obj.get("obs_uids", ()))
                entity_uid = str(obj.get("id"))
                if (
                    item.target_lineage_uid in lineages
                    or item.target_origin_obs_uid in members
                    or (
                        item.target_entity_uid is not None
                        and item.target_entity_uid == entity_uid
                    )
                ):
                    matches.append(index)
            return tuple(matches)

        def persistent_merge_guard(
            source: Mapping[str, Any], target: Mapping[str, Any]
        ) -> str | None:
            assessment = assess_identity_boundaries(
                self._identity_for_object(source),
                self._identity_for_object(target),
                identity_boundaries,
            )
            if assessment.disposition == BoundaryDisposition.VETO:
                return "persistent_create_instance_boundary"
            return None

        controller = None
        historical_controller = None
        if mode == ReplayMode.TEMPORAL_CORRUPTION:
            if corruption_plan is None:
                raise ValueError("TEMPORAL_CORRUPTION requires a corruption plan")
            plan = (
                corruption_plan
                if isinstance(corruption_plan, CorruptionPlan)
                else CorruptionPlan.from_mapping(corruption_plan)
            )
            controller = ControlledCorruptionController(plan)
        elif corruption_plan is not None:
            raise ValueError(
                "corruption plan is only valid in TEMPORAL_CORRUPTION mode"
            )
        if historical_anchor_plan is not None:
            if mode not in {
                ReplayMode.ANCHOR_ONLY_REPAIR,
                ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            }:
                raise ValueError(
                    "historical anchor plan is only valid in proposed modes"
                )
            plan = (
                historical_anchor_plan
                if isinstance(historical_anchor_plan, CorruptionPlan)
                else CorruptionPlan.from_mapping(historical_anchor_plan)
            )
            historical_controller = ControlledCorruptionController(plan)

        if initial_objects is None:
            objects = MapObjectList()
        else:
            objects = copy.deepcopy(initial_objects)
        self._initialize_identity_metadata(objects)
        map_edges = MapEdgeMapping(objects)
        by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_frame[_frame_index(str(row["frame_uid"]))].append(row)
        for values in by_frame.values():
            values.sort(
                key=lambda row: (
                    int(row.get("filtered_det_idx", 0)),
                    str(row["obs_uid"]),
                )
            )

        decisions: list[dict[str, Any]] = []
        postprocess_decisions: list[dict[str, Any]] = []
        counts = {"denoise": 0, "filter": 0, "merge": 0}
        started = time.perf_counter()
        for frame in range(int(frame_start), int(frame_end) + 1):
            frame_rows = by_frame.get(frame, ())
            if frame_rows:
                geometry_by_index: list[SparseRepairConstraint | None] = []
                for row in frame_rows:
                    obs_uid = str(row["obs_uid"])
                    association = self.provenance.get_association_for_obs(obs_uid)
                    event_uid = str(association["event_uid"])
                    event_sequence = self.provenance.sequence(association)
                    active_geometry = [
                        item
                        for item in primitives
                        if item.constraint_type
                        == ConstraintType.RESTORE_OBSERVATION_GEOMETRY
                        and item.is_active(
                            obs_uid=obs_uid,
                            event_uid=event_uid,
                            event_sequence=event_sequence,
                        )
                        and (
                            mode == ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY
                            or (
                                mode == ReplayMode.ANCHOR_ONLY_REPAIR
                                and item.applies_at_event_uid == event_uid
                            )
                        )
                    ]
                    if len(active_geometry) > 1:
                        raise SparseReplayError(
                            f"multiple active geometry overlays for {obs_uid}"
                        )
                    geometry_by_index.append(
                        active_geometry[0] if active_geometry else None
                    )
                geometry_traces: list[dict[str, Any]] = [{} for _ in frame_rows]
                detections = DetectionList(
                    [
                        self._materialize(
                            str(row["obs_uid"]),
                            geometry_contract=(
                                geometry_by_index[index].geometry_contract
                                if geometry_by_index[index] is not None
                                else None
                            ),
                            geometry_audit=geometry_traces[index],
                        )
                        for index, row in enumerate(frame_rows)
                    ]
                )
                restored_indices = [
                    index
                    for index, item in enumerate(geometry_by_index)
                    if item is not None
                ]
                if frozen_recorded_frame is not None and frame == int(
                    frozen_recorded_frame
                ):
                    (
                        historical_default,
                        score_matrix,
                    ) = self._frozen_recorded_frame_details(frame_rows, objects)
                    natural = list(historical_default)
                    native_default_sources = [
                        "RECORDED_FRAME_START_ASSOCIATION" for _ in frame_rows
                    ]
                    if restored_indices:
                        recomputed_natural, recomputed_scores = self._natural_details(
                            detections, objects
                        )
                        for index in restored_indices:
                            natural[index] = recomputed_natural[index]
                            score_matrix[index, :] = recomputed_scores[index, :]
                            native_default_sources[
                                index
                            ] = "RECOMPUTED_AFTER_GEOMETRY_OVERLAY"
                else:
                    natural, score_matrix = self._natural_details(detections, objects)
                    historical_default = list(natural)
                    native_default_sources = [
                        (
                            "RECOMPUTED_AFTER_GEOMETRY_OVERLAY"
                            if index in restored_indices
                            else "RECOMPUTED_NATIVE_MATCHER"
                        )
                        for index in range(len(frame_rows))
                    ]
                applied = list(natural)
                constraint_decisions: list[ConstraintDecision] = [
                    ConstraintDecision(ConstraintAction.NO_CONSTRAINT)
                    for _ in frame_rows
                ]
                redirect_source_lineages: list[tuple[str, ...]] = [
                    () for _ in frame_rows
                ]
                boundary_observation_identities: list[IdentityRecord | None] = [
                    None for _ in frame_rows
                ]
                boundary_resolutions: list[BoundaryMatchResolution | None] = [
                    None for _ in frame_rows
                ]
                created_boundaries: list[dict[str, Any] | None] = [
                    None for _ in frame_rows
                ]
                if controller is not None:
                    applied = controller.apply(
                        frame_idx=frame,
                        detection_list=detections,
                        objects=objects,
                        original_match_indices=applied,
                    )
                    historical_default = list(applied)
                elif historical_controller is not None:
                    historical_default = historical_controller.apply(
                        frame_idx=frame,
                        detection_list=detections,
                        objects=objects,
                        original_match_indices=historical_default,
                    )
                    applied = list(historical_default)
                if mode in {
                    ReplayMode.ANCHOR_ONLY_REPAIR,
                    ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
                }:
                    for index, row in enumerate(frame_rows):
                        obs_uid = str(row["obs_uid"])
                        association = self.provenance.get_association_for_obs(obs_uid)
                        candidates = self._candidate_targets(
                            objects, score_matrix[index]
                        )
                        event_sequence = self.provenance.sequence(association)
                        decision = engine.resolve_for_observation(
                            obs_uid=obs_uid,
                            event_uid=str(association["event_uid"]),
                            event_sequence=event_sequence,
                            natural_match=natural[index],
                            natural_candidates=candidates,
                            anchor_only=mode == ReplayMode.ANCHOR_ONLY_REPAIR,
                        )
                        if (
                            decision.action == ConstraintAction.NO_CONSTRAINT
                            and mode == ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY
                            and persistent_redirects
                        ):
                            observation_lineages = set(
                                self._lineages_for_observation(obs_uid)
                            )
                            redirect_targets: set[int] = set()
                            redirect_uids: set[str] = set()
                            redirect_sources: set[str] = set()
                            redirect_error = ""
                            for specification in persistent_redirects:
                                primitive = specification["primitive"]
                                anchor_sequence = int(specification["anchor_sequence"])
                                if event_sequence <= anchor_sequence:
                                    continue
                                sources = set(specification["source_lineages"])
                                if not observation_lineages.intersection(sources):
                                    continue
                                matches = target_indices_for_constraint(
                                    primitive, objects
                                )
                                if len(matches) != 1:
                                    redirect_error = (
                                        "persistent_redirect_target_not_active"
                                        if not matches
                                        else "persistent_redirect_target_ambiguous"
                                    )
                                    break
                                redirect_targets.add(matches[0])
                                redirect_uids.add(primitive.constraint_uid)
                                redirect_sources.update(sources)
                            if redirect_error:
                                decision = ConstraintDecision(
                                    ConstraintAction.DEFER,
                                    constraint_uids=tuple(sorted(redirect_uids)),
                                    reason=redirect_error,
                                )
                            elif len(redirect_targets) > 1:
                                decision = ConstraintDecision(
                                    ConstraintAction.DEFER,
                                    constraint_uids=tuple(sorted(redirect_uids)),
                                    reason="persistent_redirect_targets_conflict",
                                )
                            elif redirect_targets:
                                decision = ConstraintDecision(
                                    ConstraintAction.FORCE_TARGET,
                                    target_index=next(iter(redirect_targets)),
                                    constraint_uids=tuple(sorted(redirect_uids)),
                                    reason="persistent_lineage_redirect",
                                )
                                redirect_source_lineages[index] = tuple(
                                    sorted(redirect_sources)
                                )
                        constraint_decisions[index] = decision
                        if decision.action == ConstraintAction.DEFER:
                            raise SparseReplayDeferred(
                                obs_uid=obs_uid, reason=decision.reason
                            )
                        applied[index] = _resolved_constraint_match(
                            decision,
                            native_match=natural[index],
                            historical_default_match=historical_default[index],
                        )
                        if (
                            applied[index] is None
                            and decision.action == ConstraintAction.FORCE_CREATE
                        ):
                            source_identity = self._identity_for_observation(obs_uid)
                            created_identity_uid = (
                                decision.created_identity_uid
                                or decision.created_lineage_uid
                                or "revision-lineage:" + obs_uid
                            )
                            detections[index]["id"] = _uuid(
                                decision.created_entity_uid,
                                "revision-created:" + obs_uid,
                            )
                            write_identity_record(
                                detections[index],
                                IdentityRecord.build(
                                    provenance_lineage_uids=(
                                        source_identity.provenance_lineage_uids
                                    ),
                                    effective_identity_uids=(created_identity_uid,),
                                    evidence_observation_uids=(obs_uid,),
                                    complete=source_identity.complete,
                                    source="explicit_create_instance",
                                ),
                            )
                            boundary_target_index = historical_default[index]
                            if boundary_target_index is None:
                                boundary_target_index = natural[index]
                            boundary_audit: dict[str, Any] = {
                                "boundary_created": False,
                                "created_identity_uid": created_identity_uid,
                                "target_index": boundary_target_index,
                            }
                            if (
                                mode == ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY
                                and decision.separate_from_identity_uids
                            ):
                                boundary = IdentityBoundary.build(
                                    left_identity_uids=(created_identity_uid,),
                                    right_identity_uids=(
                                        decision.separate_from_identity_uids
                                    ),
                                    evidence_refs=decision.constraint_uids
                                    + (
                                        str(association["event_uid"]),
                                        obs_uid,
                                    ),
                                    source="CREATE_INSTANCE_EXPLICIT_DIFFERENT",
                                )
                                if all(
                                    item.boundary_uid != boundary.boundary_uid
                                    for item in identity_boundaries
                                ):
                                    identity_boundaries.append(boundary)
                                boundary_audit = {
                                    "boundary_created": True,
                                    "boundary_source": "EXPLICIT_PAIR_EVIDENCE",
                                    **boundary.as_dict(),
                                }
                            elif (
                                mode == ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY
                                and boundary_target_index is not None
                            ):
                                target_identity = self._identity_for_object(
                                    objects[boundary_target_index]
                                )
                                if (
                                    target_identity.effective_identity_uids
                                    and created_identity_uid
                                    not in target_identity.effective_identity_uids
                                ):
                                    boundary = IdentityBoundary.build(
                                        left_identity_uids=(created_identity_uid,),
                                        right_identity_uids=(
                                            target_identity.effective_identity_uids
                                        ),
                                        evidence_refs=decision.constraint_uids
                                        + (
                                            str(association["event_uid"]),
                                            obs_uid,
                                        ),
                                    )
                                    if all(
                                        item.boundary_uid != boundary.boundary_uid
                                        for item in identity_boundaries
                                    ):
                                        identity_boundaries.append(boundary)
                                    boundary_audit = {
                                        "boundary_created": True,
                                        **boundary.as_dict(),
                                    }
                                else:
                                    boundary_audit[
                                        "reason"
                                    ] = "TARGET_IDENTITY_UNKNOWN_OR_ALREADY_SAME"
                            else:
                                boundary_audit["reason"] = "NO_OVERRIDDEN_TARGET"
                            created_boundaries[index] = boundary_audit
                        elif (
                            applied[index] is not None
                            and decision.action == ConstraintAction.FORCE_TARGET
                            and decision.reason
                            in {
                                "explicit_positive_constraint",
                                "persistent_lineage_redirect",
                            }
                        ):
                            source_identity = self._identity_for_observation(obs_uid)
                            target_identity = self._identity_for_object(
                                objects[applied[index]]
                            )
                            if target_identity.effective_identity_uids:
                                write_identity_record(
                                    detections[index],
                                    IdentityRecord.build(
                                        provenance_lineage_uids=(
                                            source_identity.provenance_lineage_uids
                                        ),
                                        effective_identity_uids=(
                                            target_identity.effective_identity_uids
                                        ),
                                        evidence_observation_uids=(obs_uid,),
                                        complete=source_identity.complete,
                                        source="explicit_identity_redirect",
                                    ),
                                )
                        if (
                            mode == ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY
                            and identity_boundaries
                            and policy.create_association_boundary
                            and decision.action == ConstraintAction.NO_CONSTRAINT
                        ):
                            observation_identity = self._identity_for_object(
                                detections[index]
                            )
                            boundary_candidates = list(candidates)
                            if applied[index] is not None and all(
                                item.index != applied[index]
                                for item in boundary_candidates
                            ):
                                obj = objects[applied[index]]
                                obj_identity = self._identity_for_object(obj)
                                boundary_candidates.append(
                                    CandidateTarget.build(
                                        index=applied[index],
                                        entity_uid=str(obj.get("id")),
                                        lineage_uids=(
                                            obj_identity.effective_identity_uids
                                        ),
                                        provenance_lineage_uids=(
                                            obj_identity.provenance_lineage_uids
                                        ),
                                        identity_complete=obj_identity.complete,
                                        member_obs_uids=obj.get("obs_uids", ()),
                                        score=float("-inf"),
                                        eligible=False,
                                    )
                                )
                            resolution = (
                                resolve_persistent_instance_boundary_match_detailed(
                                    applied[index],
                                    boundary_candidates,
                                    observation_identity,
                                    identity_boundaries,
                                )
                            )
                            boundary_observation_identities[
                                index
                            ] = observation_identity
                            boundary_resolutions[index] = resolution
                            applied[index] = resolution.resolved_match
                            if (
                                resolution.overrode_match
                                and resolution.resolved_match is None
                            ):
                                detections[index]["id"] = _uuid(
                                    None,
                                    "revision-boundary-created:" + obs_uid,
                                )
                            elif (
                                resolution.overrode_match
                                and resolution.resolved_match is not None
                            ):
                                alternative_identity = self._identity_for_object(
                                    objects[resolution.resolved_match]
                                )
                                write_identity_record(
                                    detections[index],
                                    IdentityRecord.build(
                                        provenance_lineage_uids=(
                                            observation_identity.provenance_lineage_uids
                                        ),
                                        effective_identity_uids=(
                                            alternative_identity.effective_identity_uids
                                        ),
                                        evidence_observation_uids=(obs_uid,),
                                        complete=observation_identity.complete,
                                        source="pair_boundary_alternative",
                                    ),
                                )
                for index, row in enumerate(frame_rows):
                    obs_uid = str(row["obs_uid"])
                    association = self.provenance.get_association_for_obs(obs_uid)
                    detail = constraint_decisions[index]
                    candidate_rows = self._candidate_targets(
                        objects, score_matrix[index]
                    )
                    decisions.append(
                        {
                            "frame_idx": frame,
                            "obs_uid": obs_uid,
                            "event_uid": str(association["event_uid"]),
                            "native_default_source": native_default_sources[index],
                            "geometry_restoration": geometry_traces[index] or None,
                            "natural_match": natural[index],
                            "historical_default_match": historical_default[index],
                            "applied_match": applied[index],
                            "natural_target_origin_obs_uid": _target_origin_observation(
                                objects, natural[index]
                            ),
                            "historical_default_target_origin_obs_uid": (
                                _target_origin_observation(
                                    objects, historical_default[index]
                                )
                            ),
                            "applied_target_origin_obs_uid": _target_origin_observation(
                                objects, applied[index]
                            ),
                            "threshold_semantics": _threshold_semantics_trace(
                                score_matrix[index],
                                float(self.cfg["sim_threshold"]),
                                natural[index],
                            ),
                            "natural_candidates": [
                                {
                                    "index": item.index,
                                    "entity_uid": item.entity_uid,
                                    "lineage_uids": list(item.lineage_uids),
                                    "provenance_lineage_uids": list(
                                        item.provenance_lineage_uids
                                    ),
                                    "identity_complete": item.identity_complete,
                                    "score": item.score,
                                    "score_minus_threshold": item.score
                                    - float(self.cfg["sim_threshold"]),
                                    "eligible": item.eligible,
                                }
                                for item in candidate_rows[:10]
                            ],
                            "persistent_lineage_redirect_source_lineages": list(
                                redirect_source_lineages[index]
                            ),
                            "created_identity_boundary": created_boundaries[index],
                            "persistent_create_instance_boundary": {
                                "observation_identity": (
                                    boundary_observation_identities[index].as_dict()
                                    if boundary_observation_identities[index]
                                    is not None
                                    else None
                                ),
                                "default_match": historical_default[index],
                                **(
                                    boundary_resolutions[index].as_dict()
                                    if boundary_resolutions[index] is not None
                                    else {
                                        "resolved_match": applied[index],
                                        "forbidden_indices": [],
                                        "unknown_indices": [],
                                        "overrode_match": False,
                                        "candidate_assessments": [],
                                    }
                                ),
                            },
                            "constraint": detail.as_dict(),
                            "constraint_hit": detail.constrained,
                            "constraint_overrode_natural": natural[index]
                            != applied[index],
                            "constraint_overrode_native_natural": natural[index]
                            != applied[index],
                            "constraint_overrode_historical": historical_default[index]
                            != applied[index],
                            "constraint_changed_default_decision": historical_default[
                                index
                            ]
                            != applied[index],
                            "intervention_overrode_natural": (
                                (
                                    controller is not None
                                    or historical_controller is not None
                                )
                                and natural[index] != historical_default[index]
                            ),
                        }
                    )
                objects = self._merge(detections, objects, applied)
                map_edges.update_objects_list(objects)

            is_final = frame == (
                self.final_frame
                if final_scene_frame is None
                else int(final_scene_frame)
            )
            objects, map_edges = self._postprocess(
                objects=objects,
                map_edges=map_edges,
                frame=frame,
                is_final=is_final,
                counts=counts,
                merge_guard=(
                    persistent_merge_guard
                    if mode == ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY
                    and identity_boundaries
                    and policy.create_postprocess_boundary
                    else None
                ),
                postprocess_trace=postprocess_decisions,
            )
        if controller is not None:
            controller.finalize()
        if historical_controller is not None:
            historical_controller.finalize()
        runtime_ms = (time.perf_counter() - started) * 1000.0
        state = self._state(
            objects=objects,
            mode=mode,
            scope=scope,
            runtime_ms=runtime_ms,
            replayed_observations=len(rows),
            decisions=decisions,
            postprocess_counts=counts,
            constraint_count=len(primitives),
            intervention_count=(
                controller.applied_count if controller is not None else 0
            ),
            historical_anchor_count=(
                historical_controller.applied_count
                if historical_controller is not None
                else 0
            ),
            snapshot_runtime_ms=snapshot_runtime_ms,
            postprocess_decisions=postprocess_decisions,
            component_policy=policy,
            identity_boundaries=identity_boundaries,
        )
        return state, objects

    def replay_global(
        self,
        *,
        mode: ReplayMode,
        constraints: Iterable[SparseRepairConstraint | Mapping[str, Any]] = (),
        corruption_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        historical_anchor_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        component_policy: (ReplayComponentPolicy | Mapping[str, Any] | None) = None,
    ) -> dict[str, Any]:
        state, _ = self.replay_global_with_objects(
            mode=mode,
            constraints=constraints,
            corruption_plan=corruption_plan,
            historical_anchor_plan=historical_anchor_plan,
            component_policy=component_policy,
        )
        return state

    def replay_global_with_objects(
        self,
        *,
        mode: ReplayMode,
        constraints: Iterable[SparseRepairConstraint | Mapping[str, Any]] = (),
        corruption_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        historical_anchor_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        component_policy: (ReplayComponentPolicy | Mapping[str, Any] | None) = None,
    ) -> tuple[dict[str, Any], Any]:
        """Return the state plus raw objects for strict payload-fidelity gates."""

        return self._execute(
            mode=mode,
            rows=self._all_rows,
            frame_start=0,
            frame_end=self.final_frame,
            constraints=constraints,
            corruption_plan=corruption_plan,
            historical_anchor_plan=historical_anchor_plan,
            component_policy=component_policy,
            scope="full_temporal_same_constraint",
        )

    def replay_prefix(self, *, anchor_frame: int) -> tuple[dict[str, Any], Any]:
        if anchor_frame <= 0:
            from conceptgraph.slam.slam_classes import MapObjectList

            objects = MapObjectList()
            state = self._state(
                objects=objects,
                mode=ReplayMode.NATURAL_REPLAY,
                scope="pre_anchor_prefix",
                runtime_ms=0.0,
                replayed_observations=0,
                decisions=[],
                postprocess_counts={"denoise": 0, "filter": 0, "merge": 0},
                constraint_count=0,
                intervention_count=0,
            )
            return state, objects
        rows = [
            row
            for row in self._all_rows
            if _frame_index(str(row["frame_uid"])) < int(anchor_frame)
        ]
        return self._execute(
            mode=ReplayMode.NATURAL_REPLAY,
            rows=rows,
            frame_start=0,
            frame_end=int(anchor_frame) - 1,
            final_scene_frame=self.final_frame,
            scope="pre_anchor_prefix",
        )

    def advance_recorded_frame_prefix(
        self,
        *,
        objects: Any,
        prefix_state: Mapping[str, Any],
        anchor_obs_uid: str,
    ) -> tuple[dict[str, Any], Any]:
        """Apply only same-frame events that precede the anchor.

        Association targets come from the immutable event ledger because the native
        mapper computes a frame's match matrix before sequentially merging its
        detections. This reconstructs history; it is never used for suffix decisions.
        """

        from conceptgraph.slam.slam_classes import DetectionList, MapEdgeMapping

        anchor_row = self.provenance.get_observation(anchor_obs_uid)
        frame = _frame_index(str(anchor_row["frame_uid"]))
        frame_rows = [
            row
            for row in self._all_rows
            if _frame_index(str(row["frame_uid"])) == frame
        ]
        positions = [
            index
            for index, row in enumerate(frame_rows)
            if str(row["obs_uid"]) == anchor_obs_uid
        ]
        if len(positions) != 1:
            raise SparseReplayError("anchor observation is not unique in its frame")
        prior_rows = frame_rows[: positions[0]]
        if not prior_rows:
            return dict(prefix_state), objects
        objects = copy.deepcopy(objects)
        map_edges = MapEdgeMapping(objects)
        decisions = list(prefix_state.get("decision_trace") or ())
        started = time.perf_counter()
        for row in prior_rows:
            obs_uid = str(row["obs_uid"])
            association = self.provenance.get_association_for_obs(obs_uid)
            detection = DetectionList([self._materialize(obs_uid)])
            natural, score_matrix = self._natural_details(detection, objects)
            applied = self._recorded_match_index(association, objects, obs_uid=obs_uid)
            candidates = self._candidate_targets(objects, score_matrix[0])
            decisions.append(
                {
                    "frame_idx": frame,
                    "obs_uid": obs_uid,
                    "event_uid": str(association["event_uid"]),
                    "native_default_source": "RECORDED_IMMUTABLE_PREFIX",
                    "natural_match": natural[0],
                    "applied_match": applied,
                    "recorded_prefix_decision": True,
                    "threshold_semantics": _threshold_semantics_trace(
                        score_matrix[0],
                        float(self.cfg["sim_threshold"]),
                        natural[0],
                    ),
                    "natural_candidates": [
                        {
                            "index": item.index,
                            "entity_uid": item.entity_uid,
                            "lineage_uids": list(item.lineage_uids),
                            "score": item.score,
                            "score_minus_threshold": item.score
                            - float(self.cfg["sim_threshold"]),
                            "eligible": item.eligible,
                        }
                        for item in candidates[:10]
                    ],
                    "constraint": ConstraintDecision(
                        ConstraintAction.NO_CONSTRAINT
                    ).as_dict(),
                    "constraint_hit": False,
                    "constraint_overrode_natural": False,
                    "intervention_overrode_natural": False,
                }
            )
            objects = self._merge(detection, objects, [applied])
            map_edges.update_objects_list(objects)
        added_runtime = (time.perf_counter() - started) * 1000.0
        state = self._state(
            objects=objects,
            mode=ReplayMode.NATURAL_REPLAY,
            scope="pre_anchor_event_prefix",
            runtime_ms=float(prefix_state.get("runtime_ms", 0.0)) + added_runtime,
            replayed_observations=int(prefix_state.get("replayed_observations", 0))
            + len(prior_rows),
            decisions=decisions,
            postprocess_counts=prefix_state.get("postprocess_counts")
            or {"denoise": 0, "filter": 0, "merge": 0},
            constraint_count=0,
            intervention_count=0,
        )
        return state, objects

    @staticmethod
    def _overlay_current(
        *,
        current_state: Mapping[str, Any],
        replay_state: Mapping[str, Any],
        dependency_obs: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        replay_rows = [
            dict(row)
            for row in replay_state.get("objects") or ()
            if set(str(item) for item in row.get("member_observation_uids") or ())
            & dependency_obs
        ]
        claimed = {
            str(item)
            for row in replay_rows
            for item in row.get("member_observation_uids") or ()
        }
        affected_uids = {str(row["entity_uid"]) for row in replay_rows}
        outside_rows = []
        partial = []
        for source in current_state.get("objects") or ():
            row = dict(source)
            uid = str(row["entity_uid"])
            members = set(
                str(item) for item in row.get("member_observation_uids") or ()
            )
            overlap = members & claimed
            if uid in affected_uids or overlap == members:
                continue
            if overlap:
                partial.append(uid)
                continue
            outside_rows.append(row)
        objects = outside_rows + replay_rows
        membership = {
            str(row["entity_uid"]): list(row.get("member_observation_uids") or ())
            for row in objects
        }
        value = dict(replay_state)
        value["objects"] = objects
        value["membership"] = membership
        value["scope"] = "dependency_bounded_suffix_overlay"
        value["state_hash"] = _json_hash(membership)
        diagnostics = {
            "dependency_observation_count": len(dependency_obs),
            "claimed_observation_count": len(claimed),
            "replaced_entity_count": len(replay_rows),
            "outside_entity_count": len(outside_rows),
            "partial_outside_overlap_entities": sorted(partial),
            "overlay_pass": not partial,
        }
        value["overlay_diagnostics"] = diagnostics
        return value, diagnostics

    def replay_local_from_snapshot(
        self,
        *,
        mode: ReplayMode,
        snapshot_objects: Any,
        snapshot_runtime_ms: float,
        anchor_frame: int,
        snapshot_watermark_event_sequence: int,
        closure: DependencyClosure,
        constraints: Iterable[SparseRepairConstraint | Mapping[str, Any]],
        current_state: Mapping[str, Any],
        snapshot_timing: Mapping[str, Any] | None = None,
        historical_anchor_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        component_policy: (ReplayComponentPolicy | Mapping[str, Any] | None) = None,
    ) -> dict[str, Any]:
        if mode not in {
            ReplayMode.ANCHOR_ONLY_REPAIR,
            ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        }:
            raise ValueError("local sparse replay requires a proposed repair mode")
        return self.replay_suffix_from_snapshot(
            mode=mode,
            snapshot_objects=snapshot_objects,
            snapshot_runtime_ms=snapshot_runtime_ms,
            anchor_frame=anchor_frame,
            snapshot_watermark_event_sequence=snapshot_watermark_event_sequence,
            closure=closure,
            constraints=constraints,
            current_state=current_state,
            snapshot_timing=snapshot_timing,
            historical_anchor_plan=historical_anchor_plan,
            component_policy=component_policy,
        )

    def replay_suffix_from_snapshot(
        self,
        *,
        mode: ReplayMode,
        snapshot_objects: Any,
        snapshot_runtime_ms: float,
        anchor_frame: int,
        snapshot_watermark_event_sequence: int,
        closure: DependencyClosure,
        current_state: Mapping[str, Any],
        constraints: Iterable[SparseRepairConstraint | Mapping[str, Any]] = (),
        corruption_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        historical_anchor_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        snapshot_timing: Mapping[str, Any] | None = None,
        component_policy: (ReplayComponentPolicy | Mapping[str, Any] | None) = None,
    ) -> dict[str, Any]:
        suffix_started = time.perf_counter()
        mode = ReplayMode(mode)
        if mode not in {
            ReplayMode.NATURAL_REPLAY,
            ReplayMode.TEMPORAL_CORRUPTION,
            ReplayMode.ANCHOR_ONLY_REPAIR,
            ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        }:
            raise ValueError("unsupported suffix replay mode")
        initial_scoped = set(str(item) for item in closure.obs_uids)
        scoped = set(initial_scoped)
        expanded_entities: set[str] = set()
        execute_reported_total_ms = 0.0
        execute_wall_total_ms = 0.0
        overlay_wall_total_ms = 0.0
        attempts: list[dict[str, Any]] = []

        def expand_from_current(entity_uids: Iterable[str] = ()) -> int:
            added, entities = _expand_observation_scope(
                scoped,
                current_state.get("membership") or {},
                entity_uids=entity_uids,
            )
            expanded_entities.update(entities)
            return added

        # A current entity is an atomic ownership unit for commit.  If the typed
        # historical closure touches only part of it, include its remaining members
        # before replay rather than accepting a partial overwrite.
        expand_from_current()
        expansion_iterations = 0
        max_expansion_iterations = 3
        while True:
            rows = self._rows_strictly_after_watermark(
                [
                    row
                    for row in self._all_rows
                    if str(row["obs_uid"]) in scoped
                    and _frame_index(str(row["frame_uid"])) >= int(anchor_frame)
                ],
                snapshot_watermark_event_sequence,
            )
            execute_started = time.perf_counter()
            replay_state, _ = self._execute(
                mode=mode,
                rows=rows,
                frame_start=int(anchor_frame),
                frame_end=self.final_frame,
                initial_objects=snapshot_objects,
                constraints=constraints,
                corruption_plan=corruption_plan,
                historical_anchor_plan=historical_anchor_plan,
                final_scene_frame=self.final_frame,
                frozen_recorded_frame=int(anchor_frame),
                scope="dependency_bounded_suffix",
                snapshot_runtime_ms=snapshot_runtime_ms,
                component_policy=component_policy,
            )
            execute_wall_ms = (time.perf_counter() - execute_started) * 1000.0
            execute_reported_ms = float(replay_state.get("runtime_ms", 0.0))
            execute_wall_total_ms += execute_wall_ms
            execute_reported_total_ms += execute_reported_ms
            overlay_started = time.perf_counter()
            overlaid, diagnostics = self._overlay_current(
                current_state=current_state,
                replay_state=replay_state,
                dependency_obs=scoped,
            )
            overlay_wall_ms = (time.perf_counter() - overlay_started) * 1000.0
            overlay_wall_total_ms += overlay_wall_ms
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "scoped_observation_count": len(scoped),
                    "replayed_observation_count": len(rows),
                    "execute_reported_ms": execute_reported_ms,
                    "execute_wall_ms": execute_wall_ms,
                    "overlay_wall_ms": overlay_wall_ms,
                    "partial_outside_overlap_entity_count": len(
                        diagnostics["partial_outside_overlap_entities"]
                    ),
                }
            )
            partial = diagnostics["partial_outside_overlap_entities"]
            if not partial or expansion_iterations >= max_expansion_iterations:
                break
            added = expand_from_current(partial)
            if not added:
                break
            expansion_iterations += 1
        overlaid["closure"] = closure.as_dict()
        overlaid["closure_expansion_count"] = len(expanded_entities)
        overlaid["closure_expansion_iterations"] = expansion_iterations
        overlaid["closure_expanded_entity_uids"] = sorted(expanded_entities)
        overlaid["snapshot_watermark_event_sequence"] = int(
            snapshot_watermark_event_sequence
        )
        overlaid["closure_initial_observation_count"] = len(initial_scoped)
        overlaid["closure_effective_observation_count"] = len(scoped)
        overlaid["closure_expanded_observation_count"] = len(scoped - initial_scoped)
        suffix_total_wall_ms = (time.perf_counter() - suffix_started) * 1000.0
        timing = {
            "suffix_total_wall_ms": suffix_total_wall_ms,
            "suffix_execute_reported_total_ms": execute_reported_total_ms,
            "suffix_execute_wall_total_ms": execute_wall_total_ms,
            "suffix_overlay_wall_total_ms": overlay_wall_total_ms,
            "suffix_orchestration_wall_ms": max(
                0.0,
                suffix_total_wall_ms - execute_wall_total_ms - overlay_wall_total_ms,
            ),
            "suffix_replay_attempt_count": len(attempts),
            "suffix_replay_attempts": attempts,
            "snapshot": dict(snapshot_timing or {}),
            "timing_basis": {
                "suffix_total_wall_ms": (
                    "FULL_SUFFIX_CALL_INCLUDING_SCOPE_EXPANSION_EXECUTION_AND_OVERLAY"
                ),
                "suffix_execute_reported_total_ms": (
                    "SUM_OF_MAPPER_EXECUTION_INTERNAL_TIMERS_ACROSS_ATTEMPTS"
                ),
                "suffix_execute_wall_total_ms": (
                    "SUM_OF_MAPPER_EXECUTION_CALL_WALL_TIMES_ACROSS_ATTEMPTS"
                ),
            },
        }
        overlaid["timing"] = timing
        # Keep the legacy field, but correct it to the complete suffix-call wall
        # time rather than only the last execution attempt.
        overlaid["runtime_ms"] = suffix_total_wall_ms
        return overlaid
