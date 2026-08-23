from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constraints import (
    CandidateTarget,
    ConstraintAction,
    ConstraintDecision,
    ConstraintEngine,
    ReplayMode,
    SparseRepairConstraint,
)
from .corruption import ControlledCorruptionController
from .index import ProvenanceIndex
from .materialize import ObservationMaterializer
from .models import CorruptionPlan, DependencyClosure


class SparseReplayError(RuntimeError):
    pass


class SparseReplayDeferred(SparseReplayError):
    def __init__(self, *, obs_uid: str, reason: str) -> None:
        super().__init__(f"constraint deferred at {obs_uid}: {reason}")
        self.obs_uid = obs_uid
        self.reason = reason


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


def _object_summary(obj: Mapping[str, Any], entity_uid: str | None = None) -> dict[str, Any]:
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


def _target_origin_observation(objects: Sequence[Mapping[str, Any]], index: int | None) -> str | None:
    if index is None or index < 0 or index >= len(objects):
        return None
    members = [str(item) for item in objects[index].get("obs_uids", ())]
    return members[0] if members else None


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
            for field in ("target_object_version_before", "target_object_version_after"):
                version_uid = association.get(field)
                if version_uid in self.provenance.object_versions:
                    version = self.provenance.get_object_version(str(version_uid))
                    lineage = version.get("lineage_uid")
                    if lineage:
                        lineages.add(str(lineage))
        result = tuple(sorted(lineages))
        self._obs_lineages[obs_uid] = result
        return result

    def _lineages_for_object(self, obj: Mapping[str, Any]) -> tuple[str, ...]:
        explicit = set(str(item) for item in obj.get("revision_lineage_uids", ()))
        for obs_uid in obj.get("obs_uids", ()):
            explicit.update(self._lineages_for_observation(str(obs_uid)))
        return tuple(sorted(explicit))

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
            index for index, obj in enumerate(objects) if str(obj.get("id")) == target_uid
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
            version_uids = list(
                association.get("candidate_object_version_uids") or ()
            )
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

    def _materialize(self, obs_uid: str) -> dict[str, Any]:
        preferred = self._preferred_entity(obs_uid)
        return self.materializer.materialize(obs_uid, preferred_uid=preferred)

    def _natural_details(self, detections: Any, objects: Any) -> tuple[list[int | None], Any]:
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
        threshold = float(self.cfg["sim_threshold"])
        candidates = [
            CandidateTarget.build(
                index=index,
                entity_uid=str(obj.get("id", index)),
                lineage_uids=self._lineages_for_object(obj),
                member_obs_uids=(str(item) for item in obj.get("obs_uids", ())),
                score=float(score_row[index]),
                eligible=float(score_row[index]) >= threshold,
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
    ) -> dict[str, Any]:
        rows = [_object_summary(obj) for obj in objects]
        membership = {
            row["entity_uid"]: row["member_observation_uids"] for row in rows
        }
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
            "postprocess_counts": dict(postprocess_counts),
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
    ) -> tuple[dict[str, Any], Any]:
        from conceptgraph.slam.slam_classes import DetectionList, MapEdgeMapping, MapObjectList

        mode = ReplayMode(mode)
        primitives = tuple(
            item
            if isinstance(item, SparseRepairConstraint)
            else SparseRepairConstraint.from_mapping(item)
            for item in constraints
        )
        engine = ConstraintEngine(primitives)
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
            raise ValueError("corruption plan is only valid in TEMPORAL_CORRUPTION mode")
        if historical_anchor_plan is not None:
            if mode not in {
                ReplayMode.ANCHOR_ONLY_REPAIR,
                ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            }:
                raise ValueError("historical anchor plan is only valid in proposed modes")
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
        map_edges = MapEdgeMapping(objects)
        by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_frame[_frame_index(str(row["frame_uid"]))].append(row)
        for values in by_frame.values():
            values.sort(
                key=lambda row: (int(row.get("filtered_det_idx", 0)), str(row["obs_uid"]))
            )

        decisions: list[dict[str, Any]] = []
        counts = {"denoise": 0, "filter": 0, "merge": 0}
        started = time.perf_counter()
        for frame in range(int(frame_start), int(frame_end) + 1):
            frame_rows = by_frame.get(frame, ())
            if frame_rows:
                detections = DetectionList(
                    [self._materialize(str(row["obs_uid"])) for row in frame_rows]
                )
                if frozen_recorded_frame is not None and frame == int(
                    frozen_recorded_frame
                ):
                    natural, score_matrix = self._frozen_recorded_frame_details(
                        frame_rows, objects
                    )
                    native_default_source = "RECORDED_FRAME_START_ASSOCIATION"
                else:
                    natural, score_matrix = self._natural_details(detections, objects)
                    native_default_source = "RECOMPUTED_NATIVE_MATCHER"
                applied = list(natural)
                historical_default = list(natural)
                constraint_decisions: list[ConstraintDecision] = [
                    ConstraintDecision(ConstraintAction.NO_CONSTRAINT)
                    for _ in frame_rows
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
                        candidates = self._candidate_targets(objects, score_matrix[index])
                        decision = engine.resolve_for_observation(
                            obs_uid=obs_uid,
                            event_uid=str(association["event_uid"]),
                            event_sequence=self.provenance.sequence(association),
                            natural_match=natural[index],
                            natural_candidates=candidates,
                            anchor_only=mode == ReplayMode.ANCHOR_ONLY_REPAIR,
                        )
                        constraint_decisions[index] = decision
                        if decision.action == ConstraintAction.DEFER:
                            raise SparseReplayDeferred(obs_uid=obs_uid, reason=decision.reason)
                        applied[index] = _resolved_constraint_match(
                            decision,
                            native_match=natural[index],
                            historical_default_match=historical_default[index],
                        )
                        if applied[index] is None and decision.created_entity_uid:
                            detections[index]["id"] = _uuid(
                                decision.created_entity_uid,
                                "revision-created:" + obs_uid,
                            )
                for index, row in enumerate(frame_rows):
                    obs_uid = str(row["obs_uid"])
                    association = self.provenance.get_association_for_obs(obs_uid)
                    detail = constraint_decisions[index]
                    candidate_rows = self._candidate_targets(objects, score_matrix[index])
                    decisions.append(
                        {
                            "frame_idx": frame,
                            "obs_uid": obs_uid,
                            "event_uid": str(association["event_uid"]),
                            "native_default_source": native_default_source,
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
                            "natural_candidates": [
                                {
                                    "index": item.index,
                                    "entity_uid": item.entity_uid,
                                    "lineage_uids": list(item.lineage_uids),
                                    "score": item.score,
                                    "eligible": item.eligible,
                                }
                                for item in candidate_rows[:10]
                            ],
                            "constraint": detail.as_dict(),
                            "constraint_hit": detail.constrained,
                            "constraint_overrode_natural": natural[index] != applied[index],
                            "constraint_overrode_native_natural": natural[index]
                            != applied[index],
                            "constraint_overrode_historical": historical_default[index]
                            != applied[index],
                            "constraint_changed_default_decision": historical_default[index]
                            != applied[index],
                            "intervention_overrode_natural": (
                                (controller is not None or historical_controller is not None)
                                and natural[index] != historical_default[index]
                            ),
                        }
                    )
                objects = self._merge(detections, objects, applied)
                map_edges.update_objects_list(objects)

            is_final = frame == (
                self.final_frame if final_scene_frame is None else int(final_scene_frame)
            )
            objects, map_edges = self._postprocess(
                objects=objects,
                map_edges=map_edges,
                frame=frame,
                is_final=is_final,
                counts=counts,
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
            intervention_count=(controller.applied_count if controller is not None else 0),
            historical_anchor_count=(
                historical_controller.applied_count
                if historical_controller is not None
                else 0
            ),
            snapshot_runtime_ms=snapshot_runtime_ms,
        )
        return state, objects

    def replay_global(
        self,
        *,
        mode: ReplayMode,
        constraints: Iterable[SparseRepairConstraint | Mapping[str, Any]] = (),
        corruption_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        historical_anchor_plan: CorruptionPlan | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state, _ = self.replay_global_with_objects(
            mode=mode,
            constraints=constraints,
            corruption_plan=corruption_plan,
            historical_anchor_plan=historical_anchor_plan,
        )
        return state

    def replay_global_with_objects(
        self,
        *,
        mode: ReplayMode,
        constraints: Iterable[SparseRepairConstraint | Mapping[str, Any]] = (),
        corruption_plan: CorruptionPlan | Mapping[str, Any] | None = None,
        historical_anchor_plan: CorruptionPlan | Mapping[str, Any] | None = None,
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
            index for index, row in enumerate(frame_rows) if str(row["obs_uid"]) == anchor_obs_uid
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
            applied = self._recorded_match_index(
                association, objects, obs_uid=obs_uid
            )
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
                    "natural_candidates": [
                        {
                            "index": item.index,
                            "entity_uid": item.entity_uid,
                            "lineage_uids": list(item.lineage_uids),
                            "score": item.score,
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
            members = set(str(item) for item in row.get("member_observation_uids") or ())
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
        historical_anchor_plan: CorruptionPlan | Mapping[str, Any] | None = None,
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
            historical_anchor_plan=historical_anchor_plan,
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
    ) -> dict[str, Any]:
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
            )
            overlaid, diagnostics = self._overlay_current(
                current_state=current_state,
                replay_state=replay_state,
                dependency_obs=scoped,
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
        return overlaid
