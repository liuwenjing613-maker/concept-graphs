from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .cases import (
    apply_controlled_membership_corruption,
    canonical_membership,
    invert_membership,
    stable_entity_uid,
)
from .index import ProvenanceIndex
from .materialize import ObservationMaterializer


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _object_summary(obj: Mapping[str, Any], entity_uid: str) -> dict[str, Any]:
    points = np.asarray(obj["pcd"].points, dtype=np.float64)
    if len(points):
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
    else:
        minimum = maximum = np.full(3, np.nan)
    bbox = obj["bbox"]
    center = np.asarray(bbox.get_center(), dtype=np.float64)
    extent = np.asarray(bbox.extent, dtype=np.float64)
    members = tuple(sorted(dict.fromkeys(str(item) for item in obj.get("obs_uids", ()))))
    point_digest = hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()
    return {
        "entity_uid": str(entity_uid),
        "member_observation_uids": list(members),
        "num_detections": int(obj.get("num_detections", len(members))),
        "n_points": int(len(points)),
        "bbox_center": center.tolist(),
        "bbox_extent": extent.tolist(),
        "aabb_min": minimum.tolist(),
        "aabb_max": maximum.tolist(),
        "class_name": str(obj.get("class_name", "")),
        "point_digest": point_digest,
    }


def _serialized_object_summary(obj: Mapping[str, Any], entity_uid: str) -> dict[str, Any]:
    points = np.asarray(obj["pcd_np"], dtype=np.float64)
    bbox_points = np.asarray(obj["bbox_np"], dtype=np.float64)
    minimum = points.min(axis=0) if len(points) else np.full(3, np.nan)
    maximum = points.max(axis=0) if len(points) else np.full(3, np.nan)
    bbox_min = bbox_points.min(axis=0) if len(bbox_points) else minimum
    bbox_max = bbox_points.max(axis=0) if len(bbox_points) else maximum
    members = tuple(sorted(dict.fromkeys(str(item) for item in obj.get("obs_uids", ()))))
    return {
        "entity_uid": str(entity_uid),
        "member_observation_uids": list(members),
        "num_detections": int(obj.get("num_detections", len(members))),
        "n_points": int(len(points)),
        "bbox_center": ((bbox_min + bbox_max) / 2.0).tolist(),
        "bbox_extent": (bbox_max - bbox_min).tolist(),
        "aabb_min": minimum.tolist(),
        "aabb_max": maximum.tolist(),
        "class_name": str(obj.get("class_name", "")),
        "point_digest": hashlib.sha256(
            np.ascontiguousarray(points, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def _identity_for_object(
    members: Iterable[str], clean_owner: Mapping[str, str], assigned: set[str]
) -> str:
    votes = Counter(clean_owner[item] for item in members if item in clean_owner)
    if votes:
        identity, _ = votes.most_common(1)[0]
        if identity not in assigned:
            assigned.add(identity)
            return identity
    return stable_entity_uid(members, prefix="replay")


class CounterfactualReplayEngine:
    """Historical replay over exact observation payloads in a shadow state."""

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
        self.clean_membership = canonical_membership(
            {
                str(row["object_uid"]): row.get("member_observation_uids") or ()
                for row in provenance.final_membership
            }
        )
        self.clean_owner = invert_membership(self.clean_membership)
        self._clean_state_cache: dict[str, Any] | None = None

    def _pcd_path(self) -> Path:
        paths = sorted(self.provenance.experiment_root.glob("pcd_*.pkl.gz"))
        if len(paths) != 1:
            raise FileNotFoundError(f"expected one frozen map, found {len(paths)}")
        return paths[0]

    def clean_state(self) -> dict[str, Any]:
        if self._clean_state_cache is not None:
            return copy.deepcopy(self._clean_state_cache)
        with gzip.open(self._pcd_path(), "rb") as handle:
            payload = pickle.load(handle)
        summaries = []
        seen: set[str] = set()
        for obj in payload["objects"]:
            uid = str(obj["id"])
            members = tuple(str(item) for item in obj.get("obs_uids", ()))
            if uid not in self.clean_membership:
                uid = _identity_for_object(members, self.clean_owner, seen)
            seen.add(uid)
            summaries.append(_serialized_object_summary(obj, uid))
        state = {
            "branch": "clean",
            "scope": "frozen_baseline",
            "membership": {key: list(value) for key, value in self.clean_membership.items()},
            "objects": summaries,
            "edges": [],
            "runtime_ms": 0.0,
            "replayed_observations": 0,
            "replayed_events": 0,
            "total_events": len(self.provenance.events),
            "source_hashes": self.provenance.source_hashes(),
        }
        state["state_hash"] = _json_hash(state["membership"])
        self._clean_state_cache = state
        return copy.deepcopy(state)

    def _ordered_observations(self, members: Iterable[str]) -> list[str]:
        return sorted(
            set(str(item) for item in members),
            key=lambda item: (
                self.provenance.sequence(self.provenance.get_association_for_obs(item)),
                item,
            ),
        )

    def _postprocess_denoise(self, objects: Any) -> Any:
        from conceptgraph.slam.utils import denoise_objects

        return denoise_objects(
            downsample_voxel_size=float(self.cfg["downsample_voxel_size"]),
            dbscan_remove_noise=bool(self.cfg["dbscan_remove_noise"]),
            dbscan_eps=float(self.cfg["dbscan_eps"]),
            dbscan_min_points=int(self.cfg["dbscan_min_points"]),
            spatial_sim_type=str(self.cfg["spatial_sim_type"]),
            device=str(self.cfg["device"]),
            objects=objects,
        )

    def _natural_match(self, detection_list: Any, objects: Any) -> list[int | None]:
        if not objects:
            return [None] * len(detection_list)
        from conceptgraph.slam.mapping import (
            aggregate_similarities,
            compute_spatial_similarities,
            compute_visual_similarities,
            match_detections_to_objects,
        )

        spatial = compute_spatial_similarities(
            str(self.cfg["spatial_sim_type"]),
            detection_list,
            objects,
            float(self.cfg["downsample_voxel_size"]),
        )
        visual = compute_visual_similarities(detection_list, objects)
        aggregate = aggregate_similarities(
            str(self.cfg["match_method"]), float(self.cfg["phys_bias"]), spatial, visual
        )
        return match_detections_to_objects(aggregate, float(self.cfg["sim_threshold"]))

    def _merge(self, detection_list: Any, objects: Any, matches: list[int | None]) -> Any:
        from conceptgraph.slam.mapping import merge_obj_matches

        return merge_obj_matches(
            detection_list=detection_list,
            objects=objects,
            match_indices=matches,
            downsample_voxel_size=float(self.cfg["downsample_voxel_size"]),
            dbscan_remove_noise=bool(self.cfg["dbscan_remove_noise"]),
            dbscan_eps=float(self.cfg["dbscan_eps"]),
            dbscan_min_points=int(self.cfg["dbscan_min_points"]),
            spatial_sim_type=str(self.cfg["spatial_sim_type"]),
            device=str(self.cfg["device"]),
        )

    def _overlay_local(
        self,
        replay_objects: Any,
        desired_membership: Mapping[str, Iterable[str]],
        affected_obs: set[str],
        *,
        branch: str,
        runtime_ms: float,
        decision_trace: list[dict[str, Any]],
        replayed_events: int,
    ) -> dict[str, Any]:
        baseline = self.clean_state()
        affected_clean_entities = {
            owner for obs, owner in self.clean_owner.items() if obs in affected_obs
        }
        object_rows = [
            row for row in baseline["objects"] if row["entity_uid"] not in affected_clean_entities
        ]
        desired_owner = invert_membership(desired_membership)
        used: set[str] = set()
        for obj in replay_objects:
            members = [str(item) for item in obj.get("obs_uids", ())]
            votes = Counter(desired_owner[item] for item in members if item in desired_owner)
            entity_uid = votes.most_common(1)[0][0] if votes else stable_entity_uid(members)
            if entity_uid in used:
                entity_uid = stable_entity_uid(members, prefix="replay_fragment")
            used.add(entity_uid)
            object_rows.append(_object_summary(obj, entity_uid))

        membership = {
            row["entity_uid"]: row["member_observation_uids"] for row in object_rows
        }
        state = {
            "branch": branch,
            "scope": "dependency_local",
            "membership": membership,
            "objects": object_rows,
            "edges": [],
            "runtime_ms": runtime_ms,
            "replayed_observations": len(affected_obs),
            "replayed_events": replayed_events,
            "total_events": len(self.provenance.events),
            "forced_decisions": sum(item["forced"] for item in decision_trace),
            "decision_trace": decision_trace,
            "source_hashes": self.provenance.source_hashes(),
            "postprocess_merge_policy": "constraint_partition_preserving",
        }
        state["state_hash"] = _json_hash(state["membership"])
        return state

    def replay_local(self, case: Mapping[str, Any], *, branch: str) -> dict[str, Any]:
        branch = branch.lower()
        if branch not in {"corrupted", "repaired"}:
            raise ValueError("local branch must be corrupted or repaired")
        from conceptgraph.slam.slam_classes import DetectionList, MapObjectList
        from conceptgraph.slam.utils import processing_needed

        desired = (
            apply_controlled_membership_corruption(self.clean_membership, case)
            if branch == "corrupted"
            else self.clean_membership
        )
        affected_obs = {
            str(obs)
            for members in (case.get("affected_clean_groups") or {}).values()
            for obs in members
        }
        desired_local = {
            entity: members
            for entity, members in desired.items()
            if set(members) & affected_obs
        }
        owner = invert_membership(desired_local)
        observations = self._ordered_observations(affected_obs)
        by_frame: dict[int, list[str]] = defaultdict(list)
        for obs_uid in observations:
            row = self.provenance.get_observation(obs_uid)
            frame = int(str(row["frame_uid"]).rsplit("_f", 1)[-1])
            by_frame[frame].append(obs_uid)

        objects = MapObjectList()
        entity_order: list[str] = []
        decisions: list[dict[str, Any]] = []
        started = time.perf_counter()
        if by_frame:
            last_frame = max(by_frame)
            for frame in range(min(by_frame), last_frame + 1):
                for obs_uid in by_frame.get(frame, ()):
                    detection = DetectionList([self.materializer.materialize(obs_uid)])
                    natural = self._natural_match(detection, objects)[0]
                    desired_entity = owner[obs_uid]
                    applied = (
                        entity_order.index(desired_entity)
                        if desired_entity in entity_order
                        else None
                    )
                    objects = self._merge(detection, objects, [applied])
                    if applied is None:
                        entity_order.append(desired_entity)
                    decisions.append(
                        {
                            "obs_uid": obs_uid,
                            "event_uid": self.provenance.get_association_for_obs(obs_uid)["event_uid"],
                            "natural_match": natural,
                            "applied_match": applied,
                            "desired_entity_uid": desired_entity,
                            "forced": natural != applied,
                        }
                    )
                if objects and processing_needed(
                    int(self.cfg.get("denoise_interval", 0)),
                    bool(self.cfg.get("run_denoise_final_frame", False)),
                    frame,
                    frame == last_frame,
                ):
                    objects = self._postprocess_denoise(objects)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        replayed_events = sum(
            2 for obs_uid in affected_obs if obs_uid in self.provenance.association_for_obs
        )
        return self._overlay_local(
            objects,
            desired,
            affected_obs,
            branch=branch,
            runtime_ms=runtime_ms,
            decision_trace=decisions,
            replayed_events=replayed_events,
        )

    def final_member_refusion(self, case: Mapping[str, Any]) -> dict[str, Any]:
        from conceptgraph.slam.slam_classes import MapObjectList

        affected_obs = {
            str(obs)
            for members in (case.get("affected_clean_groups") or {}).values()
            for obs in members
        }
        local_groups = {
            entity: members
            for entity, members in self.clean_membership.items()
            if set(members) & affected_obs
        }
        started = time.perf_counter()
        objects = MapObjectList()
        for entity, members in sorted(local_groups.items()):
            objects.append(
                self.materializer.rebuild_object_from_members(
                    self._ordered_observations(members),
                    preferred_uid=entity,
                    run_final_dbscan=True,
                )
            )
        runtime_ms = (time.perf_counter() - started) * 1000.0
        return self._overlay_local(
            objects,
            self.clean_membership,
            affected_obs,
            branch="final_member_refusion",
            runtime_ms=runtime_ms,
            decision_trace=[],
            replayed_events=0,
        )

    def _best_object_for_identity(self, objects: Any, identity_uid: str) -> int | None:
        expected = set(self.clean_membership[identity_uid])
        best: tuple[int, int] | None = None
        for index, obj in enumerate(objects):
            score = len(expected & set(str(item) for item in obj.get("obs_uids", ())))
            if score and (best is None or score > best[0]):
                best = (score, index)
        return None if best is None else best[1]

    def replay_global(self, case: Mapping[str, Any], *, branch: str = "repaired") -> dict[str, Any]:
        """Full temporal reference using the original mapper's computational path."""
        branch = branch.lower()
        if branch not in {"corrupted", "repaired", "clean"}:
            raise ValueError("global branch must be clean, corrupted or repaired")
        from conceptgraph.slam.slam_classes import DetectionList, MapEdgeMapping, MapObjectList
        from conceptgraph.slam.utils import (
            denoise_objects,
            filter_objects,
            merge_objects,
            processing_needed,
        )

        rows = [
            row
            for row in self.provenance.observation_rows
            if row.get("status") == "kept" and row["obs_uid"] in self.provenance.association_for_obs
        ]
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            frame = int(str(row["frame_uid"]).rsplit("_f", 1)[-1])
            by_frame[frame].append(row)
        for values in by_frame.values():
            values.sort(key=lambda row: (int(row.get("filtered_det_idx", 0)), str(row["obs_uid"])))

        objects = MapObjectList()
        map_edges = MapEdgeMapping(objects)
        decisions: list[dict[str, Any]] = []
        affected_identities = set((case.get("affected_clean_groups") or {}).keys())
        anchor_obs = str(case["obs_uid"])
        target_identity = case.get("target_identity_uid")
        started = time.perf_counter()
        frames = sorted(by_frame)
        for frame in frames:
            frame_rows = by_frame[frame]
            detections = DetectionList(
                [self.materializer.materialize(str(row["obs_uid"])) for row in frame_rows]
            )
            if not objects:
                natural = [None] * len(detections)
                applied = list(natural)
                objects.extend(detections)
                for row in frame_rows:
                    decisions.append(
                        {
                            "obs_uid": row["obs_uid"],
                            "natural_match": None,
                            "applied_match": None,
                            "forced": False,
                        }
                    )
                map_edges.update_objects_list(objects)
                continue

            natural = self._natural_match(detections, objects)
            applied = list(natural)
            for index, row in enumerate(frame_rows):
                obs_uid = str(row["obs_uid"])
                clean_identity = self.clean_owner[obs_uid]
                if branch == "repaired" and clean_identity in affected_identities:
                    applied[index] = self._best_object_for_identity(objects, clean_identity)
                elif branch == "corrupted" and obs_uid == anchor_obs:
                    failure = str(case["failure_type"]).upper()
                    if failure == "FALSE_SPLIT":
                        applied[index] = None
                    elif failure in {"WRONG_MEMBERSHIP", "FALSE_MERGE"}:
                        if not target_identity:
                            raise ValueError("target identity is required for forced association")
                        target_index = self._best_object_for_identity(objects, str(target_identity))
                        if target_index is None:
                            raise RuntimeError("controlled corruption target is not active at anchor")
                        applied[index] = target_index
                decisions.append(
                    {
                        "obs_uid": obs_uid,
                        "natural_match": natural[index],
                        "applied_match": applied[index],
                        "forced": natural[index] != applied[index],
                    }
                )
            objects = self._merge(detections, objects, applied)
            map_edges.update_objects_list(objects)
            is_final = frame == frames[-1]
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

        runtime_ms = (time.perf_counter() - started) * 1000.0
        assigned: set[str] = set()
        summaries = []
        for obj in objects:
            members = [str(item) for item in obj.get("obs_uids", ())]
            uid = _identity_for_object(members, self.clean_owner, assigned)
            summaries.append(_object_summary(obj, uid))
        membership = {
            row["entity_uid"]: row["member_observation_uids"] for row in summaries
        }
        state = {
            "branch": f"global_{branch}",
            "scope": "full_temporal_baseline",
            "membership": membership,
            "objects": summaries,
            "edges": [],
            "runtime_ms": runtime_ms,
            "replayed_observations": len(rows),
            "replayed_events": 2 * len(rows),
            "total_events": len(self.provenance.events),
            "forced_decisions": sum(item["forced"] for item in decisions),
            "decision_trace": decisions,
            "source_hashes": self.provenance.source_hashes(),
            "original_mapper_functions": [
                "compute_spatial_similarities",
                "compute_visual_similarities",
                "aggregate_similarities",
                "match_detections_to_objects",
                "merge_obj_matches",
                "denoise_objects",
                "filter_objects",
                "merge_objects",
            ],
        }
        state["state_hash"] = _json_hash(state["membership"])
        return state
