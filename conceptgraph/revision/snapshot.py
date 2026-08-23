from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .constraints import ReplayMode
from .index import ProvenanceIndex
from .sparse_replay import SparseCounterfactualReplayEngine, SparseReplayError, _object_summary


def _snapshot_validation_pass(
    *,
    requested_count: int,
    resolution: list[Mapping[str, Any]],
    skipped: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> bool:
    """Require every requested seed to resolve before validating unique versions."""

    return bool(
        requested_count > 0
        and len(resolution) == requested_count
        and not skipped
        and rows
        and all(bool(row.get("pass")) for row in rows)
    )


@dataclass
class LocalSnapshot:
    anchor_event_uid: str
    anchor_frame: int
    watermark_event_sequence: int
    objects: Any
    state: dict[str, Any]
    seed_version_uids: tuple[str, ...]
    validation: dict[str, Any]
    config_hash: str
    source_hashes: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "anchor_event_uid": self.anchor_event_uid,
            "anchor_frame": self.anchor_frame,
            "watermark_event_sequence": self.watermark_event_sequence,
            "seed_version_uids": list(self.seed_version_uids),
            "state_hash": self.state["state_hash"],
            "object_count": len(self.state.get("objects") or ()),
            "membership": self.state.get("membership") or {},
            "validation": self.validation,
            "config_hash": self.config_hash,
            "source_hashes": self.source_hashes,
            "snapshot_runtime_ms": float(self.state.get("runtime_ms", 0.0)),
        }


class AnchorStateBuilder:
    """Reconstruct and validate the mapper state immediately before an anchor frame."""

    def __init__(
        self,
        provenance: ProvenanceIndex,
        engine: SparseCounterfactualReplayEngine | None = None,
    ) -> None:
        self.provenance = provenance
        self.engine = engine or SparseCounterfactualReplayEngine(provenance)

    def _find_object(self, objects: Any, version: Mapping[str, Any]) -> Mapping[str, Any] | None:
        object_uid = str(version["object_uid"])
        exact = [obj for obj in objects if str(obj.get("id")) == object_uid]
        if len(exact) == 1:
            return exact[0]
        origin = version.get("origin_observation_uid")
        members = set(str(item) for item in version.get("member_observation_uids") or ())
        candidates = [
            obj
            for obj in objects
            if (origin and str(origin) in set(str(item) for item in obj.get("obs_uids", ())))
            or members == set(str(item) for item in obj.get("obs_uids", ()))
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _validate_version(
        self,
        objects: Any,
        version_uid: str,
        *,
        center_atol: float,
        extent_atol: float,
        point_relative_tolerance: float,
        feature_cosine_tolerance: float,
    ) -> dict[str, Any]:
        version = self.provenance.get_object_version(version_uid)
        obj = self._find_object(objects, version)
        if obj is None:
            return {
                "version_uid": version_uid,
                "pass": False,
                "failure": "active_version_object_not_unique",
            }
        summary = _object_summary(obj, str(version["object_uid"]))
        expected_members = set(str(item) for item in version.get("member_observation_uids") or ())
        actual_members = set(summary["member_observation_uids"])
        center_error = float(
            np.linalg.norm(
                np.asarray(summary["bbox_center"], dtype=float)
                - np.asarray(version.get("bbox_center"), dtype=float)
            )
        )
        extent_error = float(
            np.linalg.norm(
                np.asarray(summary["bbox_extent"], dtype=float)
                - np.asarray(version.get("bbox_extent"), dtype=float)
            )
        )
        expected_points = int(version.get("n_points", 0))
        point_error = abs(int(summary["n_points"]) - expected_points)
        point_relative_error = point_error / max(1, expected_points)
        feature_cosine = None
        feature_pass = True
        feature_ref = version.get("clip_feature_ref")
        if feature_ref and obj.get("clip_ft") is not None:
            expected = np.asarray(self.engine.materializer.load_ref(feature_ref), dtype=float)
            actual_value = obj["clip_ft"]
            if hasattr(actual_value, "detach"):
                actual_value = actual_value.detach().cpu().numpy()
            actual = np.asarray(actual_value, dtype=float)
            denominator = float(np.linalg.norm(expected) * np.linalg.norm(actual))
            feature_cosine = float(np.dot(expected, actual) / denominator) if denominator else 0.0
            feature_pass = feature_cosine >= 1.0 - feature_cosine_tolerance
        expected_hist = {
            str(key): int(value) for key, value in (version.get("class_histogram") or {}).items()
        }
        actual_hist: dict[str, int] = {}
        for value in obj.get("class_id", ()):
            key = str(int(value))
            actual_hist[key] = actual_hist.get(key, 0) + 1
        checks = {
            "member_exact": actual_members == expected_members,
            "bbox_center_within_tolerance": center_error <= center_atol,
            "bbox_extent_within_tolerance": extent_error <= extent_atol,
            "point_count_within_tolerance": point_relative_error <= point_relative_tolerance,
            "feature_cosine_within_tolerance": feature_pass,
            "class_histogram_exact": not expected_hist or actual_hist == expected_hist,
        }
        return {
            "version_uid": version_uid,
            "object_uid": str(version["object_uid"]),
            "pass": all(checks.values()),
            "checks": checks,
            "member_symmetric_difference": sorted(actual_members ^ expected_members),
            "bbox_center_error": center_error,
            "bbox_extent_error": extent_error,
            "point_count_error": point_error,
            "point_count_relative_error": point_relative_error,
            "feature_cosine": feature_cosine,
            "expected_class_histogram": expected_hist,
            "actual_class_histogram": actual_hist,
        }

    def _resolve_pre_anchor_versions(
        self,
        seed_version_uids: Iterable[str],
        *,
        anchor_sequence: int,
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """Resolve candidate-matrix versions to the state active at the anchor.

        Native association computes one candidate matrix at the start of a frame.
        Earlier detections in that same frame can then advance a candidate from, for
        example, ``v000004`` to ``v000006`` before the anchor detection is applied.
        The benchmark case intentionally retains the matrix-time version as evidence,
        while snapshot validation must compare against the latest version from the
        same lineage strictly before the anchor event.
        """

        resolved: list[str] = []
        resolution: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for requested_uid in sorted(set(str(item) for item in seed_version_uids if item)):
            if requested_uid not in self.provenance.object_versions:
                skipped.append(
                    {"version_uid": requested_uid, "reason": "unknown_version"}
                )
                continue
            requested = self.provenance.get_object_version(requested_uid)
            requested_trigger = requested.get("trigger_event_uid")
            if requested_trigger not in self.provenance.events:
                skipped.append(
                    {"version_uid": requested_uid, "reason": "missing_trigger_event"}
                )
                continue
            requested_sequence = self.provenance.sequence(
                self.provenance.get_event(str(requested_trigger))
            )
            if requested_sequence >= anchor_sequence:
                skipped.append(
                    {"version_uid": requested_uid, "reason": "not_pre_anchor"}
                )
                continue

            lineage_uid = requested.get("lineage_uid")
            object_uid = requested.get("object_uid")
            candidates: list[tuple[int, str]] = []
            for candidate_uid, candidate in self.provenance.object_versions.items():
                same_identity = (
                    bool(lineage_uid)
                    and candidate.get("lineage_uid") == lineage_uid
                ) or (
                    not lineage_uid
                    and bool(object_uid)
                    and candidate.get("object_uid") == object_uid
                )
                if not same_identity:
                    continue
                trigger_uid = candidate.get("trigger_event_uid")
                if trigger_uid not in self.provenance.events:
                    continue
                sequence = self.provenance.sequence(
                    self.provenance.get_event(str(trigger_uid))
                )
                if sequence < anchor_sequence:
                    candidates.append((sequence, str(candidate_uid)))
            if not candidates:
                skipped.append(
                    {"version_uid": requested_uid, "reason": "no_pre_anchor_lineage_version"}
                )
                continue
            sequence, active_uid = max(candidates, key=lambda item: (item[0], item[1]))
            resolution.append(
                {
                    "requested_version_uid": requested_uid,
                    "resolved_version_uid": active_uid,
                    "requested_event_sequence": requested_sequence,
                    "resolved_event_sequence": sequence,
                    "advanced_within_prefix": active_uid != requested_uid,
                    "basis": "latest_same_lineage_strictly_before_anchor",
                }
            )
            if active_uid not in seen:
                seen.add(active_uid)
                resolved.append(active_uid)
        return resolved, resolution, skipped

    def build_pre_anchor_state(
        self,
        anchor_event_uid: str,
        dependency_seed: Iterable[str],
        *,
        center_atol: float = 2e-3,
        extent_atol: float = 2e-3,
        point_relative_tolerance: float = 0.02,
        feature_cosine_tolerance: float = 1e-5,
        strict: bool = True,
        prefix_state: Mapping[str, Any] | None = None,
        prefix_objects: Any | None = None,
    ) -> LocalSnapshot:
        anchor = self.provenance.get_event(str(anchor_event_uid))
        frame_uid = str(anchor["frame_uid"])
        anchor_frame = int(frame_uid.rsplit("_f", 1)[-1])
        anchor_sequence = self.provenance.sequence(anchor)
        if (prefix_state is None) != (prefix_objects is None):
            raise ValueError("prefix_state and prefix_objects must be supplied together")
        if prefix_state is None:
            state, objects = self.engine.replay_prefix(anchor_frame=anchor_frame)
        else:
            state, objects = dict(prefix_state), prefix_objects
        state, objects = self.engine.advance_recorded_frame_prefix(
            objects=objects,
            prefix_state=state,
            anchor_obs_uid=str(anchor["obs_uid"]),
        )
        seeds = tuple(sorted(set(str(item) for item in dependency_seed if item)))
        eligible, resolution, skipped = self._resolve_pre_anchor_versions(
            seeds,
            anchor_sequence=anchor_sequence,
        )
        rows = [
            self._validate_version(
                objects,
                version_uid,
                center_atol=center_atol,
                extent_atol=extent_atol,
                point_relative_tolerance=point_relative_tolerance,
                feature_cosine_tolerance=feature_cosine_tolerance,
            )
            for version_uid in eligible
        ]
        validation = {
            "pass": _snapshot_validation_pass(
                requested_count=len(seeds),
                resolution=resolution,
                skipped=skipped,
                rows=rows,
            ),
            "validated_version_count": len(rows),
            "requested_version_count": len(seeds),
            "resolved_request_count": len(resolution),
            "all_requested_versions_resolved": len(resolution) == len(seeds)
            and not skipped,
            "version_resolution": resolution,
            "skipped": skipped,
            "versions": rows,
            "tolerances": {
                "bbox_center_atol": center_atol,
                "bbox_extent_atol": extent_atol,
                "point_relative_tolerance": point_relative_tolerance,
                "feature_cosine_tolerance": feature_cosine_tolerance,
            },
        }
        if strict and not validation["pass"]:
            failures = [row["version_uid"] for row in rows if not row["pass"]]
            failures.extend(
                f"{row['version_uid']}:{row['reason']}" for row in skipped
            )
            raise SparseReplayError(
                "pre-anchor reconstruction mismatch for " + ", ".join(failures or ["no seed"])
            )
        config_hash = hashlib.sha256(
            json.dumps(self.engine.cfg, sort_keys=True, default=str).encode()
        ).hexdigest()
        watermark = max(
            (
                self.provenance.sequence(event)
                for event in self.provenance.events.values()
                if self.provenance.sequence(event) < anchor_sequence
            ),
            default=-1,
        )
        return LocalSnapshot(
            anchor_event_uid=str(anchor_event_uid),
            anchor_frame=anchor_frame,
            watermark_event_sequence=watermark,
            objects=objects,
            state=state,
            seed_version_uids=seeds,
            validation=validation,
            config_hash=config_hash,
            source_hashes=self.provenance.source_hashes(),
        )


class IncrementalPrefixCache:
    """Advance one scene mapper once and fork pre-anchor prefixes in time order."""

    def __init__(self, engine: SparseCounterfactualReplayEngine) -> None:
        from conceptgraph.slam.slam_classes import MapObjectList

        self.engine = engine
        self.objects = MapObjectList()
        self.completed_frame = -1
        self.runtime_ms = 0.0
        self.replayed_observations = 0
        self.decisions: list[dict[str, Any]] = []
        self.postprocess_counts = {"denoise": 0, "filter": 0, "merge": 0}
        self.state = self.engine._state(
            objects=self.objects,
            mode=ReplayMode.NATURAL_REPLAY,
            scope="incremental_pre_anchor_prefix",
            runtime_ms=0.0,
            replayed_observations=0,
            decisions=[],
            postprocess_counts=self.postprocess_counts,
            constraint_count=0,
            intervention_count=0,
        )

    def prefix_before(self, anchor_frame: int) -> tuple[dict[str, Any], Any]:
        target_frame = int(anchor_frame) - 1
        if target_frame < self.completed_frame:
            raise ValueError(
                f"incremental prefix requested out of order: {target_frame} < "
                f"{self.completed_frame}"
            )
        if target_frame > self.completed_frame:
            start = self.completed_frame + 1
            rows = [
                row
                for row in self.engine._all_rows
                if start <= int(str(row["frame_uid"]).rsplit("_f", 1)[-1]) <= target_frame
            ]
            segment, objects = self.engine._execute(
                mode=ReplayMode.NATURAL_REPLAY,
                rows=rows,
                frame_start=start,
                frame_end=target_frame,
                initial_objects=self.objects,
                final_scene_frame=self.engine.final_frame,
                scope="incremental_pre_anchor_segment",
            )
            self.objects = objects
            self.completed_frame = target_frame
            self.runtime_ms += float(segment["runtime_ms"])
            self.replayed_observations += len(rows)
            self.decisions.extend(segment.get("decision_trace") or ())
            for key in self.postprocess_counts:
                self.postprocess_counts[key] += int(
                    (segment.get("postprocess_counts") or {}).get(key, 0)
                )
            self.state = self.engine._state(
                objects=self.objects,
                mode=ReplayMode.NATURAL_REPLAY,
                scope="incremental_pre_anchor_prefix",
                runtime_ms=self.runtime_ms,
                replayed_observations=self.replayed_observations,
                decisions=self.decisions,
                postprocess_counts=self.postprocess_counts,
                constraint_count=0,
                intervention_count=0,
            )
        return self.state, self.objects
