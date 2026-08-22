from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np

from .index import ProvenanceIndex


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


class StructuralVerifier:
    def __init__(self, provenance: ProvenanceIndex) -> None:
        self.provenance = provenance

    @staticmethod
    def _check(check_id: str, passed: bool, **details: Any) -> dict[str, Any]:
        return {"id": check_id, "pass": bool(passed), "details": details}

    def verify(
        self,
        *,
        baseline_state: Mapping[str, Any],
        derived_state: Mapping[str, Any],
        closure: Mapping[str, Any],
        expected_source_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        membership = derived_state.get("membership") or {}
        owners: dict[str, list[str]] = {}
        for entity, members in membership.items():
            for obs_uid in members:
                owners.setdefault(str(obs_uid), []).append(str(entity))
        duplicates = {obs: values for obs, values in owners.items() if len(values) != 1}
        checks.append(self._check("V1_MEMBERSHIP_OWNERSHIP_VALID", not duplicates, duplicates=duplicates))

        unresolved = sorted(obs for obs in owners if obs not in self.provenance.observations)
        checks.append(self._check("V2_ALL_OBS_UIDS_RESOLVABLE", not unresolved, unresolved=unresolved))

        invalid_pcd = []
        invalid_bbox = []
        invalid_versions = []
        for obj in derived_state.get("objects") or ():
            entity = str(obj.get("entity_uid", ""))
            n_points = int(obj.get("n_points", 0))
            center = np.asarray(obj.get("bbox_center", ()), dtype=float)
            extent = np.asarray(obj.get("bbox_extent", ()), dtype=float)
            bounds = np.asarray(
                [obj.get("aabb_min", ()), obj.get("aabb_max", ())], dtype=float
            )
            if n_points <= 0 or bounds.shape != (2, 3) or not np.isfinite(bounds).all():
                invalid_pcd.append(entity)
            if (
                center.shape != (3,)
                or extent.shape != (3,)
                or not np.isfinite(center).all()
                or not np.isfinite(extent).all()
                or not np.all(extent > 0)
            ):
                invalid_bbox.append(entity)
            if entity in self.provenance.final_by_object:
                if self.provenance.get_current_version(entity) is None:
                    invalid_versions.append(entity)
            else:
                source_entities = {
                    self._clean_owner(obs)
                    for obs in obj.get("member_observation_uids") or ()
                    if obs in self.provenance.observations
                }
                if not source_entities or any(
                    source is None or self.provenance.get_current_version(source) is None
                    for source in source_entities
                ):
                    invalid_versions.append(entity)
        checks.append(self._check("V3_ACTIVE_PCD_FINITE_NONEMPTY", not invalid_pcd, entities=invalid_pcd))
        checks.append(self._check("V4_BBOX_FINITE_NONDEGENERATE", not invalid_bbox, entities=invalid_bbox))
        checks.append(self._check("V5_CURRENT_VERSION_REFERENCES_VALID", not invalid_versions, entities=invalid_versions))

        active = set(str(item) for item in membership)
        dangling = []
        self_loops = []
        for edge in derived_state.get("edges") or ():
            source = str(edge.get("source_entity_uid", ""))
            target = str(edge.get("target_entity_uid", ""))
            if source not in active or target not in active:
                dangling.append(edge)
            if source == target:
                self_loops.append(edge)
        checks.append(self._check("V6_EDGE_ENDPOINTS_ACTIVE", not dangling, dangling=dangling))
        checks.append(self._check("V7_NO_UNEXPECTED_SELF_LOOP", not self_loops, self_loops=self_loops))

        closure_obs = set(str(item) for item in closure.get("obs_uids") or ())
        baseline_rows = {
            str(row["entity_uid"]): row for row in baseline_state.get("objects") or ()
        }
        derived_rows = {
            str(row["entity_uid"]): row for row in derived_state.get("objects") or ()
        }
        outside = {
            entity
            for entity, row in baseline_rows.items()
            if not (set(row.get("member_observation_uids") or ()) & closure_obs)
        }
        outside_changes = []
        for entity in sorted(outside):
            if entity not in derived_rows or _hash(baseline_rows[entity]) != _hash(derived_rows[entity]):
                outside_changes.append(entity)
        checks.append(
            self._check(
                "V8_OUTSIDE_CLOSURE_UNCHANGED",
                not outside_changes,
                outside_entity_count=len(outside),
                changed_entities=outside_changes,
            )
        )

        actual_hashes = self.provenance.source_hashes()
        source_changes = {
            name: {"expected": digest, "actual": actual_hashes.get(name)}
            for name, digest in expected_source_hashes.items()
            if actual_hashes.get(name) != digest
        }
        checks.append(
            self._check(
                "V9_SOURCE_BASELINE_ARTIFACTS_UNCHANGED",
                not source_changes,
                changed_files=source_changes,
            )
        )
        failed = [item["id"] for item in checks if not item["pass"]]
        return {
            "pass": not failed,
            "checks": checks,
            "hard_invariant_failures": failed,
            "outside_closure_changed_entities": outside_changes,
            "source_hashes_after": actual_hashes,
        }

    def _clean_owner(self, obs_uid: str) -> str | None:
        for entity, row in self.provenance.final_by_object.items():
            if obs_uid in (row.get("member_observation_uids") or ()):
                return entity
        return None
