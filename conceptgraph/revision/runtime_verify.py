from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constraints import SparseRepairConstraint


class InvariantVerifier:
    """Production commit checks over the proposed state and immutable sources only."""

    def verify(
        self,
        *,
        state: Mapping[str, Any],
        constraints: Sequence[SparseRepairConstraint] = (),
        source_hashes_before: Mapping[str, str] | None = None,
        source_hashes_after: Mapping[str, str] | None = None,
        known_observation_uids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        membership = state.get("membership") or {}
        ownership = Counter(
            str(obs_uid)
            for members in membership.values()
            for obs_uid in (members or ())
        )
        duplicates = sorted(obs for obs, count in ownership.items() if count > 1)
        if duplicates:
            failures.append({"invariant": "R1", "duplicate_observations": duplicates})

        object_membership: dict[str, list[str]] = {}
        duplicate_entities = []
        for row in state.get("objects") or ():
            uid = str(row.get("entity_uid"))
            if uid in object_membership:
                duplicate_entities.append(uid)
            object_membership[uid] = sorted(
                str(item) for item in row.get("member_observation_uids") or ()
            )
        declared_membership = {
            str(uid): sorted(str(item) for item in members or ())
            for uid, members in membership.items()
        }
        known = (
            set(str(item) for item in known_observation_uids)
            if known_observation_uids is not None
            else None
        )
        unknown_observations = (
            sorted(set(ownership) - known) if known is not None else []
        )
        empty_observation_uids = sorted(uid for uid in ownership if not uid)
        if (
            duplicate_entities
            or object_membership != declared_membership
            or unknown_observations
            or empty_observation_uids
        ):
            failures.append(
                {
                    "invariant": "R2",
                    "duplicate_entity_uids": sorted(set(duplicate_entities)),
                    "object_membership_matches_state": object_membership
                    == declared_membership,
                    "unknown_observation_uids": unknown_observations,
                    "empty_observation_uids": empty_observation_uids,
                }
            )

        active = set(str(item) for item in membership)
        for row in state.get("objects") or ():
            uid = str(row.get("entity_uid"))
            points = int(row.get("n_points", 0))
            center = np.asarray(row.get("bbox_center"), dtype=float)
            extent = np.asarray(row.get("bbox_extent"), dtype=float)
            if points <= 0:
                failures.append({"invariant": "R3", "entity_uid": uid})
            if (
                center.shape != (3,)
                or extent.shape != (3,)
                or not np.isfinite(center).all()
                or not np.isfinite(extent).all()
                or np.any(extent <= 0)
            ):
                failures.append({"invariant": "R4", "entity_uid": uid})

        if constraints:
            expected = {item.constraint_uid for item in constraints}
            seen = {
                uid
                for decision in state.get("decision_trace") or ()
                for uid in (decision.get("constraint") or {}).get("constraint_uids", ())
            }
            missing = sorted(expected - seen)
            deferred = [
                decision.get("obs_uid")
                for decision in state.get("decision_trace") or ()
                if (decision.get("constraint") or {}).get("action") == "DEFER"
            ]
            semantic_errors = []
            constraint_by_uid = {item.constraint_uid: item for item in constraints}
            for decision in state.get("decision_trace") or ():
                detail = decision.get("constraint") or {}
                action = str(detail.get("action") or "")
                applied = decision.get("applied_match")
                target = detail.get("target_index")
                forbidden = set(int(item) for item in detail.get("forbidden_indices") or ())
                decision_uids = [
                    str(item) for item in detail.get("constraint_uids") or ()
                ]
                if applied is not None and int(applied) in forbidden:
                    semantic_errors.append(
                        {
                            "obs_uid": decision.get("obs_uid"),
                            "reason": "applied_forbidden_target",
                        }
                    )
                if action == "FORCE_TARGET" and applied != target:
                    semantic_errors.append(
                        {
                            "obs_uid": decision.get("obs_uid"),
                            "reason": "forced_target_not_applied",
                        }
                    )
                if action == "FORCE_CREATE" and applied is not None:
                    semantic_errors.append(
                        {
                            "obs_uid": decision.get("obs_uid"),
                            "reason": "forced_create_not_applied",
                        }
                    )
                for uid in decision_uids:
                    primitive = constraint_by_uid.get(uid)
                    if primitive is None:
                        semantic_errors.append(
                            {
                                "obs_uid": decision.get("obs_uid"),
                                "reason": "unknown_constraint_uid",
                                "constraint_uid": uid,
                            }
                        )
                        continue
                    if primitive.constraint_type.value in {
                        "MUST_LINK",
                        "ASSIGN_OBSERVATION",
                    } and action != "FORCE_TARGET":
                        semantic_errors.append(
                            {
                                "obs_uid": decision.get("obs_uid"),
                                "reason": "positive_constraint_not_forced",
                                "constraint_uid": uid,
                            }
                        )
                    if (
                        primitive.constraint_type.value == "CREATE_INSTANCE"
                        and action != "FORCE_CREATE"
                    ):
                        semantic_errors.append(
                            {
                                "obs_uid": decision.get("obs_uid"),
                                "reason": "create_instance_not_forced",
                                "constraint_uid": uid,
                            }
                        )
                    if primitive.constraint_type.value == "CREATE_INSTANCE":
                        owners = [
                            row
                            for row in state.get("objects") or ()
                            if primitive.obs_uid
                            in (row.get("member_observation_uids") or ())
                        ]
                        expected_lineage = (
                            primitive.created_lineage_uid
                            or "revision-lineage:" + str(primitive.obs_uid)
                        )
                        if len(owners) != 1 or expected_lineage not in set(
                            owners[0].get("revision_lineage_uids") or ()
                            if owners
                            else ()
                        ):
                            semantic_errors.append(
                                {
                                    "obs_uid": decision.get("obs_uid"),
                                    "reason": "created_instance_lineage_not_preserved",
                                    "constraint_uid": uid,
                                    "expected_lineage_uid": expected_lineage,
                                }
                            )
                    if primitive.constraint_type.value in {
                        "PARTITION_ENTITY",
                        "RELABEL",
                    }:
                        semantic_errors.append(
                            {
                                "obs_uid": decision.get("obs_uid"),
                                "reason": "primitive_not_executable_at_association_boundary",
                                "constraint_uid": uid,
                            }
                        )
            if missing or deferred or semantic_errors:
                failures.append(
                    {
                        "invariant": "R6",
                        "unseen_constraint_uids": missing,
                        "deferred_observations": deferred,
                        "semantic_errors": semantic_errors,
                    }
                )

        dangling = []
        self_loops = []
        for edge in state.get("edges") or ():
            source = str(edge.get("source_entity_uid"))
            target = str(edge.get("target_entity_uid"))
            if source not in active or target not in active:
                dangling.append((source, target))
            if source == target:
                self_loops.append((source, str(edge.get("relation")), target))
        if dangling:
            failures.append({"invariant": "R7", "dangling_edges": dangling})
        if self_loops:
            failures.append({"invariant": "R8", "self_loops": self_loops})

        overlay = state.get("overlay_diagnostics") or {}
        if overlay and not overlay.get("overlay_pass", False):
            failures.append(
                {
                    "invariant": "R11",
                    "partial_outside_overlap_entities": overlay.get(
                        "partial_outside_overlap_entities", []
                    ),
                }
            )
        if source_hashes_before is not None and source_hashes_after is not None:
            changed = sorted(
                key
                for key in set(source_hashes_before) | set(source_hashes_after)
                if source_hashes_before.get(key) != source_hashes_after.get(key)
            )
            if changed:
                failures.append({"invariant": "R10", "changed_sources": changed})

        return {
            "schema_version": "1.0.0",
            "pass": not failures,
            "hard_invariant_failures": failures,
            "checks": {
                "R1_unique_owner": not duplicates,
                "R2_evidence_and_object_refs": not any(
                    item["invariant"] == "R2" for item in failures
                ),
                "R3_R4_finite_geometry": not any(
                    item["invariant"] in {"R3", "R4"} for item in failures
                ),
                "R6_constraint_satisfied": not any(
                    item["invariant"] == "R6" for item in failures
                ),
                "R7_R8_edges_structural": not dangling and not self_loops,
                "R10_sources_immutable": not any(
                    item["invariant"] == "R10" for item in failures
                ),
                "R11_scope_overlay": not any(
                    item["invariant"] == "R11" for item in failures
                ),
            },
        }
