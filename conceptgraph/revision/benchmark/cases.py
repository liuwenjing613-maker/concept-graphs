from __future__ import annotations

import hashlib
import inspect
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping

from ..cases import ControlledCaseBuilder
from ..constraints import SparseRepairConstraint
from ..index import ProvenanceIndex


def _version_target(provenance: ProvenanceIndex, version_uid: str | None) -> dict[str, Any]:
    if not version_uid or version_uid not in provenance.object_versions:
        raise ValueError(f"constraint target version is not resolvable: {version_uid}")
    row = provenance.get_object_version(str(version_uid))
    members = tuple(str(item) for item in row.get("member_observation_uids") or ())
    origin = row.get("origin_observation_uid") or (members[0] if members else None)
    if not origin:
        raise ValueError(f"constraint target version has no immutable origin: {version_uid}")
    return {
        "version_uid": str(version_uid),
        "lineage_uid": str(row["lineage_uid"]),
        "entity_uid": str(row["object_uid"]),
        "origin_obs_uid": str(origin),
    }


def compile_sparse_constraints(
    case: Mapping[str, Any], provenance: ProvenanceIndex
) -> list[SparseRepairConstraint]:
    """Compile the known injected action into sparse, executable primitives.

    The compiler uses the original event at the injection boundary. It never embeds
    the final member partition or any later expected owner in the constraint.
    """

    anchor_uid = str(case["anchor_association_event_uid"])
    anchor = provenance.get_event(anchor_uid)
    if str(anchor.get("obs_uid")) != str(case["obs_uid"]):
        raise ValueError("case observation does not match its association anchor")
    obs_uid = str(anchor["obs_uid"])
    sequence = provenance.sequence(anchor)
    evidence_refs = (anchor_uid, str(anchor["mapping_event_uid"]))
    decision = str(anchor.get("decision", "")).upper()
    constraints: list[SparseRepairConstraint] = []

    if decision == "MERGE_TO_OBJECT":
        target = _version_target(
            provenance, str(anchor.get("target_object_version_before") or "")
        )
        constraints.append(
            SparseRepairConstraint(
                constraint_type="ASSIGN_OBSERVATION",
                obs_uid=obs_uid,
                target_lineage_uid=target["lineage_uid"],
                target_origin_obs_uid=target["origin_obs_uid"],
                target_entity_uid=target["entity_uid"],
                applies_at_event_uid=anchor_uid,
                active_from_sequence=sequence,
                source="controlled_original_action",
                evidence_refs=evidence_refs,
            )
        )
    elif decision == "CREATE_OBJECT":
        created = _version_target(
            provenance, str(anchor.get("target_object_version_after") or "")
        )
        wrong_version_uid = case.get("target_object_version_uid")
        wrong = _version_target(provenance, str(wrong_version_uid or ""))
        constraints.append(
            SparseRepairConstraint(
                constraint_type="CANNOT_LINK",
                obs_uid=obs_uid,
                target_lineage_uid=wrong["lineage_uid"],
                target_origin_obs_uid=wrong["origin_obs_uid"],
                target_entity_uid=wrong["entity_uid"],
                created_lineage_uid=created["lineage_uid"],
                created_entity_uid=created["entity_uid"],
                applies_at_event_uid=anchor_uid,
                active_from_sequence=sequence,
                source="controlled_original_action",
                evidence_refs=evidence_refs,
            )
        )
    else:
        raise ValueError(f"unsupported original association decision: {decision}")

    if str(case["failure_type"]).upper() == "WRONG_MEMBERSHIP":
        wrong_version_uid = case.get("target_object_version_uid")
        if wrong_version_uid:
            wrong = _version_target(provenance, str(wrong_version_uid))
            constraints.append(
                SparseRepairConstraint(
                    constraint_type="CANNOT_LINK",
                    obs_uid=obs_uid,
                    target_lineage_uid=wrong["lineage_uid"],
                    target_origin_obs_uid=wrong["origin_obs_uid"],
                    target_entity_uid=wrong["entity_uid"],
                    applies_at_event_uid=anchor_uid,
                    active_from_sequence=sequence,
                    source="controlled_original_action",
                    evidence_refs=evidence_refs,
                )
            )
    return constraints


def _count_bin(value: int) -> str:
    if value < 10:
        return "small"
    if value < 30:
        return "medium"
    return "large"


def _margin_bin(value: Any) -> str:
    if value is None:
        return "missing"
    margin = float(value)
    if margin < 0.05:
        return "low"
    if margin < 0.25:
        return "medium"
    return "high"


def _score_bin(value: Any) -> str:
    if value is None:
        return "missing"
    score = float(value)
    if score < 1.2:
        return "below_threshold"
    if score < 1.6:
        return "near_threshold"
    return "high"


def _quartile(frame_idx: int, frame_count: int) -> str:
    return f"Q{min(4, 1 + (4 * max(0, frame_idx)) // max(1, frame_count))}"


class BatchCaseSampler:
    """Outcome-blind, deterministic stratified sampler for controlled incidents."""

    def __init__(
        self,
        provenance: ProvenanceIndex,
        *,
        scene: str,
        seed: int = 20260823,
        frame_count: int = 200,
    ) -> None:
        self.provenance = provenance
        self.scene = str(scene)
        self.seed = int(seed)
        self.frame_count = int(frame_count)
        self.builder = ControlledCaseBuilder(provenance)

    def _strata(self, case: Mapping[str, Any]) -> dict[str, Any]:
        groups = case.get("affected_clean_groups") or {}
        source = str(case["source_identity_uid"])
        target = case.get("target_identity_uid")
        source_count = len(groups.get(source, ()))
        target_count = len(groups.get(str(target), ())) if target else 0
        frame_idx = int(case["frame_idx"])
        anchor_sequence = int(case["anchor_event_sequence"])
        descendants = 0
        version_uid = case.get("clean_target_object_version_uid")
        if version_uid in self.provenance.object_versions:
            descendants = sum(
                self.provenance.sequence(event) > anchor_sequence
                for event in self.provenance.events.values()
                if version_uid in (event.get("input_object_version_uids") or ())
            )
        return {
            "scene": self.scene,
            "failure_type": str(case["failure_type"]),
            "anchor_time_quartile": _quartile(frame_idx, self.frame_count),
            "source_member_count_bin": _count_bin(source_count),
            "target_member_count_bin": _count_bin(target_count) if target else "none",
            "association_margin_bin": _margin_bin(case.get("clean_margin")),
            "target_candidate_score_bin": _score_bin(
                (case.get("selection_metadata") or {}).get("target_score")
                or case.get("clean_top1_score")
            ),
            "descendant_event_count_bin": _count_bin(descendants),
            "crosses_denoise": frame_idx < self.frame_count - 1,
            "crosses_filter": frame_idx < self.frame_count - 1,
            "crosses_merge": frame_idx < self.frame_count - 1,
            "relation_impact": "UNSCREENED_PRIMARY",
            "maturity_bin": "eligible_by_builder",
        }
    def sample(self, failure_type: str, *, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if count < 1:
            raise ValueError("sample count must be positive")
        pool = self.builder.ranked_candidates(str(failure_type).upper())
        decorated = []
        for case in pool:
            row = dict(case)
            row["strata"] = self._strata(row)
            row["evaluation_role"] = "PRIMARY_STRATIFIED"
            decorated.append(row)

        strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        keys = (
            "anchor_time_quartile",
            "source_member_count_bin",
            "association_margin_bin",
        )
        for row in decorated:
            strata[tuple(str(row["strata"][key]) for key in keys)].append(row)
        rng = random.Random(self.seed + sum(ord(ch) for ch in str(failure_type)))
        for values in strata.values():
            rng.shuffle(values)
        stratum_order = sorted(strata)
        rng.shuffle(stratum_order)
        selected: list[dict[str, Any]] = []
        cursor = 0
        while len(selected) < min(count, len(decorated)) and stratum_order:
            key = stratum_order[cursor % len(stratum_order)]
            values = strata[key]
            if values:
                selected.append(values.pop())
            if not values:
                stratum_order.remove(key)
                cursor = 0
            else:
                cursor += 1

        selector_source = inspect.getsource(type(self))
        manifest = {
            "schema_version": "1.0.0",
            "scene": self.scene,
            "failure_type": str(failure_type).upper(),
            "evaluation_role": "PRIMARY_STRATIFIED",
            "outcome_screened": False,
            "pool_size": len(pool),
            "requested_count": count,
            "selected_count": len(selected),
            "seed": self.seed,
            "stratification_fields": list(keys),
            "selector_sha256": hashlib.sha256(selector_source.encode()).hexdigest(),
            "selected_case_uids": [str(row["case_uid"]) for row in selected],
            "selection_reason": "seeded round-robin over pre-outcome strata",
        }
        return selected, manifest

    def sample_matrix(
        self,
        *,
        failure_types: Iterable[str] = (
            "FALSE_SPLIT",
            "WRONG_MEMBERSHIP",
            "FALSE_MERGE",
        ),
        count_per_type: int = 10,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        manifests = []
        for failure_type in failure_types:
            selected, manifest = self.sample(failure_type, count=count_per_type)
            cases.extend(selected)
            manifests.append(manifest)
        return cases, {
            "schema_version": "1.0.0",
            "scene": self.scene,
            "seed": self.seed,
            "evaluation_role": "PRIMARY_STRATIFIED",
            "outcome_screened": False,
            "case_count": len(cases),
            "subsets": manifests,
        }
