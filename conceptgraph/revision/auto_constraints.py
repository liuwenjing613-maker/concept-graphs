"""Fail-closed automatic constraint generation and promotion gates.

The generator is deliberately split into four auditable layers:

1. propose from inference-only evidence;
2. normalize and bind aliases to immutable provenance/effective identity;
3. validate the exact bound constraint with counterfactual shadow evidence;
4. expose a constraint for commit only when every mandatory gate passes.

Human labels and final ownership are evaluation inputs, never proposal inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .constraints import SparseRepairConstraint


class GeneratorStage(str, Enum):
    PROPOSED = "PROPOSED"
    NORMALIZED = "NORMALIZED"
    BOUND_PENDING_SHADOW = "BOUND_PENDING_SHADOW"
    COMMIT_ELIGIBLE = "COMMIT_ELIGIBLE"
    DEFERRED = "DEFERRED"


class AutomaticAction(str, Enum):
    SAME_INSTANCE = "SAME_INSTANCE"
    SEPARATE_MEMBER_GROUPS = "SEPARATE_MEMBER_GROUPS"
    MOVE_OBSERVATION = "MOVE_OBSERVATION"
    RELABEL = "RELABEL"
    RESTORE_OBSERVATION_GEOMETRY = "RESTORE_OBSERVATION_GEOMETRY"
    PARTITION_OBSERVATION = "PARTITION_OBSERVATION"
    DEFER = "DEFER"


_FORBIDDEN_INFERENCE_KEY_FRAGMENTS = (
    "human_label",
    "human_note",
    "posthoc_gold",
    "expected_action",
    "expected_constraint",
    "expected_membership",
    "desired_owner",
    "final_owner",
    "final_membership",
    "ground_truth",
)

_REQUIRED_SHADOW_GATES = (
    "endpoint_improved",
    "collateral_safe",
    "invariants_pass",
    "source_immutable",
    "no_op_controls_pass",
    "legal_merge_control_pass",
    "component_mechanism_supported",
    "local_global_parity_pass",
    "evaluation_independent_of_generator",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _uid(prefix: str, value: Any) -> str:
    return (
        prefix + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]
    )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, Sequence):
        raise ValueError("expected a string sequence")
    return tuple(
        dict.fromkeys(str(item).strip() for item in value if str(item).strip())
    )


def forbidden_inference_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    """Return oracle-like fields recursively present in an inference payload."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            lowered = key_text.lower()
            if any(
                fragment in lowered for fragment in _FORBIDDEN_INFERENCE_KEY_FRAGMENTS
            ):
                found.append(path)
            found.extend(forbidden_inference_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            found.extend(forbidden_inference_paths(child, path))
    return tuple(sorted(set(found)))


def _canonical_groups(value: Any) -> tuple[tuple[str, ...], ...]:
    raw_groups: list[Any]
    if isinstance(value, Mapping):
        raw_groups = list(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_groups = list(value)
    else:
        raw_groups = []
    groups = []
    for raw in raw_groups:
        members = tuple(sorted(set(_string_tuple(raw))))
        if members:
            groups.append(members)
    return tuple(sorted(set(groups)))


def canonicalize_vote(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one blind proposal and compute its structural vote signature."""

    forbidden = forbidden_inference_paths(proposal)
    if forbidden:
        raise ValueError("oracle-like inference fields: " + ", ".join(forbidden))
    action = AutomaticAction(_required_text(proposal.get("action"), "action").upper())
    confidence = float(proposal.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    evidence_ids = tuple(sorted(set(_string_tuple(proposal.get("evidence_image_ids")))))
    canonical: dict[str, Any] = {
        "action": action.value,
        "confidence": confidence,
        "evidence_image_ids": list(evidence_ids),
    }
    signature_payload: dict[str, Any] = {"action": action.value}

    if action == AutomaticAction.SAME_INSTANCE:
        entities = tuple(sorted(set(_string_tuple(proposal.get("entities")))))
        canonical["entities"] = list(entities)
        signature_payload["entities"] = list(entities)
    elif action == AutomaticAction.SEPARATE_MEMBER_GROUPS:
        groups = _canonical_groups(proposal.get("groups"))
        if not groups:
            entities = tuple(sorted(set(_string_tuple(proposal.get("entities")))))
            groups = tuple((entity,) for entity in entities)
        canonical["groups"] = [list(group) for group in groups]
        signature_payload["groups"] = canonical["groups"]
    elif action == AutomaticAction.MOVE_OBSERVATION:
        for name in ("obs_key", "from_alias", "to_alias"):
            canonical[name] = _required_text(proposal.get(name), name)
            signature_payload[name] = canonical[name]
    elif action == AutomaticAction.RELABEL:
        aliases = _string_tuple(
            proposal.get("entity_alias") or proposal.get("entities")
        )
        canonical["entity_alias"] = aliases[0] if len(aliases) == 1 else ""
        canonical["label"] = _required_text(proposal.get("label"), "label").lower()
        signature_payload.update(
            {
                "entity_alias": canonical["entity_alias"],
                "label": canonical["label"],
            }
        )
    elif action in {
        AutomaticAction.RESTORE_OBSERVATION_GEOMETRY,
        AutomaticAction.PARTITION_OBSERVATION,
    }:
        canonical["obs_key"] = _required_text(proposal.get("obs_key"), "obs_key")
        signature_payload["obs_key"] = canonical["obs_key"]
        if action == AutomaticAction.PARTITION_OBSERVATION:
            contract = proposal.get("partition_contract")
            if contract is not None:
                canonical["partition_contract"] = contract
                signature_payload["partition_contract"] = contract
    canonical["signature"] = _canonical_json(signature_payload)
    return canonical


def aggregate_candidate_votes(
    votes: Iterable[Mapping[str, Any]],
    *,
    allowed_evidence_ids: Iterable[str],
    minimum_votes: int = 3,
    minimum_mean_confidence: float = 0.85,
    minimum_vote_confidence: float = 0.80,
) -> dict[str, Any]:
    """Require structural unanimity, sufficient confidence, and cited evidence."""

    rows = list(votes)
    allowed = set(str(item) for item in allowed_evidence_ids)
    canonical_rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for index, row in enumerate(rows):
        raw = row.get("constraint", row)
        try:
            canonical_rows.append(canonicalize_vote(raw))
        except (TypeError, ValueError, KeyError) as exc:
            parse_errors.append(f"vote_{index}:{type(exc).__name__}:{exc}")
    signatures = sorted(set(row["signature"] for row in canonical_rows))
    evidence_valid = bool(canonical_rows) and all(
        row["evidence_image_ids"] and set(row["evidence_image_ids"]).issubset(allowed)
        for row in canonical_rows
    )
    confidences = [float(row["confidence"]) for row in canonical_rows]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    reasons = []
    if len(rows) < minimum_votes:
        reasons.append("insufficient_vote_count")
    if parse_errors or len(canonical_rows) != len(rows):
        reasons.append("vote_schema_error")
    if len(signatures) != 1:
        reasons.append("structural_vote_disagreement")
    if mean_confidence < minimum_mean_confidence:
        reasons.append("mean_confidence_below_threshold")
    if confidences and min(confidences) < minimum_vote_confidence:
        reasons.append("individual_confidence_below_threshold")
    if not evidence_valid:
        reasons.append("evidence_citation_invalid")
    selected = canonical_rows[0] if len(signatures) == 1 and canonical_rows else None
    result = {
        "stage": GeneratorStage.NORMALIZED.value,
        "vote_count": len(rows),
        "valid_vote_count": len(canonical_rows),
        "minimum_votes": minimum_votes,
        "signature_count": len(signatures),
        "unanimous_structural_signature": len(signatures) == 1 and not parse_errors,
        "mean_confidence": mean_confidence,
        "minimum_confidence": min(confidences) if confidences else 0.0,
        "confidence_gate": (
            mean_confidence >= minimum_mean_confidence
            and bool(confidences)
            and min(confidences) >= minimum_vote_confidence
        ),
        "evidence_gate": evidence_valid,
        "oracle_like_fields_detected": False,
        "parse_errors": parse_errors,
        "defer_reasons": reasons,
        "ready_for_binding": not reasons,
        "selected_proposal": selected,
        "canonical_signature": signatures[0] if len(signatures) == 1 else None,
    }
    result["aggregate_uid"] = _uid("auto_aggregate_", result)
    return result


@dataclass(frozen=True)
class IdentityAliasBinding:
    alias: str
    entity_uid: str | None
    lineage_uid: str | None
    origin_obs_uid: str | None
    identity_uids: tuple[str, ...]
    provenance_lineage_uids: tuple[str, ...]
    complete: bool

    @classmethod
    def from_mapping(
        cls, alias: str, value: Mapping[str, Any]
    ) -> "IdentityAliasBinding":
        identities = tuple(sorted(set(_string_tuple(value.get("identity_uids")))))
        provenance = tuple(
            sorted(set(_string_tuple(value.get("provenance_lineage_uids"))))
        )
        return cls(
            alias=_required_text(alias, "alias"),
            entity_uid=value.get("entity_uid"),
            lineage_uid=value.get("lineage_uid"),
            origin_obs_uid=value.get("origin_obs_uid"),
            identity_uids=identities,
            provenance_lineage_uids=provenance,
            complete=bool(
                value.get("complete")
                and value.get("entity_uid")
                and value.get("lineage_uid")
                and value.get("origin_obs_uid")
                and identities
                and provenance
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["identity_uids"] = list(self.identity_uids)
        value["provenance_lineage_uids"] = list(self.provenance_lineage_uids)
        return value


@dataclass(frozen=True)
class IncidentBinding:
    case_uid: str
    obs_uid: str
    obs_key: str
    event_uid: str
    event_sequence: int
    observed_current_decision: str
    aliases: dict[str, IdentityAliasBinding]
    created_entity_uid: str | None = None
    created_identity_uid: str | None = None
    evidence_refs: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IncidentBinding":
        raw_aliases = value.get("aliases")
        if not isinstance(raw_aliases, Mapping):
            raise ValueError("aliases must be an object")
        aliases = {
            str(alias): IdentityAliasBinding.from_mapping(str(alias), row)
            for alias, row in raw_aliases.items()
        }
        return cls(
            case_uid=_required_text(value.get("case_uid"), "case_uid"),
            obs_uid=_required_text(value.get("obs_uid"), "obs_uid"),
            obs_key=_required_text(value.get("obs_key"), "obs_key"),
            event_uid=_required_text(value.get("event_uid"), "event_uid"),
            event_sequence=int(value.get("event_sequence")),
            observed_current_decision=_required_text(
                value.get("observed_current_decision"),
                "observed_current_decision",
            ).upper(),
            aliases=aliases,
            created_entity_uid=value.get("created_entity_uid"),
            created_identity_uid=value.get("created_identity_uid"),
            evidence_refs=tuple(sorted(set(_string_tuple(value.get("evidence_refs"))))),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_uid": self.case_uid,
            "obs_uid": self.obs_uid,
            "obs_key": self.obs_key,
            "event_uid": self.event_uid,
            "event_sequence": self.event_sequence,
            "observed_current_decision": self.observed_current_decision,
            "aliases": {
                alias: binding.as_dict()
                for alias, binding in sorted(self.aliases.items())
            },
            "created_entity_uid": self.created_entity_uid,
            "created_identity_uid": self.created_identity_uid,
            "evidence_refs": list(self.evidence_refs),
        }


def semantic_constraint_fingerprint(constraint: Mapping[str, Any]) -> str:
    """Hash only fields that can change replay behavior."""

    fields = (
        "type",
        "obs_uid",
        "target_lineage_uid",
        "target_origin_obs_uid",
        "target_entity_uid",
        "created_lineage_uid",
        "created_entity_uid",
        "created_identity_uid",
        "separate_from_identity_uids",
        "partition_contract",
        "entity_uid",
        "groups",
        "label",
        "applies_at_event_uid",
        "active_from_sequence",
        "active_until_sequence",
    )
    payload = {
        key: constraint[key]
        for key in fields
        if key in constraint and constraint[key] not in (None, [], {}, ())
    }
    return _uid("constraint_semantics_", payload)


def _deferred_compilation(
    binding: IncidentBinding,
    aggregate: Mapping[str, Any],
    *reasons: str,
) -> dict[str, Any]:
    result = {
        "case_uid": binding.case_uid,
        "stage": GeneratorStage.DEFERRED.value,
        "aggregate_uid": aggregate.get("aggregate_uid"),
        "candidate_constraint": None,
        "constraint_fingerprint": None,
        "defer_reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
        "binding": binding.as_dict(),
    }
    result["compilation_uid"] = _uid("auto_compilation_", result)
    return result


def _resolve_alias(
    binding: IncidentBinding, alias: str
) -> tuple[IdentityAliasBinding | None, str | None]:
    value = binding.aliases.get(alias)
    if value is None:
        return None, f"unknown_alias:{alias}"
    if not value.complete:
        return None, f"incomplete_identity_binding:{alias}"
    if len(value.identity_uids) != 1:
        return None, f"non_unique_effective_identity:{alias}"
    return value, None


def compile_blind_candidate(
    aggregate: Mapping[str, Any], binding: IncidentBinding
) -> dict[str, Any]:
    """Compile a normalized blind proposal into one exact sparse primitive."""

    if not aggregate.get("ready_for_binding"):
        reasons = aggregate.get("defer_reasons") or ("blind_aggregate_not_ready",)
        return _deferred_compilation(binding, aggregate, *reasons)
    proposal = aggregate.get("selected_proposal")
    if not isinstance(proposal, Mapping):
        return _deferred_compilation(binding, aggregate, "missing_selected_proposal")
    action = AutomaticAction(str(proposal["action"]))
    if action == AutomaticAction.DEFER:
        return _deferred_compilation(binding, aggregate, "generator_requested_defer")
    if action in {
        AutomaticAction.RELABEL,
        AutomaticAction.RESTORE_OBSERVATION_GEOMETRY,
    }:
        return _deferred_compilation(
            binding,
            aggregate,
            f"{action.value.lower()}_executor_and_independent_endpoint_missing",
        )
    if action == AutomaticAction.PARTITION_OBSERVATION:
        return _deferred_compilation(
            binding,
            aggregate,
            "partition_observation_pre_association_executor_not_integrated",
            "partition_observation_point_assignment_requires_hash_bound_gold",
        )

    target_alias: str | None = None
    constraint: dict[str, Any]
    if action == AutomaticAction.SAME_INSTANCE:
        entities = tuple(proposal.get("entities") or ())
        counterparts = tuple(alias for alias in entities if alias != "ANCHOR")
        if "ANCHOR" not in entities or len(counterparts) != 1:
            return _deferred_compilation(
                binding, aggregate, "same_instance_requires_anchor_and_one_context"
            )
        target_alias = counterparts[0]
    elif action == AutomaticAction.MOVE_OBSERVATION:
        if proposal.get("obs_key") != binding.obs_key:
            return _deferred_compilation(binding, aggregate, "observation_key_mismatch")
        target_alias = str(proposal.get("to_alias"))
    elif action == AutomaticAction.SEPARATE_MEMBER_GROUPS:
        groups = tuple(tuple(group) for group in proposal.get("groups") or ())
        anchor_groups = [group for group in groups if "ANCHOR" in group]
        other_groups = [group for group in groups if "ANCHOR" not in group]
        if (
            len(groups) != 2
            or len(anchor_groups) != 1
            or anchor_groups[0] != ("ANCHOR",)
            or len(other_groups) != 1
            or len(other_groups[0]) != 1
        ):
            return _deferred_compilation(
                binding,
                aggregate,
                "separation_requires_two_singleton_groups_including_anchor",
            )
        target_alias = other_groups[0][0]
    else:
        return _deferred_compilation(binding, aggregate, "unsupported_identity_action")

    target, error = _resolve_alias(binding, target_alias)
    if error or target is None:
        return _deferred_compilation(
            binding, aggregate, error or "target_binding_failed"
        )
    common = {
        "obs_uid": binding.obs_uid,
        "applies_at_event_uid": binding.event_uid,
        "active_from_sequence": binding.event_sequence,
        "active_until_sequence": binding.event_sequence,
        "source": "automatic_blind_constraint_v2",
        "evidence_refs": list(binding.evidence_refs),
    }
    if action in {
        AutomaticAction.SAME_INSTANCE,
        AutomaticAction.MOVE_OBSERVATION,
    }:
        constraint = {
            **common,
            "type": "ASSIGN_OBSERVATION",
            "target_lineage_uid": target.lineage_uid,
            "target_origin_obs_uid": target.origin_obs_uid,
            "target_entity_uid": target.entity_uid,
        }
    else:
        if binding.observed_current_decision == "CREATE":
            if not binding.created_entity_uid or not binding.created_identity_uid:
                return _deferred_compilation(
                    binding, aggregate, "created_identity_binding_incomplete"
                )
            created_entity_uid = binding.created_entity_uid
            created_identity_uid = binding.created_identity_uid
        elif binding.observed_current_decision == "ASSOCIATE":
            created_entity_uid = None
            created_identity_uid = "revision-lineage:" + binding.obs_uid
        else:
            return _deferred_compilation(
                binding,
                aggregate,
                "separate_member_groups_requires_native_create_or_associate",
            )
        constraint = {
            **common,
            "type": "CREATE_INSTANCE",
            "created_entity_uid": created_entity_uid,
            "created_lineage_uid": created_identity_uid,
            "created_identity_uid": created_identity_uid,
            "separate_from_identity_uids": list(target.identity_uids),
        }
    try:
        parsed = SparseRepairConstraint.from_mapping(constraint)
    except (TypeError, ValueError) as exc:
        return _deferred_compilation(
            binding,
            aggregate,
            f"constraint_schema_rejected:{type(exc).__name__}:{exc}",
        )
    candidate = parsed.as_dict()
    result = {
        "case_uid": binding.case_uid,
        "stage": GeneratorStage.BOUND_PENDING_SHADOW.value,
        "aggregate_uid": aggregate.get("aggregate_uid"),
        "candidate_constraint": candidate,
        "constraint_fingerprint": semantic_constraint_fingerprint(candidate),
        "defer_reasons": [],
        "target_alias": target_alias,
        "binding": binding.as_dict(),
    }
    result["compilation_uid"] = _uid("auto_compilation_", result)
    return result


def enumerate_identity_hypotheses(
    binding: IncidentBinding,
    *,
    candidate_aliases: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Compile the finite opposite identity hypotheses for independent shadowing.

    Model votes may rank these hypotheses, but they do not remove an executable
    alternative before the independent endpoint evaluator has seen it.
    """

    available = tuple(
        sorted(
            alias
            for alias, value in binding.aliases.items()
            if alias != "ANCHOR" and value.complete
        )
    )
    requested = (
        tuple(dict.fromkeys(str(alias) for alias in candidate_aliases))
        if candidate_aliases is not None
        else available
    )
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            "unknown or incomplete identity aliases: " + ", ".join(unknown)
        )

    compiled_rows = []
    for alias in requested:
        proposals = [
            {
                "action": AutomaticAction.SAME_INSTANCE.value,
                "confidence": 1.0,
                "entities": ["ANCHOR", alias],
                "evidence_image_ids": ["DETERMINISTIC_HYPOTHESIS_ENUMERATION"],
            }
        ]
        if binding.observed_current_decision in {"ASSOCIATE", "CREATE"}:
            proposals.append(
                {
                    "action": AutomaticAction.SEPARATE_MEMBER_GROUPS.value,
                    "confidence": 1.0,
                    "groups": [["ANCHOR"], [alias]],
                    "evidence_image_ids": ["DETERMINISTIC_HYPOTHESIS_ENUMERATION"],
                }
            )
        for proposal in proposals:
            canonical = canonicalize_vote(proposal)
            aggregate = {
                "ready_for_binding": True,
                "selected_proposal": canonical,
                "aggregate_uid": _uid("hypothesis_aggregate_", canonical),
                "defer_reasons": [],
            }
            compiled = compile_blind_candidate(aggregate, binding)
            compiled["hypothesis_action"] = canonical["action"]
            compiled["hypothesis_target_alias"] = alias
            compiled["hypothesis_source"] = "FINITE_DETERMINISTIC_ENUMERATION"
            compiled_rows.append(compiled)
    return compiled_rows


@dataclass(frozen=True)
class ShadowGateEvidence:
    constraint_fingerprint: str
    endpoint_improved: bool
    collateral_safe: bool
    invariants_pass: bool
    source_immutable: bool
    no_op_controls_pass: bool
    legal_merge_control_pass: bool
    component_mechanism_supported: bool
    local_global_parity_pass: bool
    evaluation_independent_of_generator: bool
    artifact_refs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ShadowGateEvidence":
        return cls(
            constraint_fingerprint=_required_text(
                value.get("constraint_fingerprint"), "constraint_fingerprint"
            ),
            endpoint_improved=bool(value.get("endpoint_improved")),
            collateral_safe=bool(value.get("collateral_safe")),
            invariants_pass=bool(value.get("invariants_pass")),
            source_immutable=bool(value.get("source_immutable")),
            no_op_controls_pass=bool(value.get("no_op_controls_pass")),
            legal_merge_control_pass=bool(value.get("legal_merge_control_pass")),
            component_mechanism_supported=bool(
                value.get("component_mechanism_supported")
            ),
            local_global_parity_pass=bool(value.get("local_global_parity_pass")),
            evaluation_independent_of_generator=bool(
                value.get("evaluation_independent_of_generator")
            ),
            artifact_refs=tuple(sorted(set(_string_tuple(value.get("artifact_refs"))))),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifact_refs"] = list(self.artifact_refs)
        return value


def decide_automatic_promotion(
    compiled: Mapping[str, Any],
    shadow: ShadowGateEvidence | None,
) -> dict[str, Any]:
    """Expose a commit constraint only after exact-fingerprint shadow validation."""

    reasons = list(compiled.get("defer_reasons") or ())
    candidate = compiled.get("candidate_constraint")
    fingerprint = compiled.get("constraint_fingerprint")
    gate_values: dict[str, bool] = {}
    fingerprint_match = False
    if (
        compiled.get("stage") != GeneratorStage.BOUND_PENDING_SHADOW.value
        or not candidate
    ):
        reasons.append("no_bound_candidate")
    elif shadow is None:
        reasons.append("shadow_evidence_missing")
    else:
        fingerprint_match = fingerprint == shadow.constraint_fingerprint
        if not fingerprint_match:
            reasons.append("shadow_constraint_fingerprint_mismatch")
        gate_values = {
            name: bool(getattr(shadow, name)) for name in _REQUIRED_SHADOW_GATES
        }
        reasons.extend(
            f"shadow_gate_failed:{name}"
            for name, passed in gate_values.items()
            if not passed
        )
        if not shadow.artifact_refs:
            reasons.append("shadow_artifact_refs_missing")
    eligible = not reasons and fingerprint_match and all(gate_values.values())
    result = {
        "case_uid": compiled.get("case_uid"),
        "stage": (
            GeneratorStage.COMMIT_ELIGIBLE.value
            if eligible
            else GeneratorStage.DEFERRED.value
        ),
        "compilation_uid": compiled.get("compilation_uid"),
        "constraint_fingerprint": fingerprint,
        "shadow_fingerprint_match": fingerprint_match,
        "shadow_gates": gate_values,
        "shadow_artifact_refs": list(shadow.artifact_refs) if shadow else [],
        "defer_reasons": list(dict.fromkeys(reasons)),
        "commit_constraint": candidate if eligible else None,
        "candidate_retained_for_audit": candidate,
    }
    result["promotion_uid"] = _uid("auto_promotion_", result)
    return result
