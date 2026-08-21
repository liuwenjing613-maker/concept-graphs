from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


FINAL_STATES = {"CORRECT", "WRONG", "UNCLEAR"}
ERROR_TYPES = {
    "NOT_APPLICABLE",
    "FALSE_MERGE",
    "FALSE_SPLIT",
    "SPURIOUS_OBJECT",
    "MISSING_OBJECT",
    "WRONG_MEMBERSHIP",
    "GEOMETRY_CORRUPTION",
    "SEMANTIC_IDENTITY_ERROR",
    "OTHER",
}
REPAIR_ACTIONS = {
    "KEEP",
    "RELABEL",
    "DELETE",
    "MERGE_WITH",
    "SPLIT_OBJECT",
    "REASSIGN_MEMBERS",
    "TRIM_GEOMETRY",
    "ABSTAIN",
}
AUDIT_CHECKS = {
    "REAL_USABLE_OBJECT",
    "SAVED_LABEL_MATCH",
    "SINGLE_PHYSICAL_OBJECT",
    "GEOMETRY_COMPLETE_AND_WELL_PLACED",
    "MEMBERSHIP_COHERENT",
    "NOT_BACKGROUND_OR_NOISE",
}
AUDIT_HYPOTHESES = {
    "CORRECT",
    "FALSE_MERGE",
    "FALSE_SPLIT",
    "SPURIOUS_OBJECT",
    "WRONG_MEMBERSHIP",
    "GEOMETRY_CORRUPTION",
    "SEMANTIC_IDENTITY_ERROR",
    "OTHER",
}
CONTEXT_RELATIONSHIPS = {
    "SAME_OBJECT_DUPLICATE",
    "SAME_OBJECT_COMPLEMENTARY_FRAGMENT",
    "DIFFERENT_OBJECT",
    "UNCLEAR",
}


class SchemaError(ValueError):
    """Raised when a VLM response cannot be made safe and unambiguous."""


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaError(f"{field_name} must be a string or null")
    value = value.strip()
    return value or None


def _confidence(value: Any, field_name: str = "confidence") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field_name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SchemaError(f"{field_name} must be in [0, 1]")
    return value


def _text_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaError(f"{field_name} must be a string list")
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(_required_text(item, f"{field_name}[{index}]"))
    return tuple(normalized)


@dataclass(frozen=True)
class AuditCheck:
    probability: float
    evidence: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], field_name: str) -> "AuditCheck":
        if not isinstance(value, Mapping):
            raise SchemaError(f"{field_name} must be an object")
        return cls(
            probability=_confidence(value.get("probability"), f"{field_name}.probability"),
            evidence=_required_text(value.get("evidence"), f"{field_name}.evidence"),
        )


@dataclass(frozen=True)
class ContextRelation:
    other_alias: str
    same_physical_object_probability: float
    relationship: str
    evidence: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "ContextRelation":
        if not isinstance(value, Mapping):
            raise SchemaError(f"context_relations[{index}] must be an object")
        relationship = _required_text(
            value.get("relationship"), f"context_relations[{index}].relationship"
        ).upper()
        if relationship not in CONTEXT_RELATIONSHIPS:
            raise SchemaError(f"unsupported context relationship: {relationship}")
        return cls(
            other_alias=_required_text(
                value.get("other_alias"), f"context_relations[{index}].other_alias"
            ).upper(),
            same_physical_object_probability=_confidence(
                value.get("same_physical_object_probability"),
                f"context_relations[{index}].same_physical_object_probability",
            ),
            relationship=relationship,
            evidence=_required_text(
                value.get("evidence"), f"context_relations[{index}].evidence"
            ),
        )


@dataclass(frozen=True)
class EvidenceAudit:
    """A forced, non-terminal VLM audit that must test every failure family."""

    schema_version: str
    target_alias: str
    best_physical_identity: str
    checks: dict[str, AuditCheck]
    context_relations: tuple[ContextRelation, ...]
    hypothesis_probabilities: dict[str, float]
    leading_hypothesis: str
    evidence_for_wrong: tuple[str, ...]
    evidence_for_correct: tuple[str, ...]
    evidence_gaps: tuple[str, ...]
    audit_summary: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceAudit":
        if not isinstance(value, Mapping):
            raise SchemaError("evidence audit must be an object")

        checks_value = value.get("checks")
        if not isinstance(checks_value, Mapping):
            raise SchemaError("checks must be an object")
        normalized_checks_value = {
            str(name).upper(): item for name, item in checks_value.items()
        }
        check_names = set(normalized_checks_value)
        if check_names != AUDIT_CHECKS:
            raise SchemaError(
                "checks must contain exactly "
                f"{sorted(AUDIT_CHECKS)}; got {sorted(check_names)}"
            )
        checks = {
            name: AuditCheck.from_mapping(normalized_checks_value[name], f"checks.{name}")
            for name in sorted(AUDIT_CHECKS)
        }

        hypotheses_value = value.get("hypothesis_probabilities")
        if not isinstance(hypotheses_value, Mapping):
            raise SchemaError("hypothesis_probabilities must be an object")
        normalized_hypotheses_value = {
            str(name).upper(): item for name, item in hypotheses_value.items()
        }
        hypothesis_names = set(normalized_hypotheses_value)
        if hypothesis_names != AUDIT_HYPOTHESES:
            raise SchemaError(
                "hypothesis_probabilities must contain exactly "
                f"{sorted(AUDIT_HYPOTHESES)}; got {sorted(hypothesis_names)}"
            )
        hypothesis_probabilities = {
            name: _confidence(
                normalized_hypotheses_value[name], f"hypothesis_probabilities.{name}"
            )
            for name in sorted(AUDIT_HYPOTHESES)
        }
        leading = _required_text(
            value.get("leading_hypothesis"), "leading_hypothesis"
        ).upper()
        if leading not in AUDIT_HYPOTHESES:
            raise SchemaError(f"unsupported leading_hypothesis: {leading}")

        relations_value = value.get("context_relations")
        if not isinstance(relations_value, list):
            raise SchemaError("context_relations must be a list")
        relations = tuple(
            ContextRelation.from_mapping(item, index)
            for index, item in enumerate(relations_value)
        )

        return cls(
            schema_version=_required_text(
                value.get("schema_version", "1.0.0"), "schema_version"
            ),
            target_alias=_required_text(value.get("target_alias"), "target_alias").upper(),
            best_physical_identity=_required_text(
                value.get("best_physical_identity"), "best_physical_identity"
            ),
            checks=checks,
            context_relations=relations,
            hypothesis_probabilities=hypothesis_probabilities,
            leading_hypothesis=leading,
            evidence_for_wrong=_text_list(
                value.get("evidence_for_wrong"), "evidence_for_wrong"
            ),
            evidence_for_correct=_text_list(
                value.get("evidence_for_correct"), "evidence_for_correct"
            ),
            evidence_gaps=_text_list(value.get("evidence_gaps"), "evidence_gaps"),
            audit_summary=_required_text(value.get("audit_summary"), "audit_summary"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Repair:
    action: str
    target_alias: str
    new_label: str | None = None
    other_alias: str | None = None
    member_view_groups: dict[str, list[str]] = field(default_factory=dict)
    rationale: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Repair":
        if not isinstance(value, Mapping):
            raise SchemaError("repair must be an object")
        action = _required_text(value.get("action"), "repair.action").upper()
        if action not in REPAIR_ACTIONS:
            raise SchemaError(f"unsupported repair.action: {action}")
        target_alias = _required_text(
            value.get("target_alias"), "repair.target_alias"
        ).upper()
        groups = value.get("member_view_groups") or {}
        if not isinstance(groups, Mapping):
            raise SchemaError("repair.member_view_groups must be an object")
        normalized_groups: dict[str, list[str]] = {}
        for group_name, aliases in groups.items():
            group_name = _required_text(group_name, "member group name")
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) and alias.strip() for alias in aliases
            ):
                raise SchemaError("each member view group must be a string list")
            normalized_groups[group_name] = [alias.strip().upper() for alias in aliases]
        return cls(
            action=action,
            target_alias=target_alias,
            new_label=_optional_text(value.get("new_label"), "repair.new_label"),
            other_alias=(
                _optional_text(value.get("other_alias"), "repair.other_alias") or None
            ),
            member_view_groups=normalized_groups,
            rationale=_required_text(value.get("rationale"), "repair.rationale"),
        )


@dataclass(frozen=True)
class Diagnosis:
    schema_version: str
    target_alias: str
    evidence_sufficient: bool
    final_state: str
    error_type: str
    confidence: float
    physical_identity: str
    diagnosis: str
    repair: Repair

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Diagnosis":
        if not isinstance(value, Mapping):
            raise SchemaError("diagnosis must be an object")
        sufficient = value.get("evidence_sufficient")
        if not isinstance(sufficient, bool):
            raise SchemaError("evidence_sufficient must be boolean")
        final_state = _required_text(value.get("final_state"), "final_state").upper()
        error_type = _required_text(value.get("error_type"), "error_type").upper()
        if final_state not in FINAL_STATES:
            raise SchemaError(f"unsupported final_state: {final_state}")
        if error_type not in ERROR_TYPES:
            raise SchemaError(f"unsupported error_type: {error_type}")
        diagnosis = cls(
            schema_version=_required_text(
                value.get("schema_version", "1.0.0"), "schema_version"
            ),
            target_alias=_required_text(value.get("target_alias"), "target_alias").upper(),
            evidence_sufficient=sufficient,
            final_state=final_state,
            error_type=error_type,
            confidence=_confidence(value.get("confidence")),
            physical_identity=_required_text(
                value.get("physical_identity"), "physical_identity"
            ),
            diagnosis=_required_text(value.get("diagnosis"), "diagnosis"),
            repair=Repair.from_mapping(value.get("repair") or {}),
        )
        diagnosis._validate_cross_fields()
        return diagnosis

    def _validate_cross_fields(self) -> None:
        if self.target_alias != self.repair.target_alias:
            raise SchemaError("diagnosis and repair target aliases differ")
        if not self.evidence_sufficient and self.final_state != "UNCLEAR":
            raise SchemaError("insufficient evidence requires UNCLEAR")
        if self.final_state == "CORRECT":
            if self.error_type != "NOT_APPLICABLE" or self.repair.action != "KEEP":
                raise SchemaError("CORRECT requires NOT_APPLICABLE + KEEP")
        elif self.final_state == "UNCLEAR":
            if self.error_type != "NOT_APPLICABLE" or self.repair.action != "ABSTAIN":
                raise SchemaError("UNCLEAR requires NOT_APPLICABLE + ABSTAIN")
        else:
            if self.error_type == "NOT_APPLICABLE":
                raise SchemaError("WRONG requires a concrete error_type")
            if self.repair.action == "KEEP":
                raise SchemaError("WRONG cannot use KEEP")
        if self.repair.action == "RELABEL" and not self.repair.new_label:
            raise SchemaError("RELABEL requires repair.new_label")
        if self.repair.action == "MERGE_WITH" and not self.repair.other_alias:
            raise SchemaError("MERGE_WITH requires repair.other_alias")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Verification:
    approve: bool
    confidence: float
    diagnosis_supported: bool
    action_supported: bool
    reason: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Verification":
        if not isinstance(value, Mapping):
            raise SchemaError("verification must be an object")
        for key in ("approve", "diagnosis_supported", "action_supported"):
            if not isinstance(value.get(key), bool):
                raise SchemaError(f"verification.{key} must be boolean")
        result = cls(
            approve=value["approve"],
            confidence=_confidence(value.get("confidence"), "verification.confidence"),
            diagnosis_supported=value["diagnosis_supported"],
            action_supported=value["action_supported"],
            reason=_required_text(value.get("reason"), "verification.reason"),
        )
        if result.approve and not (
            result.diagnosis_supported and result.action_supported
        ):
            raise SchemaError("approval requires both support flags")
        return result

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object without accepting trailing model commentary."""
    if not isinstance(text, str) or not text.strip():
        raise SchemaError("empty model response")
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"model response is not a single JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError("model response must be a JSON object")
    return value
