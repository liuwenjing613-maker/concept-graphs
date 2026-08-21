from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .evidence import EndpointEvidence
from .schemas import Diagnosis, EvidenceAudit, SchemaError, Verification


ACTION_BY_ERROR = {
    "SEMANTIC_IDENTITY_ERROR": {"RELABEL", "ABSTAIN"},
    "SPURIOUS_OBJECT": {"DELETE", "ABSTAIN"},
    "FALSE_SPLIT": {"MERGE_WITH", "ABSTAIN"},
    "FALSE_MERGE": {"SPLIT_OBJECT", "ABSTAIN"},
    "WRONG_MEMBERSHIP": {"REASSIGN_MEMBERS", "ABSTAIN"},
    "GEOMETRY_CORRUPTION": {"TRIM_GEOMETRY", "SPLIT_OBJECT", "DELETE", "ABSTAIN"},
    "MISSING_OBJECT": {"ABSTAIN"},
    "OTHER": {"ABSTAIN"},
}

MIN_DIAGNOSIS_CONFIDENCE = {
    "RELABEL": 0.85,
    "DELETE": 0.97,
    "MERGE_WITH": 0.95,
}
MIN_VERIFICATION_CONFIDENCE = {
    "RELABEL": 0.80,
    "DELETE": 0.95,
    "MERGE_WITH": 0.90,
}


@dataclass(frozen=True)
class ExecutionDecision:
    status: str
    executable: bool
    reason: str
    action: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_audit_against_evidence(
    evidence: EndpointEvidence, audit: EvidenceAudit
) -> None:
    if audit.target_alias != evidence.target_alias:
        raise SchemaError(
            f"audit targeted {audit.target_alias}, expected {evidence.target_alias}"
        )
    expected_context = set(evidence.alias_to_uid) - {evidence.target_alias}
    actual_context = [relation.other_alias for relation in audit.context_relations]
    if len(actual_context) != len(set(actual_context)):
        raise SchemaError("audit contains duplicate context aliases")
    if set(actual_context) != expected_context:
        raise SchemaError(
            "audit context aliases differ from packet: "
            f"expected={sorted(expected_context)}, got={sorted(actual_context)}"
        )


def validate_against_evidence(
    evidence: EndpointEvidence, diagnosis: Diagnosis
) -> None:
    if diagnosis.target_alias != evidence.target_alias:
        raise SchemaError(
            f"model targeted {diagnosis.target_alias}, expected {evidence.target_alias}"
        )
    action = diagnosis.repair.action
    if diagnosis.final_state == "WRONG":
        allowed = ACTION_BY_ERROR.get(diagnosis.error_type, {"ABSTAIN"})
        if action not in allowed:
            raise SchemaError(
                f"{diagnosis.error_type} cannot request {action}; allowed={sorted(allowed)}"
            )
    if action == "MERGE_WITH":
        other = (diagnosis.repair.other_alias or "").upper()
        if other not in evidence.alias_to_uid:
            raise SchemaError(f"MERGE_WITH references unknown alias: {other}")
        if other == evidence.target_alias:
            raise SchemaError("MERGE_WITH cannot target the endpoint itself")
    if action == "RELABEL":
        saved = str(evidence.target_object.get("class_name") or "").strip().casefold()
        new = str(diagnosis.repair.new_label or "").strip().casefold()
        if saved == new:
            raise SchemaError("RELABEL new_label is identical to the saved label")
    referenced_views = {
        alias
        for aliases in diagnosis.repair.member_view_groups.values()
        for alias in aliases
    }
    if any(not alias.startswith("V") for alias in referenced_views):
        raise SchemaError("member_view_groups may reference only V* representative views")


def assess_execution(
    evidence: EndpointEvidence,
    diagnosis: Diagnosis,
    verification: Verification | None,
) -> ExecutionDecision:
    action = diagnosis.repair.action
    if action in {"KEEP", "ABSTAIN"}:
        return ExecutionDecision(
            status="NO_MUTATION",
            executable=False,
            reason="The VLM requested no map mutation.",
            action=action,
        )
    if action in {"SPLIT_OBJECT", "REASSIGN_MEMBERS", "TRIM_GEOMETRY"}:
        return ExecutionDecision(
            status="NEEDS_FULL_MEMBER_PASS",
            executable=False,
            reason=(
                "Representative endpoint evidence can diagnose this repair family, but cannot "
                "safely assign every member observation. The proposal is retained without mutation."
            ),
            action=action,
        )
    if action not in MIN_DIAGNOSIS_CONFIDENCE:
        return ExecutionDecision(
            status="UNSUPPORTED_ACTION",
            executable=False,
            reason=f"No isolated executor is registered for {action}.",
            action=action,
        )
    if diagnosis.confidence < MIN_DIAGNOSIS_CONFIDENCE[action]:
        return ExecutionDecision(
            status="BELOW_DIAGNOSIS_THRESHOLD",
            executable=False,
            reason=(
                f"Diagnosis confidence {diagnosis.confidence:.3f} is below the "
                f"{MIN_DIAGNOSIS_CONFIDENCE[action]:.3f} threshold for {action}."
            ),
            action=action,
        )
    if verification is None:
        return ExecutionDecision(
            status="VERIFICATION_REQUIRED",
            executable=False,
            reason="Every mutating repair requires an independent second VLM pass.",
            action=action,
        )
    if not verification.approve:
        return ExecutionDecision(
            status="VERIFIER_REJECTED",
            executable=False,
            reason=verification.reason,
            action=action,
        )
    if verification.confidence < MIN_VERIFICATION_CONFIDENCE[action]:
        return ExecutionDecision(
            status="BELOW_VERIFICATION_THRESHOLD",
            executable=False,
            reason=(
                f"Verification confidence {verification.confidence:.3f} is below the "
                f"{MIN_VERIFICATION_CONFIDENCE[action]:.3f} threshold for {action}."
            ),
            action=action,
        )
    return ExecutionDecision(
        status="APPROVED_FOR_DERIVED_MAP",
        executable=True,
        reason="Both VLM passes support the smallest registered isolated repair.",
        action=action,
    )
