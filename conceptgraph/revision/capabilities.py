from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from .constraints import ConstraintType


class RequiredRepairCapability(str, Enum):
    """Endpoint-level capability required before any concrete constraint is built."""

    RESTORE_OBSERVATION_GEOMETRY = "RESTORE_OBSERVATION_GEOMETRY"
    SUPPRESS_SPURIOUS_OBJECT = "SUPPRESS_SPURIOUS_OBJECT"


class CapabilityDisposition(str, Enum):
    EXECUTABLE_CANDIDATE = "EXECUTABLE_CANDIDATE"
    DEFER_UNSUPPORTED = "DEFER_UNSUPPORTED"


class FeasibleRepairAction(str, Enum):
    """Finite production actions enumerated from executable payloads only."""

    NO_OP = "NO_OP"
    SAME_INSTANCE = "SAME_INSTANCE"
    SEPARATE_MEMBER_GROUPS = "SEPARATE_MEMBER_GROUPS"
    RESTORE_OBSERVATION_GEOMETRY = "RESTORE_OBSERVATION_GEOMETRY"
    SUPPRESS_OBSERVATION = "SUPPRESS_OBSERVATION"
    PARTITION_OBSERVATION = "PARTITION_OBSERVATION"


@dataclass(frozen=True)
class CapabilityResolution:
    required_capability: RequiredRepairCapability
    disposition: CapabilityDisposition
    automatic_action: str
    executable_constraint_type: ConstraintType | None
    reason: str

    @property
    def executable(self) -> bool:
        return self.executable_constraint_type is not None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["required_capability"] = self.required_capability.value
        value["disposition"] = self.disposition.value
        value["executable_constraint_type"] = (
            self.executable_constraint_type.value
            if self.executable_constraint_type is not None
            else None
        )
        value["executable"] = self.executable
        return value


_CAPABILITY_REGISTRY = {
    RequiredRepairCapability.RESTORE_OBSERVATION_GEOMETRY: CapabilityResolution(
        required_capability=RequiredRepairCapability.RESTORE_OBSERVATION_GEOMETRY,
        disposition=CapabilityDisposition.EXECUTABLE_CANDIDATE,
        automatic_action="RESTORE_OBSERVATION_GEOMETRY",
        executable_constraint_type=ConstraintType.RESTORE_OBSERVATION_GEOMETRY,
        reason="hash_bound_geometry_restoration_primitive_is_available",
    ),
    RequiredRepairCapability.SUPPRESS_SPURIOUS_OBJECT: CapabilityResolution(
        required_capability=RequiredRepairCapability.SUPPRESS_SPURIOUS_OBJECT,
        disposition=CapabilityDisposition.DEFER_UNSUPPORTED,
        automatic_action="DEFER",
        executable_constraint_type=None,
        reason="no_safe_suppress_or_delete_primitive_is_implemented",
    ),
}

_ENDPOINT_REQUIREMENTS = {
    "GEOMETRY_CORRUPTION": RequiredRepairCapability.RESTORE_OBSERVATION_GEOMETRY,
    "SPURIOUS_OBJECT": RequiredRepairCapability.SUPPRESS_SPURIOUS_OBJECT,
}


def resolve_required_capability(
    capability: RequiredRepairCapability | str,
) -> CapabilityResolution:
    return _CAPABILITY_REGISTRY[RequiredRepairCapability(capability)]


def resolve_endpoint_capability(endpoint_error_type: str) -> CapabilityResolution:
    endpoint = str(endpoint_error_type).strip().upper()
    if endpoint not in _ENDPOINT_REQUIREMENTS:
        raise ValueError(f"unsupported endpoint capability routing: {endpoint}")
    return resolve_required_capability(_ENDPOINT_REQUIREMENTS[endpoint])


def executable_constraint_types() -> tuple[str, ...]:
    return tuple(sorted(item.value for item in ConstraintType))


def enumerate_feasible_actions(
    *,
    identity_candidate_count: int = 0,
    observed_current_decision: str | None = None,
    created_identity_binding_complete: bool = False,
    geometry_contract: Mapping[str, Any] | None = None,
    partition_contract: Mapping[str, Any] | None = None,
    partition_preassociation_integrated: bool = False,
    suppression_executor_integrated: bool = False,
) -> tuple[str, ...]:
    """Enumerate executable hypotheses without consuming endpoint error labels.

    This replaces endpoint-type routing in the production proposal path. The old
    `resolve_endpoint_capability` API remains for benchmark stratification only.
    """

    if identity_candidate_count < 0:
        raise ValueError("identity_candidate_count cannot be negative")
    actions = [FeasibleRepairAction.NO_OP]
    if identity_candidate_count:
        actions.append(FeasibleRepairAction.SAME_INSTANCE)
        observed = str(observed_current_decision or "").upper()
        if observed == "ASSOCIATE" or (
            observed == "CREATE" and created_identity_binding_complete
        ):
            actions.append(FeasibleRepairAction.SEPARATE_MEMBER_GROUPS)
    if geometry_contract:
        actions.append(FeasibleRepairAction.RESTORE_OBSERVATION_GEOMETRY)
    if suppression_executor_integrated:
        actions.append(FeasibleRepairAction.SUPPRESS_OBSERVATION)
    if partition_contract and partition_preassociation_integrated:
        actions.append(FeasibleRepairAction.PARTITION_OBSERVATION)
    return tuple(action.value for action in actions)
