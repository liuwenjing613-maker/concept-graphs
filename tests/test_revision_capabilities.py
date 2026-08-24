from conceptgraph.revision.capabilities import (
    CapabilityDisposition,
    RequiredRepairCapability,
    executable_constraint_types,
    resolve_endpoint_capability,
    resolve_required_capability,
)
from conceptgraph.revision.constraints import ConstraintType


def test_geometry_capability_resolves_to_hash_bound_executor():
    result = resolve_endpoint_capability("GEOMETRY_CORRUPTION")

    assert result.disposition == CapabilityDisposition.EXECUTABLE_CANDIDATE
    assert result.automatic_action == "RESTORE_OBSERVATION_GEOMETRY"
    assert (
        result.executable_constraint_type == ConstraintType.RESTORE_OBSERVATION_GEOMETRY
    )
    assert result.executable


def test_spurious_object_fails_closed_without_suppression_primitive():
    result = resolve_endpoint_capability("SPURIOUS_OBJECT")

    assert (
        result.required_capability == RequiredRepairCapability.SUPPRESS_SPURIOUS_OBJECT
    )
    assert result.disposition == CapabilityDisposition.DEFER_UNSUPPORTED
    assert result.automatic_action == "DEFER"
    assert result.executable_constraint_type is None
    assert not result.executable


def test_registry_cannot_claim_a_delete_or_suppress_executor():
    supported = set(executable_constraint_types())

    assert "DELETE" not in supported
    assert "DELETE_ENTITY" not in supported
    assert "SUPPRESS" not in supported
    assert "SUPPRESS_OBJECT" not in supported
    assert (
        resolve_required_capability("SUPPRESS_SPURIOUS_OBJECT").as_dict()["executable"]
        is False
    )
