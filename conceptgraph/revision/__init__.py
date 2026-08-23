"""Evidence-backed counterfactual revision primitives.

The package is intentionally inert unless ``revision.enabled`` is true. It never
mutates a baseline evidence directory or source map in place.
"""

from .corruption import ControlledCorruptionController, load_corruption_plan
from .index import EvidenceIntegrityError, LineageIndex, ProvenanceIndex
from .models import (
    ConflictType,
    CorruptionPlan,
    DependencyClosure,
    RepairConstraint,
    RepairTicket,
    RevisionTransaction,
)
from .relations import AliDevBaselineRelationBackend
from .replay import CounterfactualReplayEngine
from .tracing import CausalTracer
from .transactions import ShadowTransactionManager
from .verify import StructuralVerifier

__all__ = [
    "AliDevBaselineRelationBackend",
    "CausalTracer",
    "ConflictType",
    "ControlledCorruptionController",
    "CorruptionPlan",
    "CounterfactualReplayEngine",
    "DependencyClosure",
    "EvidenceIntegrityError",
    "LineageIndex",
    "ProvenanceIndex",
    "RepairConstraint",
    "RepairTicket",
    "RevisionTransaction",
    "ShadowTransactionManager",
    "StructuralVerifier",
    "load_corruption_plan",
]
