"""Evidence-backed counterfactual revision primitives.

The package is intentionally inert unless ``revision.enabled`` is true. It never
mutates a baseline evidence directory or source map in place.
"""

from .corruption import ControlledCorruptionController, load_corruption_plan
from .constraints import (
    ConstraintAction,
    ConstraintEngine,
    ConstraintType,
    ReplayMode,
    SparseRepairConstraint,
)
from .dependency_graph import TypedDependencyGraph
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
from .runtime_verify import InvariantVerifier
from .snapshot import AnchorStateBuilder
from .sparse_replay import SparseCounterfactualReplayEngine
from .tracing import CausalTracer
from .transactions import ShadowTransactionManager
from .verify import StructuralVerifier

__all__ = [
    "AliDevBaselineRelationBackend",
    "CausalTracer",
    "ConflictType",
    "ConstraintAction",
    "ConstraintEngine",
    "ConstraintType",
    "ControlledCorruptionController",
    "CorruptionPlan",
    "CounterfactualReplayEngine",
    "DependencyClosure",
    "EvidenceIntegrityError",
    "LineageIndex",
    "InvariantVerifier",
    "ProvenanceIndex",
    "RepairConstraint",
    "RepairTicket",
    "ReplayMode",
    "RevisionTransaction",
    "ShadowTransactionManager",
    "StructuralVerifier",
    "SparseCounterfactualReplayEngine",
    "SparseRepairConstraint",
    "AnchorStateBuilder",
    "TypedDependencyGraph",
    "load_corruption_plan",
]
