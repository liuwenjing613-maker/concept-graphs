from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .index import ProvenanceIndex
from .models import DependencyClosure, RepairConstraint, RevisionTransaction
from .verify import StructuralVerifier


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


class ShadowTransactionManager:
    """Verify and atomically publish a derived result without touching baseline state."""

    def __init__(self, provenance: ProvenanceIndex, output_root: str | Path) -> None:
        self.provenance = provenance
        self.output_root = Path(output_root)
        self.verifier = StructuralVerifier(provenance)

    def prepare(
        self,
        *,
        case: Mapping[str, Any],
        trace: Mapping[str, Any],
        constraint: Mapping[str, Any],
    ) -> RevisionTransaction:
        closure = DependencyClosure.build(**trace["dependency_closure"])
        repair_constraint = RepairConstraint.from_mapping(constraint)
        base_versions = {}
        for entity in closure.entity_uids:
            version = self.provenance.get_current_version(entity)
            if version is not None:
                base_versions[entity] = str(version["object_version_uid"])
        return RevisionTransaction(
            case_uid=str(case["case_uid"]),
            causal_anchor_event_uid=str(trace["causal_anchor_event_uid"]),
            base_event_watermark=int(self.provenance.max_sequence),
            base_entity_versions=base_versions,
            read_set=tuple(sorted(set(closure.obs_uids) | set(closure.version_uids))),
            write_set=closure.entity_uids,
            dependency_closure=closure,
            repair_constraint=repair_constraint,
        )

    def verify_and_commit(
        self,
        *,
        transaction: RevisionTransaction,
        baseline_state: Mapping[str, Any],
        derived_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        case_root = self.output_root / transaction.case_uid
        shadow_path = case_root / "shadow" / "derived_state.json"
        _write_json(shadow_path, derived_state)
        transaction.shadow_output_refs = (str(shadow_path),)
        verification = self.verifier.verify(
            baseline_state=baseline_state,
            derived_state=derived_state,
            closure=transaction.dependency_closure.as_dict(),
            expected_source_hashes=baseline_state["source_hashes"],
        )
        transaction.verification = verification
        if verification["pass"]:
            derived_path = case_root / "derived" / "derived_state.json"
            _write_json(derived_path, derived_state)
            transaction.shadow_output_refs = (str(shadow_path), str(derived_path))
            transaction.commit_status = "COMMITTED"
        else:
            transaction.commit_status = "ABORTED"
        _write_json(case_root / "transaction.json", transaction.as_dict())
        _write_json(case_root / "verification.json", verification)
        return {
            "transaction": transaction.as_dict(),
            "verification": verification,
        }
