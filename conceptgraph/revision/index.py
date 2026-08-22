from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


class EvidenceIntegrityError(RuntimeError):
    """Raised when an evidence ledger is internally inconsistent."""


CORE_FILES = (
    "observations.jsonl",
    "associations.jsonl",
    "mapping_events.jsonl",
    "object_versions.jsonl",
    "object_pair_decisions.jsonl",
    "final_membership.json",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceIntegrityError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise EvidenceIntegrityError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _unique_index(rows: Iterable[dict[str, Any]], key: str, source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise EvidenceIntegrityError(f"{source}: missing {key}")
        if value in result:
            raise EvidenceIntegrityError(f"{source}: duplicate {key}={value}")
        result[value] = row
    return result


class ProvenanceIndex:
    """In-memory, read-only indexes over one frozen evidence ledger."""

    def __init__(self, root: str | Path, *, validate: bool = True) -> None:
        supplied = Path(root).expanduser().resolve()
        self.evidence_root = supplied if supplied.name == "evidence" else supplied / "evidence"
        self.experiment_root = self.evidence_root.parent
        missing = [name for name in CORE_FILES if not (self.evidence_root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing evidence files: {missing}")

        self.observation_rows = _read_jsonl(self.evidence_root / "observations.jsonl")
        self.association_rows = _read_jsonl(self.evidence_root / "associations.jsonl")
        self.mapping_event_rows = _read_jsonl(self.evidence_root / "mapping_events.jsonl")
        self.object_version_rows = _read_jsonl(self.evidence_root / "object_versions.jsonl")
        self.object_pair_decision_rows = _read_jsonl(
            self.evidence_root / "object_pair_decisions.jsonl"
        )
        with (self.evidence_root / "final_membership.json").open(encoding="utf-8") as handle:
            self.final_membership = json.load(handle)
        if not isinstance(self.final_membership, list):
            raise EvidenceIntegrityError("final_membership.json must contain a list")

        self.observations = _unique_index(self.observation_rows, "obs_uid", "observations")
        self.associations = _unique_index(self.association_rows, "event_uid", "associations")
        self.mapping_events = _unique_index(self.mapping_event_rows, "event_uid", "mapping_events")
        self.object_versions = _unique_index(
            self.object_version_rows, "object_version_uid", "object_versions"
        )
        self.events = {**self.associations, **self.mapping_events}
        if len(self.events) != len(self.associations) + len(self.mapping_events):
            raise EvidenceIntegrityError("association and mapping event UIDs overlap")

        self.association_for_obs: dict[str, dict[str, Any]] = {}
        self.versions_for_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.parent_versions: dict[str, tuple[str, ...]] = {}
        self.child_versions: dict[str, list[str]] = defaultdict(list)
        self.events_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.events_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.final_by_object: dict[str, dict[str, Any]] = {}

        for association in self.association_rows:
            obs_uid = association.get("obs_uid")
            if obs_uid in self.association_for_obs:
                raise EvidenceIntegrityError(f"multiple associations for observation {obs_uid}")
            self.association_for_obs[obs_uid] = association
        for version in self.object_version_rows:
            uid = version["object_version_uid"]
            object_uid = version.get("object_uid")
            if not isinstance(object_uid, str):
                raise EvidenceIntegrityError(f"version without object_uid: {uid}")
            self.versions_for_object[object_uid].append(version)
            parents = tuple(version.get("parent_version_uids") or ())
            self.parent_versions[uid] = parents
            for parent in parents:
                self.child_versions[parent].append(uid)
        for versions in self.versions_for_object.values():
            versions.sort(key=lambda row: int(row.get("version", 0)))
        for event in self.mapping_event_rows:
            object_uids = {
                value
                for key, value in event.items()
                if key.endswith("object_uid") and isinstance(value, str)
            }
            for object_uid in object_uids:
                self.events_by_object[object_uid].append(event)
            frame_uid = event.get("frame_uid")
            if isinstance(frame_uid, str):
                self.events_by_frame[frame_uid].append(event)
        for events in self.events_by_object.values():
            events.sort(key=self.sequence)
        for events in self.events_by_frame.values():
            events.sort(key=self.sequence)
        for row in self.final_membership:
            object_uid = row.get("object_uid")
            if not isinstance(object_uid, str) or object_uid in self.final_by_object:
                raise EvidenceIntegrityError("invalid or duplicate final object_uid")
            self.final_by_object[object_uid] = row

        if validate:
            self.validate()

    @staticmethod
    def sequence(row: dict[str, Any]) -> int:
        value = row.get("event_sequence")
        if value is not None:
            return int(value)
        event_uid = str(row.get("event_uid", ""))
        try:
            return int(event_uid.rsplit("e", 1)[-1])
        except ValueError:
            return -1

    @property
    def max_sequence(self) -> int:
        return max((self.sequence(row) for row in self.events.values()), default=-1)

    def validate(self) -> dict[str, Any]:
        errors: list[str] = []
        for obs_uid, association in self.association_for_obs.items():
            if obs_uid not in self.observations:
                errors.append(f"association references unknown observation {obs_uid}")
            mapping_uid = association.get("mapping_event_uid")
            if mapping_uid not in self.mapping_events:
                errors.append(f"association references unknown mapping event {mapping_uid}")
        for event in self.mapping_event_rows:
            obs_uid = event.get("obs_uid")
            if obs_uid is not None and obs_uid not in self.observations:
                errors.append(f"event references unknown observation {obs_uid}")
            for key in ("input_object_version_uids", "output_object_version_uids"):
                for version_uid in event.get(key) or ():
                    if version_uid not in self.object_versions:
                        errors.append(f"event references unknown version {version_uid}")
        for uid, parents in self.parent_versions.items():
            for parent in parents:
                if parent not in self.object_versions:
                    errors.append(f"version {uid} references unknown parent {parent}")
        for object_uid, final in self.final_by_object.items():
            for obs_uid in final.get("member_observation_uids") or ():
                if obs_uid not in self.observations:
                    errors.append(f"final object {object_uid} references unknown observation {obs_uid}")
        if errors:
            raise EvidenceIntegrityError("; ".join(errors[:20]))
        return {
            "pass": True,
            "observations": len(self.observations),
            "associations": len(self.associations),
            "mapping_events": len(self.mapping_events),
            "object_versions": len(self.object_versions),
            "final_objects": len(self.final_by_object),
        }

    def get_observation(self, obs_uid: str) -> dict[str, Any]:
        return self.observations[obs_uid]

    def get_association_for_obs(self, obs_uid: str) -> dict[str, Any]:
        return self.association_for_obs[obs_uid]

    def get_event(self, event_uid: str) -> dict[str, Any]:
        return self.events[event_uid]

    def get_object_version(self, version_uid: str) -> dict[str, Any]:
        return self.object_versions[version_uid]

    def get_versions_for_object(self, object_uid: str) -> tuple[dict[str, Any], ...]:
        return tuple(self.versions_for_object.get(object_uid, ()))

    def get_parent_versions(self, version_uid: str) -> tuple[str, ...]:
        return self.parent_versions.get(version_uid, ())

    def get_child_versions(self, version_uid: str) -> tuple[str, ...]:
        return tuple(self.child_versions.get(version_uid, ()))

    def get_current_version(self, object_uid: str) -> dict[str, Any] | None:
        versions = self.versions_for_object.get(object_uid, ())
        if not versions:
            return None
        if object_uid in self.final_by_object:
            active = [row for row in versions if row.get("status") == "active"]
            if active:
                return active[-1]
        return versions[-1]

    def get_member_observations(self, version_uid: str) -> tuple[str, ...]:
        return tuple(self.object_versions[version_uid].get("member_observation_uids") or ())

    def get_incident_edge_events(self, object_uid: str) -> tuple[dict[str, Any], ...]:
        result = []
        for event in self.mapping_event_rows:
            if not str(event.get("event_type", "")).startswith("EDGE_"):
                continue
            encoded = json.dumps(event, sort_keys=True)
            if object_uid in encoded:
                result.append(event)
        return tuple(sorted(result, key=self.sequence))

    def events_after(self, sequence: int) -> tuple[dict[str, Any], ...]:
        return tuple(
            sorted(
                (row for row in self.events.values() if self.sequence(row) > sequence),
                key=self.sequence,
            )
        )

    def source_hashes(self) -> dict[str, str]:
        result = {}
        for name in CORE_FILES:
            digest = hashlib.sha256()
            with (self.evidence_root / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[name] = digest.hexdigest()
        return result


class LineageIndex:
    """Version DAG plus revision-layer redirects; the baseline ledger stays read-only."""

    def __init__(self, provenance: ProvenanceIndex) -> None:
        self.provenance = provenance
        self.parents = {uid: set(values) for uid, values in provenance.parent_versions.items()}
        self.children: dict[str, set[str]] = defaultdict(set)
        for uid, parents in self.parents.items():
            for parent in parents:
                self.children[parent].add(uid)
        self.redirects: dict[str, tuple[str, ...]] = {}
        self.revision_events: list[dict[str, Any]] = []

    def resolve_descendants(self, version_uid: str) -> tuple[str, ...]:
        seen: set[str] = set()
        queue = deque([version_uid])
        while queue:
            current = queue.popleft()
            for child in self.children.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        return tuple(sorted(seen))

    def resolve_ancestors(self, version_uid: str) -> tuple[str, ...]:
        seen: set[str] = set()
        queue = deque([version_uid])
        while queue:
            current = queue.popleft()
            for parent in self.parents.get(current, ()):
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        return tuple(sorted(seen))

    def is_descendant(self, candidate: str, ancestor: str) -> bool:
        return candidate == ancestor or ancestor in self.resolve_ancestors(candidate)

    def resolve_current_entities(self, version_uid_or_lineage: str) -> tuple[str, ...]:
        if version_uid_or_lineage in self.redirects:
            return self.redirects[version_uid_or_lineage]
        versions: list[dict[str, Any]] = []
        if version_uid_or_lineage in self.provenance.object_versions:
            seed = {version_uid_or_lineage, *self.resolve_descendants(version_uid_or_lineage)}
            versions = [self.provenance.object_versions[uid] for uid in seed]
        else:
            versions = [
                row
                for row in self.provenance.object_version_rows
                if row.get("lineage_uid") == version_uid_or_lineage
            ]
        current = {
            str(row["object_uid"])
            for row in versions
            if str(row.get("object_uid")) in self.provenance.final_by_object
        }
        return tuple(sorted(current))

    def add_redirect(
        self,
        *,
        source_version_uid: str,
        target_entity_uids: Iterable[str],
        event_type: str,
        tx_id: str,
    ) -> dict[str, Any]:
        event_type = event_type.upper()
        if event_type not in {"LINEAGE_REDIRECT", "LINEAGE_SPLIT"}:
            raise ValueError(f"unsupported revision lineage event: {event_type}")
        targets = tuple(sorted(set(str(item) for item in target_entity_uids)))
        if not targets:
            raise ValueError("lineage redirect requires at least one target")
        event = {
            "event_type": event_type,
            "source_version_uid": source_version_uid,
            "target_entity_uids": list(targets),
            "tx_id": tx_id,
            "branch_id": "derived",
        }
        self.redirects[source_version_uid] = targets
        self.revision_events.append(event)
        return event

    def write_revision_events(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for event in self.revision_events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
