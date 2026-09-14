from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


CORRUPTION_TYPES = {
    "FORCE_CREATE",
    "FORCE_ASSOCIATE",
    "FORCE_POSTPROCESS_MERGE",
}
CONSTRAINT_TYPES = {
    "SAME_INSTANCE",
    "SEPARATE_MEMBER_GROUPS",
    "MOVE_OBSERVATION",
    "DEFER",
}


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    return tuple(_required_text(item, f"{name}[]") for item in value)


@dataclass(frozen=True)
class CorruptionPlan:
    case_uid: str
    frame_idx: int
    obs_uid: str
    corruption_type: str
    source_object_uid: str | None = None
    target_object_uid: str | None = None
    source_origin_obs_uid: str | None = None
    target_origin_obs_uid: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.corruption_type not in CORRUPTION_TYPES:
            raise ValueError(f"unsupported corruption_type: {self.corruption_type}")
        if self.frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        if self.corruption_type == "FORCE_ASSOCIATE" and not self.target_object_uid:
            raise ValueError("FORCE_ASSOCIATE requires target_object_uid")
        if (
            self.corruption_type == "FORCE_POSTPROCESS_MERGE"
            and (not self.source_object_uid or not self.target_object_uid)
        ):
            raise ValueError("FORCE_POSTPROCESS_MERGE requires source and target")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CorruptionPlan":
        if not isinstance(value, Mapping):
            raise ValueError("corruption plan must be an object")
        return cls(
            case_uid=_required_text(value.get("case_uid"), "case_uid"),
            frame_idx=int(value.get("frame_idx")),
            obs_uid=_required_text(value.get("obs_uid"), "obs_uid"),
            corruption_type=_required_text(
                value.get("corruption_type"), "corruption_type"
            ).upper(),
            source_object_uid=value.get("source_object_uid"),
            target_object_uid=value.get("target_object_uid"),
            source_origin_obs_uid=value.get("source_origin_obs_uid"),
            target_origin_obs_uid=value.get("target_origin_obs_uid"),
            seed=int(value.get("seed", 0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairConstraint:
    constraint_type: str
    entities: tuple[str, ...] = ()
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    obs_uid: str | None = None
    from_object_uid: str | None = None
    to_object_uid: str | None = None
    source: str = "oracle"
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.constraint_type not in CONSTRAINT_TYPES:
            raise ValueError(f"unsupported constraint type: {self.constraint_type}")
        if self.constraint_type == "SAME_INSTANCE" and len(self.entities) < 2:
            raise ValueError("SAME_INSTANCE requires at least two entities")
        if self.constraint_type == "SEPARATE_MEMBER_GROUPS":
            if len(self.groups) < 2 or any(not members for members in self.groups.values()):
                raise ValueError("SEPARATE_MEMBER_GROUPS requires two non-empty groups")
        if self.constraint_type == "MOVE_OBSERVATION":
            if not self.obs_uid or not self.from_object_uid or not self.to_object_uid:
                raise ValueError("MOVE_OBSERVATION requires obs_uid, from and to")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RepairConstraint":
        if not isinstance(value, Mapping):
            raise ValueError("repair constraint must be an object")
        kind = _required_text(
            value.get("constraint_type", value.get("type")), "constraint_type"
        ).upper()
        groups_value = value.get("groups") or {}
        if not isinstance(groups_value, Mapping):
            raise ValueError("groups must be an object")
        groups = {
            _required_text(name, "group name"): _text_tuple(members, f"groups.{name}")
            for name, members in groups_value.items()
        }
        return cls(
            constraint_type=kind,
            entities=_text_tuple(value.get("entities"), "entities"),
            groups=groups,
            obs_uid=value.get("obs_uid"),
            from_object_uid=value.get("from_object_uid", value.get("from")),
            to_object_uid=value.get("to_object_uid", value.get("to")),
            source=str(value.get("source", "oracle")),
            evidence_refs=_text_tuple(value.get("evidence_refs"), "evidence_refs"),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["type"] = value.pop("constraint_type")
        value["entities"] = list(self.entities)
        value["groups"] = {name: list(members) for name, members in self.groups.items()}
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class DependencyClosure:
    event_uids: tuple[str, ...]
    version_uids: tuple[str, ...]
    entity_uids: tuple[str, ...]
    obs_uids: tuple[str, ...]
    edge_uids: tuple[str, ...]
    start_sequence: int
    end_sequence: int

    @classmethod
    def build(
        cls,
        *,
        event_uids: Iterable[str] = (),
        version_uids: Iterable[str] = (),
        entity_uids: Iterable[str] = (),
        obs_uids: Iterable[str] = (),
        edge_uids: Iterable[str] = (),
        start_sequence: int,
        end_sequence: int,
    ) -> "DependencyClosure":
        return cls(
            event_uids=tuple(sorted(set(event_uids))),
            version_uids=tuple(sorted(set(version_uids))),
            entity_uids=tuple(sorted(set(entity_uids))),
            obs_uids=tuple(sorted(set(obs_uids))),
            edge_uids=tuple(sorted(set(edge_uids))),
            start_sequence=int(start_sequence),
            end_sequence=int(end_sequence),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RevisionTransaction:
    case_uid: str
    causal_anchor_event_uid: str
    base_event_watermark: int
    base_entity_versions: dict[str, str]
    read_set: tuple[str, ...]
    write_set: tuple[str, ...]
    dependency_closure: DependencyClosure
    repair_constraint: RepairConstraint
    shadow_output_refs: tuple[str, ...] = ()
    verification: dict[str, Any] = field(default_factory=dict)
    commit_status: str = "SHADOW"
    tx_id: str = ""

    def __post_init__(self) -> None:
        if not self.tx_id:
            stable = json.dumps(
                {
                    "case_uid": self.case_uid,
                    "anchor": self.causal_anchor_event_uid,
                    "watermark": self.base_event_watermark,
                    "constraint": self.repair_constraint.as_dict(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            self.tx_id = "tx_" + hashlib.sha256(stable.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dependency_closure"] = self.dependency_closure.as_dict()
        value["repair_constraint"] = self.repair_constraint.as_dict()
        return value


class ConflictType(str, Enum):
    DISJOINT = "DISJOINT"
    APPEND_ONLY_REBASEABLE = "APPEND_ONLY_REBASEABLE"
    LINEAGE_REBASEABLE = "LINEAGE_REBASEABLE"
    HYPOTHESIS_INVALIDATED = "HYPOTHESIS_INVALIDATED"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"


def classify_conflict(
    *,
    repair_entities: Iterable[str],
    changed_entities: Iterable[str],
    append_only_entities: Iterable[str] = (),
    lineage_redirects: Mapping[str, str] | None = None,
    removed_evidence_refs: Iterable[str] = (),
    hypothesis_evidence_refs: Iterable[str] = (),
) -> ConflictType:
    repair = set(repair_entities)
    changed = set(changed_entities)
    if set(removed_evidence_refs) & set(hypothesis_evidence_refs):
        return ConflictType.HYPOTHESIS_INVALIDATED
    if repair.isdisjoint(changed):
        return ConflictType.DISJOINT
    touched = repair & changed
    if touched and touched <= set(append_only_entities):
        return ConflictType.APPEND_ONLY_REBASEABLE
    redirects = dict(lineage_redirects or {})
    if touched and all(entity in redirects for entity in touched):
        return ConflictType.LINEAGE_REBASEABLE
    return ConflictType.UNRESOLVED_CONFLICT


TICKET_STATES = {
    "OPEN",
    "WAIT_EVIDENCE",
    "READY",
    "DIAGNOSING",
    "TRACING",
    "REPLAYING",
    "REBASING",
    "READY_TO_COMMIT",
    "COMMITTED",
    "SUPERSEDED",
    "ABORTED",
    "WAIT_STABILITY",
}
_ALLOWED_TRANSITIONS = {
    "OPEN": {"WAIT_EVIDENCE", "READY", "ABORTED"},
    "WAIT_EVIDENCE": {"READY", "SUPERSEDED", "ABORTED"},
    "READY": {"DIAGNOSING", "TRACING", "ABORTED"},
    "DIAGNOSING": {"TRACING", "WAIT_EVIDENCE", "ABORTED"},
    "TRACING": {"REPLAYING", "ABORTED"},
    "REPLAYING": {"REBASING", "READY_TO_COMMIT", "ABORTED"},
    "REBASING": {"REPLAYING", "WAIT_STABILITY", "ABORTED"},
    "READY_TO_COMMIT": {"COMMITTED", "REBASING", "ABORTED"},
    "WAIT_STABILITY": {"READY", "SUPERSEDED", "ABORTED"},
    "COMMITTED": set(),
    "SUPERSEDED": set(),
    "ABORTED": set(),
}


@dataclass
class RepairTicket:
    ticket_uid: str
    lineage_uid: str
    state: str = "OPEN"
    rebase_count: int = 0
    max_rebase_count: int = 3
    history: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, new_state: str, *, reason: str) -> None:
        new_state = new_state.upper()
        if new_state not in TICKET_STATES:
            raise ValueError(f"unknown ticket state: {new_state}")
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"invalid ticket transition: {self.state} -> {new_state}")
        if new_state == "REBASING":
            self.rebase_count += 1
            if self.rebase_count > self.max_rebase_count:
                new_state = "WAIT_STABILITY"
        self.history.append({"from": self.state, "to": new_state, "reason": reason})
        self.state = new_state

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_maturity(
    member_observation_uids: Iterable[str],
    *,
    action: str,
    bbox_extent: Iterable[float] | None,
    min_unique_frames: int = 3,
    min_members_per_split_group: int = 2,
) -> dict[str, Any]:
    members = tuple(dict.fromkeys(str(item) for item in member_observation_uids))
    frames = {item.rsplit("_r", 1)[0] for item in members}
    extent = tuple(float(item) for item in (bbox_extent or ()))
    finite_geometry = len(extent) == 3 and all(item > 0 for item in extent)
    reasons: list[str] = []
    if len(frames) < min_unique_frames:
        reasons.append("insufficient_unique_frames")
    if action.upper() == "SPLIT" and len(members) < 2 * min_members_per_split_group:
        reasons.append("insufficient_members_for_split")
    if not finite_geometry:
        reasons.append("degenerate_geometry")
    return {
        "eligible": not reasons,
        "state": "MATURE" if not reasons else "TENTATIVE",
        "reasons": reasons,
        "raw_signals": {
            "member_count": len(members),
            "unique_frame_count": len(frames),
            "bbox_extent": list(extent),
            "non_degenerate_geometry": finite_geometry,
        },
    }
