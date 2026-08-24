from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence
from .partition import ObservationPartitionContract


class ReplayMode(str, Enum):
    """Explicit replay semantics; modes must never be inferred from branch names."""

    NATURAL_REPLAY = "NATURAL_REPLAY"
    TEMPORAL_CORRUPTION = "TEMPORAL_CORRUPTION"
    LEGACY_FULL_MEMBERSHIP_ORACLE = "LEGACY_FULL_MEMBERSHIP_ORACLE"
    ANCHOR_ONLY_REPAIR = "ANCHOR_ONLY_REPAIR"
    PERSISTENT_SPARSE_CONSTRAINT_REPLAY = "PERSISTENT_SPARSE_CONSTRAINT_REPLAY"
    FINAL_MEMBER_REFUSION = "FINAL_MEMBER_REFUSION"


class ConstraintType(str, Enum):
    MUST_LINK = "MUST_LINK"
    CANNOT_LINK = "CANNOT_LINK"
    ASSIGN_OBSERVATION = "ASSIGN_OBSERVATION"
    CREATE_INSTANCE = "CREATE_INSTANCE"
    PARTITION_OBSERVATION = "PARTITION_OBSERVATION"
    PARTITION_ENTITY = "PARTITION_ENTITY"
    RELABEL = "RELABEL"
    DEFER = "DEFER"


class ConstraintAction(str, Enum):
    NO_CONSTRAINT = "NO_CONSTRAINT"
    KEEP_NATURAL = "KEEP_NATURAL"
    FORCE_CREATE = "FORCE_CREATE"
    FORCE_TARGET = "FORCE_TARGET"
    DEFER = "DEFER"


class ConstraintConflictError(ValueError):
    """Raised when simultaneously active primitives cannot be satisfied."""


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _text_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{field_name} must be a list")
    return tuple(_required_text(item, f"{field_name}[]") for item in value)


@dataclass(frozen=True)
class SparseRepairConstraint:
    """One auditable primitive consumed by the production replay path.

    Target resolution is lineage-first and may fall back to an immutable origin
    observation or entity UID. None of these fields specifies a final trajectory.
    """

    constraint_type: ConstraintType
    obs_uid: str | None = None
    refs: tuple[str, ...] = ()
    target_lineage_uid: str | None = None
    target_origin_obs_uid: str | None = None
    target_entity_uid: str | None = None
    created_lineage_uid: str | None = None
    created_entity_uid: str | None = None
    created_identity_uid: str | None = None
    separate_from_identity_uids: tuple[str, ...] = ()
    partition_contract: dict[str, Any] | None = None
    entity_uid: str | None = None
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    label: str | None = None
    reason: str | None = None
    applies_at_event_uid: str | None = None
    active_from_sequence: int | None = None
    active_until_sequence: int | None = None
    source: str = "oracle_sparse"
    evidence_refs: tuple[str, ...] = ()
    constraint_uid: str = ""

    def __post_init__(self) -> None:
        kind = ConstraintType(self.constraint_type)
        object.__setattr__(self, "constraint_type", kind)
        if (
            self.active_from_sequence is not None
            and self.active_until_sequence is not None
        ):
            if self.active_from_sequence > self.active_until_sequence:
                raise ValueError("constraint sequence interval is reversed")
        target_available = any(
            (
                self.target_lineage_uid,
                self.target_origin_obs_uid,
                self.target_entity_uid,
            )
        )
        if kind in {
            ConstraintType.MUST_LINK,
            ConstraintType.CANNOT_LINK,
            ConstraintType.ASSIGN_OBSERVATION,
        }:
            if not self.obs_uid:
                raise ValueError(f"{kind.value} requires obs_uid")
            if not target_available:
                raise ValueError(f"{kind.value} requires a target reference")
        if kind == ConstraintType.CREATE_INSTANCE:
            if not self.obs_uid:
                raise ValueError("CREATE_INSTANCE requires obs_uid")
            if target_available:
                raise ValueError("CREATE_INSTANCE must not specify a target reference")
            if (
                self.created_identity_uid
                and self.created_lineage_uid
                and self.created_identity_uid != self.created_lineage_uid
            ):
                raise ValueError(
                    "created_identity_uid conflicts with legacy created_lineage_uid"
                )
            created_identity = self.effective_created_identity_uid
            if created_identity in set(self.separate_from_identity_uids):
                raise ValueError(
                    "CREATE_INSTANCE identity cannot be separate from itself"
                )
        elif self.separate_from_identity_uids:
            raise ValueError("separate_from_identity_uids requires CREATE_INSTANCE")
        if kind == ConstraintType.PARTITION_OBSERVATION:
            if not self.obs_uid:
                raise ValueError("PARTITION_OBSERVATION requires obs_uid")
            if self.partition_contract is None:
                raise ValueError("PARTITION_OBSERVATION requires partition_contract")
            parsed_partition = ObservationPartitionContract.from_mapping(
                self.partition_contract
            )
            if parsed_partition.obs_uid != self.obs_uid:
                raise ValueError(
                    "PARTITION_OBSERVATION obs_uid does not match its contract"
                )
            object.__setattr__(self, "partition_contract", parsed_partition.as_dict())
        elif self.partition_contract is not None:
            raise ValueError("partition_contract requires PARTITION_OBSERVATION")
        if kind == ConstraintType.PARTITION_ENTITY:
            if not self.entity_uid or len(self.groups) < 2:
                raise ValueError(
                    "PARTITION_ENTITY requires an entity and at least two groups"
                )
            if any(not members for members in self.groups.values()):
                raise ValueError("PARTITION_ENTITY groups must be non-empty")
        if kind == ConstraintType.RELABEL and (not self.entity_uid or not self.label):
            raise ValueError("RELABEL requires entity_uid and label")
        if kind == ConstraintType.DEFER and not self.reason:
            raise ValueError("DEFER requires a reason")
        if not self.constraint_uid:
            payload = self.as_dict(include_uid=False)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            uid = "constraint_" + hashlib.sha256(encoded.encode()).hexdigest()[:16]
            object.__setattr__(self, "constraint_uid", uid)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SparseRepairConstraint":
        if not isinstance(value, Mapping):
            raise ValueError("constraint must be an object")
        raw_type = value.get("constraint_type", value.get("type"))
        kind = ConstraintType(_required_text(raw_type, "type").upper())
        groups_value = value.get("groups") or {}
        if not isinstance(groups_value, Mapping):
            raise ValueError("groups must be an object")
        groups = {
            _required_text(name, "group name"): _text_tuple(members, f"groups.{name}")
            for name, members in groups_value.items()
        }
        return cls(
            constraint_type=kind,
            obs_uid=value.get("obs_uid"),
            refs=_text_tuple(value.get("refs"), "refs"),
            target_lineage_uid=value.get("target_lineage_uid"),
            target_origin_obs_uid=value.get("target_origin_obs_uid"),
            target_entity_uid=value.get("target_entity_uid"),
            created_lineage_uid=value.get("created_lineage_uid"),
            created_entity_uid=value.get("created_entity_uid"),
            created_identity_uid=value.get("created_identity_uid"),
            separate_from_identity_uids=_text_tuple(
                value.get("separate_from_identity_uids"), "separate_from_identity_uids"
            ),
            partition_contract=value.get("partition_contract"),
            entity_uid=value.get("entity_uid"),
            groups=groups,
            label=value.get("label"),
            reason=value.get("reason"),
            applies_at_event_uid=value.get("applies_at_event_uid"),
            active_from_sequence=(
                int(value["active_from_sequence"])
                if value.get("active_from_sequence") is not None
                else None
            ),
            active_until_sequence=(
                int(value["active_until_sequence"])
                if value.get("active_until_sequence") is not None
                else None
            ),
            source=str(value.get("source", "oracle_sparse")),
            evidence_refs=_text_tuple(value.get("evidence_refs"), "evidence_refs"),
            constraint_uid=str(value.get("constraint_uid", "")),
        )

    def as_dict(self, *, include_uid: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["type"] = self.constraint_type.value
        value.pop("constraint_type", None)
        value["refs"] = list(self.refs)
        value["groups"] = {name: list(members) for name, members in self.groups.items()}
        value["evidence_refs"] = list(self.evidence_refs)
        value["separate_from_identity_uids"] = list(self.separate_from_identity_uids)
        if self.created_identity_uid is None:
            value.pop("created_identity_uid", None)
        if not self.separate_from_identity_uids:
            value.pop("separate_from_identity_uids", None)
        if self.partition_contract is None:
            value.pop("partition_contract", None)
        if not include_uid:
            value.pop("constraint_uid", None)
        return value

    @property
    def effective_created_identity_uid(self) -> str | None:
        return self.created_identity_uid or self.created_lineage_uid

    def is_active(self, *, obs_uid: str, event_uid: str, event_sequence: int) -> bool:
        if self.obs_uid is not None and self.obs_uid != obs_uid:
            return False
        if (
            self.applies_at_event_uid is not None
            and self.applies_at_event_uid != event_uid
        ):
            return False
        if (
            self.active_from_sequence is not None
            and event_sequence < self.active_from_sequence
        ):
            return False
        if (
            self.active_until_sequence is not None
            and event_sequence > self.active_until_sequence
        ):
            return False
        return True

    def target_key(self) -> tuple[str | None, str | None, str | None]:
        return (
            self.target_lineage_uid,
            self.target_origin_obs_uid,
            self.target_entity_uid,
        )


@dataclass(frozen=True)
class CandidateTarget:
    index: int
    entity_uid: str
    lineage_uids: tuple[str, ...]
    member_obs_uids: tuple[str, ...]
    score: float
    eligible: bool = True
    provenance_lineage_uids: tuple[str, ...] = ()
    identity_complete: bool = True

    @classmethod
    def build(
        cls,
        *,
        index: int,
        entity_uid: str,
        lineage_uids: Iterable[str] = (),
        member_obs_uids: Iterable[str] = (),
        score: float,
        eligible: bool = True,
        provenance_lineage_uids: Iterable[str] = (),
        identity_complete: bool = True,
    ) -> "CandidateTarget":
        return cls(
            index=int(index),
            entity_uid=str(entity_uid),
            lineage_uids=tuple(sorted(set(str(item) for item in lineage_uids))),
            member_obs_uids=tuple(sorted(set(str(item) for item in member_obs_uids))),
            score=float(score),
            eligible=bool(eligible),
            provenance_lineage_uids=tuple(
                sorted(set(str(item) for item in provenance_lineage_uids))
            ),
            identity_complete=bool(identity_complete),
        )


@dataclass(frozen=True)
class ConstraintDecision:
    action: ConstraintAction
    target_index: int | None = None
    constraint_uids: tuple[str, ...] = ()
    forbidden_indices: tuple[int, ...] = ()
    reason: str = ""
    created_lineage_uid: str | None = None
    created_entity_uid: str | None = None
    created_identity_uid: str | None = None
    separate_from_identity_uids: tuple[str, ...] = ()

    @property
    def constrained(self) -> bool:
        return self.action != ConstraintAction.NO_CONSTRAINT

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        value["constraint_uids"] = list(self.constraint_uids)
        value["forbidden_indices"] = list(self.forbidden_indices)
        value["separate_from_identity_uids"] = list(self.separate_from_identity_uids)
        return value


def _target_matches(
    candidate: CandidateTarget, constraint: SparseRepairConstraint
) -> bool:
    if constraint.target_lineage_uid in candidate.lineage_uids:
        return True
    if constraint.target_origin_obs_uid in candidate.member_obs_uids:
        return True
    return bool(
        constraint.target_entity_uid
        and constraint.target_entity_uid == candidate.entity_uid
    )


class ConstraintEngine:
    """Resolve sparse primitives against only the current replay state."""

    def __init__(
        self, constraints: Iterable[SparseRepairConstraint | Mapping[str, Any]]
    ) -> None:
        self.constraints = tuple(
            item
            if isinstance(item, SparseRepairConstraint)
            else SparseRepairConstraint.from_mapping(item)
            for item in constraints
        )
        self._validate_static_conflicts()

    def _validate_static_conflicts(self) -> None:
        by_scope: dict[tuple[str | None, str | None], list[SparseRepairConstraint]] = {}
        for item in self.constraints:
            by_scope.setdefault((item.obs_uid, item.applies_at_event_uid), []).append(
                item
            )
        for scope, values in by_scope.items():
            positive = {
                item.target_key()
                for item in values
                if item.constraint_type
                in {ConstraintType.MUST_LINK, ConstraintType.ASSIGN_OBSERVATION}
            }
            if len(positive) > 1:
                raise ConstraintConflictError(
                    "multiple positive targets for scope "
                    f"{scope}: {sorted(positive, key=lambda item: tuple(value or '' for value in item))}"
                )
            create = [
                item
                for item in values
                if item.constraint_type == ConstraintType.CREATE_INSTANCE
            ]
            partitions = [
                item
                for item in values
                if item.constraint_type == ConstraintType.PARTITION_OBSERVATION
            ]
            if len(partitions) > 1:
                raise ConstraintConflictError(
                    f"multiple observation partitions for scope {scope}"
                )
            if partitions and len(values) > 1:
                raise ConstraintConflictError(
                    f"observation partition cannot share association scope {scope}"
                )
            if create and positive:
                raise ConstraintConflictError(
                    f"scope {scope} requires both an existing target and a new instance"
                )
            create_identities = {
                (
                    item.effective_created_identity_uid,
                    item.created_entity_uid,
                )
                for item in create
            }
            if len(create_identities) > 1:
                raise ConstraintConflictError(
                    f"multiple created identities for scope {scope}: "
                    f"{sorted(create_identities, key=lambda item: tuple(value or '' for value in item))}"
                )
            negative = {
                item.target_key()
                for item in values
                if item.constraint_type == ConstraintType.CANNOT_LINK
            }
            overlap = positive & negative
            if overlap:
                raise ConstraintConflictError(
                    f"target is both required and forbidden for scope {scope}: {overlap}"
                )

    def resolve_for_observation(
        self,
        *,
        obs_uid: str,
        event_uid: str,
        event_sequence: int,
        natural_match: int | None,
        natural_candidates: Sequence[CandidateTarget],
        anchor_only: bool = False,
    ) -> ConstraintDecision:
        active = [
            item
            for item in self.constraints
            if item.is_active(
                obs_uid=obs_uid,
                event_uid=event_uid,
                event_sequence=event_sequence,
            )
        ]
        if anchor_only:
            active = [item for item in active if item.applies_at_event_uid == event_uid]
        if not active:
            return ConstraintDecision(ConstraintAction.NO_CONSTRAINT)
        uids = tuple(sorted(item.constraint_uid for item in active))
        deferred = [
            item for item in active if item.constraint_type == ConstraintType.DEFER
        ]
        if deferred:
            return ConstraintDecision(
                ConstraintAction.DEFER,
                constraint_uids=uids,
                reason="; ".join(item.reason or "deferred" for item in deferred),
            )

        partition = [
            item
            for item in active
            if item.constraint_type == ConstraintType.PARTITION_OBSERVATION
        ]
        if partition:
            return ConstraintDecision(
                ConstraintAction.DEFER,
                constraint_uids=uids,
                reason="partition_observation_requires_pre_association_payload_stage",
            )

        positive = [
            item
            for item in active
            if item.constraint_type
            in {ConstraintType.MUST_LINK, ConstraintType.ASSIGN_OBSERVATION}
        ]
        negative = [
            item
            for item in active
            if item.constraint_type == ConstraintType.CANNOT_LINK
        ]
        forbidden = {
            candidate.index
            for item in negative
            for candidate in natural_candidates
            if _target_matches(candidate, item)
        }

        create = [
            item
            for item in active
            if item.constraint_type == ConstraintType.CREATE_INSTANCE
        ]
        if create:
            exemplar = create[0]
            return ConstraintDecision(
                ConstraintAction.FORCE_CREATE,
                constraint_uids=uids,
                forbidden_indices=tuple(sorted(forbidden)),
                reason="explicit_create_instance_constraint",
                created_lineage_uid=exemplar.created_lineage_uid,
                created_entity_uid=exemplar.created_entity_uid,
                created_identity_uid=exemplar.effective_created_identity_uid,
                separate_from_identity_uids=tuple(
                    sorted(set(exemplar.separate_from_identity_uids))
                ),
            )

        if positive:
            matching = {
                candidate.index
                for item in positive
                for candidate in natural_candidates
                if _target_matches(candidate, item)
            }
            if len(matching) != 1:
                return ConstraintDecision(
                    ConstraintAction.DEFER,
                    constraint_uids=uids,
                    forbidden_indices=tuple(sorted(forbidden)),
                    reason=(
                        "target_not_active"
                        if not matching
                        else "target_reference_is_ambiguous"
                    ),
                )
            target = next(iter(matching))
            if target in forbidden:
                raise ConstraintConflictError("resolved target is also forbidden")
            item = positive[0]
            return ConstraintDecision(
                ConstraintAction.FORCE_TARGET,
                target_index=target,
                constraint_uids=uids,
                forbidden_indices=tuple(sorted(forbidden)),
                reason="explicit_positive_constraint",
                created_lineage_uid=item.created_lineage_uid,
                created_entity_uid=item.created_entity_uid,
            )

        if negative:
            exemplar = negative[0]
            if natural_match in forbidden:
                alternatives = [
                    item.index
                    for item in natural_candidates
                    if item.index not in forbidden and item.eligible
                ]
                if alternatives:
                    return ConstraintDecision(
                        ConstraintAction.FORCE_TARGET,
                        target_index=alternatives[0],
                        constraint_uids=uids,
                        forbidden_indices=tuple(sorted(forbidden)),
                        reason="natural_target_forbidden_use_next_candidate",
                    )
                return ConstraintDecision(
                    ConstraintAction.FORCE_CREATE,
                    constraint_uids=uids,
                    forbidden_indices=tuple(sorted(forbidden)),
                    reason="natural_target_forbidden_no_alternative",
                    created_lineage_uid=exemplar.created_lineage_uid,
                    created_entity_uid=exemplar.created_entity_uid,
                )
            return ConstraintDecision(
                ConstraintAction.KEEP_NATURAL,
                target_index=natural_match,
                constraint_uids=uids,
                forbidden_indices=tuple(sorted(forbidden)),
                reason="forbidden_target_not_selected",
                created_lineage_uid=exemplar.created_lineage_uid,
                created_entity_uid=exemplar.created_entity_uid,
            )

        return ConstraintDecision(
            ConstraintAction.KEEP_NATURAL,
            target_index=natural_match,
            constraint_uids=uids,
            reason="non_association_constraint_not_executable_at_this_boundary",
        )
