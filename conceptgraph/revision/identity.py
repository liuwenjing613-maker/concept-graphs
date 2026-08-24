"""Formal identity and provenance semantics for revision replay.

The baseline evidence ledger records immutable provenance.  A revision may
change the effective identity used for future routing, but it must never erase
or rewrite where an observation came from.  Keeping those two namespaces
separate prevents both provenance contamination and over-eager merge vetoes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, MutableMapping


PROVENANCE_LINEAGES_KEY = "revision_provenance_lineage_uids"
EFFECTIVE_IDENTITIES_KEY = "revision_identity_uids"
IDENTITY_COMPLETE_KEY = "revision_identity_complete"
LEGACY_LINEAGES_KEY = "revision_lineage_uids"


def _uids(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {str(value) for value in values if value is not None and str(value).strip()}
        )
    )


class IdentityRelation(str, Enum):
    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    UNKNOWN = "UNKNOWN"


class BoundaryDisposition(str, Enum):
    ALLOW = "ALLOW"
    VETO = "VETO"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class IdentityRecord:
    """Two-layer identity attached to one observation or active object."""

    provenance_lineage_uids: tuple[str, ...]
    effective_identity_uids: tuple[str, ...]
    evidence_observation_uids: tuple[str, ...] = ()
    complete: bool = True
    source: str = "runtime"

    @classmethod
    def build(
        cls,
        *,
        provenance_lineage_uids: Iterable[Any] = (),
        effective_identity_uids: Iterable[Any] | None = None,
        evidence_observation_uids: Iterable[Any] = (),
        complete: bool = True,
        source: str = "runtime",
    ) -> "IdentityRecord":
        provenance = _uids(provenance_lineage_uids)
        effective = (
            provenance
            if effective_identity_uids is None
            else _uids(effective_identity_uids)
        )
        return cls(
            provenance_lineage_uids=provenance,
            effective_identity_uids=effective,
            evidence_observation_uids=_uids(evidence_observation_uids),
            complete=bool(complete and provenance and effective),
            source=str(source),
        )

    @property
    def known(self) -> bool:
        return bool(self.complete and self.effective_identity_uids)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["provenance_lineage_uids"] = list(self.provenance_lineage_uids)
        value["effective_identity_uids"] = list(self.effective_identity_uids)
        value["evidence_observation_uids"] = list(self.evidence_observation_uids)
        value["known"] = self.known
        return value


@dataclass(frozen=True)
class IdentityBoundary:
    boundary_uid: str
    left_identity_uids: tuple[str, ...]
    right_identity_uids: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    source: str = "CREATE_INSTANCE"

    @classmethod
    def build(
        cls,
        *,
        left_identity_uids: Iterable[Any],
        right_identity_uids: Iterable[Any],
        evidence_refs: Iterable[Any] = (),
        source: str = "CREATE_INSTANCE",
    ) -> "IdentityBoundary":
        left = _uids(left_identity_uids)
        right = _uids(right_identity_uids)
        if not left or not right:
            raise ValueError("identity boundary requires two known non-empty sides")
        if set(left) & set(right):
            raise ValueError("identity boundary sides must be disjoint")
        payload = {
            "left_identity_uids": left,
            "right_identity_uids": right,
            "evidence_refs": _uids(evidence_refs),
            "source": str(source),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return cls(
            boundary_uid="identity_boundary_" + digest,
            left_identity_uids=left,
            right_identity_uids=right,
            evidence_refs=_uids(evidence_refs),
            source=str(source),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["left_identity_uids"] = list(self.left_identity_uids)
        value["right_identity_uids"] = list(self.right_identity_uids)
        value["evidence_refs"] = list(self.evidence_refs)
        return value


@dataclass(frozen=True)
class BoundaryAssessment:
    disposition: BoundaryDisposition
    identity_relation: IdentityRelation
    crossed_protected_identity_uids: tuple[str, ...] = ()
    crossed_boundary_uids: tuple[str, ...] = ()
    unknown_reasons: tuple[str, ...] = ()

    @property
    def veto(self) -> bool:
        return self.disposition == BoundaryDisposition.VETO

    def as_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "identity_relation": self.identity_relation.value,
            "crossed_protected_identity_uids": list(
                self.crossed_protected_identity_uids
            ),
            "crossed_boundary_uids": list(self.crossed_boundary_uids),
            "unknown_reasons": list(self.unknown_reasons),
        }


def identity_relation(left: IdentityRecord, right: IdentityRecord) -> IdentityRelation:
    if not left.known or not right.known:
        return IdentityRelation.UNKNOWN
    if set(left.effective_identity_uids) & set(right.effective_identity_uids):
        return IdentityRelation.SAME
    return IdentityRelation.DIFFERENT


def assess_protected_boundary(
    left: IdentityRecord,
    right: IdentityRecord,
    protected_identity_uids: Iterable[Any],
) -> BoundaryAssessment:
    """Classify a possible crossing without treating missing evidence as false."""

    protected = _uids(protected_identity_uids)
    relation = identity_relation(left, right)
    unknown = []
    if not left.known:
        unknown.append("left_identity_unknown")
    if not right.known:
        unknown.append("right_identity_unknown")
    if unknown:
        return BoundaryAssessment(
            BoundaryDisposition.UNKNOWN,
            relation,
            unknown_reasons=tuple(unknown),
        )
    left_ids = set(left.effective_identity_uids)
    right_ids = set(right.effective_identity_uids)
    crossed = tuple(
        identity_uid
        for identity_uid in protected
        if (identity_uid in left_ids) != (identity_uid in right_ids)
    )
    if crossed:
        return BoundaryAssessment(
            BoundaryDisposition.VETO,
            relation,
            crossed_protected_identity_uids=crossed,
        )
    return BoundaryAssessment(BoundaryDisposition.ALLOW, relation)


def assess_identity_boundaries(
    left: IdentityRecord,
    right: IdentityRecord,
    boundaries: Iterable[IdentityBoundary],
) -> BoundaryAssessment:
    """Evaluate pair-specific DIFFERENT evidence without global isolation."""

    boundary_values = tuple(boundaries)
    left_ids = set(left.effective_identity_uids)
    right_ids = set(right.effective_identity_uids)
    crossed = []
    for boundary in boundary_values:
        boundary_left = set(boundary.left_identity_uids)
        boundary_right = set(boundary.right_identity_uids)
        if (bool(left_ids & boundary_left) and bool(right_ids & boundary_right)) or (
            bool(left_ids & boundary_right) and bool(right_ids & boundary_left)
        ):
            crossed.append(boundary.boundary_uid)
    relation = identity_relation(left, right)
    if crossed:
        return BoundaryAssessment(
            BoundaryDisposition.VETO,
            IdentityRelation.DIFFERENT,
            crossed_boundary_uids=tuple(sorted(set(crossed))),
        )
    unknown = []
    if not left.effective_identity_uids:
        unknown.append("left_identity_missing")
    if not right.effective_identity_uids:
        unknown.append("right_identity_missing")
    if unknown:
        return BoundaryAssessment(
            BoundaryDisposition.UNKNOWN,
            relation,
            unknown_reasons=tuple(unknown),
        )
    return BoundaryAssessment(BoundaryDisposition.ALLOW, relation)


def record_for_observation(
    obs_uid: str, provenance_lineage_uids: Iterable[Any]
) -> IdentityRecord:
    provenance = _uids(provenance_lineage_uids)
    return IdentityRecord.build(
        provenance_lineage_uids=provenance,
        effective_identity_uids=provenance,
        evidence_observation_uids=(obs_uid,),
        complete=bool(provenance),
        source="immutable_observation_provenance",
    )


def record_for_object(
    obj: Mapping[str, Any],
    observation_lineages: Callable[[str], Iterable[Any]],
) -> IdentityRecord:
    """Read an object's immutable provenance and mutable effective identity."""

    members = _uids(obj.get("obs_uids", ()))
    derived_provenance: set[str] = set()
    unresolved = []
    for obs_uid in members:
        lineages = _uids(observation_lineages(obs_uid))
        if not lineages:
            unresolved.append(obs_uid)
        derived_provenance.update(lineages)
    explicit_provenance = _uids(obj.get(PROVENANCE_LINEAGES_KEY, ()))
    provenance = _uids((*explicit_provenance, *derived_provenance))
    explicit_identity = _uids(obj.get(EFFECTIVE_IDENTITIES_KEY, ()))
    legacy_identity = _uids(obj.get(LEGACY_LINEAGES_KEY, ()))
    effective = explicit_identity or legacy_identity or provenance
    if IDENTITY_COMPLETE_KEY in obj:
        declared_complete = bool(obj.get(IDENTITY_COMPLETE_KEY))
    else:
        declared_complete = not unresolved
    return IdentityRecord.build(
        provenance_lineage_uids=provenance,
        effective_identity_uids=effective,
        evidence_observation_uids=members,
        complete=declared_complete and bool(provenance) and bool(effective),
        source=(
            "explicit_runtime_identity"
            if explicit_identity or legacy_identity
            else "derived_from_immutable_observations"
        ),
    )


def write_identity_record(
    obj: MutableMapping[str, Any], record: IdentityRecord
) -> None:
    obj[PROVENANCE_LINEAGES_KEY] = list(record.provenance_lineage_uids)
    obj[EFFECTIVE_IDENTITIES_KEY] = list(record.effective_identity_uids)
    obj[IDENTITY_COMPLETE_KEY] = bool(record.complete)
    # Keep the V2 key as a compatibility mirror of effective identity only.
    obj[LEGACY_LINEAGES_KEY] = list(record.effective_identity_uids)


def attach_observation_identity(
    obj: MutableMapping[str, Any],
    *,
    obs_uid: str,
    provenance_lineage_uids: Iterable[Any],
    effective_identity_uids: Iterable[Any] | None = None,
) -> IdentityRecord:
    record = IdentityRecord.build(
        provenance_lineage_uids=provenance_lineage_uids,
        effective_identity_uids=effective_identity_uids,
        evidence_observation_uids=(obs_uid,),
        complete=bool(_uids(provenance_lineage_uids)),
        source="materialized_observation",
    )
    write_identity_record(obj, record)
    return record


def merge_identity_metadata(
    target: MutableMapping[str, Any], source: Mapping[str, Any]
) -> IdentityRecord:
    """Union provenance monotonically and union aliases for a legal merge."""

    target_record = IdentityRecord.build(
        provenance_lineage_uids=target.get(PROVENANCE_LINEAGES_KEY, ()),
        effective_identity_uids=(
            target.get(EFFECTIVE_IDENTITIES_KEY) or target.get(LEGACY_LINEAGES_KEY, ())
        ),
        evidence_observation_uids=target.get("obs_uids", ()),
        complete=bool(target.get(IDENTITY_COMPLETE_KEY, True)),
        source="merge_target",
    )
    source_record = IdentityRecord.build(
        provenance_lineage_uids=source.get(PROVENANCE_LINEAGES_KEY, ()),
        effective_identity_uids=(
            source.get(EFFECTIVE_IDENTITIES_KEY) or source.get(LEGACY_LINEAGES_KEY, ())
        ),
        evidence_observation_uids=source.get("obs_uids", ()),
        complete=bool(source.get(IDENTITY_COMPLETE_KEY, True)),
        source="merge_source",
    )
    merged = IdentityRecord.build(
        provenance_lineage_uids=(
            *target_record.provenance_lineage_uids,
            *source_record.provenance_lineage_uids,
        ),
        effective_identity_uids=(
            *target_record.effective_identity_uids,
            *source_record.effective_identity_uids,
        ),
        evidence_observation_uids=(
            *target_record.evidence_observation_uids,
            *source_record.evidence_observation_uids,
        ),
        complete=target_record.complete and source_record.complete,
        source="legal_merge_union",
    )
    write_identity_record(target, merged)
    return merged
