from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PARTITION_OBSERVATION_SCHEMA_VERSION = "2.0.0"
_SUPPORTED_SCHEMA_VERSIONS = {"1.0.0", PARTITION_OBSERVATION_SCHEMA_VERSION}
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PART_DISPOSITIONS = {
    "EMIT_OBSERVATION",
    "EXCLUDE_AS_CONTAMINATION",
}
_SOURCE_STAGES = {"STORED_OBSERVATION_PAYLOAD", "PRE_VOXEL_SAMPLED_PAYLOAD"}


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha256_text(value: str, field_name: str) -> str:
    normalized = _required_text(value, field_name).lower()
    if not _HASH_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return normalized


def _canonical_array(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"{field_name} cannot use object dtype")
    if array.ndim == 0:
        raise ValueError(f"{field_name} must have a point dimension")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{field_name} contains non-finite values")
    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        array = array.byteswap().newbyteorder("<")
    elif dtype.byteorder == "=":
        array = array.astype(dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def observation_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash all point-aligned arrays with names, dtypes, shapes, and bytes."""

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("observation payload must contain point-aligned arrays")
    normalized: list[tuple[str, np.ndarray]] = []
    point_count = None
    for name, value in sorted(payload.items(), key=lambda item: str(item[0])):
        field_name = _required_text(str(name), "payload field name")
        array = _canonical_array(value, f"payload.{field_name}")
        if point_count is None:
            point_count = int(array.shape[0])
        elif int(array.shape[0]) != point_count:
            raise ValueError("all observation payload arrays must share point count")
        normalized.append((field_name, array))
    digest = hashlib.sha256()
    digest.update(b"PARTITION_OBSERVATION_PAYLOAD_V1\0")
    for name, array in normalized:
        header = json.dumps(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_partition_assignment(value: Any) -> np.ndarray:
    assignment = np.asarray(value)
    if assignment.ndim != 1:
        raise ValueError("partition assignment must be one-dimensional")
    if not np.issubdtype(assignment.dtype, np.integer):
        raise ValueError("partition assignment must use integer part indices")
    if assignment.size and (
        int(assignment.min()) < 0 or int(assignment.max()) > np.iinfo(np.uint16).max
    ):
        raise ValueError("partition assignment is outside uint16 range")
    return np.ascontiguousarray(assignment, dtype="<u2")


def partition_assignment_sha256(value: Any) -> str:
    assignment = canonical_partition_assignment(value)
    digest = hashlib.sha256()
    digest.update(b"PARTITION_OBSERVATION_ASSIGNMENT_UINT16_V1\0")
    digest.update(int(assignment.size).to_bytes(8, "little"))
    digest.update(assignment.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservationPartitionPart:
    part_index: int
    part_uid: str
    identity_uid: str
    label: str | None = None

    disposition: str = "EMIT_OBSERVATION"

    def __post_init__(self) -> None:
        if int(self.part_index) < 0:
            raise ValueError("part_index must be non-negative")
        object.__setattr__(self, "part_index", int(self.part_index))
        object.__setattr__(self, "part_uid", _required_text(self.part_uid, "part_uid"))
        object.__setattr__(
            self, "identity_uid", _required_text(self.identity_uid, "identity_uid")
        )
        if self.label is not None:
            object.__setattr__(self, "label", _required_text(self.label, "label"))

        disposition = _required_text(
            self.disposition, "partition part disposition"
        ).upper()
        if disposition not in _PART_DISPOSITIONS:
            raise ValueError(f"unsupported partition part disposition: {disposition}")
        object.__setattr__(self, "disposition", disposition)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationPartitionPart":
        if not isinstance(value, Mapping):
            raise ValueError("partition part must be an object")
        return cls(
            part_index=int(value["part_index"]),
            part_uid=value.get("part_uid"),
            identity_uid=value.get("identity_uid"),
            label=value.get("label"),
            disposition=value.get("disposition", "EMIT_OBSERVATION"),
        )


@dataclass(frozen=True)
class ObservationPartitionContract:
    obs_uid: str
    source_point_count: int
    source_payload_sha256: str
    assignment_sha256: str
    parts: tuple[ObservationPartitionPart, ...]
    evidence_refs: tuple[str, ...]
    partition_uid: str = ""
    schema_version: str = PARTITION_OBSERVATION_SCHEMA_VERSION
    source_stage: str = "STORED_OBSERVATION_PAYLOAD"
    assignment_ref: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "obs_uid", _required_text(self.obs_uid, "obs_uid"))
        count = int(self.source_point_count)
        if count < 2:
            raise ValueError("source_point_count must be at least two")
        object.__setattr__(self, "source_point_count", count)
        object.__setattr__(
            self,
            "source_payload_sha256",
            _sha256_text(self.source_payload_sha256, "source_payload_sha256"),
        )
        object.__setattr__(
            self,
            "assignment_sha256",
            _sha256_text(self.assignment_sha256, "assignment_sha256"),
        )
        parts = tuple(
            part
            if isinstance(part, ObservationPartitionPart)
            else ObservationPartitionPart.from_mapping(part)
            for part in self.parts
        )
        if len(parts) < 2:
            raise ValueError("PARTITION_OBSERVATION requires at least two parts")
        indices = [part.part_index for part in parts]
        if sorted(indices) != list(range(len(parts))):
            raise ValueError("part indices must be contiguous from zero")
        if len({part.part_uid for part in parts}) != len(parts):
            raise ValueError("partition part_uids must be unique")
        if len({part.identity_uid for part in parts}) != len(parts):
            raise ValueError("partition identities must be pairwise distinct")
        object.__setattr__(
            self, "parts", tuple(sorted(parts, key=lambda part: part.part_index))
        )
        evidence = tuple(
            sorted(
                {_required_text(item, "evidence_refs[]") for item in self.evidence_refs}
            )
        )
        if not evidence:
            raise ValueError("PARTITION_OBSERVATION requires immutable evidence refs")
        object.__setattr__(self, "evidence_refs", evidence)
        schema_version = _required_text(self.schema_version, "schema_version")
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported partition schema_version: {schema_version}")
        object.__setattr__(self, "schema_version", schema_version)
        source_stage = _required_text(self.source_stage, "source_stage").upper()
        if source_stage not in _SOURCE_STAGES:
            raise ValueError(f"unsupported partition source_stage: {source_stage}")
        object.__setattr__(self, "source_stage", source_stage)
        assignment_ref = None
        if self.assignment_ref is not None:
            if not isinstance(self.assignment_ref, Mapping):
                raise ValueError("assignment_ref must be an object")
            assignment_ref = dict(self.assignment_ref)
            assignment_ref["path"] = _required_text(
                assignment_ref.get("path"), "assignment_ref.path"
            )
            assignment_ref["sha256"] = _sha256_text(
                assignment_ref.get("sha256"), "assignment_ref.sha256"
            )
            assignment_ref["format"] = _required_text(
                assignment_ref.get("format", "npz"), "assignment_ref.format"
            ).lower()
            if assignment_ref["format"] not in {"npy", "npz"}:
                raise ValueError("assignment_ref format must be npy or npz")
            if assignment_ref.get("assignment_sha256") is not None:
                referenced_assignment_hash = _sha256_text(
                    assignment_ref["assignment_sha256"],
                    "assignment_ref.assignment_sha256",
                )
                if referenced_assignment_hash != self.assignment_sha256:
                    raise ValueError("assignment_ref assignment hash mismatch")
                assignment_ref["assignment_sha256"] = referenced_assignment_hash
        if source_stage == "PRE_VOXEL_SAMPLED_PAYLOAD" and assignment_ref is None:
            raise ValueError("pre-voxel partition requires assignment_ref")
        object.__setattr__(self, "assignment_ref", assignment_ref)
        if not self.partition_uid:
            encoded = json.dumps(
                self.as_dict(include_uid=False),
                sort_keys=True,
                separators=(",", ":"),
            )
            object.__setattr__(
                self,
                "partition_uid",
                "partition_" + hashlib.sha256(encoded.encode()).hexdigest()[:16],
            )
        else:
            object.__setattr__(
                self,
                "partition_uid",
                _required_text(self.partition_uid, "partition_uid"),
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationPartitionContract":
        if not isinstance(value, Mapping):
            raise ValueError("partition_contract must be an object")
        raw_parts = value.get("parts")
        if not isinstance(raw_parts, Sequence) or isinstance(raw_parts, (str, bytes)):
            raise ValueError("partition_contract.parts must be a list")
        raw_evidence = value.get("evidence_refs")
        if not isinstance(raw_evidence, Sequence) or isinstance(
            raw_evidence, (str, bytes)
        ):
            raise ValueError("partition_contract.evidence_refs must be a list")
        return cls(
            obs_uid=value.get("obs_uid"),
            source_point_count=int(value.get("source_point_count", 0)),
            source_payload_sha256=value.get("source_payload_sha256"),
            assignment_sha256=value.get("assignment_sha256"),
            parts=tuple(
                ObservationPartitionPart.from_mapping(item) for item in raw_parts
            ),
            evidence_refs=tuple(str(item) for item in raw_evidence),
            partition_uid=str(value.get("partition_uid", "")),
            schema_version=str(
                value.get("schema_version", PARTITION_OBSERVATION_SCHEMA_VERSION)
            ),
            source_stage=value.get("source_stage", "STORED_OBSERVATION_PAYLOAD"),
            assignment_ref=value.get("assignment_ref"),
        )

    def as_dict(self, *, include_uid: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["parts"] = [asdict(part) for part in self.parts]
        value["evidence_refs"] = list(self.evidence_refs)
        if not include_uid:
            value.pop("partition_uid", None)
        return value

    def validate_assignment(self, assignment: Any) -> dict[str, Any]:
        normalized = canonical_partition_assignment(assignment)
        if int(normalized.size) != self.source_point_count:
            raise ValueError("partition assignment point count mismatch")
        if partition_assignment_sha256(normalized) != self.assignment_sha256:
            raise ValueError("partition assignment hash mismatch")
        observed = set(int(item) for item in np.unique(normalized))
        expected = set(range(len(self.parts)))
        if observed != expected:
            missing = sorted(expected - observed)
            unknown = sorted(observed - expected)
            raise ValueError(
                f"partition assignment is not exhaustive over declared parts; "
                f"missing={missing}, unknown={unknown}"
            )
        counts = {
            self.parts[index].part_uid: int(np.sum(normalized == index))
            for index in range(len(self.parts))
        }
        if any(count <= 0 for count in counts.values()):
            raise ValueError("partition parts must all be non-empty")
        return {
            "pass": True,
            "source_point_count": self.source_point_count,
            "assigned_point_count": int(sum(counts.values())),
            "part_point_counts": counts,
            "exhaustive": True,
            "disjoint": True,
            "assignment_hash_exact": True,
        }


@dataclass(frozen=True)
class PartitionedObservation:
    obs_uid: str
    parent_obs_uid: str
    part_uid: str
    identity_uid: str
    disposition: str
    provenance_observation_uids: tuple[str, ...]
    payload: dict[str, np.ndarray]
    point_count: int


@dataclass(frozen=True)
class PartitionExecutionResult:
    contract: ObservationPartitionContract
    parts: tuple[PartitionedObservation, ...]
    validation: dict[str, Any]
    excluded_parts: tuple[PartitionedObservation, ...]


def apply_observation_partition(
    contract: ObservationPartitionContract | Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    assignment: Any,
) -> PartitionExecutionResult:
    """Atomically split every point-aligned field or raise before producing parts."""

    parsed = (
        contract
        if isinstance(contract, ObservationPartitionContract)
        else ObservationPartitionContract.from_mapping(contract)
    )
    payload_hash = observation_payload_sha256(payload)
    if payload_hash != parsed.source_payload_sha256:
        raise ValueError("observation payload hash mismatch")
    normalized_payload = {
        str(name): _canonical_array(value, f"payload.{name}")
        for name, value in payload.items()
    }
    if any(
        int(array.shape[0]) != parsed.source_point_count
        for array in normalized_payload.values()
    ):
        raise ValueError("observation payload point count mismatch")
    normalized_assignment = canonical_partition_assignment(assignment)
    validation = parsed.validate_assignment(normalized_assignment)
    outputs = []
    excluded = []
    for part in parsed.parts:
        mask = normalized_assignment == part.part_index
        part_payload = {
            name: np.ascontiguousarray(array[mask])
            for name, array in normalized_payload.items()
        }
        observation = PartitionedObservation(
            obs_uid=f"{parsed.obs_uid}::partition::{part.part_uid}",
            parent_obs_uid=parsed.obs_uid,
            part_uid=part.part_uid,
            identity_uid=part.identity_uid,
            disposition=part.disposition,
            provenance_observation_uids=(parsed.obs_uid,),
            payload=part_payload,
            point_count=int(np.sum(mask)),
        )
        if part.disposition == "EMIT_OBSERVATION":
            outputs.append(observation)
        else:
            excluded.append(observation)
    return PartitionExecutionResult(
        contract=parsed,
        parts=tuple(outputs),
        validation={
            **validation,
            "emitted_part_count": len(outputs),
            "excluded_part_count": len(excluded),
            "emitted_point_count": int(sum(item.point_count for item in outputs)),
            "excluded_point_count": int(sum(item.point_count for item in excluded)),
            "source_payload_hash_exact": True,
            "atomic": True,
            "identity_complete": True,
            "provenance_preserved": True,
        },
        excluded_parts=tuple(excluded),
    )
