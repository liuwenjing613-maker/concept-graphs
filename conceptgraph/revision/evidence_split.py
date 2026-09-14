"""Frozen, oracle-free proposal/verification evidence separation.

The split is a data-integrity contract.  It does not decide whether a repair is
correct; it only proves that proposal and verification observations/artifacts do
not overlap and that verification evidence is temporally later than the anchor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .auto_constraints import forbidden_inference_paths


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _uid(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return prefix + digest[:20]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidenceReference:
    evidence_uid: str
    obs_uid: str
    frame_index: int
    sha256: str
    source_role: str
    artifact_path: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceReference":
        forbidden = forbidden_inference_paths(value)
        if forbidden:
            raise ValueError("oracle-like evidence fields: " + ", ".join(forbidden))
        obs_uid = str(value.get("obs_uid") or "").strip()
        sha256 = str(value.get("sha256") or "").lower().strip()
        source_role = str(value.get("source_role") or "").strip()
        if not obs_uid or not source_role:
            raise ValueError("obs_uid and source_role are required")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        frame_index = int(value.get("frame_index"))
        evidence_uid = str(value.get("evidence_uid") or "").strip()
        canonical = {
            "obs_uid": obs_uid,
            "frame_index": frame_index,
            "sha256": sha256,
            "source_role": source_role,
        }
        expected_uid = _uid("evidence_ref_", canonical)
        if evidence_uid and evidence_uid != expected_uid:
            raise ValueError("evidence_uid does not match canonical content")
        return cls(
            evidence_uid=expected_uid,
            obs_uid=obs_uid,
            frame_index=frame_index,
            sha256=sha256,
            source_role=source_role,
            artifact_path=(
                str(value["artifact_path"]) if value.get("artifact_path") else None
            ),
        )

    @classmethod
    def build(
        cls,
        *,
        obs_uid: str,
        frame_index: int,
        sha256: str,
        source_role: str,
        artifact_path: str | Path | None = None,
    ) -> "EvidenceReference":
        return cls.from_mapping(
            {
                "obs_uid": obs_uid,
                "frame_index": frame_index,
                "sha256": sha256,
                "source_role": source_role,
                "artifact_path": str(artifact_path) if artifact_path else None,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSplitManifest:
    incident_uid: str
    anchor_obs_uid: str
    anchor_frame: int
    minimum_verification_frame: int
    proposal: tuple[EvidenceReference, ...]
    verification: tuple[EvidenceReference, ...]
    manifest_uid: str

    @property
    def verification_available(self) -> bool:
        return bool(self.verification)

    @property
    def proposal_obs_uids(self) -> tuple[str, ...]:
        return tuple(item.obs_uid for item in self.proposal)

    @property
    def verification_obs_uids(self) -> tuple[str, ...]:
        return tuple(item.obs_uid for item in self.verification)

    @classmethod
    def build(
        cls,
        *,
        incident_uid: str,
        anchor_obs_uid: str,
        anchor_frame: int,
        proposal: Iterable[EvidenceReference | Mapping[str, Any]],
        verification: Iterable[EvidenceReference | Mapping[str, Any]],
        minimum_frame_gap: int = 1,
    ) -> "EvidenceSplitManifest":
        if minimum_frame_gap < 1:
            raise ValueError("minimum_frame_gap must be positive")
        proposal_refs = _canonical_refs(proposal)
        verification_refs = _canonical_refs(verification)
        minimum_verification_frame = int(anchor_frame) + int(minimum_frame_gap)
        early = [
            item.obs_uid
            for item in verification_refs
            if item.frame_index < minimum_verification_frame
        ]
        if early:
            raise ValueError(
                "verification evidence violates temporal embargo: " + ", ".join(early)
            )
        proposal_obs = {item.obs_uid for item in proposal_refs}
        verification_obs = {item.obs_uid for item in verification_refs}
        proposal_hashes = {item.sha256 for item in proposal_refs}
        verification_hashes = {item.sha256 for item in verification_refs}
        overlap_obs = sorted(proposal_obs & verification_obs)
        overlap_hashes = sorted(proposal_hashes & verification_hashes)
        if overlap_obs or overlap_hashes:
            raise ValueError(
                "proposal/verification evidence overlap: "
                f"obs={overlap_obs}, hashes={overlap_hashes}"
            )
        payload = {
            "incident_uid": str(incident_uid),
            "anchor_obs_uid": str(anchor_obs_uid),
            "anchor_frame": int(anchor_frame),
            "minimum_verification_frame": minimum_verification_frame,
            "proposal": [item.as_dict() for item in proposal_refs],
            "verification": [item.as_dict() for item in verification_refs],
        }
        forbidden = forbidden_inference_paths(payload)
        if forbidden:
            raise ValueError("oracle-like split fields: " + ", ".join(forbidden))
        return cls(
            incident_uid=str(incident_uid),
            anchor_obs_uid=str(anchor_obs_uid),
            anchor_frame=int(anchor_frame),
            minimum_verification_frame=minimum_verification_frame,
            proposal=proposal_refs,
            verification=verification_refs,
            manifest_uid=_uid("evidence_split_", payload),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "incident_uid": self.incident_uid,
            "anchor_obs_uid": self.anchor_obs_uid,
            "anchor_frame": self.anchor_frame,
            "minimum_verification_frame": self.minimum_verification_frame,
            "proposal": [item.as_dict() for item in self.proposal],
            "verification": [item.as_dict() for item in self.verification],
            "proposal_verification_obs_intersection": [],
            "proposal_verification_hash_intersection": [],
            "verification_available": self.verification_available,
            "manifest_uid": self.manifest_uid,
        }


def _canonical_refs(
    values: Iterable[EvidenceReference | Mapping[str, Any]],
) -> tuple[EvidenceReference, ...]:
    refs = [
        value
        if isinstance(value, EvidenceReference)
        else EvidenceReference.from_mapping(value)
        for value in values
    ]
    by_obs: dict[str, EvidenceReference] = {}
    for item in refs:
        existing = by_obs.get(item.obs_uid)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting evidence references for {item.obs_uid}")
        by_obs[item.obs_uid] = item
    return tuple(
        sorted(by_obs.values(), key=lambda item: (item.frame_index, item.obs_uid))
    )
