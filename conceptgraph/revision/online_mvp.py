"""Minimal online revision sidecar for the ali-my-new experiment.

The mapper remains the sole writer of the active map.  This module tails its
append-only evidence ledger, commits frames with a one-frame delay, aggregates
weak scanner signals into object-group tickets, and prepares oracle-free VLM
evidence.  Shadow replay is implemented at the bottom of the file so the online
control path stays small and auditable.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STREAM_FILES = (
    "frames.jsonl",
    "observations.jsonl",
    "associations.jsonl",
    "mapping_events.jsonl",
    "object_versions.jsonl",
    "object_pair_decisions.jsonl",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_uid(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]


def frame_index(value: Any) -> int:
    text = str(value or "")
    match = re.search(r"_f(\d+)", text)
    if match:
        return int(match.group(1))
    return -1


def obs_key(obs_uid: str) -> str:
    match = re.search(r"_f(\d+)_r(\d+)$", str(obs_uid))
    return f"f{int(match.group(1)):06d}_r{int(match.group(2)):04d}" if match else str(obs_uid)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normal_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


TIER_LOW = 1
TIER_MEDIUM = 2
TIER_HIGH = 3
TIER_NAMES = {TIER_LOW: "LOW", TIER_MEDIUM: "MEDIUM", TIER_HIGH: "HIGH"}

ROUTING_NO_UPDATE = "NO_UPDATE"
ROUTING_PERSISTENT = "PERSISTENT"
ROUTING_CHANGED_UNSTABLE = "CHANGED_UNSTABLE"
ROUTING_LIKELY_RESOLVED = "LIKELY_RESOLVED"
ROUTING_UNKNOWN = "UNKNOWN"

POOL_MAIN = "MAIN_POOL"
POOL_AUDIT = "AUDIT_POOL"
POOL_UNREVIEWABLE = "UNREVIEWABLE"

SIGNAL_GROUPS = {
    "AMBIGUOUS_ASSOCIATION": "ASSOCIATION",
    "NEAR_THRESHOLD_CREATE": "ASSOCIATION",
    "NEAR_THRESHOLD_ASSOCIATION": "ASSOCIATION",
    "SEMANTIC_ASSOCIATION_CONFLICT": "SEMANTIC",
    "SEMANTIC_DRIFT": "SEMANTIC",
    "GEOMETRY_JUMP": "GEOMETRY",
    "DUPLICATE_PROPOSAL_RISK": "DUPLICATE",
    "POSTPROCESS_MERGE_CONFLICT": "POSTPROCESS",
}


def _clip01(value: Any) -> float:
    number = _finite_float(value)
    return max(0.0, min(1.0, float(number or 0.0)))


def _norm(value: Any, low: float, high: float) -> float:
    number = _finite_float(value)
    if number is None or high <= low:
        return 0.0
    return _clip01((number - low) / (high - low))


def issue_signal_strength(family: str, raw: Mapping[str, Any]) -> float:
    """Auditable V2 within-family strength, never presented as a probability."""

    family = str(family).upper()
    if family == "AMBIGUOUS_ASSOCIATION":
        return _clip01(1.0 - float(raw.get("margin") or 0.0) / 0.12)
    if family == "NEAR_THRESHOLD_CREATE":
        top1 = _finite_float(raw.get("top1_score"))
        threshold = _finite_float(raw.get("sim_threshold"))
        return _clip01(
            1.0 - abs(float(top1 or 0.0) - float(threshold or 0.0)) / 0.12
        )
    if family == "NEAR_THRESHOLD_ASSOCIATION":
        top1 = _finite_float(raw.get("top1_score"))
        threshold = _finite_float(raw.get("sim_threshold"))
        return _clip01(
            1.0 - (float(top1 or 0.0) - float(threshold or 0.0)) / 0.12
        )
    if family == "SEMANTIC_ASSOCIATION_CONFLICT":
        confidence = _norm(raw.get("observation_confidence"), 0.45, 1.0)
        dominant = _norm(raw.get("target_dominant_class_ratio"), 0.70, 1.0)
        return _clip01(math.sqrt(confidence * dominant))
    if family == "GEOMETRY_JUMP":
        ratio = max(1.0, float(_finite_float(raw.get("volume_ratio")) or 1.0))
        return _clip01(math.log(ratio) / math.log(16.0))
    if family == "DUPLICATE_PROPOSAL_RISK":
        return _clip01((float(raw.get("bbox_iou") or 0.0) - 0.90) / 0.10)
    if family == "SEMANTIC_DRIFT":
        previous = max(
            1e-6, float(_finite_float(raw.get("previous_dominant_ratio")) or 0.0)
        )
        current = float(_finite_float(raw.get("current_dominant_ratio")) or 0.0)
        return _clip01(((previous - current) / previous) / 0.5)
    if family == "POSTPROCESS_MERGE_CONFLICT":
        return 0.50
    return 0.0


def _bbox_iou(first: Sequence[Any], second: Sequence[Any]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(dict(value)) + "\n")
        handle.flush()


class JsonlTail:
    """Read only complete, newline-terminated records from one growing file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.position = 0

    def poll(self) -> list[dict[str, Any]]:
        try:
            snapshot_size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if self.position > snapshot_size:
            self.position = 0
        if self.position == snapshot_size:
            return []

        # Freeze the readable boundary for this poll.  Otherwise a fast writer can
        # keep moving EOF and starve the online control loop indefinitely.
        with self.path.open("rb") as handle:
            handle.seek(self.position)
            payload = handle.read(snapshot_size - self.position)
        complete_end = payload.rfind(b"\n")
        if complete_end < 0:
            return []
        complete = payload[: complete_end + 1]
        self.position += len(complete)

        rows: list[dict[str, Any]] = []
        for raw_line in complete.splitlines():
            if not raw_line.strip():
                continue
            value = json.loads(raw_line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"{self.path}: JSONL row is not an object")
            rows.append(value)
        return rows


class LiveEvidenceLedger:
    """Small incremental index over the mapper's real evidence files."""

    def __init__(self, experiment_root: str | Path) -> None:
        self.experiment_root = Path(experiment_root).resolve()
        self.evidence_root = self.experiment_root / "evidence"
        self.tails = {
            name: JsonlTail(self.evidence_root / name) for name in STREAM_FILES
        }
        self.rows: dict[str, list[dict[str, Any]]] = {name: [] for name in STREAM_FILES}
        self.frames: dict[int, dict[str, Any]] = {}
        self.observations: dict[str, dict[str, Any]] = {}
        self.associations: dict[str, dict[str, Any]] = {}
        self.association_for_obs: dict[str, dict[str, Any]] = {}
        self.mapping_events: dict[str, dict[str, Any]] = {}
        self.object_versions: dict[str, dict[str, Any]] = {}
        self.versions_for_object: dict[str, list[dict[str, Any]]] = {}
        self.versions_for_lineage: dict[str, list[dict[str, Any]]] = {}
        self.by_frame: dict[str, dict[int, list[dict[str, Any]]]] = {
            name: {} for name in STREAM_FILES
        }
        self._committed_frames: set[int] = set()

    def poll(self, *, mapping_done: bool = False) -> list[int]:
        for name, tail in self.tails.items():
            for row in tail.poll():
                self.rows[name].append(row)
                index = frame_index(row.get("frame_uid"))
                if index >= 0:
                    self.by_frame[name].setdefault(index, []).append(row)
                if name == "frames.jsonl":
                    self.frames[index] = row
                elif name == "observations.jsonl":
                    self.observations[str(row["obs_uid"])] = row
                elif name == "associations.jsonl":
                    self.associations[str(row["event_uid"])] = row
                    self.association_for_obs[str(row["obs_uid"])] = row
                elif name == "mapping_events.jsonl":
                    self.mapping_events[str(row["event_uid"])] = row
                elif name == "object_versions.jsonl":
                    uid = str(row["object_version_uid"])
                    self.object_versions[uid] = row
                    object_uid = str(row["object_uid"])
                    lineage_uid = str(row.get("lineage_uid") or object_uid)
                    self.versions_for_object.setdefault(object_uid, []).append(row)
                    self.versions_for_lineage.setdefault(lineage_uid, []).append(row)

        seen = sorted(index for index in self.frames if index >= 0)
        eligible = seen if mapping_done else seen[:-1]
        committed = [index for index in eligible if index not in self._committed_frames]
        self._committed_frames.update(committed)
        return committed

    @property
    def max_sequence(self) -> int:
        events = [*self.associations.values(), *self.mapping_events.values()]
        return max((int(row.get("event_sequence", -1)) for row in events), default=-1)

    def max_sequence_at_frame(self, cutoff_frame: int) -> int:
        events = [*self.associations.values(), *self.mapping_events.values()]
        return max(
            (
                int(row.get("event_sequence", -1)) for row in events
                if frame_index(row.get("frame_uid")) <= int(cutoff_frame)
            ),
            default=-1,
        )

    def version_lineage(self, version_uid: Any) -> str | None:
        row = self.object_versions.get(str(version_uid))
        if not row:
            return None
        return str(row.get("lineage_uid") or row.get("object_uid"))

    def object_lineages(self, object_uid: Any) -> set[str]:
        values = self.versions_for_object.get(str(object_uid), ())
        return {
            str(row.get("lineage_uid") or row.get("object_uid")) for row in values
        }

    def latest_version_for_lineage(
        self, lineage_uid: str, *, cutoff_frame: int | None = None
    ) -> dict[str, Any] | None:
        values = self.versions_for_lineage.get(str(lineage_uid), ())
        eligible = [
            row
            for row in values
            if cutoff_frame is None or frame_index(row.get("frame_uid")) <= cutoff_frame
        ]
        return max(eligible, key=lambda row: int(row.get("version", 0))) if eligible else None


class ActiveStateResolver:
    """Resolve decision-time references to the active graph at one watermark.

    The resolver only reads ledger rows at or before ``(cutoff_frame,
    cutoff_sequence)``.  It never opens end-of-run membership artifacts.  Native
    merge redirects are compressed before observation membership is exposed.
    """

    def __init__(
        self,
        ledger: LiveEvidenceLedger,
        *,
        cutoff_frame: int,
        cutoff_sequence: int,
    ) -> None:
        self.ledger = ledger
        self.cutoff_frame = int(cutoff_frame)
        self.cutoff_sequence = int(cutoff_sequence)
        self.latest_by_object: dict[str, dict[str, Any]] = {}
        self.redirects: dict[str, str] = {}
        self.active_versions: dict[str, dict[str, Any]] = {}
        self.observation_owners: dict[str, tuple[str, ...]] = {}
        self._build()

    def _event_sequence(self, event_uid: Any) -> int:
        uid = str(event_uid or "")
        row = self.ledger.mapping_events.get(uid) or self.ledger.associations.get(uid)
        if row is not None:
            return int(row.get("event_sequence", -1))
        match = re.search(r"(?:^|_)e(\d+)(?:_|$)", uid)
        return int(match.group(1)) if match else -1

    def _eligible_version(self, row: Mapping[str, Any]) -> bool:
        frame = frame_index(row.get("frame_uid"))
        sequence = self._event_sequence(row.get("trigger_event_uid"))
        return frame <= self.cutoff_frame and (
            sequence < 0 or sequence <= self.cutoff_sequence
        )

    def _eligible_event(self, row: Mapping[str, Any]) -> bool:
        frame = frame_index(row.get("frame_uid"))
        return frame <= self.cutoff_frame and int(row.get("event_sequence", -1)) <= self.cutoff_sequence

    def _resolve_redirect(self, object_uid: Any) -> str | None:
        current = str(object_uid or "")
        if not current:
            return None
        seen: set[str] = set()
        while current in self.redirects and current not in seen:
            seen.add(current)
            current = self.redirects[current]
        if current in seen:
            return None
        for item in seen:
            self.redirects[item] = current
        return current

    def _build(self) -> None:
        eligible_versions = [
            row for row in self.ledger.object_versions.values() if self._eligible_version(row)
        ]
        for row in eligible_versions:
            object_uid = str(row.get("object_uid") or "")
            previous = self.latest_by_object.get(object_uid)
            rank = (
                self._event_sequence(row.get("trigger_event_uid")),
                int(row.get("version", 0)),
                str(row.get("object_version_uid") or ""),
            )
            previous_rank = (
                self._event_sequence(previous.get("trigger_event_uid")),
                int(previous.get("version", 0)),
                str(previous.get("object_version_uid") or ""),
            ) if previous else (-1, -1, "")
            if rank > previous_rank:
                self.latest_by_object[object_uid] = row

        merge_events = sorted(
            (
                row for row in self.ledger.mapping_events.values()
                if str(row.get("event_type") or "").upper() == "OBJECT_MERGE"
                and self._eligible_event(row)
            ),
            key=lambda row: (int(row.get("event_sequence", -1)), str(row.get("event_uid"))),
        )
        for row in merge_events:
            source = self._resolve_redirect(row.get("source_object_uid"))
            target = self._resolve_redirect(row.get("target_object_uid"))
            if source and target and source != target:
                self.redirects[source] = target

        for object_uid in tuple(self.latest_by_object):
            resolved = self._resolve_redirect(object_uid)
            if not resolved:
                continue
            row = self.latest_by_object.get(resolved)
            if row and str(row.get("status") or "").lower() == "active":
                self.active_versions[resolved] = row

        owners: dict[str, set[str]] = {}
        for owner_uid, row in self.active_versions.items():
            for obs_uid in row.get("member_observation_uids") or ():
                owners.setdefault(str(obs_uid), set()).add(owner_uid)
        self.observation_owners = {
            obs_uid: tuple(sorted(values)) for obs_uid, values in owners.items()
        }

    def active_object_uid(self, object_uid: Any) -> str | None:
        resolved = self._resolve_redirect(object_uid)
        return resolved if resolved in self.active_versions else None

    def active_version_for_object(self, object_uid: Any) -> dict[str, Any] | None:
        resolved = self.active_object_uid(object_uid)
        return self.active_versions.get(resolved or "")

    def owner_for_observation(self, obs_uid: Any) -> str | None:
        owners = self.observation_owners.get(str(obs_uid or ""), ())
        return owners[0] if len(owners) == 1 else None

    def owner_for_unit(self, obs_uids: Iterable[str]) -> str | None:
        owners = {self.owner_for_observation(uid) for uid in obs_uids}
        owners.discard(None)
        return next(iter(owners)) if len(owners) == 1 else None

    def unit_binding_complete(self, obs_uids: Iterable[str]) -> bool:
        values = tuple(str(uid) for uid in obs_uids if uid)
        return bool(values) and all(len(self.observation_owners.get(uid, ())) == 1 for uid in values)

    def has_post_event_update(
        self,
        *,
        event_sequence: int,
        object_uids: Iterable[str] = (),
        observation_uids: Iterable[str] = (),
    ) -> bool:
        owners = {
            owner for owner in (self.active_object_uid(uid) for uid in object_uids) if owner
        }
        owners.update(
            owner for owner in (self.owner_for_observation(uid) for uid in observation_uids) if owner
        )
        return any(
            self._event_sequence(self.active_versions[owner].get("trigger_event_uid"))
            > int(event_sequence)
            for owner in owners
            if owner in self.active_versions
        )

    def snapshot_manifest(self) -> dict[str, Any]:
        active_object_version_uids = {
            owner_uid: str(row.get("object_version_uid") or "")
            for owner_uid, row in sorted(self.active_versions.items())
        }
        identity_payload = {
            "schema_version": "ali_my_h_snapshot/1.0",
            "cutoff_frame": self.cutoff_frame,
            "cutoff_sequence": self.cutoff_sequence,
            "active_object_version_uids": active_object_version_uids,
            "merge_redirects": dict(sorted(self.redirects.items())),
        }
        snapshot_sha256 = hashlib.sha256(
            _canonical_json(identity_payload).encode("utf-8")
        ).hexdigest()
        return {
            **identity_payload,
            "snapshot_uid": "hsnap_" + snapshot_sha256[:16],
            "snapshot_sha256": snapshot_sha256,
            "watermark_source": "ledger_committed",
            "cutoff_frame": self.cutoff_frame,
            "cutoff_sequence": self.cutoff_sequence,
            "active_object_count": len(self.active_versions),
            "bound_observation_count": len(self.observation_owners),
            "ambiguous_observation_count": sum(
                len(owners) != 1 for owners in self.observation_owners.values()
            ),
            "merge_redirect_count": len(self.redirects),
            "final_membership_read": False,
        }


@dataclass(frozen=True)
class TaskContext:
    task_id: str = ""
    required_lineage_uids: frozenset[str] = frozenset()
    required_object_uids: frozenset[str] = frozenset()
    required_relation_uids: frozenset[str] = frozenset()
    active: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TaskContext":
        row = dict(value or {})
        return cls(
            task_id=str(row.get("task_id") or ""),
            required_lineage_uids=frozenset(
                str(item) for item in row.get("required_lineage_uids") or ()
            ),
            required_object_uids=frozenset(
                str(item) for item in row.get("required_object_uids") or ()
            ),
            required_relation_uids=frozenset(
                str(item) for item in row.get("required_relation_uids") or ()
            ),
            active=bool(row.get("active", bool(row.get("task_id")))),
        )

    def blocks(
        self,
        *,
        lineages: Iterable[str],
        objects: Iterable[str],
        relations: Iterable[str] = (),
    ) -> bool:
        if not self.active:
            return False
        return bool(
            self.required_lineage_uids.intersection(lineages)
            or self.required_object_uids.intersection(objects)
            or self.required_relation_uids.intersection(relations)
        )


@dataclass(frozen=True)
class SubIssue:
    issue_uid: str
    family: str
    anchor_event_uid: str
    anchor_obs_uid: str
    detected_frame: int
    detected_sequence: int
    object_uids: tuple[str, ...]
    lineage_uids: tuple[str, ...]
    raw_signals: dict[str, Any]
    evidence_refs: tuple[str, ...] = ()
    signal_group: str = "UNKNOWN"
    strength: float = 0.0
    latest_reconfirmed: bool = False
    independent_support_count: int = 1

    @classmethod
    def build(
        cls,
        *,
        family: str,
        anchor_event_uid: str,
        anchor_obs_uid: str,
        detected_frame: int,
        detected_sequence: int,
        object_uids: Iterable[str],
        lineage_uids: Iterable[str],
        raw_signals: Mapping[str, Any],
        evidence_refs: Iterable[str] = (),
    ) -> "SubIssue":
        payload = {
            "family": str(family),
            "anchor_event_uid": str(anchor_event_uid),
            "anchor_obs_uid": str(anchor_obs_uid),
        }
        return cls(
            issue_uid=stable_uid("issue_", payload),
            family=str(family),
            anchor_event_uid=str(anchor_event_uid),
            anchor_obs_uid=str(anchor_obs_uid),
            detected_frame=int(detected_frame),
            detected_sequence=int(detected_sequence),
            object_uids=tuple(sorted(set(str(item) for item in object_uids if item))),
            lineage_uids=tuple(sorted(set(str(item) for item in lineage_uids if item))),
            raw_signals=dict(raw_signals),
            evidence_refs=tuple(sorted(set(str(item) for item in evidence_refs if item))),
            signal_group=SIGNAL_GROUPS.get(str(family).upper(), "UNKNOWN"),
            strength=issue_signal_strength(str(family), raw_signals),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewContext:
    anchor_obs_uid: str
    primary_core_obs_uids: tuple[str, ...]
    alternative_core_obs_uids: tuple[str, ...]
    event_frame_id: int
    event_sequence: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectTicket:
    ticket_uid: str
    primary_lineage_uids: tuple[str, ...]
    primary_object_uids: tuple[str, ...]
    first_seen_frame: int
    last_seen_frame: int
    issues: list[SubIssue] = field(default_factory=list)
    state: str = "WAIT_EVIDENCE"
    task_blocking: bool = False
    affected_lineage_uids: tuple[str, ...] = ()
    affected_event_count: int = 0
    ranking_watermark: int = -1
    dispatch_frame: int | None = None
    dispatch_sequence: int | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    repair_contract: dict[str, Any] = field(default_factory=dict)
    resolution_state: str = "OPEN_UNCERTAIN"
    resolution_predicate: str = ""
    resolution_reason: str = "not_yet_refreshed"
    resolved_by: str | None = None
    resolved_frame: int | None = None
    has_post_event_update: bool = False
    current_owner_uid: str | None = None
    candidate_current_owner_uids: tuple[str, ...] = ()
    signal_strength: float = 0.0
    signal_groups: tuple[str, ...] = ()
    distinct_signal_frames: int = 0
    latest_reconfirmed: bool = False
    error_tier: int = TIER_LOW
    impact_score: float = 0.0
    impact_tier: int = TIER_LOW
    pool_since_frame: int | None = None
    representative_issue_uid: str | None = None
    review_issue_uid: str | None = None
    review_context: dict[str, Any] = field(default_factory=dict)
    event_snapshot: dict[str, Any] = field(default_factory=dict)
    event_signature: dict[str, Any] = field(default_factory=dict)
    event_update_token: tuple[tuple[str, str, str], ...] = ()
    state_history: list[dict[str, Any]] = field(default_factory=list)
    routing_state: str = ROUTING_UNKNOWN
    routing_reason: str = "not_yet_refreshed"
    routing_destination: str = POOL_MAIN
    pool_location: str = POOL_MAIN
    routing_mode: str = "shadow"
    relevant_update_count: int = 0
    stable_changed_count: int = 0
    routing_events: list[dict[str, Any]] = field(default_factory=list)
    safety_override: bool = False
    output_contract_version: str = "object_state_v2"

    def priority_key(self, current_frame: int) -> tuple[Any, ...]:
        pool_since = self.pool_since_frame if self.pool_since_frame is not None else current_frame
        wait_frames = max(0, int(current_frame) - int(pool_since))
        return (
            -int(self.error_tier),
            -int(self.impact_tier),
            -wait_frames,
            self.ticket_uid,
        )

    def as_dict(self, current_frame: int | None = None) -> dict[str, Any]:
        value = asdict(self)
        value["issues"] = [item.as_dict() for item in self.issues]
        if current_frame is not None:
            value["ticket_age_frames"] = max(0, current_frame - self.first_seen_frame)
            value["pool_wait_frames"] = (
                max(0, current_frame - self.pool_since_frame)
                if self.pool_since_frame is not None
                else 0
            )
            value["priority_tuple"] = {
                "safety_override": self.safety_override,
                "error_tier": TIER_NAMES.get(self.error_tier, str(self.error_tier)),
                "impact_tier": TIER_NAMES.get(self.impact_tier, str(self.impact_tier)),
                "pool_wait_frames": value["pool_wait_frames"],
                "signal_strength": self.signal_strength,
                "impact_score": self.impact_score,
            }
        value["error_tier_name"] = TIER_NAMES.get(self.error_tier, str(self.error_tier))
        value["impact_tier_name"] = TIER_NAMES.get(self.impact_tier, str(self.impact_tier))
        return value


class LiveDependencyTracker:
    """Typed-enough forward closure for scheduling before the frozen index exists."""

    @staticmethod
    def _event_lineages(
        event: Mapping[str, Any], ledger: LiveEvidenceLedger
    ) -> tuple[set[str], set[str]]:
        inputs: set[str] = set()
        outputs: set[str] = set()
        for field in ("input_object_version_uids", "target_object_version_before"):
            value = event.get(field)
            values = value if isinstance(value, (list, tuple)) else (value,)
            inputs.update(
                lineage
                for lineage in (ledger.version_lineage(item) for item in values if item)
                if lineage
            )
        for field in ("output_object_version_uids", "target_object_version_after"):
            value = event.get(field)
            values = value if isinstance(value, (list, tuple)) else (value,)
            outputs.update(
                lineage
                for lineage in (ledger.version_lineage(item) for item in values if item)
                if lineage
            )
        for key, value in event.items():
            if key.endswith("object_uid") and isinstance(value, str):
                inputs.update(ledger.object_lineages(value))
        return inputs, outputs

    def closure(
        self,
        *,
        ledger: LiveEvidenceLedger,
        anchor_event_uid: str,
        anchor_sequence: int,
        seed_lineages: Iterable[str],
        stop_sequence: int,
    ) -> dict[str, Any]:
        affected = set(str(item) for item in seed_lineages if item)
        event_uids: set[str] = set()
        events = sorted(
            [*ledger.associations.values(), *ledger.mapping_events.values()],
            key=lambda row: (int(row.get("event_sequence", -1)), str(row.get("event_uid"))),
        )
        for event in events:
            sequence = int(event.get("event_sequence", -1))
            if not int(anchor_sequence) <= sequence <= int(stop_sequence):
                continue
            inputs, outputs = self._event_lineages(event, ledger)
            event_uid = str(event.get("event_uid"))
            if event_uid == str(anchor_event_uid) or affected.intersection(inputs | outputs):
                event_uids.add(event_uid)
                affected.update(inputs)
                affected.update(outputs)
        return {
            "lineage_uids": tuple(sorted(affected)),
            "event_uids": tuple(sorted(event_uids)),
            "object_count": len(affected),
            "event_count": len(event_uids),
            "start_sequence": int(anchor_sequence),
            "stop_sequence": int(stop_sequence),
        }


class TicketStore:
    def __init__(self) -> None:
        self.tickets: dict[str, ObjectTicket] = {}
        self.issue_uids: set[str] = set()

    @staticmethod
    def _scope(issue: SubIssue) -> tuple[str, ...]:
        if issue.lineage_uids:
            return tuple("L:" + item for item in issue.lineage_uids)
        if issue.object_uids:
            return tuple("O:" + item for item in issue.object_uids)
        return ("OBS:" + issue.anchor_obs_uid,)

    @staticmethod
    def select_review_issue(ticket: ObjectTicket) -> SubIssue:
        return max(
            ticket.issues,
            key=lambda issue: (
                float(issue.strength),
                int(issue.detected_frame),
                issue.issue_uid,
            ),
        )

    @staticmethod
    def _ordered_core(
        obs_uids: Iterable[str],
        ledger: LiveEvidenceLedger,
        *,
        exclude: Iterable[str] = (),
        limit: int = 5,
    ) -> tuple[str, ...]:
        excluded = {str(uid) for uid in exclude}
        unique = {
            str(uid)
            for uid in obs_uids
            if uid and str(uid) not in excluded and str(uid) in ledger.observations
        }
        return tuple(
            sorted(
                unique,
                key=lambda uid: (
                    frame_index(ledger.observations[uid].get("frame_uid") or uid),
                    uid,
                ),
            )[:limit]
        )

    @classmethod
    def build_review_context(
        cls,
        issue: SubIssue,
        contract: Mapping[str, Any],
        ledger: LiveEvidenceLedger,
    ) -> ReviewContext:
        review = [
            str(uid)
            for uid in contract.get("review_unit_obs_uids") or ()
            if str(uid) in ledger.observations
        ]
        anchor = str(issue.anchor_obs_uid or "")
        if anchor not in ledger.observations:
            if not review:
                anchor = ""
            else:
                anchor = max(
                    review,
                    key=lambda uid: (
                        frame_index(ledger.observations[uid].get("frame_uid") or uid),
                        uid,
                    ),
                )
        primary = cls._ordered_core(
            contract.get("event_owner_core_obs_uids") or (),
            ledger,
            exclude=(anchor,),
        )
        alternative: tuple[str, ...] = ()
        candidate_refs = contract.get("candidate_reference_obs_uids") or {}
        for owner_uid in contract.get("candidate_owner_uids") or ():
            refs = candidate_refs.get(str(owner_uid)) or ()
            candidate_core = cls._ordered_core(
                refs,
                ledger,
                exclude=(anchor,),
            )
            if candidate_core:
                alternative = candidate_core
                break
        event = (
            ledger.associations.get(issue.anchor_event_uid)
            or ledger.mapping_events.get(issue.anchor_event_uid)
            or {}
        )
        event_frame = frame_index(event.get("frame_uid"))
        if event_frame < 0:
            event_frame = int(issue.detected_frame)
        return ReviewContext(
            anchor_obs_uid=anchor,
            primary_core_obs_uids=primary,
            alternative_core_obs_uids=alternative,
            event_frame_id=event_frame,
            event_sequence=int(
                contract.get("event_result_sequence", issue.detected_sequence)
            ),
        )

    @staticmethod
    def resolve_group_owner(
        obs_uids: Iterable[str],
        resolver: ActiveStateResolver,
    ) -> str | None:
        core = tuple(str(uid) for uid in obs_uids if uid)
        if not core:
            return None
        mapped = [
            owners[0]
            for uid in core
            for owners in [resolver.observation_owners.get(uid, ())]
            if len(owners) == 1
        ]
        if not mapped:
            return None
        counts = Counter(mapped)
        owner, count = counts.most_common(1)[0]
        required = len(core) // 2 + 1
        if count < required:
            return None
        if sum(value == count for value in counts.values()) > 1:
            return None
        return owner

    @classmethod
    def build_state_snapshot(
        cls,
        context: ReviewContext,
        resolver: ActiveStateResolver,
        *,
        frame_id: int,
        event_signature: Mapping[str, Any] | None = None,
        event_update_token: tuple[tuple[str, str, str], ...] = (),
    ) -> dict[str, Any]:
        anchor_owners = resolver.observation_owners.get(context.anchor_obs_uid, ())
        owners: dict[str, str | None] = {
            "A": anchor_owners[0] if len(anchor_owners) == 1 else None,
        }
        groups = {
            "R0": context.primary_core_obs_uids,
            "R1": context.alternative_core_obs_uids,
        }
        for role, obs_uids in groups.items():
            if obs_uids:
                owners[role] = cls.resolve_group_owner(obs_uids, resolver)

        relations: dict[str, str] = {}
        comparable = bool(context.anchor_obs_uid and owners.get("A"))
        for left, right in (("A", "R0"), ("A", "R1"), ("R0", "R1")):
            if left not in owners or right not in owners:
                continue
            first, second = owners.get(left), owners.get(right)
            key = f"{left}_{right}"
            if not first or not second:
                relations[key] = "UNKNOWN"
                comparable = False
            else:
                relations[key] = "SAME" if first == second else "DIFFERENT"

        signature: dict[str, Any] = {"relations": relations}
        anchor_owner = owners.get("A")
        anchor_version = resolver.active_versions.get(anchor_owner or "")
        anchor_label = _normal_label(
            anchor_version.get("class_name") if anchor_version else None
        )
        if anchor_label:
            signature["anchor_owner_label"] = anchor_label

        required = dict(event_signature or {})
        required_relations = required.get("relations") or {}
        for key in required_relations:
            if relations.get(key) in {None, "UNKNOWN"}:
                comparable = False
        if "anchor_owner_label" in required and not anchor_label:
            comparable = False

        token_parts: list[tuple[str, str, str]] = []
        update_observable = bool(context.anchor_obs_uid)
        for role in ("A", "R0", "R1"):
            if role not in owners:
                continue
            owner = owners.get(role)
            version = resolver.active_versions.get(owner or "")
            if not owner or version is None:
                update_observable = False
                continue
            token_parts.append(
                (role, owner, str(version.get("object_version_uid") or ""))
            )
        update_token = tuple(sorted(token_parts)) if update_observable else ()
        has_relevant_update = bool(
            update_observable
            and event_update_token
            and update_token != event_update_token
        )
        return {
            "frame_id": int(frame_id),
            "owners": owners,
            "signature": signature,
            "update_token": update_token,
            "update_observable": update_observable,
            "has_relevant_update": has_relevant_update,
            "comparable": comparable,
        }

    @staticmethod
    def _classify_routing_state(
        ticket: ObjectTicket,
        snapshot: Mapping[str, Any],
    ) -> tuple[str, str, int]:
        if (
            not ticket.event_snapshot.get("update_observable")
            or not ticket.event_snapshot.get("comparable")
            or not snapshot.get("update_observable")
        ):
            return ROUTING_UNKNOWN, "STATE_OWNER_OR_EVENT_UNOBSERVABLE", 0
        if not snapshot.get("has_relevant_update"):
            return ROUTING_NO_UPDATE, "NO_RELEVANT_OBJECT_VERSION_UPDATE", 0
        if not snapshot.get("comparable"):
            return ROUTING_UNKNOWN, "CURRENT_SIGNATURE_NOT_COMPARABLE", 0
        signature = snapshot.get("signature") or {}
        if signature == ticket.event_signature:
            return ROUTING_PERSISTENT, "UPDATED_BUT_SIGNATURE_UNCHANGED", 0

        run: list[dict[str, Any]] = []
        for item in reversed(ticket.state_history):
            if not item.get("comparable") or item.get("signature") != signature:
                break
            run.append(item)
        stable_count = len(run)
        first_frame = min(
            (int(item.get("frame_id", snapshot.get("frame_id", -1))) for item in run),
            default=int(snapshot.get("frame_id", -1)),
        )
        no_retrigger = int(ticket.last_seen_frame) < first_frame
        if stable_count >= 2 and no_retrigger:
            return (
                ROUTING_LIKELY_RESOLVED,
                "CHANGED_SIGNATURE_STABLE_ACROSS_RELEVANT_UPDATES",
                stable_count,
            )
        reason = (
            "CHANGED_SIGNATURE_RETRIGGERED"
            if stable_count >= 2 and not no_retrigger
            else "CHANGED_SIGNATURE_NOT_STABLE"
        )
        return ROUTING_CHANGED_UNSTABLE, reason, stable_count

    @staticmethod
    def _anchor_evidence_recoverable(
        context: ReviewContext,
        ledger: LiveEvidenceLedger,
    ) -> bool:
        observation = ledger.observations.get(context.anchor_obs_uid)
        if (
            not observation
            or observation.get("status") != "kept"
            or not observation.get("processed_mask_ref")
        ):
            return False
        frame = ledger.frames.get(
            frame_index(observation.get("frame_uid") or context.anchor_obs_uid)
        )
        return bool(frame and (frame.get("rgb_ref") or frame.get("rgb_path")))

    @staticmethod
    def _version_members(
        ledger: LiveEvidenceLedger, version_uid: Any, *, exclude: Iterable[str] = ()
    ) -> list[str]:
        row = ledger.object_versions.get(str(version_uid or "")) or {}
        excluded = set(str(uid) for uid in exclude)
        return [
            str(uid) for uid in row.get("member_observation_uids") or ()
            if str(uid) not in excluded
        ]

    @classmethod
    def _build_repair_contract(
        cls, issue: SubIssue, ledger: LiveEvidenceLedger
    ) -> dict[str, Any]:
        family = issue.family.upper()
        association = ledger.associations.get(issue.anchor_event_uid)
        mapping = ledger.mapping_events.get(issue.anchor_event_uid)
        review = [issue.anchor_obs_uid] if issue.anchor_obs_uid else []
        event_owner = None
        event_core: list[str] = []
        candidate_owners: list[str] = []
        candidate_refs: dict[str, list[str]] = {}
        semantic_target = None
        event_result_sequence = int(issue.detected_sequence)

        if association:
            mapping_event = ledger.mapping_events.get(
                str(association.get("mapping_event_uid") or "")
            )
            if mapping_event:
                event_result_sequence = max(
                    event_result_sequence,
                    int(mapping_event.get("event_sequence", event_result_sequence)),
                )
            event_owner = association.get("target_object_uid")
            before = association.get("target_object_version_before")
            after = association.get("target_object_version_after")
            event_core = cls._version_members(
                ledger, before or after, exclude=review
            )
            object_to_version = {
                str(object_uid): str(version_uid)
                for object_uid, version_uid in zip(
                    association.get("object_uids_before") or (),
                    association.get("candidate_object_version_uids") or (),
                )
            }
            for item in association.get("top_candidates") or ():
                object_uid = str(item.get("object_uid") or "")
                if not object_uid or object_uid == str(event_owner or ""):
                    continue
                if object_uid not in candidate_owners:
                    candidate_owners.append(object_uid)
                    candidate_refs[object_uid] = cls._version_members(
                        ledger, object_to_version.get(object_uid)
                    )
                if len(candidate_owners) >= 2:
                    break
        elif mapping and str(mapping.get("event_type") or "").upper() == "OBJECT_MERGE":
            event_owner = mapping.get("target_object_uid")
            source_before = mapping.get("source_before") or {}
            target_before = mapping.get("target_before") or {}
            review = [str(uid) for uid in source_before.get("member_observation_uids") or ()]
            event_core = [
                str(uid) for uid in target_before.get("member_observation_uids") or ()
            ]
            source_uid = str(mapping.get("source_object_uid") or "")
            if source_uid:
                candidate_owners = [source_uid]
                candidate_refs[source_uid] = list(review)
        else:
            current_version = next(
                (
                    row for row in ledger.object_versions.values()
                    if str(row.get("trigger_event_uid") or "") == issue.anchor_event_uid
                ),
                None,
            )
            if current_version:
                event_owner = current_version.get("object_uid")
                parents = current_version.get("parent_version_uids") or ()
                event_core = cls._version_members(
                    ledger, parents[0] if parents else None, exclude=review
                )

        if family in {"NEAR_THRESHOLD_CREATE", "DUPLICATE_PROPOSAL_RISK"}:
            predicate = "JOIN_CANDIDATE"
        elif family in {"SEMANTIC_ASSOCIATION_CONFLICT", "SEMANTIC_DRIFT"}:
            predicate = "ADOPT_LABEL"
            semantic_target = _normal_label(
                issue.raw_signals.get("observation_class")
                or issue.raw_signals.get("previous_class")
            ) or None
        else:
            predicate = "SEPARATE_FROM_CURRENT"

        return {
            "review_unit_obs_uids": sorted(set(review)),
            "event_owner_uid": str(event_owner or "") or None,
            "event_owner_core_obs_uids": sorted(set(event_core)),
            "candidate_owner_uids": candidate_owners,
            "candidate_reference_obs_uids": candidate_refs,
            "repair_predicate": predicate,
            "semantic_target_label": semantic_target,
            "event_result_sequence": event_result_sequence,
            "contract_built_from_event_uid": issue.anchor_event_uid,
            "executor_support": "SINGLE_OBSERVATION" if len(set(review)) == 1 else "UNSUPPORTED_MULTI_OBSERVATION",
        }

    @staticmethod
    def _event_evidence_complete(
        contract: Mapping[str, Any], ledger: LiveEvidenceLedger
    ) -> bool:
        review = [str(uid) for uid in contract.get("review_unit_obs_uids") or ()]
        if not review:
            return False
        for uid in review:
            row = ledger.observations.get(uid)
            if not row or row.get("status") != "kept" or not row.get("processed_mask_ref"):
                return False
            if frame_index(row.get("frame_uid") or uid) not in ledger.frames:
                return False
        return True

    @staticmethod
    def _label_support(
        *,
        version: Mapping[str, Any],
        target_label: str,
        ledger: LiveEvidenceLedger,
        event_frame: int,
    ) -> tuple[int, float, bool]:
        labels: list[tuple[str, int]] = []
        for uid in version.get("member_observation_uids") or ():
            row = ledger.observations.get(str(uid))
            if not row or row.get("status") != "kept":
                continue
            labels.append(
                (
                    _normal_label(row.get("class_name")),
                    frame_index(row.get("frame_uid") or uid),
                )
            )
        support = sum(label == target_label for label, _ in labels)
        dominant = support / max(1, len(labels))
        post_event = any(
            label == target_label and frame > event_frame for label, frame in labels
        )
        return support, dominant, post_event

    @classmethod
    def _resolution(
        cls,
        *,
        issue: SubIssue,
        contract: Mapping[str, Any],
        resolver: ActiveStateResolver,
        ledger: LiveEvidenceLedger,
    ) -> dict[str, Any]:
        predicate = str(contract.get("repair_predicate") or "")
        review = tuple(str(uid) for uid in contract.get("review_unit_obs_uids") or ())
        core = tuple(str(uid) for uid in contract.get("event_owner_core_obs_uids") or ())
        candidate_refs = contract.get("candidate_reference_obs_uids") or {}
        current_owner = resolver.owner_for_unit(review)
        candidate_owners = tuple(
            sorted(
                {
                    owner
                    for refs in candidate_refs.values()
                    for owner in [resolver.owner_for_unit(refs)]
                    if owner
                }
            )
        )
        relevant_objects = [
            str(contract.get("event_owner_uid") or ""),
            *[str(uid) for uid in contract.get("candidate_owner_uids") or ()],
        ]
        has_post = resolver.has_post_event_update(
            event_sequence=int(
                contract.get("event_result_sequence", issue.detected_sequence)
            ),
            object_uids=relevant_objects,
            observation_uids=review,
        )
        complete = cls._event_evidence_complete(contract, ledger)
        can_evaluate = False
        predicate_true = False
        reason = "predicate_binding_incomplete"
        details: dict[str, Any] = {}

        if predicate == "JOIN_CANDIDATE":
            can_evaluate = resolver.unit_binding_complete(review) and bool(candidate_owners)
            predicate_true = bool(
                can_evaluate and current_owner in set(candidate_owners)
            )
            reason = "review_unit_joined_recorded_candidate" if predicate_true else "review_unit_still_separate_from_recorded_candidates"
        elif predicate == "SEPARATE_FROM_CURRENT":
            core_owner = resolver.owner_for_unit(core)
            can_evaluate = resolver.unit_binding_complete(review) and resolver.unit_binding_complete(core)
            predicate_true = bool(
                can_evaluate and current_owner and core_owner and current_owner != core_owner
            )
            details["event_owner_core_current_owner_uid"] = core_owner
            reason = "review_unit_separated_from_event_owner_core" if predicate_true else "review_unit_still_with_event_owner_core"
        elif predicate == "ADOPT_LABEL":
            target = _normal_label(contract.get("semantic_target_label"))
            version = resolver.active_versions.get(current_owner or "")
            can_evaluate = bool(current_owner and version and target)
            support = 0
            dominant = 0.0
            post_label_support = False
            if can_evaluate and version:
                support, dominant, post_label_support = cls._label_support(
                    version=version,
                    target_label=target,
                    ledger=ledger,
                    event_frame=issue.detected_frame,
                )
                predicate_true = bool(
                    _normal_label(version.get("class_name")) == target
                    and support >= 2
                    and dominant >= 0.70
                    and post_label_support
                )
            details.update(
                {
                    "semantic_target_label": target or None,
                    "semantic_support_count": support,
                    "semantic_dominant_ratio": dominant,
                    "semantic_post_event_support": post_label_support,
                }
            )
            reason = "explicit_target_label_adopted_with_support" if predicate_true else "explicit_target_label_not_yet_adopted"

        if not complete:
            state = "INVALID_EVIDENCE"
            reason = "event_rgb_anchor_or_processed_mask_irrecoverable"
        elif not has_post:
            state = "OPEN_UNCERTAIN"
            reason = "no_post_event_update_use_event_result_as_latest_known_state"
        elif predicate_true:
            state = "AUTO_RESOLVED"
        elif can_evaluate:
            state = "OPEN"
        else:
            state = "OPEN_UNCERTAIN"
        return {
            "resolution_state": state,
            "resolution_predicate": predicate,
            "current_owner_uid": current_owner,
            "candidate_current_owner_uids": candidate_owners,
            "has_post_event_update": has_post,
            "can_evaluate_predicate": can_evaluate,
            "predicate_is_true": predicate_true,
            "resolution_reason": reason,
            **details,
        }

    @staticmethod
    def _p95_cap(values: Iterable[int]) -> int:
        ordered = sorted(max(0, int(value)) for value in values)
        if not ordered:
            return 1
        index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        return max(1, ordered[index])

    def upsert(self, issue: SubIssue) -> ObjectTicket:
        scope = self._scope(issue)
        ticket_uid = stable_uid("ticket_", {"scope": scope})
        ticket = self.tickets.get(ticket_uid)
        if ticket is None:
            ticket = ObjectTicket(
                ticket_uid=ticket_uid,
                primary_lineage_uids=issue.lineage_uids,
                primary_object_uids=issue.object_uids,
                first_seen_frame=issue.detected_frame,
                last_seen_frame=issue.detected_frame,
            )
            self.tickets[ticket_uid] = ticket
        ticket.last_seen_frame = max(ticket.last_seen_frame, issue.detected_frame)
        if issue.issue_uid not in self.issue_uids:
            ticket.issues.append(issue)
            ticket.issues.sort(key=lambda item: (item.detected_sequence, item.issue_uid))
            self.issue_uids.add(issue.issue_uid)
        return ticket

    def refresh(
        self,
        *,
        ledger: LiveEvidenceLedger,
        tracker: LiveDependencyTracker,
        task_context: TaskContext,
        stop_sequence: int,
        cutoff_frame: int | None = None,
        routing_mode: str = "shadow",
    ) -> None:
        if routing_mode not in {"shadow", "active"}:
            raise ValueError(f"unsupported routing mode: {routing_mode}")
        review_frame = (
            max(ledger.frames, default=-1) if cutoff_frame is None else int(cutoff_frame)
        )
        resolver = ActiveStateResolver(
            ledger,
            cutoff_frame=review_frame,
            cutoff_sequence=stop_sequence,
        )
        for ticket in self.tickets.values():
            if not ticket.issues:
                continue
            review_issue = self.select_review_issue(ticket)
            previous_review_uid = ticket.review_issue_uid
            review_changed = previous_review_uid != review_issue.issue_uid
            if review_changed:
                ticket.review_issue_uid = review_issue.issue_uid
                ticket.representative_issue_uid = review_issue.issue_uid
                ticket.repair_contract = self._build_repair_contract(
                    review_issue, ledger
                )
                context = self.build_review_context(
                    review_issue, ticket.repair_contract, ledger
                )
                ticket.review_context = context.as_dict()
                event_resolver = ActiveStateResolver(
                    ledger,
                    cutoff_frame=context.event_frame_id,
                    cutoff_sequence=context.event_sequence,
                )
                event_snapshot = self.build_state_snapshot(
                    context,
                    event_resolver,
                    frame_id=context.event_frame_id,
                )
                ticket.event_snapshot = event_snapshot
                ticket.event_signature = dict(event_snapshot["signature"])
                ticket.event_update_token = tuple(event_snapshot["update_token"])
                ticket.state_history = []
                ticket.relevant_update_count = 0
                ticket.stable_changed_count = 0
                ticket.routing_events.append(
                    {
                        "type": (
                            "REVIEW_ANCHOR_SELECTED"
                            if previous_review_uid is None
                            else "REVIEW_ANCHOR_CHANGED"
                        ),
                        "frame_id": review_frame,
                        "previous_review_issue_uid": previous_review_uid,
                        "review_issue_uid": review_issue.issue_uid,
                        "event_signature": ticket.event_signature,
                    }
                )
            else:
                context = ReviewContext(
                    anchor_obs_uid=str(ticket.review_context["anchor_obs_uid"]),
                    primary_core_obs_uids=tuple(
                        ticket.review_context.get("primary_core_obs_uids") or ()
                    ),
                    alternative_core_obs_uids=tuple(
                        ticket.review_context.get("alternative_core_obs_uids") or ()
                    ),
                    event_frame_id=int(ticket.review_context["event_frame_id"]),
                    event_sequence=int(ticket.review_context["event_sequence"]),
                )
            closure = tracker.closure(
                ledger=ledger,
                anchor_event_uid=review_issue.anchor_event_uid,
                anchor_sequence=review_issue.detected_sequence,
                seed_lineages=ticket.primary_lineage_uids,
                stop_sequence=stop_sequence,
            )
            ticket.affected_lineage_uids = closure["lineage_uids"]
            ticket.affected_event_count = int(closure["event_count"])
            ticket.ranking_watermark = int(stop_sequence)
            ticket.task_blocking = task_context.blocks(
                lineages=ticket.affected_lineage_uids,
                objects=ticket.primary_object_uids,
            )

            current_snapshot = self.build_state_snapshot(
                context,
                resolver=resolver,
                frame_id=review_frame,
                event_signature=ticket.event_signature,
                event_update_token=ticket.event_update_token,
            )
            previous_token = (
                tuple(ticket.state_history[-1].get("update_token") or ())
                if ticket.state_history
                else ()
            )
            token_changed = bool(
                current_snapshot["has_relevant_update"]
                and current_snapshot["update_token"] != previous_token
            )
            if token_changed:
                ticket.state_history.append(dict(current_snapshot))
                ticket.routing_events.append(
                    {
                        "type": "STATE_SNAPSHOT",
                        "frame_id": review_frame,
                        "review_issue_uid": review_issue.issue_uid,
                        "signature": current_snapshot["signature"],
                        "update_token": current_snapshot["update_token"],
                        "comparable": current_snapshot["comparable"],
                    }
                )
            ticket.relevant_update_count = len(ticket.state_history)
            previous_routing = ticket.routing_state
            routing_state, routing_reason, stable_count = (
                self._classify_routing_state(ticket, current_snapshot)
            )
            ticket.routing_state = routing_state
            ticket.routing_reason = routing_reason
            ticket.stable_changed_count = stable_count
            ticket.routing_mode = routing_mode
            ticket.latest_reconfirmed = routing_state == ROUTING_PERSISTENT
            ticket.has_post_event_update = bool(
                current_snapshot["has_relevant_update"]
            )
            ticket.current_owner_uid = current_snapshot["owners"].get("A")
            alternative_owner = current_snapshot["owners"].get("R1")
            ticket.candidate_current_owner_uids = tuple(
                [alternative_owner]
                if alternative_owner
                and alternative_owner != ticket.current_owner_uid
                else ()
            )
            ticket.resolution_predicate = "STATE_SIGNATURE_COMPARISON"
            ticket.resolution_reason = routing_reason
            ticket.resolved_by = None
            ticket.resolved_frame = None

            recoverable = self._anchor_evidence_recoverable(context, ledger)
            if not recoverable:
                ticket.routing_destination = POOL_UNREVIEWABLE
                ticket.pool_location = POOL_UNREVIEWABLE
                ticket.resolution_state = "INVALID_EVIDENCE"
                ticket.pool_since_frame = None
            else:
                ticket.routing_destination = (
                    POOL_AUDIT
                    if routing_state == ROUTING_LIKELY_RESOLVED
                    else POOL_MAIN
                )
                ticket.pool_location = (
                    POOL_MAIN
                    if routing_mode == "shadow"
                    and ticket.routing_destination == POOL_AUDIT
                    else ticket.routing_destination
                )
                ticket.resolution_state = (
                    "OPEN_UNCERTAIN"
                    if routing_state in {ROUTING_NO_UPDATE, ROUTING_UNKNOWN}
                    else "OPEN"
                )
                if ticket.pool_since_frame is None:
                    ticket.pool_since_frame = review_frame

            if review_changed or previous_routing != routing_state or token_changed:
                ticket.routing_events.append(
                    {
                        "type": "ROUTING_DECIDED",
                        "frame_id": review_frame,
                        "review_issue_uid": review_issue.issue_uid,
                        "routing_state": routing_state,
                        "routing_destination": ticket.routing_destination,
                        "pool_location": ticket.pool_location,
                        "reason_code": routing_reason,
                        "event_signature": ticket.event_signature,
                        "current_signature": current_snapshot["signature"],
                        "relevant_update_count": ticket.relevant_update_count,
                        "stable_changed_count": stable_count,
                        "last_issue_frame": ticket.last_seen_frame,
                    }
                )

            ticket.issues = [
                replace(
                    issue,
                    latest_reconfirmed=ticket.latest_reconfirmed,
                )
                for issue in ticket.issues
            ]
            ticket.signal_groups = tuple(
                sorted({issue.signal_group for issue in ticket.issues if issue.signal_group != "UNKNOWN"})
            )
            ticket.distinct_signal_frames = len(
                {issue.detected_frame for issue in ticket.issues}
            )
            ticket.representative_issue_uid = review_issue.issue_uid
            ticket.signal_strength = max(
                float(issue.strength) for issue in ticket.issues
            )
            strong = ticket.signal_strength >= 0.70
            corroborated = (
                len(ticket.signal_groups) >= 2
                or ticket.distinct_signal_frames >= 2
            )
            if ticket.latest_reconfirmed and (strong or corroborated):
                ticket.error_tier = TIER_HIGH
            elif ticket.latest_reconfirmed or strong or corroborated:
                ticket.error_tier = TIER_MEDIUM
            else:
                ticket.error_tier = TIER_LOW

        event_cap = self._p95_cap(
            ticket.affected_event_count for ticket in self.tickets.values()
        )
        lineage_cap = self._p95_cap(
            len(ticket.affected_lineage_uids) for ticket in self.tickets.values()
        )
        for ticket in self.tickets.values():
            event_score = min(
                1.0, math.log1p(ticket.affected_event_count) / math.log1p(event_cap)
            )
            lineage_score = min(
                1.0, math.log1p(len(ticket.affected_lineage_uids)) / math.log1p(lineage_cap)
            )
            ticket.impact_score = float(0.60 * event_score + 0.40 * lineage_score)
            if ticket.impact_score >= 0.67:
                tier = TIER_HIGH
            elif ticket.impact_score >= 0.33:
                tier = TIER_MEDIUM
            else:
                tier = TIER_LOW
            ticket.impact_tier = min(TIER_HIGH, tier + int(ticket.task_blocking))

    def ordered(
        self,
        *,
        current_frame: int,
        states: Iterable[str] = ("WAIT_EVIDENCE", "READY"),
        audit_slot: bool = False,
    ) -> list[ObjectTicket]:
        allowed = set(states)
        values = [
            ticket for ticket in self.tickets.values()
            if ticket.state in allowed
            and ticket.pool_location == (POOL_AUDIT if audit_slot else POOL_MAIN)
            and ticket.pool_since_frame is not None
        ]
        if audit_slot:
            return sorted(
                values,
                key=lambda ticket: (
                    int(ticket.pool_since_frame or current_frame),
                    ticket.ticket_uid,
                ),
            )
        return sorted(values, key=lambda ticket: ticket.priority_key(current_frame))


@dataclass(frozen=True)
class ScannerConfig:
    threshold_band: float = 0.12
    low_margin: float = 0.12
    semantic_drift_ratio: float = 0.58
    geometry_jump_ratio: float = 4.0
    duplicate_iou: float = 0.90
    min_confidence: float = 0.45


class OnlineScanner:
    """Broad weak-signal scanner.  Every result remains a hypothesis."""

    def __init__(self, config: ScannerConfig | None = None) -> None:
        self.config = config or ScannerConfig()
        self._seen_crossings: set[tuple[str, str]] = set()

    @staticmethod
    def _association_scope(
        association: Mapping[str, Any], ledger: LiveEvidenceLedger
    ) -> tuple[set[str], set[str]]:
        objects: set[str] = set()
        lineages: set[str] = set()
        target = association.get("target_object_uid")
        if target:
            objects.add(str(target))
            lineages.update(ledger.object_lineages(target))
        for item in association.get("top_candidates") or ():
            object_uid = item.get("object_uid")
            if object_uid:
                objects.add(str(object_uid))
                lineages.update(ledger.object_lineages(object_uid))
        for field in (
            "target_object_version_before",
            "target_object_version_after",
        ):
            lineage = ledger.version_lineage(association.get(field))
            if lineage:
                lineages.add(lineage)
        return objects, lineages

    def _association_issues(
        self, frame: int, ledger: LiveEvidenceLedger
    ) -> list[SubIssue]:
        issues: list[SubIssue] = []
        for association in ledger.by_frame["associations.jsonl"].get(frame, ()):
            top1 = _finite_float(association.get("top1_score"))
            margin = _finite_float(association.get("margin"))
            threshold = _finite_float(association.get("sim_threshold"))
            decision = str(association.get("decision") or "").upper()
            event_uid = str(association["event_uid"])
            obs_uid = str(association["obs_uid"])
            sequence = int(association.get("event_sequence", -1))
            objects, lineages = self._association_scope(association, ledger)
            raw = {
                "decision": decision,
                "top1_score": top1,
                "top2_score": _finite_float(association.get("top2_score")),
                "margin": margin,
                "sim_threshold": threshold,
            }
            family = None
            if (
                decision == "CREATE_OBJECT"
                and top1 is not None
                and threshold is not None
                and abs(top1 - threshold) <= self.config.threshold_band
            ):
                family = "NEAR_THRESHOLD_CREATE"
            elif (
                decision == "MERGE_TO_OBJECT"
                and margin is not None
                and margin <= self.config.low_margin
            ):
                family = "AMBIGUOUS_ASSOCIATION"
            elif (
                decision == "MERGE_TO_OBJECT"
                and top1 is not None
                and threshold is not None
                and 0 <= top1 - threshold <= self.config.threshold_band
            ):
                family = "NEAR_THRESHOLD_ASSOCIATION"
            if family:
                issues.append(
                    SubIssue.build(
                        family=family,
                        anchor_event_uid=event_uid,
                        anchor_obs_uid=obs_uid,
                        detected_frame=frame,
                        detected_sequence=sequence,
                        object_uids=objects,
                        lineage_uids=lineages,
                        raw_signals=raw,
                        evidence_refs=(event_uid, obs_uid),
                    )
                )

            observation = ledger.observations.get(obs_uid) or {}
            target_version = ledger.object_versions.get(
                str(association.get("target_object_version_before") or "")
            )
            confidence = _finite_float(observation.get("confidence")) or 0.0
            if target_version and confidence >= self.config.min_confidence:
                observed_class = _normal_label(observation.get("class_name"))
                target_class = _normal_label(target_version.get("class_name"))
                target_ratio = _finite_float(target_version.get("dominant_class_ratio"))
                if (
                    observed_class
                    and target_class
                    and observed_class != target_class
                    and (target_ratio or 0.0) >= 0.70
                ):
                    issues.append(
                        SubIssue.build(
                            family="SEMANTIC_ASSOCIATION_CONFLICT",
                            anchor_event_uid=event_uid,
                            anchor_obs_uid=obs_uid,
                            detected_frame=frame,
                            detected_sequence=sequence,
                            object_uids=objects,
                            lineage_uids=lineages,
                            raw_signals={
                                **raw,
                                "observation_class": observed_class,
                                "target_class": target_class,
                                "target_dominant_class_ratio": target_ratio,
                                "observation_confidence": confidence,
                            },
                            evidence_refs=(event_uid, obs_uid),
                        )
                    )
        return issues

    def _version_issues(self, frame: int, ledger: LiveEvidenceLedger) -> list[SubIssue]:
        issues: list[SubIssue] = []
        for current in ledger.by_frame["object_versions.jsonl"].get(frame, ()):
            object_uid = str(current["object_uid"])
            lineage = str(current.get("lineage_uid") or object_uid)
            history = sorted(
                (
                    row for row in ledger.versions_for_object.get(object_uid, ())
                    if frame_index(row.get("frame_uid")) <= frame
                ),
                key=lambda row: (
                    frame_index(row.get("frame_uid")),
                    int(row.get("version", 0)),
                ),
            )
            current_index = next(
                (
                    index for index, row in enumerate(history)
                    if str(row.get("object_version_uid"))
                    == str(current.get("object_version_uid"))
                ),
                -1,
            )
            previous = history[current_index - 1] if current_index > 0 else None
            trigger = str(current.get("trigger_event_uid") or "")
            sequence = int(
                (ledger.mapping_events.get(trigger) or ledger.associations.get(trigger) or {}).get(
                    "event_sequence", -1
                )
            )
            if previous:
                old_volume = _finite_float(previous.get("bbox_volume"))
                new_volume = _finite_float(current.get("bbox_volume"))
                if old_volume and new_volume and min(old_volume, new_volume) > 1e-6:
                    ratio = max(old_volume / new_volume, new_volume / old_volume)
                    crossing = (lineage, "GEOMETRY_JUMP")
                    if ratio >= self.config.geometry_jump_ratio and crossing not in self._seen_crossings:
                        self._seen_crossings.add(crossing)
                        issues.append(
                            SubIssue.build(
                                family="GEOMETRY_JUMP",
                                anchor_event_uid=trigger,
                                anchor_obs_uid=str(current.get("origin_observation_uid") or ""),
                                detected_frame=frame,
                                detected_sequence=sequence,
                                object_uids=(object_uid,),
                                lineage_uids=(lineage,),
                                raw_signals={
                                    "volume_ratio": ratio,
                                    "previous_volume": old_volume,
                                    "current_volume": new_volume,
                                    "operation": current.get("operation"),
                                },
                                evidence_refs=(trigger,),
                            )
                        )
                previous_ratio = _finite_float(previous.get("dominant_class_ratio")) or 1.0
                current_ratio = _finite_float(current.get("dominant_class_ratio")) or 1.0
                crossing = (lineage, "SEMANTIC_DRIFT")
                if (
                    previous_ratio >= 0.70
                    and current_ratio < self.config.semantic_drift_ratio
                    and crossing not in self._seen_crossings
                ):
                    self._seen_crossings.add(crossing)
                    issues.append(
                        SubIssue.build(
                            family="SEMANTIC_DRIFT",
                            anchor_event_uid=trigger,
                            anchor_obs_uid=str(current.get("origin_observation_uid") or ""),
                            detected_frame=frame,
                            detected_sequence=sequence,
                            object_uids=(object_uid,),
                            lineage_uids=(lineage,),
                            raw_signals={
                                "previous_dominant_ratio": previous_ratio,
                                "current_dominant_ratio": current_ratio,
                                "previous_class": previous.get("class_name"),
                                "current_class": current.get("class_name"),
                                "class_histogram": current.get("class_histogram") or {},
                            },
                            evidence_refs=(trigger,),
                        )
                    )
        return issues

    def _merge_issues(self, frame: int, ledger: LiveEvidenceLedger) -> list[SubIssue]:
        issues: list[SubIssue] = []
        for event in ledger.by_frame["mapping_events.jsonl"].get(frame, ()):
            if str(event.get("event_type")) != "OBJECT_MERGE":
                continue
            objects = {
                str(value)
                for key, value in event.items()
                if key.endswith("object_uid") and isinstance(value, str)
            }
            lineages = set().union(*(ledger.object_lineages(item) for item in objects))
            versions = [
                ledger.object_versions.get(str(uid))
                for uid in event.get("input_object_version_uids") or ()
            ]
            classes = {
                _normal_label(row.get("class_name")) for row in versions if row and row.get("class_name")
            }
            if len(classes) < 2:
                continue
            issues.append(
                SubIssue.build(
                    family="POSTPROCESS_MERGE_CONFLICT",
                    anchor_event_uid=str(event["event_uid"]),
                    anchor_obs_uid=str(event.get("obs_uid") or ""),
                    detected_frame=frame,
                    detected_sequence=int(event.get("event_sequence", -1)),
                    object_uids=objects,
                    lineage_uids=lineages,
                    raw_signals={"input_classes": sorted(classes), "event_type": "OBJECT_MERGE"},
                    evidence_refs=(str(event["event_uid"]),),
                )
            )
        return issues

    def _duplicate_issues(self, frame: int, ledger: LiveEvidenceLedger) -> list[SubIssue]:
        kept = [
            row
            for row in ledger.by_frame["observations.jsonl"].get(frame, ())
            if row.get("status") == "kept"
            and (_finite_float(row.get("confidence")) or 0.0) >= self.config.min_confidence
        ]
        candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for index, first in enumerate(kept):
            for second in kept[index + 1 :]:
                if _normal_label(first.get("class_name")) != _normal_label(second.get("class_name")):
                    continue
                iou = _bbox_iou(first.get("bbox_2d") or (), second.get("bbox_2d") or ())
                if iou >= self.config.duplicate_iou:
                    candidates.append((iou, first, second))
        issues: list[SubIssue] = []
        for iou, first, second in sorted(candidates, key=lambda item: item[0], reverse=True)[:3]:
            associations = [
                ledger.association_for_obs.get(str(first["obs_uid"])),
                ledger.association_for_obs.get(str(second["obs_uid"])),
            ]
            associations = [row for row in associations if row]
            if not associations:
                continue
            root = min(associations, key=lambda row: int(row.get("event_sequence", -1)))
            objects: set[str] = set()
            lineages: set[str] = set()
            for association in associations:
                scoped_objects, scoped_lineages = self._association_scope(association, ledger)
                objects.update(scoped_objects)
                lineages.update(scoped_lineages)
            issues.append(
                SubIssue.build(
                    family="DUPLICATE_PROPOSAL_RISK",
                    anchor_event_uid=str(root["event_uid"]),
                    anchor_obs_uid=str(root["obs_uid"]),
                    detected_frame=frame,
                    detected_sequence=int(root.get("event_sequence", -1)),
                    object_uids=objects,
                    lineage_uids=lineages,
                    raw_signals={
                        "bbox_iou": iou,
                        "class_name": first.get("class_name"),
                        "paired_obs_uids": [first["obs_uid"], second["obs_uid"]],
                    },
                    evidence_refs=(str(first["obs_uid"]), str(second["obs_uid"])),
                )
            )
        return issues

    def scan_frame(self, frame: int, ledger: LiveEvidenceLedger) -> list[SubIssue]:
        return [
            *self._association_issues(frame, ledger),
            *self._version_issues(frame, ledger),
            *self._merge_issues(frame, ledger),
            *self._duplicate_issues(frame, ledger),
        ]


@dataclass(frozen=True)
class OnlineEvidencePacket:
    ticket_uid: str
    issue_uid: str
    freeze_frame: int
    freeze_sequence: int
    evidence: Any
    association: dict[str, Any]
    alias_version_uids: dict[str, str]
    allowed_image_ids: tuple[str, ...]
    packet_manifest: dict[str, Any]


class EvidenceRouter:
    ACTIONABLE_FAMILIES = {
        "NEAR_THRESHOLD_CREATE",
        "AMBIGUOUS_ASSOCIATION",
        "NEAR_THRESHOLD_ASSOCIATION",
        "SEMANTIC_ASSOCIATION_CONFLICT",
        "DUPLICATE_PROPOSAL_RISK",
    }

    def __init__(self, experiment_root: str | Path, *, max_images: int = 6) -> None:
        self.experiment_root = Path(experiment_root).resolve()
        self.max_images = int(max_images)

    def _resolve_ref(self, ref: Mapping[str, Any]) -> Path:
        path = Path(str(ref["path"]))
        return path.resolve() if path.is_absolute() else (self.experiment_root / path).resolve()

    def _load_crop(self, row: Mapping[str, Any]):
        from PIL import Image

        ref = row.get("crop_ref") or {}
        path = self._resolve_ref(ref)
        if path.name.endswith(".pkl.gz"):
            with gzip.open(path, "rb") as handle:
                value = pickle.load(handle)
            if ref.get("index") is not None:
                value = value[int(ref["index"])]
        else:
            value = Image.open(path)
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        array = np.asarray(value)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    def _load_array(self, ref: Mapping[str, Any]) -> np.ndarray:
        path = self._resolve_ref(ref)
        key = ref.get("key")
        with np.load(path, allow_pickle=False) as value:
            if key:
                return np.asarray(value[str(key)])
            keys = list(value.keys())
            if len(keys) != 1:
                raise ValueError(f"ambiguous npz reference: {path}")
            return np.asarray(value[keys[0]])

    def _overlay(self, row: Mapping[str, Any], frame_row: Mapping[str, Any]):
        from PIL import Image

        rgb_path = Path(str(frame_row.get("rgb_path") or ""))
        if not rgb_path.is_absolute():
            rgb_path = (self.experiment_root / rgb_path).resolve()
        image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8).copy()
        ref = row.get("processed_mask_ref") or row.get("mask_ref")
        if ref:
            mask = np.asarray(self._load_array(ref), dtype=bool)
            if mask.shape == image.shape[:2]:
                tint = np.zeros_like(image)
                tint[..., 0] = 255
                image[mask] = (0.60 * image[mask] + 0.40 * tint[mask]).astype(np.uint8)
        return Image.fromarray(image)

    @staticmethod
    def _clarity(row: Mapping[str, Any]) -> float:
        confidence = _finite_float(row.get("confidence")) or 0.0
        bbox = row.get("bbox_2d") or ()
        area = 0.0
        clipped = 0.0
        if len(bbox) == 4:
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                0.0, float(bbox[3]) - float(bbox[1])
            )
            clipped = sum(float(value) <= 1.0 for value in bbox[:2]) * 0.15
        return confidence + min(0.5, math.log1p(area) / 30.0) - clipped

    def _representatives(
        self, version: Mapping[str, Any], ledger: LiveEvidenceLedger, limit: int = 2
    ) -> list[str]:
        rows = [
            ledger.observations[uid]
            for uid in version.get("member_observation_uids") or ()
            if uid in ledger.observations and ledger.observations[uid].get("status") == "kept"
        ]
        if not rows:
            return []
        clearest = max(rows, key=self._clarity)
        selected = [str(clearest["obs_uid"])]
        if limit > 1:
            farthest = max(
                rows,
                key=lambda row: (
                    abs(frame_index(row.get("frame_uid")) - frame_index(clearest.get("frame_uid"))),
                    self._clarity(row),
                ),
            )
            if str(farthest["obs_uid"]) not in selected:
                selected.append(str(farthest["obs_uid"]))
        return selected[:limit]

    def _actionable_issue(self, ticket: ObjectTicket, ledger: LiveEvidenceLedger) -> SubIssue | None:
        return next(
            (
                issue
                for issue in ticket.issues
                if issue.family in self.ACTIONABLE_FAMILIES
                and issue.anchor_event_uid in ledger.associations
                and issue.anchor_obs_uid in ledger.observations
                and self._replay_frame_evidence_valid(issue, ledger)
            ),
            None,
        )

    def _replay_frame_evidence_valid(
        self, issue: SubIssue, ledger: LiveEvidenceLedger
    ) -> bool:
        """Fail closed when a native frame matrix cannot address its ledger rows."""

        association = ledger.associations.get(issue.anchor_event_uid)
        if not association:
            return False
        index = frame_index(association.get("frame_uid"))
        matrices: dict[tuple[str, str], tuple[int, int]] = {}
        try:
            for row in ledger.by_frame["associations.jsonl"].get(index, ()):
                observation = ledger.observations.get(str(row.get("obs_uid")))
                if not observation or observation.get("status") != "kept":
                    continue
                ref = row.get("aggregate_sim_ref") or {}
                path = self._resolve_ref(ref)
                key = str(ref.get("key") or "aggregate_sim")
                cache_key = (str(path), key)
                if cache_key not in matrices:
                    with np.load(path, allow_pickle=False) as payload:
                        matrix = np.asarray(payload[key])
                    if matrix.ndim != 2:
                        return False
                    matrices[cache_key] = (int(matrix.shape[0]), int(matrix.shape[1]))
                rows, columns = matrices[cache_key]
                detection_index = int(observation.get("filtered_det_idx", -1))
                if not 0 <= detection_index < rows:
                    return False
                if len(row.get("object_uids_before") or ()) != columns:
                    return False
                if len(row.get("candidate_object_version_uids") or ()) != columns:
                    return False
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return False
        return True

    def build_v2(
        self,
        *,
        ticket: ObjectTicket,
        ledger: LiveEvidenceLedger,
        freeze_frame: int,
        freeze_sequence: int,
        output_dir: str | Path,
    ) -> OnlineEvidencePacket | None:
        """Bind one routed V2 ticket to active E aliases without rendering images."""

        if ticket.pool_location not in {POOL_MAIN, POOL_AUDIT}:
            return None
        issue = next(
            (
                item for item in ticket.issues
                if item.issue_uid == ticket.review_issue_uid
            ),
            None,
        )
        if issue is None:
            return None
        event = ledger.associations.get(issue.anchor_event_uid)
        mapping = ledger.mapping_events.get(issue.anchor_event_uid)
        if event is None and mapping is None:
            return None
        contract = dict(ticket.repair_contract)
        review = tuple(str(uid) for uid in contract.get("review_unit_obs_uids") or ())
        if not review or not self._event_evidence_complete(contract, ledger):
            return None

        resolver = ActiveStateResolver(
            ledger,
            cutoff_frame=freeze_frame,
            cutoff_sequence=freeze_sequence,
        )
        current_owner = resolver.owner_for_unit(review)
        primary_owner = resolver.active_object_uid(contract.get("event_owner_uid"))
        if primary_owner is None:
            primary_core = tuple(
                str(uid)
                for uid in ticket.review_context.get("primary_core_obs_uids") or ()
            )
            primary_owner = TicketStore.resolve_group_owner(primary_core, resolver)
        if primary_owner is None:
            primary_owner = current_owner
        primary_version = resolver.active_versions.get(primary_owner or "")
        if primary_version is None:
            return None
        alias_versions = {"E0": str(primary_version["object_version_uid"])}
        alias_owner_uids = {"E0": str(primary_owner)}
        candidate_alias_observation_uids: dict[str, list[str]] = {}
        seen_owners = {str(primary_owner)}
        candidate_refs = contract.get("candidate_reference_obs_uids") or {}
        ordered_candidates: list[tuple[str, Iterable[str]]] = []
        if current_owner and str(current_owner) != str(primary_owner):
            ordered_candidates.append((str(current_owner), review))
        for candidate_uid in contract.get("candidate_owner_uids") or ():
            refs = candidate_refs.get(str(candidate_uid)) or ()
            ordered_candidates.append((str(candidate_uid), refs))
        for candidate_uid, refs in ordered_candidates:
            owner = TicketStore.resolve_group_owner(refs, resolver)
            if owner is None:
                owner = resolver.active_object_uid(candidate_uid)
            version = resolver.active_versions.get(owner or "")
            if version is None or str(owner) in seen_owners:
                continue
            alias_versions["E1"] = str(version["object_version_uid"])
            alias_owner_uids["E1"] = str(owner)
            candidate_alias_observation_uids["E1"] = [str(uid) for uid in refs]
            seen_owners.add(str(owner))
            break

        if (
            alias_owner_uids.get("E1")
            and alias_owner_uids["E1"] == alias_owner_uids["E0"]
        ):
            return None

        if current_owner == primary_owner:
            current_assignment = "E0"
        elif current_owner and current_owner == alias_owner_uids.get("E1"):
            current_assignment = "E1"
        else:
            current_assignment = "UNKNOWN"
        event_result_sequence = int(
            contract.get("event_result_sequence", issue.detected_sequence)
        )
        newer_state_available = bool(
            resolver._event_sequence(primary_version.get("trigger_event_uid"))
            > event_result_sequence
        )

        association = dict(event or {})
        if not association and mapping:
            association = {
                "event_uid": mapping.get("event_uid"),
                "event_sequence": mapping.get("event_sequence"),
                "frame_uid": mapping.get("frame_uid"),
                "obs_uid": review[0],
                "decision": "POSTPROCESS_MERGE",
                "target_object_uid": mapping.get("target_object_uid"),
            }
        source_event = event or mapping or {}
        s_frame = frame_index(source_event.get("frame_uid"))
        if s_frame < 0:
            s_frame = int(ticket.review_context.get("event_frame_id", -1))
        s_sequence = int(
            source_event.get(
                "event_sequence", ticket.review_context.get("event_sequence", -1)
            )
        )
        d_frame = int(issue.detected_frame)
        d_sequence = int(issue.detected_sequence)
        h_frame = int(freeze_frame)
        h_sequence = int(freeze_sequence)
        if s_frame < 0 or not (s_frame <= d_frame <= h_frame):
            return None
        h_snapshot = resolver.snapshot_manifest()
        active_at_h = h_snapshot["active_object_version_uids"]
        if any(
            active_at_h.get(owner_uid) != alias_versions.get(alias)
            for alias, owner_uid in alias_owner_uids.items()
        ):
            return None
        timeline = {
            "schema_version": "ali_my_online_timeline/1.0",
            "s_frame": s_frame,
            "s_sequence": s_sequence,
            "s_event_uid": issue.anchor_event_uid,
            "d_frame": d_frame,
            "d_sequence": d_sequence,
            "d_issue_uid": issue.issue_uid,
            "h_frame": h_frame,
            "h_sequence": h_sequence,
            "h_snapshot_uid": h_snapshot["snapshot_uid"],
            "h_snapshot_sha256": h_snapshot["snapshot_sha256"],
            "h_latest_main_map_frame": h_frame,
            "c_frame": None,
            "c_sequence": None,
            "c_latest_main_map_frame": None,
            "watermark_source": "ledger_committed",
            "frame_order_valid_through_h": s_frame <= d_frame <= h_frame,
        }
        packet_manifest = {
            "schema_version": "ali_my_object_state_packet/2.1",
            "output_contract_version": "object_state_v2",
            "ticket_uid": ticket.ticket_uid,
            "issue_uid": issue.issue_uid,
            "review_issue_uid": issue.issue_uid,
            "issue": issue.as_dict(),
            "freeze_frame": int(freeze_frame),
            "freeze_sequence": int(freeze_sequence),
            "anchor_event_uid": issue.anchor_event_uid,
            "review_unit_obs_uids": list(review),
            "repair_contract": contract,
            "resolution": {
                "state": ticket.resolution_state,
                "predicate": ticket.resolution_predicate,
                "reason": ticket.resolution_reason,
                "has_post_event_update": ticket.has_post_event_update,
                "current_owner_uid": ticket.current_owner_uid,
                "candidate_current_owner_uids": list(ticket.candidate_current_owner_uids),
            },
            "routing": {
                "mode": ticket.routing_mode,
                "state": ticket.routing_state,
                "reason": ticket.routing_reason,
                "destination": ticket.routing_destination,
                "pool_location": ticket.pool_location,
                "event_signature": ticket.event_signature,
                "current_signature": (
                    ticket.state_history[-1].get("signature")
                    if ticket.state_history
                    else ticket.event_signature
                ),
                "relevant_update_count": ticket.relevant_update_count,
                "stable_changed_count": ticket.stable_changed_count,
            },
            "ranking": {
                "error_tier": TIER_NAMES.get(ticket.error_tier),
                "impact_tier": TIER_NAMES.get(ticket.impact_tier),
                "signal_strength": ticket.signal_strength,
                "impact_score": ticket.impact_score,
                "pool_since_frame": ticket.pool_since_frame,
            },
            "alias_version_uids": alias_versions,
            "alias_owner_uids": alias_owner_uids,
            "candidate_alias_observation_uids": candidate_alias_observation_uids,
            "current_assignment": current_assignment,
            "newer_state_available": newer_state_available,
            "timeline": timeline,
            "h_snapshot": h_snapshot,
            "active_snapshot": h_snapshot,
            "oracle_fields_included": False,
            "end_of_run_membership_read": False,
        }
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "packet_manifest.json", packet_manifest)
        return OnlineEvidencePacket(
            ticket_uid=ticket.ticket_uid,
            issue_uid=issue.issue_uid,
            freeze_frame=int(freeze_frame),
            freeze_sequence=int(freeze_sequence),
            evidence=None,
            association=association,
            alias_version_uids=alias_versions,
            allowed_image_ids=("I1", "I2", "I3"),
            packet_manifest=packet_manifest,
        )

    @staticmethod
    def _event_evidence_complete(
        contract: Mapping[str, Any], ledger: LiveEvidenceLedger
    ) -> bool:
        return TicketStore._event_evidence_complete(contract, ledger)

    def build(
        self,
        *,
        ticket: ObjectTicket,
        ledger: LiveEvidenceLedger,
        freeze_frame: int,
        freeze_sequence: int,
        output_dir: str | Path,
    ) -> OnlineEvidencePacket | None:
        from conceptgraph.revision.vlm import VLMIncidentEvidence

        issue = self._actionable_issue(ticket, ledger)
        if issue is None:
            return None
        association = ledger.associations[issue.anchor_event_uid]
        anchor_obs = str(association["obs_uid"])
        anchor_row = ledger.observations[anchor_obs]
        object_to_version = {
            str(object_uid): str(version_uid)
            for object_uid, version_uid in zip(
                association.get("object_uids_before") or (),
                association.get("candidate_object_version_uids") or (),
            )
        }
        contexts: list[tuple[str, str]] = []
        target_object = association.get("target_object_uid")
        target_version = association.get("target_object_version_before")
        if target_version:
            contexts.append(("CURRENT_ENTITY_CONTEXT", str(target_version)))
        alternative_rank = 0
        for item in association.get("top_candidates") or ():
            object_uid = str(item.get("object_uid") or "")
            version_uid = object_to_version.get(object_uid)
            if not version_uid:
                continue
            if target_object and object_uid == str(target_object):
                if not target_version:
                    contexts.append(("CURRENT_ENTITY_CONTEXT", version_uid))
                continue
            alternative_rank += 1
            contexts.append((f"CANDIDATE_{alternative_rank}_CONTEXT", version_uid))
            if alternative_rank >= 2:
                break

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        image_paths: list[Path] = []
        manifest: list[dict[str, Any]] = []

        def save_image(image: Any, *, alias: str, source_obs: str, reason: str) -> None:
            if len(image_paths) >= self.max_images:
                return
            image_id = f"I{len(image_paths) + 1:02d}"
            path = destination / f"{image_id}_{obs_key(source_obs)}.jpg"
            image.save(path, quality=90)
            image_paths.append(path)
            row = ledger.observations[source_obs]
            manifest.append(
                {
                    "image_id": image_id,
                    "context_alias": alias,
                    "obs_key": obs_key(source_obs),
                    "class_name": row.get("class_name"),
                    "confidence": row.get("confidence"),
                    "frame": frame_index(row.get("frame_uid")),
                    "selection_reason": reason,
                    "sha256": _sha256_file(path),
                }
            )

        frame_row = ledger.frames.get(frame_index(anchor_row.get("frame_uid")))
        if frame_row:
            save_image(
                self._overlay(anchor_row, frame_row),
                alias="ANCHOR",
                source_obs=anchor_obs,
                reason="exact trigger RGB with processed mask overlay",
            )
        save_image(
            self._load_crop(anchor_row),
            alias="ANCHOR",
            source_obs=anchor_obs,
            reason="exact trigger crop",
        )

        alias_versions: dict[str, str] = {}
        for alias, historical_version_uid in contexts:
            historical = ledger.object_versions.get(historical_version_uid)
            if not historical:
                continue
            lineage = str(historical.get("lineage_uid") or historical.get("object_uid"))
            latest = ledger.latest_version_for_lineage(lineage, cutoff_frame=freeze_frame) or historical
            alias_versions[alias] = str(latest["object_version_uid"])
            for index, member_uid in enumerate(self._representatives(latest, ledger, limit=2)):
                if len(image_paths) >= self.max_images:
                    break
                save_image(
                    self._load_crop(ledger.observations[member_uid]),
                    alias=alias,
                    source_obs=member_uid,
                    reason=(
                        "clearest context view" if index == 0 else "temporally diverse context view"
                    ),
                )

        if len(image_paths) < 3 or not alias_versions:
            return None
        observed = (
            "CREATE" if str(association.get("decision")) == "CREATE_OBJECT" else "ASSOCIATE"
        )
        scores = [
            {
                "alias": (
                    "CURRENT_ENTITY_CONTEXT"
                    if str(item.get("object_uid")) == str(target_object)
                    else f"ALTERNATIVE_{index}"
                ),
                "spatial": item.get("spatial_score"),
                "visual": item.get("visual_score"),
                "aggregate": item.get("aggregate_score"),
            }
            for index, item in enumerate(association.get("top_candidates") or (), 1)
        ][:3]
        prompt_payload = {
            "ticket_alias": ticket.ticket_uid,
            "trigger_obs_key": obs_key(anchor_obs),
            "scanner_hypotheses_not_ground_truth": [
                {"family": item.family, "raw_signals": item.raw_signals}
                for item in ticket.issues[:5]
            ],
            "observed_current_decision": observed,
            "candidate_scores": scores,
            "images": manifest,
        }
        prompt = (
            "Analyze one online 3D mapping identity ticket using only the supplied images and "
            "raw numeric evidence. Scanner hypotheses are weak triggers, not truth. Never use "
            "class name alone as physical identity proof. Choose exactly one action:\n"
            "1) SAME_INSTANCE with entities [ANCHOR, CANDIDATE_n_CONTEXT] when a native CREATE "
            "should have joined that instance;\n"
            "2) MOVE_OBSERVATION with obs_key, from_alias and to_alias when a native ASSOCIATE "
            "used the wrong owner;\n"
            "3) SEPARATE_MEMBER_GROUPS with groups [[ANCHOR],[CURRENT_ENTITY_CONTEXT]] when the "
            "anchor must not belong to its current owner;\n"
            "4) DEFER for no safe mutation, no error, dynamic-world ambiguity, geometry-only "
            "problems, or insufficient evidence.\n"
            "Return one short JSON object with keys action, confidence, entities, groups, "
            "obs_key, from_alias, to_alias, evidence_image_ids, counterevidence_image_ids, "
            "diagnosis, reason, missing_evidence. Use only aliases and image IDs shown here. "
            "The reason must be under 60 words.\n\nINCIDENT:\n"
            + json.dumps(prompt_payload, indent=2, sort_keys=True, ensure_ascii=False)
        )
        system_prompt = (
            "You are a conservative physical-instance constraint proposer for an online 3D "
            "scene graph. You do not see final membership, annotations, ground truth, or a "
            "desired answer. Unsupported or ambiguous cases must be DEFER."
        )
        evidence = VLMIncidentEvidence(
            incident_uid=ticket.ticket_uid,
            prompt=prompt,
            image_paths=tuple(image_paths),
            image_manifest=tuple(manifest),
            system_prompt=system_prompt,
        )
        packet_manifest = {
            "schema_version": "0.1.0",
            "ticket_uid": ticket.ticket_uid,
            "issue_uid": issue.issue_uid,
            "freeze_frame": int(freeze_frame),
            "freeze_sequence": int(freeze_sequence),
            "anchor_event_uid": issue.anchor_event_uid,
            "anchor_obs_key": obs_key(anchor_obs),
            "oracle_fields_included": False,
            "images": manifest,
            "alias_version_uids": alias_versions,
        }
        write_json(destination / "packet_manifest.json", packet_manifest)
        return OnlineEvidencePacket(
            ticket_uid=ticket.ticket_uid,
            issue_uid=issue.issue_uid,
            freeze_frame=int(freeze_frame),
            freeze_sequence=int(freeze_sequence),
            evidence=evidence,
            association=dict(association),
            alias_version_uids=alias_versions,
            allowed_image_ids=tuple(item["image_id"] for item in manifest),
            packet_manifest=packet_manifest,
        )


def _binding_row(version: Mapping[str, Any] | None) -> dict[str, Any]:
    if not version:
        return {
            "entity_uid": None,
            "lineage_uid": None,
            "origin_obs_uid": None,
            "identity_uids": [],
            "provenance_lineage_uids": [],
            "complete": False,
        }
    entity_uid = str(version.get("object_uid") or "")
    lineage_uid = str(version.get("lineage_uid") or entity_uid)
    members = list(version.get("member_observation_uids") or ())
    origin = str(version.get("origin_observation_uid") or (members[0] if members else ""))
    identities = list(version.get("identity_uids") or ()) or [lineage_uid]
    provenance = list(version.get("provenance_lineage_uids") or ()) or [lineage_uid]
    return {
        "entity_uid": entity_uid or None,
        "lineage_uid": lineage_uid or None,
        "origin_obs_uid": origin or None,
        "identity_uids": sorted(set(str(item) for item in identities if item)),
        "provenance_lineage_uids": sorted(set(str(item) for item in provenance if item)),
        "complete": bool(entity_uid and lineage_uid and origin and identities and provenance),
    }


def compile_vlm_response(
    *,
    packet: OnlineEvidencePacket,
    response: Mapping[str, Any],
    ledger: LiveEvidenceLedger,
) -> dict[str, Any]:
    """Bind one VLM proposal to immutable identities or fail closed."""

    from conceptgraph.revision.auto_constraints import (
        IncidentBinding,
        canonicalize_vote,
        compile_blind_candidate,
    )
    from conceptgraph.revision.vlm import normalize_incident_constraint

    proposal = dict(response.get("constraint", response))
    action = str(proposal.get("action") or "").upper()
    cited = tuple(str(item) for item in proposal.get("evidence_image_ids") or ())
    invalid_citations = sorted(set(cited) - set(packet.allowed_image_ids))
    if invalid_citations:
        return {
            "stage": "DEFERRED",
            "candidate_constraint": None,
            "defer_reasons": ["invalid_evidence_citations:" + ",".join(invalid_citations)],
            "proposal": proposal,
        }
    if action != "DEFER" and not cited:
        return {
            "stage": "DEFERRED",
            "candidate_constraint": None,
            "defer_reasons": ["executable_proposal_requires_cited_images"],
            "proposal": proposal,
        }

    observed = (
        "CREATE"
        if str(packet.association.get("decision")) == "CREATE_OBJECT"
        else "ASSOCIATE"
    )
    try:
        normalized = normalize_incident_constraint(
            proposal, observed_current_decision=observed
        )
        canonical = canonicalize_vote(normalized)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "stage": "DEFERRED",
            "candidate_constraint": None,
            "defer_reasons": [f"proposal_schema_rejected:{type(exc).__name__}:{exc}"],
            "proposal": proposal,
        }
    if canonical["action"] == "DEFER":
        return {
            "stage": "DEFERRED",
            "candidate_constraint": None,
            "defer_reasons": ["vlm_requested_defer"],
            "proposal": canonical,
        }

    aliases: dict[str, Any] = {
        "ANCHOR": {
            "entity_uid": None,
            "lineage_uid": None,
            "origin_obs_uid": str(packet.association["obs_uid"]),
            "identity_uids": [],
            "provenance_lineage_uids": [],
            "complete": False,
        }
    }
    for alias, version_uid in packet.alias_version_uids.items():
        aliases[alias] = _binding_row(ledger.object_versions.get(version_uid))

    target_after = ledger.object_versions.get(
        str(packet.association.get("target_object_version_after") or "")
    )
    created_binding = _binding_row(target_after)
    binding = IncidentBinding.from_mapping(
        {
            "case_uid": packet.ticket_uid,
            "obs_uid": str(packet.association["obs_uid"]),
            "obs_key": obs_key(str(packet.association["obs_uid"])),
            "event_uid": str(packet.association["event_uid"]),
            "event_sequence": int(packet.association.get("event_sequence", -1)),
            "observed_current_decision": observed,
            "aliases": aliases,
            "created_entity_uid": created_binding.get("entity_uid"),
            "created_identity_uid": (
                (created_binding.get("identity_uids") or [None])[0]
                if created_binding.get("complete")
                else None
            ),
            "evidence_refs": [
                f"image:{item}" for item in cited
            ]
            + [str(packet.association["event_uid"])],
        }
    )
    aggregate = {
        "aggregate_uid": stable_uid(
            "online_aggregate_", {"ticket": packet.ticket_uid, "proposal": canonical}
        ),
        "ready_for_binding": True,
        "selected_proposal": canonical,
        "defer_reasons": [],
    }
    compiled = compile_blind_candidate(aggregate, binding)
    compiled["proposal"] = canonical
    compiled["citation_check"] = {
        "pass": True,
        "cited_image_ids": list(cited),
        "allowed_image_ids": list(packet.allowed_image_ids),
    }
    return compiled


_UNIFIED_ALIAS_TO_CONTEXT = {
    "A": "ANCHOR",
    "E0": "CURRENT_ENTITY_CONTEXT",
    "E1": "CANDIDATE_1_CONTEXT",
    "E2": "CANDIDATE_2_CONTEXT",
}


def compile_unified_vlm_response(
    *,
    packet: OnlineEvidencePacket,
    result: Mapping[str, Any],
    ledger: LiveEvidenceLedger,
) -> dict[str, Any]:
    """Translate one strict candidate-ID result into one existing sparse primitive.

    The unified VLM does not author a free-form constraint.  It selects one row
    from a locally generated candidate table.  A single-anchor identity change
    uses the existing replay kernel.  A semantic candidate becomes a guarded,
    label-only pilot constraint.  Compound partitions remain audit-only and fail
    closed until a multi-constraint executor is independently validated.
    """

    output = result.get("output")
    candidates = result.get("candidates")
    raw_h_snapshot = packet.packet_manifest.get("h_snapshot") or {}
    h_snapshot_binding = {
        key: raw_h_snapshot.get(key)
        for key in (
            "snapshot_uid",
            "snapshot_sha256",
            "cutoff_frame",
            "cutoff_sequence",
            "watermark_source",
        )
        if raw_h_snapshot.get(key) is not None
    }
    audit = {
        "prompt_version": result.get("prompt_version"),
        "vlm_status": result.get("status"),
        "selected_candidate": output.get("selected_candidate") if isinstance(output, Mapping) else None,
        "output": dict(output) if isinstance(output, Mapping) else None,
        "source_h_snapshot": h_snapshot_binding,
    }

    def deferred(reason: str, *, stage: str = "DEFERRED", candidate: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "stage": stage,
            "candidate_constraint": None,
            "defer_reasons": [reason],
            "proposal": dict(output) if isinstance(output, Mapping) else None,
            "selected_candidate_row": dict(candidate) if isinstance(candidate, Mapping) else None,
            "unified_vlm": audit,
            "source_h_snapshot": h_snapshot_binding,
        }

    if str(result.get("status")) != "VALID":
        code = str(result.get("defer_code") or result.get("status") or "invalid_result")
        return deferred("unified_vlm_not_valid:" + code)
    if not isinstance(output, Mapping) or not isinstance(candidates, Sequence):
        return deferred("unified_vlm_contract_missing")

    selected_id = str(output.get("selected_candidate") or "")
    matches = [
        row for row in candidates
        if isinstance(row, Mapping) and str(row.get("id") or "") == selected_id
    ]
    if len(matches) != 1:
        return deferred("selected_candidate_not_unique_or_missing")
    candidate = matches[0]
    audit["selected_action"] = candidate.get("action")
    if str(candidate.get("axis") or "NONE") != str(output.get("decision_axis") or ""):
        return deferred("selected_candidate_axis_mismatch", candidate=candidate)

    action = str(candidate.get("action") or "").upper()
    axis = str(candidate.get("axis") or "NONE")
    if selected_id == "H0" or action == "NO_OP":
        return deferred("vlm_selected_no_op", stage="NO_OP", candidate=candidate)
    if selected_id == "DEFER" or action == "REQUEST_MORE_EVIDENCE":
        return deferred("vlm_requested_more_evidence", candidate=candidate)
    if not bool(candidate.get("executable")):
        return deferred("selected_candidate_not_executable", candidate=candidate)

    evidence_ids = [str(value) for value in output.get("evidence_ids") or ()]
    if not evidence_ids:
        return deferred("executable_candidate_requires_evidence", candidate=candidate)
    allowed_image_ids = tuple(
        str(value) for value in result.get("allowed_image_ids") or ("I1", "I2", "I3")
    )
    invalid_evidence_ids = sorted(set(evidence_ids) - set(allowed_image_ids))
    if invalid_evidence_ids:
        return deferred(
            "invalid_evidence_citations:" + ",".join(invalid_evidence_ids),
            candidate=candidate,
        )
    parameters = candidate.get("parameters")
    if not isinstance(parameters, Mapping):
        return deferred("selected_candidate_parameters_missing", candidate=candidate)

    if axis == "SEMANTIC_LABEL":
        if action != "RELABEL_ENTITY":
            return deferred("selected_semantic_action_unsupported:" + action, candidate=candidate)
        if str(parameters.get("entity") or "") != "E0":
            return deferred("semantic_relabel_must_target_E0", candidate=candidate)
        if len(set(evidence_ids)) < 2:
            return deferred("semantic_relabel_requires_two_images", candidate=candidate)
        label_text = candidate.get("label_text")
        if not isinstance(label_text, Mapping):
            return deferred("semantic_candidate_label_text_missing", candidate=candidate)

        version_uid = str(
            packet.alias_version_uids.get("CURRENT_ENTITY_CONTEXT")
            or packet.association.get("target_object_version_after")
            or packet.association.get("target_object_version_before")
            or ""
        )
        version = ledger.object_versions.get(version_uid)
        if not version:
            return deferred("semantic_target_version_unresolved", candidate=candidate)
        lineage_uid = str(version.get("lineage_uid") or version.get("object_uid") or "")
        latest_for_lineage = getattr(ledger, "latest_version_for_lineage", None)
        if lineage_uid and callable(latest_for_lineage):
            version = (
                latest_for_lineage(lineage_uid, cutoff_frame=packet.freeze_frame)
                or version
            )
        binding = _binding_row(version)
        if not binding.get("complete"):
            return deferred("semantic_target_binding_incomplete", candidate=candidate)

        current_label = str(version.get("class_name") or "").strip()
        candidate_from = str(label_text.get("from") or "").strip()
        candidate_target_label = str(label_text.get("to") or "").strip()
        output_target_label = output.get("suggested_label_for_logging")
        if not isinstance(output_target_label, str) or not output_target_label.strip():
            return deferred("semantic_output_label_missing", candidate=candidate)
        target_label = output_target_label.strip()
        if target_label != candidate_target_label:
            return deferred(
                "semantic_output_label_not_exact_candidate",
                candidate=candidate,
            )

        def comparable_label(value: Any) -> str:
            return re.sub(r"\s+\d+$", "", _normal_label(value))

        if not current_label or comparable_label(current_label) != comparable_label(candidate_from):
            return deferred("semantic_source_label_changed_since_freeze", candidate=candidate)
        if not target_label or comparable_label(target_label) == comparable_label(current_label):
            return deferred("semantic_target_label_empty_or_unchanged", candidate=candidate)

        from conceptgraph.revision.auto_constraints import semantic_constraint_fingerprint
        from conceptgraph.revision.constraints import SparseRepairConstraint

        sparse = SparseRepairConstraint.from_mapping(
            {
                "type": "RELABEL",
                "entity_uid": binding["entity_uid"],
                "target_lineage_uid": binding["lineage_uid"],
                "label": target_label,
                "expected_label": current_label,
                "reason": "vlm_output_label_matches_finite_semantic_candidate",
                "applies_at_event_uid": str(packet.association["event_uid"]),
                "active_from_sequence": int(packet.association.get("event_sequence", -1)),
                "source": "unified_vlm_semantic_pilot",
                "evidence_refs": [
                    *(f"image:{image_id}" for image_id in evidence_ids),
                    str(packet.association["event_uid"]),
                ],
            }
        )
        bound = sparse.as_dict()
        return {
            "stage": "BOUND_PENDING_SHADOW",
            "candidate_constraint": bound,
            "constraint_fingerprint": semantic_constraint_fingerprint(bound),
            "defer_reasons": [],
            "proposal": dict(output),
            "selected_candidate_row": dict(candidate),
            "unified_vlm": audit,
            "source_h_snapshot": h_snapshot_binding,
            "citation_check": {
                "pass": True,
                "cited_image_ids": evidence_ids,
                "allowed_image_ids": list(allowed_image_ids),
            },
            "semantic_pilot": {
                "target_entity_uid": binding["entity_uid"],
                "expected_label": current_label,
                "target_label": target_label,
                "label_source": "vlm_output_guarded_by_finite_candidate",
                "accuracy_validated": False,
            },
        }

    if axis != "IDENTITY":
        return deferred("selected_candidate_axis_not_executable:" + axis, candidate=candidate)

    proposal: dict[str, Any]
    if action == "PARTITION_ALIASES":
        groups = [
            [str(alias) for alias in group]
            for group in parameters.get("groups") or ()
            if isinstance(group, Sequence) and not isinstance(group, (str, bytes))
        ]
        flattened = [alias for group in groups for alias in group]
        if not groups or len(flattened) != len(set(flattened)) or "A" not in flattened:
            return deferred("invalid_alias_partition", candidate=candidate)
        anchor_group = next(group for group in groups if "A" in group)
        anchor_contexts = [alias for alias in anchor_group if alias != "A"]
        other_groups = [group for group in groups if "A" not in group]
        if len(anchor_contexts) == 0:
            return deferred("partition_changes_only_non_anchor_entities", candidate=candidate)
        if len(anchor_contexts) != 1 or any(len(group) != 1 for group in other_groups):
            return deferred(
                "compound_identity_partition_requires_multi_constraint_executor",
                candidate=candidate,
            )
        target_alias = _UNIFIED_ALIAS_TO_CONTEXT.get(anchor_contexts[0])
        if not target_alias:
            return deferred("partition_target_alias_unknown", candidate=candidate)
        proposal = {
            "action": "SAME_INSTANCE",
            "entities": ["ANCHOR", target_alias],
        }
    elif action == "SAME_INSTANCE":
        aliases = [str(value) for value in parameters.get("entities") or ()]
        mapped = [_UNIFIED_ALIAS_TO_CONTEXT.get(alias) for alias in aliases]
        if any(value is None for value in mapped):
            return deferred("same_instance_alias_unknown", candidate=candidate)
        proposal = {"action": "SAME_INSTANCE", "entities": mapped}
    elif action == "MOVE_OBSERVATION":
        if str(parameters.get("observation")) != "A":
            return deferred("move_observation_must_target_anchor", candidate=candidate)
        source = _UNIFIED_ALIAS_TO_CONTEXT.get(str(parameters.get("from")))
        target = _UNIFIED_ALIAS_TO_CONTEXT.get(str(parameters.get("to")))
        if not source or not target:
            return deferred("move_observation_alias_unknown", candidate=candidate)
        proposal = {
            "action": "MOVE_OBSERVATION",
            "obs_key": obs_key(str(packet.association["obs_uid"])),
            "from_alias": source,
            "to_alias": target,
        }
    elif action == "SEPARATE_MEMBER_GROUPS":
        mapped_groups = []
        for group in parameters.get("groups") or ():
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                return deferred("separation_group_invalid", candidate=candidate)
            mapped = [_UNIFIED_ALIAS_TO_CONTEXT.get(str(alias)) for alias in group]
            if any(value is None for value in mapped):
                return deferred("separation_alias_unknown", candidate=candidate)
            mapped_groups.append(mapped)
        proposal = {"action": "SEPARATE_MEMBER_GROUPS", "groups": mapped_groups}
    else:
        return deferred("selected_identity_action_unsupported:" + action, candidate=candidate)

    proposal.update(
        {
            "confidence": float(output.get("confidence_diagnostic", 0.0)),
            "evidence_image_ids": evidence_ids,
        }
    )
    translated_packet = replace(packet, allowed_image_ids=allowed_image_ids)
    compiled = compile_vlm_response(
        packet=translated_packet,
        response={"constraint": proposal},
        ledger=ledger,
    )
    compiled["selected_candidate_row"] = dict(candidate)
    compiled["unified_vlm"] = audit
    compiled["source_h_snapshot"] = h_snapshot_binding
    compiled["adapter_proposal"] = proposal
    return compiled


@dataclass(frozen=True)
class FrozenEvidenceView:
    root: Path
    source_experiment_root: Path
    cutoff_frame: int
    max_sequence: int
    manifest: dict[str, Any]


def _row_before_cutoff(
    row: Mapping[str, Any], *, cutoff_frame: int, max_sequence: int
) -> bool:
    index = frame_index(row.get("frame_uid"))
    if index >= 0:
        return index <= int(cutoff_frame)
    sequence = int(row.get("event_sequence", -1))
    return sequence < 0 or sequence <= int(max_sequence)


def freeze_watermarked_view(
    *,
    ledger: LiveEvidenceLedger,
    cutoff_frame: int,
    output_root: str | Path,
) -> FrozenEvidenceView:
    """Freeze complete rows through h and synthesize h's active membership."""

    root = Path(output_root).resolve()
    evidence = root / "evidence"
    manifest_path = root / "freeze_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return FrozenEvidenceView(
            root=root,
            source_experiment_root=ledger.experiment_root,
            cutoff_frame=int(existing["cutoff_frame"]),
            max_sequence=int(existing["max_sequence"]),
            manifest=existing,
        )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"incomplete frozen view already exists: {root}")
    evidence.mkdir(parents=True, exist_ok=True)
    eligible_events = [
        row
        for row in [*ledger.associations.values(), *ledger.mapping_events.values()]
        if frame_index(row.get("frame_uid")) <= int(cutoff_frame)
    ]
    max_sequence = max(
        (int(row.get("event_sequence", -1)) for row in eligible_events), default=-1
    )
    hashes: dict[str, str] = {}
    frozen_rows: dict[str, list[dict[str, Any]]] = {}
    for name in STREAM_FILES:
        selected = [
            row
            for row in ledger.rows[name]
            if _row_before_cutoff(
                row, cutoff_frame=int(cutoff_frame), max_sequence=max_sequence
            )
        ]
        frozen_rows[name] = selected
        path = evidence / name
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected:
                handle.write(_canonical_json(row) + "\n")
        hashes[name] = _sha256_file(path)

    latest_by_object: dict[str, dict[str, Any]] = {}
    for row in frozen_rows["object_versions.jsonl"]:
        object_uid = str(row["object_uid"])
        previous = latest_by_object.get(object_uid)
        if previous is None or int(row.get("version", 0)) >= int(previous.get("version", 0)):
            latest_by_object[object_uid] = row
    membership = []
    for object_uid, row in sorted(latest_by_object.items()):
        if str(row.get("status", "active")) != "active":
            continue
        membership.append(
            {
                "object_uid": object_uid,
                "status": "active",
                "class_name": row.get("class_name"),
                "class_histogram": row.get("class_histogram") or {},
                "member_observation_uids": list(
                    dict.fromkeys(str(item) for item in row.get("member_observation_uids") or ())
                ),
                "duplicate_member_observation_uids": {},
                "num_detections": int(row.get("num_detections", 0)),
                "bbox_center": row.get("bbox_center"),
                "bbox_extent": row.get("bbox_extent"),
                "n_points": int(row.get("n_points", 0)),
                "parent_or_merged_from_object_uids": [],
                "outgoing_edge_uids": [],
                "incoming_edge_uids": [],
            }
        )
    membership_path = evidence / "final_membership.json"
    write_json(membership_path, membership)
    hashes["final_membership.json"] = _sha256_file(membership_path)
    manifest = {
        "schema_version": "0.1.0",
        "cutoff_frame": int(cutoff_frame),
        "max_sequence": int(max_sequence),
        "source_experiment_root": str(ledger.experiment_root),
        "row_counts": {name: len(rows) for name, rows in frozen_rows.items()},
        "active_object_count": len(membership),
        "hashes": hashes,
        "created_at_unix": time.time(),
    }
    write_json(manifest_path, manifest)
    return FrozenEvidenceView(
        root=root,
        source_experiment_root=ledger.experiment_root,
        cutoff_frame=int(cutoff_frame),
        max_sequence=int(max_sequence),
        manifest=manifest,
    )


def _owner(state: Mapping[str, Any], observation_uid: str) -> str | None:
    owners = [
        str(entity)
        for entity, members in (state.get("membership") or {}).items()
        if str(observation_uid) in set(str(item) for item in members or ())
    ]
    return owners[0] if len(owners) == 1 else None


def _object_row(state: Mapping[str, Any], entity_uid: str | None) -> Mapping[str, Any] | None:
    if entity_uid is None:
        return None
    return next(
        (
            row
            for row in state.get("objects") or ()
            if str(row.get("entity_uid")) == str(entity_uid)
        ),
        None,
    )


def _unique_frames(members: Iterable[str]) -> int:
    return len({frame_index(item) for item in members if frame_index(item) >= 0})


def _purity(row: Mapping[str, Any] | None) -> float:
    if not row:
        return 0.0
    histogram = {
        str(key): int(value) for key, value in (row.get("class_histogram") or {}).items()
    }
    total = sum(histogram.values())
    return max(histogram.values()) / total if total else 0.0


def _semantic_state_hash(state: Mapping[str, Any]) -> str:
    labels = [
        {
            "entity_uid": str(row.get("entity_uid") or ""),
            "class_name": str(row.get("class_name") or ""),
        }
        for row in state.get("objects") or ()
    ]
    labels.sort(key=lambda row: row["entity_uid"])
    return hashlib.sha256(_canonical_json(labels).encode("utf-8")).hexdigest()


def _apply_semantic_relabels(
    state: Mapping[str, Any], constraints: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply optimistic, label-only overrides to a copied replay state."""

    derived = copy.deepcopy(dict(state))
    rows = [row for row in derived.get("objects") or () if isinstance(row, dict)]
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_entity.setdefault(str(row.get("entity_uid") or ""), []).append(row)
    reports: list[dict[str, Any]] = []
    seen_entities: set[str] = set()

    def stable_label(value: Any) -> str:
        return re.sub(r"\s+\d+$", "", _normal_label(value))

    for raw in constraints:
        entity_uid = str(raw.get("entity_uid") or "")
        expected_label = str(raw.get("expected_label") or "").strip()
        target_label = str(raw.get("label") or "").strip()
        report = {
            "constraint_uid": raw.get("constraint_uid"),
            "entity_uid": entity_uid,
            "expected_label": expected_label,
            "target_label": target_label,
            "applied": False,
            "reason": None,
        }
        matches = by_entity.get(entity_uid, [])
        if entity_uid in seen_entities:
            report["reason"] = "duplicate_semantic_target"
        elif len(matches) != 1:
            report["reason"] = "semantic_target_not_unique"
        elif not expected_label:
            report["reason"] = "expected_label_missing"
        elif not target_label:
            report["reason"] = "target_label_missing"
        else:
            row = matches[0]
            current_label = str(row.get("class_name") or "").strip()
            report["current_label"] = current_label
            if stable_label(current_label) != stable_label(expected_label):
                report["reason"] = "expected_label_mismatch"
            elif stable_label(current_label) == stable_label(target_label):
                report["reason"] = "target_label_unchanged"
            else:
                row["class_name"] = target_label
                report["applied"] = True
                report["reason"] = "label_replaced"
        seen_entities.add(entity_uid)
        reports.append(report)

    derived["semantic_label_overrides"] = reports
    derived["semantic_label_override_count"] = sum(
        bool(report["applied"]) for report in reports
    )
    derived["semantic_state_hash"] = _semantic_state_hash(derived)
    return derived, reports


def _semantic_relabel_gate(
    *,
    before_state: Mapping[str, Any],
    after_state: Mapping[str, Any],
    constraints: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    """Prove that a semantic pilot changed labels and nothing structural."""

    if not constraints:
        return {
            "all_semantic_relabels_applied": True,
            "semantic_state_changed": False,
            "expected_labels_matched": True,
            "target_labels_applied": True,
            "membership_unchanged": True,
            "membership_hash_unchanged": True,
            "object_count_conserved": True,
            "observation_count_conserved": True,
            "class_histograms_unchanged": True,
            "target_non_label_fields_unchanged": True,
            "non_target_objects_unchanged": True,
            "pass": True,
        }

    before_rows = {
        str(row.get("entity_uid") or ""): dict(row)
        for row in before_state.get("objects") or ()
    }
    after_rows = {
        str(row.get("entity_uid") or ""): dict(row)
        for row in after_state.get("objects") or ()
    }
    targets = {str(raw.get("entity_uid") or "") for raw in constraints}

    def stable_label(value: Any) -> str:
        return re.sub(r"\s+\d+$", "", _normal_label(value))

    def without_label(row: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(row)
        value.pop("class_name", None)
        return value

    expected_labels_matched = all(
        entity in before_rows
        and stable_label(before_rows[entity].get("class_name"))
        == stable_label(raw.get("expected_label"))
        for raw in constraints
        for entity in [str(raw.get("entity_uid") or "")]
    )
    target_labels_applied = all(
        entity in after_rows
        and stable_label(after_rows[entity].get("class_name"))
        == stable_label(raw.get("label"))
        for raw in constraints
        for entity in [str(raw.get("entity_uid") or "")]
    )
    target_non_label_fields_unchanged = all(
        entity in before_rows
        and entity in after_rows
        and without_label(before_rows[entity]) == without_label(after_rows[entity])
        for entity in targets
    )
    non_targets = set(before_rows) - targets
    non_target_objects_unchanged = (
        set(before_rows) == set(after_rows)
        and all(before_rows[entity] == after_rows[entity] for entity in non_targets)
    )
    before_observations = sum(
        len(values or ()) for values in (before_state.get("membership") or {}).values()
    )
    after_observations = sum(
        len(values or ()) for values in (after_state.get("membership") or {}).values()
    )
    gates = {
        "all_semantic_relabels_applied": (
            len(reports) == len(constraints)
            and all(bool(report.get("applied")) for report in reports)
        ),
        "semantic_state_changed": (
            _semantic_state_hash(before_state) != _semantic_state_hash(after_state)
        ),
        "expected_labels_matched": expected_labels_matched,
        "target_labels_applied": target_labels_applied,
        "membership_unchanged": (
            before_state.get("membership") == after_state.get("membership")
        ),
        "membership_hash_unchanged": (
            before_state.get("state_hash") == after_state.get("state_hash")
        ),
        "object_count_conserved": set(before_rows) == set(after_rows),
        "observation_count_conserved": before_observations == after_observations,
        "class_histograms_unchanged": all(
            before_rows.get(entity, {}).get("class_histogram")
            == after_rows.get(entity, {}).get("class_histogram")
            for entity in set(before_rows) | set(after_rows)
        ),
        "target_non_label_fields_unchanged": target_non_label_fields_unchanged,
        "non_target_objects_unchanged": non_target_objects_unchanged,
    }
    gates["pass"] = all(gates.values())
    return gates


def scene_health_metrics(state: Mapping[str, Any]) -> dict[str, Any]:
    objects = list(state.get("objects") or ())
    membership = state.get("membership") or {}
    owner_counts: dict[str, int] = {}
    for members in membership.values():
        for observation_uid in members or ():
            owner_counts[str(observation_uid)] = owner_counts.get(str(observation_uid), 0) + 1
    frames = [
        _unique_frames(row.get("member_observation_uids") or ()) for row in objects
    ]
    purities = [_purity(row) for row in objects]
    weights = [max(1, len(row.get("member_observation_uids") or ())) for row in objects]
    weighted_purity = (
        sum(value * weight for value, weight in zip(purities, weights)) / sum(weights)
        if weights
        else 0.0
    )
    singleton = sum(len(row.get("member_observation_uids") or ()) == 1 for row in objects)
    low_purity = sum(value < 0.70 for value in purities)
    invalid_geometry = 0
    for row in objects:
        center = np.asarray(row.get("bbox_center") or (), dtype=float)
        extent = np.asarray(row.get("bbox_extent") or (), dtype=float)
        if (
            center.shape != (3,)
            or extent.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(extent).all()
            or np.any(extent <= 0)
            or int(row.get("n_points", 0)) <= 0
        ):
            invalid_geometry += 1
    return {
        "object_count": len(objects),
        "observation_count": len(owner_counts),
        "duplicate_ownership_count": sum(value != 1 for value in owner_counts.values()),
        "singleton_object_count": singleton,
        "singleton_object_rate": singleton / len(objects) if objects else 0.0,
        "mean_unique_frames_per_object": sum(frames) / len(frames) if frames else 0.0,
        "weighted_semantic_purity": weighted_purity,
        "low_purity_object_count": low_purity,
        "low_purity_object_rate": low_purity / len(objects) if objects else 0.0,
        "invalid_geometry_object_count": invalid_geometry,
        "state_hash": state.get("state_hash"),
        "semantic_state_hash": _semantic_state_hash(state),
    }


def _constraint_target_gate(
    *,
    noop_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
    constraint: Mapping[str, Any],
) -> dict[str, Any]:
    observation_uid = str(constraint.get("obs_uid") or "")
    noop_owner = _owner(noop_state, observation_uid)
    candidate_owner = _owner(candidate_state, observation_uid)
    noop_row = _object_row(noop_state, noop_owner)
    candidate_row = _object_row(candidate_state, candidate_owner)
    candidate_frames = _unique_frames(
        (candidate_row or {}).get("member_observation_uids") or ()
    )
    purity_delta = _purity(candidate_row) - _purity(noop_row)
    constraint_type = str(constraint.get("type") or "")
    owner_changed = noop_owner != candidate_owner
    target_satisfied = owner_changed
    if constraint_type == "ASSIGN_OBSERVATION":
        target = str(constraint.get("target_entity_uid") or "")
        target_lineage = str(constraint.get("target_lineage_uid") or "")
        target_satisfied = bool(
            candidate_owner
            and (
                candidate_owner == target
                or target_lineage
                in set((candidate_row or {}).get("revision_identity_uids") or ())
                or target_lineage
                in set((candidate_row or {}).get("revision_lineage_uids") or ())
            )
        )
    return {
        "constraint_type": constraint_type,
        "noop_owner": noop_owner,
        "candidate_owner": candidate_owner,
        "owner_changed": owner_changed,
        "target_satisfied": target_satisfied,
        "candidate_support_frame_count": candidate_frames,
        "noop_owner_purity": _purity(noop_row),
        "candidate_owner_purity": _purity(candidate_row),
        "target_purity_delta": purity_delta,
        "pass": bool(target_satisfied and candidate_frames >= 2 and purity_delta >= -0.05),
    }


def _run_semantic_shadow_validation(
    *,
    provenance: Any,
    baseline_state: Mapping[str, Any],
    packet: OnlineEvidencePacket,
    compilation: Mapping[str, Any],
    constraint: Mapping[str, Any],
    destination: Path,
    started: float,
) -> dict[str, Any]:
    """Validate one VLM-selected label replacement at the frozen watermark."""

    from conceptgraph.revision.verify import StructuralVerifier

    noop_state = copy.deepcopy(dict(baseline_state))
    noop_state["semantic_state_hash"] = _semantic_state_hash(noop_state)
    candidate_state, reports = _apply_semantic_relabels(noop_state, (constraint,))
    target_entity = str(constraint.get("entity_uid") or "")
    target_row = _object_row(noop_state, target_entity)
    closure = {
        "obs_uids": sorted(
            str(value) for value in (target_row or {}).get("member_observation_uids") or ()
        ),
        "entity_uids": [target_entity] if target_entity else [],
        "event_uids": [str(packet.association.get("event_uid") or "")],
        "version_uids": [],
        "edge_uids": [],
        "start_sequence": 0,
        "end_sequence": provenance.max_sequence,
    }
    verifier = StructuralVerifier(provenance)
    expected_hashes = provenance.source_hashes()
    noop_verification = verifier.verify(
        baseline_state=baseline_state,
        derived_state=noop_state,
        closure=closure,
        expected_source_hashes=expected_hashes,
    )
    candidate_verification = verifier.verify(
        baseline_state=noop_state,
        derived_state=candidate_state,
        closure=closure,
        expected_source_hashes=expected_hashes,
    )
    semantic_gate = _semantic_relabel_gate(
        before_state=noop_state,
        after_state=candidate_state,
        constraints=(constraint,),
        reports=reports,
    )
    noop_exact = bool(
        noop_verification["pass"]
        and noop_state.get("state_hash") == baseline_state.get("state_hash")
        and _semantic_state_hash(noop_state) == _semantic_state_hash(baseline_state)
    )
    adopted = bool(
        noop_exact and candidate_verification["pass"] and semantic_gate["pass"]
    )
    result = {
        "schema_version": "0.1.0",
        "ticket_uid": packet.ticket_uid,
        "freeze_frame": packet.freeze_frame,
        "freeze_sequence": packet.freeze_sequence,
        "anchor_event_uid": str(packet.association.get("event_uid") or ""),
        "decision": "WOULD_COMMIT" if adopted else "DEFER",
        "reason": (
            "semantic_label_only_gate_pass"
            if adopted
            else "semantic_label_only_gate_not_satisfied"
        ),
        "constraint": dict(constraint),
        "constraint_fingerprint": compilation.get("constraint_fingerprint"),
        "closure": closure,
        "noop_exact_reproduction": noop_exact,
        "noop_verification": noop_verification,
        "candidate_verification": candidate_verification,
        "mechanism": {
            "constraint_hit_count": sum(bool(row.get("applied")) for row in reports),
            "constraint_override_count": sum(
                bool(row.get("applied")) for row in reports
            ),
            "partition_changed_from_noop": False,
            "semantic_label_changed": semantic_gate["semantic_state_changed"],
        },
        "target_gate": semantic_gate,
        "semantic_relabel_reports": reports,
        "evaluation_scope": "semantic_pilot_vlm_selected_label_only",
        "accuracy_validated": False,
        "noop_metrics": scene_health_metrics(noop_state),
        "candidate_metrics": scene_health_metrics(candidate_state),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(destination / "noop_state.json", noop_state)
    write_json(destination / "candidate_state.json", candidate_state)
    write_json(destination / "shadow_result.json", result)
    return result


def _run_shadow_validation_impl(
    *,
    frozen_view: FrozenEvidenceView,
    packet: OnlineEvidencePacket,
    compilation: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run NOOP and candidate from one exact snapshot to the same h."""

    from conceptgraph.revision.constraints import ReplayMode
    from conceptgraph.revision.dependency_graph import TypedDependencyGraph
    from conceptgraph.revision.index import ProvenanceIndex
    from conceptgraph.revision.snapshot import AnchorStateBuilder
    from conceptgraph.revision.sparse_replay import SparseCounterfactualReplayEngine
    from conceptgraph.revision.verify import StructuralVerifier

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    constraint = compilation.get("candidate_constraint")
    if not isinstance(constraint, Mapping):
        result = {
            "ticket_uid": packet.ticket_uid,
            "decision": "DEFER",
            "reason": "no_executable_constraint",
            "compilation": dict(compilation),
        }
        write_json(destination / "shadow_result.json", result)
        return result
    started = time.perf_counter()
    provenance = ProvenanceIndex(frozen_view.root, validate=True)
    # JSONL comes from the frozen view, while all hash-bound image/point artifacts
    # and config remain under the immutable source experiment.
    provenance.experiment_root = frozen_view.source_experiment_root
    engine = SparseCounterfactualReplayEngine(provenance)
    baseline_state = engine.replay_global(mode=ReplayMode.NATURAL_REPLAY)
    if str(constraint.get("type") or "").upper() == "RELABEL":
        return _run_semantic_shadow_validation(
            provenance=provenance,
            baseline_state=baseline_state,
            packet=packet,
            compilation=compilation,
            constraint=constraint,
            destination=destination,
            started=started,
        )
    graph = TypedDependencyGraph(provenance)
    anchor_uid = str(packet.association["event_uid"])
    anchor_sequence = int(packet.association.get("event_sequence", -1))
    seed_versions = [
        str(value)
        for value in [
            packet.association.get("target_object_version_before"),
            *(packet.association.get("candidate_object_version_uids") or ()),
        ]
        if value and str(value) in provenance.object_versions
    ]
    seed_versions = [
        uid
        for uid in dict.fromkeys(seed_versions)
        if provenance.sequence(
            provenance.get_event(str(provenance.get_object_version(uid)["trigger_event_uid"]))
        )
        < anchor_sequence
    ]
    if not seed_versions:
        result = {
            "ticket_uid": packet.ticket_uid,
            "decision": "DEFER",
            "reason": "no_pre_anchor_dependency_seed",
            "compilation": dict(compilation),
        }
        write_json(destination / "shadow_result.json", result)
        return result
    closure = graph.forward_closure(
        anchor_event_uid=anchor_uid,
        seed_version_uids=seed_versions,
        stop_watermark=provenance.max_sequence,
    )
    snapshot = AnchorStateBuilder(provenance, engine).build_pre_anchor_state(
        anchor_event_uid=anchor_uid,
        dependency_seed=seed_versions,
        strict=True,
    )
    common = {
        "mode": ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
        "snapshot_objects": snapshot.objects,
        "snapshot_runtime_ms": float(snapshot.state.get("runtime_ms", 0.0)),
        "anchor_frame": snapshot.anchor_frame,
        "snapshot_watermark_event_sequence": snapshot.watermark_event_sequence,
        "closure": closure,
        "current_state": baseline_state,
        "snapshot_timing": snapshot.state.get("timing") or {},
    }
    noop_state = engine.replay_local_from_snapshot(constraints=(), **common)
    candidate_state = engine.replay_local_from_snapshot(
        constraints=(constraint,), **common
    )
    verifier = StructuralVerifier(provenance)
    expected_hashes = provenance.source_hashes()
    noop_verification = verifier.verify(
        baseline_state=baseline_state,
        derived_state=noop_state,
        closure=closure.as_dict(),
        expected_source_hashes=expected_hashes,
    )
    candidate_verification = verifier.verify(
        baseline_state=noop_state,
        derived_state=candidate_state,
        closure=closure.as_dict(),
        expected_source_hashes=expected_hashes,
    )
    noop_exact = (
        noop_verification["pass"]
        and noop_state.get("state_hash") == baseline_state.get("state_hash")
    )
    mechanism = {
        "constraint_hit_count": int(candidate_state.get("constraint_hit_count", 0)),
        "constraint_override_count": int(
            candidate_state.get("constraint_override_count", 0)
        ),
        "partition_changed_from_noop": (
            candidate_state.get("state_hash") != noop_state.get("state_hash")
        ),
    }
    target_gate = _constraint_target_gate(
        noop_state=noop_state,
        candidate_state=candidate_state,
        constraint=constraint,
    )
    adopted = bool(
        noop_exact
        and candidate_verification["pass"]
        and mechanism["constraint_hit_count"] >= 1
        and mechanism["constraint_override_count"] >= 1
        and mechanism["partition_changed_from_noop"]
        and target_gate["pass"]
    )
    result = {
        "schema_version": "0.1.0",
        "ticket_uid": packet.ticket_uid,
        "freeze_frame": frozen_view.cutoff_frame,
        "freeze_sequence": frozen_view.max_sequence,
        "anchor_event_uid": anchor_uid,
        "decision": "WOULD_COMMIT" if adopted else "DEFER",
        "reason": (
            "same_watermark_noop_candidate_gate_pass"
            if adopted
            else "same_watermark_gate_not_satisfied"
        ),
        "constraint": dict(constraint),
        "closure": closure.as_dict(),
        "snapshot": snapshot.as_dict(),
        "noop_exact_reproduction": noop_exact,
        "noop_verification": noop_verification,
        "candidate_verification": candidate_verification,
        "mechanism": mechanism,
        "target_gate": target_gate,
        "noop_metrics": scene_health_metrics(noop_state),
        "candidate_metrics": scene_health_metrics(candidate_state),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(destination / "noop_state.json", noop_state)
    write_json(destination / "candidate_state.json", candidate_state)
    write_json(destination / "shadow_result.json", result)
    return result


def run_shadow_validation(
    *,
    frozen_view: FrozenEvidenceView,
    packet: OnlineEvidencePacket,
    compilation: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Never let one non-replayable ticket terminate the online sidecar."""

    destination = Path(output_dir)
    try:
        return _run_shadow_validation_impl(
            frozen_view=frozen_view,
            packet=packet,
            compilation=compilation,
            output_dir=destination,
        )
    except Exception as exc:
        result = {
            "schema_version": "0.1.0",
            "ticket_uid": packet.ticket_uid,
            "freeze_frame": frozen_view.cutoff_frame,
            "freeze_sequence": frozen_view.max_sequence,
            "decision": "DEFER",
            "reason": "shadow_executor_or_evidence_incompatible",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "candidate_constraint": compilation.get("candidate_constraint"),
        }
        write_json(destination / "shadow_result.json", result)
        return result


def final_scene_metric_gate(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    require_partition_change: bool = True,
) -> dict[str, bool]:
    baseline_object_count = int(baseline_metrics["object_count"])
    minimum_object_count = max(1, math.floor(0.90 * baseline_object_count))
    return {
        "partition_changed": (
            not require_partition_change
            or candidate_metrics["state_hash"] != baseline_metrics["state_hash"]
        ),
        "observation_count_conserved": (
            candidate_metrics["observation_count"] == baseline_metrics["observation_count"]
        ),
        "object_count_not_collapsed": (
            int(candidate_metrics["object_count"]) >= minimum_object_count
        ),
        "semantic_purity_not_degraded": (
            candidate_metrics["weighted_semantic_purity"]
            >= baseline_metrics["weighted_semantic_purity"] - 0.01
        ),
        "singleton_rate_not_degraded": (
            candidate_metrics["singleton_object_rate"]
            <= baseline_metrics["singleton_object_rate"] + 0.01
        ),
        "low_purity_rate_not_degraded": (
            candidate_metrics["low_purity_object_rate"]
            <= baseline_metrics["low_purity_object_rate"] + 0.02
        ),
        "no_duplicate_ownership": candidate_metrics["duplicate_ownership_count"] == 0,
        "no_invalid_geometry": candidate_metrics["invalid_geometry_object_count"] == 0,
    }


def run_final_combined_replay(
    *,
    experiment_root: str | Path,
    constraints: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Apply accepted identity constraints, then guarded label-only pilots."""

    from conceptgraph.revision.constraints import ReplayMode
    from conceptgraph.revision.dependency_graph import TypedDependencyGraph
    from conceptgraph.revision.index import ProvenanceIndex
    from conceptgraph.revision.sparse_replay import SparseCounterfactualReplayEngine
    from conceptgraph.revision.verify import StructuralVerifier

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = ProvenanceIndex(experiment_root, validate=True)
    engine = SparseCounterfactualReplayEngine(provenance)
    baseline = engine.replay_global(mode=ReplayMode.NATURAL_REPLAY)
    identity_constraints = [
        raw for raw in constraints if str(raw.get("type") or "").upper() != "RELABEL"
    ]
    semantic_constraints = [
        raw for raw in constraints if str(raw.get("type") or "").upper() == "RELABEL"
    ]
    if identity_constraints:
        candidate_before_semantics = engine.replay_global(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            constraints=identity_constraints,
        )
    else:
        candidate_before_semantics = baseline
    if semantic_constraints:
        candidate, semantic_reports = _apply_semantic_relabels(
            candidate_before_semantics, semantic_constraints
        )
    else:
        candidate = candidate_before_semantics
        semantic_reports = []
    graph = TypedDependencyGraph(provenance)
    closure_obs: set[str] = set()
    closure_entities: set[str] = set()
    closure_events: set[str] = set()
    closure_versions: set[str] = set()
    closure_edges: set[str] = set()
    starts: list[int] = []
    ends: list[int] = []
    for constraint in identity_constraints:
        anchor = str(constraint.get("applies_at_event_uid") or "")
        if anchor not in provenance.events:
            continue
        association = provenance.get_event(anchor)
        seeds = [
            str(value)
            for value in [
                association.get("target_object_version_before"),
                *(association.get("candidate_object_version_uids") or ()),
            ]
            if value and str(value) in provenance.object_versions
        ]
        closure = graph.forward_closure(
            anchor_event_uid=anchor,
            seed_version_uids=seeds,
            stop_watermark=provenance.max_sequence,
        )
        closure_obs.update(closure.obs_uids)
        closure_entities.update(closure.entity_uids)
        closure_events.update(closure.event_uids)
        closure_versions.update(closure.version_uids)
        closure_edges.update(closure.edge_uids)
        starts.append(closure.start_sequence)
        ends.append(closure.end_sequence)
    for constraint in semantic_constraints:
        entity_uid = str(constraint.get("entity_uid") or "")
        if entity_uid:
            closure_entities.add(entity_uid)
        for state in (baseline, candidate_before_semantics):
            row = _object_row(state, entity_uid)
            closure_obs.update(
                str(value) for value in (row or {}).get("member_observation_uids") or ()
            )
        anchor = str(constraint.get("applies_at_event_uid") or "")
        if anchor in provenance.events:
            closure_events.add(anchor)
            sequence = provenance.sequence(provenance.get_event(anchor))
            starts.append(sequence)
            ends.append(provenance.max_sequence)
    closure_mapping = {
        "obs_uids": sorted(closure_obs),
        "entity_uids": sorted(closure_entities),
        "event_uids": sorted(closure_events),
        "version_uids": sorted(closure_versions),
        "edge_uids": sorted(closure_edges),
        "start_sequence": min(starts, default=0),
        "end_sequence": max(ends, default=provenance.max_sequence),
    }
    verification = StructuralVerifier(provenance).verify(
        baseline_state=baseline,
        derived_state=candidate,
        closure=closure_mapping,
        expected_source_hashes=provenance.source_hashes(),
    )
    baseline_metrics = scene_health_metrics(baseline)
    candidate_metrics = scene_health_metrics(candidate)
    metric_gate = final_scene_metric_gate(
        baseline_metrics,
        candidate_metrics,
        require_partition_change=bool(identity_constraints),
    )
    semantic_gate = _semantic_relabel_gate(
        before_state=candidate_before_semantics,
        after_state=candidate,
        constraints=semantic_constraints,
        reports=semantic_reports,
    )
    requested_change_observed = bool(
        (
            identity_constraints
            and baseline_metrics["state_hash"] != candidate_metrics["state_hash"]
        )
        or (
            semantic_constraints
            and _semantic_state_hash(candidate_before_semantics)
            != _semantic_state_hash(candidate)
        )
    )
    activated = bool(
        constraints
        and verification["pass"]
        and all(metric_gate.values())
        and semantic_gate["pass"]
        and requested_change_observed
    )
    baseline_path = destination / "versions" / "v0_baseline" / "state.json"
    candidate_path = destination / "versions" / "v1_repaired" / "state.json"
    write_json(baseline_path, baseline)
    write_json(candidate_path, candidate)
    pointer = {
        "active_version": "v1_repaired" if activated else "v0_baseline",
        "state_path": str(candidate_path if activated else baseline_path),
        "previous_version": "v0_baseline" if activated else None,
        "safe_boundary": "scene_end",
        "production_global_map_mutated": False,
        "rollback_path": str(baseline_path),
        "semantic_label_pilot": bool(semantic_constraints),
        "semantic_accuracy_validated": False,
    }
    write_json(destination / "active_version.json", pointer)
    result = {
        "schema_version": "0.1.0",
        "constraint_count": len(constraints),
        "identity_constraint_count": len(identity_constraints),
        "semantic_constraint_count": len(semantic_constraints),
        "activated_repaired_version": activated,
        "verification": verification,
        "metric_gate": metric_gate,
        "semantic_gate": semantic_gate,
        "semantic_relabel_reports": semantic_reports,
        "requested_change_observed": requested_change_observed,
        "semantic_accuracy_validated": False,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_delta": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics
            if key != "state_hash"
            and isinstance(baseline_metrics[key], (int, float))
            and isinstance(candidate_metrics[key], (int, float))
        },
        "active_pointer": pointer,
    }
    write_json(destination / "final_comparison.json", result)
    return result
