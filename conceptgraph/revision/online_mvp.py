"""Minimal online revision sidecar for the ali-my-new experiment.

The mapper remains the sole writer of the active map.  This module tails its
append-only evidence ledger, commits frames with a one-frame delay, aggregates
weak scanner signals into object-group tickets, and prepares oracle-free VLM
evidence.  Shadow replay is implemented at the bottom of the file so the online
control path stays small and auditable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
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
        )

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

    def priority_key(self, current_frame: int) -> tuple[Any, ...]:
        age = max(0, int(current_frame) - self.first_seen_frame)
        return (
            -int(self.task_blocking),
            -len(self.affected_lineage_uids),
            -int(self.affected_event_count),
            -age,
            self.ticket_uid,
        )

    def as_dict(self, current_frame: int | None = None) -> dict[str, Any]:
        value = asdict(self)
        value["issues"] = [item.as_dict() for item in self.issues]
        if current_frame is not None:
            value["ticket_age_frames"] = max(0, current_frame - self.first_seen_frame)
            value["priority_tuple"] = {
                "task_blocking": self.task_blocking,
                "affected_object_count": len(self.affected_lineage_uids),
                "affected_event_count": self.affected_event_count,
                "ticket_age_frames": value["ticket_age_frames"],
            }
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
    ) -> None:
        for ticket in self.tickets.values():
            if not ticket.issues:
                continue
            root = ticket.issues[0]
            closure = tracker.closure(
                ledger=ledger,
                anchor_event_uid=root.anchor_event_uid,
                anchor_sequence=root.detected_sequence,
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

    def ordered(self, *, current_frame: int, states: Iterable[str] = ("WAIT_EVIDENCE", "READY")) -> list[ObjectTicket]:
        allowed = set(states)
        return sorted(
            (ticket for ticket in self.tickets.values() if ticket.state in allowed),
            key=lambda ticket: ticket.priority_key(current_frame),
        )


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
            history = ledger.versions_for_object.get(object_uid, ())
            previous = history[-2] if len(history) >= 2 and history[-1] is current else None
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
    baseline_metrics: Mapping[str, Any], candidate_metrics: Mapping[str, Any]
) -> dict[str, bool]:
    baseline_object_count = int(baseline_metrics["object_count"])
    minimum_object_count = max(1, math.floor(0.90 * baseline_object_count))
    return {
        "partition_changed": candidate_metrics["state_hash"] != baseline_metrics["state_hash"],
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
    """Apply disjoint accepted constraints to the complete new run, without labels."""

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
    if constraints:
        candidate = engine.replay_global(
            mode=ReplayMode.PERSISTENT_SPARSE_CONSTRAINT_REPLAY,
            constraints=constraints,
        )
    else:
        candidate = baseline
    graph = TypedDependencyGraph(provenance)
    closure_obs: set[str] = set()
    closure_entities: set[str] = set()
    closure_events: set[str] = set()
    closure_versions: set[str] = set()
    closure_edges: set[str] = set()
    starts: list[int] = []
    ends: list[int] = []
    for constraint in constraints:
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
    metric_gate = final_scene_metric_gate(baseline_metrics, candidate_metrics)
    activated = bool(
        constraints
        and verification["pass"]
        and all(metric_gate.values())
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
    }
    write_json(destination / "active_version.json", pointer)
    result = {
        "schema_version": "0.1.0",
        "constraint_count": len(constraints),
        "activated_repaired_version": activated,
        "verification": verification,
        "metric_gate": metric_gate,
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
