from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import CorruptionPlan


def _object_uid(obj: dict[str, Any]) -> str:
    return str(obj.get("id", obj.get("object_uid", "")))


def _obs_key(obs_uid: str) -> str:
    value = str(obs_uid)
    marker = value.rfind("_f")
    return value[marker:] if marker >= 0 else value


def _same_observation(first: str, second: str) -> bool:
    return str(first) == str(second) or _obs_key(first) == _obs_key(second)


def _object_has_observation(obj: dict[str, Any], obs_uid: str | None) -> bool:
    if not obs_uid:
        return False
    return any(
        _same_observation(str(member), obs_uid)
        for member in obj.get("obs_uids", ())
    )


def _object_positions(
    objects: list[dict[str, Any]], object_uid: str | None, origin_obs_uid: str | None
) -> list[int]:
    exact = [index for index, obj in enumerate(objects) if _object_uid(obj) == object_uid]
    if exact:
        return exact
    return [
        index for index, obj in enumerate(objects) if _object_has_observation(obj, origin_obs_uid)
    ]


def load_corruption_plan(path: str | Path) -> CorruptionPlan:
    with Path(path).open(encoding="utf-8") as handle:
        return CorruptionPlan.from_mapping(json.load(handle))


class ControlledCorruptionController:
    """Apply one manifest-bound intervention at the live association boundary."""

    def __init__(
        self,
        plan: CorruptionPlan,
        *,
        output_dir: str | Path | None = None,
        require_exactly_once: bool = True,
    ) -> None:
        self.plan = plan
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.require_exactly_once = require_exactly_once
        self.applied_count = 0
        self.records: list[dict[str, Any]] = []

    def apply(
        self,
        *,
        frame_idx: int,
        detection_list: Iterable[dict[str, Any]],
        objects: Iterable[dict[str, Any]],
        original_match_indices: list[int | None],
    ) -> list[int | None]:
        matches = list(original_match_indices)
        if int(frame_idx) != self.plan.frame_idx:
            return matches
        detections = list(detection_list)
        object_list = list(objects)
        positions = [
            index
            for index, detection in enumerate(detections)
            if any(
                _same_observation(str(item), self.plan.obs_uid)
                for item in detection.get("obs_uids", ())
            )
        ]
        if not positions:
            return matches
        if len(positions) != 1:
            raise RuntimeError(f"corruption observation matched {len(positions)} detections")
        if self.applied_count:
            raise RuntimeError(f"corruption {self.plan.case_uid} attempted more than once")

        detection_index = positions[0]
        original = matches[detection_index]
        original_uid = (
            _object_uid(object_list[original])
            if original is not None and 0 <= original < len(object_list)
            else None
        )
        source_matches = (
            original_uid == self.plan.source_object_uid
            or (
                original is not None
                and 0 <= original < len(object_list)
                and _object_has_observation(
                    object_list[original], self.plan.source_origin_obs_uid
                )
            )
        )
        if self.plan.source_object_uid and not source_matches:
            raise RuntimeError(
                f"corruption source drift: expected {self.plan.source_object_uid}, got {original_uid}"
            )

        if self.plan.corruption_type == "FORCE_CREATE":
            matches[detection_index] = None
            corrupted_uid = None
        elif self.plan.corruption_type == "FORCE_ASSOCIATE":
            target_positions = _object_positions(
                object_list,
                self.plan.target_object_uid,
                self.plan.target_origin_obs_uid,
            )
            if len(target_positions) != 1:
                raise RuntimeError(
                    f"target object {self.plan.target_object_uid} is not uniquely active"
                )
            matches[detection_index] = target_positions[0]
            corrupted_uid = _object_uid(object_list[target_positions[0]])
        else:
            raise RuntimeError(
                "FORCE_POSTPROCESS_MERGE must be applied by apply_postprocess_merge"
            )

        self.applied_count += 1
        record = {
            "case_uid": self.plan.case_uid,
            "event_type": "CORRUPTION_INJECTED",
            "corruption_type": self.plan.corruption_type,
            "frame_idx": int(frame_idx),
            "planned_obs_uid": self.plan.obs_uid,
            "obs_uid": str(detections[detection_index].get("obs_uids", [self.plan.obs_uid])[0]),
            "original_decision": {
                "match_index": original,
                "target_object_uid": original_uid,
            },
            "corrupted_decision": {
                "match_index": matches[detection_index],
                "target_object_uid": corrupted_uid,
            },
            "seed": self.plan.seed,
        }
        self.records.append(record)
        self._write_record(record)
        return matches

    def apply_postprocess_merge(
        self, *, frame_idx: int, objects: Iterable[dict[str, Any]]
    ) -> tuple[int, int] | None:
        if self.plan.corruption_type != "FORCE_POSTPROCESS_MERGE":
            return None
        if int(frame_idx) != self.plan.frame_idx:
            return None
        if self.applied_count:
            raise RuntimeError(f"corruption {self.plan.case_uid} attempted more than once")
        object_list = list(objects)
        by_uid = {_object_uid(obj): index for index, obj in enumerate(object_list)}
        source = by_uid.get(str(self.plan.source_object_uid))
        target = by_uid.get(str(self.plan.target_object_uid))
        if source is None or target is None or source == target:
            raise RuntimeError("postprocess corruption endpoints are not active and distinct")
        self.applied_count += 1
        record = {
            "case_uid": self.plan.case_uid,
            "event_type": "CORRUPTION_INJECTED",
            "corruption_type": self.plan.corruption_type,
            "frame_idx": int(frame_idx),
            "obs_uid": self.plan.obs_uid,
            "original_decision": {"merge": False},
            "corrupted_decision": {
                "merge": True,
                "source_object_uid": self.plan.source_object_uid,
                "target_object_uid": self.plan.target_object_uid,
            },
            "seed": self.plan.seed,
        }
        self.records.append(record)
        self._write_record(record)
        return source, target

    def finalize(self) -> None:
        if self.require_exactly_once and self.applied_count != 1:
            raise RuntimeError(
                f"corruption {self.plan.case_uid} applied {self.applied_count} times; expected 1"
            )

    def _write_record(self, record: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "corruption_events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
