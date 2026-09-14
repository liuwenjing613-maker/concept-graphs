"""Read-only evidence readiness, mapping invariants, and stage-one audit rules."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


SUPPORTED_SCHEMAS = {"0.2.0"}
FRAME_RE = re.compile(r"_f(-?\d+)")
LEGACY_REF_RE = re.compile(
    r"^(?P<path>.+?)#(?P<key>[^\[]+)(?:\[(?P<index>\d+)\])?$"
)
CLOSED_STATUSES = {
    "completed",
    "early_exit",
    "MAP_COMPLETED_EVIDENCE_VALID",
    "MAP_COMPLETED_EVIDENCE_INVALID",
}
_ARRAY_CACHE: dict[tuple[str, str], Any] = {}
_TREE_CACHE: dict[int, Any] = {}
_HASH_CACHE: dict[str, str] = {}


def _load_json(path: Path, errors: list[str], default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path.name}:{type(exc).__name__}:{exc}")
        return default


def _load_jsonl(path: Path, errors: list[str]) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("record is not an object")
                records.append(value)
            except Exception as exc:
                errors.append(
                    f"{path.name}:{line_no}:{type(exc).__name__}:{exc}"
                )
    return records


def _sha256(path: Path) -> str:
    cache_key = str(path.resolve())
    if cache_key in _HASH_CACHE:
        return _HASH_CACHE[cache_key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _HASH_CACHE[cache_key] = value
    return value


@lru_cache(maxsize=64)
def _load_container(path_string: str, fmt: str, key: Optional[str]) -> Any:
    path = Path(path_string)
    if fmt in {"pickle.gz", "pkl.gz"} or path.name.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return loaded
    selected_key = key or (loaded.files[0] if loaded.files else None)
    if selected_key is None:
        loaded.close()
        return None
    value = loaded[selected_key]
    loaded.close()
    return value


def _resolve_path(path_value: Any, evidence_dir: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (evidence_dir.parent / path).resolve()


def _parse_ref(ref: Any) -> Optional[dict]:
    if not ref:
        return None
    if isinstance(ref, dict):
        return dict(ref)
    if not isinstance(ref, str):
        return None
    match = LEGACY_REF_RE.match(ref)
    if not match:
        return {"path": ref, "format": Path(ref).suffix.lstrip(".") or "file"}
    return {
        "path": match.group("path"),
        "format": Path(match.group("path")).suffix.lstrip("."),
        "key": match.group("key"),
        "index": int(match.group("index")) if match.group("index") else None,
    }


def _load_ref_value(ref: Any, evidence_dir: Path) -> Any:
    parsed = _parse_ref(ref)
    if not parsed or not parsed.get("path"):
        return None
    path = _resolve_path(parsed["path"], evidence_dir)
    fmt = str(parsed.get("format") or "").lower()
    value = _load_container(str(path), fmt, parsed.get("key"))
    if parsed.get("index") is not None:
        value = value[int(parsed["index"])]
    return value


def _validate_ref(ref: Any, evidence_dir: Path) -> list[dict]:
    parsed = _parse_ref(ref)
    if parsed is None:
        return []
    if not parsed.get("path"):
        return [{"kind": "missing_path", "ref": parsed}]
    path = _resolve_path(parsed["path"], evidence_dir)
    if not path.exists():
        return [{"kind": "missing_file", "path": str(path), "ref": parsed}]
    issues: list[dict] = []
    if parsed.get("sha256"):
        actual = _sha256(path)
        if actual != parsed["sha256"]:
            issues.append(
                {
                    "kind": "sha256_mismatch",
                    "path": str(path),
                    "expected": parsed["sha256"],
                    "actual": actual,
                }
            )
    fmt = str(parsed.get("format") or "").lower()
    is_array_or_pickle = (
        fmt in {"npz", "npy", "pickle.gz", "pkl.gz"}
        or path.suffix.lower() in {".npz", ".npy"}
        or path.name.endswith(".pkl.gz")
    )
    if not is_array_or_pickle:
        return issues
    try:
        value = _load_ref_value(parsed, evidence_dir)
        if value is not None and hasattr(value, "shape"):
            array = np.asarray(value)
            if parsed.get("shape") is not None and list(array.shape) != list(
                parsed["shape"]
            ):
                issues.append(
                    {
                        "kind": "shape_mismatch",
                        "path": str(path),
                        "expected": parsed["shape"],
                        "actual": list(array.shape),
                    }
                )
            if parsed.get("dtype") is not None and str(array.dtype) != str(
                parsed["dtype"]
            ):
                issues.append(
                    {
                        "kind": "dtype_mismatch",
                        "path": str(path),
                        "expected": parsed["dtype"],
                        "actual": str(array.dtype),
                    }
                )
    except KeyError as exc:
        issues.append({"kind": "missing_key", "path": str(path), "key": str(exc)})
    except IndexError as exc:
        issues.append(
            {"kind": "index_out_of_range", "path": str(path), "error": str(exc)}
        )
    except Exception as exc:
        issues.append(
            {
                "kind": "unreadable_artifact",
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return issues


def _walk_refs(value: Any, label: str = "root"):
    if isinstance(value, dict):
        if value.get("path") is not None and value.get("format") is not None:
            yield label, value
            return
        for key, item in value.items():
            child = f"{label}.{key}"
            if key.endswith("_ref") and item:
                yield child, item
            elif isinstance(item, (dict, list)):
                yield from _walk_refs(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_refs(item, f"{label}[{index}]")


def _frame_idx(value: Any) -> Optional[int]:
    match = FRAME_RE.search(str(value))
    return int(match.group(1)) if match else None


def _same(a: Any, b: Any, atol: float = 1e-5) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), atol=atol, rtol=atol))
    except Exception:
        return a == b


def _finite_vec(value: Any, size: int = 3) -> bool:
    try:
        array = np.asarray(value, dtype=float)
        return array.size == size and bool(np.isfinite(array).all())
    except Exception:
        return False


class _Findings:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.items: list[dict] = []
        self._counter = 0

    def add(
        self,
        rule_ids: str | list[str],
        error_type: str,
        certainty: str,
        *,
        severity: str = "HIGH",
        scope: Optional[dict] = None,
        metrics: Optional[dict] = None,
        evidence_refs: Optional[dict] = None,
        evidence_groups: Optional[list[str]] = None,
        vetoes: Optional[list[str]] = None,
        triage: str = "DIRECT_CODE_FIX",
        recommended_action: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        self._counter += 1
        self.items.append(
            {
                "schema_version": "0.2.0",
                "finding_uid": f"finding_{self._counter:06d}",
                "run_id": self.run_id,
                "rule_ids": [rule_ids] if isinstance(rule_ids, str) else rule_ids,
                "error_type": error_type,
                "certainty": certainty,
                "severity": severity,
                "scope": scope or {},
                "metrics": metrics or {},
                "independent_evidence_groups": evidence_groups or [],
                "vetoes_triggered": vetoes or [],
                "evidence_refs": evidence_refs or {},
                "triage": triage,
                "recommended_action": recommended_action,
                "action_executed": False,
                "message": message,
            }
        )


def _array_from_ref(ref: Any, evidence_dir: Path, dtype=float) -> Optional[np.ndarray]:
    try:
        cache_key = (
            str(evidence_dir),
            json.dumps(_parse_ref(ref), sort_keys=True),
            str(np.dtype(dtype)),
        )
        if cache_key not in _ARRAY_CACHE:
            value = _load_ref_value(ref, evidence_dir)
            _ARRAY_CACHE[cache_key] = (
                None if value is None else np.asarray(value, dtype=dtype)
            )
        value = _ARRAY_CACHE[cache_key]
        return value
    except Exception:
        return None


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.asarray(b, dtype=float).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def _bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _mask_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.shape != b.shape:
        return 0.0, 0.0
    intersection = int(np.logical_and(a, b).sum())
    area_a, area_b = int(a.sum()), int(b.sum())
    union = area_a + area_b - intersection
    return (
        intersection / union if union else 0.0,
        intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0,
    )


def _normalised_center_distance(a: dict, b: dict, prefix: str = "bbox_3d") -> Optional[float]:
    center_a, center_b = a.get(f"{prefix}_center"), b.get(f"{prefix}_center")
    extent_a, extent_b = a.get(f"{prefix}_extent"), b.get(f"{prefix}_extent")
    if not all(_finite_vec(value) for value in (center_a, center_b, extent_a, extent_b)):
        return None
    denominator = 0.5 * (
        np.linalg.norm(np.asarray(extent_a, dtype=float))
        + np.linalg.norm(np.asarray(extent_b, dtype=float))
    )
    return float(
        np.linalg.norm(np.asarray(center_a, dtype=float) - np.asarray(center_b, dtype=float))
        / max(float(denominator), 1e-9)
    )


def _pcd_overlap(a: Optional[np.ndarray], b: Optional[np.ndarray], radius: float) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    try:
        from scipy.spatial import cKDTree

        tree = _TREE_CACHE.get(id(right))
        if tree is None:
            tree = cKDTree(right)
            _TREE_CACHE[id(right)] = tree
        distances, _ = tree.query(left, k=1, workers=-1)
        return float(np.mean(distances < radius))
    except Exception:
        supported = 0
        for start in range(0, len(left), 512):
            chunk = left[start : start + 512]
            distances = np.sum((chunk[:, None, :] - right[None, :, :]) ** 2, axis=2)
            supported += int(np.any(distances < radius * radius, axis=1).sum())
        return float(supported / len(left))


def _symmetric_overlap(a: Optional[np.ndarray], b: Optional[np.ndarray], radius: float) -> float:
    return min(_pcd_overlap(a, b, radius), _pcd_overlap(b, a, radius))


def _label_compatible(a: Any, b: Any) -> bool:
    left, right = str(a or "").strip().lower(), str(b or "").strip().lower()
    aliases = {
        "couch": "sofa",
        "sofa chair": "sofa",
        "armchair": "chair",
        "television": "tv",
    }
    left, right = aliases.get(left, left), aliases.get(right, right)
    return bool(left and right and (left == right or left in right or right in left))


def _event_replay(events: list[dict]) -> tuple[dict[str, list[str]], dict[str, bool]]:
    members: dict[str, list[str]] = defaultdict(list)
    active: dict[str, bool] = defaultdict(lambda: False)
    for event in sorted(events, key=lambda item: int(item.get("event_sequence") or 0)):
        event_type = event.get("event_type")
        uid, obs_uid = event.get("object_uid"), event.get("obs_uid")
        if event_type == "OBJECT_CREATE" and uid:
            members[uid] = [str(obs_uid)] if obs_uid else []
            active[uid] = True
        elif event_type == "OBS_ASSOCIATE" and uid and obs_uid:
            members[uid].append(str(obs_uid))
            active[uid] = True
        elif event_type == "OBJECT_FILTER" and uid:
            active[uid] = False
        elif event_type == "OBJECT_MERGE":
            source, target = event.get("source_object_uid"), event.get("target_object_uid")
            source_before = event.get("source_before") or event.get("before_summary") or {}
            target_before = event.get("target_before") or {}
            left = list(source_before.get("member_observation_uids", members[source]))
            right = list(target_before.get("member_observation_uids", members[target]))
            members[target] = list(dict.fromkeys(right + left))
            active[target], active[source] = True, False
    return members, active


def _audit_readiness(data: dict, findings: _Findings) -> None:
    evidence_dir, manifest = data["evidence_dir"], data["manifest"]
    missing_files = [
        name
        for name in (
            "manifest.json",
            "frames.jsonl",
            "observations.jsonl",
            "associations.jsonl",
            "mapping_events.jsonl",
            "filter_trace.jsonl",
            "object_versions.jsonl",
            "object_pair_decisions.jsonl",
            "final_membership.json",
        )
        if not (evidence_dir / name).exists()
    ]
    required_manifest = [
        "run_id",
        "scene_id",
        "status",
        "branch",
        "git_commit",
        "mapping_config_ref",
        "detection_config_ref",
        "runtime",
        "evidence_mode",
    ]
    missing_fields = [key for key in required_manifest if not manifest.get(key)]
    if manifest.get("schema_version") not in SUPPORTED_SCHEMAS or missing_files or missing_fields:
        findings.add(
            "EVI-001",
            "EVIDENCE_SCHEMA_OR_MANIFEST_INVALID",
            "CERTAIN",
            metrics={
                "schema_version": manifest.get("schema_version"),
                "missing_files": missing_files,
                "missing_fields": missing_fields,
            },
        )
    if manifest.get("status") not in CLOSED_STATUSES:
        findings.add(
            "EVI-001",
            "EVIDENCE_RUN_NOT_CLOSED",
            "CERTAIN",
            metrics={"status": manifest.get("status")},
        )
    if data["errors"]:
        findings.add(
            "EVI-001",
            "EVIDENCE_JSON_PARSE_ERROR",
            "CERTAIN",
            metrics={"errors": data["errors"]},
        )

    uid_groups = {
        "frame_uid": [item.get("frame_uid") for item in data["frames"]],
        "obs_uid": [item.get("obs_uid") for item in data["observations"]],
        "event_uid": [
            item.get("event_uid")
            for item in data["associations"] + data["events"] + data["vlm_events"]
        ],
        "object_version_uid": [
            item.get("object_version_uid") for item in data["versions"]
        ],
    }
    for field, values in uid_groups.items():
        duplicates = sorted(
            value for value, count in Counter(values).items() if value and count > 1
        )
        missing_count = sum(value is None for value in values)
        if duplicates or missing_count:
            findings.add(
                "EVI-002",
                "UID_INVALID",
                "CERTAIN",
                scope={field: duplicates},
                metrics={"missing_count": missing_count},
            )

    invalid_refs = []
    for source_name in (
        "manifest",
        "frames",
        "observations",
        "associations",
        "versions",
        "vlm_events",
    ):
        for label, ref in _walk_refs(data[source_name], source_name):
            issues = _validate_ref(ref, evidence_dir)
            if issues:
                invalid_refs.append({"field": label, "issues": issues})
    if invalid_refs:
        findings.add(
            "EVI-003",
            "ARTIFACT_REFERENCE_INVALID",
            "CERTAIN",
            metrics={"invalid_references": invalid_refs},
        )

    traces = {
        (item.get("frame_uid"), int(item.get("raw_det_idx", -1))): item
        for item in data["filter_trace"]
    }
    trace_issues = []
    for obs in data["observations"]:
        key = (obs.get("frame_uid"), int(obs.get("raw_det_idx", -1)))
        trace = traces.get(key)
        expected = "KEEP" if obs.get("status") == "kept" else "REJECT"
        if trace is None or trace.get("decision") != expected:
            trace_issues.append(
                {
                    "obs_uid": obs.get("obs_uid"),
                    "expected": expected,
                    "actual": None if trace is None else trace.get("decision"),
                }
            )
        if obs.get("status") == "kept":
            if not obs.get("processed_mask_ref") or not obs.get("pcd_ref"):
                trace_issues.append(
                    {"obs_uid": obs.get("obs_uid"), "missing": "processed_mask_or_pcd"}
                )
            if obs.get("pcd_is_sampled"):
                trace_issues.append(
                    {"obs_uid": obs.get("obs_uid"), "invalid": "sampled_pcd"}
                )
            mask = _array_from_ref(obs.get("processed_mask_ref"), evidence_dir, bool)
            if mask is not None and int(mask.sum()) != int(obs.get("processed_mask_area", -1)):
                trace_issues.append(
                    {"obs_uid": obs.get("obs_uid"), "invalid": "processed_mask_area"}
                )
            points = _array_from_ref(obs.get("pcd_ref"), evidence_dir)
            if points is not None and len(points) != int(obs.get("pcd_stored_points", -1)):
                trace_issues.append(
                    {"obs_uid": obs.get("obs_uid"), "invalid": "pcd_stored_points"}
                )
    if trace_issues:
        findings.add(
            "EVI-005",
            "OBSERVATION_OR_FILTER_TRACE_INVALID",
            "CERTAIN",
            metrics={"issues": trace_issues},
        )

    by_frame = defaultdict(list)
    for assoc in data["associations"]:
        by_frame[assoc.get("frame_uid")].append(assoc)
    for frame_uid, records in by_frame.items():
        index = _frame_idx(frame_uid)
        path = evidence_dir / "similarities" / f"frame_{index:06d}.npz"
        if index is None or not path.exists():
            findings.add(
                "EVI-004", "SIMILARITY_MATRIX_MISSING", "CERTAIN", scope={"frame_uid": frame_uid}
            )
            continue
        try:
            with np.load(path, allow_pickle=False) as matrix:
                obs_axis = [str(value) for value in matrix["observation_uids"].tolist()]
                obj_axis = [str(value) for value in matrix["object_uids"].tolist()]
                expected = (len(obs_axis), len(obj_axis))
                invalid = []
                invalid_statuses = [
                    item.get("similarity_validation")
                    for item in records
                    if item.get("similarity_evidence_valid") is False
                ]
                if invalid_statuses:
                    invalid.append(
                        {
                            "key": "similarity_evidence_valid",
                            "statuses": invalid_statuses,
                        }
                    )
                for key in ("spatial_sim", "visual_sim", "aggregate_sim"):
                    array = np.asarray(matrix[key])
                    if array.shape != expected or not np.isfinite(array).all():
                        invalid.append({"key": key, "shape": list(array.shape)})
                if [item.get("obs_uid") for item in records] != obs_axis:
                    invalid.append({"key": "observation_axis"})
                if any(item.get("object_uids_before") != obj_axis for item in records):
                    invalid.append({"key": "object_axis"})
                if invalid:
                    findings.add(
                        "EVI-004",
                        "SIMILARITY_MATRIX_INVALID",
                        "CERTAIN",
                        scope={"frame_uid": frame_uid},
                        metrics={"issues": invalid, "expected_shape": list(expected)},
                    )
        except Exception as exc:
            findings.add(
                "EVI-004",
                "SIMILARITY_MATRIX_UNREADABLE",
                "CERTAIN",
                scope={"frame_uid": frame_uid},
                message=f"{type(exc).__name__}: {exc}",
            )

    event_ids = {
        item.get("event_uid") for item in data["associations"] + data["events"]
    }
    version_ids = {item.get("object_version_uid") for item in data["versions"]}
    versions_by_object = defaultdict(list)
    version_issues = []
    for version in data["versions"]:
        versions_by_object[version.get("object_uid")].append(version)
        if version.get("trigger_event_uid") not in event_ids:
            version_issues.append(
                {"object_version_uid": version.get("object_version_uid"), "invalid": "trigger_event"}
            )
        for parent in version.get("parent_version_uids", []):
            if parent not in version_ids:
                version_issues.append(
                    {"object_version_uid": version.get("object_version_uid"), "missing_parent": parent}
                )
    final_by_uid = {item.get("object_uid"): item for item in data["final_membership"]}
    for uid, versions in versions_by_object.items():
        ordered = sorted(versions, key=lambda item: int(item.get("version", 0)))
        actual = [int(item.get("version", 0)) for item in ordered]
        if actual != list(range(1, len(actual) + 1)):
            version_issues.append({"object_uid": uid, "invalid": "non_contiguous", "actual": actual})
        current = ordered[-1]
        if uid in final_by_uid:
            if current.get("status") != "active" or set(current.get("member_observation_uids", [])) != set(
                final_by_uid[uid].get("member_observation_uids", [])
            ):
                version_issues.append({"object_uid": uid, "invalid": "final_version_mismatch"})
        elif current.get("status") == "active":
            version_issues.append({"object_uid": uid, "invalid": "orphan_active_version"})
    if set(final_by_uid) - set(versions_by_object):
        version_issues.append(
            {"missing_final_object_versions": sorted(set(final_by_uid) - set(versions_by_object))}
        )
    for event in data["events"]:
        for version_uid in event.get("output_object_version_uids", []):
            if version_uid not in version_ids:
                version_issues.append(
                    {"event_uid": event.get("event_uid"), "missing_output_version": version_uid}
                )
    if version_issues:
        findings.add(
            "EVI-006",
            "OBJECT_VERSION_CHAIN_INVALID",
            "CERTAIN",
            metrics={"issues": version_issues},
        )

    replayed, active = _event_replay(data["events"])
    replay_issues = []
    ownership = defaultdict(list)
    for uid, item in final_by_uid.items():
        members = list(item.get("member_observation_uids", []))
        if not active.get(uid) or set(replayed.get(uid, [])) != set(members):
            replay_issues.append(
                {"object_uid": uid, "replayed": replayed.get(uid, []), "final": members}
            )
        for obs_uid in members:
            ownership[obs_uid].append(uid)
    kept_uids = {
        item.get("obs_uid") for item in data["observations"] if item.get("status") == "kept"
    }
    tombstoned = {
        obs_uid
        for event in data["events"]
        if event.get("event_type") in {"OBJECT_FILTER", "OBJECT_DELETE", "OBS_INVALID"}
        for obs_uid in (event.get("before_summary") or {}).get("member_observation_uids", [])
    }
    tombstoned.update(
        event.get("obs_uid")
        for event in data["events"]
        if event.get("event_type") == "OBS_DISCARD" and event.get("obs_uid")
    )
    for obs_uid in sorted(kept_uids):
        if len(ownership.get(obs_uid, [])) != 1 and obs_uid not in tombstoned:
            replay_issues.append({"obs_uid": obs_uid, "owners": ownership.get(obs_uid, [])})
    if replay_issues:
        findings.add(
            "EVI-007",
            "FINAL_MEMBERSHIP_NOT_REPLAYABLE",
            "CERTAIN",
            metrics={"issues": replay_issues},
        )

    parent_graph = {
        item.get("event_uid"): list(item.get("parent_event_uids", []))
        for item in data["events"]
    }
    cycle = False
    visiting, visited = set(), set()

    def visit(uid):
        nonlocal cycle
        if uid in visiting:
            cycle = True
            return
        if uid in visited:
            return
        visiting.add(uid)
        for parent in parent_graph.get(uid, []):
            visit(parent)
        visiting.remove(uid)
        visited.add(uid)

    for uid in parent_graph:
        visit(uid)
    if cycle:
        findings.add("EVI-006", "EVENT_GRAPH_CYCLE", "CERTAIN")

    if manifest.get("make_edges"):
        if not data["vlm_events"]:
            findings.add("EVI-008", "VLM_EVIDENCE_MISSING", "CERTAIN")
        for item in data["vlm_events"]:
            missing = [
                key
                for key in (
                    "prompt_text",
                    "image_inputs",
                    "model_name",
                    "generation_params",
                    "raw_response",
                    "parsed_output",
                    "parser_version",
                )
                if item.get(key) is None
            ]
            if missing:
                findings.add(
                    "EVI-008",
                    "VLM_EVIDENCE_INCOMPLETE",
                    "CERTAIN",
                    scope={"event_uid": item.get("event_uid")},
                    metrics={"missing": missing},
                )


def _audit_mapping_invariants(data: dict, findings: _Findings) -> None:
    evidence_dir = data["evidence_dir"]
    active: dict[str, bool] = defaultdict(lambda: False)
    merge_accepts = Counter()
    accepted_pairs = {
        (
            item.get("merge_transaction_uid"),
            item.get("source_object_uid"),
            item.get("target_object_uid"),
        )
        for item in data["pair_decisions"]
        if item.get("decision") == "ACCEPT"
    }
    for event in sorted(data["events"], key=lambda item: int(item.get("event_sequence") or 0)):
        event_type = event.get("event_type")
        if event_type == "OBJECT_CREATE":
            active[event.get("object_uid")] = True
        elif event_type == "OBJECT_FILTER":
            active[event.get("object_uid")] = False
        elif event_type == "OBJECT_MERGE":
            source, target = event.get("source_object_uid"), event.get("target_object_uid")
            tx_uid = event.get("merge_transaction_uid") or event.get("transaction_uid")
            if source == target or not active.get(source, False) or not active.get(target, False):
                findings.add(
                    "MAP-007",
                    "ILLEGAL_MERGE_GRAPH",
                    "CERTAIN",
                    scope={"event_uid": event.get("event_uid"), "source_object_uid": source, "target_object_uid": target},
                )
            source_members = list((event.get("source_before") or {}).get("member_observation_uids", []))
            target_members = list((event.get("target_before") or {}).get("member_observation_uids", []))
            after_members = list((event.get("target_after") or {}).get("member_observation_uids", []))
            intersection = sorted(set(source_members) & set(target_members))
            if intersection:
                findings.add(
                    "MAP-006",
                    "MERGE_MEMBER_INTERSECTION",
                    "CERTAIN",
                    scope={"event_uid": event.get("event_uid")},
                    metrics={"intersection": intersection},
                )
            expected = set(source_members) | set(target_members)
            if set(after_members) != expected:
                findings.add(
                    "MAP-008",
                    "MERGE_MEMBER_UNION_MISMATCH",
                    "CERTAIN",
                    scope={"event_uid": event.get("event_uid")},
                    metrics={"expected": sorted(expected), "actual": after_members},
                )
            merge_accepts[(tx_uid, source)] += 1
            if (tx_uid, source, target) not in accepted_pairs:
                findings.add(
                    "MAP-008",
                    "MERGE_WITHOUT_ACCEPTED_CANDIDATE",
                    "CERTAIN",
                    scope={"event_uid": event.get("event_uid")},
                )
            active[source], active[target] = False, True
    for (tx_uid, source), count in merge_accepts.items():
        if count > 1:
            findings.add(
                "MAP-005",
                "MERGE_SOURCE_REUSED",
                "CERTAIN",
                scope={"merge_transaction_uid": tx_uid, "source_object_uid": source},
                metrics={"accept_count": count},
            )

    ownership = defaultdict(list)
    for item in data["final_membership"]:
        uid = item.get("object_uid")
        members = list(item.get("member_observation_uids", []))
        duplicates = {key: value for key, value in Counter(members).items() if value > 1}
        duplicates.update(item.get("duplicate_member_observation_uids") or {})
        if duplicates:
            findings.add(
                "MAP-003",
                "DUPLICATE_ACTIVE_OWNERSHIP",
                "CERTAIN",
                scope={"object_uid": uid},
                metrics={"duplicates": duplicates},
            )
        if int(item.get("num_detections", len(members))) != len(members) or len(members) != len(set(members)):
            findings.add(
                "MAP-004",
                "NUM_DETECTIONS_MISMATCH",
                "CERTAIN",
                scope={"object_uid": uid},
                metrics={"num_detections": item.get("num_detections"), "member_count": len(members), "unique_member_count": len(set(members))},
            )
        for obs_uid in set(members):
            ownership[obs_uid].append(uid)
    for obs_uid, owners in ownership.items():
        if len(owners) > 1:
            findings.add(
                "MAP-003",
                "DUPLICATE_ACTIVE_OWNERSHIP",
                "CERTAIN",
                scope={"obs_uid": obs_uid},
                metrics={"object_uids": owners},
            )

    for assoc in data["associations"]:
        index = _frame_idx(assoc.get("frame_uid"))
        if index is None:
            continue
        path = evidence_dir / "similarities" / f"frame_{index:06d}.npz"
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as matrix:
                obs_axis = [str(value) for value in matrix["observation_uids"].tolist()]
                obj_axis = [str(value) for value in matrix["object_uids"].tolist()]
                row = np.asarray(matrix["aggregate_sim"])[obs_axis.index(str(assoc.get("obs_uid")))]
            order = np.argsort(row)[::-1]
            best_idx = int(np.argmax(row)) if len(row) else None
            best_score = float(row[best_idx]) if best_idx is not None else None
            threshold = assoc.get("sim_threshold")
            if assoc.get("decision") == "DISCARD_OBSERVATION":
                if (
                    assoc.get("decision_override") != "blocking_gate_quality_discard"
                    or assoc.get("target_object_uid") is not None
                    or assoc.get("target_object_version_after") is not None
                ):
                    findings.add(
                        "MAP-001",
                        "INVALID_DISCARD_OVERRIDE",
                        "CERTAIN",
                        scope={"event_uid": assoc.get("event_uid")},
                    )
                # A quality discard is an explicit, evidenced gate override of
                # the score-based baseline, not an association mismatch.
                continue
            expected_decision = "CREATE_OBJECT" if best_score is None or best_score <= float(threshold) else "MERGE_TO_OBJECT"
            if assoc.get("decision") != expected_decision:
                findings.add(
                    "MAP-001",
                    "ASSOCIATION_DECISION_MISMATCH",
                    "CERTAIN",
                    scope={"event_uid": assoc.get("event_uid")},
                    metrics={"recorded": assoc.get("decision"), "expected": expected_decision, "best_score": best_score, "threshold": threshold},
                )
            if expected_decision == "MERGE_TO_OBJECT" and assoc.get("target_object_uid") != obj_axis[best_idx]:
                findings.add(
                    "MAP-001",
                    "ASSOCIATION_TARGET_NOT_ARGMAX",
                    "CERTAIN",
                    scope={"event_uid": assoc.get("event_uid")},
                    metrics={"recorded": assoc.get("target_object_uid"), "expected": obj_axis[best_idx]},
                )
            recorded = [item.get("object_uid") for item in assoc.get("top_candidates", [])]
            expected = [obj_axis[int(value)] for value in order[: len(recorded)]]
            top1 = float(row[order[0]]) if len(order) else None
            top2 = float(row[order[1]]) if len(order) > 1 else None
            margin = top1 - top2 if top2 is not None else None
            # Equal scores have no unique Top-K ordering.  Validate the score
            # boundary and monotonicity instead of treating a legal tie order
            # as a ledger error.
            recorded_indices = [obj_axis.index(uid) for uid in recorded]
            recorded_scores = [float(row[index]) for index in recorded_indices]
            cutoff = float(row[order[len(recorded) - 1]]) if recorded else None
            order_invalid = any(
                left + 1e-5 < right
                for left, right in zip(recorded_scores, recorded_scores[1:])
            )
            boundary_invalid = bool(
                recorded_scores
                and cutoff is not None
                and min(recorded_scores) + 1e-5 < cutoff
            )
            margin_invalid = bool(
                margin is not None
                and assoc.get("margin") is not None
                and not _same(margin, assoc.get("margin"))
            )
            if order_invalid or boundary_invalid or margin_invalid:
                findings.add(
                    "MAP-002",
                    "TOP_K_OR_MARGIN_MISMATCH",
                    "CERTAIN",
                    scope={"event_uid": assoc.get("event_uid")},
                    metrics={"recorded_order": recorded, "expected_order_tie_example": expected, "recorded_scores": recorded_scores, "top_k_cutoff": cutoff, "recorded_margin": assoc.get("margin"), "expected_margin": margin},
                )
        except Exception as exc:
            findings.add(
                "MAP-001",
                "ASSOCIATION_MATRIX_LOOKUP_FAILED",
                "CERTAIN",
                scope={"event_uid": assoc.get("event_uid")},
                message=f"{type(exc).__name__}: {exc}",
            )

    final_uids = {item.get("object_uid") for item in data["final_membership"]}
    for event in data["events"]:
        if str(event.get("event_type", "")).startswith("EDGE_"):
            source, target = event.get("source_object_uid"), event.get("target_object_uid")
            if event.get("event_type") != "EDGE_REMOVE" and (source not in final_uids or target not in final_uids):
                findings.add(
                    "MAP-009",
                    "EDGE_POINTS_TO_INACTIVE_OBJECT",
                    "CERTAIN",
                    scope={"event_uid": event.get("event_uid"), "source_object_uid": source, "target_object_uid": target},
                )


def _object_members(item: dict, obs_lookup: dict[str, dict]) -> list[dict]:
    return [obs_lookup[uid] for uid in item.get("member_observation_uids", []) if uid in obs_lookup]


def _object_feature(item: dict, obs_lookup: dict[str, dict], evidence_dir: Path) -> Optional[np.ndarray]:
    features = [
        _array_from_ref(obs.get("image_feat_ref"), evidence_dir)
        for obs in _object_members(item, obs_lookup)
    ]
    features = [value.reshape(-1) for value in features if value is not None and value.size]
    if not features:
        return None
    matrix = np.stack(features)
    scores = matrix @ matrix.T
    norms = np.linalg.norm(matrix, axis=1)
    scores = scores / np.maximum(norms[:, None] * norms[None, :], 1e-9)
    return matrix[int(np.argmax(scores.sum(axis=1)))]


def _object_pcd(item: dict, obs_lookup: dict[str, dict], evidence_dir: Path) -> Optional[np.ndarray]:
    arrays = [
        _array_from_ref(obs.get("pcd_ref"), evidence_dir)
        for obs in _object_members(item, obs_lookup)
    ]
    arrays = [value for value in arrays if value is not None and value.ndim == 2 and value.shape[1] == 3]
    return np.concatenate(arrays) if arrays else None


def _audit_semantic_rules(data: dict, findings: _Findings) -> None:
    evidence_dir = data["evidence_dir"]
    obs_lookup = {item.get("obs_uid"): item for item in data["observations"]}
    kept = [item for item in data["observations"] if item.get("status") == "kept"]
    by_frame = defaultdict(list)
    for item in kept:
        by_frame[item.get("frame_uid")].append(item)
    radius = float(data.get("mapping_config", {}).get("downsample_voxel_size") or 0.025)
    for frame_uid, items in by_frame.items():
        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                left, right = items[left_index], items[right_index]
                mask_left = _array_from_ref(left.get("processed_mask_ref"), evidence_dir, bool)
                mask_right = _array_from_ref(right.get("processed_mask_ref"), evidence_dir, bool)
                if mask_left is None or mask_right is None:
                    continue
                mask_iou, containment = _mask_metrics(mask_left, mask_right)
                if containment < 0.95 or mask_iou < 0.85:
                    continue
                image_similarity = _cosine(
                    _array_from_ref(left.get("image_feat_ref"), evidence_dir),
                    _array_from_ref(right.get("image_feat_ref"), evidence_dir),
                )
                if image_similarity is None or image_similarity < 0.95:
                    continue
                overlap = _symmetric_overlap(
                    _array_from_ref(left.get("pcd_ref"), evidence_dir),
                    _array_from_ref(right.get("pcd_ref"), evidence_dir),
                    radius,
                )
                center_distance = _normalised_center_distance(left, right)
                if (
                    overlap >= 0.80
                    or (center_distance is not None and center_distance <= 0.15)
                ):
                    compatible = _label_compatible(left.get("class_name"), right.get("class_name"))
                    findings.add(
                        "DUP-OBS-001" if compatible else "DUP-OBS-002",
                        "DUPLICATE_OBSERVATION" if compatible else "DUPLICATE_OBSERVATION_LABEL_CONFLICT",
                        "HIGH_CONFIDENCE" if compatible else "AMBIGUOUS",
                        scope={"frame_uid": frame_uid, "obs_uids": [left.get("obs_uid"), right.get("obs_uid")]},
                        metrics={"mask_iou": mask_iou, "containment": containment, "image_similarity": image_similarity, "overlap_3d_sym": overlap, "normalised_center_distance": center_distance},
                        evidence_groups=["2D mask", "3D geometry", "image semantics"],
                        vetoes=[] if compatible else ["label_conflict_or_part_whole"],
                        triage="DETERMINISTIC_REPAIR_CANDIDATE" if compatible else "VLM_REVIEW",
                        recommended_action="KEEP_ONE_OBSERVATION" if compatible else None,
                    )

    for obj in data["final_membership"]:
        members = _object_members(obj, obs_lookup)
        for left_index in range(len(members)):
            for right_index in range(left_index + 1, len(members)):
                left, right = members[left_index], members[right_index]
                if left.get("frame_uid") != right.get("frame_uid"):
                    continue
                distance = _normalised_center_distance(left, right)
                iou = _bbox_iou(left.get("bbox_2d", [0, 0, 0, 0]), right.get("bbox_2d", [0, 0, 0, 0]))
                if distance is None or iou >= 0.05 or distance <= 0.50:
                    continue
                overlap = _symmetric_overlap(
                    _array_from_ref(left.get("pcd_ref"), evidence_dir),
                    _array_from_ref(right.get("pcd_ref"), evidence_dir),
                    radius,
                )
                if distance is not None and iou < 0.05 and overlap < 0.10 and distance > 0.50:
                    compatible = _label_compatible(left.get("class_name"), right.get("class_name"))
                    findings.add(
                        "FM-001",
                        "FALSE_MERGE",
                        "HIGH_CONFIDENCE" if compatible else "AMBIGUOUS",
                        scope={"object_uid": obj.get("object_uid"), "obs_uids": [left.get("obs_uid"), right.get("obs_uid")]},
                        metrics={"bbox_iou": iou, "overlap_3d_sym": overlap, "normalised_center_distance": distance},
                        evidence_groups=["2D mask", "3D geometry", "multi-view temporal evidence"],
                        vetoes=[] if compatible else ["possible_part_whole"],
                        triage="DETERMINISTIC_REPAIR_CANDIDATE" if compatible else "VLM_REVIEW",
                        recommended_action="DETACH_AND_REASSOCIATE",
                    )

        if len(members) >= 3:
            features = [_array_from_ref(item.get("image_feat_ref"), evidence_dir) for item in members]
            for index, (member, feature) in enumerate(zip(members, features)):
                rest = [value.reshape(-1) for pos, value in enumerate(features) if pos != index and value is not None]
                if feature is None or not rest:
                    continue
                supports = [_cosine(feature, candidate) for candidate in rest]
                support = max(value for value in supports if value is not None)
                if support < 0.75:
                    findings.add(
                        "FM-002",
                        "FALSE_MERGE_SEMANTIC_OUTLIER",
                        "RISK_ONLY",
                        severity="MEDIUM",
                        scope={"object_uid": obj.get("object_uid"), "obs_uid": member.get("obs_uid")},
                        metrics={"semantic_support": support},
                        evidence_groups=["image semantics"],
                        triage="LOG_ONLY",
                    )

        center, extent = obj.get("bbox_center"), obj.get("bbox_extent")
        if int(obj.get("n_points", 0) or 0) <= 0 or not _finite_vec(center) or not _finite_vec(extent) or any(float(value) < 0 for value in (extent or [])) or not obj.get("member_observation_uids"):
            findings.add(
                "WN-001",
                "INVALID_GEOMETRY_NODE",
                "CERTAIN",
                scope={"object_uid": obj.get("object_uid")},
            )

    active = [item for item in data["final_membership"] if item.get("status", "active") == "active"]
    object_features = {item.get("object_uid"): _object_feature(item, obs_lookup, evidence_dir) for item in active}
    object_pcds = {}
    for left_index in range(len(active)):
        for right_index in range(left_index + 1, len(active)):
            left, right = active[left_index], active[right_index]
            similarity = _cosine(object_features[left.get("object_uid")], object_features[right.get("object_uid")])
            distance = _normalised_center_distance(left, right, prefix="bbox")
            compatible = _label_compatible(left.get("class_name"), right.get("class_name"))
            if similarity is None or similarity < 0.95 or distance is None or distance > 0.20 or not compatible:
                continue
            for item in (left, right):
                uid = item.get("object_uid")
                if uid not in object_pcds:
                    object_pcds[uid] = _object_pcd(item, obs_lookup, evidence_dir)
            overlap = _symmetric_overlap(
                object_pcds[left.get("object_uid")],
                object_pcds[right.get("object_uid")],
                radius,
            )
            if overlap < 0.85:
                continue
            left_frames = {obs_lookup[uid].get("frame_uid") for uid in left.get("member_observation_uids", []) if uid in obs_lookup}
            right_frames = {obs_lookup[uid].get("frame_uid") for uid in right.get("member_observation_uids", []) if uid in obs_lookup}
            co_visible_separate = 0
            for frame_uid in left_frames & right_frames:
                left_obs = [obs_lookup[uid] for uid in left.get("member_observation_uids", []) if uid in obs_lookup and obs_lookup[uid].get("frame_uid") == frame_uid]
                right_obs = [obs_lookup[uid] for uid in right.get("member_observation_uids", []) if uid in obs_lookup and obs_lookup[uid].get("frame_uid") == frame_uid]
                if any(_bbox_iou(a.get("bbox_2d"), b.get("bbox_2d")) < 0.05 for a in left_obs for b in right_obs):
                    co_visible_separate += 1
            if co_visible_separate == 0:
                findings.add(
                    "FS-001",
                    "FALSE_SPLIT",
                    "HIGH_CONFIDENCE",
                    scope={"object_uids": [left.get("object_uid"), right.get("object_uid")]},
                    metrics={"overlap_3d_sym": overlap, "image_similarity": similarity, "normalised_center_distance": distance, "co_visible_separate_count": 0},
                    evidence_groups=["3D geometry", "image semantics", "multi-view temporal evidence"],
                    triage="DETERMINISTIC_REPAIR_CANDIDATE",
                    recommended_action="MERGE_OBJECTS",
                )

    confidences = np.asarray([item.get("confidence") for item in kept if item.get("confidence") is not None], dtype=float)
    areas = np.asarray([item.get("processed_mask_area") for item in kept if item.get("processed_mask_area") is not None], dtype=float)
    points = np.asarray([item.get("n_points") for item in kept], dtype=float)
    q10_conf = float(np.quantile(confidences, 0.10)) if len(confidences) else None
    q10_area = float(np.quantile(areas, 0.10)) if len(areas) else None
    q10_points = float(np.quantile(points, 0.10)) if len(points) else None
    for obj in active:
        members = _object_members(obj, obs_lookup)
        if len({item.get("frame_uid") for item in members}) != 1 or not members:
            continue
        item = members[0]
        signals = []
        if q10_conf is not None and float(item.get("confidence") or 0) <= q10_conf:
            signals.append("low_confidence")
        if q10_area is not None and float(item.get("processed_mask_area") or 0) <= q10_area:
            signals.append("small_mask")
        if q10_points is not None and float(item.get("n_points") or 0) <= q10_points:
            signals.append("few_points")
        if len(signals) >= 3:
            findings.add(
                "WN-002",
                "WEAK_NODE_RISK",
                "AMBIGUOUS",
                severity="MEDIUM",
                scope={"object_uid": obj.get("object_uid"), "obs_uid": item.get("obs_uid")},
                metrics={"risk_signals": signals},
                evidence_groups=["2D mask", "3D geometry", "image semantics"],
                triage="VLM_REVIEW",
            )

    for assoc in data["associations"]:
        margin, top1, threshold = assoc.get("margin"), assoc.get("top1_score"), assoc.get("sim_threshold")
        if margin is not None and float(margin) <= 0.03:
            findings.add(
                "AR-001",
                "LOW_ASSOCIATION_MARGIN",
                "RISK_ONLY",
                severity="MEDIUM",
                scope={"event_uid": assoc.get("event_uid"), "obs_uid": assoc.get("obs_uid")},
                metrics={"margin": margin},
                evidence_groups=["association history"],
                triage="LOG_ONLY",
            )
        if top1 is not None and threshold is not None and 0 < float(top1) - float(threshold) <= 0.05:
            findings.add(
                "AR-002",
                "NEAR_THRESHOLD_ASSOCIATION",
                "RISK_ONLY",
                severity="MEDIUM",
                scope={"event_uid": assoc.get("event_uid"), "obs_uid": assoc.get("obs_uid")},
                metrics={"top1_score": top1, "threshold": threshold, "slack": float(top1) - float(threshold)},
                evidence_groups=["association history"],
                triage="LOG_ONLY",
            )
        if assoc.get("decision") == "CREATE_OBJECT" and top1 is not None and threshold is not None and 0 <= float(threshold) - float(top1) <= 0.05:
            findings.add(
                "AR-005",
                "CREATE_NEAR_THRESHOLD",
                "RISK_ONLY",
                severity="MEDIUM",
                scope={"event_uid": assoc.get("event_uid"), "obs_uid": assoc.get("obs_uid")},
                metrics={"top1_score": top1, "threshold": threshold},
                evidence_groups=["association history"],
                triage="LOG_ONLY",
            )


def _build_cases(data: dict, findings: list[dict], audit_dir: Path) -> None:
    cases_dir = audit_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    obs_lookup = {item.get("obs_uid"): item for item in data["observations"]}
    frames = {item.get("frame_uid"): item for item in data["frames"]}
    for finding in findings:
        case_dir = cases_dir / finding["finding_uid"]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "metrics.json").write_text(
            json.dumps(finding.get("metrics", {}), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        obs_uids = list(finding.get("scope", {}).get("obs_uids", []))
        if finding.get("scope", {}).get("obs_uid"):
            obs_uids.append(finding["scope"]["obs_uid"])
        try:
            from PIL import Image, ImageDraw

            selected = [obs_lookup[uid] for uid in dict.fromkeys(obs_uids) if uid in obs_lookup]
            if selected:
                frame = frames.get(selected[0].get("frame_uid"), {})
                rgb_ref = frame.get("rgb_ref") or frame.get("rgb_path")
                parsed = _parse_ref(rgb_ref)
                rgb_path = _resolve_path(parsed["path"], data["evidence_dir"])
                image = Image.open(rgb_path).convert("RGB")
                draw = ImageDraw.Draw(image)
                colors = ["#ff304f", "#00b8d9", "#ffab00", "#36b37e"]
                for index, obs in enumerate(selected[:4]):
                    draw.rectangle(obs.get("bbox_2d", [0, 0, 0, 0]), outline=colors[index], width=4)
                    draw.text((obs["bbox_2d"][0], obs["bbox_2d"][1]), obs.get("obs_uid", ""), fill=colors[index])
                    x1, y1, x2, y2 = [int(value) for value in obs.get("bbox_2d", [0, 0, 0, 0])]
                    width, height = x2 - x1, y2 - y1
                    margin = int(max(width, height) * 0.35)
                    crop = image.crop((max(0, x1 - margin), max(0, y1 - margin), min(image.width, x2 + margin), min(image.height, y2 + margin)))
                    crop.save(case_dir / f"context_crop_{obs['obs_uid']}.jpg", quality=92)
                image.save(case_dir / "overview.jpg", quality=92)
        except Exception as exc:
            finding.setdefault("case_builder_warnings", []).append(f"{type(exc).__name__}: {exc}")
        finding.setdefault("evidence_refs", {})["case_packet_ref"] = str(
            case_dir.relative_to(audit_dir.parent)
        )
        (case_dir / "case.json").write_text(
            json.dumps(finding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def audit_evidence(
    evidence_dir: str | Path,
    *,
    strict: bool = False,
    write: bool = True,
    run_semantic_rules: bool = True,
) -> dict:
    _ARRAY_CACHE.clear()
    _TREE_CACHE.clear()
    _HASH_CACHE.clear()
    _load_container.cache_clear()
    evidence_dir = Path(evidence_dir).resolve()
    errors: list[str] = []
    data = {
        "evidence_dir": evidence_dir,
        "errors": errors,
        "manifest": _load_json(evidence_dir / "manifest.json", errors, {}),
        "frames": _load_jsonl(evidence_dir / "frames.jsonl", errors),
        "observations": _load_jsonl(evidence_dir / "observations.jsonl", errors),
        "associations": _load_jsonl(evidence_dir / "associations.jsonl", errors),
        "events": _load_jsonl(evidence_dir / "mapping_events.jsonl", errors),
        "vlm_events": _load_jsonl(evidence_dir / "vlm_events.jsonl", errors),
        "filter_trace": _load_jsonl(evidence_dir / "filter_trace.jsonl", errors),
        "versions": _load_jsonl(evidence_dir / "object_versions.jsonl", errors),
        "pair_decisions": _load_jsonl(evidence_dir / "object_pair_decisions.jsonl", errors),
        "final_membership": _load_json(evidence_dir / "final_membership.json", errors, []),
    }
    mapping_ref = _parse_ref(data["manifest"].get("mapping_config_ref"))
    data["mapping_config"] = {}
    if mapping_ref and mapping_ref.get("path"):
        data["mapping_config"] = _load_json(
            _resolve_path(mapping_ref["path"], evidence_dir), errors, {}
        )
    run_id = str(data["manifest"].get("run_id", evidence_dir.name))
    findings = _Findings(run_id)
    _audit_readiness(data, findings)
    gate_passed = not any(
        any(str(rule).startswith("EVI-") for rule in item["rule_ids"])
        for item in findings.items
    )
    _audit_mapping_invariants(data, findings)
    if gate_passed and run_semantic_rules:
        _audit_semantic_rules(data, findings)

    audit_dir = evidence_dir.parent / "audit"
    counts = Counter(item.get("certainty") for item in findings.items)
    summary = {
        "schema_version": "0.2.0",
        "run_id": run_id,
        "evidence_dir": str(evidence_dir),
        "audit_dir": str(audit_dir),
        "gate_status": "PASS" if gate_passed else "FAIL",
        "status": "EVIDENCE_VALID" if gate_passed else "EVIDENCE_INVALID",
        "semantic_rules_executed": gate_passed and run_semantic_rules,
        "strict": bool(strict),
        "finding_count": len(findings.items),
        "certainty_counts": dict(counts),
        "rule_counts": dict(
            Counter(rule for item in findings.items for rule in item.get("rule_ids", []))
        ),
        "source_manifest_status": data["manifest"].get("status"),
        "parse_errors": errors,
    }
    if write:
        audit_dir.mkdir(parents=True, exist_ok=True)
        _build_cases(data, findings.items, audit_dir)
        with (audit_dir / "findings.jsonl").open("w", encoding="utf-8") as handle:
            for item in findings.items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        (audit_dir / "audit_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (audit_dir / "validation.json").write_text(
            json.dumps(
                {
                    "schema_version": "0.2.0",
                    "run_id": run_id,
                    "gate_status": summary["gate_status"],
                    "evidence_findings": [
                        item
                        for item in findings.items
                        if any(rule.startswith("EVI-") for rule in item["rule_ids"])
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "summary": summary,
        "findings": findings.items,
        "exit_code": 2 if strict and not gate_passed else 0,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = audit_evidence(args.evidence_dir, strict=args.strict, write=True)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
