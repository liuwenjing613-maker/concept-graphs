"""Read-only, stage-aware causal audit built on the unified evidence ledger."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import yaml

from conceptgraph.audit import evidence_audit as legacy


SCHEMA_VERSION = "1.1.0"
STAGE_ORDER = {
    "system": 0,
    "detection": 1,
    "segmentation": 2,
    "geometry": 3,
    "association": 4,
    "fusion": 5,
    "object_identity": 6,
    "caption": 7,
    "relation": 8,
}
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
CERTAINTY_ORDER = {
    "CONFIRMED_SYSTEM_ERROR": 0,
    "LIKELY_MAPPING_CONFLICT": 1,
    "AMBIGUOUS_MAPPING_RISK": 2,
    "INSUFFICIENT_EVIDENCE": 3,
    "NO_CONFLICT_FOUND": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _facts(metrics: Optional[dict]) -> list[dict]:
    return [
        {"name": str(key), "value": value}
        for key, value in (metrics or {}).items()
    ]


def _finite(values: Any) -> bool:
    try:
        array = np.asarray(values, dtype=float)
        return bool(array.size and np.isfinite(array).all())
    except Exception:
        return False


def _volume(extent: Any) -> Optional[float]:
    if not _finite(extent):
        return None
    return float(np.prod(np.maximum(np.asarray(extent, dtype=float), 0)))


def _frame_number(uid: Any) -> int:
    value = legacy._frame_idx(str(uid))
    return int(value) if value is not None else -1


@dataclass
class AuditContext:
    run_id: str
    scene_id: str
    manifest: dict
    config: dict
    audit_config: dict
    policy: dict
    policy_source: str
    evidence_root: Path
    experiment_dir: Path
    output_dir: Path
    environment_mode: str


@dataclass
class Finding:
    finding_uid: str
    checker_id: str
    stage: str
    subtype: str
    scope: dict
    certainty: str
    severity: str
    policy_context: dict
    proven_facts: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    evidence_refs: dict = field(default_factory=dict)
    route: str = "LOG_ONLY"
    repair_allowed: bool = False
    review_priority: Optional[int] = None
    review_score: Optional[float] = None
    review_score_components: dict = field(default_factory=dict)
    selection_cohort: Optional[str] = None
    selection_probability: Optional[float] = None
    sampling_weight: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class FactStore:
    """Indexed evidence plus deterministic, reusable pair/object facts."""

    def __init__(self, context: AuditContext):
        self.context = context
        errors: list[str] = []
        root = context.evidence_root
        self.errors = errors
        self.frames = legacy._load_jsonl(root / "frames.jsonl", errors)
        self.observations = legacy._load_jsonl(root / "observations.jsonl", errors)
        self.associations = legacy._load_jsonl(root / "associations.jsonl", errors)
        self.events = legacy._load_jsonl(root / "mapping_events.jsonl", errors)
        self.versions = legacy._load_jsonl(root / "object_versions.jsonl", errors)
        self.pair_decisions = legacy._load_jsonl(
            root / "object_pair_decisions.jsonl", errors
        )
        self.final_membership = legacy._load_json(
            root / "final_membership.json", errors, []
        )
        self.frame_by_uid = {item.get("frame_uid"): item for item in self.frames}
        self.obs_by_uid = {item.get("obs_uid"): item for item in self.observations}
        self.kept = [item for item in self.observations if item.get("status") == "kept"]
        self.kept_by_frame: dict[str, list[dict]] = defaultdict(list)
        for item in self.kept:
            self.kept_by_frame[item.get("frame_uid")].append(item)
        self.object_by_uid = {
            item.get("object_uid"): item for item in self.final_membership
        }
        self.ownership: dict[str, list[str]] = defaultdict(list)
        for item in self.final_membership:
            for obs_uid in item.get("member_observation_uids", []):
                self.ownership[str(obs_uid)].append(str(item.get("object_uid")))
        self.association_by_obs = {
            item.get("obs_uid"): item for item in self.associations
        }
        self.event_by_uid = {
            item.get("event_uid"): item for item in self.events
            if item.get("event_uid")
        }
        self.version_by_uid = {
            item.get("object_version_uid"): item for item in self.versions
            if item.get("object_version_uid")
        }
        self.associations_by_frame: dict[str, list[dict]] = defaultdict(list)
        for item in self.associations:
            self.associations_by_frame[item.get("frame_uid")].append(item)
        self.versions_by_object: dict[str, list[dict]] = defaultdict(list)
        for item in self.versions:
            self.versions_by_object[item.get("object_uid")].append(item)
        for items in self.versions_by_object.values():
            items.sort(key=lambda item: int(item.get("version") or 0))
        self._object_feature_cache: dict[str, Optional[np.ndarray]] = {}
        self._object_pcd_cache: dict[str, Optional[np.ndarray]] = {}
        self.mask_conflict_edges: list[dict] = []

    def array(self, ref: Any, dtype=float) -> Optional[np.ndarray]:
        return legacy._array_from_ref(ref, self.context.evidence_root, dtype)

    def member_observations(
        self, object_uid: str, object_version_uid: Optional[str] = None
    ) -> list[dict]:
        version = self.version_by_uid.get(object_version_uid, {}) if object_version_uid else {}
        item = self.object_by_uid.get(object_uid, {})
        member_uids = version.get("member_observation_uids", []) or item.get("member_observation_uids", [])
        if not member_uids:
            # Association candidates may later be merged/inactivated and therefore
            # disappear from final_membership. Their last non-empty version still
            # contains the exact observation history needed for review.
            for version in reversed(self.versions_by_object.get(object_uid, [])):
                member_uids = version.get("member_observation_uids", [])
                if member_uids:
                    break
        return [
            self.obs_by_uid[uid]
            for uid in member_uids
            if uid in self.obs_by_uid
        ]

    def object_feature(self, object_uid: str) -> Optional[np.ndarray]:
        if object_uid not in self._object_feature_cache:
            item = self.object_by_uid.get(object_uid)
            self._object_feature_cache[object_uid] = (
                legacy._object_feature(
                    item, self.obs_by_uid, self.context.evidence_root
                )
                if item
                else None
            )
        return self._object_feature_cache[object_uid]

    def object_pcd(self, object_uid: str) -> Optional[np.ndarray]:
        if object_uid not in self._object_pcd_cache:
            item = self.object_by_uid.get(object_uid)
            self._object_pcd_cache[object_uid] = (
                legacy._object_pcd(item, self.obs_by_uid, self.context.evidence_root)
                if item
                else None
            )
        return self._object_pcd_cache[object_uid]

    def separated_covisibility(self, left_uid: str, right_uid: str) -> int:
        left = self.member_observations(left_uid)
        right = self.member_observations(right_uid)
        by_left: dict[str, list[dict]] = defaultdict(list)
        by_right: dict[str, list[dict]] = defaultdict(list)
        for item in left:
            by_left[item.get("frame_uid")].append(item)
        for item in right:
            by_right[item.get("frame_uid")].append(item)
        count = 0
        for frame_uid in set(by_left) & set(by_right):
            separated = any(
                legacy._bbox_iou(a.get("bbox_2d"), b.get("bbox_2d")) < 0.05
                and (
                    legacy._normalised_center_distance(a, b) is None
                    or legacy._normalised_center_distance(a, b) > 0.30
                )
                for a in by_left[frame_uid]
                for b in by_right[frame_uid]
            )
            count += int(separated)
        return count


class LayeredAudit:
    def __init__(self, context: AuditContext, facts: FactStore):
        self.context = context
        self.facts = facts
        self.findings: list[Finding] = []
        self._counter = 0
        self._rule_counts: Counter[str] = Counter()
        self._missing_keys: set[tuple[str, str]] = set()
        self.thresholds = context.audit_config.get("thresholds", {})
        self.max_per_rule = int(
            context.audit_config.get("limits", {}).get(
                "max_findings_per_rule", 500
            )
        )
        self.radius = float(context.config.get("downsample_voxel_size") or 0.025)

    def add(
        self,
        checker_id: str,
        stage: str,
        subtype: str,
        certainty: str,
        *,
        severity: str = "MEDIUM",
        scope: Optional[dict] = None,
        metrics: Optional[dict] = None,
        hypotheses: Optional[list[dict]] = None,
        vetoes: Optional[list[str]] = None,
        missing_evidence: Optional[list[str]] = None,
        evidence_refs: Optional[dict] = None,
        route: str = "LOG_ONLY",
    ) -> Optional[Finding]:
        if self._rule_counts[checker_id] >= self.max_per_rule:
            return None
        self._counter += 1
        self._rule_counts[checker_id] += 1
        finding = Finding(
            finding_uid=f"finding_{self._counter:06d}",
            checker_id=checker_id,
            stage=stage,
            subtype=subtype,
            scope=scope or {},
            certainty=certainty,
            severity=severity,
            policy_context={
                "observation_ownership": self.context.policy.get(
                    "observation_ownership"
                ),
                "environment_mode": self.context.environment_mode,
                "policy_source": self.context.policy_source,
            },
            proven_facts=_facts(metrics),
            hypotheses=hypotheses or [],
            vetoes=vetoes or [],
            missing_evidence=missing_evidence or [],
            evidence_refs=evidence_refs or {},
            route=route,
            repair_allowed=False,
            review_priority=None,
        )
        self.findings.append(finding)
        return finding

    def missing(
        self,
        checker_id: str,
        stage: str,
        evidence_name: str,
        *,
        affected_count: int = 0,
    ) -> None:
        key = (checker_id, evidence_name)
        if key in self._missing_keys:
            return
        self._missing_keys.add(key)
        self.add(
            checker_id,
            stage,
            "EVIDENCE_NOT_AVAILABLE",
            "INSUFFICIENT_EVIDENCE",
            severity="LOW",
            metrics={"affected_count": int(affected_count)},
            missing_evidence=[evidence_name],
            route="SUPPLEMENT_EVIDENCE",
        )

    def run(self) -> tuple[list[Finding], list[dict], dict]:
        started = time.perf_counter()
        legacy_result = legacy.audit_evidence(
            self.context.evidence_root,
            strict=False,
            write=False,
            run_semantic_rules=False,
        )
        gate_passed = legacy_result["summary"]["gate_status"] == "PASS"
        self._system_findings(legacy_result["findings"])
        if "audit_policy" not in self.context.manifest:
            self.missing("SYS-001", "system", "manifest.audit_policy", affected_count=1)
        if gate_passed:
            enabled = self.context.audit_config.get("enabled_checkers", {})
            if enabled.get("detection", True):
                self._detection()
            if enabled.get("segmentation", True):
                self._segmentation()
            if enabled.get("projection_geometry", True):
                self._geometry()
            if enabled.get("association", True):
                self._association()
            if enabled.get("fusion", True):
                self._fusion()
            if enabled.get("object_identity", True):
                self._object_identity()
            if enabled.get("caption", False):
                self.missing("CAP-000", "caption", "caption checker v1 implementation")
            if enabled.get("relation", False):
                self.missing("REL-000", "relation", "relation checker v1 implementation")
        root_causes = resolve_root_causes(self.findings)
        summary = self._summary(gate_passed, root_causes, time.perf_counter() - started)
        return self.findings, root_causes, summary

    def _system_findings(self, rows: list[dict]) -> None:
        mapping = {
            "EVI-001": "SYS-001",
            "EVI-002": "SYS-001",
            "EVI-003": "SYS-001",
            "EVI-004": "SYS-002",
            "EVI-005": "SYS-001",
            "EVI-006": "SYS-008",
            "EVI-007": "SYS-008",
            "EVI-008": "SYS-001",
            "MAP-001": "SYS-003",
            "MAP-002": "SYS-002",
            "MAP-003": "SYS-004",
            "MAP-004": "SYS-005",
            "MAP-005": "SYS-006",
            "MAP-006": "SYS-007",
            "MAP-007": "SYS-006",
            "MAP-008": "SYS-007",
            "MAP-009": "SYS-009",
        }
        for row in rows:
            rule_ids = row.get("rule_ids", [])
            rule = next(
                (item for item in rule_ids if item.startswith(("EVI-", "MAP-"))),
                None,
            )
            if not rule:
                continue
            self.add(
                mapping.get(rule, "SYS-001"),
                "system",
                row.get("error_type", rule),
                "CONFIRMED_SYSTEM_ERROR",
                severity="CRITICAL" if rule.startswith("EVI-") else "HIGH",
                scope=row.get("scope"),
                metrics=row.get("metrics"),
                hypotheses=[{"name": "system_or_ledger_inconsistency", "support": rule_ids}],
                evidence_refs=row.get("evidence_refs"),
                route="BLOCK_RUN",
            )

    def _part_whole_veto(self, left: dict, right: dict) -> list[str]:
        a = str(left.get("class_name") or "").lower()
        b = str(right.get("class_name") or "").lower()
        pairs = {
            frozenset(("sofa", "pillow")),
            frozenset(("couch", "pillow")),
            frozenset(("chair", "pillow")),
            frozenset(("cabinet", "drawer")),
            frozenset(("table", "cup")),
        }
        return ["part_whole_or_support_not_excluded"] if frozenset((a, b)) in pairs else []

    def _detection(self) -> None:
        cfg = self.thresholds["duplicate_proposal"]
        missing_pair_evidence = 0
        for frame_uid, items in self.facts.kept_by_frame.items():
            for index, left in enumerate(items):
                for right in items[index + 1 :]:
                    mask_left = self.facts.array(left.get("processed_mask_ref"), bool)
                    mask_right = self.facts.array(right.get("processed_mask_ref"), bool)
                    feature_left = self.facts.array(left.get("image_feat_ref"))
                    feature_right = self.facts.array(right.get("image_feat_ref"))
                    pcd_left = self.facts.array(left.get("pcd_ref"))
                    pcd_right = self.facts.array(right.get("pcd_ref"))
                    if any(
                        value is None
                        for value in (mask_left, mask_right, feature_left, feature_right, pcd_left, pcd_right)
                    ):
                        missing_pair_evidence += 1
                        continue
                    iou, containment = legacy._mask_metrics(mask_left, mask_right)
                    clip = legacy._cosine(feature_left, feature_right)
                    if not (
                        (iou > float(cfg["mask_iou"]) or containment > float(cfg["containment"]))
                        and clip is not None
                        and clip > float(cfg["clip_similarity"])
                    ):
                        continue
                    overlap = legacy._symmetric_overlap(pcd_left, pcd_right, self.radius)
                    self.facts.mask_conflict_edges.append(
                        {
                            "frame_uid": frame_uid,
                            "obs_uids": [left.get("obs_uid"), right.get("obs_uid")],
                            "mask_iou": iou,
                            "containment": containment,
                            "clip_similarity": clip,
                            "symmetric_3d_support": overlap,
                        }
                    )
                    if overlap <= float(cfg["symmetric_3d_support"]):
                        continue
                    vetoes = self._part_whole_veto(left, right)
                    self.add(
                        "DET-001",
                        "detection",
                        "DUPLICATE_PROPOSAL",
                        "AMBIGUOUS_MAPPING_RISK" if vetoes else "LIKELY_MAPPING_CONFLICT",
                        severity="HIGH",
                        scope={"frame_uid": frame_uid, "obs_uids": [left.get("obs_uid"), right.get("obs_uid")]},
                        metrics={"mask_iou": iou, "mask_containment": containment, "clip_similarity": clip, "symmetric_3d_support": overlap},
                        hypotheses=[{"name": "duplicate_proposal", "support": ["2d_overlap", "clip_similarity", "3d_support"]}],
                        vetoes=vetoes,
                        route="VLM_REVIEW" if vetoes else "HUMAN_REVIEW",
                    )
        if missing_pair_evidence:
            self.missing("DET-001", "detection", "processed mask / CLIP / observation PCD", affected_count=missing_pair_evidence)

        confidences = np.asarray([item.get("confidence") for item in self.facts.kept if item.get("confidence") is not None], dtype=float)
        areas = np.asarray([item.get("processed_mask_area") for item in self.facts.kept if item.get("processed_mask_area") is not None], dtype=float)
        points = np.asarray([item.get("n_points") for item in self.facts.kept], dtype=float)
        q10_conf = float(np.quantile(confidences, 0.10)) if len(confidences) else None
        q10_area = float(np.quantile(areas, 0.10)) if len(areas) else None
        q10_points = float(np.quantile(points, 0.10)) if len(points) else None
        needed = int(self.thresholds["false_positive"]["required_signals_likely"])
        for obs in self.facts.kept:
            signals = []
            if q10_conf is not None and float(obs.get("confidence") or 0) <= q10_conf:
                signals.append("low_detector_confidence")
            if q10_area is not None and float(obs.get("processed_mask_area") or 0) <= q10_area:
                signals.append("small_processed_mask")
            if q10_points is not None and float(obs.get("n_points") or 0) <= q10_points:
                signals.append("few_3d_points")
            valid_depth = obs.get("valid_depth_ratio")
            if valid_depth is not None and float(valid_depth) < float(self.thresholds["false_positive"]["valid_depth_ratio"]):
                signals.append("low_valid_depth_ratio")
            owners = self.facts.ownership.get(obs.get("obs_uid"), [])
            supported_frames = 0
            if owners:
                members = self.facts.member_observations(owners[0])
                supported_frames = len({item.get("frame_uid") for item in members})
                if supported_frames <= 1:
                    signals.append("no_temporal_support")
            if len(signals) >= needed:
                self.add(
                    "DET-002", "detection", "POSSIBLE_FALSE_POSITIVE",
                    "LIKELY_MAPPING_CONFLICT" if len(signals) >= needed + 1 else "AMBIGUOUS_MAPPING_RISK",
                    severity="MEDIUM", scope={"obs_uid": obs.get("obs_uid"), "object_uid": owners[0] if owners else None},
                    metrics={"signals": signals, "supported_frame_count": supported_frames},
                    hypotheses=[{"name": "false_positive_detection", "support": signals}],
                    vetoes=["single_view_real_small_object_not_excluded"], route="HUMAN_REVIEW",
                )
        self.missing("DET-004", "detection", "GT or full-frame visibility coverage", affected_count=1)

    @staticmethod
    def _components(mask: np.ndarray) -> tuple[int, float]:
        try:
            from scipy import ndimage
            labels, count = ndimage.label(mask)
            if not count:
                return 0, 0.0
            sizes = np.bincount(labels.reshape(-1))[1:]
            sizes = np.sort(sizes)[::-1]
            return int(count), float(sizes[1] / sizes.sum()) if len(sizes) > 1 else 0.0
        except Exception:
            return 0, 0.0

    def _segmentation(self) -> None:
        cfg = self.thresholds["segmentation"]
        cluster_available = 0
        for obs in self.facts.kept:
            mask = self.facts.array(obs.get("processed_mask_ref"), bool)
            if mask is None:
                continue
            area = int(mask.sum())
            if area <= 0 or area != int(obs.get("processed_mask_area") or -1):
                self.add(
                    "SEG-001", "segmentation", "INVALID_OR_DEGENERATE_MASK",
                    "CONFIRMED_SYSTEM_ERROR", severity="CRITICAL",
                    scope={"obs_uid": obs.get("obs_uid")},
                    metrics={"artifact_area": area, "recorded_area": obs.get("processed_mask_area")},
                    hypotheses=[{"name": "mask_ledger_inconsistency", "support": ["area_mismatch_or_empty"]}],
                    route="BLOCK_RUN",
                )
            component_count, second_ratio = self._components(mask)
            raw_area = float(obs.get("pre_subtract_mask_area") or obs.get("raw_mask_area") or 0)
            loss_ratio = 1.0 - area / raw_area if raw_area > 0 else 0.0
            if loss_ratio > float(cfg["subtraction_loss_ratio"]) and second_ratio >= float(cfg["fragmented_second_component_ratio"]):
                self.add(
                    "SEG-005", "segmentation", "CONTAINMENT_SUBTRACTION_DAMAGE",
                    "AMBIGUOUS_MAPPING_RISK", severity="HIGH",
                    scope={"obs_uid": obs.get("obs_uid"), "frame_uid": obs.get("frame_uid")},
                    metrics={"loss_ratio": loss_ratio, "component_count": component_count, "second_component_ratio": second_ratio},
                    hypotheses=[{"name": "over_aggressive_containment_subtraction", "support": ["large_area_loss", "fragmented_processed_mask"]}],
                    vetoes=["part_whole_or_occlusion_not_excluded"], route="VLM_REVIEW",
                )
            pre = obs.get("pre_dbscan") or {}
            second_cluster = pre.get("second_cluster_ratio")
            if second_cluster is not None:
                cluster_available += 1
                if float(second_cluster) >= float(cfg["background_second_cluster_ratio"]):
                    self.add(
                        "SEG-002", "segmentation", "BACKGROUND_LEAKAGE_OR_UNDERSEGMENTATION",
                        "AMBIGUOUS_MAPPING_RISK", severity="HIGH",
                        scope={"obs_uid": obs.get("obs_uid"), "frame_uid": obs.get("frame_uid")},
                        metrics={"second_cluster_ratio": second_cluster, "cluster_count": pre.get("cluster_count"), "cluster_center_distance": pre.get("largest_centers_distance")},
                        hypotheses=[{"name": "background_leakage", "support": ["multiple_3d_clusters"]}, {"name": "undersegmentation", "support": ["multiple_3d_clusters"]}],
                        vetoes=["long_object_or_occlusion_not_excluded"], route="VLM_REVIEW",
                    )
            quality = []
            if obs.get("valid_depth_ratio") is not None and float(obs["valid_depth_ratio"]) < float(cfg["low_valid_depth_ratio"]):
                quality.append("low_valid_depth")
            if float(obs.get("boundary_touch_ratio") or 0) > 0.50:
                quality.append("truncated_at_boundary")
            if component_count >= 2 and second_ratio >= float(cfg["fragmented_second_component_ratio"]):
                quality.append("fragmented_mask")
            if len(quality) >= 2:
                self.add(
                    "SEG-006", "segmentation", "LOW_QUALITY_OBSERVATION",
                    "AMBIGUOUS_MAPPING_RISK", severity="LOW",
                    scope={"obs_uid": obs.get("obs_uid")}, metrics={"quality_labels": quality},
                    hypotheses=[{"name": "occlusion_or_truncation", "support": quality}],
                    route="DOWNWEIGHT_EVIDENCE",
                )
        if not cluster_available:
            self.missing("SEG-002", "segmentation", "pre-DBSCAN cluster statistics", affected_count=len(self.facts.kept))
            self.missing("SEG-003", "segmentation", "depth/3D cluster statistics", affected_count=len(self.facts.kept))

        for object_uid, item in self.facts.object_by_uid.items():
            by_frame: dict[str, list[dict]] = defaultdict(list)
            for obs in self.facts.member_observations(object_uid):
                by_frame[obs.get("frame_uid")].append(obs)
            for frame_uid, members in by_frame.items():
                for index, left in enumerate(members):
                    for right in members[index + 1 :]:
                        clip = legacy._cosine(self.facts.array(left.get("image_feat_ref")), self.facts.array(right.get("image_feat_ref")))
                        distance = legacy._normalised_center_distance(left, right)
                        if clip is not None and clip > 0.90 and distance is not None and distance <= 0.50:
                            self.add(
                                "SEG-004", "segmentation", "POSSIBLE_OVERSEGMENTATION",
                                "AMBIGUOUS_MAPPING_RISK", severity="MEDIUM",
                                scope={"object_uid": object_uid, "frame_uid": frame_uid, "obs_uids": [left.get("obs_uid"), right.get("obs_uid")]},
                                metrics={"clip_similarity": clip, "normalised_center_distance": distance},
                                hypotheses=[{"name": "segmentation_fragmentation", "support": ["same_frame_same_object", "semantic_and_geometry_continuity"]}],
                                vetoes=["legitimate_object_parts_not_excluded"], route="VLM_REVIEW",
                            )

    def _geometry(self) -> None:
        cfg = self.thresholds["geometry"]
        cluster_available = 0
        for obs in self.facts.kept:
            pcd = self.facts.array(obs.get("pcd_ref"))
            center, extent = obs.get("bbox_3d_center"), obs.get("bbox_3d_extent")
            if pcd is None or len(pcd) == 0 or not np.isfinite(pcd).all() or not _finite(center) or not _finite(extent) or any(float(value) < 0 for value in (extent or [])):
                self.add(
                    "GEO-001", "geometry", "INVALID_PCD_OR_BBOX",
                    "CONFIRMED_SYSTEM_ERROR", severity="CRITICAL",
                    scope={"obs_uid": obs.get("obs_uid")},
                    metrics={"pcd_points": None if pcd is None else len(pcd), "bbox_center": center, "bbox_extent": extent},
                    hypotheses=[{"name": "projection_or_artifact_failure", "support": ["invalid_geometry"]}], route="BLOCK_RUN",
                )
            pre = obs.get("pre_dbscan") or {}
            if pre.get("second_cluster_ratio") is not None:
                cluster_available += 1
                if float(pre["second_cluster_ratio"]) >= float(cfg["multi_cluster_second_ratio"]):
                    self.add(
                        "GEO-003", "geometry", "MULTI_CLUSTER_GEOMETRY",
                        "AMBIGUOUS_MAPPING_RISK", severity="MEDIUM",
                        scope={"obs_uid": obs.get("obs_uid")},
                        metrics=pre,
                        hypotheses=[{"name": "undersegmentation_or_background_leakage", "support": ["multi_cluster_geometry"]}],
                        vetoes=["occlusion_or_long_object_not_excluded"], route="VLM_REVIEW",
                    )
        if not cluster_available:
            self.missing("GEO-003", "geometry", "pre-DBSCAN cluster statistics", affected_count=len(self.facts.kept))
        self.missing("GEO-002", "geometry", "calibrated 2D-to-3D reprojection renderer", affected_count=len(self.facts.kept))

        for object_uid in self.facts.object_by_uid:
            members = sorted(self.facts.member_observations(object_uid), key=lambda item: _frame_number(item.get("frame_uid")))
            for before, after in zip(members, members[1:]):
                jump = legacy._normalised_center_distance(before, after)
                if jump is not None and jump > float(cfg["center_jump_normalized"]):
                    self.add(
                        "GEO-004", "geometry", "CROSS_FRAME_GEOMETRY_CONFLICT",
                        "AMBIGUOUS_MAPPING_RISK", severity="HIGH",
                        scope={"object_uid": object_uid, "obs_uids": [before.get("obs_uid"), after.get("obs_uid")]},
                        metrics={"normalised_center_jump": jump},
                        hypotheses=[{"name": "pose_or_association_error", "support": ["abrupt_center_jump"]}, {"name": "world_change", "support": []}],
                        vetoes=["new_viewpoint_or_dynamic_change_not_excluded"], route="HUMAN_REVIEW",
                    )

        for object_uid, versions in self.facts.versions_by_object.items():
            for before, after in zip(versions, versions[1:]):
                if after.get("operation") != "OBJECT_DENOISE":
                    continue
                n_before = float(before.get("n_points") or 0)
                n_after = float(after.get("n_points") or 0)
                keep_ratio = n_after / n_before if n_before else 1.0
                if keep_ratio < float(cfg["denoise_keep_ratio"]):
                    self.add(
                        "GEO-005", "geometry", "DENOISE_DESTRUCTIVE_CHANGE",
                        "LIKELY_MAPPING_CONFLICT", severity="HIGH",
                        scope={"object_uid": object_uid, "object_version_uid": after.get("object_version_uid")},
                        metrics={"point_keep_ratio": keep_ratio, "before_points": n_before, "after_points": n_after},
                        hypotheses=[{"name": "denoise_removed_main_geometry", "support": ["large_point_drop"]}], route="HUMAN_REVIEW",
                    )

        by_class: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for uid, item in self.facts.object_by_uid.items():
            vol = _volume(item.get("bbox_extent"))
            if vol is not None and vol > 0:
                by_class[str(item.get("class_name") or "unknown")].append((uid, vol))
        for values in by_class.values():
            if len(values) < 4:
                continue
            logs = np.log([value for _, value in values])
            med = float(np.median(logs)); mad = float(np.median(np.abs(logs - med)))
            if mad <= 1e-9:
                continue
            for (uid, vol), value in zip(values, logs):
                z = abs(float(value - med)) / (1.4826 * mad)
                if z > 4.0:
                    self.add(
                        "GEO-006", "geometry", "OBJECT_SCALE_OUTLIER",
                        "AMBIGUOUS_MAPPING_RISK", severity="LOW",
                        scope={"object_uid": uid}, metrics={"bbox_volume": vol, "robust_z": z},
                        hypotheses=[{"name": "scale_or_segmentation_anomaly", "support": ["class_scale_outlier"]}], route="LOG_ONLY",
                    )

    @staticmethod
    def _association_scope(
        assoc: dict,
        *,
        extra_scope: Optional[dict] = None,
        counterfactual_alternate: Optional[str] = None,
    ) -> dict:
        """Keep every decision-relevant candidate role in the finding scope."""
        candidates = [
            item for item in (assoc.get("top_candidates") or [])
            if item.get("object_uid")
        ]
        def candidate_score(item: dict, key: str) -> float:
            value = item.get(key)
            return float(value) if value is not None else -math.inf

        aggregate = sorted(
            candidates,
            key=lambda item: candidate_score(item, "aggregate_score"),
            reverse=True,
        )
        spatial = max(
            candidates,
            key=lambda item: candidate_score(item, "spatial_score"),
            default={},
        )
        visual = max(
            candidates,
            key=lambda item: candidate_score(item, "visual_score"),
            default={},
        )
        chosen_uid = assoc.get("target_object_uid")
        aggregate_uids = [item.get("object_uid") for item in aggregate[:2]]
        roles = {
            "chosen_target": chosen_uid,
            "spatial_top_candidate": spatial.get("object_uid"),
            "visual_top_candidate": visual.get("object_uid"),
            "aggregate_top1": aggregate_uids[0] if aggregate_uids else None,
            "aggregate_top2": aggregate_uids[1] if len(aggregate_uids) > 1 else None,
            "counterfactual_alternate": counterfactual_alternate,
        }
        candidate_versions = {
            str(uid): version_uid
            for uid, version_uid in zip(
                assoc.get("object_uids_before") or [],
                assoc.get("candidate_object_version_uids") or [],
            )
            if uid and version_uid
        }
        if chosen_uid and chosen_uid not in candidate_versions:
            chosen_version = assoc.get("target_object_version_before") or assoc.get("target_object_version_after")
            if chosen_version:
                candidate_versions[str(chosen_uid)] = chosen_version
        alternate_uids = []
        for uid in (
            roles["spatial_top_candidate"],
            roles["visual_top_candidate"],
            *aggregate_uids,
            counterfactual_alternate,
        ):
            if uid and uid != chosen_uid and uid not in alternate_uids:
                alternate_uids.append(uid)
        scope = {
            "event_uid": assoc.get("event_uid"),
            "obs_uid": assoc.get("obs_uid"),
            "object_uid": chosen_uid,
            "chosen_target_object_uid": chosen_uid,
            "spatial_top_object_uid": roles["spatial_top_candidate"],
            "visual_top_object_uid": roles["visual_top_candidate"],
            "aggregate_top_object_uids": aggregate_uids,
            "counterfactual_alternate_object_uid": counterfactual_alternate,
            "alternate_object_uids": alternate_uids,
            "association_candidate_roles": roles,
            "association_candidate_object_versions": candidate_versions,
        }
        if extra_scope:
            scope.update(extra_scope)
        return scope

    def _association(self) -> None:
        cfg = self.thresholds["association"]
        for assoc in self.facts.associations:
            margin = assoc.get("margin")
            top1 = assoc.get("top1_score")
            threshold = assoc.get("sim_threshold")
            if margin is not None and float(margin) <= float(cfg["low_margin"]):
                self.add(
                    "ASSOC-002", "association", "LOW_MARGIN",
                    "AMBIGUOUS_MAPPING_RISK", severity="LOW",
                    scope=self._association_scope(assoc),
                    metrics={"margin": margin, "very_low": float(margin) <= float(cfg["very_low_margin"])},
                    hypotheses=[{"name": "association_ambiguity", "support": ["low_top1_top2_margin"]}], route="LOG_ONLY",
                )
            candidates = assoc.get("top_candidates") or []
            if candidates:
                spatial_top = max(candidates, key=lambda item: float(item.get("spatial_score") or -math.inf))
                visual_top = max(candidates, key=lambda item: float(item.get("visual_score") or -math.inf))
                if spatial_top.get("object_uid") != visual_top.get("object_uid"):
                    self.add(
                        "ASSOC-003", "association", "SPATIAL_SEMANTIC_DISAGREEMENT",
                        "AMBIGUOUS_MAPPING_RISK", severity="MEDIUM",
                        scope=self._association_scope(assoc),
                        metrics={"spatial_top": spatial_top, "visual_top": visual_top},
                        hypotheses=[{"name": "false_association_or_pose_error", "support": ["rank_disagreement"]}],
                        vetoes=["same_class_neighbour_or_occlusion_not_excluded"], route="VLM_REVIEW",
                    )
                chosen = candidates[0]
                slack = float(top1) - float(threshold) if top1 is not None and threshold is not None else None
                if (
                    assoc.get("decision") == "MERGE_TO_OBJECT"
                    and float(chosen.get("visual_score") or 0) >= float(cfg["high_visual"])
                    and float(chosen.get("spatial_score") or 0) <= float(cfg["low_spatial"])
                    and slack is not None
                    and slack <= float(cfg["low_threshold_slack"])
                ):
                    self.add(
                        "ASSOC-007", "association", "HIGH_SEMANTIC_LOW_GEOMETRY_ASSOCIATION",
                        "AMBIGUOUS_MAPPING_RISK", severity="HIGH",
                        scope=self._association_scope(assoc),
                        metrics={"visual_score": chosen.get("visual_score"), "spatial_score": chosen.get("spatial_score"), "threshold_slack": slack},
                        hypotheses=[{"name": "same_looking_distant_object_or_pose_error", "support": ["high_visual", "low_spatial", "borderline_score"]}], route="VLM_REVIEW",
                    )

        for frame_uid, rows in self.facts.associations_by_frame.items():
            by_target: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_target[row.get("target_object_uid")].append(row)
            for target, group in by_target.items():
                if len(group) < 2:
                    continue
                obs = [self.facts.obs_by_uid.get(row.get("obs_uid")) for row in group]
                obs = [item for item in obs if item]
                for index, left in enumerate(obs):
                    for right in obs[index + 1 :]:
                        mask_left = self.facts.array(left.get("processed_mask_ref"), bool)
                        mask_right = self.facts.array(right.get("processed_mask_ref"), bool)
                        iou, containment = legacy._mask_metrics(mask_left, mask_right) if mask_left is not None and mask_right is not None else (0.0, 0.0)
                        distance = legacy._normalised_center_distance(left, right)
                        subtype = "MANY_TO_ONE_DUPLICATE_LIKE" if iou > 0.85 or containment > 0.95 else "MANY_TO_ONE_SEPARATED"
                        certainty = "AMBIGUOUS_MAPPING_RISK"
                        self.add(
                            "ASSOC-004", "association", subtype, certainty,
                            severity="HIGH" if subtype.endswith("SEPARATED") else "MEDIUM",
                            scope={
                                "frame_uid": frame_uid,
                                "object_uid": target,
                                "obs_uids": [left.get("obs_uid"), right.get("obs_uid")],
                                "association_candidate_roles_by_observation": {
                                    row.get("obs_uid"): self._association_scope(row).get("association_candidate_roles", {})
                                    for row in group
                                    if row.get("obs_uid") in {left.get("obs_uid"), right.get("obs_uid")}
                                },
                                "alternate_object_uids": list(dict.fromkeys(
                                    uid
                                    for row in group
                                    if row.get("obs_uid") in {left.get("obs_uid"), right.get("obs_uid")}
                                    for uid in self._association_scope(row).get("alternate_object_uids", [])
                                    if uid
                                )),
                            },
                            metrics={"mask_iou": iou, "mask_containment": containment, "normalised_center_distance": distance},
                            hypotheses=[{"name": "duplicate_proposal" if subtype.endswith("LIKE") else "false_association_or_parts", "support": ["same_frame_many_to_one"]}],
                            vetoes=["many_to_one_is_legal_policy"], route="VLM_REVIEW",
                        )

        semantic_outliers: list[tuple[str, dict, float, float]] = []
        for object_uid in self.facts.object_by_uid:
            members = self.facts.member_observations(object_uid)
            features = [self.facts.array(item.get("image_feat_ref")) for item in members]
            valid = [(item, value.reshape(-1)) for item, value in zip(members, features) if value is not None and value.size]
            if len(valid) < 4:
                continue
            matrix = np.stack([value for _, value in valid])
            norms = np.linalg.norm(matrix, axis=1)
            cosine = matrix @ matrix.T / np.maximum(norms[:, None] * norms[None, :], 1e-9)
            medoid = matrix[int(np.argmax(cosine.sum(axis=1)))]
            distances = 1.0 - (matrix @ medoid) / np.maximum(np.linalg.norm(matrix, axis=1) * np.linalg.norm(medoid), 1e-9)
            median = float(np.median(distances)); mad = float(np.median(np.abs(distances - median)))
            for (obs, _), distance in zip(valid, distances):
                robust_z = float((distance - median) / (1.4826 * mad + 1e-9))
                if robust_z > float(cfg["semantic_outlier_z"]):
                    semantic_outliers.append((object_uid, obs, robust_z, float(1.0 - distance)))
                    self.add(
                        "ASSOC-005", "association", "SEMANTIC_MEMBER_OUTLIER",
                        "AMBIGUOUS_MAPPING_RISK", severity="MEDIUM",
                        scope=(
                            self._association_scope(
                                self.facts.association_by_obs.get(obs.get("obs_uid"), {}),
                                extra_scope={"object_uid": object_uid},
                            )
                            if self.facts.association_by_obs.get(obs.get("obs_uid"))
                            else {"object_uid": object_uid, "obs_uid": obs.get("obs_uid")}
                        ),
                        metrics={"semantic_outlier_robust_z": robust_z, "semantic_support_to_medoid": float(1.0 - distance)},
                        hypotheses=[{"name": "false_association_or_visual_outlier", "support": ["robust_semantic_outlier"]}], route="LOG_ONLY",
                    )

        for object_uid, obs, robust_z, semantic_support in semantic_outliers:
            pcd = self.facts.array(obs.get("pcd_ref"))
            rest = [self.facts.array(item.get("pcd_ref")) for item in self.facts.member_observations(object_uid) if item.get("obs_uid") != obs.get("obs_uid")]
            rest = [value for value in rest if value is not None and len(value)]
            if pcd is None or not rest:
                self.missing("ASSOC-006", "association", "leave-one-out observation PCD", affected_count=1)
                continue
            core = np.concatenate(rest, axis=0)
            geo = legacy._pcd_overlap(pcd, core, self.radius)
            if geo < float(cfg["geometric_support_low"]):
                self.add(
                    "ASSOC-006", "association", "LOW_GEOMETRIC_MEMBER_SUPPORT",
                    "AMBIGUOUS_MAPPING_RISK", severity="HIGH",
                    scope=(
                        self._association_scope(
                            self.facts.association_by_obs.get(obs.get("obs_uid"), {}),
                            extra_scope={"object_uid": object_uid},
                        )
                        if self.facts.association_by_obs.get(obs.get("obs_uid"))
                        else {"object_uid": object_uid, "obs_uid": obs.get("obs_uid")}
                    ),
                    metrics={"semantic_outlier_robust_z": robust_z, "semantic_support": semantic_support, "geometric_support": geo},
                    hypotheses=[{"name": "false_association", "support": ["semantic_outlier", "low_geometric_support"]}],
                    vetoes=["new_viewpoint_or_occlusion_not_excluded"], route="VLM_REVIEW",
                )
                self._approximate_counterfactual(object_uid, obs, core, geo)

        self.missing("ASSOC-010", "association", "trajectory-level identity alignment", affected_count=1)

    def _approximate_counterfactual(self, target_uid: str, obs: dict, target_core: np.ndarray, target_geo: float) -> None:
        cfg = self.thresholds["association"]
        feature = self.facts.array(obs.get("image_feat_ref"))
        pcd = self.facts.array(obs.get("pcd_ref"))
        if feature is None or pcd is None:
            return
        target_feature = self.facts.object_feature(target_uid)
        target_sem = legacy._cosine(feature, target_feature)
        best = None
        for uid in self.facts.object_by_uid:
            if uid == target_uid:
                continue
            sem = legacy._cosine(feature, self.facts.object_feature(uid))
            if sem is None:
                continue
            geo = legacy._pcd_overlap(pcd, self.facts.object_pcd(uid), self.radius)
            score = sem + geo
            if best is None or score > best[0]:
                best = (score, uid, sem, geo)
        target_score = float(target_sem or 0) + float(target_geo)
        if best and best[0] - target_score >= float(cfg["counterfactual_gain"]) and best[2] > float(target_sem or 0) and best[3] > target_geo:
            assoc = self.facts.association_by_obs.get(obs.get("obs_uid"), {})
            self.add(
                "ASSOC-009", "association", "APPROXIMATE_LEAVE_ONE_OUT_REASSOCIATION",
                "LIKELY_MAPPING_CONFLICT", severity="HIGH",
                scope=(
                    self._association_scope(
                        assoc,
                        extra_scope={"object_uid": target_uid},
                        counterfactual_alternate=best[1],
                    )
                    if assoc
                    else {
                        "obs_uid": obs.get("obs_uid"),
                        "object_uid": target_uid,
                        "chosen_target_object_uid": target_uid,
                        "counterfactual_alternate_object_uid": best[1],
                        "alternate_object_uids": [best[1]],
                        "association_candidate_roles": {
                            "chosen_target": target_uid,
                            "counterfactual_alternate": best[1],
                        },
                    }
                ),
                metrics={"target_semantic": target_sem, "target_geometric": target_geo, "alternate_semantic": best[2], "alternate_geometric": best[3], "counterfactual_gain": best[0] - target_score, "approximation": "final_map"},
                hypotheses=[{"name": "false_association", "support": ["semantic_outlier", "low_target_geometry", "stronger_alternate"]}],
                missing_evidence=["event_time_candidate_object_versions"], route="HUMAN_REVIEW",
            )

    def _fusion(self) -> None:
        cfg = self.thresholds["fusion"]
        for object_uid, versions in self.facts.versions_by_object.items():
            for before, after in zip(versions, versions[1:]):
                if after.get("operation") not in {"OBS_ASSOCIATE", "OBJECT_MERGE"}:
                    continue
                before_center = (before.get("bbox") or {}).get("center", before.get("bbox_center"))
                after_center = (after.get("bbox") or {}).get("center", after.get("bbox_center"))
                before_extent = (before.get("bbox") or {}).get("extent", before.get("bbox_extent"))
                after_extent = (after.get("bbox") or {}).get("extent", after.get("bbox_extent"))
                center_shift = None
                if _finite(before_center) and _finite(after_center) and _finite(before_extent):
                    center_shift = float(np.linalg.norm(np.asarray(after_center) - np.asarray(before_center)) / max(float(np.linalg.norm(before_extent)), 1e-9))
                volume_before, volume_after = _volume(before_extent), _volume(after_extent)
                volume_ratio = volume_after / volume_before if volume_before and volume_after is not None else None
                points_before = float(before.get("n_points") or 0); points_after = float(after.get("n_points") or 0)
                point_ratio = points_after / points_before if points_before else None
                shocks = []
                if center_shift is not None and center_shift > float(cfg["center_shift_normalized"]): shocks.append("center_shift")
                if volume_ratio is not None and volume_ratio > float(cfg["volume_growth_ratio"]): shocks.append("bbox_volume_growth")
                if point_ratio is not None and point_ratio > float(cfg["point_growth_ratio"]): shocks.append("point_count_growth")
                if len(shocks) >= 2:
                    self.add(
                        "FUSE-007", "fusion", "FUSION_SHOCK",
                        "AMBIGUOUS_MAPPING_RISK", severity="HIGH",
                        scope={"object_uid": object_uid, "object_version_uid": after.get("object_version_uid"), "event_uid": after.get("trigger_event_uid")},
                        metrics={"signals": shocks, "center_shift_normalized": center_shift, "volume_growth_ratio": volume_ratio, "point_growth_ratio": point_ratio},
                        hypotheses=[{"name": "bad_fusion_or_large_new_view", "support": shocks}],
                        vetoes=["new_viewpoint_not_excluded"], route="VLM_REVIEW",
                    )

        proxy_count = 0
        for item in self.facts.pair_decisions:
            forward = item.get("overlap_a_to_b", item.get("overlap_forward"))
            backward = item.get("overlap_b_to_a", item.get("overlap_backward"))
            if forward is not None and backward is not None and abs(float(forward) - float(backward)) > float(cfg["overlap_asymmetry"]):
                self.add(
                    "FUSE-008", "fusion", "ASYMMETRIC_OVERLAP_RISK",
                    "AMBIGUOUS_MAPPING_RISK", severity="MEDIUM",
                    scope={"merge_transaction_uid": item.get("merge_transaction_uid"), "object_uids": [item.get("source_object_uid"), item.get("target_object_uid")]},
                    metrics={"overlap_forward": forward, "overlap_backward": backward, "decision": item.get("decision")},
                    hypotheses=[{"name": "part_whole_or_scale_asymmetry", "support": ["asymmetric_overlap"]}],
                    vetoes=["part_whole_not_excluded"], route="VLM_REVIEW",
                )
            if str(item.get("text_similarity_source", "")).upper() in {"VISUAL_PROXY", "VISUAL"}:
                proxy_count += 1
        if proxy_count:
            self.add(
                "FUSE-009", "fusion", "TEXT_SIMILARITY_IS_VISUAL_PROXY",
                "INSUFFICIENT_EVIDENCE", severity="LOW",
                metrics={"pair_decision_count": proxy_count, "independent_semantic_evidence_count": 1},
                missing_evidence=["independent text semantic feature"],
                hypotheses=[{"name": "semantic_evidence_not_independent", "support": ["visual_proxy"]}],
                route="LOG_ONLY",
            )

        filter_events = [item for item in self.facts.events if item.get("event_type") == "OBJECT_FILTER"]
        for event in filter_events:
            before = event.get("before_summary") or {}
            min_points = int(self.context.config.get("obj_min_points") or 0)
            min_detections = int(self.context.config.get("obj_min_detections") or 0)
            should_filter = int(before.get("n_points") or 0) < min_points or int(before.get("num_detections") or 0) < min_detections
            if not should_filter:
                self.add(
                    "FUSE-011", "fusion", "FILTER_POLICY_MISMATCH",
                    "CONFIRMED_SYSTEM_ERROR", severity="HIGH",
                    scope={"event_uid": event.get("event_uid"), "object_uid": event.get("object_uid")},
                    metrics={"n_points": before.get("n_points"), "num_detections": before.get("num_detections"), "obj_min_points": min_points, "obj_min_detections": min_detections},
                    hypotheses=[{"name": "filter_execution_mismatch", "support": ["policy_not_satisfied"]}], route="BLOCK_RUN",
                )

    def _object_identity(self) -> None:
        cfg = self.thresholds["object_pair"]
        active = list(self.facts.object_by_uid.values())
        for index, left in enumerate(active):
            for right in active[index + 1 :]:
                left_uid, right_uid = left.get("object_uid"), right.get("object_uid")
                clip = legacy._cosine(self.facts.object_feature(left_uid), self.facts.object_feature(right_uid))
                distance = legacy._normalised_center_distance(left, right, prefix="bbox")
                if clip is None or distance is None or clip <= float(cfg["duplicate_clip_similarity"]) or distance >= float(cfg["center_distance_normalized"]):
                    continue
                overlap = legacy._symmetric_overlap(self.facts.object_pcd(left_uid), self.facts.object_pcd(right_uid), self.radius)
                if overlap <= float(cfg["duplicate_symmetric_support"]):
                    continue
                covis = self.facts.separated_covisibility(left_uid, right_uid)
                vetoes = []
                if covis >= int(cfg["repeated_separated_covisibility_veto"]):
                    vetoes.append("repeated_separated_co_visibility")
                if self._part_whole_veto(left, right):
                    vetoes.append("part_whole_not_excluded")
                self.add(
                    "OBJ-002", "object_identity", "POSSIBLE_DUPLICATE_OBJECT",
                    "AMBIGUOUS_MAPPING_RISK" if vetoes else "LIKELY_MAPPING_CONFLICT",
                    severity="HIGH",
                    scope={"object_uids": [left_uid, right_uid]},
                    metrics={"clip_similarity": clip, "symmetric_3d_support": overlap, "normalised_center_distance": distance, "separated_covisibility_count": covis},
                    hypotheses=[{"name": "false_split_duplicate_object", "support": ["semantic_similarity", "3d_overlap", "spatial_proximity"]}],
                    vetoes=vetoes, route="VLM_REVIEW" if vetoes else "HUMAN_REVIEW",
                )

        for object_uid, item in self.facts.object_by_uid.items():
            members = self.facts.member_observations(object_uid)
            classes = Counter(str(obs.get("class_name") or "unknown") for obs in members)
            dominant_ratio = max(classes.values()) / len(members) if members else 0.0
            unique_frames = len({obs.get("frame_uid") for obs in members})
            confidences = [float(obs.get("confidence") or 0) for obs in members]
            signals = []
            if unique_frames <= 1: signals.append("single_view")
            if len(members) <= 1: signals.append("single_observation")
            if confidences and float(np.median(confidences)) <= 0.30: signals.append("low_median_confidence")
            if int(item.get("n_points") or 0) <= 10: signals.append("few_points")
            if len(signals) >= int(self.thresholds["weak_object"]["required_low_support_signals"]):
                self.add(
                    "OBJ-003", "object_identity", "LOW_SUPPORT_OBJECT",
                    "AMBIGUOUS_MAPPING_RISK", severity="MEDIUM",
                    scope={"object_uid": object_uid}, metrics={"signals": signals, "unique_frame_count": unique_frames, "member_count": len(members)},
                    hypotheses=[{"name": "weak_or_noise_node", "support": signals}],
                    vetoes=["real_small_object_not_excluded"], route="VLM_REVIEW",
                )
            if len(classes) >= int(self.thresholds["identity"]["class_count_high"]) and dominant_ratio < float(self.thresholds["identity"]["class_dominant_ratio_low"]):
                self.add(
                    "OBJ-005", "object_identity", "IDENTITY_INSTABILITY",
                    "AMBIGUOUS_MAPPING_RISK", severity="HIGH",
                    scope={"object_uid": object_uid}, metrics={"class_histogram": dict(classes), "dominant_class_ratio": dominant_ratio},
                    hypotheses=[{"name": "false_merge_or_label_instability", "support": ["high_class_entropy"]}], route="VLM_REVIEW",
                )
        self.missing("OBJ-006", "object_identity", "GT or full-frame visibility coverage", affected_count=1)

    def _summary(self, gate_passed: bool, root_causes: list[dict], elapsed: float) -> dict:
        rows = [item.to_dict() for item in self.findings]
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.context.run_id,
            "scene_id": self.context.scene_id,
            "gate_status": "PASS" if gate_passed else "FAIL",
            "mapping_mutated": False,
            "finding_count": len(rows),
            "root_cause_count": len(root_causes),
            "certainty_counts": dict(Counter(item["certainty"] for item in rows)),
            "severity_counts": dict(Counter(item["severity"] for item in rows)),
            "stage_counts": dict(Counter(item["stage"] for item in rows)),
            "checker_counts": dict(Counter(item["checker_id"] for item in rows)),
            "route_counts": dict(Counter(item["route"] for item in rows)),
            "insufficient_evidence_count": sum(item["certainty"] == "INSUFFICIENT_EVIDENCE" for item in rows),
            "policy_source": self.context.policy_source,
            "audit_config_version": self.context.audit_config.get("version"),
            "elapsed_seconds": round(float(elapsed), 3),
        }


def resolve_root_causes(findings: Iterable[Finding]) -> list[dict]:
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        if finding.stage == "system" or finding.certainty == "INSUFFICIENT_EVIDENCE":
            continue
        scope = finding.scope
        if scope.get("object_uid"):
            key = f"object:{scope['object_uid']}"
        elif scope.get("object_uids"):
            key = "objects:" + ":".join(sorted(map(str, scope["object_uids"])))
        elif scope.get("obs_uids"):
            key = "observations:" + ":".join(sorted(map(str, scope["obs_uids"])))
        elif scope.get("obs_uid"):
            key = f"observation:{scope['obs_uid']}"
        else:
            continue
        grouped[key].append(finding)
    roots = []
    for _, items in grouped.items():
        stages = {item.stage for item in items}
        if len(items) < 2 and not any(item.certainty == "LIKELY_MAPPING_CONFLICT" for item in items):
            continue
        ordered = sorted(
            items,
            key=lambda item: (
                STAGE_ORDER.get(item.stage, 99),
                SEVERITY_ORDER.get(item.severity, 99),
                CERTAINTY_ORDER.get(item.certainty, 99),
                item.finding_uid,
            ),
        )
        primary = ordered[0]
        alternatives = []
        for item in ordered[1:]:
            for hypothesis in item.hypotheses:
                name = hypothesis.get("name")
                if name and name not in alternatives:
                    alternatives.append(name)
        roots.append(
            {
                "root_cause_uid": f"rc_{len(roots) + 1:06d}",
                "primary_hypothesis": (primary.hypotheses[0].get("name") if primary.hypotheses else primary.subtype),
                "primary_stage": primary.stage,
                "supporting_findings": [item.finding_uid for item in ordered],
                "explains": sorted({item.subtype for item in items}),
                "alternative_hypotheses": alternatives[:5],
                "vetoes": sorted({value for item in items for value in item.vetoes}),
                "missing_evidence": sorted({value for item in items for value in item.missing_evidence}),
                "certainty": min((item.certainty for item in items), key=lambda value: CERTAINTY_ORDER.get(value, 99)),
            }
        )
    return roots


def _uid_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]
    return list(dict.fromkeys(str(value) for value in values if value))


def _case_object_roles(finding: Finding) -> dict[str, set[str]]:
    """Return every object needed to understand a case and why it is shown."""
    scope = finding.scope
    roles: dict[str, set[str]] = defaultdict(set)

    def add(uid: Any, role: str) -> None:
        if uid:
            roles[str(uid)].add(role)

    add(scope.get("object_uid"), "primary_object")
    for uid in _uid_list(scope.get("object_uids")):
        add(uid, "compared_object")
    add(scope.get("chosen_target_object_uid"), "chosen_target")
    add(scope.get("spatial_top_object_uid"), "spatial_top_candidate")
    add(scope.get("visual_top_object_uid"), "visual_top_candidate")
    aggregate = _uid_list(scope.get("aggregate_top_object_uids"))
    if aggregate:
        add(aggregate[0], "aggregate_top1")
    if len(aggregate) > 1:
        add(aggregate[1], "aggregate_top2")
    add(scope.get("counterfactual_alternate_object_uid"), "counterfactual_alternate")
    for uid in _uid_list(scope.get("alternate_object_uids")):
        add(uid, "alternate_candidate")
    for role, uid in (scope.get("association_candidate_roles") or {}).items():
        add(uid, str(role))
    for role_map in (scope.get("association_candidate_roles_by_observation") or {}).values():
        for role, uid in (role_map or {}).items():
            add(uid, str(role))
    return roles


def _trigger_observation_uids(finding: Finding, facts: FactStore) -> list[str]:
    scope = finding.scope
    uids = _uid_list(scope.get("obs_uids"))
    uids.extend(_uid_list(scope.get("obs_uid")))
    event_uids = _uid_list(scope.get("event_uid"))
    version_uid = scope.get("object_version_uid")
    if version_uid and version_uid in facts.version_by_uid:
        version = facts.version_by_uid[version_uid]
        event_uids.extend(_uid_list(version.get("trigger_event_uid")))
        object_uid = version.get("object_uid")
        versions = facts.versions_by_object.get(object_uid, [])
        try:
            index = next(i for i, item in enumerate(versions) if item.get("object_version_uid") == version_uid)
        except StopIteration:
            index = -1
        if index >= 0:
            before_members = set(versions[index - 1].get("member_observation_uids", [])) if index else set()
            after_members = set(version.get("member_observation_uids", []))
            uids.extend(sorted(after_members - before_members))
    for event_uid in dict.fromkeys(event_uids):
        event = facts.event_by_uid.get(event_uid, {})
        for key in ("obs_uid", "observation_uid"):
            uids.extend(_uid_list(event.get(key)))
    return [uid for uid in dict.fromkeys(uids) if uid in facts.obs_by_uid]


def _camera_position(obs: dict, facts: FactStore) -> Optional[np.ndarray]:
    frame = facts.frame_by_uid.get(obs.get("frame_uid"), {})
    pose = frame.get("pose")
    if pose is None:
        return None
    try:
        array = np.asarray(pose, dtype=float)
        if array.shape[0] >= 3 and array.shape[1] >= 4:
            return array[:3, 3]
    except Exception:
        return None
    return None


def _semantic_conflict_order(members: list[dict], facts: FactStore) -> list[dict]:
    valid = []
    for item in members:
        feature = facts.array(item.get("image_feat_ref"))
        if feature is not None and feature.size:
            valid.append((item, feature.reshape(-1)))
    if len(valid) < 2:
        return []
    matrix = np.stack([feature for _, feature in valid])
    norms = np.linalg.norm(matrix, axis=1)
    cosine = matrix @ matrix.T / np.maximum(norms[:, None] * norms[None, :], 1e-9)
    medoid_index = int(np.argmax(cosine.sum(axis=1)))
    similarity = cosine[:, medoid_index]
    return [valid[index][0] for index in np.argsort(similarity)]


def _select_object_views(
    object_uid: str,
    facts: FactStore,
    trigger_uids: set[str],
    limit: int,
    object_version_uid: Optional[str] = None,
) -> list[dict]:
    members = facts.member_observations(object_uid, object_version_uid)
    if not members or limit <= 0:
        return []
    selected: dict[str, dict] = {}

    def choose(item: Optional[dict], reason: str) -> None:
        if not item or not item.get("obs_uid"):
            return
        uid = str(item["obs_uid"])
        if uid not in selected and len(selected) >= limit:
            return
        selected.setdefault(uid, {"observation": item, "selection_reasons": []})
        if reason not in selected[uid]["selection_reasons"]:
            selected[uid]["selection_reasons"].append(reason)

    ordered_time = sorted(members, key=lambda item: (_frame_number(item.get("frame_uid")), str(item.get("obs_uid"))))
    choose(ordered_time[0], "earliest_creation_view")
    choose(max(members, key=lambda item: float(item.get("n_points") or 0)), "highest_point_contribution")
    choose(max(members, key=lambda item: float(item.get("confidence") or 0)), "highest_detector_confidence")
    trigger_members = [item for item in ordered_time if str(item.get("obs_uid")) in trigger_uids]
    choose(trigger_members[0] if trigger_members else None, "anomaly_trigger_view")
    semantic_order = _semantic_conflict_order(members, facts)
    choose(semantic_order[0] if semantic_order else None, "largest_semantic_conflict")

    anchor = _camera_position(ordered_time[0], facts)
    camera_candidates = []
    if anchor is not None:
        for item in members:
            position = _camera_position(item, facts)
            if position is not None:
                camera_candidates.append((float(np.linalg.norm(position - anchor)), item))
    choose(max(camera_candidates, key=lambda pair: pair[0])[1] if camera_candidates else ordered_time[-1], "largest_camera_viewpoint_difference")

    # Fill unused slots with temporally distant views without changing the six diagnostic labels above.
    while len(selected) < min(limit, len(members)):
        selected_frames = [_frame_number(value["observation"].get("frame_uid")) for value in selected.values()]
        remaining = [item for item in members if str(item.get("obs_uid")) not in selected]
        if not remaining:
            break
        next_item = max(
            remaining,
            key=lambda item: min(
                abs(_frame_number(item.get("frame_uid")) - frame) for frame in selected_frames
            ) if selected_frames else 0,
        )
        choose(next_item, "temporal_diversity_fill")
    return list(selected.values())


def _case_observation_selection(
    finding: Finding,
    facts: FactStore,
    limit_per_object: int,
    max_total: Optional[int] = None,
) -> list[dict]:
    trigger_uids = _trigger_observation_uids(finding, facts)
    explicit_uids = set(_uid_list(finding.scope.get("obs_uids")) + _uid_list(finding.scope.get("obs_uid")))
    object_roles = _case_object_roles(finding)
    object_versions = finding.scope.get("association_candidate_object_versions") or {}
    for obs_uid in explicit_uids:
        for owner in facts.ownership.get(obs_uid, []):
            object_roles[str(owner)].add("observation_owner")

    records: dict[str, dict] = {}

    def include(obs: dict, *, reasons: Iterable[str], object_uid: Optional[str] = None) -> None:
        uid = str(obs.get("obs_uid"))
        if not uid:
            return
        record = records.setdefault(
            uid,
            {
                "observation": obs,
                "selection_reasons": [],
                "object_uids": [],
                "object_roles": [],
            },
        )
        for reason in reasons:
            if reason not in record["selection_reasons"]:
                record["selection_reasons"].append(reason)
        if object_uid:
            if object_uid not in record["object_uids"]:
                record["object_uids"].append(object_uid)
            for role in sorted(object_roles.get(object_uid, [])):
                if role not in record["object_roles"]:
                    record["object_roles"].append(role)

    for uid in trigger_uids:
        include(
            facts.obs_by_uid[uid],
            reasons=[
                "suspect_observation" if uid in explicit_uids else "event_or_version_trigger"
            ],
        )

    per_object = {
        uid: _select_object_views(
            uid,
            facts,
            set(trigger_uids),
            limit_per_object,
            object_versions.get(uid),
        )
        for uid in object_roles
    }
    total_limit = max_total if max_total is not None else max(
        len(trigger_uids), limit_per_object * max(1, len(per_object))
    )
    # First pass guarantees at least one representative view for every candidate object.
    for object_uid, selections in per_object.items():
        if selections and (len(records) < total_limit or str(selections[0]["observation"].get("obs_uid")) in records):
            include(
                selections[0]["observation"],
                reasons=selections[0]["selection_reasons"],
                object_uid=object_uid,
            )
    # Round-robin prevents the chosen target from consuming the entire packet before alternates appear.
    for offset in range(1, limit_per_object):
        for object_uid, selections in per_object.items():
            if offset >= len(selections) or len(records) >= total_limit:
                continue
            include(
                selections[offset]["observation"],
                reasons=selections[offset]["selection_reasons"],
                object_uid=object_uid,
            )
    return list(records.values())


def _case_observations(finding: Finding, facts: FactStore, limit: int) -> list[dict]:
    """Compatibility wrapper returning mixed diagnostic views, including alternates."""
    return [
        item["observation"]
        for item in _case_observation_selection(finding, facts, limit)
    ]


def _finding_metrics(finding: Finding) -> dict:
    return {item.get("name"): item.get("value") for item in finding.proven_facts}


def _above(value: Any, threshold: float) -> float:
    if value is None:
        return 0.0
    return float(np.clip((float(value) - threshold) / max(abs(threshold), 0.05), 0.0, 3.0))


def _below(value: Any, threshold: float) -> float:
    if value is None:
        return 0.0
    return float(np.clip((threshold - float(value)) / max(abs(threshold), 0.05), 0.0, 3.0))


def _threshold_strength(finding: Finding, config: dict) -> float:
    metrics = _finding_metrics(finding)
    thresholds = config.get("thresholds", {})
    checker = finding.checker_id
    values: list[float] = []
    if checker == "DET-001":
        cfg = thresholds.get("duplicate_proposal", {})
        values = [
            _above(metrics.get("mask_iou"), float(cfg.get("mask_iou", 0.85))),
            _above(metrics.get("mask_containment"), float(cfg.get("containment", 0.95))),
            _above(metrics.get("clip_similarity"), float(cfg.get("clip_similarity", 0.90))),
            _above(metrics.get("symmetric_3d_support"), float(cfg.get("symmetric_3d_support", 0.70))),
        ]
    elif checker == "DET-002":
        needed = float(thresholds.get("false_positive", {}).get("required_signals_likely", 3))
        values = [_above(len(metrics.get("signals") or []), needed)]
    elif checker in {"SEG-002", "GEO-003"}:
        cfg = thresholds.get("segmentation" if checker == "SEG-002" else "geometry", {})
        threshold = cfg.get("background_second_cluster_ratio", cfg.get("multi_cluster_second_ratio", 0.20))
        values = [_above(metrics.get("second_cluster_ratio"), float(threshold))]
    elif checker == "SEG-004":
        values = [_above(metrics.get("clip_similarity"), 0.90), _below(metrics.get("normalised_center_distance"), 0.50)]
    elif checker == "SEG-005":
        cfg = thresholds.get("segmentation", {})
        values = [
            _above(metrics.get("loss_ratio"), float(cfg.get("subtraction_loss_ratio", 0.50))),
            _above(metrics.get("second_component_ratio"), float(cfg.get("fragmented_second_component_ratio", 0.20))),
        ]
    elif checker == "GEO-004":
        values = [_above(metrics.get("normalised_center_jump"), float(thresholds.get("geometry", {}).get("center_jump_normalized", 1.0)))]
    elif checker == "GEO-005":
        values = [_below(metrics.get("point_keep_ratio"), float(thresholds.get("geometry", {}).get("denoise_keep_ratio", 0.30)))]
    elif checker == "ASSOC-002":
        values = [_below(metrics.get("margin"), float(thresholds.get("association", {}).get("low_margin", 0.05)))]
    elif checker == "ASSOC-003":
        spatial, visual = metrics.get("spatial_top") or {}, metrics.get("visual_top") or {}
        values = [
            abs(float(spatial.get("spatial_score") or 0) - float(visual.get("spatial_score") or 0)),
            abs(float(spatial.get("visual_score") or 0) - float(visual.get("visual_score") or 0)),
        ]
    elif checker == "ASSOC-004":
        values = [
            _above(metrics.get("mask_iou"), 0.85),
            _above(metrics.get("mask_containment"), 0.95),
            _above(metrics.get("normalised_center_distance"), 0.50),
        ]
    elif checker in {"ASSOC-005", "ASSOC-006"}:
        cfg = thresholds.get("association", {})
        values = [_above(metrics.get("semantic_outlier_robust_z"), float(cfg.get("semantic_outlier_z", 3.0)))]
        if checker == "ASSOC-006":
            values.append(_below(metrics.get("geometric_support"), float(cfg.get("geometric_support_low", 0.15))))
    elif checker == "ASSOC-007":
        cfg = thresholds.get("association", {})
        values = [
            _above(metrics.get("visual_score"), float(cfg.get("high_visual", 0.85))),
            _below(metrics.get("spatial_score"), float(cfg.get("low_spatial", 0.05))),
        ]
    elif checker == "ASSOC-009":
        values = [_above(metrics.get("counterfactual_gain"), float(thresholds.get("association", {}).get("counterfactual_gain", 0.10)))]
    elif checker == "FUSE-007":
        cfg = thresholds.get("fusion", {})
        values = [
            _above(metrics.get("center_shift_normalized"), float(cfg.get("center_shift_normalized", 0.30))),
            _above(metrics.get("volume_growth_ratio"), float(cfg.get("volume_growth_ratio", 1.50))),
            _above(metrics.get("point_growth_ratio"), float(cfg.get("point_growth_ratio", 1.50))),
        ]
    elif checker == "FUSE-008":
        asymmetry = None
        if metrics.get("overlap_forward") is not None and metrics.get("overlap_backward") is not None:
            asymmetry = abs(float(metrics["overlap_forward"]) - float(metrics["overlap_backward"]))
        values = [_above(asymmetry, float(thresholds.get("fusion", {}).get("overlap_asymmetry", 0.50)))]
    elif checker == "OBJ-002":
        cfg = thresholds.get("object_pair", {})
        values = [
            _above(metrics.get("clip_similarity"), float(cfg.get("duplicate_clip_similarity", 0.90))),
            _above(metrics.get("symmetric_3d_support"), float(cfg.get("duplicate_symmetric_support", 0.75))),
            _below(metrics.get("normalised_center_distance"), float(cfg.get("center_distance_normalized", 0.20))),
        ]
    elif checker == "OBJ-005":
        cfg = thresholds.get("identity", {})
        values = [
            _below(metrics.get("dominant_class_ratio"), float(cfg.get("class_dominant_ratio_low", 0.60))),
            _above(len(metrics.get("class_histogram") or {}), float(cfg.get("class_count_high", 3))),
        ]
    valid = [value for value in values if np.isfinite(value)]
    return round(float(np.mean(valid)) if valid else 0.0, 6)


def _primary_entity_key(finding: Finding, facts: FactStore) -> str:
    scope = finding.scope
    if scope.get("object_uid"):
        return f"object:{scope['object_uid']}"
    object_uids = _uid_list(scope.get("object_uids"))
    if object_uids:
        return "objects:" + ":".join(sorted(object_uids))
    obs_uids = _uid_list(scope.get("obs_uids")) + _uid_list(scope.get("obs_uid"))
    owners = sorted({owner for uid in obs_uids for owner in facts.ownership.get(uid, [])})
    if owners:
        return "owners:" + ":".join(owners)
    if obs_uids:
        return "observations:" + ":".join(sorted(set(obs_uids)))
    if scope.get("frame_uid"):
        return f"frame:{scope['frame_uid']}"
    return f"finding:{finding.finding_uid}"


def _finding_entity_keys(finding: Finding, facts: FactStore) -> set[str]:
    scope = finding.scope
    keys = {_primary_entity_key(finding, facts)}
    for uid in _uid_list(scope.get("object_uid")) + _uid_list(scope.get("object_uids")):
        keys.add(f"object:{uid}")
    for uid in _uid_list(scope.get("obs_uid")) + _uid_list(scope.get("obs_uids")):
        keys.add(f"observation:{uid}")
        for owner in facts.ownership.get(uid, []):
            keys.add(f"object:{owner}")
    return keys


def _case_candidates(
    findings: list[Finding], facts: FactStore, config: dict
) -> tuple[list[Finding], dict[str, dict]]:
    eligible = [
        item for item in findings
        if item.certainty in {"LIKELY_MAPPING_CONFLICT", "AMBIGUOUS_MAPPING_RISK"}
    ]
    by_entity: dict[str, list[Finding]] = defaultdict(list)
    for item in eligible:
        for key in _finding_entity_keys(item, facts):
            by_entity[key].append(item)
    repeated = Counter((item.checker_id, _primary_entity_key(item, facts)) for item in eligible)
    details: dict[str, dict] = {}
    for item in eligible:
        entity_keys = _finding_entity_keys(item, facts)
        downstream = {
            other.finding_uid
            for key in entity_keys
            for other in by_entity[key]
            if STAGE_ORDER.get(other.stage, 99) > STAGE_ORDER.get(item.stage, 99)
        }
        support = {
            str(value)
            for hypothesis in item.hypotheses
            for value in (hypothesis.get("support") or [])
            if value
        }
        redundancy = repeated[(item.checker_id, _primary_entity_key(item, facts))]
        components = {
            "severity": float({"CRITICAL": 8.0, "HIGH": 6.0, "MEDIUM": 4.0, "LOW": 2.0}.get(item.severity, 0.0)),
            "certainty": float({"LIKELY_MAPPING_CONFLICT": 4.0, "AMBIGUOUS_MAPPING_RISK": 2.0}.get(item.certainty, 0.0)),
            "threshold_exceedance": round(2.5 * _threshold_strength(item, config), 6),
            "independent_evidence": round(1.25 * min(len(support), 4), 6),
            "downstream_pollution": round(1.5 * min(math.log1p(len(downstream)), 3.0), 6),
            "veto_penalty": round(-1.5 * min(len(item.vetoes), 3), 6),
            "missing_evidence_penalty": round(-1.0 * min(len(item.missing_evidence), 3), 6),
            "redundancy_penalty": round(-min(math.log1p(max(redundancy - 1, 0)), 2.0), 6),
            "independent_evidence_count": len(support),
            "downstream_finding_count": len(downstream),
            "same_checker_entity_count": redundancy,
        }
        score_keys = (
            "severity", "certainty", "threshold_exceedance", "independent_evidence",
            "downstream_pollution", "veto_penalty", "missing_evidence_penalty", "redundancy_penalty",
        )
        item.review_score = round(sum(float(components[key]) for key in score_keys), 6)
        item.review_score_components = components
        details[item.finding_uid] = {
            "primary_entity": _primary_entity_key(item, facts),
            "entity_keys": sorted(entity_keys),
            "score": item.review_score,
            "score_components": components,
        }
    ranked = sorted(eligible, key=lambda item: (-float(item.review_score or 0), item.checker_id, item.finding_uid))
    for rank, item in enumerate(ranked, 1):
        item.review_priority = rank
        details[item.finding_uid]["global_priority_rank"] = rank
    return eligible, details


def _allocate_quotas(
    counts: dict[str, int], target: int, *, minimum: int = 0, maximum: Optional[int] = None
) -> dict[str, int]:
    quotas = {key: min(value, minimum) for key, value in counts.items()}
    target = min(target, sum(counts.values()))
    while sum(quotas.values()) < target:
        available = [
            key for key, count in counts.items()
            if quotas[key] < count and (maximum is None or quotas[key] < maximum)
        ]
        if not available:
            break
        key = max(
            available,
            key=lambda name: (math.sqrt(counts[name]) / (quotas[name] + 1), counts[name], name),
        )
        quotas[key] += 1
    return quotas


def _select_case_candidates(
    findings: list[Finding], facts: FactStore, config: dict, max_cases: int
) -> tuple[list[Finding], dict]:
    settings = config.get("case_builder", {})
    eligible, details = _case_candidates(findings, facts, config)
    seed_value = settings.get("random_seed")
    seed = int(seed_value) if seed_value is not None else int(
        hashlib.sha256(facts.context.run_id.encode("utf-8")).hexdigest()[:8], 16
    )
    rng = np.random.default_rng(seed)
    calibration_target = min(
        len(eligible),
        int(round(max_cases * float(settings.get("calibration_fraction", 0.50)))),
    )
    strata: dict[str, list[Finding]] = defaultdict(list)
    for item in eligible:
        strata[f"{item.checker_id}|{item.certainty}"].append(item)
    stratum_counts = {key: len(items) for key, items in strata.items()}
    calibration_quotas = _allocate_quotas(
        stratum_counts,
        calibration_target,
        minimum=int(settings.get("min_calibration_per_stratum", 1)),
    )
    selected: list[Finding] = []
    selected_uids: set[str] = set()
    stratum_rows = []
    for key in sorted(strata):
        population = sorted(strata[key], key=lambda item: item.finding_uid)
        sample_size = calibration_quotas.get(key, 0)
        indices = sorted(rng.choice(len(population), size=sample_size, replace=False).tolist()) if sample_size else []
        probability = sample_size / len(population) if population else 0.0
        weight = 1.0 / probability if probability else None
        for index in indices:
            item = population[index]
            item.selection_cohort = "calibration_random"
            item.selection_probability = round(probability, 12)
            item.sampling_weight = round(weight, 12) if weight is not None else None
            selected.append(item)
            selected_uids.add(item.finding_uid)
        stratum_rows.append(
            {
                "stratum": key,
                "population": len(population),
                "selected": sample_size,
                "inclusion_probability": round(probability, 12),
                "sampling_weight": round(weight, 12) if weight is not None else None,
            }
        )

    priority_target = max(0, min(max_cases - len(selected), len(eligible) - len(selected)))
    remaining = [item for item in eligible if item.finding_uid not in selected_uids]
    by_checker: dict[str, list[Finding]] = defaultdict(list)
    for item in remaining:
        by_checker[item.checker_id].append(item)
    for items in by_checker.values():
        items.sort(key=lambda item: (-float(item.review_score or 0), item.finding_uid))
    checker_counts = {key: len(items) for key, items in by_checker.items()}
    checker_quotas = _allocate_quotas(
        checker_counts,
        priority_target,
        minimum=int(settings.get("min_priority_per_checker", 1)),
        maximum=int(settings.get("max_priority_per_checker", 20)),
    )
    entity_cap = int(settings.get("max_priority_cases_per_entity", 2))
    entity_usage = Counter(details[item.finding_uid]["primary_entity"] for item in selected)
    signature_seen = {
        (item.checker_id, item.subtype, details[item.finding_uid]["primary_entity"])
        for item in selected
    }
    checker_selected = Counter()
    skipped_duplicate = 0
    skipped_entity_cap = 0

    def try_select(item: Finding) -> bool:
        nonlocal skipped_duplicate, skipped_entity_cap
        primary = details[item.finding_uid]["primary_entity"]
        signature = (item.checker_id, item.subtype, primary)
        if signature in signature_seen:
            skipped_duplicate += 1
            return False
        if entity_usage[primary] >= entity_cap:
            skipped_entity_cap += 1
            return False
        item.selection_cohort = "diagnostic_priority"
        selected.append(item)
        selected_uids.add(item.finding_uid)
        signature_seen.add(signature)
        entity_usage[primary] += 1
        checker_selected[item.checker_id] += 1
        return True

    for checker in sorted(by_checker):
        for item in by_checker[checker]:
            if checker_selected[checker] >= checker_quotas.get(checker, 0):
                break
            try_select(item)
    if len(selected) < max_cases:
        for item in sorted(remaining, key=lambda value: (-float(value.review_score or 0), value.checker_id, value.finding_uid)):
            if len(selected) >= max_cases:
                break
            if item.finding_uid in selected_uids:
                continue
            if checker_selected[item.checker_id] >= int(settings.get("max_priority_per_checker", 20)):
                continue
            try_select(item)

    selected.sort(key=lambda item: (0 if item.selection_cohort == "calibration_random" else 1, item.checker_id, item.review_priority or 10**9))
    selected_rows = []
    for case_rank, item in enumerate(selected, 1):
        row = {
            "case_rank": case_rank,
            "finding_uid": item.finding_uid,
            "checker_id": item.checker_id,
            "certainty": item.certainty,
            "cohort": item.selection_cohort,
            "review_score": item.review_score,
            "review_priority": item.review_priority,
            "selection_probability": item.selection_probability,
            "sampling_weight": item.sampling_weight,
            **details[item.finding_uid],
        }
        selected_rows.append(row)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "dual_cohort_stratified_random_plus_diagnostic_priority",
        "random_seed": seed,
        "available_candidates": len(eligible),
        "max_cases": max_cases,
        "selected_count": len(selected),
        "precision_policy": "Only calibration_random may be used for weighted precision; diagnostic_priority is for root-cause discovery and must not be treated as a probability sample.",
        "cohorts": {
            "calibration_random": {
                "target": calibration_target,
                "selected": sum(1 for item in selected if item.selection_cohort == "calibration_random"),
                "stratification": ["checker_id", "certainty"],
                "strata": stratum_rows,
            },
            "diagnostic_priority": {
                "target": priority_target,
                "selected": sum(1 for item in selected if item.selection_cohort == "diagnostic_priority"),
                "checker_quotas": checker_quotas,
                "max_cases_per_checker": int(settings.get("max_priority_per_checker", 20)),
                "max_cases_per_entity": entity_cap,
                "duplicate_scope_skipped": skipped_duplicate,
                "entity_cap_skipped": skipped_entity_cap,
            },
        },
        "candidate_checker_counts": dict(Counter(item.checker_id for item in eligible)),
        "selected_checker_counts": dict(Counter(item.checker_id for item in selected)),
        "selected_cohort_counts": dict(Counter(item.selection_cohort for item in selected)),
        "selected": selected_rows,
    }
    return selected, manifest


def _safe_filename(value: Any) -> str:
    text = str(value)
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in text)


def _save_timeline(overviews: list[tuple[str, Any]], destination: Path) -> None:
    if len(overviews) <= 1:
        return
    from PIL import Image, ImageDraw, ImageOps

    thumb_size = (360, 230)
    columns = min(3, len(overviews))
    rows = int(math.ceil(len(overviews) / columns))
    canvas = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 30)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (frame_uid, overview) in enumerate(overviews):
        thumbnail = ImageOps.contain(overview, thumb_size)
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + 30)
        canvas.paste(thumbnail, (x, y + 24))
        draw.text((x + 4, y + 4), str(frame_uid), fill=(0, 0, 0))
    canvas.save(destination, quality=90)


def build_evidence_packets(
    findings: list[Finding], facts: FactStore, output_dir: Path, config: dict
) -> dict:
    settings = config.get("case_builder", {})
    if not settings.get("enabled", True):
        return {"built": 0, "warnings": []}
    max_cases = int(settings.get("max_cases", 200))
    max_images = int(settings.get("max_images_per_object", 6))
    max_total_images = int(settings.get("max_total_images_per_case", 24))
    candidates, selection_manifest = _select_case_candidates(
        findings, facts, config, max_cases
    )
    cases_root = output_dir / "cases"
    if cases_root.exists():
        shutil.rmtree(cases_root)
    cases_root.mkdir(parents=True, exist_ok=True)
    _json_dump(output_dir / "case_selection.json", selection_manifest)
    warnings = []
    built = 0
    for finding in candidates:
        selection = _case_observation_selection(
            finding, facts, max_images, max_total_images
        )
        observations = [item["observation"] for item in selection]
        case_dir = cases_root / finding.finding_uid
        case_dir.mkdir(parents=True, exist_ok=True)
        view_selection = {
            "finding_uid": finding.finding_uid,
            "selection_cohort": finding.selection_cohort,
            "observations": [
                {
                    "obs_uid": item["observation"].get("obs_uid"),
                    "frame_uid": item["observation"].get("frame_uid"),
                    "object_uids": item["object_uids"],
                    "object_roles": item["object_roles"],
                    "selection_reasons": item["selection_reasons"],
                }
                for item in selection
            ],
        }
        _json_dump(case_dir / "view_selection.json", view_selection)
        association_roles = finding.scope.get("association_candidate_roles") or {}
        if finding.stage == "association":
            role_to_selected = {}
            candidate_versions = finding.scope.get("association_candidate_object_versions") or {}
            for role, object_uid in association_roles.items():
                role_to_selected[role] = {
                    "object_uid": object_uid,
                    "object_version_uid": candidate_versions.get(object_uid),
                    "selected_observation_uids": [
                        item["observation"].get("obs_uid")
                        for item in selection
                        if object_uid in item["object_uids"]
                    ],
                }
            _json_dump(
                case_dir / "association_candidates.json",
                {
                    "suspect_observation_uids": _trigger_observation_uids(finding, facts),
                    "roles": role_to_selected,
                    "roles_by_observation": finding.scope.get("association_candidate_roles_by_observation", {}),
                    "alternate_object_uids": finding.scope.get("alternate_object_uids", []),
                },
            )
        _json_dump(
            case_dir / "metrics.json",
            {item["name"]: item["value"] for item in finding.proven_facts},
        )
        if not observations:
            warnings.append(f"{finding.finding_uid}:no_observations")
        try:
            from PIL import Image, ImageDraw
            colors = [(255, 48, 79), (0, 184, 217), (255, 171, 0), (54, 179, 126), (101, 84, 192), (255, 86, 48)]
            by_frame: dict[str, list[dict]] = defaultdict(list)
            selection_by_uid = {str(item["observation"].get("obs_uid")): item for item in selection}
            for obs in observations:
                by_frame[str(obs.get("frame_uid"))].append(obs)
            timeline_inputs = []
            overview_refs = {}
            overlay_refs = {}
            depth_refs = {}
            for frame_uid in sorted(by_frame, key=_frame_number):
                frame = facts.frame_by_uid.get(frame_uid, {})
                rgb_ref = legacy._parse_ref(frame.get("rgb_ref") or frame.get("rgb_path"))
                rgb_path = legacy._resolve_path(rgb_ref["path"], facts.context.evidence_root)
                image = Image.open(rgb_path).convert("RGB")
                overview = image.copy()
                draw = ImageDraw.Draw(overview)
                overlay = np.asarray(image, dtype=np.float32).copy()
                image_array = np.asarray(image)
                for index, obs in enumerate(by_frame[frame_uid]):
                    color = colors[index % len(colors)]
                    bbox = [int(value) for value in obs.get("bbox_2d", [0, 0, 0, 0])]
                    bbox = [
                        max(0, min(image.width, bbox[0])), max(0, min(image.height, bbox[1])),
                        max(0, min(image.width, bbox[2])), max(0, min(image.height, bbox[3])),
                    ]
                    draw.rectangle(bbox, outline=color, width=4)
                    record = selection_by_uid.get(str(obs.get("obs_uid")), {})
                    labels = record.get("object_roles") or record.get("selection_reasons") or []
                    label = f"{str(obs.get('obs_uid'))[-12:]} | {','.join(labels[:2])}"
                    draw.text((bbox[0], bbox[1]), label, fill=color)
                    mask = facts.array(obs.get("processed_mask_ref"), bool)
                    if mask is not None and mask.shape[:2] == overlay.shape[:2]:
                        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.asarray(color)
                        neutral = np.full_like(image_array, 127)
                        masked = np.where(mask[..., None], image_array, neutral)
                        Image.fromarray(masked.astype(np.uint8)).crop(tuple(bbox)).save(
                            case_dir / f"masked_crop_{obs['obs_uid']}.jpg", quality=92
                        )
                    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    margin = int(max(width, height) * 0.50)
                    crop_box = (
                        max(0, bbox[0] - margin), max(0, bbox[1] - margin),
                        min(image.width, bbox[2] + margin), min(image.height, bbox[3] + margin),
                    )
                    image.crop(crop_box).save(case_dir / f"context_crop_{obs['obs_uid']}.jpg", quality=92)
                safe_frame = _safe_filename(frame_uid)
                overview_name = f"overview_{safe_frame}.jpg"
                overlay_name = f"mask_overlay_{safe_frame}.png"
                overview.save(case_dir / overview_name, quality=92)
                Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(case_dir / overlay_name)
                timeline_inputs.append((frame_uid, overview.copy()))
                overview_refs[frame_uid] = overview_name
                overlay_refs[frame_uid] = overlay_name

                if settings.get("save_depth_overlay", True):
                    depth_ref = legacy._parse_ref(frame.get("depth_ref") or frame.get("depth_path"))
                    depth_path = legacy._resolve_path(depth_ref["path"], facts.context.evidence_root)
                    depth = np.asarray(Image.open(depth_path), dtype=float)
                    finite = depth[np.isfinite(depth) & (depth > 0)]
                    if len(finite):
                        lo, hi = np.quantile(finite, [0.02, 0.98])
                        depth_image = np.clip((depth - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
                        depth_name = f"depth_{safe_frame}.png"
                        Image.fromarray(depth_image).save(case_dir / depth_name)
                        depth_refs[frame_uid] = depth_name
            if len(timeline_inputs) > 1:
                _save_timeline(timeline_inputs, case_dir / "timeline.jpg")
            finding.evidence_refs.update(
                {
                    "frame_overviews": overview_refs,
                    "frame_mask_overlays": overlay_refs,
                    "frame_depth": depth_refs,
                    "timeline": "timeline.jpg" if len(timeline_inputs) > 1 else None,
                    "view_selection": "view_selection.json",
                    "association_candidates": "association_candidates.json" if finding.stage == "association" else None,
                }
            )
        except Exception as exc:
            warnings.append(f"{finding.finding_uid}:image:{type(exc).__name__}:{exc}")

        if settings.get("save_3d_overlay", True):
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(7, 6))
                axis = fig.add_subplot(111, projection="3d")
                plotted = 0
                for index, obs in enumerate(observations[: int(settings.get("max_3d_observations", 12))]):
                    points = facts.array(obs.get("pcd_ref"))
                    if points is None or not len(points):
                        continue
                    step = max(1, len(points) // 2000)
                    sample = points[::step]
                    axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1, alpha=0.55, label=str(obs.get("obs_uid"))[-12:])
                    plotted += 1
                if plotted:
                    axis.legend(fontsize=6)
                    axis.set_xlabel("x"); axis.set_ylabel("y"); axis.set_zlabel("z")
                    fig.tight_layout(); fig.savefig(case_dir / "pcd_overlay.png", dpi=140)
                plt.close(fig)
            except Exception as exc:
                warnings.append(f"{finding.finding_uid}:pcd:{type(exc).__name__}:{exc}")
        finding.evidence_refs["case_packet"] = str(case_dir.relative_to(output_dir.parent))
        _json_dump(case_dir / "case.json", finding.to_dict())
        built += 1
    return {
        "built": built,
        "warnings": warnings,
        "max_cases": max_cases,
        "strategy": selection_manifest["strategy"],
        "available_candidates": selection_manifest["available_candidates"],
        "selected_cohort_counts": selection_manifest["selected_cohort_counts"],
        "selected_checker_counts": selection_manifest["selected_checker_counts"],
        "selection_manifest": "case_selection.json",
    }


def load_audit_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("version"):
        raise ValueError("audit config must be a mapping with a version")
    return value


def run_layered_audit(
    experiment_dir: str | Path,
    audit_config_path: str | Path,
    *,
    build_cases: Optional[bool] = None,
    output_dir: Optional[str | Path] = None,
) -> dict:
    experiment_dir = Path(experiment_dir).resolve()
    evidence_root = experiment_dir / "evidence"
    audit_config_path = Path(audit_config_path).resolve()
    audit_config = load_audit_config(audit_config_path)
    manifest = json.loads((evidence_root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    mapping_ref = legacy._parse_ref(manifest.get("mapping_config_ref"))
    mapping_config = legacy._load_json(
        legacy._resolve_path(mapping_ref["path"], evidence_root), errors, {}
    )
    manifest_policy = manifest.get("audit_policy")
    policy = manifest_policy or audit_config.get("policy", {})
    policy_source = "manifest" if manifest_policy else "audit_config_fallback"
    if build_cases is not None:
        audit_config.setdefault("case_builder", {})["enabled"] = bool(build_cases)
    output = Path(output_dir).resolve() if output_dir else experiment_dir / str(audit_config.get("output_dir_name", "audit_v1"))
    output.mkdir(parents=True, exist_ok=True)
    context = AuditContext(
        run_id=str(manifest.get("run_id", experiment_dir.name)),
        scene_id=str(manifest.get("scene_id", mapping_config.get("scene_id", "unknown"))),
        manifest=manifest,
        config=mapping_config,
        audit_config=audit_config,
        policy=policy,
        policy_source=policy_source,
        evidence_root=evidence_root,
        experiment_dir=experiment_dir,
        output_dir=output,
        environment_mode=str(policy.get("environment_mode", audit_config.get("environment_mode", "static"))),
    )
    facts = FactStore(context)
    engine = LayeredAudit(context, facts)
    findings, root_causes, summary = engine.run()
    case_result = build_evidence_packets(findings, facts, output, audit_config)
    summary["case_builder"] = case_result
    config_copy = output / "audit_config.yaml"
    shutil.copyfile(audit_config_path, config_copy)
    metrics_cache = output / "metrics_cache"
    metrics_cache.mkdir(parents=True, exist_ok=True)
    with (metrics_cache / "mask_conflict_graph.jsonl").open("w", encoding="utf-8") as handle:
        for item in facts.mask_conflict_edges:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output / "findings.jsonl").open("w", encoding="utf-8") as handle:
        for item in findings:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    with (output / "root_causes.jsonl").open("w", encoding="utf-8") as handle:
        for item in root_causes:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    _json_dump(output / "audit_summary.json", summary)
    validation = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.run_id,
        "gate_status": summary["gate_status"],
        "system_findings": [item.to_dict() for item in findings if item.stage == "system"],
    }
    _json_dump(output / "evidence_validation.json", validation)
    audit_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.run_id,
        "source_evidence_manifest": {
            "path": str((evidence_root / "manifest.json").relative_to(experiment_dir)),
            "sha256": _sha256(evidence_root / "manifest.json"),
        },
        "audit_config": {"path": "audit_config.yaml", "sha256": _sha256(config_copy), "version": audit_config.get("version")},
        "policy": policy,
        "policy_source": policy_source,
        "enabled_checkers": audit_config.get("enabled_checkers", {}),
        "mapping_mutated": False,
        "outputs": {
            name: {"path": name, "sha256": _sha256(output / name)}
            for name in (
                "evidence_validation.json",
                "findings.jsonl",
                "root_causes.jsonl",
                "audit_summary.json",
                "case_selection.json",
            )
            if (output / name).exists()
        },
    }
    _json_dump(output / "audit_manifest.json", audit_manifest)
    return {"summary": summary, "findings": findings, "root_causes": root_causes, "output_dir": output}
