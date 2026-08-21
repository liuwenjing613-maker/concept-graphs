from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_HUMAN_KEYS = {
    "reviewer_id",
    "evidence_sufficient",
    "final_state",
    "final_error_type",
    "review_seconds",
    "human_label",
    "repair_verified",
}


class EvidenceError(ValueError):
    """Raised when an endpoint packet is unsafe or not exactly traceable."""


@dataclass(frozen=True)
class ImageEvidence:
    path: Path
    role: str
    caption: str
    detail: str = "high"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _histogram_text(histogram: dict[str, Any] | None, limit: int = 6) -> str:
    if not histogram:
        return "none"
    ranked = sorted(histogram.items(), key=lambda item: (-int(item[1]), item[0]))
    return ", ".join(f"{name}:{count}" for name, count in ranked[:limit])


def _float_vector(value: Any, size: int = 3) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != size:
        return None
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    return vector


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_float = float(numerator)
        denominator_float = float(denominator)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numerator_float) or not math.isfinite(denominator_float):
        return None
    if denominator_float == 0:
        return None
    return numerator_float / denominator_float


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


class EndpointEvidence:
    """A label-blind, hash-verified view of one final endpoint packet."""

    def __init__(self, case_dir: Path, payload: dict[str, Any]) -> None:
        self.case_dir = case_dir.resolve()
        self.payload = payload
        self.scene_id = str(payload.get("scene_id"))
        self.case_uid = str(payload.get("case_uid"))
        incident = payload.get("incident") or {}
        self.target_uids = tuple(incident.get("final_owner_uids") or ())
        if len(self.target_uids) != 1:
            raise EvidenceError(
                f"v1 requires exactly one final endpoint owner, got {len(self.target_uids)}"
            )
        self.target_uid = self.target_uids[0]
        matches = [
            obj
            for obj in payload.get("final_objects") or []
            if obj.get("object_uid") == self.target_uid
        ]
        if len(matches) != 1:
            raise EvidenceError("target final object is not uniquely represented")
        self.target_object = matches[0]
        self.target_alias = str(matches[0].get("object_alias") or "").upper()
        if not self.target_alias:
            raise EvidenceError("target alias is missing")
        self.alias_to_uid = {
            str(obj.get("object_alias") or "").upper(): str(obj.get("object_uid"))
            for obj in payload.get("final_objects") or []
            if obj.get("object_alias") and obj.get("object_uid")
        }

    @classmethod
    def load(cls, case_dir: str | Path) -> "EndpointEvidence":
        case_dir = Path(case_dir).expanduser().resolve()
        path = case_dir / "review_evidence.json"
        if not path.is_file():
            raise EvidenceError(f"missing review_evidence.json: {case_dir}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        forbidden = sorted(FORBIDDEN_HUMAN_KEYS.intersection(_walk_keys(payload)))
        if forbidden:
            raise EvidenceError(f"human-label fields found in inference input: {forbidden}")
        contract = payload.get("evidence_contract") or {}
        if contract.get("fidelity_status") != "TRACEABLE":
            raise EvidenceError("packet is not TRACEABLE")
        if contract.get("artifact_hashes_match") is not True:
            raise EvidenceError("packet artifact hashes do not match")
        if contract.get("exact_final_map_linkage") is not True:
            raise EvidenceError("packet lacks exact final-map linkage")
        if contract.get("critical_gaps"):
            raise EvidenceError("packet has a critical evidence gap")
        return cls(case_dir, payload)

    def _asset(self, name: str, role: str, caption: str, detail: str = "high") -> ImageEvidence:
        path = (self.case_dir / name).resolve()
        try:
            path.relative_to(self.case_dir)
        except ValueError as exc:
            raise EvidenceError(f"asset escapes case directory: {name}") from exc
        if not path.is_file():
            raise EvidenceError(f"missing displayed asset: {name}")
        expected = (self.payload.get("displayed_asset_sha256") or {}).get(name)
        if not expected:
            raise EvidenceError(f"asset is not hash-bound by review packet: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise EvidenceError(f"displayed asset hash mismatch: {name}")
        return ImageEvidence(path=path, role=role, caption=caption, detail=detail)

    def select_images(self, max_images: int = 8) -> list[ImageEvidence]:
        if max_images < 3:
            raise EvidenceError("max_images must be at least 3")
        selected: list[ImageEvidence] = []
        geometry = (self.payload.get("assets") or {}).get("final_object_geometry") or []
        for index, name in enumerate(geometry[:2], start=1):
            selected.append(
                self._asset(
                    name,
                    role=f"final_geometry_{index}",
                    caption=(
                        "Exact final-map geometry. Only the object marked [ENDPOINT] is judged; "
                        "other aliases are context."
                    ),
                )
            )

        # The raw/processed-mask/depth panel is often the only place where severe
        # subtraction, surface leakage, or a detached observation fragment is visible.
        # Reserve one slot before representative RGB views can consume the budget.
        used_names = {image.path.name for image in selected}
        packet_assets = self.payload.get("assets") or {}
        panels = packet_assets.get("trigger_observation_panels") or []
        for index, name in enumerate(panels[:2], start=1):
            if len(selected) >= max_images or name in used_names:
                break
            selected.append(
                self._asset(
                    name,
                    role=f"trigger_panel_Q{index}",
                    caption=(
                        "Exact trigger observation panel (RGB, raw mask, processed mask, depth "
                        "and stored observation PCD). Compare raw versus processed support; it "
                        "is supporting evidence, not the final object."
                    ),
                )
            )
            used_names.add(name)

        timeline_name = packet_assets.get("timeline")
        if (
            isinstance(timeline_name, str)
            and timeline_name
            and len(selected) < max_images
            and timeline_name not in used_names
        ):
            selected.append(
                self._asset(
                    timeline_name,
                    role="representative_timeline",
                    caption=(
                        "Frozen multi-view timeline of representative observation boxes. Use it "
                        "to assess physical coverage and repeated section-level fragmentation; "
                        "box colors are observations, not human labels or final aliases."
                    ),
                )
            )
            used_names.add(timeline_name)

        target_views = []
        for view in self.payload.get("representative_views") or []:
            if self.target_uid not in (view.get("object_uids") or []):
                continue
            reasons = view.get("selection_reasons") or []
            priority = min(
                [
                    {
                        "earliest_creation_view": 0,
                        "highest_detector_confidence": 1,
                        "largest_semantic_conflict": 2,
                        "largest_camera_viewpoint_difference": 3,
                        "highest_point_contribution": 4,
                        "anomaly_trigger_view": 5,
                    }.get(reason, 9)
                    for reason in reasons
                ]
                or [9]
            )
            target_views.append((priority, str(view.get("obs_uid")), view))
        target_views.sort(key=lambda item: (item[0], item[1]))

        for view_index, (_, _, view) in enumerate(target_views, start=1):
            if len(selected) >= max_images:
                break
            assets = view.get("assets") or {}
            context_name = assets.get("context_crop")
            if context_name and context_name not in used_names:
                selected.append(
                    self._asset(
                        context_name,
                        role=f"target_view_V{view_index}",
                        caption=(
                            f"V{view_index}: RGB context crop for {self.target_alias}; "
                            f"saved observation label={view.get('class_name')!r}."
                        ),
                    )
                )
                used_names.add(context_name)
            if len(selected) >= max_images:
                break
            mask_name = assets.get("masked_crop")
            if view_index <= 2 and mask_name and mask_name not in used_names:
                selected.append(
                    self._asset(
                        mask_name,
                        role=f"target_mask_V{view_index}",
                        caption=f"V{view_index}: exact processed-mask crop used by mapping.",
                    )
                )
                used_names.add(mask_name)

        return selected

    def _object_support_text(self, obj: dict[str, Any]) -> str:
        histogram = obj.get("observed_class_histogram") or {}
        saved_label = str(obj.get("class_name") or "")
        saved_count = sum(
            int(count)
            for label, count in histogram.items()
            if str(label).casefold() == saved_label.casefold()
        )
        return _fmt(_safe_ratio(saved_count, obj.get("member_count")))

    def _pairwise_spatial_text(self, other: dict[str, Any]) -> str:
        target_center = _float_vector(self.target_object.get("bbox_center"))
        target_extent = _float_vector(self.target_object.get("bbox_extent"))
        other_center = _float_vector(other.get("bbox_center"))
        other_extent = _float_vector(other.get("bbox_extent"))
        if not all((target_center, target_extent, other_center, other_extent)):
            return "spatial metrics unavailable"
        assert target_center and target_extent and other_center and other_extent
        center_delta = tuple(
            abs(target_center[index] - other_center[index]) for index in range(3)
        )
        center_distance = math.sqrt(sum(value * value for value in center_delta))
        half_sums = tuple(
            (target_extent[index] + other_extent[index]) / 2.0 for index in range(3)
        )
        gaps = tuple(
            max(center_delta[index] - half_sums[index], 0.0) for index in range(3)
        )
        bbox_surface_gap = math.sqrt(sum(value * value for value in gaps))
        intersections = tuple(
            min(
                target_extent[index],
                other_extent[index],
                max(half_sums[index] - center_delta[index], 0.0),
            )
            for index in range(3)
        )
        intersection_volume = math.prod(intersections)
        target_volume = math.prod(max(value, 0.0) for value in target_extent)
        other_volume = math.prod(max(value, 0.0) for value in other_extent)
        union_volume = target_volume + other_volume - intersection_volume
        bbox_iou = intersection_volume / union_volume if union_volume > 0 else None
        normalized_center_distance = math.sqrt(
            sum(
                (center_delta[index] / max(half_sums[index], 1e-9)) ** 2
                for index in range(3)
            )
        )
        return (
            f"center_distance={center_distance:.4f}; bbox_surface_gap={bbox_surface_gap:.4f}; "
            f"bbox_iou={_fmt(bbox_iou)}; normalized_center_distance="
            f"{normalized_center_distance:.4f}; target/other_member_ratio="
            f"{_fmt(_safe_ratio(self.target_object.get('member_count'), other.get('member_count')))}; "
            f"target/other_point_ratio="
            f"{_fmt(_safe_ratio(self.target_object.get('n_points'), other.get('n_points')))}"
        )

    def summary_text(self, images: list[ImageEvidence]) -> str:
        lines = [
            f"scene_id: {self.scene_id}",
            f"case_uid: {self.case_uid}",
            f"target: {self.target_alias} [ENDPOINT]",
            "",
            "Exact final objects:",
        ]
        for obj in self.payload.get("final_objects") or []:
            alias = str(obj.get("object_alias") or "?").upper()
            role = "ENDPOINT" if obj.get("object_uid") == self.target_uid else "context"
            lines.append(
                "- "
                f"{alias} [{role}]: saved_label={obj.get('class_name')!r}; "
                f"members={obj.get('member_count')}; unique_frames={obj.get('unique_frame_count')}; "
                f"points={obj.get('n_points')}; bbox_extent={obj.get('bbox_extent')}; "
                f"exact_saved_label_support={self._object_support_text(obj)}; "
                f"observed_labels={_histogram_text(obj.get('observed_class_histogram'))}"
            )
        context_objects = [
            obj
            for obj in self.payload.get("final_objects") or []
            if obj.get("object_uid") != self.target_uid
        ]
        if context_objects:
            lines.extend(["", "Deterministic endpoint-to-context geometry (descriptive only):"])
            for obj in context_objects:
                alias = str(obj.get("object_alias") or "?").upper()
                same_label = (
                    str(obj.get("class_name") or "").casefold()
                    == str(self.target_object.get("class_name") or "").casefold()
                )
                lines.append(
                    f"- {self.target_alias} vs {alias}: same_saved_label={str(same_label).lower()}; "
                    + self._pairwise_spatial_text(obj)
                )
        lines.extend(["", "Images in exact order:"])
        for index, image in enumerate(images, start=1):
            lines.append(f"- image {index} ({image.role}): {image.caption}")
        lines.extend(
            [
                "",
                "Judge only the exact final state of the endpoint. A suspicious upstream event "
                "does not make the final object wrong if it resolved correctly.",
            ]
        )
        return "\n".join(lines)

    def fingerprint(self, prompt_text: str, images: list[ImageEvidence]) -> str:
        digest = hashlib.sha256(prompt_text.encode("utf-8"))
        for image in images:
            digest.update(image.role.encode("utf-8"))
            digest.update(sha256_file(image.path).encode("ascii"))
        return digest.hexdigest()
