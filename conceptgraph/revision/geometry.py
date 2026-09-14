from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_SOURCE_ROLES = {
    "raw_mask",
    "processed_mask",
    "depth",
    "rgb",
    "original_observation_pcd",
}


class GeometryContractError(ValueError):
    """Raised when a geometry overlay is incomplete, drifted, or ambiguous."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeometryContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name).lower()
    if not _SHA256.fullmatch(text):
        raise GeometryContractError(f"{field_name} must be a lowercase SHA-256")
    return text


def _artifact_ref(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GeometryContractError(f"{field_name} must be an object")
    result = {
        "path": _required_text(value.get("path"), f"{field_name}.path"),
        "sha256": _sha(value.get("sha256"), f"{field_name}.sha256"),
        "format": _required_text(value.get("format", "npz"), f"{field_name}.format"),
    }
    for name in ("key", "index", "shape", "dtype", "role"):
        if value.get(name) is not None:
            result[name] = value[name]
    return result


def resolve_artifact_ref(
    ref: Mapping[str, Any], *, base_root: str | Path | None = None
) -> Path:
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        if base_root is None:
            raise GeometryContractError(
                f"relative artifact path has no base root: {path}"
            )
        path = Path(base_root) / path
    path = path.resolve()
    if not path.is_file():
        raise GeometryContractError(f"geometry artifact does not exist: {path}")
    actual = file_sha256(path)
    if actual != str(ref["sha256"]):
        raise GeometryContractError(
            f"geometry artifact drift: {path}; expected {ref['sha256']}, got {actual}"
        )
    return path


@dataclass(frozen=True)
class ObservationGeometryContract:
    obs_uid: str
    payload_uid: str
    replacement_pcd_ref: dict[str, Any]
    replacement_mask_ref: dict[str, Any]
    source_observation_sha256: str
    source_artifacts: tuple[dict[str, Any], ...]
    derivation: dict[str, Any]
    schema_version: str = "1.0.0"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationGeometryContract":
        if not isinstance(value, Mapping):
            raise GeometryContractError("geometry_contract must be an object")
        schema = _required_text(
            value.get("schema_version", "1.0.0"), "geometry_contract.schema_version"
        )
        if schema != "1.0.0":
            raise GeometryContractError(
                f"unsupported geometry contract schema: {schema}"
            )
        source_values = value.get("source_artifacts")
        if not isinstance(source_values, Sequence) or isinstance(
            source_values, (str, bytes)
        ):
            raise GeometryContractError(
                "geometry_contract.source_artifacts must be a list"
            )
        sources = tuple(
            _artifact_ref(item, f"source_artifacts[{index}]")
            for index, item in enumerate(source_values)
        )
        roles = [str(item.get("role") or "") for item in sources]
        if len(roles) != len(set(roles)):
            raise GeometryContractError("geometry source roles must be unique")
        missing = sorted(_REQUIRED_SOURCE_ROLES - set(roles))
        if missing:
            raise GeometryContractError(
                "geometry contract lacks source roles: " + ", ".join(missing)
            )
        derivation = value.get("derivation")
        if not isinstance(derivation, Mapping):
            raise GeometryContractError(
                "geometry_contract.derivation must be an object"
            )
        if derivation.get("algorithm") not in {
            "RAW_MASK_DEPTH_WORLD_PCD_V1",
            "EXACT_EXISTING_PAYLOAD_NOOP_V1",
        }:
            raise GeometryContractError("unsupported geometry derivation algorithm")
        if derivation.get("random_perturbation") is not False:
            raise GeometryContractError(
                "geometry restoration must disable random perturbation"
            )
        contract = cls(
            schema_version=schema,
            obs_uid=_required_text(value.get("obs_uid"), "geometry_contract.obs_uid"),
            payload_uid=_required_text(
                value.get("payload_uid"), "geometry_contract.payload_uid"
            ),
            replacement_pcd_ref=_artifact_ref(
                value.get("replacement_pcd_ref"), "replacement_pcd_ref"
            ),
            replacement_mask_ref=_artifact_ref(
                value.get("replacement_mask_ref"), "replacement_mask_ref"
            ),
            source_observation_sha256=_sha(
                value.get("source_observation_sha256"),
                "source_observation_sha256",
            ),
            source_artifacts=sources,
            derivation=dict(derivation),
        )
        expected_uid = (
            "geometry_payload_"
            + canonical_json_sha256(contract.as_dict(include_payload_uid=False))[:20]
        )
        if contract.payload_uid != expected_uid:
            raise GeometryContractError(
                f"geometry payload UID mismatch: {contract.payload_uid} != {expected_uid}"
            )
        return contract

    def as_dict(self, *, include_payload_uid: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "obs_uid": self.obs_uid,
            "replacement_pcd_ref": dict(self.replacement_pcd_ref),
            "replacement_mask_ref": dict(self.replacement_mask_ref),
            "source_observation_sha256": self.source_observation_sha256,
            "source_artifacts": [dict(item) for item in self.source_artifacts],
            "derivation": dict(self.derivation),
        }
        if include_payload_uid:
            result["payload_uid"] = self.payload_uid
        return result

    @classmethod
    def build(
        cls,
        *,
        obs_uid: str,
        replacement_pcd_ref: Mapping[str, Any],
        replacement_mask_ref: Mapping[str, Any],
        source_observation_sha256: str,
        source_artifacts: Sequence[Mapping[str, Any]],
        derivation: Mapping[str, Any],
    ) -> "ObservationGeometryContract":
        without_uid = {
            "schema_version": "1.0.0",
            "obs_uid": obs_uid,
            "replacement_pcd_ref": dict(replacement_pcd_ref),
            "replacement_mask_ref": dict(replacement_mask_ref),
            "source_observation_sha256": source_observation_sha256,
            "source_artifacts": [dict(item) for item in source_artifacts],
            "derivation": dict(derivation),
        }
        without_uid["payload_uid"] = (
            "geometry_payload_" + canonical_json_sha256(without_uid)[:20]
        )
        return cls.from_mapping(without_uid)

    def verify_source_bindings(
        self,
        observation: Mapping[str, Any],
        *,
        base_root: str | Path | None = None,
    ) -> dict[str, Any]:
        actual_observation_sha256 = canonical_json_sha256(observation)
        source_checks = []
        for ref in self.source_artifacts:
            path = resolve_artifact_ref(ref, base_root=base_root)
            source_checks.append(
                {
                    "role": ref["role"],
                    "path": str(path),
                    "sha256": ref["sha256"],
                    "pass": True,
                }
            )
        observation_pass = actual_observation_sha256 == self.source_observation_sha256
        if not observation_pass:
            raise GeometryContractError(
                f"source observation drift for {self.obs_uid}: "
                f"{actual_observation_sha256} != {self.source_observation_sha256}"
            )
        return {
            "pass": observation_pass and all(row["pass"] for row in source_checks),
            "source_observation_sha256": actual_observation_sha256,
            "source_artifact_checks": source_checks,
        }

    def load_payload(
        self,
        *,
        base_root: str | Path | None = None,
    ) -> dict[str, Any]:
        pcd_path = resolve_artifact_ref(self.replacement_pcd_ref, base_root=base_root)
        mask_path = resolve_artifact_ref(self.replacement_mask_ref, base_root=base_root)
        with np.load(pcd_path, allow_pickle=False) as archive:
            if "points" not in archive.files or "colors" not in archive.files:
                raise GeometryContractError(
                    f"replacement PCD lacks points/colors: {pcd_path}"
                )
            points = np.asarray(archive["points"], dtype=np.float64)
            colors = np.asarray(archive["colors"], dtype=np.float64)
        with np.load(mask_path, allow_pickle=False) as archive:
            key = str(self.replacement_mask_ref.get("key") or "mask")
            if key not in archive.files:
                raise GeometryContractError(
                    f"replacement mask lacks key {key}: {mask_path}"
                )
            mask = np.asarray(archive[key], dtype=bool)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or colors.shape != points.shape
            or not len(points)
            or not np.isfinite(points).all()
            or not np.isfinite(colors).all()
        ):
            raise GeometryContractError("replacement point payload is invalid")
        if mask.ndim != 2 or not mask.any():
            raise GeometryContractError("replacement mask payload is invalid")
        expected_points = self.derivation.get("replacement_points_sha256")
        expected_colors = self.derivation.get("replacement_colors_sha256")
        expected_mask = self.derivation.get("replacement_mask_array_sha256")
        actual = {
            "replacement_points_sha256": array_sha256(points),
            "replacement_colors_sha256": array_sha256(colors),
            "replacement_mask_array_sha256": array_sha256(mask),
        }
        for field, expected in (
            ("replacement_points_sha256", expected_points),
            ("replacement_colors_sha256", expected_colors),
            ("replacement_mask_array_sha256", expected_mask),
        ):
            if actual[field] != _sha(expected, f"derivation.{field}"):
                raise GeometryContractError(
                    f"replacement array drift for {field}: "
                    f"{actual[field]} != {expected}"
                )
        return {
            "points": points,
            "colors": colors,
            "mask": mask,
            "pcd_path": str(pcd_path),
            "mask_path": str(mask_path),
            **actual,
        }
