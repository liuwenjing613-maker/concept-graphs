#!/usr/bin/env python3
"""Freeze rich, oracle-free identity evidence bundles before model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from conceptgraph.revision.auto_constraints import IncidentBinding
from conceptgraph.revision.identity_evidence import IdentityEvidenceBundleBuilder
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.vlm import VLMIncidentBuilder


_FIXED_MACHINE_PANELS = (
    ("review_observation_Q1.png", "MACHINE_TRIGGER_PANEL_Q1"),
    ("review_observation_Q2.png", "MACHINE_TRIGGER_PANEL_Q2"),
    ("review_final_objects_relative.png", "CURRENT_MAP_RELATIVE_GEOMETRY"),
    ("review_final_objects_detail.png", "CURRENT_MAP_OBJECT_DETAIL"),
    ("timeline.jpg", "MACHINE_INCIDENT_TIMELINE"),
)


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _review_packet(base_run: Path, finding_uid: str) -> tuple[Path, dict[str, Any]]:
    packet = base_run / "audit_validity_gate_endpoint_v2_1" / "cases" / finding_uid
    review_path = packet / "review_evidence.json"
    if not review_path.is_file():
        raise FileNotFoundError(review_path)
    return packet, _read(review_path)


def _image_rows(
    *,
    base_evidence,
    packet: Path,
    review: Mapping[str, Any],
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for source_row, source_path in zip(
        base_evidence.image_manifest, base_evidence.image_paths
    ):
        path = Path(source_path).resolve()
        rows.append(
            {
                **source_row,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "source_role": "ASSOCIATION_CONTEXT_CROP",
                "path": str(path),
            }
        )
        paths.append(path)

    displayed_hashes = review.get("displayed_asset_sha256") or {}
    for filename, role in _FIXED_MACHINE_PANELS:
        path = (packet / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        frozen = displayed_hashes.get(filename)
        if frozen is not None and str(frozen) != actual:
            raise ValueError(f"machine panel hash drift: {path}")
        rows.append(
            {
                "image_id": f"I{len(rows) + 1:02d}",
                "context_alias": role,
                "logical_name": filename,
                "sha256": actual,
                "bytes": path.stat().st_size,
                "source_role": "MACHINE_AUDIT_PANEL",
                "path": str(path),
            }
        )
        paths.append(path)
    return paths, rows


def _prompt(bundle: Mapping[str, Any], images: list[Mapping[str, Any]]) -> str:
    prompt_payload = {
        "identity_evidence_bundle": bundle,
        "images": [
            {
                key: row.get(key)
                for key in (
                    "image_id",
                    "context_alias",
                    "obs_key",
                    "class_name",
                    "logical_name",
                    "source_role",
                )
                if row.get(key) is not None
            }
            for row in images
        ],
    }
    return (
        "Decide one physical-identity repair using only the frozen machine evidence "
        "below. The current map may be wrong; its detector signals, geometry panels, "
        "and timeline are observations, not a human answer. No human verdict, desired "
        "ownership, repaired map, or expected action is supplied.\n\n"
        "All target identifiers are finite aliases. Never invent an alias or UID. "
        "For an observed CREATE decision: choose SAME_INSTANCE only when ANCHOR should "
        "have joined exactly one candidate physical instance; choose "
        "SEPARATE_MEMBER_GROUPS only when the creation was appropriate but ANCHOR and "
        "exactly one candidate must remain different identities through later merges. "
        "Use DEFER when neither conclusion is supported. Semantic class equality alone "
        "is insufficient. Use registered appearance, 3D separation/overlap, temporal "
        "history, machine endpoint geometry, and merge timeline together.\n\n"
        "Return exactly one JSON object. For SAME_INSTANCE return action, confidence, "
        "entities=['ANCHOR','CANDIDATE_N_CONTEXT'], evidence_image_ids, and reason. "
        "For SEPARATE_MEMBER_GROUPS return action, confidence, "
        "groups=[['ANCHOR'],['CANDIDATE_N_CONTEXT']], evidence_image_ids, and reason. "
        "For DEFER return action, confidence, empty entities/groups, "
        "evidence_image_ids, and reason. Cite only listed image IDs.\n\n"
        "FROZEN EVIDENCE:\n"
        + json.dumps(prompt_payload, indent=2, sort_keys=True, ensure_ascii=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--office-run", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()

    source = _read(args.identity_manifest)
    cases = sorted(
        (
            row
            for row in source.get("cases") or ()
            if row.get("causal_disposition") == "REPLAYABLE_ASSOCIATION_CAUSE"
        ),
        key=lambda row: str(row["anchor_association_event_uid"]),
    )
    if len(cases) != 3:
        raise ValueError("expected exactly three replayable identity development cases")

    base_runs = {
        "office0": args.office_run.resolve(),
        "room0": args.room_run.resolve(),
    }
    provenance = {scene: ProvenanceIndex(path) for scene, path in base_runs.items()}
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_cases = []

    for index, case in enumerate(cases, 1):
        scene = str(case["scene_id"])
        blind_uid = f"identity_dev_{index:03d}"
        packet, review = _review_packet(
            base_runs[scene], str(case["representative_finding_uid"])
        )
        if str(review.get("case_uid")) != str(case["incident_uid"]):
            raise ValueError(f"review incident mismatch for {blind_uid}")
        built = IdentityEvidenceBundleBuilder(provenance[scene]).build(
            case_uid=blind_uid,
            association_event_uid=str(case["anchor_association_event_uid"]),
            machine_review=review,
        )
        case_dir = output_root / blind_uid
        base_evidence = VLMIncidentBuilder(provenance[scene]).build(
            {
                "anchor_association_event_uid": str(
                    case["anchor_association_event_uid"]
                ),
                "observed_current_decision": built.binding.observed_current_decision,
            },
            case_dir / "association_context_images",
        )
        image_paths, image_manifest = _image_rows(
            base_evidence=base_evidence,
            packet=packet,
            review=review,
        )
        prompt = _prompt(built.inference_bundle, image_manifest)
        binding_value = built.binding.as_dict()
        binding_value["evidence_refs"] = sorted(
            set(
                binding_value["evidence_refs"]
                + [f"{row['image_id']}:{row['sha256']}" for row in image_manifest]
            )
        )
        binding = IncidentBinding.from_mapping(binding_value)
        request = {
            "schema_version": "1.0.0",
            "blind_case_uid": blind_uid,
            "input_family": "IDENTITY_ASSOCIATION_DEVELOPMENT",
            "bundle_path": str((case_dir / "bundle.json").resolve()),
            "bundle_uid": built.inference_bundle["bundle_uid"],
            "prompt": prompt,
            "prompt_sha256": _text_sha256(prompt),
            "images": image_manifest,
            "allowed_evidence_image_ids": [row["image_id"] for row in image_manifest],
            "binding_private_path": str((case_dir / "binding.private.json").resolve()),
            "source_case_uid_not_in_prompt": True,
            "human_verdict_not_in_prompt": True,
            "expected_action_not_in_prompt": True,
            "repaired_ownership_not_in_prompt": True,
        }
        _write(case_dir / "bundle.json", built.inference_bundle)
        _write(case_dir / "binding.private.json", binding.as_dict())
        _write(case_dir / "request.frozen.json", request)
        manifest_cases.append(
            {
                "blind_case_uid": blind_uid,
                "scene_id": scene,
                "anchor_event_alias": built.inference_bundle["incident_alias"],
                "bundle_uid": built.inference_bundle["bundle_uid"],
                "bundle_path": request["bundle_path"],
                "bundle_sha256": _sha256(case_dir / "bundle.json"),
                "request_path": str((case_dir / "request.frozen.json").resolve()),
                "request_sha256": _sha256(case_dir / "request.frozen.json"),
                "binding_private_path": request["binding_private_path"],
                "binding_private_sha256": _sha256(case_dir / "binding.private.json"),
                "image_count": len(image_paths),
                "prompt_sha256": request["prompt_sha256"],
                "development_source_case_uid": str(case["case_uid"]),
                "development_gold_excluded_from_request": True,
            }
        )

    result = {
        "schema_version": "1.0.0",
        "manifest_uid": "identity_evidence_bundle_v1_20260824",
        "role": "DEVELOPMENT_NOT_HOLDOUT",
        "frozen_before_model_responses": True,
        "candidate_aliases_deterministic": True,
        "current_corrupted_map_evidence_allowed": True,
        "human_answers_excluded": True,
        "identity_manifest_sha256": _sha256(args.identity_manifest.resolve()),
        "case_count": len(manifest_cases),
        "cases": manifest_cases,
    }
    _write(args.output_manifest, result)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_manifest": str(args.output_manifest.resolve()),
                "case_count": len(manifest_cases),
                "bundle_uids": [row["bundle_uid"] for row in manifest_cases],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
