#!/usr/bin/env python3
"""Assemble a frozen incident-level validation root from scene audit outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "2.1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_scene(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("scene must be SCENE_ID=/absolute/experiment/path")
    scene_id, raw_path = value.split("=", 1)
    scene_id = scene_id.strip()
    path = Path(raw_path).expanduser().resolve()
    if not scene_id or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid scene specification: {value}")
    return scene_id, path


def worklist_rows(
    scene_id: str,
    experiment_dir: Path,
    validation_root: Path,
    audit_dir_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_dir = experiment_dir / audit_dir_name
    selection_path = audit_dir / "case_selection.json"
    selection = read_json(selection_path)
    if selection.get("annotation_unit") != "incident":
        raise ValueError(f"{selection_path} is not an incident-level selection manifest")
    if selection.get("strategy") != "incident_deduplicated_dual_cohort_endpoint_review":
        raise ValueError(f"unexpected incident sampling strategy in {selection_path}")
    rows = []
    for item in selection.get("selected") or []:
        incident_uid = str(item.get("incident_uid") or "")
        representative_uid = str(item.get("representative_finding_uid") or item.get("finding_uid") or "")
        source_case = audit_dir / "cases" / representative_uid / "case.json"
        if not incident_uid or not representative_uid or not source_case.is_file():
            raise ValueError(f"invalid selected incident in {selection_path}: {incident_uid}")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "annotation_unit": "incident",
                "scene_id": scene_id,
                "case_uid": incident_uid,
                "incident_uid": incident_uid,
                "representative_finding_uid": representative_uid,
                "finding_uid": representative_uid,
                "checker_id": item.get("checker_id"),
                "stage": item.get("stage"),
                "subtype": item.get("subtype"),
                "checker_ids": item.get("checker_ids") or [],
                "stages": item.get("stages") or [],
                "subtypes": item.get("subtypes") or [],
                "member_finding_uids": item.get("member_finding_uids") or [],
                "blocked_checker_ids": item.get("blocked_checker_ids") or [],
                "trigger_observation_uids": item.get("trigger_observation_uids") or [],
                "representative_trigger_observation_uids": item.get(
                    "representative_trigger_observation_uids"
                ) or item.get("trigger_observation_uids") or [],
                "all_trigger_observation_uids": item.get(
                    "all_trigger_observation_uids"
                ) or item.get("trigger_observation_uids") or [],
                "final_owner_uids": item.get("final_owner_uids") or [],
                "machine_resolution_status": item.get("machine_resolution_status"),
                "identity_kind": item.get("identity_kind"),
                "cohort": item.get("cohort"),
                "case_rank": item.get("case_rank"),
                "review_score": item.get("review_score"),
                "review_priority": item.get("review_priority"),
                "selection_probability": item.get("selection_probability"),
                "sampling_weight": item.get("sampling_weight"),
                "case_dir": str(validation_root / "cases" / scene_id / representative_uid),
                "reviewer_id": None,
                "evidence_sufficient": None,
                "final_state": None,
                "final_error_type": None,
                "review_seconds": None,
                "notes": "",
            }
        )
    reviewable_count = int((selection.get("deduplication") or {}).get("reviewable_incident_count", -1))
    is_endpoint_census = reviewable_count >= 0 and len(rows) == reviewable_count
    return rows, {
        "scene_id": scene_id,
        "experiment_dir": str(experiment_dir),
        "audit_dir": str(audit_dir),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selected_incident_count": len(rows),
        "reviewable_endpoint_count": reviewable_count,
        "selection_mode": "endpoint_census" if is_endpoint_census else "dual_cohort_sample",
        "deduplication": selection.get("deduplication") or {},
        "selected_cohort_counts": selection.get("selected_cohort_counts") or {},
        "weighted_precision_allowed": selection.get("weighted_precision_allowed"),
    }


def assemble(
    validation_root: Path,
    scenes: list[tuple[str, Path]],
    *,
    audit_dir_name: str,
    config_path: Path | None = None,
    review_readme: Path | None = None,
    parity_dir: Path | None = None,
) -> dict[str, Any]:
    validation_root = validation_root.expanduser().resolve()
    if validation_root.exists():
        raise FileExistsError(f"refusing to overwrite existing validation root: {validation_root}")
    if len({scene_id for scene_id, _ in scenes}) != len(scenes):
        raise ValueError("scene IDs must be unique")
    validation_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{validation_root.name}.building-", dir=validation_root.parent)
    ).resolve()
    try:
        all_rows: list[dict[str, Any]] = []
        scene_records = []
        for scene_id, experiment_dir in scenes:
            rows, record = worklist_rows(
                scene_id, experiment_dir, validation_root, audit_dir_name
            )
            all_rows.extend(rows)
            scene_records.append(record)
            run_link = staging_root / "runs" / scene_id / "formal"
            case_link = staging_root / "cases" / scene_id
            run_link.parent.mkdir(parents=True, exist_ok=True)
            case_link.parent.mkdir(parents=True, exist_ok=True)
            run_link.symlink_to(experiment_dir, target_is_directory=True)
            case_link.symlink_to(experiment_dir / audit_dir_name / "cases", target_is_directory=True)

        keys = [(row["scene_id"], row["incident_uid"]) for row in all_rows]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate scene/incident key in combined worklist")
        all_rows.sort(
            key=lambda row: (
                str(row.get("cohort")),
                str(row.get("scene_id")),
                int(row.get("case_rank") or 0),
                str(row.get("incident_uid")),
            )
        )
        worklist_path = staging_root / "labels" / "r1_worklist.jsonl"
        write_jsonl(worklist_path, all_rows)
        (staging_root / "metrics").mkdir(parents=True, exist_ok=True)
        (staging_root / "config").mkdir(parents=True, exist_ok=True)
        if config_path is not None:
            shutil.copyfile(config_path.resolve(), staging_root / "config" / config_path.name)
        if review_readme is not None:
            shutil.copyfile(review_readme.resolve(), staging_root / "labels" / "README.md")
        parity_record = None
        if parity_dir is not None:
            parity_dir = parity_dir.resolve()
            parity_report = parity_dir / "parity_report.json"
            if not parity_report.is_file() or read_json(parity_report).get("status") != "PASS":
                raise ValueError(f"parity directory does not contain a PASS report: {parity_dir}")
            (staging_root / "parity").symlink_to(parity_dir, target_is_directory=True)
            parity_record = {
                "path": str(parity_dir),
                "report_sha256": sha256_file(parity_report),
                "status": "PASS",
            }
        (staging_root / "decision.md").write_text(
            "# Incident-level validation gate\n\n"
            "Status: `PENDING R1 ENDPOINT LABELS`\n\n"
            "This gate evaluates unique unresolved final-map incidents. Root-cause and repair claims "
            "must be generated only after confirmed endpoint errors and verified by replay.\n",
            encoding="utf-8",
        )
        full_endpoint_census = bool(scene_records) and all(
            record.get("selection_mode") == "endpoint_census" for record in scene_records
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "annotation_unit": "incident",
            "strategy": "incident_deduplicated_endpoint_first",
            "audit_dir_name": audit_dir_name,
            "validation_root": str(validation_root),
            "case_count": len(all_rows),
            "scene_count": len(scene_records),
            "selection_mode": "endpoint_census" if full_endpoint_census else "dual_cohort_sample",
            "full_endpoint_census": full_endpoint_census,
            "cohort_counts": dict(
                sorted(
                    {
                        str(cohort): sum(row.get("cohort") == cohort for row in all_rows)
                        for cohort in {row.get("cohort") for row in all_rows}
                    }.items()
                )
            ),
            "worklist_sha256": sha256_file(worklist_path),
            "parity": parity_record,
            "scenes": scene_records,
        }
        write_json(staging_root / "incident_worklist_manifest.json", manifest)
        os.replace(staging_root, validation_root)
        return manifest
    except Exception:
        if staging_root.exists() and staging_root.parent == validation_root.parent:
            shutil.rmtree(staging_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--scene", action="append", required=True, type=parse_scene)
    parser.add_argument("--audit-dir-name", default="audit_validity_gate_endpoint_v2_1")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--review-readme", type=Path)
    parser.add_argument("--parity-dir", type=Path)
    args = parser.parse_args()
    manifest = assemble(
        args.validation_root,
        args.scene,
        audit_dir_name=args.audit_dir_name,
        config_path=args.config,
        review_readme=args.review_readme,
        parity_dir=args.parity_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
