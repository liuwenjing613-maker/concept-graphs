#!/usr/bin/env python3
"""Build blind identity-repair intake directly from machine incident worklists."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from conceptgraph.revision.auto_constraints import forbidden_inference_paths
from conceptgraph.revision.evidence_split import sha256_file
from conceptgraph.revision.identity_evidence import IdentityEvidenceBundleBuilder
from conceptgraph.revision.index import ProvenanceIndex


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


def _blind_uid(scene_id: str, incident_uid: str) -> str:
    digest = hashlib.sha256(f"{scene_id}:{incident_uid}".encode("utf-8")).hexdigest()
    return "identity_machine_" + digest[:20]


def _frame_index(obs_uid: str) -> int:
    match = re.search(r"_f(\d{6})(?:_|$)", str(obs_uid))
    if match is None:
        raise ValueError(f"cannot parse frame from {obs_uid}")
    return int(match.group(1))


def _machine_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    scene_id: str,
    excluded_incidents: set[str],
) -> list[dict[str, Any]]:
    candidates = [
        dict(row)
        for row in rows
        if str(row.get("stage")) == "association"
        and str(row.get("incident_uid")) not in excluded_incidents
        and not row.get("blocked_checker_ids")
    ]
    return sorted(
        candidates,
        key=lambda row: (
            -float(row.get("review_score") or row.get("score") or 0.0),
            int(row.get("case_rank") or 10**9),
            str(row.get("incident_uid") or ""),
        ),
    )


def _has_machine_future_visibility(
    review: Mapping[str, Any],
    *,
    association_event_uid: str,
    minimum_frame_gap: int,
) -> bool:
    matches = [
        row
        for row in review.get("association_decisions") or ()
        if str(row.get("event_uid")) == association_event_uid
    ]
    if len(matches) != 1:
        return False
    association = matches[0]
    anchor_frame = _frame_index(str(association["obs_uid"]))
    relevant = {
        str(association.get("target_object_uid") or ""),
        *(
            str(row.get("object_uid") or "")
            for row in (association.get("candidates") or ())[:1]
        ),
    }
    relevant.discard("")
    resolved = set(relevant)
    for row in review.get("objects") or ():
        if str(row.get("object_uid")) in relevant:
            resolved.update(str(item) for item in row.get("resolved_final_uids") or ())
    return any(
        str(row.get("object_uid")) in resolved
        and int(row.get("last_frame") or -1) >= anchor_frame + minimum_frame_gap
        for row in review.get("final_objects") or ()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--office-selection", required=True, type=Path)
    parser.add_argument("--room-selection", required=True, type=Path)
    parser.add_argument("--office-run", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--private-audit", required=True, type=Path)
    parser.add_argument("--per-scene", type=int, default=1)
    parser.add_argument("--minimum-frame-gap", type=int, default=3)
    parser.add_argument("--exclude-incident", action="append", default=[])
    args = parser.parse_args()

    if args.per_scene < 1:
        raise ValueError("--per-scene must be positive")
    if args.minimum_frame_gap < 1:
        raise ValueError("--minimum-frame-gap must be positive")
    base_runs = {
        "office0": args.office_run.resolve(),
        "room0": args.room_run.resolve(),
    }
    selection_paths = {
        "office0": args.office_selection.resolve(),
        "room0": args.room_selection.resolve(),
    }
    selections = {scene_id: _read(path) for scene_id, path in selection_paths.items()}
    provenance = {
        scene_id: ProvenanceIndex(base_run) for scene_id, base_run in base_runs.items()
    }
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    excluded = set(str(item) for item in args.exclude_incident)

    runtime_cases = []
    audit_cases = []
    rejected = []
    for scene_id in ("office0", "room0"):
        accepted = 0
        for source in _machine_candidates(
            selections[scene_id].get("selected") or (),
            scene_id=scene_id,
            excluded_incidents=excluded,
        ):
            incident_uid = str(source["incident_uid"])
            finding_uid = str(source["representative_finding_uid"])
            packet = (
                base_runs[scene_id]
                / "audit_validity_gate_endpoint_v2_1"
                / "cases"
                / finding_uid
            )
            try:
                case = _read(packet / "case.json")
                review = _read(packet / "review_evidence.json")
                event_uid = str(case["scope"]["event_uid"])
                if str(review.get("case_uid")) != incident_uid:
                    raise ValueError("machine review incident mismatch")
                if not _has_machine_future_visibility(
                    review,
                    association_event_uid=event_uid,
                    minimum_frame_gap=args.minimum_frame_gap,
                ):
                    raise ValueError("no machine-visible future identity evidence")
                blind_case_uid = _blind_uid(scene_id, incident_uid)
                built = IdentityEvidenceBundleBuilder(provenance[scene_id]).build(
                    case_uid=blind_case_uid,
                    association_event_uid=event_uid,
                    machine_review=review,
                )
                candidate = built.binding.aliases.get("CANDIDATE_1_CONTEXT")
                if candidate is None or not candidate.complete:
                    raise ValueError("candidate-1 identity binding is incomplete")
            except (FileNotFoundError, KeyError, ValueError) as exc:
                rejected.append(
                    {
                        "scene_id": scene_id,
                        "source_incident_uid": incident_uid,
                        "source_finding_uid": finding_uid,
                        "reason": str(exc),
                    }
                )
                continue

            case_dir = output_root / blind_case_uid
            bundle_path = case_dir / "bundle.machine_only.json"
            binding_path = case_dir / "binding.private.json"
            _write(bundle_path, built.inference_bundle)
            _write(binding_path, built.binding.as_dict())
            runtime_cases.append(
                {
                    "case_uid": blind_case_uid,
                    "scene_id": scene_id,
                    "bundle_path": str(bundle_path.resolve()),
                    "bundle_sha256": sha256_file(bundle_path),
                    "binding_path": str(binding_path.resolve()),
                    "binding_sha256": sha256_file(binding_path),
                }
            )
            audit_cases.append(
                {
                    "blind_case_uid": blind_case_uid,
                    "scene_id": scene_id,
                    "source_incident_uid": incident_uid,
                    "source_finding_uid": finding_uid,
                    "source_machine_review_score": float(
                        source.get("review_score") or source.get("score") or 0.0
                    ),
                    "source_selection_manifest": str(selection_paths[scene_id]),
                    "source_selection_manifest_sha256": sha256_file(
                        selection_paths[scene_id]
                    ),
                    "human_labels_loaded": False,
                }
            )
            accepted += 1
            if accepted >= args.per_scene:
                break
        if accepted < args.per_scene:
            raise ValueError(
                f"{scene_id}: only {accepted} machine incidents passed intake; "
                f"need {args.per_scene}"
            )

    runtime_manifest = {
        "schema_version": "1.0.0",
        "role": "AUTONOMOUS_MACHINE_INCIDENT_RUNTIME_INPUT",
        "candidate_source": "MACHINE_INCIDENT_WORKLIST_ONLY",
        "selection_policy": (
            "highest machine review score with association anchor, complete "
            "candidate binding, and future endpoint visibility"
        ),
        "case_count": len(runtime_cases),
        "cases": runtime_cases,
        "runtime_human_or_gold_loaded": False,
    }
    forbidden = forbidden_inference_paths(runtime_manifest)
    if forbidden:
        raise ValueError(
            "machine intake leaked oracle-like runtime fields: " + ", ".join(forbidden)
        )
    runtime_path = args.runtime_manifest.resolve()
    _write(runtime_path, runtime_manifest)
    private_audit = {
        "schema_version": "1.0.0",
        "role": "PRIVATE_MACHINE_INTAKE_TRACE_NOT_INFERENCE_INPUT",
        "runtime_manifest_path": str(runtime_path),
        "runtime_manifest_sha256": sha256_file(runtime_path),
        "human_labels_loaded": False,
        "accepted_cases": audit_cases,
        "rejected_machine_cases": rejected,
    }
    _write(args.private_audit.resolve(), private_audit)
    print(
        json.dumps(
            {
                "status": "PASS",
                "case_count": len(runtime_cases),
                "runtime_manifest": str(runtime_path),
                "blind_case_uids": [row["case_uid"] for row in runtime_cases],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
