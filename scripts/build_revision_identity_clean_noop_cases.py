#!/usr/bin/env python3
"""Freeze human-curated clean identity cases behind a blind runtime manifest.

This is a curation utility, not an inference component. It may read development
labels to select endpoints that humans judged correct, but it writes the label
mapping only to a separate post-hoc key. The runtime manifest and evidence
bundles contain machine/provenance evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _eligible_clean_rows(
    rows: Iterable[Mapping[str, Any]], scene_id: str
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
        if str(row.get("scene_id")) == scene_id
        and str(row.get("annotation_unit")) == "incident"
        and str(row.get("stage")) == "association"
        and str(row.get("evidence_sufficient")).upper() == "YES"
        and str(row.get("final_state")).upper() == "CORRECT"
    ]
    return sorted(
        selected,
        key=lambda row: (
            int(row.get("case_rank") or 10**9),
            str(row.get("case_uid") or ""),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--office-run", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--posthoc-key", required=True, type=Path)
    parser.add_argument("--per-scene", type=int, default=1)
    parser.add_argument(
        "--exclude-source-case",
        action="append",
        default=[],
        help="Development source case UID to exclude after a protocol failure",
    )
    args = parser.parse_args()

    if args.per_scene < 1:
        raise ValueError("--per-scene must be positive")
    excluded_source_cases = set(str(item) for item in args.exclude_source_case)
    labels_path = args.labels.resolve()
    labels = _read_jsonl(labels_path)
    base_runs = {
        "office0": args.office_run.resolve(),
        "room0": args.room_run.resolve(),
    }
    provenance = {
        scene_id: ProvenanceIndex(base_run) for scene_id, base_run in base_runs.items()
    }
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_cases = []
    posthoc_cases = []
    for scene_id in ("office0", "room0"):
        accepted = 0
        rejection_rows = []
        for row in _eligible_clean_rows(labels, scene_id):
            if str(row.get("case_uid")) in excluded_source_cases:
                continue
            source_case_dir = Path(str(row["case_dir"])).resolve()
            case = _read(source_case_dir / "case.json")
            review = _read(source_case_dir / "review_evidence.json")
            try:
                blind_case_uid = f"identity_clean_noop_{scene_id}_{accepted + 1:03d}"
                built = IdentityEvidenceBundleBuilder(provenance[scene_id]).build(
                    case_uid=blind_case_uid,
                    association_event_uid=str(case["scope"]["event_uid"]),
                    machine_review=review,
                )
                candidate = built.binding.aliases.get("CANDIDATE_1_CONTEXT")
                if built.binding.observed_current_decision != "CREATE":
                    raise ValueError("clean control is not an observed CREATE")
                if candidate is None or not candidate.complete:
                    raise ValueError("candidate-1 identity binding is incomplete")
            except (KeyError, ValueError) as exc:
                rejection_rows.append(
                    {
                        "source_case_uid": str(row.get("case_uid")),
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
            posthoc_cases.append(
                {
                    "blind_case_uid": blind_case_uid,
                    "scene_id": scene_id,
                    "source_case_uid": str(row["case_uid"]),
                    "source_finding_uid": str(row["finding_uid"]),
                    "source_label_record_sha256": _sha256_json(row),
                    "human_endpoint_state": "CORRECT",
                    "human_evidence_sufficient": "YES",
                    "expected_runtime_action": "NO_OP",
                }
            )
            accepted += 1
            if accepted >= args.per_scene:
                break
        if accepted < args.per_scene:
            raise ValueError(
                f"{scene_id}: found {accepted} eligible clean CREATE cases; "
                f"need {args.per_scene}; rejections={rejection_rows}"
            )

    runtime_manifest = {
        "schema_version": "1.0.0",
        "role": "BLIND_RUNTIME_INPUT",
        "case_count": len(runtime_cases),
        "cases": runtime_cases,
        "human_or_gold_fields_present": False,
        "posthoc_key_path_present": False,
    }
    forbidden = forbidden_inference_paths(runtime_manifest)
    if forbidden:
        raise ValueError(
            "runtime clean-control manifest leaked oracle fields: "
            + ", ".join(forbidden)
        )
    runtime_path = args.runtime_manifest.resolve()
    _write(runtime_path, runtime_manifest)

    posthoc_key = {
        "schema_version": "1.0.0",
        "role": "POSTHOC_ONLY_DO_NOT_FEED_TO_RUNTIME",
        "curation_policy": (
            "one deterministic human-confirmed CORRECT, evidence-sufficient, "
            "association-stage observed-CREATE control per scene"
        ),
        "labels_path": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "runtime_manifest_path": str(runtime_path),
        "runtime_manifest_sha256": sha256_file(runtime_path),
        "case_count": len(posthoc_cases),
        "cases": posthoc_cases,
    }
    _write(args.posthoc_key.resolve(), posthoc_key)
    print(
        json.dumps(
            {
                "status": "PASS",
                "case_count": len(runtime_cases),
                "runtime_manifest": str(runtime_path),
                "posthoc_key": str(args.posthoc_key.resolve()),
                "blind_case_uids": [row["case_uid"] for row in runtime_cases],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
