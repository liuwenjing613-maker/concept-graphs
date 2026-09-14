#!/usr/bin/env python3
"""Remove previously exposed controls from a frozen blind pilot intake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from conceptgraph.revision.auto_constraints import forbidden_inference_paths
from conceptgraph.revision.evidence_split import sha256_file
from scripts.freeze_revision_identity_selective_v0 import _read, _write


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--private-audit", required=True, type=Path)
    parser.add_argument("--exclude-incident", action="append", required=True)
    parser.add_argument("--output-runtime-manifest", required=True, type=Path)
    parser.add_argument("--output-private-audit", required=True, type=Path)
    args = parser.parse_args()

    runtime_path = args.runtime_manifest.resolve()
    private_path = args.private_audit.resolve()
    runtime = _read(runtime_path)
    private = _read(private_path)
    excluded = {str(item) for item in args.exclude_incident}
    source_by_case = {
        str(row["case_uid"]): str(row["source_incident_uid"])
        for row in private.get("eligible_cases") or ()
    }
    kept = [
        dict(row)
        for row in runtime.get("cases") or ()
        if source_by_case.get(str(row["case_uid"])) not in excluded
    ]
    removed = [
        {
            "case_uid": str(row["case_uid"]),
            "source_incident_uid": source_by_case.get(str(row["case_uid"])),
            "reason": "PRIOR_DEVELOPMENT_CONTROL_OVERLAP",
        }
        for row in runtime.get("cases") or ()
        if source_by_case.get(str(row["case_uid"])) in excluded
    ]
    if not kept:
        raise ValueError("control-overlap exclusion removed every blind case")
    output = {
        **runtime,
        "role": "CMVIC_ROOM0_BLIND_PILOT_RUNTIME_INPUT_CONTROL_DISJOINT",
        "selection_policy": (
            str(runtime.get("selection_policy"))
            + "_THEN_EXCLUDE_PREVIOUSLY_EXPOSED_DEVELOPMENT_CONTROLS"
        ),
        "source_runtime_manifest_path": str(runtime_path),
        "source_runtime_manifest_sha256": sha256_file(runtime_path),
        "excluded_prior_control_incident_count": len(excluded),
        "case_count": len(kept),
        "cases": kept,
    }
    output.pop("eligible_pool", None)
    output["eligible_pool_count_before_control_exclusion"] = runtime.get(
        "eligible_pool_count"
    )
    output["runtime_human_or_gold_loaded"] = False
    forbidden = forbidden_inference_paths(output)
    if forbidden:
        raise ValueError("oracle-like filtered runtime fields: " + ", ".join(forbidden))
    output_path = args.output_runtime_manifest.resolve()
    _write(output_path, output)
    private_output: dict[str, Any] = {
        "schema_version": "1.0.0",
        "role": "PRIVATE_CONTROL_DISJOINT_FILTER_TRACE_NOT_INFERENCE_INPUT",
        "source_runtime_manifest_path": str(runtime_path),
        "source_runtime_manifest_sha256": sha256_file(runtime_path),
        "source_private_audit_path": str(private_path),
        "source_private_audit_sha256": sha256_file(private_path),
        "output_runtime_manifest_path": str(output_path),
        "output_runtime_manifest_sha256": sha256_file(output_path),
        "human_labels_loaded_for_filter": False,
        "filter_basis": "CASE_UID_PREVIOUSLY_EXPOSED_AS_DEVELOPMENT_CONTROL",
        "excluded_incident_uids": sorted(excluded),
        "removed_cases": removed,
        "kept_case_uids": [str(row["case_uid"]) for row in kept],
    }
    _write(args.output_private_audit.resolve(), private_output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "input_count": len(runtime.get("cases") or ()),
                "removed_count": len(removed),
                "output_count": len(kept),
                "output_runtime_manifest": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
