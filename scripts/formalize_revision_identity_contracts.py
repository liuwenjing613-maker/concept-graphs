#!/usr/bin/env python3
"""Derive explicit pairwise identity contracts from frozen causal evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.constraints import SparseRepairConstraint
from conceptgraph.revision.index import ProvenanceIndex


def _scene_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--scene-base-run must use SCENE=PATH: {value}")
        scene_id, raw_path = value.split("=", 1)
        scene_id = scene_id.strip()
        if not scene_id or scene_id in result:
            raise ValueError(f"invalid or duplicate scene id: {scene_id!r}")
        result[scene_id] = Path(raw_path).expanduser().resolve()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lineages_for_merge_side(
    provenance: ProvenanceIndex,
    event: Mapping[str, Any],
    entity_uid: str,
) -> tuple[str, ...]:
    lineages: set[str] = set()
    prefix = f"{entity_uid}@"
    for version_uid in event.get("input_object_version_uids") or ():
        if not str(version_uid).startswith(prefix):
            continue
        version = provenance.object_versions.get(str(version_uid)) or {}
        lineage_uid = version.get("lineage_uid")
        if lineage_uid:
            lineages.add(str(lineage_uid))
        for obs_uid in version.get("member_observation_uids") or ():
            try:
                association = provenance.get_association_for_obs(str(obs_uid))
                after_uid = association.get("target_object_version_after")
                origin_version = provenance.object_versions.get(str(after_uid)) or {}
                origin_lineage = origin_version.get("lineage_uid")
                if origin_lineage:
                    lineages.add(str(origin_lineage))
            except (KeyError, TypeError):
                continue
    return tuple(sorted(lineages))


def _derive_pair_contract(
    provenance: ProvenanceIndex,
    constraint: Mapping[str, Any],
) -> dict[str, Any]:
    created_entity_uid = str(constraint.get("created_entity_uid") or "")
    created_identity_uid = str(
        constraint.get("created_identity_uid")
        or constraint.get("created_lineage_uid")
        or ("revision-lineage:" + str(constraint.get("obs_uid") or ""))
    )
    if not created_entity_uid or not created_identity_uid:
        raise ValueError("CREATE_INSTANCE lacks a stable created entity or identity")

    explicit = tuple(
        sorted(
            {
                str(item)
                for item in constraint.get("separate_from_identity_uids") or ()
                if str(item)
            }
        )
    )
    if explicit:
        return {
            "created_identity_uid": created_identity_uid,
            "separate_from_identity_uids": explicit,
            "evidence_event_uids": tuple(),
            "counterpart_entity_uids": tuple(),
            "derivation": "PREEXISTING_EXPLICIT_CONTRACT",
        }

    candidates: list[dict[str, Any]] = []
    anchor_sequence = int(constraint.get("active_from_sequence") or -1)
    for evidence_ref in constraint.get("evidence_refs") or ():
        event = provenance.events.get(str(evidence_ref))
        if not event or str(event.get("event_type")) != "OBJECT_MERGE":
            continue
        if provenance.sequence(event) <= anchor_sequence:
            continue
        source_uid = str(event.get("source_object_uid") or "")
        target_uid = str(event.get("target_object_uid") or "")
        if created_entity_uid == source_uid:
            counterpart_uid = target_uid
        elif created_entity_uid == target_uid:
            counterpart_uid = source_uid
        else:
            continue
        lineages = _lineages_for_merge_side(provenance, event, counterpart_uid)
        if not counterpart_uid or not lineages:
            raise ValueError(
                f"cannot resolve counterpart identity for evidence event {evidence_ref}"
            )
        candidates.append(
            {
                "event_uid": str(event["event_uid"]),
                "counterpart_entity_uid": counterpart_uid,
                "counterpart_identity_uids": lineages,
            }
        )

    counterpart_entities = {item["counterpart_entity_uid"] for item in candidates}
    if len(counterpart_entities) != 1:
        raise ValueError(
            "CREATE_INSTANCE needs exactly one evidence-bound merge counterpart; "
            f"created_entity_uid={created_entity_uid} candidates={candidates}"
        )
    separate = tuple(
        sorted(
            {
                identity_uid
                for item in candidates
                for identity_uid in item["counterpart_identity_uids"]
                if identity_uid != created_identity_uid
            }
        )
    )
    if not separate:
        raise ValueError(
            f"CREATE_INSTANCE pair boundary is empty for {created_entity_uid}"
        )
    return {
        "created_identity_uid": created_identity_uid,
        "separate_from_identity_uids": separate,
        "evidence_event_uids": tuple(
            sorted({item["event_uid"] for item in candidates})
        ),
        "counterpart_entity_uids": tuple(sorted(counterpart_entities)),
        "derivation": "FROZEN_OBJECT_MERGE_EVIDENCE",
    }


def formalize_manifest(
    manifest: Mapping[str, Any],
    provenances: Mapping[str, ProvenanceIndex],
    *,
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    derived = copy.deepcopy(dict(manifest))
    contracts: list[dict[str, Any]] = []
    for case in derived.get("cases") or ():
        scene_id = str(case.get("scene_id") or "")
        provenance = provenances.get(scene_id)
        if provenance is None:
            raise ValueError(f"missing --scene-base-run for {scene_id}")
        for constraint in case.get("constraints") or ():
            if str(constraint.get("type") or "").upper() != "CREATE_INSTANCE":
                continue
            contract = _derive_pair_contract(provenance, constraint)
            constraint["created_identity_uid"] = contract["created_identity_uid"]
            constraint["separate_from_identity_uids"] = list(
                contract["separate_from_identity_uids"]
            )
            SparseRepairConstraint.from_mapping(constraint)
            contracts.append(
                {
                    "case_uid": str(case["case_uid"]),
                    "scene_id": scene_id,
                    "obs_uid": str(constraint["obs_uid"]),
                    **{
                        key: list(value) if isinstance(value, tuple) else value
                        for key, value in contract.items()
                    },
                }
            )

    derived["identity_semantics_version"] = "2.0.0"
    derived["identity_contract_formalization"] = {
        "source_manifest_sha256": source_manifest_sha256,
        "derivation_uses_replay_outcomes": False,
        "derivation_inputs": [
            "frozen CREATE_INSTANCE constraint",
            "frozen evidence_refs",
            "immutable OBJECT_MERGE event",
            "immutable object-version lineage",
        ],
        "unknown_or_ambiguous_contract_policy": "FAIL_CLOSED",
        "contract_count": len(contracts),
        "contracts": contracts,
    }
    return derived, derived["identity_contract_formalization"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--scene-base-run",
        action="append",
        default=[],
        metavar="SCENE=PATH",
    )
    args = parser.parse_args()

    source = args.input_manifest.resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    scene_paths = _scene_paths(args.scene_base_run)
    provenances = {
        scene_id: ProvenanceIndex(path)
        for scene_id, path in sorted(scene_paths.items())
    }
    derived, audit = formalize_manifest(
        manifest,
        provenances,
        source_manifest_sha256=_sha256(source),
    )
    output = args.output_manifest.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_manifest": str(output),
                "contract_count": audit["contract_count"],
                "contracts": audit["contracts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
