#!/usr/bin/env python3
"""Schedule hash-frozen wide-frame evidence requested by the outcome critic."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from conceptgraph.revision.autonomous_identity import (
    build_pairwise_critic_prompt,
    evenly_spaced,
    heldout_assignments_distinguishable,
)
from conceptgraph.revision.evidence_split import sha256_file
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.vlm import _resolve_ref


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


def _payload(request: Mapping[str, Any]) -> dict[str, Any]:
    marker = "FROZEN HELD-OUT OUTCOMES:\n"
    prompt = str(request["prompt"])
    if marker not in prompt:
        raise ValueError("critic prompt does not contain frozen payload")
    value = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(value, dict):
        raise ValueError("critic payload must be one object")
    return value


def _balanced_crop_ids(
    *, state_summaries: Mapping[str, Mapping[str, Any]], limit: int
) -> tuple[str, ...]:
    states = list(state_summaries.values())
    finest = max(
        states,
        key=lambda state: int(state.get("group_count_on_heldout_evidence", 0)),
    )
    groups = [
        tuple(str(item) for item in group.get("evidence_ids") or ())
        for group in finest.get("groups") or ()
        if group.get("evidence_ids")
    ]
    if not groups:
        raise ValueError("no held-out state groups are available for scheduling")
    allocation = [0 for _ in groups]
    remaining = min(limit, sum(len(group) for group in groups))
    while remaining:
        progressed = False
        for index, group in enumerate(groups):
            if allocation[index] >= len(group):
                continue
            allocation[index] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            break
    selected = []
    for group, count in zip(groups, allocation):
        if count:
            selected.extend(evenly_spaced(group, limit=count))
    return tuple(dict.fromkeys(selected))


def _wide_frame_source(provenance: ProvenanceIndex, obs_uid: str) -> Path:
    observation = provenance.get_observation(obs_uid)
    crop_ref = observation.get("crop_ref")
    if not isinstance(crop_ref, Mapping):
        raise FileNotFoundError(f"observation has no crop reference: {obs_uid}")
    crop_path = _resolve_ref(provenance, crop_ref)
    frame_dir = crop_path.parent.name
    source = crop_path.parent.parent.parent / "vis" / f"{frame_dir}.jpg"
    if not source.is_file():
        raise FileNotFoundError(source)
    return source.resolve()


def _freeze_padded_local_context(
    *,
    provenance: ProvenanceIndex,
    obs_uid: str,
    destination: Path,
    expansion_factor: float = 5.0,
    minimum_output_edge: int = 512,
) -> dict[str, Any]:
    """Freeze a local RGB neighborhood without inventing image information."""

    if expansion_factor < 1.0:
        raise ValueError("expansion_factor must be at least one")
    observation = provenance.get_observation(obs_uid)
    bbox = observation.get("bbox_2d")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise ValueError(f"observation has no valid 2D box: {obs_uid}")
    source = _wide_frame_source(provenance, obs_uid)
    with Image.open(source) as raw:
        image = raw.convert("RGB")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        pad_x = width * (expansion_factor - 1.0) / 2.0
        pad_y = height * (expansion_factor - 1.0) / 2.0
        bounds = (
            max(0, math.floor(x1 - pad_x)),
            max(0, math.floor(y1 - pad_y)),
            min(image.width, math.ceil(x2 + pad_x)),
            min(image.height, math.ceil(y2 + pad_y)),
        )
        crop = image.crop(bounds)
        longest = max(crop.size)
        scale = max(1.0, float(minimum_output_edge) / max(1, longest))
        if scale > 1.0:
            crop = crop.resize(
                (
                    max(1, round(crop.width * scale)),
                    max(1, round(crop.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        crop.save(destination, format="PNG")
    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_crop_bounds_xyxy": list(bounds),
        "expansion_factor": expansion_factor,
        "output_size": list(crop.size),
    }


def _critic_requested_more_evidence(
    case_uid: str, critic_results: Mapping[str, Any]
) -> bool:
    rows = [
        row
        for row in critic_results.get("results") or ()
        if str(row.get("case_uid")) == case_uid
    ]
    if not rows or any(row.get("status") != "PASS" for row in rows):
        return False
    for row in rows:
        needed = (row.get("response") or {}).get("critic", {}).get("needed_evidence")
        values = (needed,) if isinstance(needed, str) else tuple(needed or ())
        for value in values:
            normalized = str(value).strip().lower()
            if normalized and not normalized.startswith(
                ("none", "n/a", "not needed", "no additional")
            ):
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", required=True, type=Path)
    parser.add_argument("--critic-results", required=True, type=Path)
    parser.add_argument("--office-run", required=True, type=Path)
    parser.add_argument("--room-run", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--context-image-count", type=int, default=4)
    args = parser.parse_args()

    protocol_path = args.freeze_protocol.resolve()
    protocol = _read(protocol_path)
    critic_path = args.critic_results.resolve()
    critic = _read(critic_path)
    if str(critic.get("freeze_protocol_sha256")) != sha256_file(protocol_path):
        raise ValueError("critic results are not bound to the supplied freeze protocol")
    if protocol.get("runtime_human_or_gold_loaded") is not False:
        raise ValueError("source freeze protocol did not pass oracle isolation")

    output_root = args.output_root.resolve()
    protocol_output = output_root / "scheduled_freeze_protocol.json"
    if protocol_output.exists():
        raise FileExistsError(f"refusing to overwrite {protocol_output}")
    output_root.mkdir(parents=True, exist_ok=True)
    base_runs = {
        "office0": args.office_run.resolve(),
        "room0": args.room_run.resolve(),
    }
    contexts = {}
    scheduled_requests = []
    scheduled_cases = []
    unschedulable_cases = []
    for case in protocol.get("cases") or ():
        case_uid = str(case["case_uid"])
        if not _critic_requested_more_evidence(case_uid, critic):
            continue
        scene_id = str(case["scene_id"])
        if scene_id not in contexts:
            contexts[scene_id] = ProvenanceIndex(base_runs[scene_id])
        provenance = contexts[scene_id]
        source_requests = [
            row
            for row in protocol.get("critic_requests") or ()
            if str(row.get("case_uid")) == case_uid
        ]
        if not source_requests:
            raise ValueError(f"{case_uid}: source critic requests are absent")
        first_path = Path(str(source_requests[0]["path"])).resolve()
        if sha256_file(first_path) != str(source_requests[0]["sha256"]):
            raise ValueError(f"{case_uid}: source request hash drift")
        first_request = _read(first_path)
        first_payload = _payload(first_request)
        if not heldout_assignments_distinguishable(first_payload["executed_states"]):
            unschedulable_cases.append(
                {
                    "case_uid": case_uid,
                    "scene_id": scene_id,
                    "reason": "NO_STATE_SPECIFIC_HELDOUT_ASSIGNMENT",
                    "scheduler_action": "STOP_WITHOUT_FUTILE_WIDE_CONTEXT",
                }
            )
            continue
        selected_ids = _balanced_crop_ids(
            state_summaries=first_payload["executed_states"],
            limit=args.context_image_count,
        )
        image_by_id = {
            str(row["evidence_id"]): row for row in first_request.get("images") or ()
        }
        context_rows = []
        seen_sources = set()
        for context_index, crop_id in enumerate(selected_ids, 1):
            crop_row = image_by_id[crop_id]
            obs_uid = str(crop_row["obs_uid"])
            source = _wide_frame_source(provenance, obs_uid)
            if source in seen_sources:
                continue
            seen_sources.add(source)
            context_id = f"C{context_index:02d}"
            destination = (
                output_root / case_uid / "wide_context" / f"{context_id}.jpg"
            ).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            context_rows.append(
                {
                    "evidence_id": context_id,
                    "obs_uid": obs_uid + "#WIDE_FRAME_CONTEXT",
                    "frame_index": int(crop_row["frame_index"]),
                    "class_name": "scene context",
                    "sha256": sha256_file(destination),
                    "path": str(destination),
                    "source_role": "SCHEDULED_HELDOUT_WIDE_FRAME_CONTEXT",
                    "view_type": "WIDE_FRAME_CONTEXT",
                    "linked_crop_evidence_ids": [crop_id],
                    "source_path": str(source),
                    "source_sha256": sha256_file(source),
                }
            )
            local_id = f"L{context_index:02d}"
            local_destination = (
                output_root / case_uid / "local_context" / f"{local_id}.png"
            ).resolve()
            local_audit = _freeze_padded_local_context(
                provenance=provenance,
                obs_uid=obs_uid,
                destination=local_destination,
            )
            context_rows.append(
                {
                    "evidence_id": local_id,
                    "obs_uid": obs_uid + "#PADDED_LOCAL_CONTEXT",
                    "frame_index": int(crop_row["frame_index"]),
                    "class_name": "local scene context",
                    "sha256": sha256_file(local_destination),
                    "path": str(local_destination),
                    "source_role": "SCHEDULED_HELDOUT_PADDED_LOCAL_CONTEXT",
                    "view_type": "PADDED_LOCAL_CONTEXT",
                    "linked_crop_evidence_ids": [crop_id],
                    **local_audit,
                }
            )
        if not context_rows:
            raise ValueError(f"{case_uid}: scheduler found no wide context images")

        case_request_rows = []
        for source_row in source_requests:
            source_path = Path(str(source_row["path"])).resolve()
            if sha256_file(source_path) != str(source_row["sha256"]):
                raise ValueError(f"source request hash drift: {source_path}")
            source_request = _read(source_path)
            source_payload = _payload(source_request)
            combined_images = list(source_request.get("images") or ()) + context_rows
            prompt = build_pairwise_critic_prompt(
                incident_uid=case_uid,
                evidence_rows=combined_images,
                state_summaries=source_payload["executed_states"],
            )
            request_uid = str(source_request["request_uid"]) + "_EVIDENCE_ROUND_1"
            destination = (
                output_root / case_uid / "critic_requests" / f"{request_uid}.json"
            )
            request = {
                **{
                    key: value
                    for key, value in source_request.items()
                    if key
                    not in {
                        "request_uid",
                        "prompt",
                        "prompt_sha256",
                        "images",
                        "allowed_evidence_ids",
                    }
                },
                "request_uid": request_uid,
                "parent_request_uid": str(source_request["request_uid"]),
                "parent_request_sha256": str(source_row["sha256"]),
                "evidence_schedule_round": 1,
                "scheduler_trigger": "OUTCOME_CRITIC_REQUESTED_ADDITIONAL_CONTEXT",
                "prompt": prompt,
                "prompt_sha256": __import__("hashlib")
                .sha256(prompt.encode("utf-8"))
                .hexdigest(),
                "images": combined_images,
                "allowed_evidence_ids": [
                    str(row["evidence_id"]) for row in combined_images
                ],
                "scheduled_context_evidence_ids": [
                    str(row["evidence_id"]) for row in context_rows
                ],
            }
            _write(destination, request)
            manifest_row = {
                "request_uid": request_uid,
                "parent_request_uid": str(source_request["request_uid"]),
                "case_uid": case_uid,
                "pair_uid": str(source_request["pair_uid"]),
                "order_index": int(source_request["order_index"]),
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
            }
            scheduled_requests.append(manifest_row)
            case_request_rows.append(manifest_row)
        scheduled_cases.append(
            {
                "case_uid": case_uid,
                "scene_id": scene_id,
                "context_image_count": len(context_rows),
                "context_images": context_rows,
                "critic_requests": case_request_rows,
            }
        )

    scheduled_protocol = {
        "schema_version": "1.0.0",
        "role": "DEVELOPMENT_SHADOW_EVIDENCE_SCHEDULE_ROUND_1",
        "parent_freeze_protocol_path": str(protocol_path),
        "parent_freeze_protocol_sha256": sha256_file(protocol_path),
        "trigger_critic_results_path": str(critic_path),
        "trigger_critic_results_sha256": sha256_file(critic_path),
        "case_count": len(scheduled_cases),
        "request_count": len(scheduled_requests),
        "cases": scheduled_cases,
        "critic_requests": scheduled_requests,
        "unschedulable_case_count": len(unschedulable_cases),
        "unschedulable_cases": unschedulable_cases,
        "runtime_human_or_gold_loaded": False,
        "candidate_states_changed": False,
        "proposal_evidence_reused_by_critic": False,
        "scheduled_evidence_role": "HELDOUT_FUTURE_WIDE_FRAME_CONTEXT",
        "production_commit_permitted": False,
    }
    _write(protocol_output, scheduled_protocol)
    print(
        json.dumps(
            {
                "status": (
                    "PASS"
                    if scheduled_requests
                    else "NO_SCHEDULABLE_ADDITIONAL_EVIDENCE"
                ),
                "case_count": len(scheduled_cases),
                "request_count": len(scheduled_requests),
                "unschedulable_case_count": len(unschedulable_cases),
                "output": str(protocol_output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
