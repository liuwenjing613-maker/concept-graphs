#!/usr/bin/env python3
"""Run one fresh stride-10 map with the minimal ali-my-new online sidecar."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Running ``python scripts/run_ali_my_new_online.py`` sets sys.path[0] to the
# scripts directory. Make the repository importable without relying on a shell's
# pre-existing PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.online_mvp import (
    EvidenceRouter,
    LiveDependencyTracker,
    LiveEvidenceLedger,
    OnlineEvidencePacket,
    OnlineScanner,
    TaskContext,
    TicketStore,
    append_jsonl,
    write_json,
)
from scripts.validate_unified_vlm_v2 import (
    PROMPT_VERSION as UNIFIED_VLM_PROMPT_VERSION,
    FrozenRun,
    PreflightDefer,
    call_vlm,
    prepare_case,
    write_case_html,
    write_root_html,
)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _candidate_pool_passes(
    dispatched: int,
    *,
    drain_all_candidates: bool,
) -> tuple[bool, ...]:
    """Return audit-slot choices in the exact order used for one dispatch."""

    preferred_audit = (dispatched + 1) % 10 == 0
    if drain_all_candidates:
        return (preferred_audit, not preferred_audit)
    return (preferred_audit,)


def _mapping_command(args: argparse.Namespace, experiment_root: Path) -> list[str]:
    worktree = Path(args.worktree).resolve()
    return [
        str(Path(args.python).resolve()),
        "conceptgraph/slam/rerun_realtime_mapping.py",
        f"dataset_root={Path(args.dataset_root).resolve()}",
        f"dataset_config={Path(args.dataset_config).resolve()}",
        f"scene_id={args.scene}",
        f"start={args.start}",
        f"end={args.end}",
        f"stride={args.stride}",
        f"image_height={args.image_height}",
        f"image_width={args.image_width}",
        "make_edges=false",
        "use_rerun=false",
        "save_rerun=false",
        f"force_detection={str(args.force_detection).lower()}",
        f"save_detections={str(args.save_detections).lower()}",
        f"detections_exp_suffix={args.detections_exp_suffix}",
        f"exp_suffix={args.exp_suffix}",
        "save_video=false",
        "save_objects_all_frames=false",
        "save_pcd=false",
        "save_json=true",
        "periodically_save_pcd=false",
        "save_evidence=true",
        "evidence_mode=strict",
        "evidence_save_observation_pcd=true",
        "device=cuda",
        "revision.enabled=false",
    ]


def _run_unified_vlm(
    *,
    experiment_root: Path,
    output_root: Path,
    packet: OnlineEvidencePacket,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    timeout: float,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Prepare event/current/diagnostic views, then make one strict V2 call."""

    prepare_started = time.monotonic()
    case_root = output_root / "vlm_object_state_v2"
    try:
        run = FrozenRun(experiment_root, online_subdir=output_root.name)
        case = prepare_case(run, packet.ticket_uid, case_root)
    except PreflightDefer as exc:
        case_dir = case_root / packet.ticket_uid
        failure = {
            "ticket_uid": packet.ticket_uid,
            "status": "PREFLIGHT_DEFER",
            "defer_code": exc.code,
            "defer_detail": exc.detail,
            "prepare_elapsed_seconds": time.monotonic() - prepare_started,
            "api_call_attempted": False,
            "prompt_version": UNIFIED_VLM_PROMPT_VERSION,
        }
        write_json(case_dir / "validation.json", failure)
        return failure
    except Exception as exc:
        case_dir = case_root / packet.ticket_uid
        failure = {
            "ticket_uid": packet.ticket_uid,
            "status": "PREPARE_ERROR",
            "error": f"{type(exc).__name__}:{exc}",
            "prepare_elapsed_seconds": time.monotonic() - prepare_started,
            "api_call_attempted": False,
            "prompt_version": UNIFIED_VLM_PROMPT_VERSION,
        }
        write_json(case_dir / "validation.json", failure)
        return failure

    prepare_elapsed = time.monotonic() - prepare_started
    if prepare_only:
        validation = {
            "status": "PREPARED_ONLY",
            "vlm_called": False,
            "prepare_elapsed_seconds": prepare_elapsed,
            "prompt_version": UNIFIED_VLM_PROMPT_VERSION,
        }
        write_json(case.case_dir / "validation.json", validation)
        write_case_html(case.case_dir)
        return {
            "ticket_uid": packet.ticket_uid,
            "status": "PREPARED_ONLY",
            "output": None,
            "available_identity_targets": list(case.available_identity_targets),
            "available_semantic_targets": list(case.available_semantic_targets),
            "case_dir": str(case.case_dir),
            "prepare_elapsed_seconds": prepare_elapsed,
            "api_call_attempted": False,
            "prompt_version": UNIFIED_VLM_PROMPT_VERSION,
        }
    result = call_vlm(
        case,
        api_key,
        base_url,
        model,
        timeout,
        reasoning_effort=reasoning_effort,
    )
    result.update(
        {
            "available_identity_targets": list(case.available_identity_targets),
            "available_semantic_targets": list(case.available_semantic_targets),
            "allowed_image_ids": ["I1", "I2", "I3"],
            "case_dir": str(case.case_dir),
            "prepare_elapsed_seconds": prepare_elapsed,
            "api_call_attempted": True,
            "prompt_version": UNIFIED_VLM_PROMPT_VERSION,
        }
    )
    return result


def _safe_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticket_uid": response.get("ticket_uid"),
        "status": response.get("status"),
        "output": response.get("output"),
        "available_identity_targets": response.get("available_identity_targets") or [],
        "available_semantic_targets": response.get("available_semantic_targets") or [],
        "allowed_image_ids": response.get("allowed_image_ids") or [],
        "model": response.get("model"),
        "response_id": response.get("response_id"),
        "usage": response.get("usage") or {},
        "elapsed_seconds": response.get("elapsed_seconds"),
        "prepare_elapsed_seconds": response.get("prepare_elapsed_seconds"),
        "api_call_attempted": bool(response.get("api_call_attempted")),
        "prompt_version": response.get("prompt_version"),
        "case_dir": response.get("case_dir"),
        "defer_code": response.get("defer_code"),
        "defer_detail": response.get("defer_detail"),
        "error": response.get("error"),
        "timeline": response.get("timeline"),
        "source_h_snapshot": response.get("source_h_snapshot"),
        "completion_binding": response.get("completion_binding"),
    }


def _ticket_state_from_vlm_response(response: Mapping[str, Any]) -> str:
    """Map a response to ticket state without treating input-only prep as failure."""

    status = str(response.get("status") or "")
    if status == "PREPARED_ONLY":
        return "WAIT_EVIDENCE"
    if status != "VALID":
        return "ABORTED"
    output = response.get("output") or {}
    if output.get("identity_target") == "E0" and output.get("semantic_target") == "L0":
        return "NO_ACTION"
    return "DIAGNOSED"


def _bind_completion_timeline(
    *,
    packet: OnlineEvidencePacket,
    response: dict[str, Any],
    ledger: LiveEvidenceLedger,
    latest_committed: int,
    output_root: Path,
) -> dict[str, Any]:
    """Persist C while keeping the conclusion bound to the immutable H snapshot."""

    timeline = dict(packet.packet_manifest.get("timeline") or {})
    c_frame = max(int(packet.freeze_frame), int(latest_committed))
    c_sequence = ledger.max_sequence_at_frame(c_frame)
    timeline.update(
        {
            "c_frame": c_frame,
            "c_sequence": c_sequence,
            "c_latest_main_map_frame": c_frame,
            "frame_order_valid": (
                int(timeline.get("s_frame", -1))
                <= int(timeline.get("d_frame", -1))
                <= int(timeline.get("h_frame", -1))
                <= c_frame
            ),
        }
    )
    if not timeline["frame_order_valid"]:
        raise ValueError("online timeline violates S <= D <= H <= C")
    h_snapshot = packet.packet_manifest.get("h_snapshot") or {}
    source_h_snapshot = {
        key: h_snapshot.get(key)
        for key in (
            "snapshot_uid",
            "snapshot_sha256",
            "cutoff_frame",
            "cutoff_sequence",
            "watermark_source",
        )
    }
    completion_binding = {
        "schema_version": "ali_my_vlm_completion_binding/1.0",
        "ticket_uid": packet.ticket_uid,
        "vlm_status": response.get("status"),
        "source_h_snapshot": source_h_snapshot,
        "saved_at_c": {
            "frame": c_frame,
            "sequence": c_sequence,
            "latest_main_map_frame": c_frame,
        },
        "constraint_source_must_match_h_snapshot": True,
    }
    packet.packet_manifest["timeline"] = timeline
    packet.packet_manifest["completion_binding"] = completion_binding
    packet_path = output_root / "vlm" / packet.ticket_uid / "evidence" / "packet_manifest.json"
    write_json(packet_path, packet.packet_manifest)
    write_json(
        output_root / "vlm" / packet.ticket_uid / "completion_manifest.json",
        completion_binding,
    )

    response["timeline"] = timeline
    response["source_h_snapshot"] = source_h_snapshot
    response["completion_binding"] = completion_binding
    case_dir_value = response.get("case_dir")
    case_dir = (
        Path(str(case_dir_value))
        if case_dir_value
        else output_root / "vlm_object_state_v2" / packet.ticket_uid
    )
    manifest_path = case_dir / "case_manifest.json"
    if manifest_path.is_file():
        case_manifest = _read_json(manifest_path)
        case_manifest["timeline"] = timeline
        case_manifest["completion_binding"] = completion_binding
        cutoff_audit = dict(case_manifest.get("cutoff_audit") or {})
        cutoff_audit["s_lte_d_lte_h_lte_c"] = timeline["frame_order_valid"]
        cutoff_audit["c_binds_same_h_snapshot"] = (
            source_h_snapshot.get("snapshot_uid") == timeline.get("h_snapshot_uid")
            and source_h_snapshot.get("snapshot_sha256")
            == timeline.get("h_snapshot_sha256")
        )
        case_manifest["cutoff_audit"] = cutoff_audit
        write_json(manifest_path, case_manifest)
    validation_path = case_dir / "validation.json"
    if validation_path.is_file():
        validation = _read_json(validation_path)
        validation["timeline"] = timeline
        validation["source_h_snapshot"] = source_h_snapshot
        write_json(validation_path, validation)
    if manifest_path.is_file():
        write_case_html(case_dir)
    return response


def _tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        values = handle.readlines()
    return "".join(values[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", default="/home/chenkejun/beauty/conceptgraphs"
    )
    parser.add_argument(
        "--worktree",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--python", default="/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python"
    )
    parser.add_argument(
        "--dataset-root",
        help="Dataset root containing <scene>/results and <scene>/traj.txt",
    )
    parser.add_argument(
        "--dataset-config",
        help="ConceptGraphs dataset camera YAML (defaults to Replica)",
    )
    parser.add_argument("--scene", default="room0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--image-height", type=int, default=680)
    parser.add_argument("--image-width", type=int, default=1200)
    parser.add_argument("--force-detection", action="store_true")
    parser.add_argument("--save-detections", action="store_true")
    parser.add_argument("--exp-suffix", required=True)
    parser.add_argument("--detections-exp-suffix", default="room0_detections_stride10")
    parser.add_argument("--reuse-experiment-root", type=Path)
    parser.add_argument("--output-subdir", default="online_mvp")
    parser.add_argument("--mapping-gpu", default="0")
    parser.add_argument("--replay-gpu", default="1")
    parser.add_argument("--api-key-count", type=int, default=5)
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=None,
        help="worker count; one or more workers may share each memory-only key",
    )
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help="optional OpenAI-compatible reasoning_effort passed to every VLM call",
    )
    parser.add_argument(
        "--routing-mode",
        choices=("shadow", "active"),
        default="shadow",
        help="candidate routing mode; active moves likely-resolved tickets to AUDIT_POOL",
    )
    parser.add_argument("--max-vlm-tickets", type=int, default=15)
    parser.add_argument(
        "--drain-all-candidates",
        action="store_true",
        help="after mapping ends, process every remaining main/audit candidate",
    )
    parser.add_argument(
        "--ticket-uid",
        action="append",
        default=[],
        help="optional exact ticket allowlist for frozen validation",
    )
    parser.add_argument("--max-shadow-tickets", type=int, default=3)
    parser.add_argument("--min-ticket-age-frames", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--task-context", type=Path)
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument(
        "--prepare-vlm-only",
        action="store_true",
        help="build and audit V2 evidence cards without making API calls",
    )
    args = parser.parse_args()

    if args.stride != 10:
        raise ValueError("ali-my-new MVP is intentionally frozen to stride=10")
    if args.api_key_count < 1 and not args.no_vlm:
        raise ValueError("at least one API key is required")
    if args.api_concurrency is not None and args.api_concurrency < 1:
        raise ValueError("api-concurrency must be positive")
    if Path(args.output_subdir).name != args.output_subdir:
        raise ValueError("output-subdir must be one plain directory name")
    project_root = Path(args.project_root).resolve()
    worktree = Path(args.worktree).resolve()
    args.dataset_root = str(
        Path(args.dataset_root).resolve()
        if args.dataset_root
        else project_root / "data" / "Replica"
    )
    args.dataset_config = str(
        Path(args.dataset_config).resolve()
        if args.dataset_config
        else worktree
        / "conceptgraph"
        / "dataset"
        / "dataconfigs"
        / "replica"
        / "replica.yaml"
    )
    dataset_root = Path(args.dataset_root)
    if args.reuse_experiment_root:
        experiment_root = args.reuse_experiment_root.resolve()
    else:
        experiment_root = dataset_root / args.scene / "exps" / args.exp_suffix
    output_root = experiment_root / args.output_subdir
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite online output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    live_log = output_root / "online_events.jsonl"
    task_context = TaskContext.from_mapping(_read_json(args.task_context))

    api_keys: list[str] = []
    if not args.no_vlm and not args.prepare_vlm_only:
        for index in range(args.api_key_count):
            value = getpass.getpass(
                f"API key {index + 1}/{args.api_key_count} (memory only): "
            ).strip()
            if not value:
                raise ValueError("empty API key")
            api_keys.append(value)

    worker_count = max(
        1,
        int(args.api_concurrency or len(api_keys) or args.api_key_count),
    )

    protocol = {
        "schema_version": "0.1.0",
        "experiment_root": str(experiment_root),
        "output_subdir": args.output_subdir,
        "worktree": str(worktree),
        "dataset_root": str(dataset_root),
        "dataset_config": args.dataset_config,
        "scene": args.scene,
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "force_detection": args.force_detection,
        "save_detections": args.save_detections,
        "mapping_gpu": args.mapping_gpu,
        "replay_gpu": args.replay_gpu,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "routing_mode": args.routing_mode,
        "base_url": args.base_url,
        "api_credential_slots": len(api_keys),
        "api_concurrency": 0 if args.no_vlm else worker_count,
        "api_keys_persisted": False,
        "drain_all_candidates": args.drain_all_candidates,
        "one_call_per_ticket": True,
        "prepare_vlm_only": args.prepare_vlm_only,
        "ticket_uid_allowlist": list(args.ticket_uid),
        "vlm_prompt_version": UNIFIED_VLM_PROMPT_VERSION,
        "vlm_input_image_count": 3,
        "vlm_output_contract": "object_state_v2",
        "vlm_parser_enabled": False,
        "repair_execution_enabled": False,
        "compound_partition_policy": "diagnose_and_report_executor_unsupported",
        "semantic_policy": "declarative_label_target_shadow_diagnosis_only",
        "semantic_accuracy_validated": False,
        "annotations_loaded": False,
        "ground_truth_loaded": False,
        "make_edges": False,
        "commit_scope": "experiment_local_version_pointer_at_scene_end",
        "task_context": {
            "task_id": task_context.task_id,
            "active": task_context.active,
            "required_lineage_count": len(task_context.required_lineage_uids),
            "required_object_count": len(task_context.required_object_uids),
            "required_relation_count": len(task_context.required_relation_uids),
        },
        "started_at_unix": time.time(),
    }
    write_json(output_root / "protocol.json", protocol)

    mapping_process: subprocess.Popen[str] | None = None
    mapping_log = output_root / "mapping.log"
    mapping_handle = None
    if not args.reuse_experiment_root:
        command = _mapping_command(args, experiment_root)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.mapping_gpu)
        environment["PYTHONPATH"] = str(worktree)
        # Reuse the server workspace caches instead of silently downloading
        # multi-gigabyte detector/encoder weights into the login user's cache.
        environment.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
        environment.setdefault(
            "HF_HOME", str(project_root / "models" / "huggingface")
        )
        environment.setdefault("TORCH_HOME", str(project_root / "models" / "torch"))
        environment.setdefault(
            "YOLO_CONFIG_DIR", str(project_root / ".config" / "Ultralytics")
        )
        environment.setdefault(
            "MPLCONFIGDIR", str(project_root / ".config" / "matplotlib")
        )
        mapping_handle = mapping_log.open("w", encoding="utf-8", newline="\n")
        mapping_process = subprocess.Popen(
            command,
            cwd=worktree,
            env=environment,
            stdout=mapping_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"MAPPER_STARTED pid={mapping_process.pid} root={experiment_root}", flush=True)
    else:
        print(f"REUSING_COMPLETED_RUN root={experiment_root}", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.replay_gpu)
    ledger = LiveEvidenceLedger(experiment_root)
    scanner = OnlineScanner()
    tracker = LiveDependencyTracker()
    tickets = TicketStore()
    router = EvidenceRouter(experiment_root, max_images=6)
    vlm_executor = ThreadPoolExecutor(max_workers=worker_count)
    clients = (
        [api_keys[index % len(api_keys)] for index in range(worker_count)]
        if api_keys
        else ([""] * worker_count if args.prepare_vlm_only else [])
    )
    slot_futures: dict[int, Future[dict[str, Any]]] = {}
    slot_packets: dict[int, OnlineEvidencePacket] = {}
    locked_lineages: set[str] = set()
    dispatched = 0
    api_calls_attempted = 0
    latest_committed = -1
    diagnosed_results: list[dict[str, Any]] = []
    terminal_vlm_tickets: set[str] = set()
    dispatch_records: list[dict[str, Any]] = []

    try:
        idle_rounds_after_done = 0
        while True:
            mapping_done = mapping_process is None or mapping_process.poll() is not None
            committed = ledger.poll(mapping_done=mapping_done)
            for committed_frame in committed:
                latest_committed = max(latest_committed, committed_frame)
                issues = scanner.scan_frame(committed_frame, ledger)
                for issue in issues:
                    ticket = tickets.upsert(issue)
                    append_jsonl(
                        live_log,
                        {
                            "type": "ISSUE_UPSERT",
                            "frame": committed_frame,
                            "ticket_uid": ticket.ticket_uid,
                            "issue": issue.as_dict(),
                        },
                    )
                if not args.reuse_experiment_root:
                    tickets.refresh(
                        ledger=ledger,
                        tracker=tracker,
                        task_context=task_context,
                        stop_sequence=ledger.max_sequence_at_frame(committed_frame),
                        cutoff_frame=committed_frame,
                        routing_mode=args.routing_mode,
                    )
                if committed_frame % 10 == 0:
                    print(
                        "WATERMARK "
                        f"frame={committed_frame} sequence={ledger.max_sequence_at_frame(committed_frame)} "
                        f"tickets={len(tickets.tickets)} dispatched={dispatched} "
                        "repair=disabled",
                        flush=True,
                    )

            if committed and args.reuse_experiment_root:
                # A completed ledger arrives as one batch; ranking every historical
                # frame would repeat the same full causal closure hundreds of times.
                tickets.refresh(
                    ledger=ledger,
                    tracker=tracker,
                    task_context=task_context,
                    stop_sequence=ledger.max_sequence_at_frame(latest_committed),
                    cutoff_frame=latest_committed,
                    routing_mode=args.routing_mode,
                )

            for slot, future in list(slot_futures.items()):
                if not future.done():
                    continue
                packet = slot_packets.pop(slot)
                del slot_futures[slot]
                ticket = tickets.tickets[packet.ticket_uid]
                try:
                    raw_response = future.result()
                    raw_response = _bind_completion_timeline(
                        packet=packet,
                        response=raw_response,
                        ledger=ledger,
                        latest_committed=latest_committed,
                        output_root=output_root,
                    )
                    response = _safe_response(raw_response)
                    if response.get("api_call_attempted"):
                        api_calls_attempted += 1
                    response_path = output_root / "vlm" / packet.ticket_uid / "response.json"
                    write_json(response_path, response)
                    output = response.get("output") or {}
                    valid = response.get("status") == "VALID"
                    prepared_only = response.get("status") == "PREPARED_ONLY"
                    parser_status = (
                        "NOT_CALLED_INPUT_ONLY"
                        if prepared_only
                        else "DISABLED_FOR_FIRST_VALIDATION"
                    )
                    ticket.attempts.append(
                        {
                            "stage": "VLM",
                            "status": response.get("status"),
                            "response_path": str(response_path),
                            "output_contract": "object_state_v2",
                            "parser_status": parser_status,
                            "timeline": response.get("timeline"),
                            "source_h_snapshot": response.get("source_h_snapshot"),
                        }
                    )
                    if valid:
                        diagnosed_results.append(
                            {
                                "ticket_uid": packet.ticket_uid,
                                "output": output,
                                "freeze_frame": packet.freeze_frame,
                                "freeze_sequence": packet.freeze_sequence,
                                "timeline": response.get("timeline"),
                                "source_h_snapshot": response.get("source_h_snapshot"),
                            }
                        )
                    ticket.state = _ticket_state_from_vlm_response(response)
                    terminal_vlm_tickets.add(ticket.ticket_uid)
                    locked_lineages.difference_update(ticket.primary_lineage_uids)
                    append_jsonl(
                        live_log,
                        {
                            "type": (
                                "VLM_INPUT_PREPARED" if prepared_only else "VLM_COMPLETED"
                            ),
                            "ticket_uid": packet.ticket_uid,
                            "slot": slot,
                            "vlm_status": response.get("status"),
                            "object_state_v2": output if valid else None,
                            "parser_status": parser_status,
                            "timeline": response.get("timeline"),
                            "source_h_snapshot": response.get("source_h_snapshot"),
                        },
                    )
                except Exception as exc:
                    ticket.state = "ABORTED"
                    ticket.attempts.append(
                        {
                            "stage": "VLM",
                            "status": "ERROR",
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
                    terminal_vlm_tickets.add(ticket.ticket_uid)
                    locked_lineages.difference_update(ticket.primary_lineage_uids)
                    append_jsonl(
                        live_log,
                        {
                            "type": "VLM_FAILED",
                            "ticket_uid": packet.ticket_uid,
                            "error": f"{type(exc).__name__}:{exc}",
                            "h_frame": packet.freeze_frame,
                            "h_sequence": packet.freeze_sequence,
                            "h_snapshot_uid": (
                                packet.packet_manifest.get("h_snapshot") or {}
                            ).get("snapshot_uid"),
                            "latest_main_map_frame": max(
                                int(packet.freeze_frame), int(latest_committed)
                            ),
                        },
                    )

            if not args.no_vlm and dispatched < args.max_vlm_tickets:
                free_slots = [index for index in range(len(clients)) if index not in slot_futures]
                for slot in free_slots:
                    if dispatched >= args.max_vlm_tickets:
                        break
                    selected = None
                    for audit_slot in _candidate_pool_passes(
                        dispatched,
                        drain_all_candidates=args.drain_all_candidates,
                    ):
                        for ticket in tickets.ordered(
                            current_frame=max(0, latest_committed),
                            audit_slot=audit_slot,
                        ):
                            if ticket.ticket_uid in terminal_vlm_tickets:
                                continue
                            if args.ticket_uid and ticket.ticket_uid not in set(args.ticket_uid):
                                continue
                            if ticket.dispatch_frame is not None:
                                continue
                            age = max(0, latest_committed) - ticket.first_seen_frame
                            if (
                                age < args.min_ticket_age_frames
                                and not (mapping_done and args.drain_all_candidates)
                            ):
                                continue
                            if locked_lineages.intersection(ticket.primary_lineage_uids):
                                continue
                            packet = router.build_v2(
                                ticket=ticket,
                                ledger=ledger,
                                freeze_frame=max(0, latest_committed),
                                freeze_sequence=ledger.max_sequence_at_frame(
                                    max(0, latest_committed)
                                ),
                                output_dir=output_root / "vlm" / ticket.ticket_uid / "evidence",
                            )
                            if packet is None:
                                if mapping_done and args.drain_all_candidates:
                                    rank = len(dispatch_records) + 1
                                    record = {
                                        "dispatch_rank": rank,
                                        "ticket_uid": ticket.ticket_uid,
                                        "disposition": "NO_EVIDENCE_PACKET",
                                        "pool_location": ticket.pool_location,
                                        "routing_state": ticket.routing_state,
                                        "priority": ticket.as_dict(
                                            max(0, latest_committed)
                                        ).get("priority_tuple"),
                                    }
                                    dispatch_records.append(record)
                                    terminal_vlm_tickets.add(ticket.ticket_uid)
                                    case_dir = (
                                        output_root
                                        / "vlm_object_state_v2"
                                        / ticket.ticket_uid
                                    )
                                    write_json(
                                        case_dir / "validation.json",
                                        {
                                            "status": "NO_EVIDENCE_PACKET",
                                            "api_call_attempted": False,
                                            "dispatch_rank": rank,
                                        },
                                    )
                                    append_jsonl(
                                        live_log,
                                        {
                                            "type": "VLM_SKIPPED_NO_PACKET",
                                            **record,
                                        },
                                    )
                                continue
                            selected = (ticket, packet)
                            break
                        if selected is not None:
                            break
                    if selected is None:
                        break
                    ticket, packet = selected
                    dispatch_rank = len(dispatch_records) + 1
                    dispatch_record = {
                        "dispatch_rank": dispatch_rank,
                        "ticket_uid": ticket.ticket_uid,
                        "disposition": "VLM_DISPATCHED",
                        "pool_location": ticket.pool_location,
                        "routing_state": ticket.routing_state,
                        "h_frame": packet.freeze_frame,
                        "h_sequence": packet.freeze_sequence,
                        "priority": ticket.as_dict(max(0, latest_committed)).get(
                            "priority_tuple"
                        ),
                    }
                    dispatch_records.append(dispatch_record)
                    ticket.state = "DIAGNOSING"
                    ticket.dispatch_frame = packet.freeze_frame
                    ticket.dispatch_sequence = packet.freeze_sequence
                    locked_lineages.update(ticket.primary_lineage_uids)
                    slot_packets[slot] = packet
                    slot_futures[slot] = vlm_executor.submit(
                        _run_unified_vlm,
                        experiment_root=experiment_root,
                        output_root=output_root,
                        packet=packet,
                        api_key=clients[slot],
                        base_url=args.base_url,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        timeout=300.0,
                        prepare_only=args.prepare_vlm_only,
                    )
                    dispatched += 1
                    append_jsonl(
                        live_log,
                        {
                            "type": "VLM_STARTED",
                            "dispatch_rank": dispatch_rank,
                            "ticket_uid": ticket.ticket_uid,
                            "slot": slot,
                            "freeze_frame": packet.freeze_frame,
                            "freeze_sequence": packet.freeze_sequence,
                            "h_snapshot_uid": (
                                packet.packet_manifest.get("h_snapshot") or {}
                            ).get("snapshot_uid"),
                            "latest_main_map_frame": max(0, latest_committed),
                            "priority": ticket.as_dict(max(0, latest_committed)).get(
                                "priority_tuple"
                            ),
                        },
                    )

            active_work = bool(slot_futures)
            if mapping_done:
                if mapping_process is not None and mapping_process.returncode not in (0, None):
                    break
                if args.drain_all_candidates:
                    target_uids = {
                        ticket_uid
                        for ticket_uid in tickets.tickets
                        if not args.ticket_uid or ticket_uid in set(args.ticket_uid)
                    }
                    pending_uids = target_uids - terminal_vlm_tickets
                    if active_work:
                        idle_rounds_after_done = 0
                    elif not pending_uids:
                        break
                    else:
                        idle_rounds_after_done += 1
                    if idle_rounds_after_done >= 3:
                        append_jsonl(
                            live_log,
                            {
                                "type": "VLM_DRAIN_INCOMPLETE",
                                "pending_ticket_uids": sorted(pending_uids),
                                "max_vlm_tickets": args.max_vlm_tickets,
                            },
                        )
                        break
                    time.sleep(max(0.05, args.poll_seconds))
                    continue
                if active_work:
                    idle_rounds_after_done = 0
                else:
                    idle_rounds_after_done += 1
                # Three final rounds allow newly committed last-frame evidence to dispatch.
                if idle_rounds_after_done >= 3:
                    break
            time.sleep(max(0.05, args.poll_seconds))
    finally:
        vlm_executor.shutdown(wait=True, cancel_futures=False)
        if mapping_handle is not None:
            mapping_handle.close()

    mapping_returncode = 0 if mapping_process is None else int(mapping_process.returncode or 0)
    ledger.poll(mapping_done=True)
    tickets.refresh(
        ledger=ledger,
        tracker=tracker,
        task_context=task_context,
        stop_sequence=ledger.max_sequence_at_frame(max(ledger.frames, default=-1)),
        cutoff_frame=max(ledger.frames, default=-1),
        routing_mode=args.routing_mode,
    )
    write_json(
        output_root / "tickets.json",
        {
            "ticket_count": len(tickets.tickets),
            "tickets": [
                ticket.as_dict(max(0, latest_committed))
                for ticket in sorted(tickets.tickets.values(), key=lambda item: item.ticket_uid)
            ],
        },
    )
    if mapping_returncode != 0:
        failure = {
            "status": "MAPPING_FAILED",
            "returncode": mapping_returncode,
            "mapping_log_tail": _tail(mapping_log),
        }
        write_json(output_root / "run_failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return 2

    target_ticket_uids = {
        ticket_uid
        for ticket_uid in tickets.tickets
        if not args.ticket_uid or ticket_uid in set(args.ticket_uid)
    }
    undrained_ticket_uids = sorted(target_ticket_uids - terminal_vlm_tickets)
    write_json(
        output_root / "dispatch_order.json",
        {
            "candidate_count": len(target_ticket_uids),
            "ordered_candidate_count": len(dispatch_records),
            "undrained_ticket_uids": undrained_ticket_uids,
            "records": dispatch_records,
        },
    )
    visualization_root = output_root / "vlm_object_state_v2"
    if visualization_root.is_dir():
        write_root_html(
            visualization_root,
            ordered_ticket_uids=[row["ticket_uid"] for row in dispatch_records],
        )
    resolution_counts: dict[str, int] = {}
    error_tier_counts: dict[str, int] = {}
    for ticket in tickets.tickets.values():
        resolution_counts[ticket.resolution_state] = (
            resolution_counts.get(ticket.resolution_state, 0) + 1
        )
        tier = str(ticket.as_dict().get("error_tier_name"))
        error_tier_counts[tier] = error_tier_counts.get(tier, 0) + 1
    output_counts: dict[str, int] = {}
    for row in diagnosed_results:
        output = row.get("output") or {}
        key = f"{output.get('identity_target')}+{output.get('semantic_target')}"
        output_counts[key] = output_counts.get(key, 0) + 1
    summary = {
        "schema_version": "ali_my_houxuan_vlm/2.0",
        "status": (
            "COMPLETED"
            if not undrained_ticket_uids
            else "COMPLETED_WITH_UNDRAINED_CANDIDATES"
        ),
        "experiment_root": str(experiment_root),
        "fresh_online_mapping": not bool(args.reuse_experiment_root),
        "processed_committed_frame_count": len(ledger._committed_frames),
        "final_frame": max(ledger._committed_frames, default=-1),
        "final_event_sequence": ledger.max_sequence_at_frame(
            max(ledger._committed_frames, default=-1)
        ),
        "ticket_count": len(tickets.tickets),
        "resolution_counts": resolution_counts,
        "error_tier_counts": error_tier_counts,
        "vlm_dispatched": dispatched,
        "candidate_target_count": len(target_ticket_uids),
        "candidate_ordered_count": len(dispatch_records),
        "undrained_candidate_count": len(undrained_ticket_uids),
        "undrained_ticket_uids": undrained_ticket_uids,
        "vlm_api_calls_attempted": api_calls_attempted,
        "vlm_valid_output_count": len(diagnosed_results),
        "vlm_output_counts": output_counts,
        "vlm_prompt_version": UNIFIED_VLM_PROMPT_VERSION,
        "reasoning_effort": args.reasoning_effort,
        "routing_mode": args.routing_mode,
        "drain_all_candidates": args.drain_all_candidates,
        "api_concurrency": 0 if args.no_vlm else worker_count,
        "vlm_output_contract": "object_state_v2",
        "object_state_parser_enabled": False,
        "repair_execution_enabled": False,
        "shadow_started": 0,
        "combined_constraint_count": 0,
        "semantic_accuracy_validated": False,
        "activated_repaired_version": False,
        "visualization_index": str(visualization_root / "index.html"),
        "annotations_loaded": False,
        "ground_truth_loaded": False,
        "api_keys_persisted": False,
        "finished_at_unix": time.time(),
    }
    write_json(output_root / "run_summary.json", summary)
    print("ONLINE_MVP_COMPLETED", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
