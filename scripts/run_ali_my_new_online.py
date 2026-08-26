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

from conceptgraph.revision.online_mvp import (
    EvidenceRouter,
    LiveDependencyTracker,
    LiveEvidenceLedger,
    OnlineEvidencePacket,
    OnlineScanner,
    TaskContext,
    TicketStore,
    append_jsonl,
    compile_vlm_response,
    freeze_watermarked_view,
    run_final_combined_replay,
    run_shadow_validation,
    write_json,
)
from conceptgraph.revision.vlm import OpenAICompatibleConstraintClient


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _mapping_command(args: argparse.Namespace, experiment_root: Path) -> list[str]:
    worktree = Path(args.worktree).resolve()
    root = Path(args.project_root).resolve()
    return [
        str(Path(args.python).resolve()),
        "conceptgraph/slam/rerun_realtime_mapping.py",
        f"dataset_root={root / 'data' / 'Replica'}",
        f"dataset_config={worktree / 'conceptgraph' / 'dataset' / 'dataconfigs' / 'replica' / 'replica.yaml'}",
        f"scene_id={args.scene}",
        f"start={args.start}",
        f"end={args.end}",
        f"stride={args.stride}",
        "image_height=680",
        "image_width=1200",
        "make_edges=false",
        "use_rerun=false",
        "save_rerun=false",
        "force_detection=false",
        "save_detections=false",
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


def _safe_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "incident_uid": response.get("incident_uid"),
        "constraint": response.get("constraint"),
        "model": response.get("model"),
        "response_id": response.get("response_id"),
        "usage": response.get("usage") or {},
        "elapsed_seconds": response.get("elapsed_seconds"),
    }


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
        default="/home/chenkejun/beauty/conceptgraphs/code/official/ali-my-new",
    )
    parser.add_argument(
        "--python", default="/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python"
    )
    parser.add_argument("--scene", default="room0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--exp-suffix", required=True)
    parser.add_argument("--detections-exp-suffix", default="room0_detections_stride10")
    parser.add_argument("--reuse-experiment-root", type=Path)
    parser.add_argument("--output-subdir", default="online_mvp")
    parser.add_argument("--mapping-gpu", default="0")
    parser.add_argument("--replay-gpu", default="1")
    parser.add_argument("--api-key-count", type=int, default=5)
    parser.add_argument("--base-url", default="https://api.pinaic.com/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--max-vlm-tickets", type=int, default=15)
    parser.add_argument("--max-shadow-tickets", type=int, default=3)
    parser.add_argument("--min-ticket-age-frames", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--task-context", type=Path)
    parser.add_argument("--no-vlm", action="store_true")
    args = parser.parse_args()

    if args.stride != 10:
        raise ValueError("ali-my-new MVP is intentionally frozen to stride=10")
    if args.api_key_count < 1 and not args.no_vlm:
        raise ValueError("at least one API key is required")
    if Path(args.output_subdir).name != args.output_subdir:
        raise ValueError("output-subdir must be one plain directory name")
    project_root = Path(args.project_root).resolve()
    worktree = Path(args.worktree).resolve()
    if args.reuse_experiment_root:
        experiment_root = args.reuse_experiment_root.resolve()
    else:
        experiment_root = (
            project_root / "data" / "Replica" / args.scene / "exps" / args.exp_suffix
        )
    output_root = experiment_root / args.output_subdir
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite online output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    live_log = output_root / "online_events.jsonl"
    task_context = TaskContext.from_mapping(_read_json(args.task_context))

    api_keys: list[str] = []
    if not args.no_vlm:
        for index in range(args.api_key_count):
            value = getpass.getpass(
                f"API key {index + 1}/{args.api_key_count} (memory only): "
            ).strip()
            if not value:
                raise ValueError("empty API key")
            api_keys.append(value)

    protocol = {
        "schema_version": "0.1.0",
        "experiment_root": str(experiment_root),
        "output_subdir": args.output_subdir,
        "worktree": str(worktree),
        "scene": args.scene,
        "start": args.start,
        "end": args.end,
        "stride": args.stride,
        "mapping_gpu": args.mapping_gpu,
        "replay_gpu": args.replay_gpu,
        "model": args.model,
        "base_url": args.base_url,
        "api_credential_slots": len(api_keys),
        "api_keys_persisted": False,
        "one_call_per_ticket": True,
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
    vlm_executor = ThreadPoolExecutor(max_workers=max(1, len(api_keys)))
    shadow_executor = ThreadPoolExecutor(max_workers=1)
    clients = [
        OpenAICompatibleConstraintClient(
            api_key=key,
            base_url=args.base_url,
            model=args.model,
            timeout_seconds=300.0,
        )
        for key in api_keys
    ]
    slot_futures: dict[int, Future[dict[str, Any]]] = {}
    slot_packets: dict[int, OnlineEvidencePacket] = {}
    shadow_futures: dict[str, Future[dict[str, Any]]] = {}
    shadow_compilations: dict[str, dict[str, Any]] = {}
    locked_lineages: set[str] = set()
    dispatched = 0
    shadows_started = 0
    latest_committed = -1
    accepted_results: list[dict[str, Any]] = []
    terminal_vlm_tickets: set[str] = set()

    def submit_shadow(
        packet: OnlineEvidencePacket,
        compilation: dict[str, Any],
    ) -> None:
        nonlocal shadows_started
        if shadows_started >= args.max_shadow_tickets:
            return
        freeze_root = output_root / "frozen" / (
            f"{packet.ticket_uid}_f{packet.freeze_frame:06d}"
        )
        frozen = freeze_watermarked_view(
            ledger=ledger,
            cutoff_frame=packet.freeze_frame,
            output_root=freeze_root,
        )
        shadow_dir = output_root / "shadow" / packet.ticket_uid
        shadow_futures[packet.ticket_uid] = shadow_executor.submit(
            run_shadow_validation,
            frozen_view=frozen,
            packet=packet,
            compilation=compilation,
            output_dir=shadow_dir,
        )
        shadow_compilations[packet.ticket_uid] = compilation
        shadows_started += 1
        tickets.tickets[packet.ticket_uid].state = "REPLAYING"
        append_jsonl(
            live_log,
            {
                "type": "SHADOW_STARTED",
                "ticket_uid": packet.ticket_uid,
                "freeze_frame": packet.freeze_frame,
                "freeze_sequence": frozen.max_sequence,
            },
        )

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
                        stop_sequence=ledger.max_sequence,
                    )
                if committed_frame % 10 == 0:
                    print(
                        "WATERMARK "
                        f"frame={committed_frame} sequence={ledger.max_sequence} "
                        f"tickets={len(tickets.tickets)} dispatched={dispatched} "
                        f"shadow={shadows_started}",
                        flush=True,
                    )

            if committed and args.reuse_experiment_root:
                # A completed ledger arrives as one batch; ranking every historical
                # frame would repeat the same full causal closure hundreds of times.
                tickets.refresh(
                    ledger=ledger,
                    tracker=tracker,
                    task_context=task_context,
                    stop_sequence=ledger.max_sequence,
                )

            for slot, future in list(slot_futures.items()):
                if not future.done():
                    continue
                packet = slot_packets.pop(slot)
                del slot_futures[slot]
                ticket = tickets.tickets[packet.ticket_uid]
                try:
                    raw_response = future.result()
                    response = _safe_response(raw_response)
                    response_path = output_root / "vlm" / packet.ticket_uid / "response.json"
                    write_json(response_path, response)
                    compilation = compile_vlm_response(
                        packet=packet,
                        response=response,
                        ledger=ledger,
                    )
                    write_json(
                        output_root / "vlm" / packet.ticket_uid / "compilation.json",
                        compilation,
                    )
                    ticket.attempts.append(
                        {
                            "stage": "VLM",
                            "status": "PASS",
                            "response_path": str(response_path),
                            "compilation_stage": compilation.get("stage"),
                        }
                    )
                    if compilation.get("candidate_constraint"):
                        submit_shadow(packet, compilation)
                    else:
                        ticket.state = "ABORTED"
                        terminal_vlm_tickets.add(ticket.ticket_uid)
                        locked_lineages.difference_update(ticket.primary_lineage_uids)
                    append_jsonl(
                        live_log,
                        {
                            "type": "VLM_COMPLETED",
                            "ticket_uid": packet.ticket_uid,
                            "slot": slot,
                            "compilation_stage": compilation.get("stage"),
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
                        },
                    )

            for ticket_uid, future in list(shadow_futures.items()):
                if not future.done():
                    continue
                del shadow_futures[ticket_uid]
                ticket = tickets.tickets[ticket_uid]
                try:
                    result = future.result()
                    ticket.state = (
                        "READY_TO_COMMIT"
                        if result.get("decision") == "WOULD_COMMIT"
                        else "ABORTED"
                    )
                    ticket.attempts.append(
                        {
                            "stage": "SHADOW",
                            "status": result.get("decision"),
                            "result_path": str(
                                output_root / "shadow" / ticket_uid / "shadow_result.json"
                            ),
                        }
                    )
                    if result.get("decision") == "WOULD_COMMIT":
                        accepted_results.append(
                            {
                                "ticket_uid": ticket_uid,
                                "result": result,
                                "compilation": shadow_compilations[ticket_uid],
                            }
                        )
                    append_jsonl(
                        live_log,
                        {
                            "type": "SHADOW_COMPLETED",
                            "ticket_uid": ticket_uid,
                            "decision": result.get("decision"),
                            "reason": result.get("reason"),
                        },
                    )
                except Exception as exc:
                    ticket.state = "ABORTED"
                    ticket.attempts.append(
                        {
                            "stage": "SHADOW",
                            "status": "ERROR",
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
                    append_jsonl(
                        live_log,
                        {
                            "type": "SHADOW_FAILED",
                            "ticket_uid": ticket_uid,
                            "error": f"{type(exc).__name__}:{exc}",
                        },
                    )
                terminal_vlm_tickets.add(ticket_uid)
                locked_lineages.difference_update(ticket.primary_lineage_uids)

            if not args.no_vlm and dispatched < args.max_vlm_tickets:
                free_slots = [index for index in range(len(clients)) if index not in slot_futures]
                for slot in free_slots:
                    if dispatched >= args.max_vlm_tickets:
                        break
                    selected = None
                    for ticket in tickets.ordered(current_frame=max(0, latest_committed)):
                        if ticket.ticket_uid in terminal_vlm_tickets:
                            continue
                        if ticket.dispatch_frame is not None:
                            continue
                        if max(0, latest_committed) - ticket.first_seen_frame < args.min_ticket_age_frames:
                            continue
                        if locked_lineages.intersection(ticket.primary_lineage_uids):
                            continue
                        packet = router.build(
                            ticket=ticket,
                            ledger=ledger,
                            freeze_frame=max(0, latest_committed),
                            freeze_sequence=ledger.max_sequence,
                            output_dir=output_root / "vlm" / ticket.ticket_uid / "evidence",
                        )
                        if packet is None:
                            continue
                        selected = (ticket, packet)
                        break
                    if selected is None:
                        break
                    ticket, packet = selected
                    ticket.state = "DIAGNOSING"
                    ticket.dispatch_frame = packet.freeze_frame
                    ticket.dispatch_sequence = packet.freeze_sequence
                    locked_lineages.update(ticket.primary_lineage_uids)
                    slot_packets[slot] = packet
                    slot_futures[slot] = vlm_executor.submit(
                        clients[slot].complete, packet.evidence
                    )
                    dispatched += 1
                    append_jsonl(
                        live_log,
                        {
                            "type": "VLM_STARTED",
                            "ticket_uid": ticket.ticket_uid,
                            "slot": slot,
                            "freeze_frame": packet.freeze_frame,
                            "priority": ticket.as_dict(max(0, latest_committed)).get(
                                "priority_tuple"
                            ),
                        },
                    )

            active_work = bool(slot_futures or shadow_futures)
            if mapping_done:
                if mapping_process is not None and mapping_process.returncode not in (0, None):
                    break
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
        shadow_executor.shutdown(wait=True, cancel_futures=False)
        if mapping_handle is not None:
            mapping_handle.close()

    mapping_returncode = 0 if mapping_process is None else int(mapping_process.returncode or 0)
    ledger.poll(mapping_done=True)
    tickets.refresh(
        ledger=ledger,
        tracker=tracker,
        task_context=task_context,
        stop_sequence=ledger.max_sequence,
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

    selected_constraints: list[dict[str, Any]] = []
    used_observations: set[str] = set()
    used_lineages: set[str] = set()
    for row in accepted_results:
        ticket = tickets.tickets[row["ticket_uid"]]
        constraint = row["compilation"].get("candidate_constraint")
        if not isinstance(constraint, dict):
            continue
        observation_uid = str(constraint.get("obs_uid") or "")
        if observation_uid in used_observations:
            continue
        if used_lineages.intersection(ticket.primary_lineage_uids):
            continue
        selected_constraints.append(constraint)
        used_observations.add(observation_uid)
        used_lineages.update(ticket.primary_lineage_uids)

    final_comparison = run_final_combined_replay(
        experiment_root=experiment_root,
        constraints=selected_constraints,
        output_dir=output_root / "final",
    )
    summary = {
        "schema_version": "0.1.0",
        "status": "COMPLETED",
        "experiment_root": str(experiment_root),
        "processed_committed_frame_count": len(ledger._committed_frames),
        "final_frame": max(ledger._committed_frames, default=-1),
        "final_event_sequence": ledger.max_sequence,
        "ticket_count": len(tickets.tickets),
        "vlm_dispatched": dispatched,
        "shadow_started": shadows_started,
        "shadow_would_commit_count": len(accepted_results),
        "combined_constraint_count": len(selected_constraints),
        "activated_repaired_version": final_comparison[
            "activated_repaired_version"
        ],
        "baseline_metrics": final_comparison["baseline_metrics"],
        "candidate_metrics": final_comparison["candidate_metrics"],
        "metric_delta": final_comparison["metric_delta"],
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
