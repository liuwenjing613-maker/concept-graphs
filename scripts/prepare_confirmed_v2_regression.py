#!/usr/bin/env python3
"""Rebuild V2 evidence for a confirmed legacy suite at each original watermark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from conceptgraph.revision.online_mvp import (
    EvidenceRouter,
    LiveDependencyTracker,
    LiveEvidenceLedger,
    OnlineScanner,
    TaskContext,
    TicketStore,
)
from scripts.validate_unified_vlm_v1 import write_json
from scripts.validate_unified_vlm_v2 import (
    FrozenRun,
    PreflightDefer,
    prepare_case,
    write_case_html,
    write_root_html,
)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _find_ticket(
    tickets: TicketStore,
    *,
    legacy_ticket_uid: str,
    issue_uid: str,
    anchor_event_uid: str,
):
    direct = tickets.tickets.get(legacy_ticket_uid)
    if direct is not None:
        return direct
    matches = [
        ticket
        for ticket in tickets.tickets.values()
        if any(
            issue.issue_uid == issue_uid
            or issue.anchor_event_uid == anchor_event_uid
            for issue in ticket.issues
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _build_packet(
    *,
    source_root: Path,
    output_subdir: str,
    legacy_manifest: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    freeze_frame = int(legacy_manifest["freeze_frame"])
    freeze_sequence = int(legacy_manifest["freeze_sequence"])
    ledger = LiveEvidenceLedger(source_root)
    committed = ledger.poll(mapping_done=True)
    scanner = OnlineScanner()
    tickets = TicketStore()
    for frame in sorted(frame for frame in committed if frame <= freeze_frame):
        for issue in scanner.scan_frame(frame, ledger):
            tickets.upsert(issue)
    tickets.refresh(
        ledger=ledger,
        tracker=LiveDependencyTracker(),
        task_context=TaskContext(),
        stop_sequence=freeze_sequence,
        cutoff_frame=freeze_frame,
    )
    ticket = _find_ticket(
        tickets,
        legacy_ticket_uid=str(legacy_manifest["ticket_uid"]),
        issue_uid=str(legacy_manifest["issue_uid"]),
        anchor_event_uid=str(legacy_manifest["anchor_event_uid"]),
    )
    if ticket is None:
        raise RuntimeError("legacy issue did not map uniquely into the V2 object pool")
    online_root = source_root / output_subdir
    online_root.mkdir(parents=True, exist_ok=True)
    # FrozenRun indexes online_events.jsonl unconditionally.  V2 packets carry
    # their own cutoff-valid issue, so this isolated replay root intentionally
    # has an empty event stream instead of copying legacy V1 decisions.
    (online_root / "online_events.jsonl").touch(exist_ok=True)
    packet_dir = online_root / "vlm" / ticket.ticket_uid / "evidence"
    packet = EvidenceRouter(source_root, max_images=6).build_v2(
        ticket=ticket,
        ledger=ledger,
        freeze_frame=freeze_frame,
        freeze_sequence=freeze_sequence,
        output_dir=packet_dir,
    )
    if packet is None:
        raise RuntimeError(
            f"V2 router did not emit a packet; resolution={ticket.resolution_state}"
        )
    return ticket.ticket_uid, {
        "legacy_ticket_uid": legacy_manifest["ticket_uid"],
        "v2_ticket_uid": ticket.ticket_uid,
        "source_online_experiment": str(source_root),
        "freeze_frame": freeze_frame,
        "freeze_sequence": freeze_sequence,
        "resolution_state": ticket.resolution_state,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--packet-subdir", required=True)
    args = parser.parse_args()
    if Path(args.packet_subdir).name != args.packet_subdir:
        raise ValueError("packet-subdir must be one plain directory name")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_root}")
    args.output_root.mkdir(parents=True)

    suite = _load(args.legacy_root / "run_summary.json")
    requested = [str(uid) for uid in suite.get("tickets_requested") or ()]
    results: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    for legacy_uid in requested:
        legacy_dir = args.legacy_root / legacy_uid
        manifest = _load(legacy_dir / "case_manifest.json")
        source_root = Path(manifest["source_online_experiment"])
        try:
            v2_uid, row = _build_packet(
                source_root=source_root,
                output_subdir=args.packet_subdir,
                legacy_manifest=manifest,
            )
            run = FrozenRun(source_root, online_subdir=args.packet_subdir)
            case = prepare_case(run, v2_uid, args.output_root)
            write_case_html(case.case_dir)
            row["status"] = "PREPARED"
            row["i1_quality_status"] = _load(case.case_dir / "case_manifest.json").get(
                "i1_quality_status"
            )
            results.append(row)
        except (PreflightDefer, RuntimeError, FileNotFoundError, ValueError) as exc:
            results.append(
                {
                    "legacy_ticket_uid": legacy_uid,
                    "status": getattr(exc, "code", "FAILED"),
                    "detail": getattr(exc, "detail", str(exc)),
                }
            )

    # Read prior judgments only after every V2 request artifact is frozen.
    # They are evaluation labels and can never influence evidence routing.
    for legacy_uid in requested:
        old_output = _load(args.legacy_root / legacy_uid / "vlm_output.json")
        expected.append(
            {
                "legacy_ticket_uid": legacy_uid,
                "legacy_selected_candidate": old_output.get("selected_candidate"),
                "v2_expected": (
                    "UNRESOLVED+COMPOUND_STATE_REQUIRED"
                    if old_output.get("selected_candidate") == "H4"
                    else "E1+NOT_EVALUATED"
                ),
            }
        )

    write_json(
        args.output_root / "regression_expectations_posthoc.json",
        {
            "not_in_vlm_request": True,
            "source": "confirmed identity_partition_true5 legacy review",
            "expectations": expected,
        },
    )
    write_json(
        args.output_root / "run_summary.json",
        {
            "schema_version": "ali_my_confirmed_v2_regression/1.0",
            "api_calls_attempted": 0,
            "map_or_repair_mutation": False,
            "legacy_vlm_output_used_in_request": False,
            "requested": len(requested),
            "prepared": sum(row.get("status") == "PREPARED" for row in results),
            "results": results,
        },
    )
    write_root_html(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
