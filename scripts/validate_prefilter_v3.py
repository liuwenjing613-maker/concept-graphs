#!/usr/bin/env python3
"""Replay a completed online evidence ledger through the V3 prefilter."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from conceptgraph.revision.online_mvp import (
    POOL_AUDIT,
    POOL_MAIN,
    POOL_UNREVIEWABLE,
    EvidenceRouter,
    LiveDependencyTracker,
    LiveEvidenceLedger,
    OnlineScanner,
    TaskContext,
    TicketStore,
)


TERMINAL_STATES = {"NO_ACTION", "DIAGNOSED", "ABORTED", "REPAIRED"}
ALL_LOCATIONS = (POOL_MAIN, POOL_AUDIT, POOL_UNREVIEWABLE, "IN_FLIGHT", "COMPLETED")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _location(ticket: Any) -> str:
    if ticket.state == "DIAGNOSING":
        return "IN_FLIGHT"
    if ticket.state in TERMINAL_STATES:
        return "COMPLETED"
    return str(ticket.pool_location)


def _slim_ticket(ticket: Any, current_frame: int) -> dict[str, Any]:
    issue = TicketStore.select_review_issue(ticket)
    context = ticket.review_context
    return {
        "ticket_uid": ticket.ticket_uid,
        "review_issue_uid": ticket.review_issue_uid,
        "review_family": issue.family if issue else None,
        "review_strength": issue.strength if issue else None,
        "review_frame": issue.detected_frame if issue else None,
        "routing_state": ticket.routing_state,
        "routing_reason": ticket.routing_reason,
        "proposed_destination": ticket.routing_destination,
        "actual_location": _location(ticket),
        "routing_mode": ticket.routing_mode,
        "latest_reconfirmed": ticket.latest_reconfirmed,
        "relevant_update_count": ticket.relevant_update_count,
        "stable_changed_count": ticket.stable_changed_count,
        "event_signature": ticket.event_signature,
        "latest_signature": (
            ticket.state_history[-1].get("signature") if ticket.state_history else None
        ),
        "review_context": dict(context) if context else None,
        "priority": list(ticket.priority_key(current_frame)),
        "issue_count": len(ticket.issues),
        "first_seen_frame": ticket.first_seen_frame,
        "last_seen_frame": ticket.last_seen_frame,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routing-mode", choices=("shadow", "active"), default="shadow")
    parser.add_argument(
        "--packet-root",
        type=Path,
        help="Optional fresh online-style root where VLM packet manifests are written.",
    )
    parser.add_argument(
        "--ticket",
        action="append",
        dest="selected_tickets",
        help="Ticket to build; repeat for a fixed regression set. Defaults to all tickets.",
    )
    args = parser.parse_args()

    ledger = LiveEvidenceLedger(args.source_root)
    committed = sorted(ledger.poll(mapping_done=True))
    if not committed:
        raise RuntimeError("no committed online frames found")

    scanner = OnlineScanner()
    tracker = LiveDependencyTracker()
    tickets = TicketStore()
    raw_issue_uids: list[str] = []
    timeline: list[dict[str, Any]] = []

    for frame in committed:
        issues = scanner.scan_frame(frame, ledger)
        raw_issue_uids.extend(issue.issue_uid for issue in issues)
        for issue in issues:
            tickets.upsert(issue)
        sequence = ledger.max_sequence_at_frame(frame)
        tickets.refresh(
            ledger=ledger,
            tracker=tracker,
            task_context=TaskContext(),
            stop_sequence=sequence,
            cutoff_frame=frame,
            routing_mode=args.routing_mode,
        )
        state_counts = Counter(ticket.routing_state for ticket in tickets.tickets.values())
        proposed_counts = Counter(
            ticket.routing_destination for ticket in tickets.tickets.values()
        )
        location_counts = Counter(_location(ticket) for ticket in tickets.tickets.values())
        invalid_locations = sorted(set(location_counts) - set(ALL_LOCATIONS))
        conservation_total = sum(location_counts[name] for name in ALL_LOCATIONS)
        timeline.append(
            {
                "frame": frame,
                "sequence": sequence,
                "issues_added": len(issues),
                "raw_issues_cumulative": len(raw_issue_uids),
                "unique_issue_uids_cumulative": len(set(raw_issue_uids)),
                "unique_tickets": len(tickets.tickets),
                "routing_states": dict(sorted(state_counts.items())),
                "proposed_destinations": dict(sorted(proposed_counts.items())),
                "actual_locations": {
                    name: int(location_counts.get(name, 0)) for name in ALL_LOCATIONS
                },
                "conservation": {
                    "counted": conservation_total,
                    "expected": len(tickets.tickets),
                    "pass": conservation_total == len(tickets.tickets)
                    and not invalid_locations,
                    "invalid_locations": invalid_locations,
                },
            }
        )

    final_frame = committed[-1]
    ticket_rows = [
        _slim_ticket(ticket, final_frame)
        for ticket in sorted(
            tickets.tickets.values(), key=lambda item: item.priority_key(final_frame)
        )
    ]
    final_locations = Counter(row["actual_location"] for row in ticket_rows)
    final_states = Counter(row["routing_state"] for row in ticket_rows)
    final_proposed = Counter(row["proposed_destination"] for row in ticket_rows)
    stored_issue_uids = [
        issue.issue_uid for ticket in tickets.tickets.values() for issue in ticket.issues
    ]
    packet_build = None
    if args.packet_root is not None:
        if args.packet_root.exists():
            raise FileExistsError(f"packet root already exists: {args.packet_root}")
        args.packet_root.mkdir(parents=True)
        router = EvidenceRouter(args.source_root)
        requested = list(args.selected_tickets or (row["ticket_uid"] for row in ticket_rows))
        packet_rows = []
        for ticket_uid in requested:
            ticket = tickets.tickets.get(ticket_uid)
            if ticket is None:
                packet_rows.append(
                    {"ticket_uid": ticket_uid, "status": "TICKET_NOT_FOUND"}
                )
                continue
            packet = router.build_v2(
                ticket=ticket,
                ledger=ledger,
                freeze_frame=final_frame,
                freeze_sequence=ledger.max_sequence_at_frame(final_frame),
                output_dir=args.packet_root / "vlm" / ticket_uid / "evidence",
            )
            packet_rows.append(
                {
                    "ticket_uid": ticket_uid,
                    "status": "PACKET_BUILT" if packet is not None else "PACKET_UNREVIEWABLE",
                    "review_issue_uid": ticket.review_issue_uid,
                    "routing_state": ticket.routing_state,
                    "proposed_destination": ticket.routing_destination,
                    "actual_location": _location(ticket),
                    "current_assignment": (
                        packet.packet_manifest.get("current_assignment") if packet else None
                    ),
                    "aliases": (
                        sorted(packet.packet_manifest.get("alias_version_uids") or {})
                        if packet else []
                    ),
                }
            )
        packet_build = {
            "packet_root": str(args.packet_root),
            "requested": len(requested),
            "built": sum(row["status"] == "PACKET_BUILT" for row in packet_rows),
            "unreviewable": sum(
                row["status"] == "PACKET_UNREVIEWABLE" for row in packet_rows
            ),
            "not_found": sum(row["status"] == "TICKET_NOT_FOUND" for row in packet_rows),
            "rows": packet_rows,
        }
    result = {
        "schema_version": "candidate_prefilter_v3_shadow_replay",
        "source_root": str(args.source_root),
        "routing_mode": args.routing_mode,
        "causal_replay": {
            "fresh_online_mapping": False,
            "future_graph_used_for_decision": False,
            "method": "frame-by-frame scanner and cutoff-bound state resolver",
            "first_frame": committed[0],
            "last_frame": final_frame,
            "committed_frame_count": len(committed),
        },
        "summary": {
            "raw_issue_events": len(raw_issue_uids),
            "unique_issue_uids": len(set(raw_issue_uids)),
            "stored_issue_uids": len(set(stored_issue_uids)),
            "unique_tickets": len(ticket_rows),
            "routing_states": dict(sorted(final_states.items())),
            "proposed_destinations": dict(sorted(final_proposed.items())),
            "actual_locations": {
                name: int(final_locations.get(name, 0)) for name in ALL_LOCATIONS
            },
            "issue_conservation_pass": set(raw_issue_uids) == set(stored_issue_uids),
            "ticket_conservation_pass": all(row["conservation"]["pass"] for row in timeline),
            "shadow_keeps_audit_in_main": args.routing_mode == "shadow",
        },
        "timeline": timeline,
        "tickets": ticket_rows,
        "packet_build": packet_build,
    }
    _write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
