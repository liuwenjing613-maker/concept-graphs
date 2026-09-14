from conceptgraph.revision.models import (
    ConflictType,
    RepairTicket,
    assess_maturity,
    classify_conflict,
)


def test_four_online_conflict_classes():
    assert classify_conflict(
        repair_entities=["o1"], changed_entities=["o9"]
    ) == ConflictType.DISJOINT
    assert classify_conflict(
        repair_entities=["o1"],
        changed_entities=["o1"],
        append_only_entities=["o1"],
    ) == ConflictType.APPEND_ONLY_REBASEABLE
    assert classify_conflict(
        repair_entities=["o1"],
        changed_entities=["o1"],
        lineage_redirects={"o1": "o7"},
    ) == ConflictType.LINEAGE_REBASEABLE
    assert classify_conflict(
        repair_entities=["o1"],
        changed_entities=["o1"],
        removed_evidence_refs=["obs7"],
        hypothesis_evidence_refs=["obs7"],
    ) == ConflictType.HYPOTHESIS_INVALIDATED


def test_ticket_stops_rebase_loop():
    ticket = RepairTicket("t1", "l1", state="REPLAYING", max_rebase_count=1)
    ticket.transition("REBASING", reason="new tail")
    assert ticket.state == "REBASING"
    ticket.transition("REPLAYING", reason="tail replay")
    ticket.transition("REBASING", reason="new tail again")
    assert ticket.state == "WAIT_STABILITY"


def test_maturity_preserves_raw_signals_and_defers_weak_groups():
    weak = assess_maturity(
        ["run_f000001_r0", "run_f000001_r1"],
        action="SPLIT",
        bbox_extent=[1.0, 0.0, 1.0],
    )
    assert not weak["eligible"]
    assert weak["state"] == "TENTATIVE"
    assert "insufficient_unique_frames" in weak["reasons"]
    assert "degenerate_geometry" in weak["reasons"]

    mature = assess_maturity(
        [
            "run_f000001_r0",
            "run_f000002_r0",
            "run_f000003_r0",
            "run_f000004_r0",
        ],
        action="SPLIT",
        bbox_extent=[1.0, 2.0, 3.0],
    )
    assert mature["eligible"]
    assert mature["state"] == "MATURE"
