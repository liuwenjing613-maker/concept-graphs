from scripts.build_revision_cmvic_room0_pilot_intake_v0 import (
    QUARTILES,
    stratified_round_robin,
)


def _pool():
    rows = []
    for quartile_index, (quartile, _, _) in enumerate(QUARTILES):
        for item_index in range(3):
            rows.append(
                {
                    "case_uid": f"case_{quartile_index}_{item_index}",
                    "source_incident_uid": f"incident_{quartile_index}_{item_index}",
                    "anchor_quartile": quartile,
                    "review_score": 1000.0 - item_index,
                }
            )
    return rows


def test_stratified_selector_is_seeded_balanced_and_score_blind():
    pool = _pool()
    selected = stratified_round_robin(pool, seed="fixed", limit=8)
    mutated = [dict(row, review_score=-float(row["review_score"])) for row in pool]
    selected_after_score_reversal = stratified_round_robin(
        mutated, seed="fixed", limit=8
    )

    assert [row["case_uid"] for row in selected] == [
        row["case_uid"] for row in selected_after_score_reversal
    ]
    assert {
        quartile: sum(row["anchor_quartile"] == quartile for row in selected)
        for quartile, _, _ in QUARTILES
    } == {quartile: 2 for quartile, _, _ in QUARTILES}
