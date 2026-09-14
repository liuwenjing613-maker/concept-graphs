from __future__ import annotations

import pytest

from conceptgraph.revision.relation_impact import (
    canonical_edge_map,
    edge_diff,
    gold_metrics,
    indexed_predictions,
    membership_changed_entities,
    restrict_edges,
    split_stable_gold,
)


def _edge(source: str, relation: str, target: str, support: int = 1):
    return {
        "source_entity_uid": source,
        "relation": relation,
        "target_entity_uid": target,
        "num_detections": support,
    }


def _gold(source: int, target: int, predicate: str, label: bool):
    return {
        "gold_uid": f"{source}-{target}-{predicate}-{label}",
        "source_gt_object_id": source,
        "target_gt_object_id": target,
        "predicate": predicate,
        "label": label,
    }


def test_edge_diff_distinguishes_support_from_set_change():
    native = canonical_edge_map([_edge("a", "on top of", "b", 13)])
    sparse = canonical_edge_map([_edge("a", "on top of", "b", 12)])

    result = edge_diff(native, sparse)

    assert not result["exact"]
    assert result["added"] == []
    assert result["removed"] == []
    assert result["support_changed"] == [
        {
            "edge": ["a", "on top of", "b"],
            "reference_support": 13,
            "candidate_support": 12,
            "delta": -1,
        }
    ]
    assert result["changed_entity_uids"] == ["a", "b"]


def test_index_mapping_keeps_unknown_endpoints_outside_frozen_namespace():
    edges = canonical_edge_map(
        [
            _edge("a", "on top of", "b", 2),
            _edge("new", "under", "a", 1),
        ]
    )

    predictions, unmapped = indexed_predictions(edges, {"a": 4, "b": 9})

    assert predictions == {(4, 9, "on")}
    assert len(unmapped) == 1
    assert unmapped[0]["missing_endpoint_uids"] == ["new"]


def test_gold_metrics_are_direction_and_label_scoped():
    rows = [
        _gold(1, 2, "on", True),
        _gold(2, 1, "under", True),
        _gold(2, 1, "on", False),
        _gold(1, 2, "under", False),
    ]

    metrics = gold_metrics(rows, {(1, 2, "on"), (2, 1, "on")})

    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["balanced_accuracy"] == 0.5


def test_identity_changed_endpoints_are_removed_from_comparable_gold():
    rows = [
        _gold(1, 2, "on", True),
        _gold(2, 1, "under", True),
        _gold(3, 4, "on", False),
    ]

    stable, impacted = split_stable_gold(rows, {2})

    assert [row["gold_uid"] for row in stable] == ["3-4-on-False"]
    assert len(impacted) == 2


def test_membership_change_and_outside_relation_scope():
    native_membership = {"a": ["o1", "o2"], "b": ["o3"], "c": ["o4"]}
    sparse_membership = {
        "a": ["o1"],
        "b": ["o3"],
        "c": ["o4"],
        "new": ["o2"],
    }
    edges = canonical_edge_map(
        [
            _edge("a", "on top of", "b"),
            _edge("b", "under", "c"),
            _edge("new", "under", "a"),
        ]
    )

    changed = membership_changed_entities(native_membership, sparse_membership)
    outside = restrict_edges(edges, changed)

    assert changed == {"a", "new"}
    assert outside == {("b", "under", "c"): 1}


@pytest.mark.parametrize(
    "edge",
    [
        _edge("a", "near", "b"),
        _edge("a", "on top of", "a"),
        _edge("a", "on top of", "b", 0),
    ],
)
def test_invalid_aggregated_edges_fail_closed(edge):
    with pytest.raises(ValueError):
        canonical_edge_map([edge])
