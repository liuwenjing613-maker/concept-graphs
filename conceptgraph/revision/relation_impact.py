from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


PREDICATE_MAP = {"on top of": "on", "under": "under"}
EdgeKey = tuple[str, str, str]
IndexedTriple = tuple[int, int, str]


def canonical_edge_map(
    edges: Iterable[Mapping[str, Any]],
) -> dict[EdgeKey, int]:
    """Validate and canonicalize an aggregated directed relation graph."""

    result: dict[EdgeKey, int] = {}
    for edge in edges:
        source = str(edge.get("source_entity_uid") or "")
        target = str(edge.get("target_entity_uid") or "")
        relation = str(edge.get("relation") or "")
        support = int(edge.get("num_detections", 0))
        if not source or not target or not relation:
            raise ValueError("relation edge has an empty field")
        if source == target:
            raise ValueError(f"self-loop relation edge: {source}")
        if relation not in PREDICATE_MAP:
            raise ValueError(f"unsupported relation predicate: {relation}")
        if support <= 0:
            raise ValueError("relation support must be positive")
        key = (source, relation, target)
        if key in result:
            raise ValueError(f"duplicate relation edge: {key}")
        result[key] = support
    return result


def edge_diff(
    reference: Mapping[EdgeKey, int],
    candidate: Mapping[EdgeKey, int],
) -> dict[str, Any]:
    """Return set and support changes without conflating the two."""

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    added = sorted(candidate_keys - reference_keys)
    removed = sorted(reference_keys - candidate_keys)
    support_changed = sorted(
        key
        for key in reference_keys & candidate_keys
        if int(reference[key]) != int(candidate[key])
    )
    changed = added + removed + support_changed
    return {
        "exact": not changed,
        "added": [{"edge": list(key), "support": int(candidate[key])} for key in added],
        "removed": [
            {"edge": list(key), "support": int(reference[key])} for key in removed
        ],
        "support_changed": [
            {
                "edge": list(key),
                "reference_support": int(reference[key]),
                "candidate_support": int(candidate[key]),
                "delta": int(candidate[key]) - int(reference[key]),
            }
            for key in support_changed
        ],
        "changed_edge_count": len(changed),
        "changed_entity_uids": sorted(
            {
                endpoint
                for source, _relation, target in changed
                for endpoint in (source, target)
            }
        ),
    }


def indexed_predictions(
    edges: Mapping[EdgeKey, int],
    entity_to_index: Mapping[str, int],
) -> tuple[set[IndexedTriple], list[dict[str, Any]]]:
    """Map UUID-bound edges to the frozen formal-map index namespace."""

    predictions: set[IndexedTriple] = set()
    unmapped = []
    for (source, relation, target), support in sorted(edges.items()):
        if source not in entity_to_index or target not in entity_to_index:
            unmapped.append(
                {
                    "edge": [source, relation, target],
                    "support": int(support),
                    "missing_endpoint_uids": sorted(
                        endpoint
                        for endpoint in (source, target)
                        if endpoint not in entity_to_index
                    ),
                }
            )
            continue
        triple = (
            int(entity_to_index[source]),
            int(entity_to_index[target]),
            PREDICATE_MAP[relation],
        )
        if triple in predictions:
            raise ValueError(f"index mapping collapsed duplicate relation: {triple}")
        predictions.add(triple)
    return predictions, unmapped


def gold_key(row: Mapping[str, Any]) -> IndexedTriple:
    return (
        int(row["source_gt_object_id"]),
        int(row["target_gt_object_id"]),
        str(row["predicate"]),
    )


def gold_metrics(
    rows: Iterable[Mapping[str, Any]],
    predicted: set[IndexedTriple],
) -> dict[str, Any]:
    """Evaluate only explicitly labeled pair-directions."""

    materialized = list(rows)
    counts = Counter()
    for row in materialized:
        label = bool(row["label"])
        present = gold_key(row) in predicted
        counts[
            "tp" if label and present else "fn" if label else "fp" if present else "tn"
        ] += 1
    tp, fp, fn, tn = (int(counts[name]) for name in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "label_count": len(materialized),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / max(1, len(materialized)),
        "precision_within_labeled_pair_scope": precision,
        "recall_within_labeled_pair_scope": recall,
        "specificity_within_labeled_pair_scope": specificity,
        "f1_within_labeled_pair_scope": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def split_stable_gold(
    rows: Iterable[Mapping[str, Any]],
    impacted_indices: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Move identity-changed endpoints out of the comparable gold scope."""

    stable = []
    impacted = []
    for source in rows:
        row = dict(source)
        key = gold_key(row)
        destination = (
            impacted
            if key[0] in impacted_indices or key[1] in impacted_indices
            else stable
        )
        destination.append(row)
    return stable, impacted


def restrict_edges(
    edges: Mapping[EdgeKey, int],
    excluded_entities: set[str],
) -> dict[EdgeKey, int]:
    return {
        key: int(support)
        for key, support in edges.items()
        if key[0] not in excluded_entities and key[2] not in excluded_entities
    }


def membership_changed_entities(
    reference: Mapping[str, Iterable[str]],
    candidate: Mapping[str, Iterable[str]],
) -> set[str]:
    all_entities = set(reference) | set(candidate)
    return {
        str(entity)
        for entity in all_entities
        if set(str(item) for item in reference.get(entity, ()))
        != set(str(item) for item in candidate.get(entity, ()))
    }
