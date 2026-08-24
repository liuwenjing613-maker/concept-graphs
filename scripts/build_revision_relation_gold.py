#!/usr/bin/env python3
"""Freeze and evaluate a small direction-aware relation gold set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triple(value: Iterable[Any]) -> tuple[int, int, str]:
    source, target, predicate = value
    return int(source), int(target), str(predicate)


def _gold_uid(source: int, target: int, predicate: str, label: bool) -> str:
    payload = f"{source}|{predicate}|{target}|{int(label)}"
    return "relation_gold_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    source = _read(args.source_evaluation)
    mapped_gt = {
        _triple(row)
        for row in (source.get("audit") or {}).get("mapped_gt_triples") or ()
    }
    explicit_on = sorted(row for row in mapped_gt if row[2] == "on")
    if not explicit_on:
        raise ValueError("no mapped explicit ReplicaSSG 'on' relations")
    rows = []
    for upper, support, _ in explicit_on:
        if upper == support:
            raise ValueError("self relation cannot define directional gold")
        labels = [
            (upper, support, "on", True, "REPLICASSG_EXPLICIT"),
            (support, upper, "under", True, "INVERSE_OF_EXPLICIT_ON"),
            (support, upper, "on", False, "DIRECTIONAL_ANTISYMMETRY"),
            (upper, support, "under", False, "DIRECTIONAL_ANTISYMMETRY"),
        ]
        for source_id, target_id, predicate, label, derivation in labels:
            rows.append(
                {
                    "gold_uid": _gold_uid(source_id, target_id, predicate, label),
                    "source_gt_object_id": source_id,
                    "target_gt_object_id": target_id,
                    "predicate": predicate,
                    "label": label,
                    "derivation": derivation,
                    "source_explicit_triple": [upper, support, "on"],
                }
            )
    keys = [
        (
            row["source_gt_object_id"],
            row["target_gt_object_id"],
            row["predicate"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("logical closure produced duplicate/conflicting gold keys")
    manifest = {
        "schema_version": "1.0.0",
        "gold_uid": "room0_mapped_on_direction_gold_20260824",
        "scene_id": "room0",
        "frozen_before_scoped_prediction_evaluation": True,
        "selection_uses_prediction_presence": False,
        "selection_rule": (
            "all mapped ReplicaSSG explicit 'on' triples; no support threshold "
            "or prediction-dependent filtering"
        ),
        "semantic_contract": {
            "on_inverse": "under",
            "distinct_rigid_objects": True,
            "directional_antisymmetry": True,
            "negative_labels_are_pair_local": True,
        },
        "scope_limit": (
            "Only labels the four on/under directions for each explicit mapped "
            "support pair. Predictions on all other pairs remain UNKNOWN and are "
            "excluded from precision claims."
        ),
        "source_artifact": {
            "path": str(Path(args.source_evaluation).resolve()),
            "sha256": _sha256(args.source_evaluation),
            "mapped_gt_relation_count": len(mapped_gt),
            "mapped_explicit_on_count": len(explicit_on),
        },
        "label_count": len(rows),
        "positive_count": sum(bool(row["label"]) for row in rows),
        "negative_count": sum(not bool(row["label"]) for row in rows),
        "relations": rows,
    }
    _write(args.output, manifest)
    return manifest


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        (
            "TP"
            if row["label"] and row["predicted"]
            else "FN"
            if row["label"]
            else "FP"
            if row["predicted"]
            else "TN"
        )
        for row in rows
    )
    tp, fp, fn, tn = (counts[name] for name in ("TP", "FP", "FN", "TN"))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / max(1, len(rows)),
        "precision_within_labeled_pair_scope": precision,
        "recall_within_labeled_pair_scope": recall,
        "specificity_within_labeled_pair_scope": specificity,
        "f1_within_labeled_pair_scope": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read(args.gold)
    expected_sha = str(manifest["source_artifact"]["sha256"])
    actual_sha = _sha256(args.source_evaluation)
    if actual_sha != expected_sha:
        raise ValueError("relation evaluation source hash changed after freeze")
    source = _read(args.source_evaluation)
    predicted = {
        _triple(row)
        for row in (source.get("audit") or {}).get("predicted_triples") or ()
    }
    rows = []
    for item in manifest["relations"]:
        key = (
            int(item["source_gt_object_id"]),
            int(item["target_gt_object_id"]),
            str(item["predicate"]),
        )
        rows.append({**item, "predicted": key in predicted})
    result = {
        "schema_version": "1.0.0",
        "evaluation_role": "SMALL_DIRECTION_AWARE_RELATION_GOLD",
        "gold_uid": manifest["gold_uid"],
        "pass": True,
        "source_hash_exact": True,
        "label_count": len(rows),
        "metrics": _metrics(rows),
        "by_derivation": {
            derivation: _metrics(
                [row for row in rows if row["derivation"] == derivation]
            )
            for derivation in sorted({row["derivation"] for row in rows})
        },
        "rows": rows,
        "unknown_prediction_count_outside_gold_scope": len(
            predicted
            - {
                (
                    int(row["source_gt_object_id"]),
                    int(row["target_gt_object_id"]),
                    str(row["predicate"]),
                )
                for row in rows
            }
        ),
        "precision_is_scope_restricted": True,
        "population_claim_allowed": False,
    }
    _write(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "evaluate"), required=True)
    parser.add_argument("--source-evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gold", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze(args)
        summary = {
            "status": "PASS",
            "gold_uid": result["gold_uid"],
            "label_count": result["label_count"],
            "positive_count": result["positive_count"],
            "negative_count": result["negative_count"],
        }
    else:
        if args.gold is None:
            parser.error("evaluate requires --gold")
        result = evaluate(args)
        summary = {
            "status": "PASS",
            "gold_uid": result["gold_uid"],
            "metrics": result["metrics"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
