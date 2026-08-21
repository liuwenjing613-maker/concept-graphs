from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evidence import sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _confusion(rows: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    labels = sorted({label for pair in rows for label in pair})
    table = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for truth, prediction in rows:
        table[truth][prediction] += 1
    return table


def evaluate(run_root: Path, labels_path: Path) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_used_for_inference") is not False:
        raise ValueError("run manifest does not prove label-blind inference")
    labels = _load_jsonl(labels_path)
    truth_by_case = {
        str(row.get("case_uid") or row.get("incident_uid")): row for row in labels
    }
    results: dict[str, dict[str, Any]] = {}
    for path in sorted((run_root / "cases").glob("*/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("labels_used_for_inference") is not False:
            raise ValueError(f"result is not label blind: {path}")
        results[str(row["case_uid"])] = row

    joined = [(truth_by_case[key], results[key]) for key in sorted(truth_by_case.keys() & results.keys())]
    state_pairs: list[tuple[str, str]] = []
    type_pairs: list[tuple[str, str]] = []
    binary_tp = binary_fp = binary_fn = binary_tn = 0
    brier_values: list[float] = []
    scene_rows: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for truth, result in joined:
        prediction = result["diagnosis"]
        truth_state = str(truth["final_state"])
        pred_state = str(prediction["final_state"])
        state_pairs.append((truth_state, pred_state))
        scene_rows[str(truth["scene_id"])].append((truth, result))
        if truth_state != "UNCLEAR":
            truth_wrong = truth_state == "WRONG"
            pred_wrong = pred_state == "WRONG"
            if truth_wrong and pred_wrong:
                binary_tp += 1
            elif not truth_wrong and pred_wrong:
                binary_fp += 1
            elif truth_wrong and not pred_wrong:
                binary_fn += 1
            else:
                binary_tn += 1
            probability_wrong = (
                float(prediction["confidence"])
                if pred_wrong
                else (1.0 - float(prediction["confidence"]) if pred_state == "CORRECT" else 0.5)
            )
            brier_values.append((probability_wrong - float(truth_wrong)) ** 2)
        if truth_state == "WRONG" and pred_state == "WRONG":
            type_pairs.append((str(truth["final_error_type"]), str(prediction["error_type"])))

    precision = _safe_div(binary_tp, binary_tp + binary_fp)
    recall = _safe_div(binary_tp, binary_tp + binary_fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    execution_counts = Counter(
        str(result["execution"]["status"]) for _, result in joined
    )
    action_counts = Counter(
        str(result["diagnosis"]["repair"]["action"]) for _, result in joined
    )
    per_scene = {}
    for scene_id, rows in sorted(scene_rows.items()):
        exact = sum(
            truth["final_state"] == result["diagnosis"]["final_state"]
            for truth, result in rows
        )
        per_scene[scene_id] = {
            "count": len(rows),
            "state_exact": exact,
            "state_accuracy": _safe_div(exact, len(rows)),
        }
    return {
        "schema_version": "1.0.0",
        "created_at": _utc_now(),
        "run_root": str(run_root),
        "run_manifest_sha256": sha256_file(manifest_path),
        "labels_path": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "labels_used_only_after_inference": True,
        "truth_case_count": len(truth_by_case),
        "result_case_count": len(results),
        "matched_case_count": len(joined),
        "state_confusion": _confusion(state_pairs),
        "state_exact": sum(truth == pred for truth, pred in state_pairs),
        "state_accuracy": _safe_div(sum(truth == pred for truth, pred in state_pairs), len(state_pairs)),
        "wrong_detection": {
            "tp": binary_tp,
            "fp": binary_fp,
            "fn": binary_fn,
            "tn": binary_tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier_score": _safe_div(sum(brier_values), len(brier_values)),
        },
        "error_type_confusion_when_both_wrong": _confusion(type_pairs),
        "error_type_exact_when_both_wrong": sum(t == p for t, p in type_pairs),
        "error_type_accuracy_when_both_wrong": _safe_div(
            sum(t == p for t, p in type_pairs), len(type_pairs)
        ),
        "execution_status_counts": dict(sorted(execution_counts.items())),
        "repair_action_counts": dict(sorted(action_counts.items())),
        "per_scene": per_scene,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a completed label-blind VLM run.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args.run_root.resolve(), args.labels.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"matched": result["matched_case_count"], "state_accuracy": result["state_accuracy"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
