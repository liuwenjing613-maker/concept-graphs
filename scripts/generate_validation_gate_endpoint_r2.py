#!/usr/bin/env python3
"""Build a small, R1-answer-blind endpoint repeat-review subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "2.1.0"
FINAL_STATES = {"CORRECT", "WRONG", "UNCLEAR"}
ERROR_TYPES = {
    "NOT_APPLICABLE",
    "FALSE_MERGE",
    "FALSE_SPLIT",
    "SPURIOUS_OBJECT",
    "MISSING_OBJECT",
    "WRONG_MEMBERSHIP",
    "GEOMETRY_CORRUPTION",
    "SEMANTIC_IDENTITY_ERROR",
    "OTHER",
}
LABEL_FIELDS = {
    "reviewer_id",
    "evidence_sufficient",
    "final_state",
    "final_error_type",
    "review_seconds",
    "notes",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    value = str(row.get("scene_id") or ""), str(
        row.get("incident_uid") or row.get("case_uid") or ""
    )
    if not all(value):
        raise ValueError("row needs scene_id and incident_uid")
    return value


def index_unique(rows: Iterable[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        key = case_key(row)
        if key in result:
            raise ValueError(f"duplicate {name} key: {key}")
        result[key] = row
    return result


def validate_r1_label(row: dict[str, Any], key: tuple[str, str]) -> None:
    if row.get("reviewer_id") != "R1":
        raise ValueError(f"R1 freeze contains a non-R1 row: {key}")
    evidence = row.get("evidence_sufficient")
    state = row.get("final_state")
    error_type = row.get("final_error_type")
    if evidence not in {"YES", "NO"} or state not in FINAL_STATES or error_type not in ERROR_TYPES:
        raise ValueError(f"invalid R1 endpoint label: {key}")
    if evidence == "NO" and state != "UNCLEAR":
        raise ValueError(f"R1 evidence NO must map to UNCLEAR: {key}")
    if evidence == "YES" and state == "UNCLEAR":
        raise ValueError(f"R1 evidence YES cannot map to UNCLEAR: {key}")
    if (state == "WRONG") == (error_type == "NOT_APPLICABLE"):
        raise ValueError(f"R1 final state and error type conflict: {key}")
    if error_type == "OTHER" and not str(row.get("notes") or "").strip():
        raise ValueError(f"R1 OTHER needs notes: {key}")


def stable_key(row: dict[str, Any], seed: int, namespace: str = "select") -> tuple[str, str, str]:
    scene, uid = case_key(row)
    digest = hashlib.sha256(f"{seed}:{namespace}:{scene}:{uid}".encode()).hexdigest()
    return digest, scene, uid


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if total < 0 or total > population:
        raise ValueError(f"cannot allocate {total} from population {population}")
    if total == 0:
        return {name: 0 for name in counts}
    ideals = {name: total * count / population for name, count in counts.items()}
    result = {name: min(counts[name], math.floor(ideals[name])) for name in counts}
    order = sorted(counts, key=lambda name: (-(ideals[name] - result[name]), name))
    while sum(result.values()) < total:
        changed = False
        for name in order:
            if result[name] < counts[name]:
                result[name] += 1
                changed = True
                if sum(result.values()) == total:
                    break
        if not changed:
            raise ValueError("allocation exhausted before reaching target")
    return result


def state_targets(labels: list[dict[str, Any]], size: int) -> dict[str, int]:
    counts = Counter(str(row["final_state"]) for row in labels)
    unclear = min(counts["UNCLEAR"], 2, size)
    remaining = size - unclear
    observed_wrong_types = {
        str(row["final_error_type"])
        for row in labels
        if row["final_state"] == "WRONG"
    }
    proportional_wrong = round(size * counts["WRONG"] / len(labels))
    wrong = max(proportional_wrong, min(len(observed_wrong_types), remaining))
    wrong = min(wrong, counts["WRONG"], remaining)
    correct = size - unclear - wrong
    if correct > counts["CORRECT"]:
        transfer = correct - counts["CORRECT"]
        correct -= transfer
        wrong += transfer
    if correct < 0 or wrong > counts["WRONG"]:
        raise ValueError("requested R2 size cannot satisfy endpoint-state targets")
    return {"CORRECT": correct, "WRONG": wrong, "UNCLEAR": unclear}


def state_scene_targets(
    labels: list[dict[str, Any]], targets: dict[str, int], scene_targets: dict[str, int]
) -> dict[str, dict[str, int]]:
    scenes = sorted(scene_targets)
    matrix: dict[str, dict[str, int]] = {}
    ideals: dict[str, dict[str, float]] = {}
    for state, target in targets.items():
        counts = Counter(str(row["scene_id"]) for row in labels if row["final_state"] == state)
        for scene in scenes:
            counts.setdefault(scene, 0)
        matrix[state] = largest_remainder(dict(counts), target)
        denominator = sum(counts.values())
        ideals[state] = {
            scene: (target * counts[scene] / denominator if denominator else 0.0)
            for scene in scenes
        }

    def totals() -> Counter[str]:
        return Counter({scene: sum(matrix[state][scene] for state in matrix) for scene in scenes})

    available = Counter((str(row["final_state"]), str(row["scene_id"])) for row in labels)
    while totals() != Counter(scene_targets):
        current = totals()
        over = [scene for scene in scenes if current[scene] > scene_targets[scene]]
        under = [scene for scene in scenes if current[scene] < scene_targets[scene]]
        candidates = []
        for source in over:
            for destination in under:
                for state in matrix:
                    if matrix[state][source] <= 0:
                        continue
                    if matrix[state][destination] >= available[(state, destination)]:
                        continue
                    before = abs(matrix[state][source] - ideals[state][source]) + abs(
                        matrix[state][destination] - ideals[state][destination]
                    )
                    after = abs(matrix[state][source] - 1 - ideals[state][source]) + abs(
                        matrix[state][destination] + 1 - ideals[state][destination]
                    )
                    candidates.append((after - before, state, source, destination))
        if not candidates:
            raise ValueError("cannot reconcile state and scene quotas")
        _, state, source, destination = min(candidates)
        matrix[state][source] -= 1
        matrix[state][destination] += 1
    return matrix


def select_subset(
    worklist: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    size: int = 24,
    seed: int = 20260821,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    work_index = index_unique(worklist, "R1 worklist")
    label_index = index_unique(labels, "frozen R1 labels")
    if set(work_index) != set(label_index):
        raise ValueError("frozen R1 labels must exactly cover the R1 endpoint census")
    if size <= 0 or size > len(worklist):
        raise ValueError(f"subset size must be in [1, {len(worklist)}]")
    for key, label in label_index.items():
        validate_r1_label(label, key)

    merged = []
    for key, meta in work_index.items():
        row = dict(meta)
        row.update({field: label_index[key].get(field) for field in LABEL_FIELDS})
        merged.append(row)

    state_quota = state_targets(merged, size)
    scene_counts = Counter(str(row["scene_id"]) for row in merged)
    scene_quota = largest_remainder(dict(scene_counts), size)
    matrix = state_scene_targets(merged, state_quota, scene_quota)
    remaining = {state: dict(values) for state, values in matrix.items()}
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()

    # Preserve at least one example of every R1 endpoint-error type so error-type
    # stability is not silently untestable. R1 answers are never copied to R2.
    wrong_types = sorted(
        {str(row["final_error_type"]) for row in merged if row["final_state"] == "WRONG"},
        key=lambda value: (
            sum(row["final_error_type"] == value for row in merged),
            value,
        ),
    )
    if len(wrong_types) > state_quota["WRONG"]:
        raise ValueError("R2 WRONG quota is too small to cover observed endpoint-error types")
    for error_type in wrong_types:
        candidates = [
            row
            for row in merged
            if row["final_state"] == "WRONG"
            and row["final_error_type"] == error_type
            and case_key(row) not in selected_keys
            and remaining["WRONG"].get(str(row["scene_id"]), 0) > 0
        ]
        if not candidates:
            raise ValueError(f"cannot cover R1 error type within scene quotas: {error_type}")
        chosen = min(
            candidates,
            key=lambda row: (
                -remaining["WRONG"][str(row["scene_id"])],
                stable_key(row, seed, f"error:{error_type}"),
            ),
        )
        selected.append(chosen)
        selected_keys.add(case_key(chosen))
        remaining["WRONG"][str(chosen["scene_id"])] -= 1

    for state in ("UNCLEAR", "WRONG", "CORRECT"):
        for scene in sorted(scene_quota):
            need = remaining[state][scene]
            pool = sorted(
                (
                    row
                    for row in merged
                    if row["final_state"] == state
                    and str(row["scene_id"]) == scene
                    and case_key(row) not in selected_keys
                ),
                key=lambda row: stable_key(row, seed, f"fill:{state}:{scene}"),
            )
            if len(pool) < need:
                raise ValueError(f"not enough cases for R2 stratum {state}/{scene}")
            for chosen in pool[:need]:
                selected.append(chosen)
                selected_keys.add(case_key(chosen))
            remaining[state][scene] = 0

    if len(selected) != size or any(value for row in remaining.values() for value in row.values()):
        raise AssertionError("R2 selection did not satisfy its declared quotas")

    selected = sorted(selected, key=lambda row: stable_key(row, seed, "blind-order"))
    output = []
    for rank, row in enumerate(selected, 1):
        clean = {key: value for key, value in work_index[case_key(row)].items() if key not in LABEL_FIELDS}
        clean.update(
            {
                "r2_subset_rank": rank,
                "r2_selection_seed": seed,
                "r2_selection_method": "deterministic_scene_state_and_error_type_stratified_repeat_sample",
                "r2_r1_answers_exposed_to_page": False,
            }
        )
        output.append(clean)

    design = {
        "schema_version": "1.0.0",
        "purpose": "same-evidence endpoint repeatability check",
        "selection_method": "deterministic scene/state stratification with observed error-type coverage",
        "selection_seed": seed,
        "population_size": len(worklist),
        "subset_size": size,
        "scene_targets": scene_quota,
        "r1_state_targets": state_quota,
        "state_scene_targets": matrix,
        "observed_r1_error_types_covered": wrong_types,
        "r1_answers_exposed_in_r2_worklist": False,
        "r1_answers_exposed_by_review_service": False,
        "interpretation": (
            "This purposive repeat sample tests decision stability across endpoint states and error types; "
            "it is not a second estimate of the population endpoint-error rate."
        ),
    }
    return output, design


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--r1-labels", required=True, type=Path)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    root = args.validation_root.resolve()
    labels_dir = root / "labels"
    worklist_path = labels_dir / "r1_worklist.jsonl"
    r2_worklist_path = labels_dir / "r2_worklist.jsonl"
    r2_review_manifest_path = root / "r2_review_evidence_manifest.json"
    selection_manifest_path = root / "r2_selection_manifest.json"
    outputs = (r2_worklist_path, r2_review_manifest_path, selection_manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing R2 inputs: " + ", ".join(existing))

    r1_labels_path = args.r1_labels.resolve()
    worklist = read_jsonl(worklist_path)
    labels = read_jsonl(r1_labels_path)
    subset, design = select_subset(worklist, labels, size=args.size, seed=args.seed)
    full_review_manifest_path = root / "review_evidence_manifest.json"
    full_review_manifest = read_json(full_review_manifest_path)
    full_cases = index_unique(full_review_manifest.get("cases") or [], "review evidence manifest")
    subset_keys = {case_key(row) for row in subset}
    if not subset_keys <= set(full_cases):
        raise ValueError("R2 subset is not fully covered by the frozen review-evidence manifest")

    labels_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(r2_worklist_path, subset)
    r2_review_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": full_review_manifest.get("status"),
        "worklist_sha256": sha256_file(r2_worklist_path),
        "case_count": len(subset),
        "source_review_evidence_manifest_sha256": sha256_file(full_review_manifest_path),
        "all_artifact_hashes_match": full_review_manifest.get("all_artifact_hashes_match"),
        "all_available_final_objects_link_exactly": full_review_manifest.get(
            "all_available_final_objects_link_exactly"
        ),
        "cases": [full_cases[case_key(row)] for row in subset],
    }
    r2_review_manifest_path.write_text(
        json.dumps(r2_review_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    design.update(
        {
            "r1_worklist_sha256": sha256_file(worklist_path),
            "frozen_r1_labels": str(r1_labels_path),
            "frozen_r1_labels_sha256": sha256_file(r1_labels_path),
            "r2_worklist": str(r2_worklist_path),
            "r2_worklist_sha256": sha256_file(r2_worklist_path),
            "r2_review_evidence_manifest": str(r2_review_manifest_path),
            "r2_review_evidence_manifest_sha256": sha256_file(r2_review_manifest_path),
        }
    )
    selection_manifest_path.write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "r2_case_count": len(subset),
                "worklist": str(r2_worklist_path),
                "review_manifest": str(r2_review_manifest_path),
                "selection_manifest": str(selection_manifest_path),
                "design": design,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
