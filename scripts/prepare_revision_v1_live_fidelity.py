from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path
from typing import Any, Mapping


DEFAULT_COUNTS = {
    "FALSE_MERGE": 4,
    "FALSE_SPLIT": 3,
    "WRONG_MEMBERSHIP": 3,
}


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_counts(values: list[str]) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_COUNTS)
    result: dict[str, int] = {}
    for value in values:
        failure_type, separator, raw_count = value.partition("=")
        failure_type = failure_type.strip().upper()
        if not separator or failure_type not in DEFAULT_COUNTS:
            raise ValueError(
                "--count must be one of FALSE_MERGE=N, FALSE_SPLIT=N, "
                "WRONG_MEMBERSHIP=N"
            )
        count = int(raw_count)
        if count < 1:
            raise ValueError("live fidelity count must be positive")
        if failure_type in result:
            raise ValueError(f"duplicate live fidelity count for {failure_type}")
        result[failure_type] = count
    return result


def select(
    cases: list[dict[str, Any]],
    *,
    seed: int,
    counts: Mapping[str, int] = DEFAULT_COUNTS,
) -> list[dict[str, Any]]:
    selected = []
    for failure_type, count in counts.items():
        pool = sorted(
            (row for row in cases if str(row["failure_type"]) == failure_type),
            key=lambda row: str(row["case_uid"]),
        )
        rng = random.Random(seed + sum(ord(ch) for ch in failure_type))
        rng.shuffle(pool)
        if len(pool) < count:
            raise ValueError(f"{failure_type}: need {count} cases, found {len(pool)}")
        selected.extend(pool[:count])
    return selected


def validate_predeclared_source(
    selected: list[dict[str, Any]], source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    source_ids = [str(item) for item in source_manifest.get("selected_case_uids") or ()]
    selected_ids = [str(row["case_uid"]) for row in selected]
    missing = sorted(set(selected_ids) - set(source_ids))
    source_valid = bool(
        not source_manifest.get("outcome_screened", False)
        and source_manifest.get("frozen_before_new_live_outcomes", False)
        and len(source_ids) == len(set(source_ids))
    )
    if missing or not source_valid:
        raise ValueError(
            "staged live subset is not backed by a valid pre-outcome source manifest: "
            f"missing={missing}, source_valid={source_valid}"
        )
    return {
        "all_selected_cases_predeclared": True,
        "source_case_count": len(source_ids),
        "selected_case_count": len(selected_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an outcome-blind stratified live fidelity subset"
    )
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--count",
        action="append",
        default=[],
        metavar="FAILURE_TYPE=N",
        help="override staged counts; repeat once per included failure type",
    )
    parser.add_argument(
        "--source-frozen-manifest",
        help=(
            "earlier pre-live manifest from which every staged case must have been "
            "predeclared"
        ),
    )
    args = parser.parse_args()
    counts = parse_counts(args.count)
    primary_root = Path(args.primary_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_path = output_root / "live_fidelity_selection_manifest.json"
    cases_path = output_root / "cases.json"
    if manifest_path.exists() or cases_path.exists():
        if not (manifest_path.exists() and cases_path.exists()):
            raise RuntimeError("frozen live fidelity manifest is incomplete")
        print(json.dumps({"status": "REUSED_FROZEN", "case_count": len(_read(cases_path))}))
        return

    primary_manifest = _read(primary_root / "manifests" / "case_selection_manifest.json")
    if bool(primary_manifest.get("outcome_screened")):
        raise RuntimeError("primary manifest was outcome-screened")
    selected = select(
        list(_read(primary_root / "manifests" / "cases.json")),
        seed=args.seed,
        counts=counts,
    )
    source_manifest_path = (
        Path(args.source_frozen_manifest).resolve()
        if args.source_frozen_manifest
        else None
    )
    source_validation = None
    if source_manifest_path is not None:
        source_validation = validate_predeclared_source(
            selected, _read(source_manifest_path)
        )
    selector_sha = hashlib.sha256(inspect.getsource(select).encode()).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "seed": args.seed,
        "evaluation_role": "LIVE_SIMULATOR_FIDELITY",
        "outcome_screened": False,
        "frozen_before_new_live_outcomes": source_manifest_path is None,
        "frozen_before_fidelity_comparison_outcomes": True,
        "selection_timing_note": (
            "staged subset derived after some live maps existed, but before any "
            "live/simulator comparison result"
            if source_manifest_path is not None
            else "selection frozen before live mapping"
        ),
        "source_frozen_manifest": (
            str(source_manifest_path) if source_manifest_path is not None else None
        ),
        "source_manifest_validation": source_validation,
        "primary_manifest": str(
            (primary_root / "manifests" / "case_selection_manifest.json").resolve()
        ),
        "selection_counts": counts,
        "selector_sha256": selector_sha,
        "selected_case_uids": [str(row["case_uid"]) for row in selected],
    }
    _write(manifest_path, manifest)
    _write(cases_path, selected)
    for case in selected:
        _write(output_root / "cases" / f"{case['case_uid']}.json", case)
        _write(
            output_root / "plans" / f"{case['case_uid']}.json",
            case["corruption_plan"],
        )
    print(json.dumps({"status": "FROZEN", "case_count": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
