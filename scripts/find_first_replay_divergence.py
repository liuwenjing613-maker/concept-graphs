from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    for key in ("decision_trace", "trace", "events"):
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"no trace list in {source}")


def first_divergence(
    reference: Sequence[Mapping[str, Any]], replayed: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    fields = (
        "obs_uid",
        "event_uid",
        "decision",
        "natural_match",
        "applied_match",
        "object_versions",
        "member_observation_uids",
        "postprocess_state",
    )
    limit = max(len(reference), len(replayed))
    for index in range(limit):
        left = reference[index] if index < len(reference) else None
        right = replayed[index] if index < len(replayed) else None
        if left is None or right is None:
            return {
                "index": index,
                "reason": "trace_length_mismatch",
                "reference": left,
                "replayed": right,
            }
        differences = {
            field: {"reference": left.get(field), "replayed": right.get(field)}
            for field in fields
            if left.get(field) != right.get(field)
        }
        if differences:
            return {
                "index": index,
                "reason": "field_mismatch",
                "frame": right.get("frame_idx", left.get("frame_idx")),
                "obs_uid": right.get("obs_uid", left.get("obs_uid")),
                "differences": differences,
                "reference_candidates": left.get("natural_candidates"),
                "replayed_candidates": right.get("natural_candidates"),
                "reference": dict(left),
                "replayed": dict(right),
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate the first replay trace divergence")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--replayed", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = {
        "schema_version": "1.0.0",
        "first_divergence": first_divergence(
            _read_trace(args.reference), _read_trace(args.replayed)
        ),
    }
    result["pass"] = result["first_divergence"] is None
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
