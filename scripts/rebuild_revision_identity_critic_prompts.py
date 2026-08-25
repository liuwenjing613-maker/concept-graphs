#!/usr/bin/env python3
"""Rebuild frozen critic prompts with current policy and unchanged evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from conceptgraph.revision.autonomous_identity import build_pairwise_critic_prompt
from conceptgraph.revision.evidence_split import sha256_file


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _payload(request: Mapping[str, Any]) -> dict[str, Any]:
    marker = "FROZEN HELD-OUT OUTCOMES:\n"
    prompt = str(request["prompt"])
    if marker not in prompt:
        raise ValueError("critic prompt does not contain frozen payload")
    payload = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(payload, dict):
        raise ValueError("critic payload must be one object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prompt-revision", required=True)
    args = parser.parse_args()

    protocol_path = args.freeze_protocol.resolve()
    protocol = _read(protocol_path)
    if protocol.get("runtime_human_or_gold_loaded") is not False:
        raise ValueError("source protocol did not pass oracle isolation")
    output_root = args.output_root.resolve()
    output_protocol = output_root / "rebuilt_prompt_protocol.json"
    if output_protocol.exists():
        raise FileExistsError(f"refusing to overwrite {output_protocol}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for source_row in protocol.get("critic_requests") or ():
        source_path = Path(str(source_row["path"])).resolve()
        if sha256_file(source_path) != str(source_row["sha256"]):
            raise ValueError(f"source request hash drift: {source_path}")
        request = _read(source_path)
        payload = _payload(request)
        images = list(request.get("images") or ())
        for image in images:
            image_path = Path(str(image["path"])).resolve()
            if sha256_file(image_path) != str(image["sha256"]):
                raise ValueError(f"source image hash drift: {image_path}")
        prompt = build_pairwise_critic_prompt(
            incident_uid=str(request["case_uid"]),
            evidence_rows=images,
            state_summaries=payload["executed_states"],
        )
        destination = (
            output_root
            / str(request["case_uid"])
            / "critic_requests"
            / f"{request['request_uid']}.json"
        )
        rebuilt = {
            **{
                key: value
                for key, value in request.items()
                if key not in {"prompt", "prompt_sha256"}
            },
            "parent_request_path": str(source_path),
            "parent_request_sha256": str(source_row["sha256"]),
            "prompt_revision": str(args.prompt_revision),
            "evidence_changed": False,
            "state_summaries_changed": False,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        _write(destination, rebuilt)
        rows.append(
            {
                "request_uid": str(request["request_uid"]),
                "case_uid": str(request["case_uid"]),
                "pair_uid": str(request["pair_uid"]),
                "order_index": int(request["order_index"]),
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
            }
        )

    if not rows:
        raise ValueError("source protocol contains no critic requests")
    rebuilt_protocol = {
        "schema_version": "1.0.0",
        "role": "DEVELOPMENT_PROMPT_POLICY_ABLATION",
        "parent_freeze_protocol_path": str(protocol_path),
        "parent_freeze_protocol_sha256": sha256_file(protocol_path),
        "prompt_revision": str(args.prompt_revision),
        "case_count": len({row["case_uid"] for row in rows}),
        "request_count": len(rows),
        "cases": list(protocol.get("cases") or ()),
        "critic_requests": rows,
        "runtime_human_or_gold_loaded": False,
        "candidate_states_changed": False,
        "evidence_changed": False,
        "production_commit_permitted": False,
    }
    _write(output_protocol, rebuilt_protocol)
    print(
        json.dumps(
            {
                "status": "PASS",
                "case_count": rebuilt_protocol["case_count"],
                "request_count": len(rows),
                "output": str(output_protocol),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
