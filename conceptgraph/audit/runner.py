"""CLI for the read-only layered causal audit v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conceptgraph.audit.layered_audit import run_layered_audit


def _bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--audit_config", type=Path, required=True)
    parser.add_argument("--build_cases", type=_bool, default=None)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    result = run_layered_audit(
        args.experiment_dir,
        args.audit_config,
        build_cases=args.build_cases,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0 if result["summary"]["gate_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
