from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.experiment import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one controlled revision case")
    parser.add_argument("--case-root", required=True)
    args = parser.parse_args()
    root = Path(args.case_root)
    metrics = read_json(root / "metrics.json")
    transaction = read_json(root / "transaction.json")
    result = {
        "case_uid": metrics["case_uid"],
        "failure_type": metrics["failure_type"],
        "pass": metrics["pass"],
        "acceptance": metrics["acceptance"],
        "local_vs_global": metrics["local_vs_global"],
        "transaction_status": transaction["commit_status"],
        "hard_invariant_failures": transaction["verification"]["hard_invariant_failures"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
