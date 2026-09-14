from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.experiment import build_aggregate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Revision Kernel metrics")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    print(json.dumps(build_aggregate_report(args.run_root), indent=2))


if __name__ == "__main__":
    main()
