from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.experiment import write_json
from conceptgraph.revision.index import ProvenanceIndex
from conceptgraph.revision.materialize import ObservationMaterializer


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exact replay materialization")
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    provenance = ProvenanceIndex(args.base_run)
    with (provenance.experiment_root / "config_params.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    materializer = ObservationMaterializer(provenance, cfg)
    observations = sorted(provenance.association_for_obs)
    failures = []
    failed_checks: Counter[str] = Counter()
    started = time.perf_counter()
    for index, obs_uid in enumerate(observations, 1):
        try:
            result = materializer.fidelity(obs_uid)
            if not result["pass"]:
                failures.append(result)
                failed_checks.update(
                    key for key, passed in result["checks"].items() if not passed
                )
        except Exception as exc:
            failures.append(
                {"obs_uid": obs_uid, "pass": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            failed_checks["exception"] += 1
        if index % 250 == 0:
            print(f"validated {index}/{len(observations)}", flush=True)
    elapsed = time.perf_counter() - started
    output = {
        "pass": not failures,
        "observation_count": len(observations),
        "passed_count": len(observations) - len(failures),
        "failure_count": len(failures),
        "failed_checks": dict(failed_checks),
        "failures": failures[:100],
        "elapsed_seconds": elapsed,
        "exactness_policy": "no approximation; exact PCD points, CLIP feature, class and obs UID",
        "source_hashes": provenance.source_hashes(),
    }
    write_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "failures"}, indent=2))
    raise SystemExit(0 if output["pass"] else 1)


if __name__ == "__main__":
    main()
