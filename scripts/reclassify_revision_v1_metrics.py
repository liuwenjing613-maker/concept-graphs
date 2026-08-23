from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conceptgraph.revision.benchmark.experiment_v1 import (
    _method_equivalent,
    aggregate_results,
    aligned_relation_metrics,
    classify_repair_outcome,
    selected_metric_paths,
    write_json,
)
from conceptgraph.revision.constraints import SparseRepairConstraint
from conceptgraph.revision.evaluate import evaluate_state
from conceptgraph.revision.runtime_verify import InvariantVerifier


OUTCOME_LABELS = {
    "CORRUPTION_SELF_HEALED_NO_FINAL_EFFECT",
    "CONSTRAINT_INSUFFICIENT",
    "COLLATERAL_DAMAGE",
}
DERIVED_ABLATION_LABELS = {
    "CONSTRAINT_NON_CAUSAL_ABLATION_EQUIVALENT",
    "NATURAL_RECOMPUTE_BASELINE_EQUIVALENT",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reclassify(root: Path, *, check_only: bool = False) -> dict[str, Any]:
    audits = []
    metric_paths, selection_integrity = selected_metric_paths(root)
    for path in metric_paths:
        row = _read(path)
        if row.get("status") != "COMPLETED":
            continue
        methods = row.get("methods") or {}
        branches = path.parent / "branches"
        reference_path = branches / "reference.json"
        if not reference_path.exists():
            raise FileNotFoundError(
                f"cannot audit stored relation metrics without {reference_path}"
            )
        reference_state = _read(reference_path)
        case = _read(path.parent / "case.json")
        affected = {
            str(item)
            for members in (case.get("affected_clean_groups") or {}).values()
            for item in members
        }
        known_observations = {
            str(item)
            for members in (reference_state.get("membership") or {}).values()
            for item in members
        }
        relation_state_changes: dict[str, bool] = {}
        for method_name, method in methods.items():
            branch_path = branches / f"{method_name}.json"
            if not branch_path.exists():
                raise FileNotFoundError(
                    f"cannot audit stored relation metrics without {branch_path}"
                )
            branch_state = _read(branch_path)
            refreshed = evaluate_state(
                reference_state,
                branch_state,
                affected_observations=affected,
            )
            relation = aligned_relation_metrics(reference_state, branch_state)
            previous_relation = method.get("relation") or {}
            relation_state_changes[method_name] = any(
                previous_relation.get(key) != relation.get(key)
                for key in (
                    "edge_state_match",
                    "edge_relation_match",
                    "edge_support_match",
                    "edge_set_f1_to_clean",
                    "support_absolute_error",
                )
            )
            refreshed["relation"] = relation
            methods[method_name] = refreshed

        historical = methods["historical_anchor_no_repair"]
        natural = methods["natural_recompute_ablation"]
        anchor = methods["anchor_only_local"]
        persistent = methods["persistent_sparse_local"]
        persistent_state = _read(branches / "persistent_sparse_local.json")
        anchor_state = _read(branches / "anchor_only_local.json")
        constraints = [
            SparseRepairConstraint.from_mapping(item)
            for item in _read(path.parent / "constraint.json")
        ]
        verifier = InvariantVerifier()
        anchor_verification = verifier.verify(
            state=anchor_state,
            constraints=constraints,
            source_hashes_before=reference_state.get("source_hashes"),
            source_hashes_after=anchor_state.get("source_hashes"),
            known_observation_uids=known_observations,
        )
        persistent_verification = verifier.verify(
            state=persistent_state,
            constraints=constraints,
            source_hashes_before=reference_state.get("source_hashes"),
            source_hashes_after=persistent_state.get("source_hashes"),
            known_observation_uids=known_observations,
        )
        runtime_verification = {
            "anchor_only": anchor_verification,
            "persistent_sparse": persistent_verification,
            "pass": anchor_verification["pass"] and persistent_verification["pass"],
            "refreshed_from_stored_branches": True,
        }
        row["runtime_verification"] = runtime_verification
        no_constraint_equivalent = _method_equivalent(historical, persistent)
        natural_recompute_equivalent = _method_equivalent(natural, persistent)
        anchor_persistent_equivalent = _method_equivalent(anchor, persistent)
        constraint_diagnostics = row.setdefault("constraint_diagnostics", {})
        constraint_diagnostics.update(
            {
                "no_constraint_equivalent": no_constraint_equivalent,
                "natural_recompute_equivalent": natural_recompute_equivalent,
                "anchor_persistent_equivalent": anchor_persistent_equivalent,
                "supports_sparse_constraint_causal_claim": bool(
                    persistent_state.get("constraint_historical_override_count", 0) > 0
                    and not no_constraint_equivalent
                ),
            }
        )
        outcome = classify_repair_outcome(
            corrupted_method=methods["temporal_corrupted_local"],
            persistent_method=methods["persistent_sparse_local"],
            verification_pass=bool((row.get("runtime_verification") or {}).get("pass")),
        )
        taxonomy = set(str(item) for item in row.get("failure_taxonomy") or ())
        taxonomy.difference_update(OUTCOME_LABELS)
        taxonomy.difference_update(DERIVED_ABLATION_LABELS)
        if not any(outcome["damage_dimensions"].values()):
            taxonomy.add("CORRUPTION_SELF_HEALED_NO_FINAL_EFFECT")
        elif not any(outcome["improved_dimensions"].values()):
            taxonomy.add("CONSTRAINT_INSUFFICIENT")
        if not outcome["collateral_safe"]:
            taxonomy.add("COLLATERAL_DAMAGE")
        if natural_recompute_equivalent:
            taxonomy.add("NATURAL_RECOMPUTE_BASELINE_EQUIVALENT")
        if (
            int(persistent_state.get("constraint_hit_count", 0)) > 0
            and int(persistent_state.get("constraint_override_count", 0)) == 0
            and no_constraint_equivalent
        ):
            taxonomy.add("CONSTRAINT_NON_CAUSAL_ABLATION_EQUIVALENT")
        before = {
            "pass": bool(row.get("pass")),
            "damage_dimensions": row.get("damage_dimensions"),
            "improved_dimensions": row.get("improved_dimensions"),
            "collateral_safe": row.get("collateral_safe"),
            "failure_taxonomy": sorted(row.get("failure_taxonomy") or ()),
        }
        after = {
            "pass": bool(outcome["pass"]),
            "damage_dimensions": outcome["damage_dimensions"],
            "improved_dimensions": outcome["improved_dimensions"],
            "collateral_safe": outcome["collateral_safe"],
            "failure_taxonomy": sorted(taxonomy),
        }
        classification_changed = before != after
        changed = classification_changed or any(relation_state_changes.values())
        audits.append(
            {
                "case_uid": row.get("case_uid", path.parent.name),
                "changed": changed,
                "classification_changed": classification_changed,
                "before": before,
                "after": after,
                "relation_metrics_refreshed_from_stored_branches": True,
                "all_method_metrics_refreshed_from_stored_branches": True,
                "relation_methods_refreshed": sorted(methods),
                "relation_state_changed": sorted(
                    name
                    for name, changed in relation_state_changes.items()
                    if changed
                ),
            }
        )
        if not check_only:
            row.update(after)
            row["outcome_classification"] = {
                "schema_version": "1.2.0",
                "source": "STORED_BRANCH_STATES_AND_METHOD_METRICS",
                "thresholds": outcome["thresholds"],
                "reclassified": changed,
            }
            write_json(path, row)
            write_json(path.parent / "runtime_verification.json", runtime_verification)
    result = {
        "schema_version": "1.0.0",
        "check_only": check_only,
        "selection_integrity": selection_integrity,
        "completed_case_count": len(audits),
        "changed_case_count": sum(bool(row["changed"]) for row in audits),
        "cases": audits,
    }
    if not check_only:
        result["aggregate"] = aggregate_results(root)
        write_json(root / "outcome_reclassification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reclassify stored V1 outcomes with declared numeric tolerances"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = reclassify(Path(args.output_root).resolve(), check_only=args.check_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
