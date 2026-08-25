#!/usr/bin/env python3
"""Audit that the V0 production decision path has one semantic threshold."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


PRODUCTION_MODULES = (
    "conceptgraph/revision/evidence_split.py",
    "conceptgraph/revision/capabilities.py",
    "conceptgraph/revision/candidate_verifier.py",
    "conceptgraph/revision/selective_commit.py",
    "conceptgraph/revision/shadow_critic.py",
    "conceptgraph/revision/runtime_verify.py",
)

LEGACY_SEMANTIC_GATES = (
    "endpoint_improved",
    "no_op_controls_pass",
    "legal_merge_control_pass",
    "component_mechanism_supported",
    "local_global_parity_pass",
    "minimum_mean_confidence",
    "minimum_vote_confidence",
    "structural_vote_disagreement",
)

ORACLE_FIELD_FRAGMENTS = (
    "human_label",
    "expected_action",
    "expected_constraint",
    "desired_owner",
    "final_owner",
    "ground_truth",
    "posthoc_gold",
)


def _comparison_text(source: str, tree: ast.AST) -> list[str]:
    return [
        ast.get_source_segment(source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    modules = []
    all_source = []
    for relative in PRODUCTION_MODULES:
        path = root / relative
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        comparisons = _comparison_text(source, tree)
        modules.append(
            {
                "path": relative,
                "comparison_count": len(comparisons),
                "commit_threshold_comparisons": [
                    item
                    for item in comparisons
                    if "benefit_probability" in item and "commit_threshold" in item
                ],
            }
        )
        all_source.append(source)
    joined = "\n".join(all_source)
    legacy_hits = sorted(item for item in LEGACY_SEMANTIC_GATES if item in joined)
    oracle_hits = sorted(item for item in ORACLE_FIELD_FRAGMENTS if item in joined)
    threshold_comparisons = [
        item for module in modules for item in module["commit_threshold_comparisons"]
    ]
    checks = {
        "production_semantic_commit_threshold_equals_one": len(threshold_comparisons)
        == 1,
        "legacy_gate_stack_absent_from_production_path": not legacy_hits,
        "oracle_fields_absent_from_production_source": not oracle_hits,
        "vlm_self_confidence_not_compared_to_threshold": not any(
            "confidence" in item
            and "0.0 <= confidence <= 1.0" not in item
            and "confidence is not None" not in item
            for module in modules
            for item in _comparison_text(
                (root / module["path"]).read_text(encoding="utf-8"),
                ast.parse((root / module["path"]).read_text(encoding="utf-8")),
            )
        ),
    }
    result = {
        "schema_version": "1.0.0",
        "production_modules": modules,
        "semantic_commit_threshold_count": len(threshold_comparisons),
        "semantic_commit_threshold_comparisons": threshold_comparisons,
        "legacy_gate_hits": legacy_hits,
        "oracle_field_hits": oracle_hits,
        "checks": checks,
        "pass": all(checks.values()),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    print(encoded, end="")
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
