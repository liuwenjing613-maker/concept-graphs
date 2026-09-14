from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


FORBIDDEN_IDENTIFIERS = {
    "clean_membership",
    "clean_owner",
    "oracle_constraint",
    "affected_clean_groups",
    "expected_final_owner",
    "gt_object_identity",
    "gt_relation",
}
FORBIDDEN_IMPORT_PARTS = {"benchmark", "evaluate"}
DEFAULT_RUNTIME_FILES = (
    "conceptgraph/revision/constraints.py",
    "conceptgraph/revision/sparse_replay.py",
    "conceptgraph/revision/dependency_graph.py",
    "conceptgraph/revision/snapshot.py",
    "conceptgraph/revision/runtime_verify.py",
)


def audit_file(path: Path) -> list[dict[str, object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[dict[str, object]] = []
    for node in ast.walk(tree):
        identifier = None
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, ast.Attribute):
            identifier = node.attr
        elif isinstance(node, ast.arg):
            identifier = node.arg
        elif isinstance(node, ast.keyword):
            identifier = node.arg
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Covers dictionary/subscript keys such as state["clean_owner"].
            identifier = node.value
        if identifier in FORBIDDEN_IDENTIFIERS:
            violations.append(
                {
                    "path": str(path),
                    "line": int(getattr(node, "lineno", 0)),
                    "kind": "forbidden_identifier",
                    "value": identifier,
                }
            )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            else:
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                parts = set(module.split("."))
                overlap = parts & FORBIDDEN_IMPORT_PARTS
                if overlap:
                    violations.append(
                        {
                            "path": str(path),
                            "line": int(getattr(node, "lineno", 0)),
                            "kind": "forbidden_runtime_import",
                            "value": module,
                        }
                    )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit proposed revision runtime paths")
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_RUNTIME_FILES))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [Path(value) if Path(value).is_absolute() else root / value for value in args.paths]
    violations = [item for path in paths for item in audit_file(path)]
    result = {
        "schema_version": "1.0.0",
        "pass": not violations,
        "runtime_file_count": len(paths),
        "forbidden_identifiers": sorted(FORBIDDEN_IDENTIFIERS),
        "violations": violations,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
