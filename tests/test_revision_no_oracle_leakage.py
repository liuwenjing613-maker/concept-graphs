from pathlib import Path

from scripts.audit_revision_oracle_leakage import DEFAULT_RUNTIME_FILES, audit_file


def test_proposed_runtime_paths_have_no_benchmark_trajectory_access():
    root = Path(__file__).resolve().parents[1]
    violations = [
        item
        for relative in DEFAULT_RUNTIME_FILES
        for item in audit_file(root / relative)
    ]
    assert violations == []


def test_audit_rejects_a_forbidden_runtime_identifier(tmp_path):
    source = tmp_path / "bad_runtime.py"
    source.write_text("def bad(engine):\n    return engine.clean_owner\n", encoding="utf-8")
    violations = audit_file(source)
    assert len(violations) == 1
    assert violations[0]["value"] == "clean_owner"


def test_audit_rejects_forbidden_string_key_access(tmp_path):
    source = tmp_path / "bad_subscript_runtime.py"
    source.write_text(
        'def bad(state):\n    return state["clean_membership"]\n',
        encoding="utf-8",
    )
    violations = audit_file(source)
    assert len(violations) == 1
    assert violations[0]["value"] == "clean_membership"
