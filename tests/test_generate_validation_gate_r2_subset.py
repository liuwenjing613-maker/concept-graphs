from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


LOCAL = Path(__file__).with_name("generate_validation_gate_r2_subset.py")
REPOSITORY = Path(__file__).parents[1] / "scripts" / "generate_validation_gate_r2_subset.py"
PATH = LOCAL if LOCAL.exists() else REPOSITORY
SPEC = spec_from_file_location("generate_validation_gate_r2_subset", PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixture_rows():
    rows = []
    for cohort in MODULE.COHORTS:
        for scene in ("room0", "office0"):
            for stage in MODULE.REQUIRED_STAGES:
                for number in range(3):
                    uid = f"{cohort}-{scene}-{stage}-{number}"
                    rows.append(
                        {
                            "scene_id": scene,
                            "case_uid": uid,
                            "finding_uid": uid,
                            "cohort": cohort,
                            "stage": stage,
                        }
                    )
    return rows


def test_subset_is_balanced_and_covers_required_strata():
    selected = MODULE.select_subset(fixture_rows(), per_cohort=8, seed=7)
    assert len(selected) == 16
    for cohort in MODULE.COHORTS:
        cohort_rows = [row for row in selected if row["cohort"] == cohort]
        assert len(cohort_rows) == 8
        assert {row["stage"] for row in cohort_rows} >= set(MODULE.REQUIRED_STAGES)
        assert {row["scene_id"] for row in cohort_rows} == {"room0", "office0"}


def test_subset_is_deterministic():
    first = MODULE.select_subset(fixture_rows(), per_cohort=8, seed=7)
    second = MODULE.select_subset(list(reversed(fixture_rows())), per_cohort=8, seed=7)
    assert [row["case_uid"] for row in first] == [row["case_uid"] for row in second]
