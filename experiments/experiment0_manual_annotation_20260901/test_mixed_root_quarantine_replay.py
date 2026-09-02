from run_mixed_root_quarantine_replay import (
    audit_gt_partition,
    quarantine_observation,
)


def _gt(obs_uid: str, gt_id: int) -> dict:
    return {
        "obs_uid": obs_uid,
        "gt_assignment_eligible": True,
        "gt_top_id": gt_id,
        "gt_purity": 1.0,
        "mask_mixed": False,
        "mask_two_foreground": False,
    }


def test_quarantine_observation_removes_exact_anchor() -> None:
    state = {
        "membership": {"a": ["old", "mixed"], "b": ["new"]},
        "objects": [
            {"entity_uid": "a", "member_observation_uids": ["old", "mixed"]},
            {"entity_uid": "b", "member_observation_uids": ["new"]},
        ],
    }
    result, audit = quarantine_observation(state, "mixed")
    assert result["membership"] == {"a": ["old"], "b": ["new"]}
    assert result["objects"][0]["member_observation_uids"] == ["old"]
    assert audit["native_owner_count"] == 1
    assert not audit["membership_contains_anchor_after"]


def test_complete_gt_partition_audit_distinguishes_instances() -> None:
    gt = {
        "a1": _gt("a1", 15),
        "a2": _gt("a2", 15),
        "b1": _gt("b1", 19),
        "b2": _gt("b2", 19),
    }
    clean = audit_gt_partition(
        {"old": ["a1", "a2"], "new": ["b1", "b2"]},
        gt,
        target_probe_obs_uid="a1",
    )
    mixed = audit_gt_partition(
        {"mixed": ["a1", "a2", "b1", "b2"]},
        gt,
        target_probe_obs_uid="a1",
    )
    assert clean["best_entities_are_distinct"]
    assert clean["per_gt"]["19"]["best_entity"]["f1"] == 1.0
    assert not mixed["best_entities_are_distinct"]
    assert mixed["per_gt"]["19"]["best_entity"]["precision"] == 0.5
