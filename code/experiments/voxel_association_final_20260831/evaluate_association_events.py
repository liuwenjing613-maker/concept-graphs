#!/usr/bin/env python3
"""Causal event-level audit for historical association uncertainty.

The evaluator consumes matrices and object versions captured at the original
online decision. Replica instance GT is used only after mapping for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


BASE = Path("/home/chenkejun/beauty/conceptgraphs")
ONLINE_ROOT = BASE / "results/experiments/online_label_trigger_v1_20260831"
SCENES = {
    "dev": {
        "room0": ONLINE_ROOT / "dev/room0/scene_summary.json",
        "office0": ONLINE_ROOT / "dev/office0/scene_summary.json",
    },
    "holdout": {
        "room1": ONLINE_ROOT / "holdout/room1/scene_summary.json",
        "office1": ONLINE_ROOT / "holdout/office1/scene_summary.json",
    },
}
BG_LABELS = {"wall", "floor", "ceiling", "unknown", "undefined", "background"}
FEATURE_ORDER = [
    "risk_low_margin",
    "risk_near_threshold",
    "risk_modality_disagreement",
    "risk_semantic_residual",
    "risk_spatial_residual",
]
GT_PURITY = 0.90
GT_SUPPORT = 0.90
GT_TOP_PIXELS = 25
HISTORY_MIN_OBS = 3
HISTORY_MIN_FRAMES = 3
HISTORY_DOMINANT_RATIO = 0.80
BOOTSTRAP_SEED = 20260831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-rule", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def reliable_gt(row: dict) -> bool:
    label = str(row.get("gt_top_label") or "").strip().lower()
    return bool(
        row.get("gt_assignment_eligible")
        and safe_float(row.get("gt_purity")) is not None
        and float(row["gt_purity"]) >= GT_PURITY
        and safe_float(row.get("gt_supported_fraction")) is not None
        and float(row["gt_supported_fraction"]) >= GT_SUPPORT
        and int(row.get("gt_top_pixels") or 0) >= GT_TOP_PIXELS
        and row.get("gt_top_id") is not None
        and label not in BG_LABELS
    )


@dataclass(frozen=True)
class Identity:
    gt_id: int
    reliable_observations: int
    unique_frames: int
    dominant_ratio: float


def historical_identity(
    version_uid: str | None,
    current_raw_frame: int,
    versions: dict[str, dict],
    gt_by_obs: dict[str, dict],
) -> tuple[Identity | None, str]:
    if not version_uid or version_uid not in versions:
        return None, "missing_version"
    rows = []
    for obs_uid in versions[version_uid].get("member_observation_uids") or []:
        gt = gt_by_obs.get(str(obs_uid))
        if gt is None or not reliable_gt(gt):
            continue
        if int(gt.get("raw_frame", gt.get("frame_idx", -1))) >= current_raw_frame:
            continue
        rows.append(gt)
    if len(rows) < HISTORY_MIN_OBS:
        return None, "history_lt3_reliable"
    frames = {int(row.get("raw_frame", row.get("frame_idx", -1))) for row in rows}
    if len(frames) < HISTORY_MIN_FRAMES:
        return None, "history_lt3_frames"
    counts = Counter(int(row["gt_top_id"]) for row in rows)
    gt_id, count = counts.most_common(1)[0]
    ratio = count / len(rows)
    if ratio < HISTORY_DOMINANT_RATIO:
        return None, "history_ambiguous"
    return Identity(gt_id, len(rows), len(frames), ratio), "ok"


def metric_values(y: np.ndarray, score: np.ndarray) -> dict:
    valid = np.isfinite(score)
    y = np.asarray(y[valid], dtype=int)
    score = np.asarray(score[valid], dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "n": int(len(y)),
            "positives": int(y.sum()) if len(y) else 0,
            "prevalence": float(y.mean()) if len(y) else None,
            "auroc": None,
            "average_precision": None,
            "ap_lift": None,
            "top20_error_rate": None,
            "bottom20_error_rate": None,
            "top_bottom_ratio": None,
        }
    order = np.argsort(score, kind="stable")
    k = max(1, int(math.ceil(0.2 * len(y))))
    bottom = y[order[:k]]
    top = y[order[-k:]]
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, score))
    bottom_rate = float(bottom.mean())
    top_rate = float(top.mean())
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": ap,
        "ap_lift": ap - prevalence,
        "top20_error_rate": top_rate,
        "bottom20_error_rate": bottom_rate,
        "top_bottom_ratio": (top_rate / bottom_rate) if bottom_rate > 0 else (math.inf if top_rate > 0 else None),
    }


def cluster_bootstrap(rows: list[dict], feature: str, iterations: int) -> dict:
    usable = [row for row in rows if safe_float(row.get(feature)) is not None]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in usable:
        groups[str(row["cluster_uid"])].append(row)
    cluster_ids = sorted(groups)
    if len(cluster_ids) < 2:
        return {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled = [row for cluster in selected for row in groups[str(cluster)]]
        y = np.asarray([row["is_error"] for row in sampled], dtype=int)
        score = np.asarray([row[feature] for row in sampled], dtype=float)
        current = metric_values(y, score)
        for key in ("auroc", "average_precision", "ap_lift"):
            if current.get(key) is not None and math.isfinite(float(current[key])):
                samples[key].append(float(current[key]))
    output = {"cluster_count": len(cluster_ids), "iterations_requested": iterations}
    for key, values in samples.items():
        output[f"{key}_ci95"] = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        output[f"{key}_valid_iterations"] = len(values)
    return output


def load_scene(scene: str, summary_path: Path) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    summary = read_json(summary_path)
    exp_root = Path(summary["exp_root"])
    evidence_root = exp_root / "evidence"
    gt_path = summary_path.parent / "observation_gt.jsonl"
    gt_by_obs = {str(row["obs_uid"]): row for row in read_jsonl(gt_path)}
    versions = {str(row["object_version_uid"]): row for row in read_jsonl(evidence_root / "object_versions.jsonl")}
    associations = list(read_jsonl(evidence_root / "associations.jsonl"))
    frame_cache: dict[str, dict[str, object]] = {}
    records: list[dict] = []
    exclusions = Counter()
    integrity = Counter()

    for association in associations:
        obs_uid = str(association.get("obs_uid"))
        obs_gt = gt_by_obs.get(obs_uid)
        if obs_gt is None:
            exclusions["missing_observation_gt"] += 1
            continue
        if not reliable_gt(obs_gt):
            exclusions["observation_gt_unreliable"] += 1
            continue
        if not association.get("similarity_evidence_valid"):
            exclusions["invalid_similarity_evidence"] += 1
            continue
        object_uids = [str(value) for value in association.get("object_uids_before") or []]
        if not object_uids:
            exclusions["no_candidates"] += 1
            continue
        ref = association.get("aggregate_sim_ref") or {}
        relative = str(ref.get("path") or "")
        matrix_path = exp_root / relative
        cache_key = str(matrix_path)
        if cache_key not in frame_cache:
            with np.load(matrix_path, allow_pickle=False) as bundle:
                frame_cache[cache_key] = {
                    "observation_uids": [str(value) for value in bundle["observation_uids"].tolist()],
                    "object_uids": [str(value) for value in bundle["object_uids"].tolist()],
                    "spatial_sim": np.asarray(bundle["spatial_sim"], dtype=float),
                    "visual_sim": np.asarray(bundle["visual_sim"], dtype=float),
                    "aggregate_sim": np.asarray(bundle["aggregate_sim"], dtype=float),
                }
        matrices = frame_cache[cache_key]
        if matrices["object_uids"] != object_uids:
            exclusions["matrix_object_order_mismatch"] += 1
            continue
        try:
            row_index = matrices["observation_uids"].index(obs_uid)
        except ValueError:
            exclusions["matrix_observation_missing"] += 1
            continue
        spatial = np.asarray(matrices["spatial_sim"])[row_index]
        visual = np.asarray(matrices["visual_sim"])[row_index]
        aggregate = np.asarray(matrices["aggregate_sim"])[row_index]
        if not (len(spatial) == len(visual) == len(aggregate) == len(object_uids)):
            exclusions["matrix_shape_mismatch"] += 1
            continue
        order = np.argsort(aggregate)[::-1]
        top1_idx = int(order[0])
        top2_idx = int(order[1]) if len(order) > 1 else None
        top1 = float(aggregate[top1_idx])
        top2 = float(aggregate[top2_idx]) if top2_idx is not None else None
        recorded_top1 = safe_float(association.get("top1_score"))
        if recorded_top1 is None or abs(recorded_top1 - top1) > 1e-5:
            exclusions["top1_reconstruction_mismatch"] += 1
            continue
        integrity["matrix_rows_validated"] += 1
        raw_frame = int(obs_gt.get("raw_frame", obs_gt.get("frame_idx")))
        version_uids = association.get("candidate_object_version_uids") or []
        if len(version_uids) != len(object_uids):
            exclusions["candidate_version_alignment_mismatch"] += 1
            continue
        identities = []
        identity_reasons = []
        for version_uid in version_uids:
            identity, reason = historical_identity(version_uid, raw_frame, versions, gt_by_obs)
            identities.append(identity)
            identity_reasons.append(reason)

        decision = str(association.get("decision"))
        obs_gt_id = int(obs_gt["gt_top_id"])
        threshold = float(association["sim_threshold"])
        base = {
            "scene": scene,
            "event_uid": association.get("event_uid"),
            "obs_uid": obs_uid,
            "raw_frame": raw_frame,
            "decision": decision,
            "obs_gt_id": obs_gt_id,
            "obs_gt_label": obs_gt.get("gt_top_label"),
            "obs_gt_purity": float(obs_gt["gt_purity"]),
            "candidate_count": len(object_uids),
            "top1_aggregate": top1,
            "top2_aggregate": top2,
            "threshold": threshold,
            "risk_low_margin": (-(top1 - top2)) if top2 is not None else None,
            "risk_near_threshold": -abs(top1 - threshold),
            "risk_modality_disagreement": float(int(np.argmax(spatial) != np.argmax(visual))),
            "spatial_winner_uid": object_uids[int(np.argmax(spatial))],
            "semantic_winner_uid": object_uids[int(np.argmax(visual))],
            "top1_candidate_uid": object_uids[top1_idx],
        }

        if decision == "MERGE_TO_OBJECT":
            target_uid = str(association.get("target_object_uid"))
            if target_uid not in object_uids:
                exclusions["merge_target_not_candidate"] += 1
                continue
            target_idx = object_uids.index(target_uid)
            identity = identities[target_idx]
            if identity is None:
                exclusions[f"merge_target_{identity_reasons[target_idx]}"] += 1
                continue
            base.update(
                {
                    "event_family": "merge",
                    "cluster_uid": target_uid,
                    "target_object_uid": target_uid,
                    "target_gt_id": identity.gt_id,
                    "target_history_observations": identity.reliable_observations,
                    "target_history_frames": identity.unique_frames,
                    "target_history_dominant_ratio": identity.dominant_ratio,
                    "is_error": int(identity.gt_id != obs_gt_id),
                    "risk_semantic_residual": float(np.max(visual) - visual[target_idx]),
                    "risk_spatial_residual": float(np.max(spatial) - spatial[target_idx]),
                    "correct_candidate_exists": int(any(item is not None and item.gt_id == obs_gt_id for item in identities)),
                }
            )
            records.append(base)
        elif decision == "CREATE_OBJECT":
            identity = identities[top1_idx]
            if identity is None:
                exclusions[f"create_top1_{identity_reasons[top1_idx]}"] += 1
                continue
            base.update(
                {
                    "event_family": "create",
                    "cluster_uid": object_uids[top1_idx],
                    "target_object_uid": None,
                    "target_gt_id": identity.gt_id,
                    "target_history_observations": identity.reliable_observations,
                    "target_history_frames": identity.unique_frames,
                    "target_history_dominant_ratio": identity.dominant_ratio,
                    "is_error": int(identity.gt_id == obs_gt_id),
                    "risk_semantic_residual": None,
                    "risk_spatial_residual": None,
                    "correct_candidate_exists": int(identity.gt_id == obs_gt_id),
                }
            )
            records.append(base)
        else:
            exclusions["unknown_decision"] += 1

    audit = {
        "scene": scene,
        "summary_path": str(summary_path),
        "exp_root": str(exp_root),
        "associations_total": len(associations),
        "gt_rows": len(gt_by_obs),
        "object_versions": len(versions),
        "evaluable_records": len(records),
        "evaluable_merge": sum(row["event_family"] == "merge" for row in records),
        "evaluable_create": sum(row["event_family"] == "create" for row in records),
        "exclusions": dict(sorted(exclusions.items())),
        "integrity": dict(sorted(integrity.items())),
        "runtime_seconds": time.perf_counter() - started,
    }
    return records, audit


def evaluate_scene(scene: str, records: list[dict], iterations: int) -> list[dict]:
    rows = []
    for family in ("merge", "create"):
        family_rows = [row for row in records if row["event_family"] == family]
        features = FEATURE_ORDER if family == "merge" else FEATURE_ORDER[:3]
        for feature in features:
            usable = [row for row in family_rows if safe_float(row.get(feature)) is not None]
            y = np.asarray([row["is_error"] for row in usable], dtype=int)
            score = np.asarray([row[feature] for row in usable], dtype=float)
            metric = metric_values(y, score)
            bootstrap = cluster_bootstrap(usable, feature, iterations)
            positive_clusters = len({row["cluster_uid"] for row in usable if row["is_error"]})
            negative_clusters = len({row["cluster_uid"] for row in usable if not row["is_error"]})
            rows.append(
                {
                    "scene": scene,
                    "event_family": family,
                    "feature": feature,
                    "positive_clusters": positive_clusters,
                    "negative_clusters": negative_clusters,
                    **metric,
                    **bootstrap,
                }
            )
    return rows


def select_dev_rule(metric_rows: list[dict]) -> dict:
    by_key = {(row["scene"], row["event_family"], row["feature"]): row for row in metric_rows}
    dev_scenes = sorted(SCENES["dev"])
    viability = {}
    for scene in dev_scenes:
        reference = by_key.get((scene, "merge", "risk_near_threshold"), {})
        viability[scene] = {
            "n": reference.get("n", 0),
            "positives": reference.get("positives", 0),
            "negatives": (reference.get("n", 0) - reference.get("positives", 0)),
            "positive_clusters": reference.get("positive_clusters", 0),
            "negative_clusters": reference.get("negative_clusters", 0),
        }
    viable = all(
        item["n"] >= 50
        and item["positives"] >= 10
        and item["negatives"] >= 10
        and item["positive_clusters"] >= 5
        and item["negative_clusters"] >= 5
        for item in viability.values()
    )
    if not viable:
        return {"status": "STOP_DATA_UNSUPPORTED", "selected_merge_feature": None, "viability": viability}

    candidates = []
    for order_index, feature in enumerate(FEATURE_ORDER):
        rows = [by_key.get((scene, "merge", feature)) for scene in dev_scenes]
        if any(row is None or row.get("auroc") is None or row.get("ap_lift") is None for row in rows):
            continue
        if all(row["auroc"] > 0.55 and row["ap_lift"] > 0 for row in rows):
            candidates.append(
                {
                    "feature": feature,
                    "min_auroc": min(row["auroc"] for row in rows),
                    "min_ap_lift": min(row["ap_lift"] for row in rows),
                    "order_index": order_index,
                }
            )
    if not candidates:
        return {"status": "STOP_EVENT_SIGNAL", "selected_merge_feature": None, "viability": viability}
    selected = max(candidates, key=lambda row: (row["min_auroc"], row["min_ap_lift"], -row["order_index"]))
    return {
        "status": "CONTINUE_TO_HOLDOUT",
        "selected_merge_feature": selected["feature"],
        "selection": selected,
        "eligible_features": candidates,
        "viability": viability,
        "feature_order": FEATURE_ORDER,
        "gt_thresholds": {
            "purity": GT_PURITY,
            "supported_fraction": GT_SUPPORT,
            "top_pixels": GT_TOP_PIXELS,
            "history_min_observations": HISTORY_MIN_OBS,
            "history_min_frames": HISTORY_MIN_FRAMES,
            "history_dominant_ratio": HISTORY_DOMINANT_RATIO,
        },
    }


def holdout_decision(metric_rows: list[dict], selected_feature: str) -> dict:
    selected = [
        row
        for row in metric_rows
        if row["event_family"] == "merge" and row["feature"] == selected_feature
    ]
    if len(selected) != len(SCENES["holdout"]):
        return {"status": "STOP_HOLDOUT_INCOMPLETE", "selected_feature": selected_feature}
    if any(row.get("auroc") is None or row.get("ap_lift") is None for row in selected):
        return {"status": "STOP_HOLDOUT_UNEVALUABLE", "selected_feature": selected_feature}
    if any(row["auroc"] <= 0.5 or row["ap_lift"] <= 0 for row in selected):
        status = "STOP_HOLDOUT_REVERSAL"
    elif all(row["auroc"] >= 0.70 and row["ap_lift"] >= 0.10 for row in selected):
        status = "GO_EVENT_SIGNAL"
    else:
        status = "MODIFY_EVENT_WEAK"
    return {
        "status": status,
        "selected_feature": selected_feature,
        "scene_results": {
            row["scene"]: {
                "n": row["n"],
                "positives": row["positives"],
                "auroc": row["auroc"],
                "ap_lift": row["ap_lift"],
            }
            for row in selected
        },
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    output = args.output_root / args.split
    output.mkdir(parents=True, exist_ok=True)
    all_records = []
    audits = []
    metric_rows = []
    for scene, summary_path in SCENES[args.split].items():
        print(f"[START] {args.split}/{scene}: load causal association evidence", flush=True)
        records, audit = load_scene(scene, summary_path)
        all_records.extend(records)
        audits.append(audit)
        metric_rows.extend(evaluate_scene(scene, records, args.bootstrap))
        print(
            f"[DONE] {scene}: evaluable={len(records)} "
            f"merge={audit['evaluable_merge']} create={audit['evaluable_create']} "
            f"time={audit['runtime_seconds']:.2f}s",
            flush=True,
        )

    write_jsonl(output / "event_records.jsonl", all_records)
    write_csv(output / "event_records.csv", all_records)
    write_csv(output / "event_metrics.csv", metric_rows)
    atomic_json(output / "integrity_audit.json", audits)

    if args.split == "dev":
        decision = select_dev_rule(metric_rows)
        frozen = {
            **decision,
            "split": "dev",
            "preregistered": True,
            "bootstrap_iterations": args.bootstrap,
        }
        atomic_json(output / "frozen_rule.json", frozen)
    else:
        if args.frozen_rule is None:
            raise SystemExit("--frozen-rule is required for holdout")
        frozen = read_json(args.frozen_rule)
        selected_feature = frozen.get("selected_merge_feature")
        if not selected_feature:
            raise SystemExit("DEV rule stopped; holdout evaluation is prohibited")
        decision = holdout_decision(metric_rows, str(selected_feature))
        atomic_json(output / "holdout_decision.json", decision)

    summary = {
        "split": args.split,
        "scenes": list(SCENES[args.split]),
        "records": len(all_records),
        "runtime_seconds": time.perf_counter() - started,
        "decision": decision,
    }
    atomic_json(output / "summary.json", summary)
    (output / "READY").write_text("READY\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
