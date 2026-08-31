#!/usr/bin/env python3
"""Selected uncertainty and size-confound checks for voxel-trigger re-audit."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--repetitions", type=int, default=5000)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def residualize(rows: list[dict], source: str) -> dict[tuple[str, int], float]:
    output = {}
    for scene in ("room0", "office0"):
        selected = [row for row in rows if row["scene"] == scene]
        score = np.asarray([float(row[source]) for row in selected], dtype=float)
        x = np.column_stack(
            (
                np.ones(len(selected)),
                [float(row["score__size_num_detections"]) for row in selected],
                [float(row["score__size_object_voxels"]) for row in selected],
            )
        )
        beta = np.linalg.lstsq(x, score, rcond=None)[0]
        ranked = percentile(score - x @ beta)
        for row, value in zip(selected, ranked):
            output[(scene, int(row["object_index"]))] = float(value)
    return output


def selected_arrays(rows: list[dict], scene: str, target: str, score: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["scene"] == scene and row[target] != ""]
    y = np.asarray([int(row[target]) for row in selected], dtype=np.uint8)
    values = np.asarray([float(row[score]) for row in selected], dtype=float)
    return y, values


def uncertainty(y: np.ndarray, score: np.ndarray, rng: np.random.Generator, repetitions: int) -> dict:
    prevalence = float(y.mean())
    auroc = float(roc_auc_score(y, score))
    ap = float(average_precision_score(y, score))
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    auc_samples = np.empty(repetitions, dtype=float)
    ap_lift_samples = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        indices = np.concatenate(
            (
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            )
        )
        sample_y = y[indices]
        sample_score = score[indices]
        auc_samples[iteration] = roc_auc_score(sample_y, sample_score)
        ap_lift_samples[iteration] = average_precision_score(sample_y, sample_score) - prevalence
    permutations = max(repetitions, 10000)
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(y)
        exceed += roc_auc_score(shuffled, score) >= auroc
    order = np.argsort(-score, kind="stable")
    k = max(1, int(math.ceil(0.2 * len(y))))
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "auroc": auroc,
        "auroc_ci_low": float(np.quantile(auc_samples, 0.025)),
        "auroc_ci_high": float(np.quantile(auc_samples, 0.975)),
        "average_precision": ap,
        "ap_lift": float(ap - prevalence),
        "ap_lift_ci_low": float(np.quantile(ap_lift_samples, 0.025)),
        "ap_lift_ci_high": float(np.quantile(ap_lift_samples, 0.975)),
        "permutation_p_auroc_gt_half": float((exceed + 1) / (permutations + 1)),
        "top20_error_rate": float(y[order[:k]].mean()),
        "bottom20_error_rate": float(y[order[-k:]].mean()),
    }


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.input / "object_scores_5cm.csv")
    nonspatial_residual = residualize(rows, "score__nonspatial_owner_label_entropy")
    for row in rows:
        row["score__nonspatial_owner_label_entropy_size_residual"] = nonspatial_residual[
            (row["scene"], int(row["object_index"]))
        ]

    selections = [
        ("strict unified", "target__sidecar_strict_unified", "frame-balanced voxel disagreement", "score__framebalanced_mean_disagreement"),
        ("strict unified", "target__sidecar_strict_unified", "frame-balanced voxel disagreement | size residual", "score__framebalanced_mean_disagreement_size_residual"),
        ("strict unified", "target__sidecar_strict_unified", "nonspatial label entropy", "score__nonspatial_owner_label_entropy"),
        ("strict unified", "target__sidecar_strict_unified", "nonspatial label entropy | size residual", "score__nonspatial_owner_label_entropy_size_residual"),
        ("strict unified", "target__sidecar_strict_unified", "number of detections", "score__size_num_detections"),
        ("conservative mask", "target__sidecar_conservative_mask", "frame-balanced voxel disagreement", "score__framebalanced_mean_disagreement"),
        ("conservative mask", "target__sidecar_conservative_mask", "frame-balanced voxel disagreement | size residual", "score__framebalanced_mean_disagreement_size_residual"),
        ("strict association", "target__sidecar_strict_association", "second-label 3D region | size residual", "score__voxel_second_region_size_residual"),
        ("strict association", "target__sidecar_strict_association", "mean owner entropy | size residual", "score__mean_owner_entropy_size_residual"),
        ("conservative association", "target__sidecar_conservative_association", "second-label 3D region | size residual", "score__voxel_second_region_size_residual"),
        ("conservative association", "target__sidecar_conservative_association", "mean owner entropy | size residual", "score__mean_owner_entropy_size_residual"),
    ]
    rng = np.random.default_rng(args.seed)
    output = []
    for task, target, display_score, score in selections:
        for scene in ("room0", "office0"):
            y, values = selected_arrays(rows, scene, target, score)
            if len(np.unique(y)) != 2:
                continue
            output.append(
                {
                    "scene": scene,
                    "task": task,
                    "score": display_score,
                    **uncertainty(y, values, rng, args.repetitions),
                }
            )
    write_csv(args.output / "selected_uncertainty.csv", output)
    write_csv(args.output / "object_scores_5cm_with_nonspatial_residual.csv", rows)

    plotted = [
        row
        for row in output
        if row["task"] in {"strict unified", "conservative association"}
        and row["score"]
        in {
            "frame-balanced voxel disagreement",
            "frame-balanced voxel disagreement | size residual",
            "second-label 3D region | size residual",
            "mean owner entropy | size residual",
            "nonspatial label entropy | size residual",
            "number of detections",
        }
    ]
    labels = []
    values = []
    lows = []
    highs = []
    colors = []
    for row in plotted:
        labels.append(f"{row['scene']}\n{row['task']}\n{row['score']}")
        values.append(row["auroc"])
        lows.append(row["auroc"] - row["auroc_ci_low"])
        highs.append(row["auroc_ci_high"] - row["auroc"])
        colors.append("#3973ac" if row["scene"] == "room0" else "#d65f5f")
    fig, axis = plt.subplots(figsize=(16, 8))
    x = np.arange(len(labels))
    axis.bar(x, values, color=colors)
    axis.errorbar(x, values, yerr=np.asarray([lows, highs]), fmt="none", color="black", capsize=4)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("AUROC with stratified bootstrap 95% interval")
    axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    axis.set_title("Voxel-trigger re-audit: size confounding and association-specific signal")
    fig.tight_layout()
    fig.savefig(args.output / "02_selected_uncertainty.png", dpi=180)
    plt.close(fig)
    (args.output / "READY").write_text("ready\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
