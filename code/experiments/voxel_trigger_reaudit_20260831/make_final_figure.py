#!/usr/bin/env python3
"""Create the final compact re-audit figure from frozen CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def value(data: list[dict], **query: object) -> float:
    matched = [row for row in data if all(str(row[key]) == str(expected) for key, expected in query.items())]
    if len(matched) != 1:
        raise ValueError((query, len(matched)))
    return float(matched[0]["auroc"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    metrics = rows(args.root / "metrics.csv")
    construction = rows(args.root / "construction_audit.csv")
    support = rows(args.root / "support_controlled" / "support_controlled_metrics.csv")
    uncertainty = rows(args.root / "statistical" / "selected_uncertainty.csv")

    scenes = ["room0", "office0"]
    colors = {"room0": "#377eb8", "office0": "#e15759"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axis = axes[0, 0]
    x = np.arange(2)
    same_frame = []
    multilabel = []
    for scene in scenes:
        row = next(r for r in construction if r["scene"] == scene and float(r["scale"]) == 0.05)
        same_frame.append(float(row["same_frame_extra_link_fraction"]))
        multilabel.append(float(row["voxels_with_same_frame_multilabel_fraction"]))
    axis.bar(x - 0.18, same_frame, 0.36, label="extra links from same frame")
    axis.bar(x + 0.18, multilabel, 0.36, label="voxels with same-frame multi-label")
    axis.set_xticks(x, scenes)
    axis.set_ylim(0, 0.6)
    axis.set_ylabel("fraction")
    axis.set_title("A. Original observation votes are correlated")
    axis.legend(fontsize=9)

    axis = axes[0, 1]
    score_specs = [
        ("frame-balanced\nvoxel", "framebalanced_mean_disagreement"),
        ("voxel after\nsize control", "framebalanced_mean_disagreement_size_residual"),
        ("nonspatial\nlabel entropy", "nonspatial_owner_label_entropy"),
        ("nonspatial after\nsize control", "nonspatial_owner_label_entropy_size_residual"),
        ("observation\ncount only", "size_num_detections"),
    ]
    x = np.arange(len(score_specs))
    width = 0.36
    for offset, scene in enumerate(scenes):
        vals = []
        for _, score in score_specs:
            if score == "nonspatial_owner_label_entropy_size_residual":
                match = next(
                    row
                    for row in uncertainty
                    if row["scene"] == scene
                    and row["task"] == "strict unified"
                    and row["score"] == "nonspatial label entropy | size residual"
                )
                vals.append(float(match["auroc"]))
            else:
                vals.append(
                    value(
                        metrics,
                        scene=scene,
                        scale=0.05,
                        target="sidecar_strict_unified",
                        score=score,
                    )
                )
        axis.bar(x + (offset - 0.5) * width, vals, width, label=scene, color=colors[scene])
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1.05)
    axis.set_xticks(x, [label for label, _ in score_specs], fontsize=9)
    axis.set_ylabel("AUROC")
    axis.set_title("B. Unified/mask signal is dominated by support size")
    axis.legend()

    axis = axes[1, 0]
    for scene in scenes:
        vals = [
            value(
                metrics,
                scene=scene,
                scale=scale,
                target="sidecar_conservative_association",
                score="voxel_second_region_size_residual",
            )
            for scale in (0.025, 0.05, 0.1)
        ]
        axis.plot([2.5, 5, 10], vals, marker="o", linewidth=2, label=scene, color=colors[scene])
        for x_value, y_value in zip([2.5, 5, 10], vals):
            axis.text(x_value, y_value + 0.018, f"{y_value:.2f}", ha="center", fontsize=9)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0.45, 0.95)
    axis.set_xticks([2.5, 5, 10])
    axis.set_xlabel("voxel size (cm)")
    axis.set_ylabel("AUROC")
    axis.set_title("C. Second-label 3D region retains association signal")
    axis.legend()

    axis = axes[1, 1]
    matrix = np.zeros((3, 4), dtype=float)
    for row_index, minimum_support in enumerate((3, 5, 10)):
        for column_index, threshold in enumerate((0.1, 0.2, 0.3, 0.4)):
            scene_values = []
            for scene in scenes:
                matched = [
                    row
                    for row in support
                    if row["scene"] == scene
                    and row["family"] == "association"
                    and int(row["minimum_support"]) == minimum_support
                    and float(row["error_fraction_threshold"]) == threshold
                    and row["score"] == "voxel_second_region_residual"
                ]
                scene_values.append(float(matched[0]["auroc"]))
            matrix[row_index, column_index] = min(scene_values)
    image = axis.imshow(matrix, vmin=0.45, vmax=0.85, cmap="YlGnBu", aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center")
    axis.set_xticks(np.arange(4), ["0.10", "0.20", "0.30", "0.40"])
    axis.set_yticks(np.arange(3), ["≥3", "≥5", "≥10"])
    axis.set_xlabel("wrong-association fraction threshold")
    axis.set_ylabel("minimum pure observations")
    axis.set_title("D. Worst-scene AUROC under support-controlled GT")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Simple voxel trigger re-audit: reject the unified score, retain a narrow association hypothesis", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.root / "03_final_reaudit_decision.png", dpi=190)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
