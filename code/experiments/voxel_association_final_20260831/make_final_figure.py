#!/usr/bin/env python3
"""Render the preregistered final STOP decision figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCENES = ["room0", "office0", "room1", "office1"]
COLORS = ["#4C78A8", "#72B7B2", "#F58518", "#BAB0AC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def selected_rows(root: Path) -> dict[str, dict]:
    output = {}
    for split in ("dev", "holdout"):
        for row in read_csv(root / split / "event_metrics.csv"):
            if row["event_family"] == "merge" and row["feature"] == "risk_near_threshold":
                output[row["scene"]] = row
    return output


def create_counts(root: Path) -> dict[str, tuple[int, int]]:
    counts = {}
    for split in ("dev", "holdout"):
        path = root / split / "event_records.csv"
        rows = read_csv(path)
        for scene in SCENES:
            selected = [row for row in rows if row["scene"] == scene and row["event_family"] == "create"]
            if selected:
                counts[scene] = (len(selected), sum(int(row["is_error"]) for row in selected))
            elif any(row["scene"] == scene for row in rows):
                counts[scene] = (0, 0)
    return counts


def main() -> int:
    args = parse_args()
    rows = selected_rows(args.results_root)
    creates = create_counts(args.results_root)
    auc = [as_float(rows[scene].get("auroc")) for scene in SCENES]
    lift = [as_float(rows[scene].get("ap_lift")) for scene in SCENES]
    positives = [int(rows[scene]["positives"]) for scene in SCENES]
    totals = [int(rows[scene]["n"]) for scene in SCENES]
    negatives = [total - positive for total, positive in zip(totals, positives)]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    x = np.arange(len(SCENES))

    ax = axes[0, 0]
    for i, value in enumerate(auc):
        if value is None:
            ax.bar(i, 0, color=COLORS[i], alpha=0.25)
            ax.text(i, 0.53, "N/A\n0 positives", ha="center", va="bottom", fontweight="bold")
        else:
            ax.bar(i, value, color=COLORS[i])
            ax.text(i, value + 0.018, f"{value:.3f}", ha="center", va="bottom")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1, label="random = 0.50")
    ax.axhline(0.7, color="#C44E52", linestyle=":", linewidth=1.5, label="GO = 0.70")
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x, SCENES)
    ax.set_ylabel("AUROC")
    ax.set_title("A. Frozen event score: distance to threshold")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[0, 1]
    for i, value in enumerate(lift):
        if value is None:
            ax.bar(i, 0, color=COLORS[i], alpha=0.25)
            ax.text(i, 0.012, "N/A", ha="center", va="bottom", fontweight="bold")
        else:
            ax.bar(i, value, color=COLORS[i])
            ax.text(i, value + 0.004, f"{value:+.3f}", ha="center", va="bottom")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.axhline(0.1, color="#C44E52", linestyle=":", linewidth=1.5, label="GO = +0.10")
    ax.set_ylim(-0.01, 0.13)
    ax.set_xticks(x, SCENES)
    ax.set_ylabel("AP lift over prevalence")
    ax.set_title("B. Increment beyond error base rate")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1, 0]
    y = np.arange(len(SCENES))
    ax.barh(y, negatives, color="#B9C2CC", label="correct merge")
    ax.barh(y, positives, left=negatives, color="#D95F5F", label="wrong merge")
    for i, (n, p) in enumerate(zip(totals, positives)):
        ax.text(n + max(totals) * 0.012, i, f"{p}/{n} errors", va="center")
    ax.set_yticks(y, SCENES)
    ax.invert_yaxis()
    ax.set_xlabel("Strictly evaluable MERGE events")
    ax.set_title("C. HOLDOUT office1 has no positive event")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1, 1]
    ax.axis("off")
    create_text = " / ".join(f"{scene} {creates.get(scene, (0, 0))[1]}/{creates.get(scene, (0, 0))[0]}" for scene in SCENES)
    lines = [
        ("FINAL: STOP", 0.96, 17, "#B22222", "bold"),
        ("Do not build V→A or connect it to online tickets.", 0.82, 11, "#222222", "bold"),
        ("Why", 0.66, 12, "#222222", "bold"),
        ("• DEV signal was marginal (worst AUC 0.557; AP lift +0.004).", 0.56, 10, "#222222", "normal"),
        ("• Frozen room1 result remained weak (AUC 0.611; AP lift +0.026).", 0.45, 10, "#222222", "normal"),
        ("• office1 had 0/109 wrong MERGE events: ranking is unevaluable.", 0.34, 10, "#222222", "normal"),
        (f"• CREATE error/evaluable: {create_text}.", 0.23, 10, "#222222", "normal"),
        ("Keep voxels for indexing/scope/provenance; keep traces for audit only.", 0.08, 10, "#1F5A85", "bold"),
    ]
    for text, ypos, size, color, weight in lines:
        ax.text(0.02, ypos, text, transform=ax.transAxes, fontsize=size, color=color, fontweight=weight, va="top")
    ax.set_title("D. Preregistered decision", loc="left")

    fig.suptitle("Final validation: voxel persistence × association uncertainty", fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "status": "STOP"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
