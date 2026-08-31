#!/usr/bin/env python3
"""Remove repeated-error support bias with fixed-support sensitivity grids."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reaudit", type=Path, required=True)
    parser.add_argument("--module-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def binary(y: np.ndarray, score: np.ndarray) -> dict:
    prevalence = float(y.mean())
    if len(np.unique(y)) != 2:
        return {
            "n": int(len(y)),
            "positives": int(y.sum()),
            "prevalence": prevalence,
            "auroc": None,
            "average_precision": None,
            "ap_lift": None,
        }
    ap = float(average_precision_score(y, score))
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": ap,
        "ap_lift": float(ap - prevalence),
    }


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.module_root.resolve()))
    import reaudit_voxel_trigger as rv

    args.output.mkdir(parents=True, exist_ok=True)
    score_rows = read_csv(
        args.reaudit / "statistical" / "object_scores_5cm_with_nonspatial_residual.csv"
    )
    scores_by_scene = {
        scene: {int(row["object_index"]): row for row in score_rows if row["scene"] == scene}
        for scene in ("room0", "office0")
    }
    score_fields = {
        "voxel_second_region_residual": "score__voxel_second_region_size_residual",
        "voxel_owner_entropy_residual": "score__mean_owner_entropy_size_residual",
        "framebalanced_disagreement_residual": "score__framebalanced_mean_disagreement_size_residual",
        "nonspatial_label_entropy_residual": "score__nonspatial_owner_label_entropy_size_residual",
        "num_detections": "score__size_num_detections",
    }
    metric_rows: list[dict] = []
    correlation_rows: list[dict] = []
    support_rows: list[dict] = []
    object_audits: list[dict] = []
    for scene in ("room0", "office0"):
        scene_root = args.source_root / "full" / scene
        observations = rv.read_jsonl(scene_root / "observations.jsonl")
        object_rows = rv.read_jsonl(
            scene_root / "voxel_0p050" / "all_history" / "objects.jsonl"
        )
        audits, _ = rv.build_sidecar_targets(observations, object_rows, strict=True)
        foreground = [int(row["object_index"]) for row in object_rows if not row["is_background"]]
        for index in foreground:
            object_audits.append(
                {
                    "scene": scene,
                    "object_index": index,
                    **audits[index],
                }
            )
        for support in (3, 5, 10):
            association_indices = [index for index in foreground if audits[index]["pure"] >= support]
            mask_indices = [index for index in foreground if audits[index]["eligible"] >= support]
            support_rows.extend(
                [
                    {
                        "scene": scene,
                        "family": "association",
                        "minimum_support": support,
                        "n": len(association_indices),
                        "fraction_ge_0p1": int(sum(audits[index]["wrong_fraction"] >= 0.1 for index in association_indices)),
                    },
                    {
                        "scene": scene,
                        "family": "mask",
                        "minimum_support": support,
                        "n": len(mask_indices),
                        "fraction_ge_0p1": int(sum(audits[index]["mixed"] / max(audits[index]["eligible"], 1) >= 0.1 for index in mask_indices)),
                    },
                ]
            )
            for family, indices, fraction_field in (
                ("association", association_indices, "wrong_fraction"),
                ("mask", mask_indices, None),
            ):
                if not indices:
                    continue
                fractions = np.asarray(
                    [
                        audits[index][fraction_field]
                        if fraction_field
                        else audits[index]["mixed"] / max(audits[index]["eligible"], 1)
                        for index in indices
                    ],
                    dtype=float,
                )
                for score_name, score_field in score_fields.items():
                    values = np.asarray(
                        [float(scores_by_scene[scene][index][score_field]) for index in indices],
                        dtype=float,
                    )
                    rho, p_value = spearmanr(values, fractions)
                    correlation_rows.append(
                        {
                            "scene": scene,
                            "family": family,
                            "minimum_support": support,
                            "score": score_name,
                            "n": len(indices),
                            "spearman_rho": float(rho),
                            "spearman_p": float(p_value),
                        }
                    )
                    for threshold in (0.1, 0.2, 0.3, 0.4):
                        labels = fractions >= threshold
                        metric_rows.append(
                            {
                                "scene": scene,
                                "family": family,
                                "minimum_support": support,
                                "error_fraction_threshold": threshold,
                                "score": score_name,
                                **binary(labels.astype(np.uint8), values),
                            }
                        )
    write_csv(args.output / "support_counts.csv", support_rows)
    write_csv(args.output / "support_controlled_metrics.csv", metric_rows)
    write_csv(args.output / "fraction_correlations.csv", correlation_rows)
    write_csv(args.output / "strict_object_audits.csv", object_audits)
    (args.output / "READY").write_text("ready\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
