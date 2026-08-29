#!/usr/bin/env python3
"""Render a reproducible two-scene Error Atlas from frozen audit outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "oracle"))
from evaluate_geometry_semantics import load_map, sha256_file  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sampled_points(map_path: Path, scores: dict[int, float], seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points, identities, quality = [], [], []
    for index, obj in enumerate(load_map(map_path)["objects"]):
        xyz = np.asarray(obj["pcd_np"], dtype=np.float64)
        if len(xyz) > 1200:
            xyz = xyz[rng.choice(len(xyz), 1200, replace=False)]
        points.append(xyz)
        identities.append(np.full(len(xyz), index))
        quality.append(np.full(len(xyz), scores.get(index, 0.0)))
    return np.concatenate(points), np.concatenate(identities), np.concatenate(quality)


def scene_figures(scene: str, source: Path, output: Path) -> dict:
    summary = json.loads((source / "audit_summary.json").read_text(encoding="utf-8"))
    matrices = np.load(source / "overlap_matrices.npz")
    pred_rows = read_csv(source / "predicted_object_summary.csv")
    gt_rows = read_csv(source / "gt_instance_summary.csv")
    pred_scores = {int(row["predicted_index"]): float(row["maximum_purity"]) for row in pred_rows}
    gt_scores = {int(row["gt_index"]): float(row["maximum_coverage"]) for row in gt_rows}
    pred_xyz, pred_id, pred_quality = sampled_points(Path(summary["prediction_map"]), pred_scores, 11)
    gt_xyz, gt_id, gt_quality = sampled_points(Path(summary["gt_map"]), gt_scores, 17)
    lo = np.minimum(pred_xyz[:, [0, 2]].min(axis=0), gt_xyz[:, [0, 2]].min(axis=0))
    hi = np.maximum(pred_xyz[:, [0, 2]].max(axis=0), gt_xyz[:, [0, 2]].max(axis=0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    panels = [
        (gt_xyz, gt_id, "GT observable instance IDs", "tab20", None),
        (pred_xyz, pred_id, "ali-dev predicted object IDs", "tab20", None),
        (pred_xyz, pred_quality, "Predicted-object maximum purity", "magma", (0, 1)),
        (gt_xyz, gt_quality, "GT-instance maximum coverage", "viridis", (0, 1)),
    ]
    for ax, (xyz, color, title, cmap, limits) in zip(axes.flat, panels):
        kwargs = {"s": 0.7, "c": color, "cmap": cmap, "rasterized": True}
        if limits:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        image = ax.scatter(xyz[:, 0], xyz[:, 2], **kwargs)
        ax.set(title=title, xlabel="x (m)", ylabel="z (m)", xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
        ax.set_aspect("equal")
        if limits:
            fig.colorbar(image, ax=ax, fraction=0.045, label="score")
    overview = output / f"{scene}_overview.png"
    fig.savefig(overview, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    image = ax.imshow(matrices["iou"].T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set(title=f"{scene}: Predicted Object ↔ GT Instance voxel IoU (5 cm)", xlabel="predicted object matrix index", ylabel="GT instance matrix index")
    fig.colorbar(image, ax=ax, label="voxel IoU")
    heatmap = output / f"{scene}_overlap_heatmap.png"
    fig.savefig(heatmap, dpi=170)
    plt.close(fig)

    purities = np.asarray([float(row["maximum_purity"]) for row in pred_rows])
    coverages = np.asarray([float(row["maximum_coverage"]) for row in gt_rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].hist(purities, bins=np.linspace(0, 1, 21), color="#b23a48"); axes[0].set(title="Maximum purity per prediction", xlabel="purity", ylabel="objects", xlim=(0, 1))
    axes[1].hist(coverages, bins=np.linspace(0, 1, 21), color="#287271"); axes[1].set(title="Maximum coverage per GT", xlabel="coverage", ylabel="instances", xlim=(0, 1))
    distributions = output / f"{scene}_distributions.png"
    fig.savefig(distributions, dpi=170)
    plt.close(fig)
    return {"summary": summary, "overview": overview.name, "heatmap": heatmap.name, "distributions": distributions.name}


def table(rows: list[dict], fields: list[str]) -> str:
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matching = args.matching_root.resolve()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    rendered = {scene: scene_figures(scene, matching / scene / "voxel0p05", output) for scene in ("room0", "office0")}
    sections = []
    for scene, item in rendered.items():
        summary = item["summary"]
        pred_fields = ["predicted_index", "predicted_label", "maximum_purity", "best_gt_instance_id", "best_gt_label", "overlap_gt_count_purity_0p01"]
        gt_fields = ["gt_instance_id", "gt_label", "maximum_coverage", "best_predicted_index", "best_predicted_label", "overlap_prediction_count_coverage_0p01"]
        sections.append(f"<section><h2>{scene}</h2><p>Frozen maps: {summary['counts']['predicted_objects']} predictions, {summary['counts']['observable_gt_instances']} observable GT instances.</p><img src='{item['overview']}'><img src='{item['heatmap']}'><img src='{item['distributions']}'><h3>Worst prediction purity (Top 20)</h3>{table(summary['tails']['worst_prediction_purity'], pred_fields)}<h3>Worst GT coverage (Top 20)</h3>{table(summary['tails']['worst_gt_coverage'], gt_fields)}<h3>Most contaminated predictions</h3>{table(summary['tails']['most_contaminated_predictions'], pred_fields)}<h3>Most fragmented GT instances</h3>{table(summary['tails']['most_fragmented_gt'], gt_fields)}</section>")
    document = "<!doctype html><html><head><meta charset='utf-8'><title>Error Atlas</title><style>body{font-family:Arial,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#222}img{width:100%;margin:10px 0;border:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px}th,td{border:1px solid #ddd;padding:5px;text-align:left}th{background:#f2f2f2}h2{margin-top:50px}</style></head><body><h1>Task-Weighted Error Impact Audit: Error Atlas</h1><p>Continuous measurements at 5 cm. Colors and Top-20 lists are diagnostic views, not frozen error labels.</p>" + "".join(sections) + "</body></html>"
    atlas = output / "error_atlas.html"; atlas.write_text(document, encoding="utf-8")
    manifest = {"schema_version": "1.0.0", "atlas": str(atlas), "files": {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "atlas_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"atlas": str(atlas), "files": sorted(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
