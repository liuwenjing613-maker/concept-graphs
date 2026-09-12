from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
from PIL import Image, ImageDraw


REVIEW_STATUSES = ("CLEAN", "MIXED", "UNSCORABLE")
SCOPE_MASK_QUALITY = "MASK_QUALITY"
SCOPE_FALSE_DISCARD_CLEAN = "FALSE_DISCARD_CLEAN"
SCOPE_CORRECT_DISCARD_MIXED = "CORRECT_DISCARD_MIXED"
SCOPE_VLM_USABLE_GT_MIXED = "VLM_USABLE_GT_MIXED"
REVIEW_SCOPES = (
    SCOPE_VLM_USABLE_GT_MIXED,
    SCOPE_FALSE_DISCARD_CLEAN,
    SCOPE_CORRECT_DISCARD_MIXED,
    SCOPE_MASK_QUALITY,
)
STATUS_COLORS = {
    "CLEAN": np.asarray([62, 205, 145], dtype=np.float32),
    "MIXED": np.asarray([255, 171, 64], dtype=np.float32),
    "UNSCORABLE": np.asarray([229, 105, 161], dtype=np.float32),
}
MASK_BOX_COLOR = (0, 255, 96)
INDEX_FIELDS = (
    "obs_uid",
    "frame_uid",
    "frame_index",
    "source_frame_id",
    "filtered_det_idx",
    "class_name",
    "quality_status",
    "quality_reason",
    "mask_area",
    "valid_depth_ratio",
    "matched_gt_points",
    "gt_support_ratio",
    "top1_gt_instance",
    "top1_gt_class",
    "top1_purity",
    "top2_gt_instance",
    "top2_purity",
    "purity_margin",
    "core_matched_gt_points",
    "core_top1_gt_instance",
    "core_top1_purity",
    "core_agrees_with_full_mask",
    "gt_instance_counts",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_bounds(mask: np.ndarray, padding_ratio: float = 0.12) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    if not len(rows):
        return 0, 0, width, height
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    padding = max(24, int(max(x1 - x0, y1 - y0) * padding_ratio))
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def thicken(binary: np.ndarray, radius: int = 2) -> np.ndarray:
    thick = np.asarray(binary, dtype=bool).copy()
    height, width = thick.shape
    source = thick.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if not dx and not dy:
                continue
            y0, y1 = max(0, dy), min(height, height + dy)
            x0, x1 = max(0, dx), min(width, width + dx)
            thick[y0:y1, x0:x1] |= source[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return thick


def render_rgb_mask(
    rgb_path: Path,
    mask: np.ndarray,
    *,
    status: str,
    view: str,
    max_side: int,
) -> tuple[bytes, tuple[int, int]]:
    with Image.open(rgb_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if rgb.shape[:2] != mask.shape:
        raise ValueError(
            f"RGB/mask shape mismatch: rgb={rgb.shape[:2]}, mask={mask.shape}"
        )
    output = rgb.copy()
    color = np.asarray(MASK_BOX_COLOR, dtype=np.float32)
    output[mask] = np.clip(
        0.95 * output[mask].astype(np.float32) + 0.05 * color,
        0,
        255,
    ).astype(np.uint8)
    boundary = thicken(mask_boundary(mask), radius=2)
    output[boundary] = np.asarray(MASK_BOX_COLOR, dtype=np.uint8)
    image = Image.fromarray(output, mode="RGB")
    rows, cols = np.nonzero(mask)
    if len(rows):
        rectangle = (
            int(cols.min()),
            int(rows.min()),
            int(cols.max()),
            int(rows.max()),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(rectangle, outline=(0, 0, 0), width=7)
        draw.rectangle(rectangle, outline=MASK_BOX_COLOR, width=4)
    if view == "crop":
        image = image.crop(mask_bounds(mask))
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        size = tuple(max(1, int(round(value * scale))) for value in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=76, method=4)
    return buffer.getvalue(), image.size


class ReviewDataset:
    def __init__(
        self,
        run_dir: Path,
        metrics_dir: Path,
        *,
        verify_hashes: bool = False,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.metrics_dir = metrics_dir.resolve()
        self.verify_hashes = bool(verify_hashes)
        self._verified_paths: set[Path] = set()
        gate_config_path = self.run_dir / "blocking_association_gate/config.json"
        self.gate_config = (
            json.loads(gate_config_path.read_text(encoding="utf-8"))
            if gate_config_path.is_file()
            else {}
        )

        cohort_events: dict[str, dict[str, Any]] = {}
        event_path = self.metrics_dir / "event_decisions.jsonl"
        if event_path.is_file():
            for event in iter_jsonl(event_path):
                obs_uid = str(event.get("obs_uid"))
                status = str(event.get("quality_status"))
                final_result = dict(event.get("final_result") or {})
                final_category = str(final_result.get("category") or "")
                scopes = []
                if status == "CLEAN" and final_category == "FALSE_DISCARD":
                    scopes.append(SCOPE_FALSE_DISCARD_CLEAN)
                elif status == "MIXED" and final_category == "CORRECT_DISCARD":
                    scopes.append(SCOPE_CORRECT_DISCARD_MIXED)
                mixed_not_discarded = status == "MIXED" and final_category in {
                    "MIXED_MASK_ATTACHMENT",
                    "MIXED_MASK_NEW",
                }
                if not scopes and not mixed_not_discarded:
                    continue
                cohort_events[obs_uid] = {
                    "event_review_scopes": scopes,
                    "mixed_not_discarded_candidate": mixed_not_discarded,
                    "baseline_category": (event.get("baseline_result") or {}).get("category"),
                    "final_category": final_category,
                    "changed": event.get("changed"),
                    "decision_source": event.get("decision_source"),
                    "route_reason": event.get("route_reason"),
                    "event_sequence": event.get("event_sequence"),
                }

        gate_path = self.run_dir / "blocking_association_gate/events.jsonl"
        if gate_path.is_file():
            for gate in iter_jsonl(gate_path):
                obs_uid = str(gate.get("current_observation_uid"))
                event = cohort_events.get(obs_uid)
                if not event or not event.get("mixed_not_discarded_candidate"):
                    continue
                quality_path = (
                    self.run_dir
                    / "blocking_association_gate/events"
                    / str(gate.get("event_id"))
                    / "quality/result.json"
                )
                quality = (
                    json.loads(quality_path.read_text(encoding="utf-8"))
                    if quality_path.is_file()
                    else {}
                )
                quality_value = dict(quality.get("value") or {})
                audit = dict(gate.get("audit_scores_hidden_from_vlm") or {})
                iou = dict(gate.get("candidate_iou_prefilter_hidden_from_vlm") or {})
                trigger = dict(gate.get("trigger") or {})
                event.update(
                    {
                        "gate_event_found": True,
                        "gate_event_id": gate.get("event_id"),
                        "gate_schema": gate.get("schema_version"),
                        "gate_mode": gate.get("mode"),
                        "gate_trigger_kind": trigger.get("kind"),
                        "gate_trigger_reasons": trigger.get("reasons") or [],
                        "trigger_top1": trigger.get("top1"),
                        "trigger_top2": trigger.get("top2"),
                        "trigger_margin": trigger.get("margin"),
                        "trigger_threshold_distance": trigger.get("threshold_distance"),
                        "candidate_scores": audit.get("candidate_scores") or {},
                        "sim_threshold": audit.get("sim_threshold"),
                        "margin_threshold": audit.get("margin_threshold"),
                        "threshold_distance": audit.get("threshold_distance"),
                        "candidate_iou_threshold": iou.get("iou_threshold"),
                        "candidate_count": len(gate.get("candidate_alias_to_object_index") or {}),
                        "vlm_quality_status": quality_value.get("status"),
                        "vlm_quality_reason": quality_value.get("reason"),
                        "vlm_quality_error": quality.get("error"),
                        "quality_attempt_count": len(quality.get("attempts") or []),
                        "quality_timeout_count": quality.get("timeout_count"),
                        "model_choice": (gate.get("model_output") or {}).get("choice"),
                    }
                )
                if quality_value.get("status") == "USABLE" and not quality.get("error"):
                    event["event_review_scopes"].append(SCOPE_VLM_USABLE_GT_MIXED)

        labels: dict[str, dict[str, Any]] = {}
        labels_path = self.metrics_dir / "observation_labels.jsonl"
        for row in iter_jsonl(labels_path):
            obs_uid = str(row["obs_uid"])
            status = str(row.get("quality_status"))
            if status not in {"MIXED", "UNSCORABLE"} and obs_uid not in cohort_events:
                continue
            compact = {key: row.get(key) for key in INDEX_FIELDS}
            cohorts = []
            if status in {"MIXED", "UNSCORABLE"}:
                cohorts.append(SCOPE_MASK_QUALITY)
            if obs_uid in cohort_events:
                cohorts.extend(cohort_events[obs_uid].get("event_review_scopes") or [])
                compact.update(cohort_events[obs_uid])
            compact["review_scopes"] = cohorts
            labels[obs_uid] = compact
        if not labels:
            raise ValueError(f"no matching observations in {labels_path}")

        needed = set(labels)
        observations: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(self.run_dir / "evidence/observations.jsonl"):
            obs_uid = str(row.get("obs_uid"))
            if obs_uid not in needed:
                continue
            observations[obs_uid] = {
                "processed_mask_ref": row.get("processed_mask_ref"),
                "bbox_2d": row.get("bbox_2d"),
                "confidence": row.get("confidence"),
                "processed_mask_area": row.get("processed_mask_area"),
                "removed_pixel_count": row.get("removed_pixel_count"),
            }
        missing = sorted(needed - set(observations))
        if missing:
            raise ValueError(f"missing observation evidence for {len(missing)} rows")

        needed_frames = {str(row["frame_uid"]) for row in labels.values()}
        frames: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(self.run_dir / "evidence/frames.jsonl"):
            frame_uid = str(row.get("frame_uid"))
            if frame_uid in needed_frames:
                frames[frame_uid] = {"rgb_ref": row.get("rgb_ref")}
        missing_frames = sorted(needed_frames - set(frames))
        if missing_frames:
            raise ValueError(f"missing frame evidence for {len(missing_frames)} rows")

        self.frames = frames
        self.rows = []
        for obs_uid, label in labels.items():
            row = dict(label)
            row.update(observations[obs_uid])
            self.rows.append(row)
        self.rows.sort(
            key=lambda row: (
                str(row["quality_status"]),
                str(row.get("discard_cohort") or ""),
                str(row["quality_reason"]),
                int(row.get("frame_index") or 0),
                int(row.get("filtered_det_idx") or 0),
                str(row["obs_uid"]),
            )
        )
        self.by_uid = {str(row["obs_uid"]): row for row in self.rows}

    def _resolve(self, ref: Mapping[str, Any]) -> Path:
        path = (self.run_dir / str(ref["path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = ref.get("sha256")
        if self.verify_hashes and expected and path not in self._verified_paths:
            actual = sha256_file(path)
            if actual != str(expected):
                raise ValueError(f"evidence hash drift: {path}")
            self._verified_paths.add(path)
        return path

    def _mask(self, row: Mapping[str, Any]) -> np.ndarray:
        ref = dict(row.get("processed_mask_ref") or {})
        path = self._resolve(ref)
        with np.load(path, allow_pickle=False) as archive:
            key = str(ref.get("key") or "mask")
            if key not in archive:
                raise KeyError(f"{path} is missing mask key {key}")
            return np.ascontiguousarray(np.asarray(archive[key], dtype=bool))

    def render(self, obs_uid: str, *, view: str, max_side: int) -> tuple[bytes, tuple[int, int]]:
        row = self.by_uid[obs_uid]
        frame = self.frames[str(row["frame_uid"])]
        rgb_path = self._resolve(dict(frame["rgb_ref"]))
        return render_rgb_mask(
            rgb_path,
            self._mask(row),
            status=str(row["quality_status"]),
            view=view,
            max_side=max_side,
        )

    def summary(self) -> dict[str, Any]:
        statuses = Counter(str(row["quality_status"]) for row in self.rows)
        reasons = Counter(str(row["quality_reason"]) for row in self.rows)
        scopes = Counter(
            scope for row in self.rows for scope in row.get("review_scopes") or ()
        )
        return {
            "total": len(self.rows),
            "status_counts": dict(sorted(statuses.items())),
            "reason_counts": dict(sorted(reasons.items())),
            "scope_counts": {scope: scopes[scope] for scope in REVIEW_SCOPES},
            "scopes": list(REVIEW_SCOPES),
            "statuses": list(REVIEW_STATUSES),
            "reasons": sorted(reasons),
            "gate_config": {
                key: self.gate_config.get(key)
                for key in (
                    "schema_version",
                    "model",
                    "reasoning_effort",
                    "sim_threshold",
                    "margin_threshold",
                    "threshold_distance",
                    "threshold_scope",
                    "review_all_new",
                    "mask_change_enabled",
                    "support_window",
                    "support_min_history",
                    "support_reference_min",
                    "support_drop_threshold",
                    "association_top_k",
                    "create_top_k",
                    "candidate_iou_filter_enabled",
                    "candidate_iou_threshold",
                )
                if key in self.gate_config
            },
            "disk_policy": "on-demand RGB+mask WebP; rendered images are not written to disk",
        }

    def query(
        self,
        *,
        scope: str,
        status: str,
        reason: str,
        text: str,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        needle = text.strip().lower()
        filtered = []
        for row in self.rows:
            if scope not in (row.get("review_scopes") or ()):
                continue
            if status != "ALL" and row["quality_status"] != status:
                continue
            if reason != "ALL" and row["quality_reason"] != reason:
                continue
            searchable = " ".join(
                str(row.get(key) or "")
                for key in ("obs_uid", "frame_uid", "source_frame_id", "class_name")
            ).lower()
            if needle and needle not in searchable:
                continue
            filtered.append(row)
        total = len(filtered)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), pages)
        start = (page - 1) * page_size
        result_rows = []
        for row in filtered[start : start + page_size]:
            public = {key: row.get(key) for key in INDEX_FIELDS}
            public.update(
                {
                    "bbox_2d": row.get("bbox_2d"),
                    "confidence": row.get("confidence"),
                    "processed_mask_area": row.get("processed_mask_area"),
                    "removed_pixel_count": row.get("removed_pixel_count"),
                    "discard_cohort": row.get("discard_cohort"),
                    "baseline_category": row.get("baseline_category"),
                    "final_category": row.get("final_category"),
                    "changed": row.get("changed"),
                    "decision_source": row.get("decision_source"),
                    "route_reason": row.get("route_reason"),
                    "event_sequence": row.get("event_sequence"),
                    "gate_event_found": row.get("gate_event_found"),
                    "gate_event_id": row.get("gate_event_id"),
                    "gate_schema": row.get("gate_schema"),
                    "gate_mode": row.get("gate_mode"),
                    "gate_trigger_kind": row.get("gate_trigger_kind"),
                    "gate_trigger_reasons": row.get("gate_trigger_reasons"),
                    "trigger_top1": row.get("trigger_top1"),
                    "trigger_top2": row.get("trigger_top2"),
                    "trigger_margin": row.get("trigger_margin"),
                    "trigger_threshold_distance": row.get("trigger_threshold_distance"),
                    "candidate_scores": row.get("candidate_scores"),
                    "sim_threshold": row.get("sim_threshold"),
                    "margin_threshold": row.get("margin_threshold"),
                    "threshold_distance": row.get("threshold_distance"),
                    "candidate_iou_threshold": row.get("candidate_iou_threshold"),
                    "candidate_count": row.get("candidate_count"),
                    "vlm_quality_status": row.get("vlm_quality_status"),
                    "vlm_quality_reason": row.get("vlm_quality_reason"),
                    "vlm_quality_error": row.get("vlm_quality_error"),
                    "quality_attempt_count": row.get("quality_attempt_count"),
                    "quality_timeout_count": row.get("quality_timeout_count"),
                    "model_choice": row.get("model_choice"),
                    "image_crop": f"/image/{quote(str(row['obs_uid']))}.webp?view=crop",
                    "image_full": f"/image/{quote(str(row['obs_uid']))}.webp?view=full",
                }
            )
            result_rows.append(public)
        return {
            "rows": result_rows,
            "total": total,
            "page": page,
            "pages": pages,
            "page_size": page_size,
        }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Observation mask audit</title>
<style>
:root{color-scheme:dark;--bg:#0b1017;--panel:#121a24;--line:#2a394a;--text:#eaf1f8;--muted:#9fb0c1;--cyan:#32e6e2;--orange:#ffab40;--pink:#e569a1;--green:#3ecd91;--red:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI Variable","Microsoft YaHei UI",sans-serif}
header{position:sticky;top:0;z-index:4;background:rgba(11,16,23,.96);border-bottom:1px solid var(--line);padding:14px 20px}
h1{font-size:19px;margin:0 0 10px}.controls{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
select,input,button{background:#172230;color:var(--text);border:1px solid #34485e;border-radius:7px;padding:8px 10px}input{min-width:270px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}
.summary{margin-left:auto;color:var(--muted)}.runinfo{margin-top:10px;color:#b9c8d7;font-size:12px}.runinfo b{color:white}main{padding:18px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;align-items:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}.image{height:230px;background:#05080d;position:relative;overflow:hidden;cursor:zoom-in}
.image img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}.meta{padding:11px 13px}.topline{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.badge,.outcome,.vlm{font-weight:750;font-size:12px;padding:3px 7px;border-radius:999px}.CLEAN{background:rgba(62,205,145,.16);color:#79e8b8}.MIXED{background:rgba(255,171,64,.16);color:#ffc271}.UNSCORABLE{background:rgba(229,105,161,.16);color:#ff9dca}.outcome.BAD{background:rgba(255,107,107,.16);color:#ff9696}.outcome.GOOD{background:rgba(62,205,145,.16);color:#79e8b8}.vlm{background:rgba(255,107,107,.16);color:#ff9696}
.reason{color:var(--cyan);font-family:ui-monospace,monospace;font-size:12px}.uid{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted);overflow-wrap:anywhere;margin:7px 0}
.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 12px;color:#c4d1df;font-size:12px}.stats b{color:white}.gate{border-top:1px solid var(--line);margin-top:9px;padding-top:8px;color:#b9c8d7;font-size:12px}.gate b{color:white}.gate-reason{color:#ffb5b5}.hint{color:var(--muted);font-size:12px;margin-top:8px}.empty{padding:60px;text-align:center;color:var(--muted)}
@media(max-width:700px){input{min-width:100%}.summary{margin-left:0}.grid{grid-template-columns:1fr}main{padding:10px}}
</style>
</head>
<body><header><h1>Observation 掩码与 VLM 丢弃审计</h1><div class="controls">
<select id="scope"><option value="VLM_USABLE_GT_MIXED">GT 为 MIXED，VLM 判 USABLE</option><option value="FALSE_DISCARD_CLEAN">VLM 错误丢弃 CLEAN</option><option value="CORRECT_DISCARD_MIXED">VLM 正确丢弃 MIXED</option><option value="MASK_QUALITY">MIXED / UNSCORABLE 全集</option></select>
<select id="status"><option value="ALL">全部状态</option><option>CLEAN</option><option>MIXED</option><option>UNSCORABLE</option></select>
<select id="reason"><option value="ALL">全部原因</option></select>
<input id="search" placeholder="搜索 obs UID、帧或检测类别">
<button id="prev">上一页</button><button id="next">下一页</button>
<span class="summary" id="summary"></span></div><div class="runinfo" id="runinfo"></div></header>
<main><div class="grid" id="grid"></div></main>
<script>
let page=1; const size=10; const $=id=>document.getElementById(id);
const pct=v=>v==null?'—':(100*Number(v)).toFixed(1)+'%';
const val=v=>v==null?'—':v;
const num=v=>v==null?'—':Number(v).toFixed(3).replace(/0+$/,'').replace(/\.$/,'');
async function init(){const s=await fetch('/api/summary').then(r=>r.json());for(const x of s.reasons){const o=document.createElement('option');o.value=x;o.textContent=x;$('reason').append(o)}const labels={VLM_USABLE_GT_MIXED:'GT 为 MIXED，VLM 判 USABLE',FALSE_DISCARD_CLEAN:'VLM 错误丢弃 CLEAN',CORRECT_DISCARD_MIXED:'VLM 正确丢弃 MIXED',MASK_QUALITY:'MIXED / UNSCORABLE 全集'};for(const o of $('scope').options)o.textContent=`${labels[o.value]}（${s.scope_counts[o.value]}）`;const c=s.gate_config||{};$('runinfo').innerHTML=`<b>门控配置</b> · ${val(c.model)} · reasoning ${val(c.reasoning_effort)} · sim ${val(c.sim_threshold)} · margin&lt;${val(c.margin_threshold)} · create distance≤${val(c.threshold_distance)} · candidate IoU&gt;${val(c.candidate_iou_threshold)} 去重 · Top-K ${val(c.association_top_k)}/${val(c.create_top_k)} · all-new 复核 ${c.review_all_new?'on':'off'} · mask-change ${c.mask_change_enabled?'on':'off'}`;syncScope();load()}
function syncScope(){const scope=$('scope').value;const fixed=scope==='FALSE_DISCARD_CLEAN'?'CLEAN':(scope==='CORRECT_DISCARD_MIXED'||scope==='VLM_USABLE_GT_MIXED')?'MIXED':'ALL';$('status').value=fixed;$('status').disabled=scope!=='MASK_QUALITY'}
async function load(){const q=new URLSearchParams({scope:$('scope').value,status:$('status').value,reason:$('reason').value,q:$('search').value,page,size});const d=await fetch('/api/rows?'+q).then(r=>r.json());page=d.page;$('summary').textContent=`${d.total} 条 · 每页 10 张 · 第 ${d.page}/${d.pages} 页 · 高亮绿色框与轮廓标示 mask`;$('prev').disabled=page<=1;$('next').disabled=page>=d.pages;const g=$('grid');g.innerHTML='';if(!d.rows.length){g.innerHTML='<div class="empty">没有匹配条目</div>';return}for(const r of d.rows){const c=document.createElement('article');c.className='card';const counts=Object.entries(r.gt_instance_counts||{}).sort((a,b)=>Number(b[1])-Number(a[1])).slice(0,3).map(([k,v])=>`${k}:${v}`).join(' · ')||'—';const scores=Object.entries(r.candidate_scores||{}).map(([k,v])=>`${k}:${num(v)}`).join(' · ')||'—';const actual=[r.trigger_top1!=null?`top1 ${num(r.trigger_top1)}`:'',r.trigger_top2!=null?`top2 ${num(r.trigger_top2)}`:'',r.trigger_margin!=null?`margin ${num(r.trigger_margin)}`:'',r.trigger_threshold_distance!=null?`distance ${num(r.trigger_threshold_distance)}`:''].filter(Boolean).join(' · ')||'—';const outcome=r.final_category?`<span class="outcome ${r.final_category==='FALSE_DISCARD'?'BAD':'GOOD'}">${r.final_category}</span>`:'';const vlm=r.vlm_quality_status?`<span class="vlm">VLM ${r.vlm_quality_status}</span>`:'';const transition=r.final_category?`<br>动作判定 ${val(r.baseline_category)} → ${r.final_category}`:'';const gate=r.gate_event_found?`<div class="gate"><b>VLM 质量理由：</b><span class="gate-reason">${val(r.vlm_quality_reason)}</span><br><b>触发：</b>${val(r.gate_trigger_kind)} · ${(r.gate_trigger_reasons||[]).join(' + ')||'—'}<br><b>本例分数：</b>${actual}<br><b>候选分数：</b>${scores}<br><b>固定门限：</b>sim ${val(r.sim_threshold)} · margin ${val(r.margin_threshold)} · distance ${val(r.threshold_distance)} · IoU ${val(r.candidate_iou_threshold)}<br><b>候选：</b>${val(r.candidate_count)} 个 · <b>model choice：</b>${val(r.model_choice)} · <b>最终：</b>${val(r.final_category)}</div>`:'';c.innerHTML=`<div class="image"><img loading="lazy" decoding="async" src="${r.image_crop}" data-crop="${r.image_crop}" data-full="${r.image_full}" data-view="crop" alt="${r.quality_status} mask"></div><div class="meta"><div class="topline"><span class="badge ${r.quality_status}">${r.quality_status}</span>${vlm}${outcome}<span class="reason">${r.quality_reason}</span></div><div class="uid">${r.obs_uid}</div><div class="stats"><span>类别 <b>${val(r.class_name)}</b></span><span>帧 <b>${val(r.frame_index)}</b></span><span>GT 身份 <b>${val(r.top1_gt_instance)}</b></span><span>GT 类别 <b>${val(r.top1_gt_class)}</b></span><span>有效深度 <b>${pct(r.valid_depth_ratio)}</b></span><span>GT 支持 <b>${pct(r.gt_support_ratio)}</b></span><span>Top-1 purity <b>${pct(r.top1_purity)}</b></span><span>Top-2 purity <b>${pct(r.top2_purity)}</b></span><span>匹配点 <b>${val(r.matched_gt_points)}</b></span><span>mask 面积 <b>${val(r.mask_area)}</b></span></div>${gate}<div class="hint">GT 计数 ${counts}${transition}<br>点击图像切换局部裁剪 / 完整 RGB 帧</div></div>`;const img=c.querySelector('img');c.querySelector('.image').onclick=()=>{const full=img.dataset.view==='crop';img.src=full?img.dataset.full:img.dataset.crop;img.dataset.view=full?'full':'crop'};g.append(c)}}
$('scope').onchange=()=>{page=1;syncScope();load()};for(const id of ['status','reason'])$(id).onchange=()=>{page=1;load()};let timer;$('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{page=1;load()},250)};$('prev').onclick=()=>{page--;load();scrollTo(0,0)};$('next').onclick=()=>{page++;load();scrollTo(0,0)};init();
</script></body></html>"""


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def make_handler(dataset: ReviewDataset, max_side: int):
    image_slots = threading.BoundedSemaphore(2)

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(20.0)

        def send_payload(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_payload(HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/favicon.ico":
                    self.send_payload(b"", "image/x-icon", 204)
                    return
                if parsed.path == "/api/summary":
                    self.send_payload(json_bytes(dataset.summary()), "application/json")
                    return
                if parsed.path == "/api/rows":
                    query = parse_qs(parsed.query)
                    scope = query.get("scope", [SCOPE_VLM_USABLE_GT_MIXED])[0].upper()
                    status = query.get("status", ["ALL"])[0].upper()
                    reason = query.get("reason", ["ALL"])[0]
                    text = query.get("q", [""])[0]
                    page = max(1, int(query.get("page", ["1"])[0]))
                    page_size = min(48, max(1, int(query.get("size", ["10"])[0])))
                    if scope not in set(REVIEW_SCOPES):
                        raise ValueError("invalid scope")
                    if status not in {"ALL", *REVIEW_STATUSES}:
                        raise ValueError("invalid status")
                    payload = dataset.query(
                        scope=scope,
                        status=status,
                        reason=reason,
                        text=text,
                        page=page,
                        page_size=page_size,
                    )
                    self.send_payload(json_bytes(payload), "application/json")
                    return
                if parsed.path.startswith("/image/") and parsed.path.endswith(".webp"):
                    obs_uid = parsed.path[len("/image/") : -len(".webp")]
                    if obs_uid not in dataset.by_uid:
                        self.send_payload(b"not found", "text/plain", 404)
                        return
                    query = parse_qs(parsed.query)
                    view = query.get("view", ["crop"])[0]
                    if view not in {"crop", "full"}:
                        raise ValueError("view must be crop or full")
                    with image_slots:
                        payload, _ = dataset.render(obs_uid, view=view, max_side=max_side)
                    self.send_payload(payload, "image/webp")
                    return
                self.send_payload(b"not found", "text/plain", 404)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self.send_payload(str(exc).encode("utf-8"), "text/plain; charset=utf-8", 400)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("review_masks: " + format % args + "\n")

    return Handler


def check_dataset(dataset: ReviewDataset, max_side: int) -> dict[str, Any]:
    samples = {}
    for status in REVIEW_STATUSES:
        row = next((item for item in dataset.rows if item["quality_status"] == status), None)
        if row is None:
            continue
        payload, size = dataset.render(str(row["obs_uid"]), view="crop", max_side=max_side)
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        samples[status] = {
            "obs_uid": row["obs_uid"],
            "encoded_bytes": len(payload),
            "rendered_size": list(size),
        }
    max_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"summary": dataset.summary(), "samples": samples, "peak_rss_mib": max_rss_kib / 1024.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Low-memory RGB mask and VLM discard review"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate two renders and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if not 160 <= args.max_side <= 2400:
        raise ValueError("--max-side must be in [160, 2400]")
    run_dir = args.run_dir.resolve()
    metrics_dir = (args.metrics_dir or run_dir / "observation_gt_metrics").resolve()
    dataset = ReviewDataset(
        run_dir,
        metrics_dir,
        verify_hashes=args.verify_hashes,
    )
    if args.check:
        print(json.dumps(check_dataset(dataset, args.max_side), ensure_ascii=False, indent=2))
        return 0
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(dataset, args.max_side))
    server.daemon_threads = True
    print(
        f"Reviewing {len(dataset.rows)} masks at http://{args.bind}:{args.port} "
        "(at most two image renders run concurrently; generated WebP images are not saved)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
