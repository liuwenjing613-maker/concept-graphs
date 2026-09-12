from __future__ import annotations

import argparse
import hashlib
import io
import json
import mimetypes
import resource
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw


PROTOCOL = "vlm_new_decision_manual_review_v1_20260912"
TARGET_DECISION_SOURCE = "v7_staged_vlm"
PRIMARY_CATEGORIES = ("DUPLICATE_CREATION", "AMBIGUOUS_NEW")
REVIEW_CHOICES = (
    "CONFIRM_DUPLICATE",
    "CONFIRM_CORRECT_NEW",
    "INSUFFICIENT_VISUAL_EVIDENCE",
    "GT_OR_MASK_LABEL_ERROR",
)
MASK_COLOR = (33, 235, 176)


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


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def thicken(binary: np.ndarray, radius: int = 2) -> np.ndarray:
    thick = np.asarray(binary, dtype=bool).copy()
    source = thick.copy()
    height, width = thick.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            y0, y1 = max(0, dy), min(height, height + dy)
            x0, x1 = max(0, dx), min(width, width + dx)
            thick[y0:y1, x0:x1] |= source[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return thick


def mask_bounds(mask: np.ndarray, padding_ratio: float = 0.14) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    if not len(rows):
        return 0, 0, width, height
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    padding = max(28, int(max(x1 - x0, y1 - y0) * padding_ratio))
    return max(0, x0 - padding), max(0, y0 - padding), min(width, x1 + padding), min(height, y1 + padding)


def render_rgb_mask(rgb_path: Path, mask: np.ndarray, view: str, max_side: int) -> bytes:
    with Image.open(rgb_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"RGB/mask shape mismatch: rgb={rgb.shape[:2]}, mask={mask.shape}")
    output = rgb.copy()
    color = np.asarray(MASK_COLOR, dtype=np.float32)
    output[mask] = np.clip(0.86 * output[mask].astype(np.float32) + 0.14 * color, 0, 255).astype(np.uint8)
    output[thicken(mask_boundary(mask), radius=2)] = np.asarray(MASK_COLOR, dtype=np.uint8)
    image = Image.fromarray(output, mode="RGB")
    rows, cols = np.nonzero(mask)
    if len(rows):
        box = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=(0, 0, 0), width=7)
        draw.rectangle(box, outline=MASK_COLOR, width=4)
    if view == "crop":
        image = image.crop(mask_bounds(mask))
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(tuple(max(1, round(v * scale)) for v in image.size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=80, method=4)
    return buffer.getvalue()


class NewDecisionReviewDataset:
    def __init__(self, run_dir: Path, metrics_dir: Path, verify_hashes: bool = False) -> None:
        self.run_dir = run_dir.resolve()
        self.metrics_dir = metrics_dir.resolve()
        self.verify_hashes = bool(verify_hashes)
        self._verified_paths: set[Path] = set()

        decisions = []
        for row in iter_jsonl(self.metrics_dir / "event_decisions.jsonl"):
            final_action = dict(row.get("final_action") or {})
            if final_action.get("kind") != "NEW":
                continue
            if row.get("decision_source") != TARGET_DECISION_SOURCE:
                continue
            decisions.append(row)
        if not decisions:
            raise ValueError("no v7_staged_vlm final NEW decisions found")
        needed = {str(row["obs_uid"]) for row in decisions}

        labels = {
            str(row["obs_uid"]): row
            for row in iter_jsonl(self.metrics_dir / "observation_labels.jsonl")
            if str(row.get("obs_uid")) in needed
        }
        associations = {
            str(row["obs_uid"]): row
            for row in iter_jsonl(self.run_dir / "evidence/associations.jsonl")
            if str(row.get("obs_uid")) in needed
        }
        gates = {
            str(row["current_observation_uid"]): row
            for row in iter_jsonl(self.run_dir / "blocking_association_gate/events.jsonl")
            if str(row.get("current_observation_uid")) in needed
        }
        identities: dict[str, dict[str, dict[str, Any]]] = {uid: {} for uid in needed}
        for row in iter_jsonl(self.metrics_dir / "prediction_identities.jsonl"):
            obs_uid = str(row.get("obs_uid"))
            if obs_uid in needed:
                identities[obs_uid][str(row["object_uid"])] = row

        observations = {
            str(row["obs_uid"]): row
            for row in iter_jsonl(self.run_dir / "evidence/observations.jsonl")
            if str(row.get("obs_uid")) in needed
        }
        needed_frames = {str(labels[uid]["frame_uid"]) for uid in needed if uid in labels}
        self.frames = {
            str(row["frame_uid"]): row
            for row in iter_jsonl(self.run_dir / "evidence/frames.jsonl")
            if str(row.get("frame_uid")) in needed_frames
        }

        missing = {
            "labels": needed - labels.keys(),
            "associations": needed - associations.keys(),
            "gates": needed - gates.keys(),
            "observations": needed - observations.keys(),
        }
        bad = {key: sorted(value) for key, value in missing.items() if value}
        if bad:
            raise ValueError(f"missing evidence rows: { {key: len(value) for key, value in bad.items()} }")

        self.rows: list[dict[str, Any]] = []
        self.allowed_gate_images: dict[tuple[str, str], Path] = {}
        for decision in decisions:
            obs_uid = str(decision["obs_uid"])
            label = labels[obs_uid]
            association = associations[obs_uid]
            gate = gates[obs_uid]
            identity_map = identities[obs_uid]
            object_uids = list(association.get("object_uids_before") or [])
            alias_to_index = dict(gate.get("candidate_alias_to_object_index") or {})
            scores = dict((gate.get("audit_scores_hidden_from_vlm") or {}).get("candidate_scores") or {})
            gt_value = label.get("assigned_gt_instance")
            gt_id = int(gt_value) if gt_value is not None else None
            canonical_by_gt = dict(decision.get("canonical_prediction_by_gt") or {})
            canonical_uid = canonical_by_gt.get(str(gt_id)) if gt_id is not None else None
            evidence_paths = {
                str(item.get("path")): str(item.get("label") or "")
                for item in gate.get("evidence") or []
                if item.get("path")
            }
            event_id = str(gate["event_id"])
            for name in evidence_paths:
                candidate_path = (self.run_dir / "blocking_association_gate/events" / event_id / name).resolve()
                event_root = (self.run_dir / "blocking_association_gate/events" / event_id).resolve()
                if candidate_path.parent == event_root and candidate_path.is_file():
                    self.allowed_gate_images[(event_id, name)] = candidate_path

            candidates = []
            for alias, index_value in sorted(alias_to_index.items()):
                index = int(index_value)
                uid = object_uids[index] if 0 <= index < len(object_uids) else None
                identity = dict(identity_map.get(str(uid)) or {}) if uid else {}
                image_name = f"candidate_{alias}.jpg"
                candidates.append(
                    {
                        "alias": alias,
                        "object_index": index,
                        "object_uid": uid,
                        "score": scores.get(alias),
                        "identity_status": identity.get("identity_status") or "MISSING",
                        "gt_instance": identity.get("gt_instance"),
                        "identity_purity": identity.get("identity_purity"),
                        "support_count": identity.get("dominant_support_count"),
                        "is_obs_gt": gt_id is not None and identity.get("gt_instance") == gt_id,
                        "is_obs_gt_canonical": bool(canonical_uid and uid == canonical_uid),
                        "image": f"/gate-image/{quote(event_id)}/{quote(image_name)}" if (event_id, image_name) in self.allowed_gate_images else None,
                    }
                )

            same_gt = []
            uncertain = []
            for uid, identity in identity_map.items():
                status = str(identity.get("identity_status") or "MISSING")
                if status == "RELIABLE" and gt_id is not None and identity.get("gt_instance") == gt_id:
                    same_gt.append(
                        {
                            "object_uid": uid,
                            "canonical": uid == canonical_uid,
                            "identity_purity": identity.get("identity_purity"),
                            "support_count": identity.get("dominant_support_count"),
                        }
                    )
                elif status != "RELIABLE":
                    uncertain.append(uid)
            same_gt.sort(key=lambda value: (not value["canonical"], value["object_uid"]))
            final_result = dict(decision.get("final_result") or {})
            baseline_result = dict(decision.get("baseline_result") or {})
            trigger = dict(gate.get("trigger") or {})
            model_output = dict(gate.get("model_output") or {})
            quality_name = "quality.jpg"
            row = {
                "obs_uid": obs_uid,
                "association_event_uid": decision.get("association_event_uid"),
                "event_id": event_id,
                "event_sequence": decision.get("event_sequence"),
                "frame_uid": label.get("frame_uid"),
                "frame_index": label.get("frame_index"),
                "source_frame_id": label.get("source_frame_id"),
                "class_name": label.get("class_name"),
                "category": final_result.get("category"),
                "baseline_category": baseline_result.get("category"),
                "quality_status": label.get("quality_status"),
                "quality_reason": label.get("quality_reason"),
                "gt_instance": gt_id,
                "gt_class": label.get("top1_gt_class"),
                "top1_purity": label.get("top1_purity"),
                "gt_support_ratio": label.get("gt_support_ratio"),
                "mask_area": label.get("mask_area"),
                "valid_depth_ratio": label.get("valid_depth_ratio"),
                "gt_instance_counts": label.get("gt_instance_counts") or {},
                "changed": decision.get("changed"),
                "baseline_action": (decision.get("baseline_action") or {}).get("kind"),
                "final_action": (decision.get("final_action") or {}).get("kind"),
                "route_reason": decision.get("route_reason"),
                "model_choice": model_output.get("choice"),
                "model_confidence": model_output.get("confidence"),
                "trigger_kind": trigger.get("kind"),
                "trigger_reasons": trigger.get("reasons") or [],
                "trigger_top1": trigger.get("top1"),
                "trigger_top2": trigger.get("top2"),
                "trigger_margin": trigger.get("margin"),
                "trigger_threshold_distance": trigger.get("threshold_distance"),
                "canonical_uid": canonical_uid,
                "same_gt_existing": same_gt,
                "uncertain_existing_count": len(uncertain),
                "uncertain_existing_sample": uncertain[:6],
                "candidate_count": len(candidates),
                "candidates": candidates,
                "current_gate_image": f"/gate-image/{quote(event_id)}/{quality_name}" if (event_id, quality_name) in self.allowed_gate_images else None,
                "current_mask_crop": f"/mask-image/{quote(obs_uid)}.webp?view=crop",
                "current_mask_full": f"/mask-image/{quote(obs_uid)}.webp?view=full",
                "processed_mask_ref": observations[obs_uid].get("processed_mask_ref"),
            }
            self.rows.append(row)

        category_rank = {"DUPLICATE_CREATION": 0, "AMBIGUOUS_NEW": 1}
        self.rows.sort(key=lambda row: (category_rank.get(str(row["category"]), 9), int(row.get("frame_index") or 0), str(row["obs_uid"])))
        self.by_uid = {str(row["obs_uid"]): row for row in self.rows}
        self.run_id = str(self.rows[0]["obs_uid"]).split("_f", 1)[0]

    def _resolve_ref(self, ref: Mapping[str, Any]) -> Path:
        path = (self.run_dir / str(ref["path"])).resolve()
        # Frame refs may deliberately resolve from the experiment directory back
        # to the scene-level ``results/`` directory.  The ref comes from frozen
        # evidence, never from an HTTP parameter, so only existence is required.
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = ref.get("sha256")
        if self.verify_hashes and expected and path not in self._verified_paths:
            if sha256_file(path) != str(expected):
                raise ValueError(f"evidence hash drift: {path}")
            self._verified_paths.add(path)
        return path

    def render_mask(self, obs_uid: str, view: str, max_side: int) -> bytes:
        row = self.by_uid[obs_uid]
        ref = dict(row.get("processed_mask_ref") or {})
        with np.load(self._resolve_ref(ref), allow_pickle=False) as archive:
            mask = np.asarray(archive[str(ref.get("key") or "mask")], dtype=bool)
        frame = self.frames[str(row["frame_uid"])]
        rgb_path = self._resolve_ref(dict(frame["rgb_ref"]))
        return render_rgb_mask(rgb_path, mask, view, max_side)

    def gate_image(self, event_id: str, name: str) -> Path | None:
        return self.allowed_gate_images.get((event_id, name))

    def summary(self) -> dict[str, Any]:
        categories = Counter(str(row["category"]) for row in self.rows)
        transitions = Counter(f"{row['baseline_category']}->{row['category']}" for row in self.rows)
        return {
            "protocol": PROTOCOL,
            "run_id": self.run_id,
            "total": len(self.rows),
            "category_counts": dict(sorted(categories.items())),
            "transition_counts": dict(sorted(transitions.items())),
            "primary_categories": list(PRIMARY_CATEGORIES),
            "review_choices": list(REVIEW_CHOICES),
            "storage_key": f"{PROTOCOL}:{self.run_id}",
            "review_scope": "all v7_staged_vlm final NEW decisions",
        }

    def query(self, category: str, text: str) -> dict[str, Any]:
        needle = text.strip().lower()
        rows = []
        for row in self.rows:
            if category != "ALL" and row["category"] != category:
                continue
            searchable = " ".join(
                str(row.get(key) or "")
                for key in (
                    "obs_uid",
                    "event_id",
                    "frame_uid",
                    "class_name",
                    "gt_instance",
                    "gt_class",
                    "category",
                    "baseline_category",
                    "route_reason",
                )
            ).lower()
            if needle and needle not in searchable:
                continue
            public = {key: value for key, value in row.items() if key != "processed_mask_ref"}
            rows.append(public)
        return {"rows": rows, "total": len(rows)}


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM NEW forensic review</title>
<style>
:root{color-scheme:dark;--ink:#e9eee9;--muted:#8d9993;--paper:#111513;--panel:#171c19;--panel2:#202622;--line:#39423d;--acid:#c8ff37;--mint:#21ebb0;--amber:#ffb44a;--red:#ff654f;--blue:#61baff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 82% -20%,#34452e 0,transparent 34%),linear-gradient(135deg,#0d100f,#151a17 46%,#0b0e0c);color:var(--ink);font:14px/1.45 "Microsoft YaHei UI","Noto Sans CJK SC",sans-serif;min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.13;background-image:repeating-linear-gradient(0deg,transparent 0 3px,#fff 4px);mix-blend-mode:overlay}
header{position:sticky;top:0;z-index:10;background:rgba(13,16,15,.93);backdrop-filter:blur(15px);border-bottom:1px solid var(--line);padding:13px 24px}.brand{display:flex;align-items:end;gap:15px}.kicker{font:800 11px/1 ui-monospace,monospace;letter-spacing:.2em;color:var(--acid)}h1{font-size:21px;line-height:1;margin:0;letter-spacing:-.03em}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}select,input,button,textarea{font:inherit;color:var(--ink);background:#1b211e;border:1px solid #47514b;border-radius:2px;padding:8px 10px}select,input{height:36px}input{min-width:250px}button{cursor:pointer;transition:.18s ease}button:hover{border-color:var(--acid);transform:translateY(-1px)}button:disabled{opacity:.35;cursor:not-allowed;transform:none}.counter{margin-left:auto;font:700 12px ui-monospace,monospace;color:var(--muted)}
main{max-width:1680px;margin:auto;padding:20px 24px 140px}.case-head{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:20px;align-items:start;margin-bottom:13px}.case-no{font:900 clamp(28px,4vw,56px)/.9 Georgia,serif;letter-spacing:-.06em}.case-no small{display:block;font:700 10px/1.4 ui-monospace,monospace;letter-spacing:.16em;color:var(--muted);margin-bottom:9px}.tags{display:flex;justify-content:flex-end;gap:7px;flex-wrap:wrap}.tag{border:1px solid var(--line);padding:5px 8px;font:800 11px ui-monospace,monospace;letter-spacing:.05em;background:#131714}.tag.dc{color:var(--red);border-color:#873c32}.tag.an{color:var(--amber);border-color:#7d6134}.tag.clean{color:var(--mint);border-color:#28735a}.tag.changed{color:var(--blue);border-color:#315d78}
.evidence{display:grid;grid-template-columns:1.18fr repeat(3,1fr);gap:8px}.shot{position:relative;background:#050706;border:1px solid var(--line);min-width:0;overflow:hidden}.shot.current{border-color:#57804a}.shot.canonical{border-color:var(--red);box-shadow:inset 0 0 0 1px var(--red)}.shot-head{height:47px;display:flex;align-items:center;justify-content:space-between;padding:8px 10px;background:linear-gradient(90deg,#1b211e,#111512);border-bottom:1px solid var(--line)}.alias{font:900 18px Georgia,serif}.relation{font:800 10px ui-monospace,monospace;color:var(--muted);text-align:right}.relation.same{color:var(--red)}.photo{height:min(46vh,520px);display:flex;align-items:center;justify-content:center;background:#060807}.photo img{width:100%;height:100%;object-fit:contain}.photo .missing{color:#66716b;font:12px ui-monospace,monospace}.shot-foot{padding:8px 10px;min-height:65px;color:#aeb8b2;font:11px/1.5 ui-monospace,monospace}.shot-foot b{color:#f0f4f1}.toggle{position:absolute;right:8px;top:56px;padding:5px 7px;font-size:10px;background:#0e1511cc}
.analysis{display:grid;grid-template-columns:1.2fr 1fr .9fr;gap:8px;margin-top:8px}.block{background:var(--panel);border:1px solid var(--line);padding:13px;min-width:0}.block h2{font:800 10px ui-monospace,monospace;letter-spacing:.17em;color:var(--acid);margin:0 0 10px}.verdict{font:800 18px/1.35 Georgia,"Microsoft YaHei UI",serif}.logic{margin-top:8px;color:#b9c3bd}.logic strong{color:white}.facts{display:grid;grid-template-columns:1fr 1fr;gap:7px 15px}.fact{border-bottom:1px solid #2e3631;padding-bottom:5px}.fact span{display:block;color:var(--muted);font:10px ui-monospace,monospace}.fact b{display:block;margin-top:2px;font-size:13px;overflow-wrap:anywhere}.uids{font:11px/1.6 ui-monospace,monospace;color:#aab5ae;overflow-wrap:anywhere}.uids em{color:var(--red);font-style:normal}.empty{text-align:center;padding:100px;color:var(--muted)}
.review{position:fixed;z-index:20;left:0;right:0;bottom:0;background:rgba(12,15,13,.96);backdrop-filter:blur(18px);border-top:1px solid #4a554e;padding:10px 24px}.review-inner{max-width:1680px;margin:auto;display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center}.review-title{font:900 11px ui-monospace,monospace;color:var(--acid);letter-spacing:.12em}.choices{display:flex;gap:6px;flex-wrap:wrap}.choice{padding:9px 12px;font-size:12px}.choice.active{background:var(--acid);border-color:var(--acid);color:#111;font-weight:900}.choice[data-v="CONFIRM_DUPLICATE"].active{background:var(--red);border-color:var(--red);color:#fff}.note{width:100%;height:37px;resize:none}.review-actions{display:flex;gap:6px}.progress{font:700 11px ui-monospace,monospace;color:var(--muted);white-space:nowrap}.nav{display:flex;gap:6px}.key{opacity:.55;font:10px ui-monospace,monospace}.toast{position:fixed;right:24px;bottom:128px;background:var(--acid);color:#10130f;padding:9px 13px;font-weight:800;transform:translateY(20px);opacity:0;transition:.2s;z-index:30}.toast.on{opacity:1;transform:none}
@media(max-width:1100px){.evidence{grid-template-columns:1fr 1fr}.photo{height:380px}.analysis{grid-template-columns:1fr}.review-inner{grid-template-columns:1fr}.review-title{display:none}.review{max-height:190px;overflow:auto}main{padding-bottom:210px}}
@media(max-width:650px){header,main,.review{padding-left:10px;padding-right:10px}.brand{align-items:start;flex-direction:column}.counter{margin-left:0}.evidence{grid-template-columns:1fr}.photo{height:330px}.case-head{grid-template-columns:1fr}.tags{justify-content:flex-start}.choices{display:grid;grid-template-columns:1fr 1fr}.choice{padding:8px 4px;font-size:10px}}
</style></head><body>
<header><div class="brand"><div class="kicker">FORENSIC CASE DESK / OFFICE25050</div><h1>VLM 新建决策人工复核</h1></div><div class="controls"><select id="category"></select><input id="search" placeholder="搜索 observation、GT 或 baseline 类别"><div class="nav"><button id="prev">← 上一例</button><button id="next">下一例 →</button></div><button id="export">导出复核 JSON</button><span class="counter" id="counter"></span></div></header>
<main id="main"><div class="empty">正在建立证据索引…</div></main>
<section class="review"><div class="review-inner"><div class="review-title">HUMAN VERDICT</div><div><div class="choices" id="choices"><button class="choice" data-v="CONFIRM_DUPLICATE"><span class="key">1</span> 确认重复新建</button><button class="choice" data-v="CONFIRM_CORRECT_NEW"><span class="key">2</span> 应为正确新建</button><button class="choice" data-v="INSUFFICIENT_VISUAL_EVIDENCE"><span class="key">3</span> 视觉证据不足</button><button class="choice" data-v="GT_OR_MASK_LABEL_ERROR"><span class="key">4</span> GT / 掩码标签错误</button></div><textarea class="note" id="note" placeholder="可选备注：为何同意/推翻脚本标签"></textarea></div><div class="review-actions"><span class="progress" id="progress"></span><button id="clear">清除此例</button></div></div></section><div class="toast" id="toast"></div>
<script>
const $=id=>document.getElementById(id), esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=v=>v==null?'—':(Number(v)*100).toFixed(1)+'%'; const num=v=>v==null?'—':Number(v).toFixed(3).replace(/0+$/,'').replace(/\.$/,''); const short=v=>v?String(v).slice(0,8):'—';
let summary={},rows=[],index=0,annotations={},noteTimer; const labels={ALL:'全部 VLM→NEW',DUPLICATE_CREATION:'重复新建',AMBIGUOUS_NEW:'不确定新建',MIXED_MASK_NEW:'混合掩码新建',UNSCORABLE_OBSERVATION:'不可评分新建'};
function toast(s){$('toast').textContent=s;$('toast').classList.add('on');setTimeout(()=>$('toast').classList.remove('on'),1300)}
function loadStore(){try{annotations=JSON.parse(localStorage.getItem(summary.storage_key)||'{}')}catch{annotations={}}} function saveStore(){localStorage.setItem(summary.storage_key,JSON.stringify(annotations));updateProgress()}
function current(){return rows[index]}
function setAnnotation(value){const r=current();if(!r)return;const old=annotations[r.obs_uid]||{};annotations[r.obs_uid]={...old,manual_label:value,note:$('note').value||'',reviewed_at:new Date().toISOString(),auto_category:r.category,event_id:r.event_id,association_event_uid:r.association_event_uid,gt_instance:r.gt_instance};saveStore();renderReview();toast('已保存在当前浏览器')}
function updateProgress(){const visible=rows.filter(r=>annotations[r.obs_uid]?.manual_label).length;const total=Object.values(annotations).filter(x=>x?.manual_label).length;$('progress').textContent=`当前筛选 ${visible}/${rows.length} · 总计 ${total}/${summary.total}`}
async function fetchRows(){const q=new URLSearchParams({category:$('category').value,q:$('search').value});const d=await fetch('/api/cases?'+q).then(r=>r.json());rows=d.rows;index=Math.min(index,Math.max(0,rows.length-1));render()}
function evidenceShot(title,img,foot,cls='',toggle=''){return `<article class="shot ${cls}"><div class="shot-head"><span class="alias">${esc(title)}</span>${foot.head||''}</div><div class="photo">${img?`<img src="${esc(img)}" alt="${esc(title)}">`:'<span class="missing">NO IMAGE EVIDENCE</span>'}</div>${toggle}<div class="shot-foot">${foot.body||''}</div></article>`}
function render(){const r=current();$('prev').disabled=!r||index<=0;$('next').disabled=!r||index>=rows.length-1;$('counter').textContent=r?`${index+1} / ${rows.length}`:'0 / 0';if(!r){$('main').innerHTML='<div class="empty">没有匹配案例</div>';renderReview();return}
 const catClass=r.category==='DUPLICATE_CREATION'?'dc':r.category==='AMBIGUOUS_NEW'?'an':'';const currentFoot={head:`<span class="relation">OBSERVATION / GT ${esc(r.gt_instance)}</span>`,body:`mask 类别 <b>${esc(r.class_name)}</b> · GT 类别 <b>${esc(r.gt_class)}</b><br>purity <b>${pct(r.top1_purity)}</b> · GT support <b>${pct(r.gt_support_ratio)}</b> · mask area <b>${esc(r.mask_area)}</b>`};
 let shots=evidenceShot('CURRENT',r.current_mask_crop,currentFoot,'current',`<button class="toggle" id="toggleMask">切换完整帧</button>`);for(const c of r.candidates){const relation=c.is_obs_gt_canonical?'同 GT · CANONICAL':c.is_obs_gt?'同 GT · FRAGMENT':c.identity_status==='RELIABLE'?`其他 GT ${c.gt_instance}`:'身份不确定';const cls=c.is_obs_gt_canonical?'canonical':'';shots+=evidenceShot(`CANDIDATE ${c.alias}`,c.image,{head:`<span class="relation ${c.is_obs_gt?'same':''}">${esc(relation)}</span>`,body:`score <b>${num(c.score)}</b> · object #${esc(c.object_index)} / ${esc(short(c.object_uid))}<br>identity <b>${esc(c.identity_status)}</b> · purity <b>${pct(c.identity_purity)}</b> · support <b>${esc(c.support_count)}</b>`},cls)}
 const same=r.same_gt_existing||[];let autoLogic;if(r.category==='DUPLICATE_CREATION'){autoLogic=`当前 observation 为 <strong>CLEAN / GT ${esc(r.gt_instance)}</strong>，事件前已有 <strong>${same.length}</strong> 个可靠的同-GT预测对象；再次 NEW 因而记为重复新建。`}else if(r.category==='AMBIGUOUS_NEW'){autoLogic=`当前 observation 为 <strong>CLEAN / GT ${esc(r.gt_instance)}</strong>，事件前没有可靠同-GT对象，但存在 <strong>${esc(r.uncertain_existing_count)}</strong> 个身份不确定对象；因此脚本不能证明 NEW 正确，只能记为 AMBIGUOUS_NEW。`}else if(r.category==='MIXED_MASK_NEW'){autoLogic=`当前 observation 的掩码包含多个 GT 实例，NEW 不计入 clean observation 的关联准确率；请人工判断它是否仍具有可用的主体。`}else if(r.category==='UNSCORABLE_OBSERVATION'){autoLogic=`当前掩码的有效实例 GT 支持不足，无法可靠赋予 GT 身份；该 NEW 事件不进入确定性正确/错误统计。`}else{autoLogic=`该事件按 <strong>${esc(r.category)}</strong> 记录，请结合当前掩码、候选投影和冻结对象身份人工复核。`}
 const sameHtml=same.length?same.map(x=>`<div><em>${x.canonical?'CANONICAL':'FRAGMENT'}</em> ${esc(x.object_uid)} · purity ${pct(x.identity_purity)} · support ${esc(x.support_count)}</div>`).join(''):'<div>无可靠同-GT对象</div>';const unknown=(r.uncertain_existing_sample||[]).map(x=>short(x)).join(' · ')||'—';
 $('main').innerHTML=`<div class="case-head"><div class="case-no"><small>CASE ${String(index+1).padStart(2,'0')} / FRAME ${esc(r.frame_index)} / EVENT ${esc(r.event_sequence)}</small>${esc(r.class_name)} → GT ${esc(r.gt_instance)}</div><div class="tags"><span class="tag ${catClass}">${esc(r.category)}</span><span class="tag clean">${esc(r.quality_status)}</span>${r.changed?'<span class="tag changed">VLM CHANGED</span>':''}<span class="tag">${esc(r.baseline_action)} → ${esc(r.final_action)}</span></div></div><section class="evidence">${shots}</section><section class="analysis"><div class="block"><h2>AUTOMATIC VERDICT</h2><div class="verdict">${esc(r.category)}</div><div class="logic">${autoLogic}</div></div><div class="block"><h2>DECISION TRACE</h2><div class="facts"><div class="fact"><span>baseline category</span><b>${esc(r.baseline_category)}</b></div><div class="fact"><span>VLM choice</span><b>${esc(r.model_choice)}</b></div><div class="fact"><span>route</span><b>${esc(r.route_reason)}</b></div><div class="fact"><span>trigger</span><b>${esc(r.trigger_kind)} / ${esc((r.trigger_reasons||[]).join(' + '))}</b></div><div class="fact"><span>top1 / top2</span><b>${num(r.trigger_top1)} / ${num(r.trigger_top2)}</b></div><div class="fact"><span>changed</span><b>${esc(r.changed)}</b></div></div></div><div class="block"><h2>FROZEN IDENTITY EVIDENCE</h2><div class="uids">${sameHtml}<br>不确定对象 ${esc(r.uncertain_existing_count)} 个：${esc(unknown)}<br><br>obs: ${esc(r.obs_uid)}<br>gate: ${esc(r.event_id)}</div></div></section>`;
 const toggle=$('toggleMask');if(toggle){let full=false;toggle.onclick=()=>{full=!full;const img=toggle.closest('.shot').querySelector('img');img.src=full?r.current_mask_full:r.current_mask_crop;toggle.textContent=full?'切换局部掩码':'切换完整帧'}}renderReview();window.scrollTo({top:0,behavior:'smooth'})}
function renderReview(){const r=current(),a=r?annotations[r.obs_uid]||{}:{};for(const b of document.querySelectorAll('.choice'))b.classList.toggle('active',b.dataset.v===a.manual_label);$('note').value=a.note||'';updateProgress()}
function step(delta){if(!rows.length)return;index=Math.max(0,Math.min(rows.length-1,index+delta));render()}
async function init(){summary=await fetch('/api/summary').then(r=>r.json());loadStore();const categories=['ALL',...Object.keys(summary.category_counts)];for(const c of categories){const o=document.createElement('option');o.value=c;o.textContent=`${labels[c]||c}（${c==='ALL'?summary.total:summary.category_counts[c]}）`;$('category').append(o)}$('category').value='DUPLICATE_CREATION';await fetchRows()}
for(const b of document.querySelectorAll('.choice'))b.onclick=()=>setAnnotation(b.dataset.v);$('clear').onclick=()=>{const r=current();if(r){delete annotations[r.obs_uid];saveStore();renderReview()}};$('note').oninput=()=>{clearTimeout(noteTimer);noteTimer=setTimeout(()=>{const r=current();if(r&&annotations[r.obs_uid]){annotations[r.obs_uid].note=$('note').value;saveStore()}},300)};$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('category').onchange=()=>{index=0;fetchRows()};let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{index=0;fetchRows()},220)};
$('export').onclick=()=>{const payload={protocol:summary.protocol,run_id:summary.run_id,exported_at:new Date().toISOString(),automatic_counts:summary.category_counts,review_choices:summary.review_choices,annotations:Object.entries(annotations).map(([obs_uid,value])=>({obs_uid,...value}))};const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`${summary.run_id}_vlm_new_manual_review.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.addEventListener('keydown',e=>{if(e.target.matches('input,textarea,select'))return;if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);if(['1','2','3','4'].includes(e.key))setAnnotation(summary.review_choices[Number(e.key)-1])});init();
</script></body></html>'''


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def make_handler(dataset: NewDecisionReviewDataset, max_side: int):
    image_slots = threading.BoundedSemaphore(3)

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(30.0)

        def send_payload(self, payload: bytes, content_type: str, status: int = 200, cache: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", cache)
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
                if parsed.path == "/api/cases":
                    query = parse_qs(parsed.query)
                    category = query.get("category", ["DUPLICATE_CREATION"])[0]
                    text = query.get("q", [""])[0]
                    allowed = {"ALL", *(str(row["category"]) for row in dataset.rows)}
                    if category not in allowed:
                        raise ValueError("invalid category")
                    self.send_payload(json_bytes(dataset.query(category, text)), "application/json")
                    return
                if parsed.path.startswith("/mask-image/") and parsed.path.endswith(".webp"):
                    obs_uid = unquote(parsed.path[len("/mask-image/") : -len(".webp")])
                    if obs_uid not in dataset.by_uid:
                        self.send_payload(b"not found", "text/plain", 404)
                        return
                    view = parse_qs(parsed.query).get("view", ["crop"])[0]
                    if view not in {"crop", "full"}:
                        raise ValueError("view must be crop or full")
                    with image_slots:
                        payload = dataset.render_mask(obs_uid, view, max_side)
                    self.send_payload(payload, "image/webp", cache="private, max-age=3600")
                    return
                if parsed.path.startswith("/gate-image/"):
                    parts = parsed.path.split("/", 3)
                    if len(parts) != 4:
                        raise ValueError("invalid gate image path")
                    event_id, name = unquote(parts[2]), unquote(parts[3])
                    path = dataset.gate_image(event_id, name)
                    if path is None:
                        self.send_payload(b"not found", "text/plain", 404)
                        return
                    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    self.send_payload(path.read_bytes(), content_type, cache="private, max-age=3600")
                    return
                self.send_payload(b"not found", "text/plain", 404)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self.send_payload(str(exc).encode("utf-8"), "text/plain; charset=utf-8", 400)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("review_new_decisions: " + format % args + "\n")

    return Handler


def check_dataset(dataset: NewDecisionReviewDataset, max_side: int) -> dict[str, Any]:
    samples = {}
    for category in PRIMARY_CATEGORIES:
        row = next((item for item in dataset.rows if item["category"] == category), None)
        if row is None:
            continue
        payload = dataset.render_mask(str(row["obs_uid"]), "crop", max_side)
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        candidate_files = [
            dataset.gate_image(str(row["event_id"]), f"candidate_{candidate['alias']}.jpg")
            for candidate in row["candidates"]
        ]
        samples[category] = {
            "obs_uid": row["obs_uid"],
            "rendered_mask_bytes": len(payload),
            "candidate_images_present": sum(path is not None for path in candidate_files),
            "same_gt_existing": len(row["same_gt_existing"]),
            "uncertain_existing_count": row["uncertain_existing_count"],
        }
    return {
        "summary": dataset.summary(),
        "samples": samples,
        "peak_rss_mib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser review for VLM final NEW decisions")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-side", type=int, default=900)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if not 240 <= args.max_side <= 2400:
        raise ValueError("--max-side must be in [240, 2400]")
    run_dir = args.run_dir.resolve()
    metrics_dir = (args.metrics_dir or run_dir / "observation_gt_metrics").resolve()
    dataset = NewDecisionReviewDataset(run_dir, metrics_dir, args.verify_hashes)
    if args.check:
        print(json.dumps(check_dataset(dataset, args.max_side), ensure_ascii=False, indent=2))
        return 0
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(dataset, args.max_side))
    server.daemon_threads = True
    print(f"Reviewing {len(dataset.rows)} VLM final NEW cases at http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
