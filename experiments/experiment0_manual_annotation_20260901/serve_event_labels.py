#!/usr/bin/env python3
"""Loopback-only two-stage annotation UI for Experiment 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from label_logic import derive_event_label, validate_blind_label, validate_final_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if raw.strip():
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {path}:{line_no}")
                rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AnnotationStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.cases_root = (self.root / "cases").resolve()
        self.labels_root = self.root / "labels"
        self.labels_root.mkdir(parents=True, exist_ok=True)
        self.drafts_path = self.labels_root / "blind_drafts.jsonl"
        self.labels_path = self.labels_root / "event_labels.jsonl"
        self.lock = threading.RLock()
        self._worklist_mtime_ns = -1
        self.manifest: dict[str, Any] = {}
        self.worklist: list[dict[str, Any]] = []
        self.by_case: dict[str, dict[str, Any]] = {}
        self.drafts: dict[str, dict[str, Any]] = {}
        self.labels: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.drafts_path):
            self.drafts[str(row["case_uid"])] = row
        for row in read_jsonl(self.labels_path):
            self.labels[str(row["case_uid"])] = row
        self.reload(force=True)

    def reload(self, force: bool = False) -> None:
        worklist_path = self.root / "worklist.jsonl"
        manifest_path = self.root / "manifest.json"
        if not worklist_path.is_file() or not manifest_path.is_file():
            raise ValueError("packet root 缺少 manifest.json/worklist.jsonl")
        mtime_ns = worklist_path.stat().st_mtime_ns
        if not force and mtime_ns == self._worklist_mtime_ns:
            return
        manifest = read_json(manifest_path)
        if manifest.get("worklist_sha256") != sha256_file(worklist_path):
            raise ValueError("worklist 与 manifest 哈希不一致")
        worklist = read_jsonl(worklist_path)
        if int(manifest.get("case_count", -1)) != len(worklist):
            raise ValueError("manifest case_count 与 worklist 不一致")
        by_case: dict[str, dict[str, Any]] = {}
        for row in worklist:
            case_uid = str(row.get("case_uid") or "")
            if not case_uid or case_uid in by_case:
                raise ValueError("worklist 含空或重复 case_uid")
            by_case[case_uid] = row
        self.manifest = manifest
        self.worklist = worklist
        self.by_case = by_case
        self._worklist_mtime_ns = mtime_ns

    def _case_dir(self, case_uid: str) -> Path:
        row = self.by_case.get(case_uid)
        if row is None:
            raise FileNotFoundError("unknown case")
        path = Path(str(row["case_dir"])).resolve()
        if path != self.cases_root and self.cases_root not in path.parents:
            raise ValueError("case_dir escaped packet root")
        return path

    def _case_files(self, case_uid: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
        case_dir = self._case_dir(case_uid)
        public_path = case_dir / "case_public.json"
        private_path = case_dir / "case_private.json"
        public = read_json(public_path)
        private = read_json(private_path)
        if private.get("source_public_sha256") != sha256_file(public_path):
            raise ValueError("public/private case binding failed")
        if public.get("case_uid") != case_uid or private.get("case_uid") != case_uid:
            raise ValueError("case UID binding failed")
        for name, expected in (public.get("displayed_asset_sha256") or {}).items():
            asset = (case_dir / str(name)).resolve()
            if case_dir not in asset.parents or not asset.is_file():
                raise ValueError(f"asset missing: {name}")
            if sha256_file(asset) != expected:
                raise ValueError(f"asset hash changed: {name}")
        return public, private, case_dir

    def status(self) -> dict[str, Any]:
        with self.lock:
            self.reload()
            completed = [index for index, row in enumerate(self.worklist) if str(row["case_uid"]) in self.labels]
            drafted = [index for index, row in enumerate(self.worklist) if str(row["case_uid"]) in self.drafts and str(row["case_uid"]) not in self.labels]
            return {
                "completed": len(completed),
                "drafted": len(drafted),
                "total": len(self.worklist),
                "completed_indices": completed,
                "drafted_indices": drafted,
                "mapper_latest_frame": self.manifest.get("mapper_latest_frame"),
                "ready_through_frame": self.manifest.get("ready_through_frame"),
                "mapper_complete": self.manifest.get("mapper_complete"),
            }

    def case_payload(self, index: int) -> dict[str, Any]:
        with self.lock:
            self.reload()
            if index < 0 or index >= len(self.worklist):
                raise IndexError("case index out of range")
            row = self.worklist[index]
            case_uid = str(row["case_uid"])
            public, private, _ = self._case_files(case_uid)
            draft = self.drafts.get(case_uid)
            label = self.labels.get(case_uid)
            payload = {
                "index": index,
                "position": index + 1,
                "total": len(self.worklist),
                "public": public,
                "draft": draft.get("blind") if draft else None,
                "label": label,
                "stage": "complete" if label else ("reveal" if draft else "blind"),
                "progress": self.status(),
            }
            if draft:
                target_code = str(private["selected_target_code"])
                candidate = next(row for row in private["candidates"] if row["code"] == target_code)
                payload["reveal"] = {
                    "selected_target_code": target_code,
                    "top1_score": private["association_event"].get("top1_score"),
                    "top2_score": private["association_event"].get("top2_score"),
                    "margin": private["association_event"].get("margin"),
                    "threshold": private["association_event"].get("sim_threshold"),
                    "selected_candidate_scores": {
                        "spatial": candidate.get("spatial_score"),
                        "visual": candidate.get("visual_score"),
                        "aggregate": candidate.get("aggregate_score"),
                    },
                }
            return payload

    def _atomic_rows(self, path: Path, values: dict[str, dict[str, Any]]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self.worklist:
                value = values.get(str(row["case_uid"]))
                if value is not None:
                    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def save_blind(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.reload()
            case_uid = str(payload.get("case_uid") or "")
            if case_uid not in self.by_case:
                raise ValueError("未知 case")
            public, _, _ = self._case_files(case_uid)
            codes = {str(row["code"]) for row in public["candidates"]}
            blind = validate_blind_label(payload, codes)
            saved = {
                "case_uid": case_uid,
                "event_uid": public["event_uid"],
                "blind": blind,
                "blind_saved_at_utc": utc_now(),
                "blind_submitted_when_mapper_latest_frame": self.manifest.get("mapper_latest_frame"),
            }
            self.drafts[case_uid] = saved
            self._atomic_rows(self.drafts_path, self.drafts)
            return {"saved": case_uid, "status": self.status()}

    def save_final(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.reload()
            case_uid = str(payload.get("case_uid") or "")
            draft = self.drafts.get(case_uid)
            if draft is None:
                raise ValueError("请先完成盲标")
            public, private, _ = self._case_files(case_uid)
            final = validate_final_label(payload, draft["blind"])
            derived = derive_event_label(draft["blind"], final, str(private["selected_target_code"]))
            label = {
                "schema_version": "experiment0-human-event-label/1.0",
                "case_uid": case_uid,
                "event_uid": public["event_uid"],
                "scene": public["scene"],
                "source_frame": public.get("source_frame"),
                "event_frame_idx": public.get("event_frame_idx"),
                "blind": draft["blind"],
                "final": final,
                "derived": derived,
                "selected_target_code": private["selected_target_code"],
                "saved_at_utc": utc_now(),
                "submitted_when_mapper_latest_frame": self.manifest.get("mapper_latest_frame"),
                "timeline": {
                    "s_event_frame": public.get("event_frame_idx"),
                    "d_human_submission_mapper_frame": self.manifest.get("mapper_latest_frame"),
                    "h": None,
                    "c": None,
                },
            }
            self.labels[case_uid] = label
            self._atomic_rows(self.labels_path, self.labels)
            return {"saved": case_uid, "derived": derived, "status": self.status()}

    def asset_path(self, case_uid: str, name: str) -> Path:
        with self.lock:
            self.reload()
            public, _, case_dir = self._case_files(case_uid)
            if name not in (public.get("displayed_asset_sha256") or {}):
                raise FileNotFoundError("asset not declared")
            path = (case_dir / name).resolve()
            if case_dir not in path.parents or not path.is_file():
                raise FileNotFoundError("invalid asset")
            return path


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>实验 0 关联事件标注</title>
<style>
:root{--bg:#f4f6fa;--card:#fff;--ink:#1c2538;--muted:#69738a;--line:#dfe4ee;--blue:#315efb;--green:#13845f;--red:#b8364f;--amber:#a86908}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:20;padding:12px 18px;background:rgba(244,246,250,.96);border-bottom:1px solid var(--line)}
.bar{max-width:1500px;margin:auto;display:flex;gap:14px;align-items:center}.title{font-size:18px;font-weight:800}.progress{height:10px;flex:1;background:#e0e5ef;border-radius:99px;overflow:hidden}.progress i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),#7b92ff)}
main{max-width:1500px;margin:18px auto;padding:0 18px;display:grid;grid-template-columns:minmax(0,1.55fr) minmax(370px,.72fr);gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 26px rgba(29,43,70,.07)}.evidence,.form{padding:18px}.form{position:sticky;top:78px;align-self:start;max-height:calc(100vh - 95px);overflow:auto}
.notice{padding:10px 12px;border-radius:9px;background:#edf2ff;margin:10px 0}.notice.warn{background:#fff5e5;border:1px solid #e6b65c}.notice.good{background:#ebf8f3;border:1px solid #92cdb8}.muted{color:var(--muted)}.mono{font:12px ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
.current{display:grid;grid-template-columns:1.25fr .75fr;gap:10px}.current img,.candidate img{display:block;width:100%;border-radius:9px;border:1px solid var(--line);cursor:zoom-in}.candidate-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.candidate{padding:12px;border:1px solid var(--line);border-radius:11px;background:#fbfcff}.candidate h3{margin:0 0 7px}.candidate-code{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--blue);color:white;margin-right:7px}
.section{border-top:1px solid var(--line);margin-top:18px;padding-top:17px}h1{font-size:23px;margin:0 0 6px}h2{font-size:18px;margin:0 0 10px}.field{margin:0 0 14px}.field label{display:block;font-weight:750;margin-bottom:5px}select,textarea,input{width:100%;padding:9px 10px;border:1px solid #cbd2df;border-radius:9px;background:white;font:inherit}textarea{min-height:70px;resize:vertical}.check-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:7px}.check{display:flex;align-items:center;gap:7px;padding:8px;border:1px solid var(--line);border-radius:8px}.check input{width:auto}.actions{display:grid;grid-template-columns:1fr 1.5fr 1fr;gap:8px;position:sticky;bottom:-18px;background:white;padding:12px 0 18px;border-top:1px solid var(--line)}button{border:0;border-radius:9px;padding:10px 12px;font-weight:750;cursor:pointer}button.primary{background:var(--blue);color:white}button.secondary{background:#edf0f6}button:disabled{opacity:.45}.error{color:var(--red);font-weight:700}.saved{color:var(--green);font-weight:700}.derived{font-size:17px;font-weight:800;color:var(--green)}.hidden{display:none}
@media(max-width:980px){main{grid-template-columns:1fr}.form{position:static;max-height:none}.candidate-grid,.current{grid-template-columns:1fr}}
</style></head>
<body><header><div class="bar"><div class="title">实验 0 · 关联事件两阶段标注</div><div class="progress"><i id="bar"></i></div><div id="count" class="muted"></div></div></header>
<main><section class="card evidence"><div id="caseMeta" class="muted"></div><h1 id="caseTitle"></h1>
<div class="notice">先只按现实物理实例判断。检测类别、mapper 选择、候选排名、分数、自动 GT 与抽样组在盲标阶段全部隐藏。</div>
<div class="section"><h2>当前 observation</h2><div class="current"><img id="currentContext"><img id="currentCrop"></div><div id="currentStats" class="muted"></div></div>
<div class="section"><h2>事件前候选节点 A–E</h2><div class="muted">每张图只使用该候选在事件发生前的历史；青色为候选历史，粉色为当前 observation。可以多选同一实例已有的 split 节点。</div><div id="candidates" class="candidate-grid"></div></div>
</section>
<aside class="card form"><div id="stageNotice"></div><div id="savedState" class="muted"></div>
<form id="blindForm"><h2>阶段 A：盲标身份</h2>
<div class="field"><label>1. 当前 processed mask 是什么情况？</label><select id="observationQuality" required><option value="">请选择</option><option>CLEAN_SINGLE_INSTANCE</option><option>BORDERLINE_SINGLE_INSTANCE</option><option>MIXED_MULTIPLE_INSTANCES</option><option>BACKGROUND_OR_FRAGMENT</option><option>DUPLICATE_PROPOSAL_SAME_FRAME</option><option>DYNAMIC_POSE_DEPTH_ERROR</option><option>GRANULARITY_AMBIGUOUS</option><option>INSUFFICIENT</option></select></div>
<div class="field"><label>2. 哪些候选与当前 observation 是同一物理实例？</label><div id="matchChecks" class="check-grid"></div><div class="muted">候选可多选；NONE_SHOWN 或 UNCERTAIN 必须单独选。</div></div>
<div class="field"><label>3. 用一句短语描述物理实例（可选）</label><textarea id="physicalNote" placeholder="例如：床右侧靠墙的白色床头柜；不要只写类别名"></textarea></div>
<button class="primary" type="submit">保存盲标并揭示 mapper 选择</button></form>
<form id="finalForm" class="hidden"><h2>阶段 B：揭示后裁决</h2><div id="revealBox" class="notice warn"></div>
<div class="field"><label>1. 系统所选节点在关联前是什么状态？</label><select id="targetState"><option value="">请选择</option><option>CLEAN_SINGLE_INSTANCE</option><option>ALREADY_CONTAMINATED</option><option>UNCERTAIN</option></select></div>
<div class="field"><label>2. 若 A–E 没有匹配，完整事件时地图里是否已有正确节点？</label><select id="outsideStatus"><option value="">请选择</option><option>NOT_NEEDED</option><option>MATCH_EXISTS_OUTSIDE</option><option>NO_MATCHING_NODE_EXISTS</option><option>UNCHECKED</option></select></div>
<div class="field"><label>3. 证据是否足够？</label><select id="evidenceStatus"><option value="">请选择</option><option>YES</option><option>PARTIAL</option><option>NO</option></select></div>
<div class="field"><label>4. 置信度 1–5</label><select id="confidence"><option value="">请选择</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div>
<div class="field"><label>5. 备注（PARTIAL/NO 必填）</label><textarea id="notes" placeholder="写明缺少哪一个视角、历史或粒度依据"></textarea></div>
<button class="primary" type="submit">保存最终标签</button></form>
<div id="completeBox" class="hidden"><h2>该例已完成</h2><div id="derived" class="derived"></div></div>
<div id="message"></div><div class="actions"><button id="prev" class="secondary">上一例</button><button id="nextOpen" class="primary">下一未完成</button><button id="next" class="secondary">下一例</button></div>
</aside></main>
<script>
let index=0,payload=null,blindStarted=performance.now(),finalStarted=null;
const $=id=>document.getElementById(id);const asset=(caseId,name)=>`/asset/${encodeURIComponent(caseId)}/${encodeURIComponent(name)}`;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function api(url,opts){const r=await fetch(url,opts);const j=await r.json();if(!r.ok)throw new Error(j.error||'请求失败');return j;}
function zoom(e){window.open(e.target.src,'_blank');}
function fillProgress(p){$('count').textContent=`完成 ${p.completed}/${p.total} · 盲标草稿 ${p.drafted} · mapper ${p.mapper_latest_frame??'-'}`;$('bar').style.width=(p.total?100*p.completed/p.total:0)+'%';}
function renderCase(data){payload=data;const p=data.public,caseId=p.case_uid;fillProgress(data.progress);$('caseMeta').textContent=`${data.position}/${data.total} · scene=${p.scene} · source frame=${p.source_frame} · event step=${p.event_frame_idx}`;$('caseTitle').textContent=`Case ${caseId}`;
$('currentContext').src=asset(caseId,p.current.context_asset);$('currentCrop').src=asset(caseId,p.current.crop_asset);$('currentContext').onclick=zoom;$('currentCrop').onclick=zoom;$('currentStats').textContent=`mask=${p.current.mask_area??'-'} px · valid depth=${p.current.valid_depth_ratio??'-'} · points=${p.current.stored_point_count??'-'}`;
$('candidates').innerHTML=p.candidates.map(c=>`<article class="candidate"><h3><span class="candidate-code">${esc(c.code)}</span>候选 ${esc(c.code)}</h3><div class="muted">显示 ${c.displayed_history_count}/${c.history_observation_count} 条历史 · ${c.history_frame_count} 帧</div><img src="${asset(caseId,c.history_asset)}"><img src="${asset(caseId,c.pcd_asset)}"></article>`).join('');document.querySelectorAll('.candidate img').forEach(img=>img.onclick=zoom);
$('matchChecks').innerHTML=p.candidates.map(c=>`<label class="check"><input type="checkbox" name="match" value="${esc(c.code)}">候选 ${esc(c.code)}</label>`).join('')+`<label class="check"><input type="checkbox" name="match" value="NONE_SHOWN">NONE_SHOWN</label><label class="check"><input type="checkbox" name="match" value="UNCERTAIN">UNCERTAIN</label>`;
document.querySelectorAll('input[name=match]').forEach(box=>box.onchange=()=>{if(box.checked&&(box.value==='NONE_SHOWN'||box.value==='UNCERTAIN'))document.querySelectorAll('input[name=match]').forEach(other=>{if(other!==box)other.checked=false});else if(box.checked)document.querySelectorAll('input[name=match]').forEach(other=>{if(other.value==='NONE_SHOWN'||other.value==='UNCERTAIN')other.checked=false});});
$('blindForm').classList.toggle('hidden',data.stage!=='blind');$('finalForm').classList.toggle('hidden',data.stage!=='reveal');$('completeBox').classList.toggle('hidden',data.stage!=='complete');$('savedState').textContent=data.stage==='blind'?'尚未保存':data.stage==='reveal'?'盲标已锁定；正在揭示后裁决':'最终标签已保存';
if(data.draft){$('observationQuality').value=data.draft.observation_quality;$('physicalNote').value=data.draft.physical_instance_note||'';}
if(data.reveal){const r=data.reveal;$('revealBox').innerHTML=`系统实际选择：<b>候选 ${esc(r.selected_target_code)}</b><br>selected spatial=${fmt(r.selected_candidate_scores.spatial)}, visual=${fmt(r.selected_candidate_scores.visual)}, aggregate=${fmt(r.selected_candidate_scores.aggregate)}<br>top1=${fmt(r.top1_score)}, top2=${fmt(r.top2_score)}, margin=${fmt(r.margin)}, threshold=${fmt(r.threshold)}`;finalStarted=performance.now();}
if(data.label)$('derived').textContent=`${data.label.derived.derived_status} → ${data.label.derived.derived_action}`;
$('prev').disabled=index<=0;$('next').disabled=index>=data.total-1;blindStarted=performance.now();$('message').textContent='';}
function fmt(v){return v===null||v===undefined?'-':Number(v).toFixed(4)}
async function load(i){index=Math.max(0,i);try{renderCase(await api(`/api/case?index=${index}`))}catch(e){$('message').innerHTML=`<p class="error">${esc(e.message)}</p>`}}
$('blindForm').onsubmit=async e=>{e.preventDefault();const matches=[...document.querySelectorAll('input[name=match]:checked')].map(x=>x.value);try{await api('/api/blind',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case_uid:payload.public.case_uid,observation_quality:$('observationQuality').value,matching_candidate_codes:matches,physical_instance_note:$('physicalNote').value,blind_review_seconds:(performance.now()-blindStarted)/1000})});await load(index)}catch(err){$('message').innerHTML=`<p class="error">${esc(err.message)}</p>`}};
$('finalForm').onsubmit=async e=>{e.preventDefault();try{const r=await api('/api/final',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case_uid:payload.public.case_uid,target_state:$('targetState').value,outside_candidate_status:$('outsideStatus').value,evidence_sufficient:$('evidenceStatus').value,confidence:$('confidence').value,notes:$('notes').value,final_review_seconds:(performance.now()-(finalStarted||performance.now()))/1000})});$('message').innerHTML=`<p class="saved">已保存：${esc(r.derived.derived_status)}</p>`;await load(index)}catch(err){$('message').innerHTML=`<p class="error">${esc(err.message)}</p>`}};
$('prev').onclick=()=>load(index-1);$('next').onclick=()=>load(index+1);$('nextOpen').onclick=async()=>{const s=await api('/api/status');let target=-1;for(let i=index+1;i<s.total;i++)if(!s.completed_indices.includes(i)){target=i;break}if(target<0)for(let i=0;i<s.total;i++)if(!s.completed_indices.includes(i)){target=i;break}if(target>=0)load(target);else $('message').innerHTML='<p class="saved">全部完成</p>'};load(0);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    store: AnnotationStore

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, payload: Any, status=HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1024 * 1024:
            raise ValueError("invalid request body")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/api/status":
                self._json(self.store.status())
                return
            if parsed.path == "/api/case":
                query = parse_qs(parsed.query)
                self._json(self.store.case_payload(int(query.get("index", ["0"])[0])))
                return
            if parsed.path.startswith("/asset/"):
                parts = parsed.path.split("/", 3)
                if len(parts) != 4:
                    raise FileNotFoundError("bad asset path")
                path = self.store.asset_path(unquote(parts[2]), unquote(parts[3]))
                data = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(data)
                return
            raise FileNotFoundError("not found")
        except (FileNotFoundError, IndexError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            body = self._body()
            if parsed.path == "/api/blind":
                self._json(self.store.save_blind(body))
                return
            if parsed.path == "/api/final":
                self._json(self.store.save_final(body))
                return
            raise FileNotFoundError("not found")
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("标注服务只能绑定 loopback；请用 SSH tunnel 访问")
    store = AnnotationStore(args.packet_root)
    Handler.store = store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({
        "status": "SERVING",
        "url": f"http://{args.host}:{args.port}/",
        "packet_root": str(args.packet_root.resolve()),
        "cases": store.status()["total"],
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

