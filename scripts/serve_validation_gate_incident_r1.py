#!/usr/bin/env python3
"""Loopback-only endpoint-first R1 UI for unique validation incidents."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mimetypes
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


SCHEMA_VERSION = "2.1.0"
REVIEW_FILENAME = "review_evidence.json"
REVIEW_MANIFEST_FILENAME = "review_evidence_manifest.json"
LABEL_FIELDS = (
    "reviewer_id",
    "evidence_sufficient",
    "final_state",
    "final_error_type",
    "review_seconds",
    "notes",
)
FINAL_STATES = {"CORRECT", "WRONG", "UNCLEAR"}
ERROR_TYPES = {
    "NOT_APPLICABLE",
    "FALSE_MERGE",
    "FALSE_SPLIT",
    "SPURIOUS_OBJECT",
    "MISSING_OBJECT",
    "WRONG_MEMBERSHIP",
    "GEOMETRY_CORRUPTION",
    "SEMANTIC_IDENTITY_ERROR",
    "OTHER",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    scene_id = str(row.get("scene_id") or "")
    incident_uid = str(row.get("incident_uid") or row.get("case_uid") or "")
    if not scene_id or not incident_uid:
        raise ValueError("worklist row needs scene_id and incident_uid")
    return scene_id, incident_uid


def validate_label(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = {field: payload.get(field) for field in LABEL_FIELDS}
    cleaned["reviewer_id"] = "R1"
    missing = [
        field
        for field in ("evidence_sufficient", "final_state", "final_error_type", "review_seconds")
        if cleaned.get(field) is None or cleaned.get(field) == ""
    ]
    if missing:
        raise ValueError("请完成这些字段：" + "、".join(missing))
    if cleaned["evidence_sufficient"] not in {"YES", "NO"}:
        raise ValueError("evidence_sufficient 的值不合法")
    if cleaned["final_state"] not in FINAL_STATES:
        raise ValueError("final_state 的值不合法")
    if cleaned["final_error_type"] not in ERROR_TYPES:
        raise ValueError("final_error_type 的值不合法")
    try:
        seconds = float(cleaned["review_seconds"])
    except (TypeError, ValueError) as exc:
        raise ValueError("review_seconds 必须是非负数") from exc
    if seconds < 0:
        raise ValueError("review_seconds 必须是非负数")
    cleaned["review_seconds"] = round(seconds, 1)
    notes = cleaned.get("notes")
    cleaned["notes"] = None if notes is None or not str(notes).strip() else str(notes).strip()

    evidence = cleaned["evidence_sufficient"]
    final_state = cleaned["final_state"]
    error_type = cleaned["final_error_type"]
    if evidence == "NO" and final_state != "UNCLEAR":
        raise ValueError("证据不足时，最终状态必须选 UNCLEAR，不能把看不清当成正确或错误")
    if evidence == "YES" and final_state == "UNCLEAR":
        raise ValueError("证据充分时必须判断 CORRECT 或 WRONG")
    if final_state == "WRONG" and error_type == "NOT_APPLICABLE":
        raise ValueError("最终状态为 WRONG 时请选择一种可见的最终错误类型")
    if final_state != "WRONG" and error_type != "NOT_APPLICABLE":
        raise ValueError("最终状态不是 WRONG 时，错误类型必须为 NOT_APPLICABLE")
    if error_type == "OTHER" and not cleaned["notes"]:
        raise ValueError("错误类型为 OTHER 时，请在备注中用一句话说明可见错误")
    return cleaned


class ReviewStore:
    def __init__(self, validation_root: Path):
        self.root = validation_root.resolve()
        self.labels_dir = self.root / "labels"
        self.worklist_path = self.labels_dir / "r1_worklist.jsonl"
        self.labels_path = self.labels_dir / "labels_r1.jsonl"
        self.worklist = read_jsonl(self.worklist_path)
        manifest_path = self.root / REVIEW_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ValueError(f"缺少 {REVIEW_MANIFEST_FILENAME}，R1 暂停")
        self.review_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.review_manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("review evidence schema 不兼容")
        if self.review_manifest.get("status") not in {"READY", "READY_WITH_DECLARED_LIMITATIONS"}:
            raise ValueError("review evidence manifest 尚未就绪")
        if self.review_manifest.get("worklist_sha256") != hashlib.sha256(self.worklist_path.read_bytes()).hexdigest():
            raise ValueError("review evidence 与当前 incident worklist 不一致")
        if int(self.review_manifest.get("case_count", -1)) != len(self.worklist):
            raise ValueError("review evidence 案例数与 incident worklist 不一致")

        self.by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.worklist:
            if row.get("annotation_unit") != "incident":
                raise ValueError("R1 worklist 不是 incident 级")
            key = case_key(row)
            if key in self.by_key:
                raise ValueError(f"duplicate incident: {key}")
            self.by_key[key] = row
        self.manifest_by_key = {}
        for item in self.review_manifest.get("cases") or []:
            key = (str(item.get("scene_id") or ""), str(item.get("incident_uid") or item.get("case_uid") or ""))
            if not all(key) or key in self.manifest_by_key:
                raise ValueError("review evidence manifest 含无效或重复 incident")
            self.manifest_by_key[key] = item
        if set(self.manifest_by_key) != set(self.by_key):
            raise ValueError("review evidence manifest 与 incident worklist 集合不一致")

        self.display_rows = sorted(
            self.worklist,
            key=lambda row: hashlib.sha256(
                f"R1_ENDPOINT_BLIND_V2:{case_key(row)[0]}:{case_key(row)[1]}".encode()
            ).hexdigest(),
        )
        self.labels: dict[tuple[str, str], dict[str, Any]] = {}
        self.lock = threading.Lock()
        if self.labels_path.exists():
            for row in read_jsonl(self.labels_path):
                key = case_key(row)
                if key not in self.by_key:
                    raise ValueError(f"labels_r1 contains unknown incident: {key}")
                self.labels[key] = row

    @property
    def total(self) -> int:
        return len(self.display_rows)

    def status(self) -> dict[str, Any]:
        completed_indices = [
            index for index, row in enumerate(self.display_rows) if case_key(row) in self.labels
        ]
        return {
            "completed": len(self.labels),
            "total": self.total,
            "complete": len(self.labels) == self.total,
            "completed_indices": completed_indices,
        }

    def _case_dir(self, row: dict[str, Any]) -> Path:
        path = Path(str(row["case_dir"])).resolve()
        expected_root = (self.root / "cases" / str(row["scene_id"])).resolve()
        if expected_root != path and expected_root not in path.parents:
            raise ValueError("case_dir escaped expected scene root")
        return path

    def _assets(self, case_dir: Path, review: dict[str, Any]) -> list[str]:
        declared = review.get("displayed_asset_sha256")
        if not isinstance(declared, dict):
            raise ValueError("人类证据投影缺少页面图片哈希清单")
        names = []
        for name, expected_sha in declared.items():
            relative = Path(str(name))
            path = (case_dir / relative).resolve()
            if case_dir != path and case_dir not in path.parents:
                raise ValueError("页面图片路径逃逸案例目录")
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not path.is_file():
                raise ValueError(f"页面图片不存在或类型不允许：{name}")
            if sha256_file(path) != expected_sha:
                raise ValueError(f"页面图片在证据生成后发生变化：{name}")
            names.append(relative.as_posix())
        priority = {
            "review_final_objects_relative.png": 0,
            "review_final_objects_detail.png": 1,
            "timeline.jpg": 2,
            "pcd_overlay.png": 3,
        }
        return sorted(names, key=lambda name: (priority.get(name, 4), name))

    def case_payload(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.total:
            raise IndexError("case index out of range")
        row = self.display_rows[index]
        key = case_key(row)
        case_dir = self._case_dir(row)
        case_path = case_dir / "case.json"
        review_path = case_dir / REVIEW_FILENAME
        if not review_path.is_file():
            raise ValueError(f"缺少 incident 人类证据投影：{key[0]}/{key[1]}")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        manifest_item = self.manifest_by_key[key]
        if manifest_item.get("review_evidence_sha256") != sha256_file(review_path):
            raise ValueError(f"人类证据投影与顶层 manifest 哈希不一致：{key[0]}/{key[1]}")
        if review.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"人类证据投影版本不兼容：{key[0]}/{key[1]}")
        if review.get("scene_id") != key[0] or review.get("case_uid") != key[1]:
            raise ValueError(f"人类证据投影绑定了错误 incident：{key[0]}/{key[1]}")
        if review.get("source_case_json_sha256") != sha256_file(case_path):
            raise ValueError(f"case.json 在人类证据生成后发生变化：{key[0]}/{key[1]}")
        incident = review.get("incident") or {}
        if incident.get("incident_uid") != key[1]:
            raise ValueError(f"review evidence incident UID mismatch：{key[0]}/{key[1]}")

        # R1 is deliberately endpoint-blind to checker name, stage and score.
        safe_review = copy.deepcopy(review)
        for field in ("finding_uid", "checker_id", "stage", "subtype"):
            safe_review.pop(field, None)
        safe_incident = safe_review.get("incident") or {}
        for field in ("representative_finding_uid", "member_finding_uids", "checker_ids", "stages", "subtypes", "blocked_checker_ids"):
            safe_incident.pop(field, None)
        label = self.labels.get(key)
        return {
            "index": index,
            "position": index + 1,
            "total": self.total,
            "scene_id": key[0],
            "case_uid": key[1],
            "review_evidence": safe_review,
            "assets": self._assets(case_dir, review),
            "label": {field: label.get(field) for field in LABEL_FIELDS} if label else None,
            "completed": label is not None,
            "progress": self.status(),
        }

    def asset_path(self, scene_id: str, incident_uid: str, relative: str) -> Path:
        row = self.by_key.get((scene_id, incident_uid))
        if row is None:
            raise FileNotFoundError("unknown incident")
        case_dir = self._case_dir(row)
        target = (case_dir / relative).resolve()
        if case_dir != target and case_dir not in target.parents:
            raise FileNotFoundError("invalid asset path")
        if not target.is_file():
            raise FileNotFoundError("missing asset")
        review = json.loads((case_dir / REVIEW_FILENAME).read_text(encoding="utf-8"))
        name = target.relative_to(case_dir).as_posix()
        expected_sha = (review.get("displayed_asset_sha256") or {}).get(name)
        if not expected_sha or sha256_file(target) != expected_sha:
            raise FileNotFoundError("asset is not declared by the frozen review evidence or its hash changed")
        return target

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = (str(payload.get("scene_id") or ""), str(payload.get("case_uid") or ""))
        if key not in self.by_key:
            raise ValueError("未知 incident，未保存")
        label = validate_label(payload)
        with self.lock:
            output = dict(self.by_key[key])
            output.update(label)
            self.labels[key] = output
            temporary = self.labels_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for row in self.worklist:
                    saved = self.labels.get(case_key(row))
                    if saved is not None:
                        handle.write(json.dumps(saved, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.labels_path)
        result = self.status()
        result["saved_case"] = f"{key[0]}/{key[1]}"
        return result


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R1 最终 endpoint 复核</title>
<style>
:root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687289;--line:#dfe4ee;--blue:#315efb;--green:#15825d;--red:#b83232;--amber:#a46600;--shadow:0 8px 28px rgba(28,42,70,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}header{position:sticky;top:0;z-index:20;background:rgba(245,247,251,.96);border-bottom:1px solid var(--line);padding:12px 20px}.bar{max-width:1480px;margin:auto;display:flex;gap:15px;align-items:center}.title{font-size:18px;font-weight:800}.progress{height:10px;background:#e2e7f0;border-radius:99px;overflow:hidden;flex:1}.progress i{display:block;height:100%;background:var(--blue);width:0}.muted{color:var(--muted)}main{max-width:1480px;margin:18px auto;padding:0 18px;display:grid;grid-template-columns:minmax(0,1.55fr) minmax(360px,.72fr);gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}.evidence,.form{padding:18px}.form{position:sticky;top:78px;align-self:start;max-height:calc(100vh - 96px);overflow:auto}.eyebrow{color:var(--blue);font-weight:750}.case-title{font-size:23px;margin:3px 0}.notice{padding:11px 13px;border-radius:9px;margin:12px 0;background:#eef3ff}.pass{background:#ecf8f2;border:1px solid #8bcbb4}.warning{background:#fff7e8;border:1px solid #e4b45f}.section{border-top:1px solid var(--line);margin-top:19px;padding-top:18px}.section h2{font-size:18px;margin:0 0 9px}.step{display:inline-grid;place-items:center;width:25px;height:25px;border-radius:50%;background:var(--blue);color:white;font-size:13px;margin-right:7px}.contract{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.contract div,.entity{border:1px solid var(--line);border-radius:9px;padding:10px;background:#fbfcff}.contract b{display:block;font-size:12px;color:var(--muted)}.gallery{display:grid;grid-template-columns:1fr;gap:11px}.figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fafbfe}.figure img{display:block;width:100%;cursor:zoom-in}.figure figcaption{padding:8px 10px;color:var(--muted);font-size:12px}.cards{display:grid;gap:9px}.entity h3{font-size:15px;margin:0 0 6px}.kv{display:grid;grid-template-columns:minmax(130px,.42fr) minmax(0,1fr);gap:3px 10px}.kv>div:nth-child(odd){color:var(--muted)}.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.views{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.view{border:1px solid var(--line);border-radius:9px;padding:8px}.view-images{display:grid;grid-template-columns:1fr 1fr;gap:5px}.view img{width:100%;border-radius:6px;cursor:zoom-in}details{border:1px solid var(--line);border-radius:9px;padding:9px 11px;margin:10px 0}summary{font-weight:750;cursor:pointer}.field{margin-bottom:14px}.field label{font-weight:750;display:block;margin-bottom:5px}.required:after{content:" *";color:var(--red)}select,textarea{width:100%;border:1px solid #cbd2df;border-radius:9px;padding:9px 10px;background:white;font:inherit}textarea{min-height:68px}.help{font-size:12px;color:var(--muted)}.logic{padding:9px 10px;border-radius:8px;background:#f2f4f8;font-size:13px}.logic.error{background:#fff0f0;color:#972b2b}.logic.ok{background:#ecf8f2;color:#176447}.actions{display:grid;grid-template-columns:1fr 1.35fr 1fr;gap:8px;position:sticky;bottom:-18px;background:white;padding:12px 0 18px;border-top:1px solid var(--line)}button{border:0;border-radius:9px;padding:10px;font-weight:750;cursor:pointer}.primary{background:var(--blue);color:white}.secondary{background:#edf0f6}.saved{color:var(--green);font-weight:750}.error{color:var(--red);font-weight:750}.hidden{display:none}.type-guide dt{font-weight:750}.type-guide dd{margin:0 0 8px;color:var(--muted)}
@media(max-width:980px){main{grid-template-columns:1fr}.form{position:static;max-height:none}.contract,.views{grid-template-columns:1fr}}
</style></head><body>
<header><div class="bar"><div class="title">R1 最终 endpoint 复核</div><div class="progress"><i id="progressBar"></i></div><div id="progressText" class="muted">加载中</div></div></header>
<main><section class="card evidence">
<div id="position" class="eyebrow"></div><h1 class="case-title">判断最终地图，而不是判断规则</h1><div id="caseMeta" class="muted"></div>
<div class="notice"><b>同一组 active final objects 只展示一次。</b>不同 observation、不同 checker、不同阶段只作为这个 final endpoint 的内部证据，不再让你重复判断同一最终对象。R1 不要求你判断阶段根因或修复动作。</div>
<div id="contractStatus"></div><div id="contractGrid" class="contract"></div><details id="gapWrap" class="hidden"><summary>证据缺口</summary><div id="gaps"></div></details>

<div class="section"><h2><span class="step">1</span>先看最终地图对象</h2>
<div class="notice pass">这是从哈希锁定的最终 map pickle 直接读取的对象、完整成员集合和点云。你的核心问题只有一个：<b>最终地图中可见的对象状态是否正确？</b></div>
<div id="finalRecords" class="cards"></div><div id="finalGallery" class="gallery"></div></div>

<div class="section"><h2><span class="step">2</span>用代表视图核对对象身份与成员</h2>
<div class="muted">视图只用于回答最终对象由什么组成；页面明确显示抽样覆盖，不把代表视图冒充全部成员。</div><div id="objectRecords" class="cards"></div><div id="representativeViews" class="views"></div></div>

<div class="section"><h2><span class="step">3</span>必要时查看触发 observation 上下文</h2>
<details><summary>展开与最终对象严格同源的代表性 RGB / mask / depth / 3D</summary><div class="notice warning">这里显示的是一条精确的代表性触发上下文，不冒充该 endpoint 的全部报警历史；完整 linked triggers 已冻结供后续专家追踪。R1 不据此猜测“最早哪一步错了”。</div><div id="triggerGallery" class="gallery"></div><div id="triggerRecords" class="cards"></div></details>
<details><summary>展开系统当时的关联记录</summary><div id="decisionRecords" class="cards"></div></details></div>

<div class="section"><h2><span class="step">4</span>追溯材料</h2><details><summary>展开旧 packet 图片</summary><div class="notice warning"><code>pcd_overlay.png</code> 是抽样 observation 叠加，不是最终对象；最终判断以上面第 1 节为准。</div><div id="rawGallery" class="gallery"></div></details></div>
</section>

<aside class="card form"><h2>只填写最终状态</h2><div id="savedState" class="muted">尚未保存</div>
<details open><summary>三个字段怎么选</summary>
<p><b>证据充分 YES：</b>最终点云、对象视图和成员足以让你选 CORRECT 或 WRONG。<br><b>证据不足 NO：</b>最终对象缺失、关键视角缺失或仍有两种同样合理的解释；随后固定选 UNCLEAR。</p>
<p><b>CORRECT：</b>最终身份、节点数、成员和几何没有可见错误。上游曾出现重复 proposal 或低 margin，但最终状态正确，也选它。<br><b>WRONG：</b>错误仍真实存在于最终地图。<br><b>UNCLEAR：</b>当前证据不能可靠区分对错，不等于误报。</p>
<dl class="type-guide"><dt>FALSE_MERGE 错融</dt><dd>两个或更多真实物体被保留在同一最终节点。</dd><dt>FALSE_SPLIT 错分</dt><dd>同一真实物体仍保留成多个最终节点。</dd><dt>SPURIOUS_OBJECT 伪对象</dt><dd>最终节点只是噪声、残片或背景。</dd><dt>MISSING_OBJECT 缺失对象</dt><dd>应该存在的真实对象在最终地图中没有有效节点。</dd><dt>WRONG_MEMBERSHIP 成员错归属</dt><dd>有效 observation 被放进不属于它的最终对象。</dd><dt>GEOMETRY_CORRUPTION 几何损坏</dt><dd>最终点云、位置、尺度或形状明显错误。</dd><dt>SEMANTIC_IDENTITY_ERROR 语义身份错</dt><dd>几何节点存在，但其稳定身份明显错误；同义词不算。</dd></dl>
</details>
<div class="field"><label class="required">证据足以判断最终状态吗</label><select id="evidence_sufficient"><option value="">请选择</option><option value="YES">YES — 足以判断</option><option value="NO">NO — 不足，不能猜</option></select></div>
<div class="field"><label class="required">最终地图对象状态</label><select id="final_state"><option value="">请选择</option><option value="CORRECT">CORRECT — 最终正确</option><option value="WRONG">WRONG — 最终仍有错</option><option value="UNCLEAR">UNCLEAR — 无法可靠判断</option></select></div>
<div class="field"><label class="required">可见的最终错误类型</label><select id="final_error_type"><option value="">请选择</option><option value="NOT_APPLICABLE">NOT_APPLICABLE — 不适用</option><option value="FALSE_MERGE">FALSE_MERGE — 错融</option><option value="FALSE_SPLIT">FALSE_SPLIT — 错分</option><option value="SPURIOUS_OBJECT">SPURIOUS_OBJECT — 伪对象</option><option value="MISSING_OBJECT">MISSING_OBJECT — 缺失对象</option><option value="WRONG_MEMBERSHIP">WRONG_MEMBERSHIP — 成员错归属</option><option value="GEOMETRY_CORRUPTION">GEOMETRY_CORRUPTION — 几何损坏</option><option value="SEMANTIC_IDENTITY_ERROR">SEMANTIC_IDENTITY_ERROR — 语义身份错</option><option value="OTHER">OTHER — 其他</option></select><div class="help">只有 WRONG 才选择具体类型；其余固定为 NOT_APPLICABLE。</div></div>
<div class="field"><label>备注</label><textarea id="notes" placeholder="可留空；OTHER 时请说明"></textarea></div>
<div id="logicHint" class="logic">填写后检查逻辑一致性。</div><div id="message"></div>
<div class="actions"><button id="prev" class="secondary">上一例</button><button id="save" class="primary">保存本例</button><button id="next" class="secondary">下一例</button></div>
</aside></main>
<script>
const $=id=>document.getElementById(id);let payload=null,current=0,openedAt=Date.now(),priorSeconds=0;const fields=['evidence_sufficient','final_state','final_error_type','notes'];
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}function fmt(v,d=4){if(v===null||v===undefined||v==='')return '—';if(typeof v==='number')return Number.isInteger(v)?String(v):v.toFixed(d);if(typeof v==='object')return JSON.stringify(v);return String(v);}function assetUrl(n){return `/asset?scene=${encodeURIComponent(payload.scene_id)}&case=${encodeURIComponent(payload.case_uid)}&file=${encodeURIComponent(n)}`;}function kv(rows){return `<div class="kv">${rows.map(([k,v])=>`<div>${esc(k)}</div><div>${esc(fmt(v))}</div>`).join('')}</div>`;}function fig(root,n,c){if(!n)return;const f=document.createElement('figure');f.className='figure';const i=document.createElement('img');i.loading='lazy';i.src=assetUrl(n);i.onclick=()=>window.open(i.src,'_blank');const cap=document.createElement('figcaption');cap.textContent=c||n;f.append(i,cap);root.appendChild(f);}
function contract(r){const c=r.evidence_contract||{},g=c.critical_gaps||[],bad=g.some(x=>x.critical);$('contractStatus').className='notice '+(bad?'warning':'pass');$('contractStatus').innerHTML=bad?'<b>本例存在关键证据缺口。</b>若它阻断最终状态判断，请选证据不足。':'<b>证据已对齐。</b>页面资产、ledger 引用和最终 map object 均可追溯。';$('contractGrid').innerHTML=`<div><b>Artifact 哈希</b>${c.artifact_hashes_match?'一致':'异常'}</div><div><b>最终对象核对</b>${c.exact_final_map_linkage?'完整一致':'异常'}</div><div><b>证据状态</b>${esc(c.fidelity_status||'')}</div>`;$('gapWrap').classList.toggle('hidden',!g.length);$('gaps').innerHTML=g.map(x=>`<div class="notice warning"><b>${esc(x.code)}</b><br>${esc(x.message)}</div>`).join('');}
function finalObjects(r){const root=$('finalRecords');root.innerHTML='';const o=r.final_outcome||{},i=r.incident||{},c=i.trigger_context_coverage||{};root.innerHTML=`<div class="notice pass"><b>结构事实：${esc(o.machine_resolution_status||o.status||'')}</b><br>final endpoint：${(i.final_owner_uids||[]).length} 个 active object；内部关联 triggers：${fmt(c.linked_total??(i.all_trigger_observation_uids||[]).length,0)}；本页代表性上下文：${fmt(c.displayed??(i.trigger_observation_uids||[]).length,0)}。该结构事实不是人工答案。</div>`;for(const x of r.final_objects||[]){const card=document.createElement('div');card.className='entity';const role=x.endpoint_role==='INCIDENT_FINAL_OWNER'?'本 endpoint 的最终对象':'上下文候选最终对象';card.innerHTML=`<h3>${esc(x.object_alias)} · ${esc(x.class_name||'unknown')} · ${role}</h3>${kv([['完整成员 / 帧数',`${fmt(x.member_count,0)} / ${fmt(x.unique_frame_count,0)}`],['帧范围',`${fmt(x.first_frame,0)} – ${fmt(x.last_frame,0)}`],['最终点数',x.n_points],['bbox center',x.bbox_center],['bbox extent',x.bbox_extent],['成员类别统计',x.observed_class_histogram],['merge ancestry',x.parent_or_merged_from_object_uids],['成员与 final pickle',x.membership_matches_final_output?'一致':'不一致'],['点数与 final pickle',x.point_count_matches_final_output?'一致':'不一致']])}<div class="mono">${esc(x.object_uid)}</div>`;root.appendChild(card);}const gallery=$('finalGallery');gallery.innerHTML='';for(const n of (r.assets||{}).final_object_geometry||[])fig(gallery,n,n.includes('relative')?'最终对象统一世界坐标':'最终对象逐个放大');}
function objects(r){const root=$('objectRecords');root.innerHTML='';for(const x of r.objects||[]){const v=x.trigger_or_decision_version||{},c=x.representative_view_coverage||{};const card=document.createElement('div');card.className='entity';card.innerHTML=`<h3>${esc(x.object_alias)} · ${esc(x.final_status)}</h3>${kv([['当时类别',v.class_name],['当时成员 / 点数',v.object_version_uid?`${fmt(v.member_count,0)} / ${fmt(v.n_points,0)}`:'未记录角色版本'],['代表视图覆盖',`${fmt(c.selected,0)} / ${fmt(c.total,0)}`],['最终去向',(x.resolved_final_aliases||[]).join(', ')||'无 active final object']])}`;root.appendChild(card);}const views=$('representativeViews');views.innerHTML='';for(const x of r.representative_views||[]){const card=document.createElement('div');card.className='view';card.innerHTML=`<b>${esc((x.object_aliases||[]).join(', ')||'未绑定对象')} · frame ${esc(fmt(x.frame_number,0))} · ${esc(x.class_name||'unknown')}</b><div class="view-images"></div><div class="mono">${esc(x.obs_uid)}</div>`;const images=card.querySelector('.view-images');for(const [kind,n] of Object.entries(x.assets||{})){if(!n)continue;const img=document.createElement('img');img.loading='lazy';img.src=assetUrl(n);img.alt=kind;img.onclick=()=>window.open(img.src,'_blank');images.appendChild(img);}views.appendChild(card);}}
function context(r){const gallery=$('triggerGallery');gallery.innerHTML='';for(const x of r.trigger_observations||[])fig(gallery,x.panel_asset,`${x.observation_alias}：同源 RGB / raw mask / processed mask / depth / post-DBSCAN PCD`);const records=$('triggerRecords');records.innerHTML='';for(const x of r.trigger_observations||[]){const card=document.createElement('div');card.className='entity';card.innerHTML=`<h3>${esc(x.observation_alias)} · ${esc(x.class_name||'unknown')}</h3>${kv([['帧',x.frame_number],['置信度',x.confidence],['raw → processed mask',`${fmt(x.raw_mask_area,0)} → ${fmt(x.processed_mask_area,0)}`],['有效深度比例',x.valid_depth_ratio],['post-DBSCAN 点数',x.n_points],['最终归属',(x.final_owner_aliases||[]).join(', ')||'无 active owner']])}<div class="mono">${esc(x.obs_uid)}</div>`;records.appendChild(card);}const decisions=$('decisionRecords');decisions.innerHTML='';for(const x of r.association_decisions||[]){const card=document.createElement('div');card.className='entity';card.innerHTML=`<h3>${esc(x.obs_uid)}</h3>${kv([['系统动作',x.decision],['目标对象',`${x.target_object_alias||''} · ${x.target_object_uid||''}`],['Top1 / Top2',`${fmt(x.top1_score)} / ${fmt(x.top2_score)}`],['margin',x.margin],['阈值',x.sim_threshold]])}`;decisions.appendChild(card);}}
function raw(a){const root=$('rawGallery');root.innerHTML='';for(const n of a.filter(x=>!x.startsWith('review_')))fig(root,n,n==='pcd_overlay.png'?'抽样 observation 点云叠加（不是最终 object）':n);}function progress(p){$('progressBar').style.width=(100*p.completed/p.total)+'%';$('progressText').textContent=`已完成 ${p.completed} / ${p.total}`;}
function normalize(){const e=$('evidence_sufficient').value,s=$('final_state').value,t=$('final_error_type');if(e==='NO'){$('final_state').value='UNCLEAR';t.value='NOT_APPLICABLE';}else if(s==='CORRECT'||s==='UNCLEAR'){t.value='NOT_APPLICABLE';}t.disabled=$('final_state').value!=='WRONG';}
function logic(){normalize();const e=$('evidence_sufficient').value,s=$('final_state').value,t=$('final_error_type').value,n=$('notes').value.trim(),p=[];if(e==='YES'&&s==='UNCLEAR')p.push('证据充分时必须选 CORRECT 或 WRONG。');if(e==='NO'&&s!=='UNCLEAR')p.push('证据不足时必须选 UNCLEAR。');if(s==='WRONG'&&(!t||t==='NOT_APPLICABLE'))p.push('WRONG 时请选择具体错误类型。');if(s!=='WRONG'&&t&&t!=='NOT_APPLICABLE')p.push('不是 WRONG 时错误类型应为 NOT_APPLICABLE。');if(t==='OTHER'&&!n)p.push('OTHER 时请写备注。');const r=$('logicHint');r.className='logic '+(p.length?'error':'ok');r.textContent=p.length?p.join(' '):'当前选择逻辑一致。';}
function setForm(l){for(const f of fields)$(f).value=l?.[f]??'';priorSeconds=Number(l?.review_seconds||0);$('savedState').className=l?'saved':'muted';$('savedState').textContent=l?'本例已有保存记录，可修改后覆盖':'尚未保存';openedAt=Date.now();logic();}
async function load(index){$('message').textContent='';const res=await fetch('/api/case?index='+index,{cache:'no-store'});const data=await res.json();if(!res.ok){$('message').className='error';$('message').textContent=data.error||'读取失败';return;}payload=data;current=data.index;progress(data.progress);$('position').textContent=`第 ${data.position} / ${data.total} 个 final endpoint`;$('caseMeta').textContent=`场景 ${data.scene_id} · endpoint ${data.case_uid}`;contract(data.review_evidence);finalObjects(data.review_evidence);objects(data.review_evidence);context(data.review_evidence);raw(data.assets);setForm(data.label);$('prev').disabled=current===0;$('next').disabled=current===data.total-1;window.scrollTo({top:0,behavior:'smooth'});}
async function save(){normalize();const body={scene_id:payload.scene_id,case_uid:payload.case_uid,review_seconds:priorSeconds+(Date.now()-openedAt)/1000};for(const f of fields)body[f]=$(f).value;const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await res.json();if(!res.ok){$('message').className='error';$('message').textContent=data.error||'保存失败';return false;}$('message').className='saved';$('message').textContent=`已保存。总进度 ${data.completed}/${data.total}`;priorSeconds=body.review_seconds;openedAt=Date.now();progress(data);return true;}
$('prev').onclick=()=>load(Math.max(0,current-1));$('next').onclick=()=>load(Math.min(payload.total-1,current+1));$('save').onclick=save;for(const f of fields)$(f).addEventListener('change',logic);document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();save();}});load(0);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    store: ReviewStore

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.address_string()} {fmt % args}")

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                data = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif parsed.path == "/api/status":
                self._send_json(self.store.status())
            elif parsed.path == "/api/case":
                self._send_json(self.store.case_payload(int(query.get("index", ["0"])[0])))
            elif parsed.path == "/asset":
                path = self.store.asset_path(
                    query.get("scene", [""])[0],
                    query.get("case", [""])[0],
                    query.get("file", [""])[0],
                )
                data = path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, IndexError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/save":
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("expected JSON object")
            self._send_json(self.store.save(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = ReviewStore(args.validation_root)
    Handler.store = store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "status": "READY",
                "protocol": "final_endpoint_r1_v2_1",
                "url": f"http://{args.host}:{args.port}",
                "completed": store.status()["completed"],
                "total": store.total,
                "labels": str(store.labels_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
