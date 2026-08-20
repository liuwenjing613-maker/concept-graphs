#!/usr/bin/env python3
"""Local-only browser UI for Audit Validity Gate R1 human review.

The reviewer sees evidence packets but not cohort, rule certainty, sampling
weight, or review score. Saves are atomic and preserve the frozen worklist
metadata. The server intentionally binds to loopback by default.
"""

from __future__ import annotations

import argparse
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


LABEL_FIELDS = (
    "reviewer_id",
    "evidence_sufficient",
    "finding_correct",
    "root_stage_correct",
    "physical_interpretation",
    "downstream_harm",
    "harm_confidence",
    "repair_action",
    "repair_locality",
    "repair_confidence",
    "alternative_explanation",
    "review_seconds",
    "notes",
)

ENUMS = {
    "evidence_sufficient": {"YES", "NO", "PARTIAL"},
    "finding_correct": {"YES", "NO", "UNCERTAIN"},
    "root_stage_correct": {"YES", "NO", "UNCERTAIN", "NOT_APPLICABLE"},
    "downstream_harm": {
        "NONE",
        "LOCAL_WEIGHTING_BIAS",
        "WRONG_OBSERVATION_MEMBERSHIP",
        "FALSE_SPLIT_DUPLICATE_NODE",
        "FALSE_MERGE_IDENTITY_POLLUTION",
        "GEOMETRY_CORRUPTION",
        "RELATION_POLLUTION",
        "UNKNOWN",
    },
    "repair_action": {
        "NONE",
        "DROP_OBSERVATION",
        "REASSIGN_OBSERVATION",
        "MERGE_OBJECTS",
        "SPLIT_OBJECT",
        "RECOMPUTE_GEOMETRY",
        "DOWNWEIGHT_EVIDENCE",
        "NEED_MORE_VIEW",
        "UNKNOWN",
    },
    "repair_locality": {"LOCAL", "MULTI_OBJECT", "GLOBAL", "NOT_APPLICABLE"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(value)
    return rows


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    scene = str(row.get("scene_id", ""))
    uid = str(row.get("case_uid") or row.get("finding_uid") or "")
    if not scene or not uid:
        raise ValueError("worklist row needs scene_id and case_uid/finding_uid")
    return scene, uid


def validate_label(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = {field: payload.get(field) for field in LABEL_FIELDS}
    cleaned["reviewer_id"] = "R1"
    required = set(LABEL_FIELDS) - {"alternative_explanation", "notes"}
    missing = sorted(field for field in required if cleaned.get(field) is None or cleaned.get(field) == "")
    if missing:
        raise ValueError("请完成这些字段：" + "、".join(missing))
    for field, allowed in ENUMS.items():
        if cleaned.get(field) not in allowed:
            raise ValueError(f"{field} 的值不合法")
    for field in ("harm_confidence", "repair_confidence"):
        value = cleaned.get(field)
        if isinstance(value, bool):
            raise ValueError(f"{field} 必须是 1–5")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} 必须是 1–5") from exc
        if number < 1 or number > 5:
            raise ValueError(f"{field} 必须是 1–5")
        cleaned[field] = number
    try:
        seconds = float(cleaned.get("review_seconds"))
    except (TypeError, ValueError) as exc:
        raise ValueError("review_seconds 必须是非负数") from exc
    if seconds < 0:
        raise ValueError("review_seconds 必须是非负数")
    cleaned["review_seconds"] = round(seconds, 1)
    for field in ("physical_interpretation",):
        value = cleaned.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("请用一句短语写明你看到的真实物理关系")
        cleaned[field] = value.strip()
    for field in ("alternative_explanation", "notes"):
        value = cleaned.get(field)
        cleaned[field] = None if value is None or not str(value).strip() else str(value).strip()
    return cleaned


class ReviewStore:
    def __init__(self, validation_root: Path):
        self.root = validation_root.resolve()
        self.labels_dir = self.root / "labels"
        self.worklist_path = self.labels_dir / "r1_worklist.jsonl"
        self.labels_path = self.labels_dir / "labels_r1.jsonl"
        self.worklist = read_jsonl(self.worklist_path)
        self.by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.worklist:
            key = case_key(row)
            if key in self.by_key:
                raise ValueError(f"duplicate worklist case: {key}")
            self.by_key[key] = row
        self.display_rows = sorted(
            self.worklist,
            key=lambda row: hashlib.sha256(
                f"R1_BLIND_V1:{case_key(row)[0]}:{case_key(row)[1]}".encode()
            ).hexdigest(),
        )
        self.labels: dict[tuple[str, str], dict[str, Any]] = {}
        self.lock = threading.Lock()
        if self.labels_path.exists():
            for row in read_jsonl(self.labels_path):
                key = case_key(row)
                if key not in self.by_key:
                    raise ValueError(f"labels_r1 contains unknown case: {key}")
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

    def _assets(self, case_dir: Path) -> list[str]:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        names = [
            path.relative_to(case_dir).as_posix()
            for path in case_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ]
        priority = {
            "timeline.jpg": 0,
            "pcd_overlay.png": 1,
        }
        return sorted(names, key=lambda name: (priority.get(name, 2), name))

    def case_payload(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.total:
            raise IndexError("case index out of range")
        row = self.display_rows[index]
        key = case_key(row)
        case_dir = self._case_dir(row)
        case_json_path = case_dir / "case.json"
        case_json = json.loads(case_json_path.read_text(encoding="utf-8"))
        safe_case = {
            field: case_json.get(field)
            for field in (
                "finding_uid",
                "checker_id",
                "stage",
                "subtype",
                "scope",
                "proven_facts",
                "hypotheses",
                "vetoes",
                "missing_evidence",
                "policy_context",
            )
        }
        label = self.labels.get(key)
        return {
            "index": index,
            "position": index + 1,
            "total": self.total,
            "scene_id": key[0],
            "case_uid": key[1],
            "case": safe_case,
            "assets": self._assets(case_dir),
            "label": {field: label.get(field) for field in LABEL_FIELDS} if label else None,
            "completed": label is not None,
            "progress": self.status(),
        }

    def asset_path(self, scene: str, uid: str, relative: str) -> Path:
        key = (scene, uid)
        row = self.by_key.get(key)
        if row is None:
            raise FileNotFoundError("unknown case")
        case_dir = self._case_dir(row)
        target = (case_dir / relative).resolve()
        if case_dir != target and case_dir not in target.parents:
            raise FileNotFoundError("invalid asset path")
        if not target.is_file():
            raise FileNotFoundError("missing asset")
        return target

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = (str(payload.get("scene_id", "")), str(payload.get("case_uid", "")))
        if key not in self.by_key:
            raise ValueError("未知案例，未保存")
        label = validate_label(payload)
        with self.lock:
            output = dict(self.by_key[key])
            output.update(label)
            self.labels[key] = output
            tmp = self.labels_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                for row in self.worklist:
                    saved = self.labels.get(case_key(row))
                    if saved is not None:
                        handle.write(json.dumps(saved, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.labels_path)
        result = self.status()
        result["saved_case"] = f"{key[0]}/{key[1]}"
        return result


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R1 人工复核</title>
<style>
:root{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#687289;--line:#dfe4ee;--blue:#315efb;--green:#15825d;--red:#bb3030;--shadow:0 8px 28px rgba(28,42,70,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:20;background:rgba(245,247,251,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:12px 20px}
.bar{max-width:1500px;margin:auto;display:flex;gap:16px;align-items:center}.title{font-weight:750;font-size:18px;white-space:nowrap}.progress{height:10px;background:#e2e7f0;border-radius:99px;overflow:hidden;flex:1}.progress>i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),#6d8aff);width:0}.count{font-variant-numeric:tabular-nums;color:var(--muted)}
main{max-width:1500px;margin:18px auto;padding:0 18px;display:grid;grid-template-columns:minmax(0,1.55fr) minmax(360px,.8fr);gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}
.evidence{padding:18px}.form{padding:18px;position:sticky;top:77px;align-self:start;max-height:calc(100vh - 95px);overflow:auto}.eyebrow{color:var(--blue);font-weight:700}.case-title{font-size:22px;margin:3px 0 8px}.muted{color:var(--muted)}
.notice{padding:10px 12px;background:#eef3ff;border-radius:9px;margin:12px 0}.facts{display:grid;gap:8px;margin:14px 0}.fact{padding:10px 12px;border-left:4px solid #7b91d9;background:#f7f9fe;border-radius:6px;white-space:pre-wrap}.fact.warn{border-left-color:#e1a029;background:#fff9ed}.fact.veto{border-left-color:#b64b63;background:#fff4f6}
details{border:1px solid var(--line);border-radius:10px;padding:9px 12px;margin:12px 0}summary{cursor:pointer;font-weight:700}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fafbfe}.figure img{display:block;width:100%;height:auto;cursor:zoom-in}.figure figcaption{padding:7px 9px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}
h2{font-size:18px;margin:0 0 12px}.field{margin-bottom:13px}.field label{display:block;font-weight:700;margin-bottom:5px}.required:after{content:" *";color:var(--red)}select,textarea,input{width:100%;border:1px solid #cbd2df;border-radius:9px;padding:9px 10px;background:white;color:var(--ink);font:inherit}textarea{min-height:68px;resize:vertical}.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}.help{font-size:12px;color:var(--muted);margin-top:3px}.actions{display:grid;grid-template-columns:1fr 1.4fr 1fr;gap:8px;position:sticky;bottom:-18px;background:white;padding:12px 0 18px;border-top:1px solid var(--line)}button{border:0;border-radius:9px;padding:10px 12px;font-weight:700;cursor:pointer}button.primary{background:var(--blue);color:white}button.secondary{background:#edf0f6;color:var(--ink)}button:disabled{opacity:.45;cursor:not-allowed}.saved{color:var(--green);font-weight:700}.error{color:var(--red);font-weight:700}.done{padding:25px;text-align:center}.hidden{display:none}
@media(max-width:980px){main{grid-template-columns:1fr}.form{position:static;max-height:none}.gallery{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="bar"><div class="title">R1 人工复核</div><div class="progress"><i id="progressBar"></i></div><div class="count" id="progressText">加载中</div></div></header>
<main id="main">
<section class="card evidence">
  <div class="eyebrow" id="position"></div><h1 class="case-title" id="caseTitle"></h1><div class="muted" id="caseMeta"></div>
  <div class="notice">只根据你看到的证据判断。页面刻意不显示这是随机队列还是优先队列，也不显示规则自己的 certainty 和分数。</div>
  <details open><summary>这条规则在怀疑什么</summary><div class="facts" id="hypotheses"></div></details>
  <details><summary>已记录的事实</summary><div class="facts" id="facts"></div></details>
  <details><summary>否决条件与缺失证据</summary><div class="facts" id="limits"></div></details>
  <h2>可视证据</h2><div class="gallery" id="gallery"></div>
</section>
<aside class="card form">
  <h2>你的人工判断</h2><div id="savedState" class="muted">尚未保存</div>
  <div class="field"><label class="required">证据够不够判断</label><select id="evidence_sufficient"><option value="">请选择</option><option>YES</option><option>PARTIAL</option><option>NO</option></select><div class="help">看不清或缺关键视角时选 PARTIAL/NO，不要勉强。</div></div>
  <div class="field"><label class="required">规则指出的问题是真的吗</label><select id="finding_correct"><option value="">请选择</option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select></div>
  <div class="field"><label class="required">根因阶段定位正确吗</label><select id="root_stage_correct"><option value="">请选择</option><option>YES</option><option>NO</option><option>UNCERTAIN</option><option>NOT_APPLICABLE</option></select></div>
  <div class="field"><label class="required">你看到的真实物理关系</label><textarea id="physical_interpretation" placeholder="例如：同一把椅子的重复 proposal；两个不同物体被错融；只是视角遮挡"></textarea></div>
  <div class="field"><label class="required">对最终地图的危害</label><select id="downstream_harm"><option value="">请选择</option><option>NONE</option><option>LOCAL_WEIGHTING_BIAS</option><option>WRONG_OBSERVATION_MEMBERSHIP</option><option>FALSE_SPLIT_DUPLICATE_NODE</option><option>FALSE_MERGE_IDENTITY_POLLUTION</option><option>GEOMETRY_CORRUPTION</option><option>RELATION_POLLUTION</option><option>UNKNOWN</option></select></div>
  <div class="field"><label class="required">危害判断置信度（1–5）</label><select id="harm_confidence"><option value="">请选择</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div>
  <div class="field"><label class="required">最合适的修复动作</label><select id="repair_action"><option value="">请选择</option><option>NONE</option><option>DROP_OBSERVATION</option><option>REASSIGN_OBSERVATION</option><option>MERGE_OBJECTS</option><option>SPLIT_OBJECT</option><option>RECOMPUTE_GEOMETRY</option><option>DOWNWEIGHT_EVIDENCE</option><option>NEED_MORE_VIEW</option><option>UNKNOWN</option></select></div>
  <div class="two"><div class="field"><label class="required">修复范围</label><select id="repair_locality"><option value="">请选择</option><option>LOCAL</option><option>MULTI_OBJECT</option><option>GLOBAL</option><option>NOT_APPLICABLE</option></select></div><div class="field"><label class="required">修复置信度（1–5）</label><select id="repair_confidence"><option value="">请选择</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div></div>
  <div class="field"><label>其他可能解释</label><textarea id="alternative_explanation" placeholder="可留空"></textarea></div>
  <div class="field"><label>备注</label><textarea id="notes" placeholder="尤其记录你犹豫的原因"></textarea></div>
  <div id="message"></div>
  <div class="actions"><button class="secondary" id="prev">上一例</button><button class="primary" id="save">保存本例</button><button class="secondary" id="next">下一例</button></div>
</aside>
</main>
<script>
const fields=['evidence_sufficient','finding_correct','root_stage_correct','physical_interpretation','downstream_harm','harm_confidence','repair_action','repair_locality','repair_confidence','alternative_explanation','notes'];
let current=0, payload=null, openedAt=Date.now(), priorSeconds=0;
const $=id=>document.getElementById(id);
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function blocks(id,values,cls='fact'){const root=$(id);root.innerHTML='';const items=Array.isArray(values)?values:(values?[values]:[]);if(!items.length){root.innerHTML='<div class="muted">无</div>';return;}for(const value of items){const div=document.createElement('div');div.className=cls;div.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);root.appendChild(div);}}
function updateProgress(p){$('progressBar').style.width=(100*p.completed/p.total)+'%';$('progressText').textContent=`已完成 ${p.completed} / ${p.total}`;}
function setForm(label){for(const field of fields){$(field).value=label?.[field]??'';}priorSeconds=Number(label?.review_seconds||0);$('savedState').className=label?'saved':'muted';$('savedState').textContent=label?'本例已有保存记录，可修改后再次保存':'尚未保存';openedAt=Date.now();}
async function loadCase(index){$('message').textContent='';const res=await fetch('/api/case?index='+index,{cache:'no-store'});if(!res.ok){$('message').className='error';$('message').textContent='读取案例失败';return;}payload=await res.json();current=payload.index;updateProgress(payload.progress);$('position').textContent=`第 ${payload.position} / ${payload.total} 例`;$('caseTitle').textContent=`${payload.case.checker_id||''} · ${payload.case.subtype||'风险案例'}`;$('caseMeta').textContent=`场景 ${payload.scene_id} · 阶段 ${payload.case.stage||''} · 案例 ${payload.case_uid}`;blocks('hypotheses',payload.case.hypotheses);blocks('facts',payload.case.proven_facts);const limits=[];(payload.case.vetoes||[]).forEach(x=>limits.push('VETO: '+(typeof x==='string'?x:JSON.stringify(x))));(payload.case.missing_evidence||[]).forEach(x=>limits.push('MISSING: '+(typeof x==='string'?x:JSON.stringify(x))));blocks('limits',limits,'fact warn');const g=$('gallery');g.innerHTML='';for(const name of payload.assets){const fig=document.createElement('figure');fig.className='figure';const img=document.createElement('img');img.loading='lazy';img.src=`/asset?scene=${encodeURIComponent(payload.scene_id)}&case=${encodeURIComponent(payload.case_uid)}&file=${encodeURIComponent(name)}`;img.alt=name;img.onclick=()=>window.open(img.src,'_blank');const cap=document.createElement('figcaption');cap.textContent=name;fig.append(img,cap);g.appendChild(fig);}setForm(payload.label);$('prev').disabled=current===0;$('next').disabled=current===payload.total-1;window.scrollTo({top:0,behavior:'smooth'});}
async function save(){const body={scene_id:payload.scene_id,case_uid:payload.case_uid,review_seconds:priorSeconds+(Date.now()-openedAt)/1000};for(const field of fields)body[field]=$(field).value;const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await res.json();if(!res.ok){$('message').className='error';$('message').textContent=data.error||'保存失败';return false;}$('message').className='saved';$('message').textContent=`已保存。总进度 ${data.completed}/${data.total}`;priorSeconds=body.review_seconds;openedAt=Date.now();updateProgress(data);return true;}
$('save').onclick=save;$('prev').onclick=()=>loadCase(Math.max(0,current-1));$('next').onclick=()=>loadCase(Math.min(payload.total-1,current+1));document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();save();}});loadCase(0);
</script>
</body></html>"""


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
                index = int(query.get("index", ["0"])[0])
                self._send_json(self.store.case_payload(index))
            elif parsed.path == "/asset":
                path = self.store.asset_path(
                    query.get("scene", [""])[0],
                    query.get("case", [""])[0],
                    query.get("file", [""])[0],
                )
                data = path.read_bytes()
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
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
        except Exception as exc:  # keep browser response readable; traceback remains in server log
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"server error: {exc}")
            raise

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
    parser = argparse.ArgumentParser()
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
