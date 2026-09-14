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

REVIEW_EVIDENCE_SCHEMA = "1.0.0"
REVIEW_EVIDENCE_FILENAME = "review_evidence.json"
REVIEW_MANIFEST_FILENAME = "review_evidence_manifest.json"

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    evidence = cleaned["evidence_sufficient"]
    finding = cleaned["finding_correct"]
    root = cleaned["root_stage_correct"]
    harm = cleaned["downstream_harm"]
    repair = cleaned["repair_action"]
    locality = cleaned["repair_locality"]
    if evidence == "YES" and (
        finding == "UNCERTAIN"
        or root == "UNCERTAIN"
        or harm == "UNKNOWN"
        or repair in {"UNKNOWN", "NEED_MORE_VIEW"}
    ):
        raise ValueError(
            "证据为 YES 表示所有核心结论都能落定：不能再选 UNCERTAIN、UNKNOWN 或 NEED_MORE_VIEW"
        )
    if evidence == "PARTIAL" and not cleaned["notes"]:
        raise ValueError("证据为 PARTIAL 时，请在备注中写清缺少哪一环、哪些结论不能确认")
    if finding == "UNCERTAIN" and (
        root != "UNCERTAIN"
        or harm != "UNKNOWN"
        or repair != "NEED_MORE_VIEW"
        or locality != "NOT_APPLICABLE"
    ):
        raise ValueError(
            "finding 为 UNCERTAIN 时不能继续猜根因、危害或修复：请选 root=UNCERTAIN、"
            "harm=UNKNOWN、repair=NEED_MORE_VIEW、locality=NOT_APPLICABLE"
        )
    if evidence == "NO" and finding != "UNCERTAIN":
        raise ValueError("证据为 NO 时，finding 必须选 UNCERTAIN，不能把看不清当成 YES 或 NO")
    if finding == "NO" and (
        root != "NOT_APPLICABLE"
        or harm != "NONE"
        or repair != "NONE"
        or locality != "NOT_APPLICABLE"
    ):
        raise ValueError(
            "finding 为 NO 时应选 root=NOT_APPLICABLE、harm=NONE、"
            "repair=NONE、locality=NOT_APPLICABLE"
        )
    if harm == "NONE" and repair != "NONE":
        raise ValueError("最终地图危害为 NONE 时，修复动作应选 NONE")
    if repair == "NONE" and harm != "NONE":
        raise ValueError("修复动作为 NONE 时，最终地图危害也应为 NONE")
    if finding == "YES" and root == "NOT_APPLICABLE":
        raise ValueError("finding 为 YES 时根因阶段存在，不能选 NOT_APPLICABLE")
    if repair == "NONE" and locality != "NOT_APPLICABLE":
        raise ValueError("修复动作为 NONE 时，修复范围应选 NOT_APPLICABLE")
    if repair == "NEED_MORE_VIEW" and locality != "NOT_APPLICABLE":
        raise ValueError("修复动作为 NEED_MORE_VIEW 时，当前没有可执行范围，请选 NOT_APPLICABLE")
    if repair in {"REASSIGN_OBSERVATION", "MERGE_OBJECTS", "SPLIT_OBJECT"} and locality != "MULTI_OBJECT":
        raise ValueError(f"{repair} 会同时改变多个对象，请选 MULTI_OBJECT")
    return cleaned


class ReviewStore:
    def __init__(self, validation_root: Path):
        self.root = validation_root.resolve()
        self.labels_dir = self.root / "labels"
        self.worklist_path = self.labels_dir / "r1_worklist.jsonl"
        self.labels_path = self.labels_dir / "labels_r1.jsonl"
        self.worklist = read_jsonl(self.worklist_path)
        self.review_manifest_path = self.root / REVIEW_MANIFEST_FILENAME
        if not self.review_manifest_path.is_file():
            raise ValueError(
                f"缺少 {REVIEW_MANIFEST_FILENAME}；R1 暂停，先生成与系统证据对齐的人类证据包"
            )
        self.review_manifest = json.loads(
            self.review_manifest_path.read_text(encoding="utf-8")
        )
        if self.review_manifest.get("schema_version") != REVIEW_EVIDENCE_SCHEMA:
            raise ValueError("review evidence schema 不兼容")
        if self.review_manifest.get("status") not in {
            "READY",
            "READY_WITH_DECLARED_LIMITATIONS",
        }:
            raise ValueError("review evidence manifest 尚未就绪")
        expected_worklist_sha = hashlib.sha256(self.worklist_path.read_bytes()).hexdigest()
        if self.review_manifest.get("worklist_sha256") != expected_worklist_sha:
            raise ValueError("review evidence 与当前 R1 worklist 不一致")
        if int(self.review_manifest.get("case_count", -1)) != len(self.worklist):
            raise ValueError("review evidence 案例数与 R1 worklist 不一致")
        self.by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.worklist:
            key = case_key(row)
            if key in self.by_key:
                raise ValueError(f"duplicate worklist case: {key}")
            self.by_key[key] = row
        self.review_manifest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for item in self.review_manifest.get("cases") or []:
            key = (str(item.get("scene_id") or ""), str(item.get("case_uid") or ""))
            if not all(key) or key in self.review_manifest_by_key:
                raise ValueError("review evidence manifest 含无效或重复案例")
            self.review_manifest_by_key[key] = item
        if set(self.review_manifest_by_key) != set(self.by_key):
            raise ValueError("review evidence manifest 的案例集合与 R1 worklist 不一致")
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

    def _assets(self, case_dir: Path, review_evidence: dict[str, Any]) -> list[str]:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        declared = review_evidence.get("displayed_asset_sha256")
        if not isinstance(declared, dict):
            raise ValueError("人类证据投影缺少页面图片哈希清单")
        names = []
        for name, expected_sha in declared.items():
            relative = Path(str(name))
            path = (case_dir / relative).resolve()
            if case_dir != path and case_dir not in path.parents:
                raise ValueError("页面图片路径逃逸案例目录")
            if path.suffix.lower() not in extensions or not path.is_file():
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
        case_json_path = case_dir / "case.json"
        case_json = json.loads(case_json_path.read_text(encoding="utf-8"))
        review_path = case_dir / REVIEW_EVIDENCE_FILENAME
        if not review_path.is_file():
            raise ValueError(f"缺少人类证据投影：{key[0]}/{key[1]}")
        review_evidence = json.loads(review_path.read_text(encoding="utf-8"))
        manifest_item = self.review_manifest_by_key[key]
        if manifest_item.get("review_evidence_sha256") != sha256_file(review_path):
            raise ValueError(f"人类证据投影与顶层 manifest 哈希不一致：{key[0]}/{key[1]}")
        if review_evidence.get("schema_version") != REVIEW_EVIDENCE_SCHEMA:
            raise ValueError(f"人类证据投影版本不兼容：{key[0]}/{key[1]}")
        if review_evidence.get("scene_id") != key[0] or review_evidence.get("case_uid") != key[1]:
            raise ValueError(f"人类证据投影绑定了错误案例：{key[0]}/{key[1]}")
        case_sha = sha256_file(case_json_path)
        if review_evidence.get("source_case_json_sha256") != case_sha:
            raise ValueError(f"case.json 在人类证据生成后发生变化：{key[0]}/{key[1]}")
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
            "review_evidence": review_evidence,
            "assets": self._assets(case_dir, review_evidence),
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
        review_path = case_dir / REVIEW_EVIDENCE_FILENAME
        review_evidence = json.loads(review_path.read_text(encoding="utf-8"))
        name = target.relative_to(case_dir).as_posix()
        expected_sha = (review_evidence.get("displayed_asset_sha256") or {}).get(name)
        if not expected_sha:
            raise FileNotFoundError("asset is not declared by the frozen review evidence")
        if sha256_file(target) != expected_sha:
            raise FileNotFoundError("asset hash changed after review evidence generation")
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
.notice{padding:10px 12px;background:#eef3ff;border-radius:9px;margin:12px 0}.notice.warning{background:#fff4e5;border:1px solid #efb55a}.notice.pass{background:#eefaf5;border:1px solid #8bcbb4}.facts{display:grid;gap:8px;margin:14px 0}.fact{padding:10px 12px;border-left:4px solid #7b91d9;background:#f7f9fe;border-radius:6px;white-space:pre-wrap}.fact.warn{border-left-color:#e1a029;background:#fff9ed}.fact.veto{border-left-color:#b64b63;background:#fff4f6}
details{border:1px solid var(--line);border-radius:10px;padding:9px 12px;margin:12px 0}summary{cursor:pointer;font-weight:700}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.gallery.one{grid-template-columns:1fr}.figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fafbfe}.figure img{display:block;width:100%;height:auto;cursor:zoom-in}.figure figcaption{padding:8px 10px;color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.section{border-top:1px solid var(--line);margin-top:20px;padding-top:18px}.section h2{display:flex;align-items:center;gap:8px}.step{display:inline-grid;place-items:center;width:25px;height:25px;border-radius:50%;background:var(--blue);color:#fff;font-size:13px}.question{font-size:17px;font-weight:750;padding:12px 14px;border-radius:10px;background:#f0f4ff;margin:10px 0}.contract{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:12px 0}.contract>div,.stat{border:1px solid var(--line);border-radius:9px;padding:9px 10px;background:#fbfcff}.contract b,.stat b{display:block;font-size:12px;color:var(--muted)}.contract span,.stat span{font-weight:700}.cards{display:grid;gap:10px}.entity{border:1px solid var(--line);border-radius:10px;padding:12px;background:#fbfcff}.entity h3{margin:0 0 7px;font-size:15px}.pills{display:flex;flex-wrap:wrap;gap:5px;margin:6px 0}.pill{display:inline-block;border-radius:99px;padding:2px 8px;background:#e8edfb;color:#294a9d;font-size:12px}.pill.final{background:#e1f5ec;color:#116447}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.kv{display:grid;grid-template-columns:minmax(130px,.45fr) minmax(0,1fr);gap:3px 10px;margin:7px 0}.kv>div:nth-child(odd){color:var(--muted)}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}th{background:#f4f6fb;white-space:nowrap}.view-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.view-card{border:1px solid var(--line);border-radius:9px;padding:9px}.view-images{display:grid;grid-template-columns:1fr 1fr;gap:5px}.view-images img{width:100%;border-radius:6px;cursor:zoom-in}.logic{font-size:13px;padding:8px 10px;border-radius:8px;background:#f7f8fb;margin:6px 0}.logic.error{background:#fff0f0;color:#9b2727}.logic.ok{background:#ecf8f2;color:#176447}
.guide{background:#fbfcff}.guide h3{font-size:14px;margin:13px 0 4px}.guide ul{margin:4px 0 9px;padding-left:20px}.guide li{margin:4px 0}.guide code{font-size:12px;color:#2446ad}.shortcut{padding:8px 10px;border-radius:8px;background:#f0f4ff;margin:7px 0}
h2{font-size:18px;margin:0 0 12px}.field{margin-bottom:13px}.field label{display:block;font-weight:700;margin-bottom:5px}.required:after{content:" *";color:var(--red)}select,textarea,input{width:100%;border:1px solid #cbd2df;border-radius:9px;padding:9px 10px;background:white;color:var(--ink);font:inherit}textarea{min-height:68px;resize:vertical}.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}.help{font-size:12px;color:var(--muted);margin-top:3px}.actions{display:grid;grid-template-columns:1fr 1.4fr 1fr;gap:8px;position:sticky;bottom:-18px;background:white;padding:12px 0 18px;border-top:1px solid var(--line)}button{border:0;border-radius:9px;padding:10px 12px;font-weight:700;cursor:pointer}button.primary{background:var(--blue);color:white}button.secondary{background:#edf0f6;color:var(--ink)}button:disabled{opacity:.45;cursor:not-allowed}.saved{color:var(--green);font-weight:700}.error{color:var(--red);font-weight:700}.done{padding:25px;text-align:center}.hidden{display:none}
@media(max-width:980px){main{grid-template-columns:1fr}.form{position:static;max-height:none}.gallery,.view-grid,.contract{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="bar"><div class="title">R1 人工复核</div><div class="progress"><i id="progressBar"></i></div><div class="count" id="progressText">加载中</div></div></header>
<main id="main">
<section class="card evidence">
  <div class="eyebrow" id="position"></div><h1 class="case-title" id="caseTitle"></h1><div class="muted" id="caseMeta"></div>
  <div class="notice">页面仍隐藏随机/优先队列、规则 certainty、sampling weight 和 review score。现在每项展示都区分：<b>系统触发时的精确记录</b>、<b>供人理解的代表视图</b>、<b>最终地图中的真实 object</b>。</div>
  <div id="contractStatus"></div>
  <div class="contract" id="contractGrid"></div>
  <details id="contractGapsWrap" class="hidden" open><summary>为什么这里不能假装“证据完整”</summary><div class="facts" id="contractGaps"></div></details>

  <div class="section">
    <h2><span class="step">1</span>先明确这例到底要判断什么</h2>
    <div class="question" id="reviewQuestion"></div>
    <details open><summary>规则提出的错误假设（不是答案）</summary><div class="facts" id="hypotheses"></div></details>
    <details><summary>系统实际使用的触发事实</summary><div class="facts" id="facts"></div></details>
    <details><summary>已知反例、原 finding 声明的缺失证据</summary><div class="facts" id="limits"></div></details>
  </div>

  <div class="section" id="decisionSection">
    <h2><span class="step">2</span>系统当时做了什么决定</h2>
    <div class="muted">这里直接来自 associations / object_versions；不是根据图片事后猜出的。</div>
    <div id="decisionRecords"></div>
  </div>

  <div class="section">
    <h2><span class="step">3</span>可疑 observation：同一份 RGB / mask / depth / 3D</h2>
    <div class="muted">每张六联图由 ledger 引用的原始 artifact 生成。橙色表示 containment subtraction 删除的像素；右下角明确是 DBSCAN 后保存的 observation PCD。</div>
    <div class="gallery one" id="triggerGallery"></div>
    <div class="cards" id="triggerRecords"></div>
  </div>

  <div class="section">
    <h2><span class="step">4</span>对象身份与代表视图</h2>
    <div class="muted">对象视图是从完整成员中按创建帧、贡献、置信度、语义冲突、视角差异等规则抽取的代表样本；页面会写清“显示了多少 / 总共有多少”，不会把抽样当成全部。</div>
    <div class="cards" id="objectRecords"></div>
    <details open><summary>展开代表性 2D 视图</summary><div class="view-grid" id="representativeViews"></div></details>
  </div>

  <div class="section">
    <h2><span class="step">5</span>最终地图 object：判断危害必须看这里</h2>
    <div class="notice pass">下图直接读取 manifest 用 SHA-256 锁定的最终 map pickle，并逐对象核对 UID、完整成员集合和点数。统一坐标图用于看对象是否重合/分离；单对象图用于看自身几何。</div>
    <div class="cards" id="finalRecords"></div>
    <div class="gallery one" id="finalGallery"></div>
  </div>

  <div class="section">
    <h2><span class="step">6</span>原始 packet 材料（追溯用）</h2>
    <details><summary>展开全部旧版图片</summary><div class="notice warning">其中 <code>pcd_overlay.png</code> 只是被抽中 observation 的点云叠加，<b>不是最终 object</b>；最终 object 以上一节为准。</div><div class="gallery" id="gallery"></div></details>
  </div>
</section>
<aside class="card form">
  <h2>你的人工判断</h2><div id="savedState" class="muted">尚未保存</div>
  <details class="guide" open><summary>先看：每个选项到底怎么选</summary>
    <div class="shortcut"><b>固定顺序：</b>①先看证据对齐状态；②判断现实中是否真有建图错误；③找最早根因；④再看最终 object 判断危害；⑤最后才选修复。<b>阈值触发不等于错误，真实错误也可能最终无害。</b></div>
    <h3>证据够不够判断</h3><ul>
      <li><code>YES</code>：你能把“是否真的出错、根因阶段、最终危害、可执行修复”都落定；不能再搭配 <code>UNCERTAIN</code>、<code>UNKNOWN</code> 或 <code>NEED_MORE_VIEW</code>。</li>
      <li><code>PARTIAL</code>：能判断其中一部分，例如 mask 明显有问题，但缺少某个历史点云快照，无法确认根因或最终危害。必须在备注写清缺哪一环；这类案例不进入准确率分母，只进入证据覆盖率。</li>
      <li><code>NO</code>：连核心物理关系都无法可靠判断，或关键证据与系统记录无法对齐。不要凭规则名、阈值或类别文字猜；该例只记为证据覆盖失败，不记为误报。</li>
      <li>页面出现橙色“关键视觉缺口”时，不是强制你选 NO；它提醒你核对这个缺口是否正好阻断本例判断。若阻断，选 PARTIAL/NO。</li>
    </ul>
    <h3>这个候选对应真实建图错误吗</h3><ul>
      <li><code>YES</code>：现实物理关系与系统 identity/成员/几何/检测决定确实冲突。仅仅“margin 很低”“数值过阈值”不能算 YES。</li>
      <li><code>NO</code>：触发数值可能完全真实，但它对应正常视角变化、遮挡、合法部件、多 proposal 合法归入同一 object，或正确的新建对象；因此不是建图错误。</li>
      <li><code>UNCERTAIN</code>：错误解释与正常解释仍都说得通。证据为 NO 时必须选它，不能把看不清当作误报。</li>
    </ul>
    <h3>根因阶段定位正确吗</h3><ul>
      <li><code>YES</code>：最早、最主要的问题确实发生在页面显示的 detection / segmentation / geometry / association / fusion / object identity 阶段。</li>
      <li><code>NO</code>：异常是真的，但根因来自别的阶段。例如 association 看似错，其实最早是 mask 分割错。</li>
      <li><code>UNCERTAIN</code>：能确认异常，却无法判断最早从哪个阶段开始。</li>
      <li><code>NOT_APPLICABLE</code>：finding 为 NO，根因问题不存在。证据为 NO 而 finding=UNCERTAIN 时应选 UNCERTAIN，不是 NOT_APPLICABLE。</li>
    </ul>
    <h3>真实物理关系</h3><ul><li>不要复述规则名，用一句自然语言写你看到的世界，例如“同一把椅子的两个重复 mask”“两件不同家具被吸进同一对象”“只是扶手与椅子本体的部件关系”。</li></ul>
    <h3>对最终地图的危害</h3><ul>
      <li><code>NONE</code>：finding 为 NO；或错误虽真实，但最终对象身份、成员、几何和特征没有受到可见影响。必须看完“最终地图 object”再选。</li>
      <li><code>LOCAL_WEIGHTING_BIAS</code>：对象身份基本正确，但重复/低质量 observation 让类别计数、CLIP 特征或融合权重产生局部偏差。</li>
      <li><code>WRONG_OBSERVATION_MEMBERSHIP</code>：一个有效 observation 被放进了错误对象。</li>
      <li><code>FALSE_SPLIT_DUPLICATE_NODE</code>：一个真实物体被错误保留成两个或更多地图节点。</li>
      <li><code>FALSE_MERGE_IDENTITY_POLLUTION</code>：两个或更多真实物体被错误融合成同一个节点。</li>
      <li><code>GEOMETRY_CORRUPTION</code>：点云、bbox、位置或尺度被明显拉坏，即使对象身份可能仍对。</li>
      <li><code>RELATION_POLLUTION</code>：错误传播到场景图关系/边。本轮正式运行未启用 edge，通常不选它。</li>
      <li><code>UNKNOWN</code>：怀疑有害，但看不出具体伤害类型或无法确认最终是否受影响。</li>
    </ul>
    <h3>最合适的修复动作</h3><ul>
      <li><code>NONE</code>：finding 为 NO，或异常真实但无下游危害。</li>
      <li><code>DROP_OBSERVATION</code>：某个 observation 本身是重复、伪检或严重坏数据，直接移除最合适。</li>
      <li><code>REASSIGN_OBSERVATION</code>：observation 本身有效，只是归到了错误对象，应移动到另一个对象。</li>
      <li><code>MERGE_OBJECTS</code>：一个真实物体被拆成多个重复节点，应把节点合并。</li>
      <li><code>SPLIT_OBJECT</code>：多个真实物体被错融成一个节点，应拆开。</li>
      <li><code>RECOMPUTE_GEOMETRY</code>：成员和身份大体正确，主要是 mask/projection/点云/bbox 几何需要重算。</li>
      <li><code>DOWNWEIGHT_EVIDENCE</code>：observation 仍有部分价值，不应删除，但应降低它对融合或语义的影响。</li>
      <li><code>NEED_MORE_VIEW</code>：必须补更多视角或时间信息才能安全决定怎么修。</li>
      <li><code>UNKNOWN</code>：确认有害，却无法给出安全、明确的动作。</li>
    </ul>
    <h3>修复范围</h3><ul>
      <li><code>LOCAL</code>：只动一个 observation 或一个对象内部。</li>
      <li><code>MULTI_OBJECT</code>：需要同时处理两个或更多对象；重归属、对象 merge/split 通常选它。</li>
      <li><code>GLOBAL</code>：问题来自全局阈值、策略或系统规则，需要影响大量案例的改动。</li>
      <li><code>NOT_APPLICABLE</code>：repair action 为 NONE 或 NEED_MORE_VIEW，当前没有可执行修复。</li>
    </ul>
    <h3>两个置信度的 1–5</h3><ul>
      <li><code>1</code>：几乎是猜测；<code>2</code>：弱证据；<code>3</code>：更可能是，但仍有合理反例；<code>4</code>：证据很清楚；<code>5</code>：多种证据直接一致，几乎没有合理替代解释。</li>
      <li>危害置信度只评价“伤害类型判断”；修复置信度只评价“这个动作是否安全合适”，不要因为 finding 很明显就一律填 5。</li>
    </ul>
    <h3>常见组合</h3><ul>
      <li>误报：<code>finding=NO</code>、<code>root=NOT_APPLICABLE</code>、<code>harm=NONE</code>、<code>repair=NONE</code>。</li>
      <li>真实但无害：<code>finding=YES</code>、<code>harm=NONE</code>、<code>repair=NONE</code>。</li>
      <li>完全证据不足：<code>evidence=NO</code>、<code>finding=UNCERTAIN</code>、<code>root=UNCERTAIN</code>、<code>harm=UNKNOWN</code>、<code>repair=NEED_MORE_VIEW</code>、<code>locality=NOT_APPLICABLE</code>。</li>
      <li>部分证据不足：<code>evidence=PARTIAL</code>；其余项按你实际能确认的范围填，备注必须写缺失环节。</li>
      <li>一个物体变多个节点：<code>FALSE_SPLIT_DUPLICATE_NODE + MERGE_OBJECTS + MULTI_OBJECT</code>。</li>
      <li>多个物体错融一个节点：<code>FALSE_MERGE_IDENTITY_POLLUTION + SPLIT_OBJECT + MULTI_OBJECT</code>。</li>
      <li>观测放错对象：<code>WRONG_OBSERVATION_MEMBERSHIP + REASSIGN_OBSERVATION + MULTI_OBJECT</code>。</li>
    </ul>
  </details>
  <div class="field"><label class="required">证据够不够判断</label><select id="evidence_sufficient"><option value="">请选择</option><option>YES</option><option>PARTIAL</option><option>NO</option></select><div class="help">看不清或缺关键视角时选 PARTIAL/NO，不要勉强。</div></div>
  <div class="field"><label class="required">这个候选对应真实建图错误吗</label><select id="finding_correct"><option value="">请选择</option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select><div class="help">判断现实中的错误，不是判断触发数值是否真实。</div></div>
  <div class="field"><label class="required">根因阶段定位正确吗</label><select id="root_stage_correct"><option value="">请选择</option><option>YES</option><option>NO</option><option>UNCERTAIN</option><option>NOT_APPLICABLE</option></select></div>
  <div class="field"><label class="required">你看到的真实物理关系</label><textarea id="physical_interpretation" placeholder="例如：同一把椅子的重复 proposal；两个不同物体被错融；只是视角遮挡"></textarea></div>
  <div class="field"><label class="required">对最终地图的危害</label><select id="downstream_harm"><option value="">请选择</option><option>NONE</option><option>LOCAL_WEIGHTING_BIAS</option><option>WRONG_OBSERVATION_MEMBERSHIP</option><option>FALSE_SPLIT_DUPLICATE_NODE</option><option>FALSE_MERGE_IDENTITY_POLLUTION</option><option>GEOMETRY_CORRUPTION</option><option>RELATION_POLLUTION</option><option>UNKNOWN</option></select></div>
  <div class="field"><label class="required">危害判断置信度（1–5）</label><select id="harm_confidence"><option value="">请选择</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div>
  <div class="field"><label class="required">最合适的修复动作</label><select id="repair_action"><option value="">请选择</option><option>NONE</option><option>DROP_OBSERVATION</option><option>REASSIGN_OBSERVATION</option><option>MERGE_OBJECTS</option><option>SPLIT_OBJECT</option><option>RECOMPUTE_GEOMETRY</option><option>DOWNWEIGHT_EVIDENCE</option><option>NEED_MORE_VIEW</option><option>UNKNOWN</option></select></div>
  <div class="two"><div class="field"><label class="required">修复范围</label><select id="repair_locality"><option value="">请选择</option><option>LOCAL</option><option>MULTI_OBJECT</option><option>GLOBAL</option><option>NOT_APPLICABLE</option></select></div><div class="field"><label class="required">修复置信度（1–5）</label><select id="repair_confidence"><option value="">请选择</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div></div>
  <div class="field"><label>其他可能解释</label><textarea id="alternative_explanation" placeholder="可留空"></textarea></div>
  <div class="field"><label>备注</label><textarea id="notes" placeholder="尤其记录你犹豫的原因"></textarea></div>
  <div id="logicHint" class="logic">填写后这里会检查选项之间是否自洽，不会替你做判断。</div>
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
const REVIEW_QUESTIONS={
  'DET-001':'同一帧中的这些 proposal，是否其实重复指向同一个真实物体，并构成一次检测错误？',
  'DET-002':'这个弱观测是真实的小物体，还是噪声、背景或无法成立的伪检？',
  'SEG-002':'这个 processed mask 是否把背景或两个不同深度/物体错误包进了同一个 observation？',
  'SEG-004':'同一帧、同一 object 下的两个 observation 是一个物体被切碎，还是两个合法部件/实例？',
  'SEG-005':'containment subtraction 是否删掉了本应保留的主体，使 processed mask 明显破碎或残缺？',
  'GEO-003':'这个 observation 在 3D 中是否真的混入多个不应同属一体的 cluster，而非长物体或遮挡造成？',
  'GEO-004':'相邻观测的 3D 中心突跳，是否说明错误关联/位姿问题，而非视角、遮挡或真实动态？',
  'GEO-005':'去噪是否错误删除了对象主体，而不是只清除噪声点？',
  'ASSOC-002':'低 margin 只是“难选”；请判断系统最终 CREATE/MERGE 的身份决定是否真的选错，而不是把低 margin 本身当错误。',
  'ASSOC-003':'空间最优与视觉最优不一致时，系统最终身份决定是否真的把 observation 放错了对象？',
  'ASSOC-004':'同一帧多个 observation 进入同一对象，是合法的重复/部件归并，还是把不同真实物体错误塞进同一节点？',
  'ASSOC-005':'这个语义离群 observation 是否真的不属于最终对象，还是外观变化、遮挡或类别词波动？',
  'FUSE-007':'融合造成的中心/尺度/点数突变是否是坏融合，而不是同一物体新增了一个合理的大视角？',
  'OBJ-003':'这个低支持 final object 是真实的小物体节点，还是噪声/残片/无效节点？',
  'OBJ-005':'一个 final object 内类别高度不稳定，是否来自 false merge，而非同一物体的类别同义词或视角变化？'
};
const REASON_ZH={
  suspect_observation:'规则直接指向',event_or_version_trigger:'事件/版本触发',earliest_creation_view:'最早创建视图',highest_point_contribution:'点贡献最高',highest_detector_confidence:'检测置信度最高',anomaly_trigger_view:'异常触发视图',largest_semantic_conflict:'语义冲突最大',largest_camera_viewpoint_difference:'相机视角差最大',temporal_diversity_fill:'补充时间跨度'
};
const ROLE_ZH={
  primary_object:'主要对象',compared_object:'比较对象',chosen_target:'系统目标',association_target:'关联结果',spatial_top_candidate:'空间第一',visual_top_candidate:'视觉第一',aggregate_top1:'综合第一',aggregate_top2:'综合第二',alternate_candidate:'备选对象',counterfactual_alternate:'反事实备选',final_owner_of_trigger:'触发观测的最终归属',observation_owner:'观测归属',association_top1:'候选第1',association_top2:'候选第2',association_top3:'候选第3'
};
function fmt(v,digits=4){if(v===null||v===undefined||v==='')return '—';if(typeof v==='number')return Number.isInteger(v)?String(v):v.toFixed(digits);if(typeof v==='object')return JSON.stringify(v);return String(v);}
function assetUrl(name){return `/asset?scene=${encodeURIComponent(payload.scene_id)}&case=${encodeURIComponent(payload.case_uid)}&file=${encodeURIComponent(name)}`;}
function makeFigure(root,name,caption){if(!name)return;const fig=document.createElement('figure');fig.className='figure';const img=document.createElement('img');img.loading='lazy';img.src=assetUrl(name);img.alt=caption||name;img.onclick=()=>window.open(img.src,'_blank');const cap=document.createElement('figcaption');cap.textContent=caption||name;fig.append(img,cap);root.appendChild(fig);}
function pills(values,kind=''){return (values||[]).map(x=>`<span class="pill ${kind}">${esc(ROLE_ZH[x]||REASON_ZH[x]||x)}</span>`).join('');}
function kv(rows){return `<div class="kv">${rows.map(([k,v])=>`<div>${esc(k)}</div><div>${esc(fmt(v))}</div>`).join('')}</div>`;}
function renderContract(review){const c=review.evidence_contract||{};const gaps=c.critical_gaps||[];const critical=gaps.some(x=>x.critical);const root=$('contractStatus');root.className='notice '+(critical?'warning':'pass');root.innerHTML=critical?'<b>本例存在已声明的关键视觉缺口。</b> 请判断它是否阻断你的结论；系统不会把缺失快照伪装成别的图。':'<b>证据投影可追溯。</b> 触发记录、引用 artifact 与最终 map object 已对齐。';const grid=$('contractGrid');grid.innerHTML=`<div><b>Artifact 哈希</b><span>${c.artifact_hashes_match?'一致':'异常'}</span></div><div><b>最终结果核对</b><span>${c.exact_final_map_linkage?'完整一致（含“无 active object”）':'异常'}</span></div><div><b>页面证据状态</b><span>${esc(c.fidelity_status||'')}</span></div>`;const wrap=$('contractGapsWrap');wrap.classList.toggle('hidden',!gaps.length);if(gaps.length){const facts=gaps.map(x=>`${x.critical?'关键缺口':'提示'} · ${x.code}\n${x.message}`);blocks('contractGaps',facts,'fact warn');}else{$('contractGaps').innerHTML='';}}
function renderDecisions(review){const root=$('decisionRecords');root.innerHTML='';const rows=review.association_decisions||[];$('decisionSection').style.display=rows.length?'block':'none';for(const item of rows){const card=document.createElement('div');card.className='entity';let table='<div class="table-wrap"><table><thead><tr><th>排名</th><th>对象</th><th>空间分</th><th>视觉分</th><th>综合分</th><th>决策时版本</th></tr></thead><tbody>';for(const c of item.candidates||[]){table+=`<tr><td>${c.rank}</td><td>${esc(c.object_alias)}<br><span class="mono">${esc(c.object_uid)}</span></td><td>${esc(fmt(c.spatial_score))}</td><td>${esc(fmt(c.visual_score))}</td><td>${esc(fmt(c.aggregate_score))}</td><td class="mono">${esc(c.object_version_uid||'—')}</td></tr>`;}table+='</tbody></table></div>';card.innerHTML=`<h3>${esc(item.obs_uid)}</h3>${kv([['系统动作',item.decision],['目标对象',`${item.target_object_alias} · ${item.target_object_uid}`],['Top1 / Top2',`${fmt(item.top1_score)} / ${fmt(item.top2_score)}`],['margin',item.margin],['阈值',item.sim_threshold],['相似度证据有效',item.similarity_evidence_valid]])}${table}`;root.appendChild(card);}}
function renderTriggers(review){const gallery=$('triggerGallery');gallery.innerHTML='';for(const item of review.trigger_observations||[]){makeFigure(gallery,item.panel_asset,`${item.observation_alias}：系统引用的同一份 RGB / raw mask / processed mask / depth / post-DBSCAN PCD`);}const root=$('triggerRecords');root.innerHTML='';for(const item of review.trigger_observations||[]){const card=document.createElement('div');card.className='entity';card.innerHTML=`<h3>${esc(item.observation_alias)} · ${esc(item.class_name||'unknown')} <span class="mono">${esc(item.obs_uid)}</span></h3>${kv([['帧',item.frame_number],['检测置信度',item.confidence],['raw → processed mask',`${fmt(item.raw_mask_area,0)} → ${fmt(item.processed_mask_area,0)}`],['被 subtraction 删除',item.removed_pixel_count],['有效深度比例',item.valid_depth_ratio],['post-DBSCAN 点数',item.n_points],['pre-DBSCAN 统计',item.pre_dbscan],['最终归属',(item.final_owner_aliases||[]).join(', ')||'无 active final owner']])}`;root.appendChild(card);}}
function renderObjects(review){const root=$('objectRecords');root.innerHTML='';for(const item of review.objects||[]){const version=item.trigger_or_decision_version||{};const coverage=item.representative_view_coverage||{};const card=document.createElement('div');card.className='entity';card.innerHTML=`<h3>${esc(item.object_alias)} <span class="mono">${esc(item.object_uid)}</span></h3><div class="pills">${pills(item.roles)}</div>${kv([['触发/决策时版本',version.object_version_uid],['当时类别',version.class_name],['当时成员 / 点数',version.object_version_uid?`${fmt(version.member_count,0)} / ${fmt(version.n_points,0)}`:'未记录到该角色版本'],['当时操作',version.operation],['代表视图覆盖',`${fmt(coverage.selected,0)} / ${fmt(coverage.total,0)}`],['最终状态',item.final_status],['最终去向',(item.resolved_final_aliases||[]).join(', ')||'不在 active final map']])}`;root.appendChild(card);}}
function renderRepresentative(review){const root=$('representativeViews');root.innerHTML='';for(const item of review.representative_views||[]){const card=document.createElement('div');card.className='view-card';const roles=[...(item.object_aliases||[]),...(item.object_roles||[])];card.innerHTML=`<b>${esc((item.object_aliases||[]).join(', ')||'未绑定对象')} · frame ${esc(fmt(item.frame_number,0))} · ${esc(item.class_name||'unknown')}</b><div class="pills">${pills(item.object_roles)}${pills(item.selection_reasons)}</div><div class="view-images"></div><div class="mono">${esc(item.obs_uid)}</div>`;const imgs=card.querySelector('.view-images');for(const [kind,name] of Object.entries(item.assets||{})){if(!name)continue;const wrap=document.createElement('div');const img=document.createElement('img');img.loading='lazy';img.src=assetUrl(name);img.alt=kind;img.onclick=()=>window.open(img.src,'_blank');const label=document.createElement('div');label.className='muted';label.textContent=kind==='masked_crop'?'processed mask 内像素':'带上下文 RGB';wrap.append(img,label);imgs.appendChild(wrap);}root.appendChild(card);}}
function renderFinal(review){const root=$('finalRecords');root.innerHTML='';const outcome=review.final_outcome||{};const outcomeCard=document.createElement('div');outcomeCard.className='notice pass';outcomeCard.innerHTML=`<b>最终结果：${esc(outcome.status||'未记录')}</b><br>${esc(outcome.message||'')}`;root.appendChild(outcomeCard);for(const item of review.final_objects||[]){const card=document.createElement('div');card.className='entity';card.innerHTML=`<h3>${esc(item.object_alias)} · ${esc(item.class_name||'unknown')} <span class="pill final">active final object</span></h3>${kv([['完整成员数 / 视角数',`${fmt(item.member_count,0)} / ${fmt(item.unique_frame_count,0)}`],['帧范围',`${fmt(item.first_frame,0)} – ${fmt(item.last_frame,0)}`],['最终点数',item.n_points],['最终 bbox center',item.bbox_center],['最终 bbox extent',item.bbox_extent],['成员类别统计',item.observed_class_histogram],['由哪些对象 merge 而来',item.parent_or_merged_from_object_uids],['完整成员与 final pickle',item.membership_matches_final_output?'一致':'不一致'],['点数与 final pickle',item.point_count_matches_final_output?'一致':'不一致'],['完整 PCD SHA-256',item.pcd_sha256]])}<div class="mono">${esc(item.object_uid)}</div>`;root.appendChild(card);}const gallery=$('finalGallery');gallery.innerHTML='';for(const name of (review.assets||{}).final_object_geometry||[]){makeFigure(gallery,name,name.includes('relative')?'最终对象：统一世界坐标（判断重合/分离）':'最终对象：逐对象放大（判断自身几何）');}}
function renderRaw(assets){const root=$('gallery');root.innerHTML='';for(const name of assets.filter(x=>!x.startsWith('review_'))){makeFigure(root,name,name==='pcd_overlay.png'?'抽样 observation 点云叠加（不是最终 object）':name);}}
function updateProgress(p){$('progressBar').style.width=(100*p.completed/p.total)+'%';$('progressText').textContent=`已完成 ${p.completed} / ${p.total}`;}
function logicCheck(){const e=$('evidence_sufficient').value,f=$('finding_correct').value,r=$('root_stage_correct').value,h=$('downstream_harm').value,a=$('repair_action').value,l=$('repair_locality').value,n=$('notes').value.trim();const problems=[];if(e==='YES'&&(f==='UNCERTAIN'||r==='UNCERTAIN'||h==='UNKNOWN'||a==='UNKNOWN'||a==='NEED_MORE_VIEW'))problems.push('证据为 YES 时，核心结论不能仍是 UNCERTAIN、UNKNOWN 或 NEED_MORE_VIEW。');if(e==='PARTIAL'&&!n)problems.push('证据为 PARTIAL 时，请在备注写清缺少哪一环。');if(e==='NO'&&f!=='UNCERTAIN')problems.push('证据为 NO 时，finding 必须是 UNCERTAIN，不能把看不清当成真或假。');if(f==='UNCERTAIN'&&(r!=='UNCERTAIN'||h!=='UNKNOWN'||a!=='NEED_MORE_VIEW'||l!=='NOT_APPLICABLE'))problems.push('finding 为 UNCERTAIN 时，不应继续猜根因、危害或修复。');if(f==='NO'&&(r!=='NOT_APPLICABLE'||h!=='NONE'||a!=='NONE'||l!=='NOT_APPLICABLE'))problems.push('finding 为 NO 时，根因不适用、危害 NONE、修复 NONE。');if(f==='YES'&&r==='NOT_APPLICABLE')problems.push('finding 为 YES 时，根因阶段不能选 NOT_APPLICABLE。');if(h==='NONE'&&a&&a!=='NONE')problems.push('危害 NONE 与非 NONE 修复动作矛盾。');if(a==='NONE'&&h&&h!=='NONE')problems.push('修复 NONE 与非 NONE 危害矛盾。');if((a==='NONE'||a==='NEED_MORE_VIEW')&&l&&l!=='NOT_APPLICABLE')problems.push('当前没有可执行动作时，修复范围应为 NOT_APPLICABLE。');if(['REASSIGN_OBSERVATION','MERGE_OBJECTS','SPLIT_OBJECT'].includes(a)&&l&&l!=='MULTI_OBJECT')problems.push(`${a} 应选 MULTI_OBJECT。`);const root=$('logicHint');root.className='logic '+(problems.length?'error':'ok');root.textContent=problems.length?problems.join(' '):'目前已填写选项在逻辑上自洽；这不代表判断一定正确。';}
function setForm(label){for(const field of fields){$(field).value=label?.[field]??'';}priorSeconds=Number(label?.review_seconds||0);$('savedState').className=label?'saved':'muted';$('savedState').textContent=label?'本例已有保存记录，可修改后再次保存':'尚未保存';openedAt=Date.now();logicCheck();}
async function loadCase(index){$('message').textContent='';const res=await fetch('/api/case?index='+index,{cache:'no-store'});if(!res.ok){let data={};try{data=await res.json();}catch{}$('message').className='error';$('message').textContent=data.error||'读取案例失败';return;}payload=await res.json();current=payload.index;updateProgress(payload.progress);$('position').textContent=`第 ${payload.position} / ${payload.total} 例`;$('caseTitle').textContent=`${payload.case.checker_id||''} · ${payload.case.subtype||'风险案例'}`;$('caseMeta').textContent=`场景 ${payload.scene_id} · 阶段 ${payload.case.stage||''} · 案例 ${payload.case_uid}`;$('reviewQuestion').textContent=REVIEW_QUESTIONS[payload.case.checker_id]||'判断这个候选是否对应现实中的建图错误，以及它是否伤害最终对象图。';blocks('hypotheses',payload.case.hypotheses);blocks('facts',payload.case.proven_facts);const limits=[];(payload.case.vetoes||[]).forEach(x=>limits.push('VETO: '+(typeof x==='string'?x:JSON.stringify(x))));(payload.case.missing_evidence||[]).forEach(x=>limits.push('MISSING: '+(typeof x==='string'?x:JSON.stringify(x))));blocks('limits',limits,'fact warn');renderContract(payload.review_evidence);renderDecisions(payload.review_evidence);renderTriggers(payload.review_evidence);renderObjects(payload.review_evidence);renderRepresentative(payload.review_evidence);renderFinal(payload.review_evidence);renderRaw(payload.assets);setForm(payload.label);$('prev').disabled=current===0;$('next').disabled=current===payload.total-1;window.scrollTo({top:0,behavior:'smooth'});}
async function save(){const body={scene_id:payload.scene_id,case_uid:payload.case_uid,review_seconds:priorSeconds+(Date.now()-openedAt)/1000};for(const field of fields)body[field]=$(field).value;const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await res.json();if(!res.ok){$('message').className='error';$('message').textContent=data.error||'保存失败';return false;}$('message').className='saved';$('message').textContent=`已保存。总进度 ${data.completed}/${data.total}`;priorSeconds=body.review_seconds;openedAt=Date.now();updateProgress(data);return true;}
$('save').onclick=save;$('prev').onclick=()=>loadCase(Math.max(0,current-1));$('next').onclick=()=>loadCase(Math.min(payload.total-1,current+1));for(const field of fields){$(field).addEventListener('change',logicCheck);}document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();save();}});loadCase(0);
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
