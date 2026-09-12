from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote


PROTOCOL = "vlm_gate_manual_review_with_gt_v1_20260912"


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


def action_choice(
    action: Mapping[str, Any],
    gate: Mapping[str, Any],
    association: Mapping[str, Any],
) -> str:
    kind = str(action.get("kind") or "UNRESOLVED")
    if kind != "ATTACH":
        return kind
    object_uids = [str(item) for item in association.get("object_uids_before") or []]
    target_uid = str(action.get("object_uid") or "")
    try:
        target_index = object_uids.index(target_uid)
    except ValueError:
        return f"ATTACH:{target_uid[:8] or '?'}"
    reverse_alias = {
        int(index): str(alias)
        for alias, index in (gate.get("candidate_alias_to_object_index") or {}).items()
    }
    return reverse_alias.get(target_index, f"ATTACH#{target_index}")


def gt_oracle_choice(
    quality_status: str | None,
    canonical_uid: str | None,
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[str, str]:
    """Describe the GT-supported action without conflating it with execution."""
    candidates = list(candidates)
    if quality_status == "MIXED":
        return "DISCARD", "MIXED observation"
    if quality_status == "UNSCORABLE":
        return "UNCERTAIN", "GT evidence is unscorable"
    if quality_status != "CLEAN":
        return "UNCERTAIN", f"quality={quality_status or 'missing'}"

    canonical = next((item for item in candidates if item.get("canonical")), None)
    if canonical is not None:
        return f"Candidate {canonical['alias']}", "same GT canonical prediction"

    same_gt = [item for item in candidates if item.get("same_gt")]
    if same_gt:
        aliases = "/".join(str(item["alias"]) for item in same_gt)
        return f"Candidate {aliases}", "same GT fragment prediction"

    if canonical_uid:
        return "NOT SHOWN", "same-GT canonical prediction exists outside shown candidates"
    uncertain_shown = [
        item
        for item in candidates
        if item.get("identity_status") != "RELIABLE"
    ]
    if uncertain_shown:
        aliases = "/".join(str(item["alias"]) for item in uncertain_shown)
        return "UNCERTAIN", f"shown candidate {aliases} has unresolved identity"
    return "NEW", "all shown candidates are reliably assigned to other GT instances"


def build_cases(
    run_dir: Path,
    metrics_dir: Path,
    output_dir: Path,
    verify_images: bool,
) -> list[dict[str, Any]]:
    decisions = list(iter_jsonl(metrics_dir / "event_decisions.jsonl"))
    labels = {
        str(row["obs_uid"]): row
        for row in iter_jsonl(metrics_dir / "observation_labels.jsonl")
    }
    gates = {
        str(row["current_observation_uid"]): row
        for row in iter_jsonl(run_dir / "blocking_association_gate/events.jsonl")
    }
    associations = {
        str(row["obs_uid"]): row
        for row in iter_jsonl(run_dir / "evidence/associations.jsonl")
    }
    identities: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(metrics_dir / "prediction_identities.jsonl"):
        identities[str(row["obs_uid"])][str(row["object_uid"])] = row

    blind_root = run_dir / "blocking_association_gate/human_annotation_blind"
    missing: dict[str, list[str]] = defaultdict(list)
    cases: list[dict[str, Any]] = []
    for decision in decisions:
        obs_uid = str(decision["obs_uid"])
        gate = gates.get(obs_uid)
        label = labels.get(obs_uid)
        association = associations.get(obs_uid)
        if gate is None:
            missing["gate"].append(obs_uid)
            continue
        if label is None:
            missing["label"].append(obs_uid)
            continue
        if association is None:
            missing["association"].append(obs_uid)
            continue
        case_id = str(gate.get("human_annotation_case_id") or "")
        case_dir = blind_root / "cases" / case_id
        case_json_path = case_dir / "case.json"
        if not case_json_path.is_file():
            missing["blind_case"].append(obs_uid)
            continue
        blind_case = json.loads(case_json_path.read_text(encoding="utf-8"))
        blind_images = {
            str(item["path"]): item for item in blind_case.get("images") or []
        }
        image_rows = []
        for order, evidence in enumerate(gate.get("evidence") or [], 1):
            name = str(evidence.get("path") or "")
            blind_meta = blind_images.get(name)
            image_path = case_dir / name
            if blind_meta is None or not image_path.is_file():
                missing["image"].append(f"{obs_uid}:{name}")
                continue
            expected = str(evidence.get("sha256") or "")
            if expected and str(blind_meta.get("sha256") or "") != expected:
                raise ValueError(f"blind image metadata hash mismatch: {case_id}/{name}")
            if verify_images and expected and sha256_file(image_path) != expected:
                raise ValueError(f"blind image content hash mismatch: {case_id}/{name}")
            relative = os.path.relpath(image_path, output_dir).replace(os.sep, "/")
            image_rows.append(
                {
                    "order": order,
                    "label": evidence.get("label"),
                    "name": name,
                    "url": "/".join(quote(part) for part in relative.split("/")),
                    "sha256": expected,
                    "width": blind_meta.get("width"),
                    "height": blind_meta.get("height"),
                }
            )

        object_uids = [str(item) for item in association.get("object_uids_before") or []]
        scores = dict(
            (gate.get("audit_scores_hidden_from_vlm") or {}).get("candidate_scores")
            or {}
        )
        identity_map = identities.get(obs_uid, {})
        gt_value = label.get("assigned_gt_instance")
        gt_id = int(gt_value) if gt_value is not None else None
        canonical_uid = None
        if gt_id is not None:
            canonical_uid = (decision.get("canonical_prediction_by_gt") or {}).get(str(gt_id))
        candidates = []
        for alias, index_value in sorted(
            (gate.get("candidate_alias_to_object_index") or {}).items()
        ):
            index = int(index_value)
            uid = object_uids[index] if 0 <= index < len(object_uids) else None
            identity = dict(identity_map.get(str(uid)) or {}) if uid else {}
            candidate_gt = identity.get("gt_instance")
            candidates.append(
                {
                    "alias": str(alias),
                    "object_index": index,
                    "object_uid": uid,
                    "score": scores.get(str(alias)),
                    "identity_status": identity.get("identity_status") or "MISSING",
                    "gt_instance": candidate_gt,
                    "identity_purity": identity.get("identity_purity"),
                    "support_count": identity.get("dominant_support_count"),
                    "same_gt": gt_id is not None and candidate_gt == gt_id,
                    "canonical": bool(canonical_uid and uid == canonical_uid),
                }
            )

        trigger = dict(gate.get("trigger") or {})
        hidden_scores = dict(gate.get("audit_scores_hidden_from_vlm") or {})
        model_output = dict(gate.get("model_output") or {})
        baseline_result = dict(decision.get("baseline_result") or {})
        final_result = dict(decision.get("final_result") or {})
        execution_result = dict(decision.get("execution_result") or {})
        staged_decision = dict(decision.get("vlm_staged_decision") or {})
        merge_aliases = [
            str(alias)
            for alias in (staged_decision.get("same_aliases") or [])
            if alias is not None
        ]
        vlm_staged_kind = staged_decision.get("kind")
        vlm_intent = (
            "MERGE " + "+".join(merge_aliases)
            if vlm_staged_kind == "MERGE_REVIEW" and merge_aliases
            else model_output.get("choice")
        )
        baseline_choice = action_choice(
            dict(decision.get("baseline_action") or {}), gate, association
        )
        final_choice = action_choice(
            dict(decision.get("final_action") or {}), gate, association
        )
        oracle_choice, oracle_reason = gt_oracle_choice(
            label.get("quality_status"),
            str(canonical_uid) if canonical_uid else None,
            candidates,
        )
        cases.append(
            {
                "case_id": case_id,
                "event_id": gate.get("event_id"),
                "event_sequence": decision.get("event_sequence"),
                "obs_uid": obs_uid,
                "frame_index": decision.get("frame_index"),
                "source_frame_id": label.get("source_frame_id"),
                "class_name": label.get("class_name"),
                "images": image_rows,
                "allowed_choices": [
                    *[item["alias"] for item in candidates],
                    "NEW",
                    "DISCARD",
                    "UNCERTAIN",
                ],
                "candidates": candidates,
                "trigger_kind": trigger.get("kind"),
                "trigger_reasons": trigger.get("reasons") or [],
                "top1": trigger.get("top1"),
                "top2": trigger.get("top2"),
                "margin": trigger.get("margin"),
                "sim_threshold": hidden_scores.get("sim_threshold"),
                "margin_threshold": hidden_scores.get("margin_threshold"),
                "threshold_distance": (
                    trigger.get("threshold_distance")
                    if trigger.get("threshold_distance") is not None
                    else hidden_scores.get("threshold_distance")
                ),
                "baseline_choice": baseline_choice,
                "vlm_choice": vlm_intent,
                "vlm_raw_choice": model_output.get("choice"),
                "vlm_staged_kind": vlm_staged_kind,
                "vlm_staged_reason": staged_decision.get("reason_code"),
                "vlm_merge_aliases": merge_aliases,
                "vlm_confidence": model_output.get("confidence"),
                "final_choice": final_choice,
                "execution_category": execution_result.get("category"),
                "merge_review_evaluation": final_result.get(
                    "merge_review_evaluation"
                ),
                "linked_merge_reviews": decision.get("linked_merge_reviews") or [],
                "execution_matches_vlm": bool(
                    decision.get("final_action_matches_association")
                ),
                "changed": bool(decision.get("changed")),
                "route_reason": decision.get("route_reason"),
                "baseline_category": baseline_result.get("category"),
                "final_category": final_result.get("category"),
                "new_action_context": final_result.get("new_action_context"),
                "global_uncertain_prediction_count": final_result.get(
                    "global_uncertain_prediction_count"
                ),
                "shown_uncertain_prediction_count": final_result.get(
                    "shown_uncertain_prediction_count"
                ),
                "oracle_choice": oracle_choice,
                "oracle_reason": oracle_reason,
                "quality_status": label.get("quality_status"),
                "quality_reason": label.get("quality_reason"),
                "gt_instance": gt_id,
                "gt_class": label.get("top1_gt_class"),
                "gt_top1_purity": label.get("top1_purity"),
                "gt_top2_instance": label.get("top2_gt_instance"),
                "gt_top2_purity": label.get("top2_purity"),
                "gt_support_ratio": label.get("gt_support_ratio"),
                "valid_depth_ratio": label.get("valid_depth_ratio"),
                "mask_area": label.get("mask_area"),
                "prompt_urls": {
                    name: "/".join(
                        quote(part)
                        for part in os.path.relpath(case_dir / name, output_dir)
                        .replace(os.sep, "/")
                        .split("/")
                    )
                    for name in ("system_prompt.txt", "user_prompt.txt")
                    if (case_dir / name).is_file()
                },
            }
        )
    if missing:
        raise ValueError(
            "missing review inputs: "
            + ", ".join(f"{key}={len(value)}" for key, value in sorted(missing.items()))
        )
    cases.sort(key=lambda row: (int(row.get("event_sequence") or 0), str(row["event_id"])))
    if len(cases) != len(decisions):
        raise ValueError(f"case count mismatch: cases={len(cases)}, decisions={len(decisions)}")
    return cases


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>VLM Gate / Human Verification Desk</title>
<style>
:root{color-scheme:dark;--bg:#090d0d;--paper:#111717;--panel:#161e1d;--line:#34413e;--text:#edf3ef;--muted:#91a09a;--acid:#d7ff45;--cyan:#3de2cf;--orange:#ffad42;--red:#ff6257;--blue:#72a7ff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;min-height:100vh;background:radial-gradient(circle at 80% -10%,#26443b 0,transparent 30%),linear-gradient(135deg,#080b0b,#111817 52%,#080b0a);color:var(--text);font:14px/1.45 Bahnschrift,"Microsoft YaHei UI",sans-serif}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.08;background-image:repeating-linear-gradient(90deg,transparent 0 79px,#caffee 80px),repeating-linear-gradient(0deg,transparent 0 79px,#caffee 80px)}
button,input,select,textarea{font:inherit}.topbar{position:sticky;top:0;z-index:20;padding:12px 20px;background:#0b100fed;border-bottom:1px solid var(--line);backdrop-filter:blur(18px)}
.brand{display:flex;align-items:flex-end;gap:13px}.eyebrow{font:800 10px ui-monospace,monospace;letter-spacing:.2em;color:var(--acid)}h1{margin:0;font:800 22px/1 Georgia,"Microsoft YaHei UI",serif}.run{margin-left:auto;color:var(--muted);font:10px ui-monospace,monospace}.filters{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:11px}select,input,textarea{color:var(--text);background:#18201f;border:1px solid #465550;border-radius:3px;padding:8px 10px}input{min-width:250px}.filters button,.nav button,.review button{color:var(--text);background:#18201f;border:1px solid #465550;border-radius:3px;padding:8px 11px;cursor:pointer}.filters button:hover,.nav button:hover,.review button:hover{border-color:var(--acid)}.counter{margin-left:auto;font:800 11px ui-monospace,monospace;color:var(--muted)}
main{position:relative;max-width:1780px;margin:auto;padding:18px 20px 190px}.casehead{display:grid;grid-template-columns:1fr auto;gap:16px;align-items:start;margin-bottom:12px}.case-title small{display:block;color:var(--muted);font:800 10px ui-monospace,monospace;letter-spacing:.12em}.case-title strong{display:block;font:900 clamp(28px,3vw,48px)/1 Georgia,"Microsoft YaHei UI",serif;letter-spacing:-.04em}.badges{display:flex;justify-content:flex-end;gap:6px;flex-wrap:wrap}.badge{padding:5px 8px;border:1px solid var(--line);background:#111716;font:800 10px ui-monospace,monospace}.badge.clean{color:var(--cyan)}.badge.mixed{color:var(--orange)}.badge.error{color:var(--red)}.badge.changed{color:var(--blue)}
.decision-rail{display:grid;grid-template-columns:1fr 44px 1fr 44px 1.3fr;align-items:stretch;margin-bottom:8px}.decision{padding:11px 13px;background:var(--panel);border:1px solid var(--line)}.decision span{display:block;color:var(--muted);font:800 9px ui-monospace,monospace;letter-spacing:.15em}.decision b{display:block;font:900 21px Georgia,serif;margin-top:4px}.decision small{color:var(--muted)}.decision.verdict.error{border-color:var(--red)}.decision.verdict.good{border-color:var(--cyan)}.decision.verdict.ambiguous{border-color:var(--orange)}.arrow{display:grid;place-items:center;color:var(--acid);font-size:20px}
.evidence{display:grid;grid-template-columns:1.14fr repeat(3,minmax(0,1fr));gap:8px}.shot{min-width:0;background:#050707;border:1px solid var(--line)}.shot.gt-oracle{border-color:var(--acid);box-shadow:inset 0 0 0 1px var(--acid)}.shot.vlm-selected{outline:2px solid var(--blue);outline-offset:-4px}.shot.current{border-color:#658a58}.shot-head{min-height:45px;padding:8px 10px;display:flex;justify-content:space-between;align-items:center;background:linear-gradient(90deg,#1c2523,#111716);border-bottom:1px solid var(--line)}.shot-head b{font:900 17px Georgia,serif}.shot-head span{color:var(--muted);font:800 9px ui-monospace,monospace;text-align:right}.shot img{display:block;width:100%;height:min(47vh,530px);object-fit:contain;cursor:zoom-in}.shot-foot{min-height:69px;padding:8px 10px;color:#aebbb6;font:11px/1.55 ui-monospace,monospace}.shot-foot strong{color:white}.oracle-tag{color:var(--acid);font-weight:900}.selected-tag,.merge-tag{color:var(--blue);font-weight:900}.audit-ok{color:var(--muted)}.audit-mismatch{color:var(--red);font-weight:900}.missing{height:300px;display:grid;place-items:center;color:var(--muted)}
.details{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:8px;margin-top:8px}.card{background:var(--panel);border:1px solid var(--line);padding:12px;min-width:0}.card h2{margin:0 0 9px;color:var(--acid);font:900 10px ui-monospace,monospace;letter-spacing:.15em}.metric-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:7px}.metric{padding-bottom:5px;border-bottom:1px solid #2d3835}.metric span{display:block;color:var(--muted);font:9px ui-monospace,monospace}.metric b{display:block;overflow-wrap:anywhere}.scores{display:flex;gap:6px;flex-wrap:wrap}.score{padding:6px 9px;background:#0d1211;border:1px solid var(--line);font:800 11px ui-monospace,monospace}.ids{font:11px/1.65 ui-monospace,monospace;color:#a9b6b1;overflow-wrap:anywhere}.ids em{color:var(--red);font-style:normal}.links a{color:var(--cyan)}
.review{position:fixed;left:0;right:0;bottom:0;z-index:30;background:#0a0e0ded;border-top:1px solid #53615d;backdrop-filter:blur(18px);padding:10px 20px}.review-inner{max-width:1780px;margin:auto;display:grid;grid-template-columns:auto 1fr 320px auto;gap:10px;align-items:center}.review-label{font:900 10px ui-monospace,monospace;color:var(--acid);letter-spacing:.14em}.choices{display:flex;gap:5px;flex-wrap:wrap}.choice.active{background:var(--acid);border-color:var(--acid);color:#111;font-weight:900}.choice[data-choice="DISCARD"].active{background:var(--red);border-color:var(--red);color:white}.review textarea{height:38px;min-height:38px;resize:vertical}.review-actions{display:flex;gap:5px;flex-wrap:wrap}.review .save{background:#235c50}.review .nextblank{border-color:var(--orange)}.progress{color:var(--muted);font:10px ui-monospace,monospace;white-space:nowrap}.issue{display:flex;gap:5px;align-items:center;color:var(--orange);font-size:11px}
#zoom{display:none;position:fixed;inset:0;z-index:50;background:#000e;padding:18px;align-items:center;justify-content:center}#zoom.open{display:flex}#zoom img{max-width:98vw;max-height:96vh;object-fit:contain}.empty{padding:120px;text-align:center;color:var(--muted)}
@media(max-width:1200px){.evidence{grid-template-columns:1fr 1fr}.details{grid-template-columns:1fr}.review-inner{grid-template-columns:1fr}.review-label{display:none}.review{max-height:220px;overflow:auto}main{padding-bottom:250px}}
@media(max-width:700px){.topbar,main,.review{padding-left:9px;padding-right:9px}.run{display:none}.casehead{grid-template-columns:1fr}.badges{justify-content:flex-start}.decision-rail{grid-template-columns:1fr}.arrow{height:24px;transform:rotate(90deg)}.evidence{grid-template-columns:1fr}.shot img{height:360px}.counter{margin-left:0}}
</style></head><body>
<header class="topbar"><div class="brand"><div class="eyebrow">FORENSIC GATE DESK / GT REVEALED</div><h1>VLM 门控事件人工复核</h1><div class="run" id="run"></div></div><div class="filters"><select id="category"></select><select id="route"></select><select id="quality"></select><select id="changed"><option value="ALL">全部动作</option><option value="YES">仅 VLM 改变</option><option value="NO">仅未改变</option></select><input id="search" placeholder="搜索 event / observation / class"><div class="nav"><button id="prev">←</button><button id="next">→</button><button id="nextBlank">下一未复核</button></div><button id="exportJsonl">导出 JSONL</button><button id="exportJson">导出全部状态</button><button id="import">导入</button><input id="importFile" type="file" hidden><span class="counter" id="counter"></span></div></header>
<main id="main"><div class="empty">正在装载冻结证据…</div></main>
<section class="review"><div class="review-inner"><div class="review-label">HUMAN ACTION</div><div><div id="choices" class="choices"></div><label class="issue"><input id="issue" type="checkbox"> GT / mask / evidence 有问题</label></div><textarea id="note" placeholder="人工复核依据（可选）"></textarea><div class="review-actions"><select id="confidence"><option value="">置信度</option><option value="high">高</option><option value="medium">中</option><option value="low">低</option></select><button class="save" id="save">保存并下一例</button><button id="clear">清除</button><span class="progress" id="progress"></span></div></div></section>
<div id="zoom"><img alt="放大证据"></div>
<script>const CASES=__CASES__;const META=__META__;
const $=id=>document.getElementById(id),esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const num=v=>v==null?'—':Number(v).toFixed(4).replace(/0+$/,'').replace(/\.$/,'');const pct=v=>v==null?'—':(Number(v)*100).toFixed(1)+'%';const short=v=>v?String(v).slice(0,8):'—';
let rows=[],index=0,labels={},draft=null;const KEY=META.storage_key;try{labels=JSON.parse(localStorage.getItem(KEY)||'{}')}catch{labels={}}
function persist(){localStorage.setItem(KEY,JSON.stringify(labels));progress()}function choiceText(v){return /^[A-Z]$/.test(v)?`Candidate ${v}`:v==='NEW'?'NEW 新建':v==='DISCARD'?'DISCARD 丢弃':'UNCERTAIN 不确定'}
function filters(){const cat=$('category').value,route=$('route').value,quality=$('quality').value,changed=$('changed').value,q=$('search').value.trim().toLowerCase();rows=CASES.filter(c=>(cat==='ALL'||c.final_category===cat)&&(route==='ALL'||c.route_reason===route)&&(quality==='ALL'||c.quality_status===quality)&&(changed==='ALL'||(changed==='YES')===c.changed)&&(!q||[c.event_id,c.obs_uid,c.class_name,c.gt_class,c.baseline_category,c.final_category].join(' ').toLowerCase().includes(q)));index=Math.min(index,Math.max(0,rows.length-1));render()}
function options(id,values,label){const e=$(id);e.innerHTML=`<option value="ALL">${label}</option>`+[...values].sort().map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}
function badgeClass(v){return /WRONG|FALSE|DUPLICATE/.test(v)?'error':v==='CLEAN'?'clean':v==='MIXED'?'mixed':''}
function outcomeClass(v){return /WRONG|FALSE|DUPLICATE|MIXED_MASK_(ATTACHMENT|NEW)/.test(v)?'error':/^CORRECT_/.test(v)?'good':/^AMBIGUOUS_/.test(v)?'ambiguous':''}
function outcomeText(v){return /^CORRECT_/.test(v)?'V1 判定：正确':/WRONG|FALSE|DUPLICATE|MIXED_MASK_(ATTACHMENT|NEW)/.test(v)?'V1 判定：错误':/^AMBIGUOUS_/.test(v)?'V1 判定：身份不确定':v==='UNSCORABLE_OBSERVATION'?'V1 判定：不可计分':'V1 判定：未解析'}
function figure(c,img){const alias=(img.name.match(/candidate_([A-Z])/)||[])[1],cand=c.candidates.find(x=>x.alias===alias);let foot='原始 VLM 输入 · '+esc(img.width)+'×'+esc(img.height);const classes=[img.name==='quality.jpg'?'current':''];if(cand){const rel=cand.canonical?'同 GT · CANONICAL':cand.same_gt?'同 GT · FRAGMENT':cand.identity_status==='RELIABLE'?'其他 GT '+esc(cand.gt_instance):'身份不确定';const inMerge=(c.vlm_merge_aliases||[]).includes(alias),selected=c.vlm_raw_choice===alias;const tags=[cand.same_gt?'<span class="oracle-tag">GT ORACLE</span>':'',inMerge?'<span class="merge-tag">VLM MERGE SET</span>':'',selected?'<span class="selected-tag">VLM SELECTED</span>':''].filter(Boolean).join(' · ');foot=`${tags}${tags?'<br>':''}score <strong>${num(cand.score)}</strong> · object #${esc(cand.object_index)} / ${esc(short(cand.object_uid))}<br>${rel} · identity ${esc(cand.identity_status)} · purity ${pct(cand.identity_purity)}`;if(cand.same_gt)classes.push('gt-oracle');if(inMerge||selected)classes.push('vlm-selected')}return `<article class="shot ${classes.filter(Boolean).join(' ')}"><div class="shot-head"><b>${esc(img.label)}</b><span>SHA ${esc(short(img.sha256))}</span></div><img src="${esc(img.url)}" alt="${esc(img.label)}"><div class="shot-foot">${foot}</div></article>`}
function metric(k,v){return `<div class="metric"><span>${esc(k)}</span><b>${esc(v)}</b></div>`}
function mergeAudit(c){if(c.vlm_staged_kind!=='MERGE_REVIEW')return '';const linked=(c.linked_merge_reviews||[]).map(x=>{const eventual=x.eventually_merged?`MERGED @ ${x.executed_event_id||x.executed_frame_idx||'later'}`:'NOT MERGED';return `${esc(x.event_id||x.pair_key||'pair')} · vote ${esc(x.model_choice??'—')} · immediate ${esc(x.execution_at_review??'—')} · eventual ${esc(eventual)}`}).join('<br>');return `<br>VLM staged intent: <strong>${esc(c.vlm_choice)}</strong> · ${esc(c.merge_review_evaluation??'—')}<br>execution fallback: <strong>${esc(c.final_choice)}</strong> · ${esc(c.execution_category??'—')}<br>merge lifecycle: ${linked||'no newly linked pair event'}`}
function render(){const c=rows[index];$('prev').disabled=!c||index===0;$('next').disabled=!c||index>=rows.length-1;$('counter').textContent=c?`${index+1}/${rows.length} · 全部 ${CASES.length}`:`0/${rows.length}`;if(!c){$('main').innerHTML='<div class="empty">当前筛选没有事件</div>';review();return}const scores=c.candidates.map(x=>`<span class="score">${esc(x.alias)} ${num(x.score)}</span>`).join('');const reasons=(c.trigger_reasons||[]).join(' + ')||'—';const imgs=c.images.map(x=>figure(c,x)).join('');const auditClass=c.execution_matches_vlm?'audit-ok':'audit-mismatch';const auditText=c.execution_matches_vlm?'与 VLM 请求一致':'与 VLM 请求不一致';const newAudit=c.new_action_context?`<br>NEW context: <strong>${esc(c.new_action_context)}</strong><br>uncertain identities: shown ${esc(c.shown_uncertain_prediction_count??0)} / global ${esc(c.global_uncertain_prediction_count??0)}`:'';const stagedAudit=mergeAudit(c);$('main').innerHTML=`<div class="casehead"><div class="case-title"><small>${esc(c.case_id)} · FRAME ${esc(c.frame_index)} · EVENT #${esc(c.event_sequence)} · ${esc(c.event_id)}</small><strong>${esc(c.class_name)} → GT ${esc(c.gt_instance)}</strong></div><div class="badges"><span class="badge ${badgeClass(c.quality_status)}">${esc(c.quality_status)}</span><span class="badge ${badgeClass(c.final_category)}">${esc(c.final_category)}</span>${c.changed?'<span class="badge changed">VLM CHANGED</span>':''}</div></div><section class="decision-rail"><div class="decision"><span>BASELINE ACTION</span><b>${esc(c.baseline_choice)}</b><small>${esc(c.baseline_category)}</small></div><div class="arrow">→</div><div class="decision"><span>VLM INTENT</span><b>${esc(c.vlm_choice)}</b><small>${c.vlm_staged_kind==='MERGE_REVIEW'?`fallback ${esc(c.vlm_raw_choice)} · ${esc(c.vlm_staged_reason)}`:`confidence ${esc(c.vlm_confidence)}`}</small></div><div class="arrow">→</div><div class="decision verdict ${outcomeClass(c.final_category)}"><span>V1 EVALUATION</span><b>${esc(c.final_category)}</b><small>${esc(outcomeText(c.final_category))}</small></div></section><section class="evidence">${imgs}</section><section class="details"><div class="card"><h2>TRIGGER PARAMETERS</h2><div class="metric-grid">${metric('kind / reasons',c.trigger_kind+' / '+reasons)}${metric('top1 / top2',num(c.top1)+' / '+num(c.top2))}${metric('margin',num(c.margin))}${metric('sim threshold',num(c.sim_threshold))}${metric('margin threshold',num(c.margin_threshold))}${metric('threshold distance',num(c.threshold_distance))}</div><div class="scores" style="margin-top:10px">${scores}</div></div><div class="card"><h2>OFFICIAL GT / ORACLE</h2><div class="metric-grid">${metric('GT oracle action',c.oracle_choice)}${metric('oracle basis',c.oracle_reason)}${metric('GT class / instance',(c.gt_class??'—')+' / '+(c.gt_instance??'—'))}${metric('top1 purity',pct(c.gt_top1_purity))}${metric('top2 instance / purity',(c.gt_top2_instance??'—')+' / '+pct(c.gt_top2_purity))}${metric('GT support',pct(c.gt_support_ratio))}${metric('valid depth',pct(c.valid_depth_ratio))}${metric('mask area',c.mask_area)}</div></div><div class="card"><h2>TRACE / EXECUTION AUDIT</h2><div class="ids">applied action: <strong>${esc(c.final_choice)}</strong> · <span class="${auditClass}">${esc(auditText)}</span><br>route: ${esc(c.route_reason)}<br>quality reason: <em>${esc(c.quality_reason)}</em>${newAudit}${stagedAudit}<br>obs: ${esc(c.obs_uid)}<br>event: ${esc(c.event_id)}<br><span class="links">${Object.entries(c.prompt_urls).map(([n,u])=>`<a href="${esc(u)}" target="_blank">${esc(n)}</a>`).join(' · ')}</span></div></div></section>`;document.querySelectorAll('.shot img').forEach(img=>img.onclick=()=>{$('#zoom img').src=img.src;$('zoom').classList.add('open')});review();window.scrollTo({top:0,behavior:'smooth'})}
function review(){const c=rows[index];if(!c){$('choices').innerHTML='';return}const saved=labels[c.event_id]||{};draft=saved.choice||null;$('choices').innerHTML=c.allowed_choices.map(v=>`<button class="choice ${draft===v?'active':''}" data-choice="${esc(v)}">${esc(choiceText(v))}</button>`).join('');document.querySelectorAll('.choice').forEach(b=>b.onclick=()=>{draft=b.dataset.choice;reviewDraft()});$('note').value=saved.note||'';$('confidence').value=saved.confidence||'';$('issue').checked=!!saved.issue;progress()}
function reviewDraft(){document.querySelectorAll('.choice').forEach(b=>b.classList.toggle('active',b.dataset.choice===draft))}
function save(next=true){const c=rows[index];if(!c||!draft)return;labels[c.event_id]={case_id:c.case_id,event_id:c.event_id,obs_uid:c.obs_uid,choice:draft,confidence:$('confidence').value||null,issue:$('issue').checked,note:$('note').value.trim(),baseline_choice:c.baseline_choice,vlm_choice:c.vlm_choice,final_choice:c.final_choice,auto_category:c.final_category,reviewed_at:new Date().toISOString()};persist();if(next)nextBlank()}
function progress(){const n=Object.values(labels).filter(x=>x.choice).length;$('progress').textContent=`已复核 ${n}/${CASES.length}`}
function step(d){if(!rows.length)return;index=Math.max(0,Math.min(rows.length-1,index+d));render()}
function nextBlank(){if(!rows.length)return;for(let n=1;n<=rows.length;n++){const i=(index+n)%rows.length;if(!labels[rows[i].event_id]?.choice){index=i;render();return}}step(1)}
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function exportJsonl(){const out=CASES.map(c=>labels[c.event_id]).filter(x=>x?.choice);download('room0_gate_human_review.jsonl',out.map(JSON.stringify).join('\n')+'\n','application/x-ndjson')}
function exportJson(){download('room0_gate_human_review_state.json',JSON.stringify({protocol:META.protocol,exported_at:new Date().toISOString(),labels},null,2),'application/json')}
async function importFile(file){const text=await file.text();let vals=[];try{const v=JSON.parse(text);vals=v.labels?Object.values(v.labels):Array.isArray(v)?v:[v]}catch{vals=text.split(/\r?\n/).filter(Boolean).map(JSON.parse)}for(const v of vals)if(v.event_id)labels[v.event_id]=v;persist();review()}
$('run').textContent=`${META.run_id} · ${META.source_evaluation_protocol}`;options('category',new Set(CASES.map(c=>c.final_category)),'全部评测结果');options('route',new Set(CASES.map(c=>c.route_reason)),'全部 route');options('quality',new Set(CASES.map(c=>c.quality_status)),'全部 GT quality');for(const id of ['category','route','quality','changed'])$(id).onchange=()=>{index=0;filters()};let timer;$('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{index=0;filters()},180)};$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('nextBlank').onclick=nextBlank;$('save').onclick=()=>save(true);$('clear').onclick=()=>{const c=rows[index];if(c){delete labels[c.event_id];persist();review()}};$('exportJsonl').onclick=exportJsonl;$('exportJson').onclick=exportJson;$('import').onclick=()=>$('importFile').click();$('importFile').onchange=e=>e.target.files[0]&&importFile(e.target.files[0]);$('zoom').onclick=()=>$('zoom').classList.remove('open');document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')step(-1);else if(e.key==='ArrowRight')step(1);else{const c=rows[index],k=e.key.toUpperCase(),v=k==='N'?'NEW':k==='D'?'DISCARD':k==='U'?'UNCERTAIN':k;if(c?.allowed_choices.includes(v)){draft=v;reviewDraft()}}};filters();
</script></body></html>'''


def generate(run_dir: Path, metrics_dir: Path, output_dir: Path, verify_images: bool) -> Path:
    run_dir = run_dir.resolve()
    metrics_dir = metrics_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        evaluation_metrics = json.loads(
            (metrics_dir / "metrics.json").read_text(encoding="utf-8")
        )
        cases = build_cases(run_dir, metrics_dir, output_dir, verify_images)
        run_id = str(cases[0]["obs_uid"]).split("_f", 1)[0]
        meta = {
            "protocol": PROTOCOL,
            "source_evaluation_protocol": evaluation_metrics.get("protocol"),
            "run_id": run_id,
            "case_count": len(cases),
            "storage_key": f"{PROTOCOL}:{run_id}",
            "category_counts": dict(sorted(Counter(str(c["final_category"]) for c in cases).items())),
            "route_counts": dict(sorted(Counter(str(c["route_reason"]) for c in cases).items())),
            "quality_counts": dict(sorted(Counter(str(c["quality_status"]) for c in cases).items())),
            "changed_count": sum(bool(c["changed"]) for c in cases),
            "decision_rail_semantics": [
                "baseline_action",
                "vlm_staged_intent",
                "evaluation_category",
            ],
            "candidate_highlight_semantics": {
                "green": "GT-supported same-instance candidate",
                "blue": "candidate selected by the VLM or included in its merge set",
            },
            "images_are_exact_vlm_inputs": True,
            "image_content_hashes_verified": bool(verify_images),
            "source_fingerprints": {
                "gate_events": sha256_file(run_dir / "blocking_association_gate/events.jsonl"),
                "event_decisions": sha256_file(metrics_dir / "event_decisions.jsonl"),
                "observation_labels": sha256_file(metrics_dir / "observation_labels.jsonl"),
            },
        }
        payload = json.dumps(cases, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        meta_payload = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        html = HTML.replace("__CASES__", payload).replace("__META__", meta_payload)
        (output_dir / "index.html").write_text(html, encoding="utf-8")
        (output_dir / "review_manifest.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_dir
    except Exception:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()
        output_dir.rmdir()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate all-gate VLM v1 human review webpage")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    metrics_dir = (args.metrics_dir or run_dir / "observation_gt_v1_replica_official").resolve()
    output = generate(run_dir, metrics_dir, args.out, args.verify_images)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
