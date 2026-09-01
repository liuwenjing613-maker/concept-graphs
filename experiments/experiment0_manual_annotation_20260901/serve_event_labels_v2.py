#!/usr/bin/env python3
"""Loopback-only two-stage annotation UI for ATTACH/NEW routing schema v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import serve_event_labels as legacy_server
from label_logic_v2 import (
    derive_routing_label,
    validate_blind_label,
    validate_final_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    return parser.parse_args()


class AnnotationStoreV2(legacy_server.AnnotationStore):
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
                action = str(private["original_action_type"])
                target_code = private.get("original_target_code")
                selected_scores = None
                if action == "ATTACH_EXISTING":
                    candidate = next(
                        candidate
                        for candidate in private["candidates"]
                        if candidate["code"] == target_code
                    )
                    selected_scores = {
                        "spatial": candidate.get("spatial_score"),
                        "visual": candidate.get("visual_score"),
                        "aggregate": candidate.get("aggregate_score"),
                    }
                payload["reveal"] = {
                    "original_action_type": action,
                    "original_target_code": target_code,
                    "top1_score": private["association_event"].get("top1_score"),
                    "top2_score": private["association_event"].get("top2_score"),
                    "margin": private["association_event"].get("margin"),
                    "threshold": private["association_event"].get("sim_threshold"),
                    "selected_candidate_scores": selected_scores,
                }
            return payload

    def save_blind(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.reload()
            case_uid = str(payload.get("case_uid") or "")
            if case_uid not in self.by_case:
                raise ValueError("未知 case")
            if case_uid in self.drafts or case_uid in self.labels:
                raise ValueError("该 case 的盲标已经锁定；如需纠错请另写裁决覆盖层")
            public, _, _ = self._case_files(case_uid)
            codes = {str(row["code"]) for row in public["candidates"]}
            blind = validate_blind_label(payload, codes)
            saved = {
                "schema_version": "experiment0-human-routing-blind/2.0",
                "case_uid": case_uid,
                "event_uid": public["event_uid"],
                "blind": blind,
                "blind_saved_at_utc": legacy_server.utc_now(),
                "blind_submitted_when_mapper_latest_frame": self.manifest.get(
                    "mapper_latest_frame"
                ),
            }
            self.drafts[case_uid] = saved
            self._atomic_rows(self.drafts_path, self.drafts)
            return {"saved": case_uid, "status": self.status()}

    def save_final(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.reload()
            case_uid = str(payload.get("case_uid") or "")
            if case_uid in self.labels:
                raise ValueError("该 case 的最终标签已经保存；如需纠错请另写裁决覆盖层")
            draft = self.drafts.get(case_uid)
            if draft is None:
                raise ValueError("请先完成盲标")
            public, private, _ = self._case_files(case_uid)
            original_action = str(private["original_action_type"])
            final = validate_final_label(payload, draft["blind"], original_action)
            derived = derive_routing_label(
                draft["blind"],
                final,
                original_action,
                private.get("original_target_code"),
            )
            label = {
                "schema_version": "experiment0-human-identity-routing-label/2.0",
                "case_uid": case_uid,
                "event_uid": public["event_uid"],
                "scene": public["scene"],
                "source_frame": public.get("source_frame"),
                "event_frame_idx": public.get("event_frame_idx"),
                "tminus_snapshot_sha256": public.get("tminus_snapshot_sha256"),
                "blind": draft["blind"],
                "reveal": {
                    "original_action_type": original_action,
                    "original_target_code": private.get("original_target_code"),
                },
                "final": final,
                "derived": derived,
                "saved_at_utc": legacy_server.utc_now(),
                "submitted_when_mapper_latest_frame": self.manifest.get(
                    "mapper_latest_frame"
                ),
                "timeline": {
                    "s_source_frame": public.get("source_frame"),
                    "s_processed_frame_idx": public.get("event_frame_idx"),
                    "mapper_latest_frame_at_event": public.get(
                        "mapper_latest_frame_at_event"
                    ),
                    "d_human_submission_mapper_frame": self.manifest.get(
                        "mapper_latest_frame"
                    ),
                    "h": None,
                    "c": None,
                },
            }
            self.labels[case_uid] = label
            self._atomic_rows(self.labels_path, self.labels)
            return {"saved": case_uid, "derived": derived, "status": self.status()}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"v2 HTML patch expected one occurrence: {old[:80]!r}")
    return text.replace(old, new, 1)


def make_html() -> str:
    html = legacy_server.HTML
    html = replace_once(html, "<title>实验 0 关联事件标注</title>", "<title>实验 0 身份路由 v2 · R2 校准</title>")
    html = replace_once(
        html,
        "实验 0 · 关联事件两阶段标注",
        "实验 0 · ATTACH/NEW 身份路由 v2 · R2 校准",
    )
    html = replace_once(
        html,
        "检测类别、mapper 选择、候选排名、分数、自动 GT 与抽样组在盲标阶段全部隐藏。",
        "检测类别、mapper 的 ATTACH/NEW 动作、目标、排名、分数、自动 GT 与抽样组在盲标阶段全部隐藏。",
    )
    html = replace_once(
        html,
        "<div class=\"section\"><h2>事件前候选节点 A–E</h2>",
        "<div class=\"section\"><h2>事件前 t^- 候选节点</h2>",
    )
    html = replace_once(
        html,
        "<form id=\"blindForm\"><h2>阶段 A：盲标身份</h2>",
        "<form id=\"blindForm\"><h2>阶段 A：盲标身份</h2><div class=\"notice\"><b>quality 三问，按顺序判断：</b><br>① mask 是否同时包含两个可分物体？是 → MIXED。<br>② 若现实中仍是一个完整物理实例，即使画面只看到一部分、被遮挡或被截断，也选 CLEAN/BORDERLINE；<b>局部可见不等于粒度歧义</b>。<br>③ 只有无法稳定决定“它是独立物体，还是另一物体的一部分”时，才选 GRANULARITY，并在实例描述中写清 part-whole 边界。<br><b>同类别、同材质、属于同一窗/家具系统或彼此相邻，都不等于同一实例；必须结合位置、形状和历史。</b></div>",
    )
    html = replace_once(
        html,
        "<div class=\"field\"><label>3. 用一句短语描述物理实例（可选）</label>",
        "<div class=\"field\"><label>3. 身份判断证据状态</label><select id=\"identityEvidence\" required><option value=\"\">请选择</option><option>SUFFICIENT_FOR_IDENTITY</option><option>PARTIAL</option><option>INSUFFICIENT</option></select></div>\n<div class=\"field\"><label>4. 用一句短语描述物理实例（一般可选；GRANULARITY 必须写明 part-whole 边界）</label>",
    )
    html = replace_once(
        html,
        "<div class=\"field\"><label>1. 系统所选节点在关联前是什么状态？</label><select id=\"targetState\"><option value=\"\">请选择</option><option>CLEAN_SINGLE_INSTANCE</option><option>ALREADY_CONTAMINATED</option><option>UNCERTAIN</option></select></div>",
        "<div class=\"field\"><label>1. 原始 ATTACH 目标在事件前是什么状态？NEW 请选择 NOT_APPLICABLE</label><select id=\"targetState\"><option value=\"\">请选择</option><option>CLEAN_SINGLE_INSTANCE</option><option>ALREADY_CONTAMINATED</option><option>UNCERTAIN</option><option>NOT_APPLICABLE</option></select></div>",
    )
    html = replace_once(
        html,
        "<div class=\"field\"><label>2. 若 A–E 没有匹配，完整事件时地图里是否已有正确节点？</label><select id=\"outsideStatus\"><option value=\"\">请选择</option><option>NOT_NEEDED</option><option>MATCH_EXISTS_OUTSIDE</option><option>NO_MATCHING_NODE_EXISTS</option><option>UNCHECKED</option></select></div>",
        "<div class=\"field\"><label>2. 完整事件时 t^- 地图检查结果</label><select id=\"outsideStatus\"><option value=\"\">请选择</option><option>NOT_NEEDED_MATCH_SHOWN</option><option>MATCH_EXISTS_OUTSIDE</option><option>NO_MATCHING_NODE_EXISTS</option><option>UNCHECKED</option></select></div><div class=\"field\"><label>3. 候选外匹配节点 UID（仅 MATCH_EXISTS_OUTSIDE；多个用逗号分隔）</label><textarea id=\"outsideUids\"></textarea></div>",
    )
    html = replace_once(
        html,
        "<div class=\"field\"><label>3. 证据是否足够？</label><select id=\"evidenceStatus\"><option value=\"\">请选择</option><option>YES</option><option>PARTIAL</option><option>NO</option></select></div>",
        "<div class=\"field\"><label>4. 因果备注（可选；这里只记录事实，不直接判 root/cascade）</label><textarea id=\"causalNote\" placeholder=\"例如：目标在本事件前已经混入另一实例\"></textarea></div>",
    )
    html = html.replace("<label>4. 置信度 1–5</label>", "<label>5. 置信度 1–5</label>", 1)
    html = html.replace("<label>5. 备注（PARTIAL/NO 必填）</label>", "<label>6. 备注（PARTIAL/INSUFFICIENT 必填）</label>", 1)
    html = replace_once(
        html,
        "if(data.draft){$('observationQuality').value=data.draft.observation_quality;$('physicalNote').value=data.draft.physical_instance_note||'';}",
        "if(data.draft){$('observationQuality').value=data.draft.observation_quality;$('identityEvidence').value=data.draft.identity_evidence_status;$('physicalNote').value=data.draft.physical_instance_note||'';}",
    )
    html = replace_once(
        html,
        "if(data.reveal){const r=data.reveal;$('revealBox').innerHTML=`系统实际选择：<b>候选 ${esc(r.selected_target_code)}</b><br>selected spatial=${fmt(r.selected_candidate_scores.spatial)}, visual=${fmt(r.selected_candidate_scores.visual)}, aggregate=${fmt(r.selected_candidate_scores.aggregate)}<br>top1=${fmt(r.top1_score)}, top2=${fmt(r.top2_score)}, margin=${fmt(r.margin)}, threshold=${fmt(r.threshold)}`;finalStarted=performance.now();}",
        "if(data.reveal){const r=data.reveal;if(r.original_action_type==='NEW'){$('revealBox').innerHTML=`系统原始动作：<b>NEW</b><br>top1=${fmt(r.top1_score)}, top2=${fmt(r.top2_score)}, margin=${fmt(r.margin)}, threshold=${fmt(r.threshold)}`;$('targetState').value='NOT_APPLICABLE';$('targetState').disabled=true}else{const s=r.selected_candidate_scores;$('revealBox').innerHTML=`系统原始动作：<b>ATTACH(候选 ${esc(r.original_target_code)})</b><br>selected spatial=${fmt(s.spatial)}, visual=${fmt(s.visual)}, aggregate=${fmt(s.aggregate)}<br>top1=${fmt(r.top1_score)}, top2=${fmt(r.top2_score)}, margin=${fmt(r.margin)}, threshold=${fmt(r.threshold)}`;$('targetState').disabled=false}finalStarted=performance.now();}",
    )
    html = replace_once(
        html,
        "if(data.label)$('derived').textContent=`${data.label.derived.derived_status} → ${data.label.derived.derived_action}`;",
        "if(data.label)$('derived').textContent=`${data.label.derived.routing_label} → ${data.label.derived.correct_action_type}`;",
    )
    html = replace_once(
        html,
        "$('blindForm').onsubmit=async e=>{e.preventDefault();const matches=[...document.querySelectorAll('input[name=match]:checked')].map(x=>x.value);try{",
        "$('blindForm').onsubmit=async e=>{e.preventDefault();const matches=[...document.querySelectorAll('input[name=match]:checked')].map(x=>x.value);const quality=$('observationQuality').value;const evidence=$('identityEvidence').value;const note=$('physicalNote').value.trim();const summary=`确认锁定盲标？提交后不可修改。\\n\\nquality: ${quality||'(未选)'}\\n同一实例候选: ${matches.join(', ')||'(未选)'}\\n身份证据: ${evidence||'(未选)'}\\n实例描述: ${note||'(空)'}\\n\\n再次检查：局部可见 ≠ GRANULARITY；同类/同材质/相邻 ≠ 同一实例。`;if(!window.confirm(summary))return;try{",
    )
    html = replace_once(
        html,
        "physical_instance_note:$('physicalNote').value,blind_review_seconds",
        "identity_evidence_status:$('identityEvidence').value,physical_instance_note:$('physicalNote').value,blind_review_seconds",
    )
    html = replace_once(
        html,
        "target_state:$('targetState').value,outside_candidate_status:$('outsideStatus').value,evidence_sufficient:$('evidenceStatus').value,confidence:$('confidence').value,notes:$('notes').value,final_review_seconds",
        "target_pre_state:$('targetState').value,full_map_status:$('outsideStatus').value,outside_matching_node_uids:$('outsideUids').value.split(',').map(x=>x.trim()).filter(Boolean),causal_note:$('causalNote').value,confidence:$('confidence').value,notes:$('notes').value,final_review_seconds",
    )
    html = replace_once(
        html,
        "已保存：${esc(r.derived.derived_status)}",
        "已保存：${esc(r.derived.routing_label)}",
    )
    return html


HTML = make_html()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("标注服务只能绑定 loopback；请用 SSH tunnel 访问")
    store = AnnotationStoreV2(args.packet_root)
    legacy_server.HTML = HTML
    legacy_server.Handler.store = store
    server = legacy_server.ThreadingHTTPServer(
        (args.host, args.port), legacy_server.Handler
    )
    print(
        json.dumps(
            {
                "status": "SERVING_V2",
                "url": f"http://{args.host}:{args.port}/",
                "packet_root": str(args.packet_root.resolve()),
                "cases": store.status()["total"],
            },
            ensure_ascii=False,
        ),
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
    raise SystemExit(main())
