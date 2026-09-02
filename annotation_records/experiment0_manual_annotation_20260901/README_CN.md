# 实验 0 标注与结果快照（2026-09-02）

## 当前有效数据

- `v2_large_room0_r1/`：正式大批量标注。179/179 页完成，其中 162 个独立事件、17 个盲重复；`labels/` 保存原始草稿与最终答案，`worklist*` 和 `manifest*` 保存选例与完整性信息。
- `v2_r2_schema_trial_room0/`：15 个 R2 身份路由复核样本及盲重复裁决。结合正式大批量标注去重后，共得到 174 个独立人工事件。
- `v2_large_room0_r1/analysis_20260902/episodes/`：合并后的人工事件、14 个身份路由错误 episode 和根/级联分析输入。
- `v2_large_room0_r1/analysis_20260902/`：质量统计、最小回放指标、完整成员分区审计。判断实验结论时应读完整成员审计，不能只看少量 endpoint probe。

## 辅助与历史数据

- `calibration_room0/`、`r1_v2_schema_trial_room0/`：用于校准标注定义和一致性，不能估计自然错误率。
- `v2_schema_trial_room0/`：旧版有缺陷的试标，保留用于追溯，但不计入 174 个正式独立事件。
- `v2_r2_candidate_pool_room0/`、`v2_r2_mixed_supplement_room0/`、`v1_to_v2_overlay/`：R2 选例、补充和旧标签迁移依据。
- `identity_routing_v2_audit_room0/`：私有自动 GT 路由审计，仅用于标注后核验。
- `corrected_gt_audit_room0/`、`corrected_association_audit/`：校正后的离线评测 sidecar 与摘要；它们不进入在线 mapper 决策。

## 当前结论边界

- 179 页盲重复在路线、可判定性、质量类型和实例身份上均为 100% 一致。
- 自然概率队列中 148 个可判定事件发现 5 个错误，点估计 3.378%；这只是 room0 抽样，不等于全场景全部事件。
- 两个假分裂案例能够恢复，但原生后续流程也会自愈，因此只构成单类可行性证据。
- `v2_r2_013` 的 CREATE 修复虽有 100% 新实体精确率，完整实例召回率仅 13.9%，属于局部修复，不可宣称整体方法成立。
- 更早的 frame138 是同类双实例混合 mask，整张 ATTACH 或 NEW 都不正确；当前正在验证“隔离后自然恢复”，尚无最终结论。

## 未纳入 Git 的产物

证据图片、逐 case 页面、服务日志、旧备份和数百 MB 的回放分支状态不进入 Git。它们仍保存在服务器原始结果目录，可由这里的 manifest、事件 UID 和脚本复现；仓库内没有密码或 API Key。
