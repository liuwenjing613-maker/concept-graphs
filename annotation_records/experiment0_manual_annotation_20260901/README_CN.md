# 实验 0 标注与结果快照（2026-09-02）

## 当前有效数据

- `v2_large_room0_r1/`：正式大批量标注。179/179 页完成，其中 162 个独立事件、17 个盲重复；`labels/` 保存原始草稿与最终答案，`worklist*` 和 `manifest*` 保存选例与完整性信息。
- `v2_r2_schema_trial_room0/`：15 个 R2 身份路由复核样本及盲重复裁决。结合正式大批量标注去重后，共得到 174 个独立人工事件。
- `v2_large_room0_r1/analysis_20260902/episodes/`：合并后的人工事件、14 个身份路由错误 episode 和根/级联分析输入。
- `v2_large_room0_r1/analysis_20260902/`：质量统计、最小回放指标、完整成员分区审计。判断实验结论时应读完整成员审计，不能只看少量 endpoint probe。
- `v2_large_room0_r1/analysis_20260902/core_scope_audit/`：按主论文最新定义重审 14 个错误、150 个概率样本和完整流自动 root 候选；同时保存逐例表、严格分母和 Wilson 区间。
- `v2_large_room0_r1/analysis_20260902/identity_boundary_margin_generalization/`：对三个 CREATE 回放的低分差规则做跨案例审计，仅用于说明探索性阈值尚不能冻结。
- `determinism_room0_20260902/`：同一冻结配置从空图重跑 room0 的确定性证据，包括两次 manifest、strict evidence 摘要、UID 配对、最终地图比较和完整审计报告。

## 辅助与历史数据

- `calibration_room0/`、`r1_v2_schema_trial_room0/`：用于校准标注定义和一致性，不能估计自然错误率。
- `v2_schema_trial_room0/`：旧版有缺陷的试标，保留用于追溯，但不计入 174 个正式独立事件。
- `v2_r2_candidate_pool_room0/`、`v2_r2_mixed_supplement_room0/`、`v1_to_v2_overlay/`：R2 选例、补充和旧标签迁移依据。
- `identity_routing_v2_audit_room0/`：私有自动 GT 路由审计，仅用于标注后核验。
- `corrected_gt_audit_room0/`、`corrected_association_audit/`：校正后的离线评测 sidecar 与摘要；它们不进入在线 mapper 决策。

## 当前结论边界

- 179 页盲重复在路线、可判定性、质量类型和实例身份上均为 100% 一致。
- 自然概率队列中 148 个可判定事件发现 5 个事件级路由错误，点估计 3.378%；这是“clean observation 条件下的路由动作错误率”，不是 root false-attach 发生率。
- 14 个错误中有 5 个 false split；其余 9 个 ATTACH 错误的精确 `t^-` 目标均已被更早历史污染。因此现有 14 例中，符合主论文严格定义的已确认自然 root false-attach 为 0。
- 概率样本按“当前 observation 纯净、精确 `t^-` 目标因果干净、至少两条目标历史”过滤后，严格分母为 69，root 为 0，Wilson 95% 上界约 5.27%。该结果只来自 room0 开发场景，不能解释为问题不存在。
- 完整流旧自动逻辑给出 6 个 nominal root 候选；深审计后 5 个明确预污染，1 个因前一帧存在单条双实例泄漏而需人工裁决。自动 GT 只能选例，不能代替正式标签。
- 两个假分裂案例能够恢复，但原生后续流程也会自愈，因此只构成单类可行性证据。
- `v2_r2_013` 的 CREATE 修复虽有 100% 新实体精确率，完整实例召回率仅 13.9%，属于局部修复，不可宣称整体方法成立。
- 更早的 frame138 是同类双实例混合 mask，整张 ATTACH 或 NEW 都不正确。后续 oracle 过滤与 CREATE 可达到接近完整分离，但使用了校正 GT 选帧和事后阈值，只能作为旁支机制上界，不能计入主论文 false-attach 证据。
- 当前不能声称方法可行；必须先补齐自然 root 的人工确认，并在 office0 与未见场景从空图严格在线重复实验 0。
- room0 baseline 双跑确定性通过：7,507 个 association、399 帧 trace、全部规范化证据账本、400 个相似度文件及最终 72 个对象的数值状态一致。因此当前 root/cascade 修正不是随机重跑漂移造成的。

## 未纳入 Git 的产物

证据图片、逐 case 页面、服务日志、旧备份和数百 MB 的回放分支状态不进入 Git。它们仍保存在服务器原始结果目录，可由这里的 manifest、事件 UID 和脚本复现；仓库内没有密码或 API Key。
