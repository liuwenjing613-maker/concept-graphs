# 实验 0：人工校准标注复核结果（2026-09-01）

## 结论

当前状态为 **HOLD：暂不启动 room2 正式实验**。

原因不是简单的“动作一致率没有达到 90%”，而是本轮校准集的自动正负分层使用了已被后续修订明确作废的旧 Habitat semantic sidecar。旧 sidecar 存在可见视口不匹配，因此这 20 个基础案例并不是预期的 8 个错误、8 个正确和 4 个排除案例。现有人工标签显示标注者对多数可核对负例和 mask 质量判断是合理的，但尚未验证能否稳定识别真实关联错误。

## 1. 人工标注完整性

- 基础案例：20/20 完成。
- 暗重复：4/4 完成。
- 标签文件：24/24 完成，无缺行。
- 基础案例证据充分性：20/20 选择 `YES`。
- 基础案例动作：`KEEP=10`、`REASSIGN=1`、`DEFER=4`、`EXCLUDE=5`。
- 暗重复可纳入性一致：4/4（100%）。
- 暗重复精确标签一致：3/4（75%）。
- 暗重复最终动作一致：3/4（75%）。

唯一的暗重复核心分歧是同一事件的 `calibration_016` / `calibration_024`：

- 第一次：同实例候选只选 C，mapper 目标是 D，得到 `REASSIGN`；
- 重复次：同实例候选选 C+D，mapper 目标仍是 D，得到 `KEEP`；
- 两次均选择证据 `YES`、置信度 5，因此这是身份判断的真实分歧，不能当作备注或置信度差异忽略。

## 2. 自动分层存在的关键错误

本轮私有校准 worklist 的分层来自旧的 Habitat semantic sidecar。项目最新修订已经明确：该 sidecar 与 RGB/depth 的可见视口不一致，旧结果不能引用；正确 sidecar 应使用当前深度和位姿反投影到 ReplicaSSG 标注网格，并以 3 cm 门限做最近邻赋值。

直接抽查可见明显错位：

- `calibration_002` 页面当前 mask 是枕头；旧 sidecar 给出 `table`、purity=0.9285；校正 sidecar 给出 `cushion`、purity=0.9969。
- 多个页面中的椅子、枕头和灯，在旧记录中被归为 `blinds`、`rug` 或其他无关实例。
- `calibration_020` 校正后为 `ceiling`、purity=0.9986；人工却标为 `CLEAN_SINGLE_INSTANCE`，该例应按协议作为背景排除并进入裁决。

因此旧的 `AUTO_ERROR/AUTO_CORRECT/AUTO_MASK_EXCLUSION` 组名只能保留作审计，不能再作为校准真值或抽样依据。

## 3. 校正 sidecar 后的复算

已在服务器用校正 sidecar重新生成 room0 的 7,507 条 observation GT，并重新计算关联事件：

- 严格可评估 `MERGE_TO_OBJECT`：4,700 个；
- 校正自动错误：29 个；
- 错误涉及 target cluster：6 个；
- 其中 t^- 候选集中存在正确节点：4 个；不存在：25 个。

将现有 20 个基础案例与校正事件记录连接：

- 10 个案例满足严格自动评估条件；
- 这 10 个全部是校正后的正确关联，现有校准集中没有一个被严格校正 GT 确认的错误正例；
- 人工对其中 8 个给出 `KEEP`，1 个保守 `DEFER`，另 1 个就是 `016/024` 的重复分歧；
- 旧的 8 个 `AUTO_ERROR` 中，能被校正严格评估的 4 个全部变成正确关联。

mask 质量的校正审计也支持大部分人工判断：

- `017`：chair，纯净，人工标 CLEAN，合理；
- `018`：table+book mixed，人工标 MIXED，合理；
- `019`：stool，纯净，人工标 CLEAN，合理；
- `020`：ceiling 背景，人工标 CLEAN，需要纠正。

这说明人工协议并非整体失效，但当前样本只能证明“多数负例/排除例能标”，不能证明“真实错误能识别”。

## 4. 不能使用的结论

- 当前汇总中的 `1/11` root false-attach 比例不能解释为自然发生率；该队列原本就是校准抽样，而且自动分层已经失效。
- 不能据当前 20 例声称标注器具有错误召回能力，因为严格校正后没有错误正例。
- 不能为了通过门槛覆盖或修改原始暗重复标签；原始标签必须保持不变，裁决另写 overlay。

## 5. 最小修正方案

不要求重做现有 20+4 例。保留原始标签，并新增一个小型补充校准：

1. 从校正后的 29 个错误事件中选 6 个基础案例，优先每个 target cluster 取 1 个，并覆盖“已有正确候选”和“需要新建节点”两种动作；
2. 选择当前 mask 高纯、目标历史稳定、页面证据清楚的案例，避免用模糊正例测试标注者；
3. 在 6 个案例中加入 3 个随机暗重复，共新增 9 次标注；
4. 另做一次显式裁决：`016/024` 最终身份、`020` 背景排除，以及 `UNCERTAIN` 与证据 `YES/PARTIAL` 的使用规则；
5. 不机械套用小样本 90% 门槛。放行重点是：没有系统性概念误解；补充正例能被识别或有理由地 `DEFER`；3 个新增暗重复的核心动作一致；所有分歧均有裁决记录。

通过后再冻结 schema，并从空图严格在线启动 room2。room2 仍需把概率抽样队列与高召回案例挖掘队列分开，前者才可估计自然发生率。

## 6. 服务器产物

- 原始标注汇总：`/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_manual_annotation_20260901/calibration_room0/annotation_summary.json`
- 校正 observation GT：`/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_manual_annotation_20260901/corrected_gt_audit_room0/`
- 校正关联复算：`/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_manual_annotation_20260901/corrected_association_audit/`
- 校正 sidecar：`/home/chenkejun/beauty/conceptgraphs/results/experiments/mask_first_order_semantic_audit_20260831/sidecars_geometry_full/room0/`

