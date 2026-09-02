# 实验0：office0 主范围自然错误筛查总结

日期：2026-09-02

## 一句话结论

office0 完整在线 baseline 中自动初筛出 2 个“本应 NEW、却 ATTACH 到旧节点”的名义候选；检查错误发生前的精确目标版本后，2 个目标都已含当前物理实例，因此都是已有污染的后续级联，严格独立 root 候选为 0。

这说明筛查链条可以工作，也再次暴露了“混合 mask 先污染节点，之后纯 observation 被污染节点继续吸附”的机制；但 room0 和 office0 仍没有获得满足主论文口径的自然 root 正例，所以目前不能据此声称方法有效。

## 1. 本阶段使用了什么

- 场景：Replica `office0`；
- baseline：从空图开始的严格在线 run，处理帧为 `0:2000:5`，共 400 帧；
- baseline 状态：`MAP_COMPLETED_EVIDENCE_VALID`，保存 observation PCD 和完整关联证据；
- GT：使用与该轨迹逐帧几何对齐的 ReplicaSSG sidecar，共 400 帧，对齐检查通过；
- 当前 observation：3,106 条；
- 人工标签：本阶段没有把自动 GT 当人工真值，自动结果只用于挑选少量值得复核的事件。

## 2. 自动路由初筛

3,106 条关联中：

- 2,103 条具有可编译的 GT 历史；
- 1,849 条满足可靠自动 GT 条件；
- 1,831 条为 `CORRECT_ATTACH`；
- 14 条为 `CORRECT_NEW`；
- 2 条为 `SHOULD_HAVE_BEEN_NEW`，属于当前主问题的名义候选；
- 2 条为 `WRONG_NEW_FALSE_SPLIT`，不属于 false-attach 主范围；
- 其余因当前 GT 不可靠或历史不足/含混而不做自动定论。

## 3. 为什么 2 个名义候选都不是独立 root

### 候选 A：frame 87，desk-organizer → tissue-paper 节点

- 当前 observation：GT44 `desk-organizer`，GT 纯度 0.975；
- 错误前目标版本含 9 条 observation：GT28 `tissue-paper` 8 条，GT44 `desk-organizer` 1 条；
- 更早的 frame 74 已出现一个 `desk-organizer + tissue-paper` 混合 mask，并在 event 1134 被挂入该目标；
- 所以 frame 87 发生前，目标节点已经含当前 desk-organizer 的信息；frame 87 是级联，不是第一次污染。

### 候选 B：frame 258，blinds → door 节点

- 当前 observation：GT15 `blinds`，GT 纯度 0.996；
- 错误前目标版本含 102 条 observation：GT16 `door` 97 条，GT15 `blinds` 5 条；
- 目标历史里有 30 条混合 mask，29 条为两个前景实例；
- 在当前事件前，已有 29 条历史 observation 明确包含 blinds，最早可追到 frame 13 / event 322；
- 所以这是长期 `door + blinds` 污染后的再次吸附，明显不是独立 root。

## 4. 与标注结果如何配合

room0 的人工标注负责回答“当前 observation 应该属于谁、mapper 动作是否正确”；本阶段增加了另一个不可缺少的判断：错误发生前，目标节点是否干净。

只有同时满足以下条件，才能计为实验0主范围的自然 root false attach：

1. 当前 observation 是单一、身份足够清楚的物理实例；
2. mapper 实际执行 `ATTACH_EXISTING`；
3. 当前实例与目标实例不是同一物理对象；
4. 精确 `t^-` 目标至少有可核验历史且在事件前仍是单一干净实例；
5. 当前物理实例没有通过更早的纯 mask 或混合 mask 进入该目标。

这避免把同一条污染链上的多个后续错误重复计算为多个 root。

## 5. 当前实验0能说明什么、不能说明什么

可以说明：

- room0 的完整在线 baseline 已通过双跑确定性复验，当前结论不是随机漂移造成；
- room0 与 office0 都观察到混合 mask 进入节点后引起后续错误吸附的链式机制；
- “当前身份判断 + 精确事件前目标历史”的审计流程能够排除伪 root。

不能说明：

- 还没有自然、独立、目标事前干净的 root 正例；
- 因而尚不能评价 proposed repair 对主问题的成功率；
- 自动 GT 初筛不是正式人工标签，也不能给出正式自然错误率；
- 本阶段没有重跑 office0 baseline，因此没有新增可比较 timing；使用的是已完成且证据有效的严格在线 run。

## 6. 下一步

room0、office0 两个开发场景的最小筛查已经完成。下一步进入未见 Replica 场景的 pilot：先检查现有严格在线 run 是否证据完整，再用同一套低成本自动筛查找出少量主范围候选；只有候选通过精确 `t^-` 历史检查后，才交给人工标注。这样可以避免再次进行上百条低信息量全量标注。

机器可读结果位于：

- `corrected_gt_audit_office0/`：逐 observation 的校正 GT 与完整性结果；
- `identity_routing_v2_audit_office0/`：私有 GT 路由初筛；
- `auto_core_candidate_audit_office0_20260902/`：2 个候选的精确事件前历史审计。
