# 实验 0：在线关联事件人工标注协议（冻结前校准版）

日期：2026-09-01  
状态：`CALIBRATION_ONLY`。先验证标注是否可靠；未见场景尚未解封，当前不能做实验 0 结论。

## 1. 这次到底标什么

唯一分析单位是一次历史关联事件：

```text
e_t = 当前 observation z_t + 系统在 t^- 时刻选择的 persistent node o_t
```

主问题只研究：一个可用、近似单一真实实例的 observation 被 `MERGE_TO_OBJECT` 到错误节点。输出动作只有：

- `KEEP`：系统所选节点就是同一物理实例；
- `REASSIGN`：系统所选节点不是同一实例，但 t^- 已有一个或多个合法节点；
- `NEW`：系统所选节点不是同一实例，而且 t^- 尚无该实例节点；
- `DEFER/EXCLUDE`：证据不足，或根因属于 mixed mask、背景残片、重复 proposal、动态/pose/depth、粒度歧义等其他问题。

本实验不把 false split、mask repair、语义改名、关系错误或动态物体混入 false-attach 主指标。

## 2. 为什么采用“在线捕获、允许延迟裁决”

强迫人在事件发生的那一帧立即给最终答案并不可靠：同类相邻物体、遮挡和局部视图常需多视角才能确认。正确做法是：

1. mapper 从 `frame=0` 空图开始严格在线运行；
2. 每个事件发生时立即冻结当前 observation、全部候选、所选节点的 `t^-` 版本、分数矩阵和 artifact 哈希；
3. sidecar 等到下一帧出现，确认上一帧写完，再生成不可变标注包；最后一帧以 evidence manifest 完成为准；
4. 人可在建图同时处理已闭合的包，也可稍后处理；任何未来图像只能帮助裁决世界真值，不能改变包内的事件时状态；
5. 标注过程不修改 canonical map，不把人工答案反馈给 mapper。

因此“在线”约束的是数据捕获和状态边界，不是逼标注者在信息不足时猜答案。

## 3. 两阶段盲标

### 阶段 A：不显示 mapper 选择和分数

页面展示：

- 当前 RGB、processed-mask 全图和裁剪；
- 候选 A–E 各自严格来自 `t^-` 的历史 observation；
- 当前 observation 与每个候选历史的三视图 3D 对照；
- 不显示 object UID、候选分数、排名，也不标出系统最后选了谁。

标注者先回答：

1. 当前 processed mask 是否近似一个真实物理实例；
2. A–E 中哪些与当前 observation 是同一物理实例；同一实例已有多个 split 节点时可以多选；
3. 若看不清，必须选 `UNCERTAIN`，不能用检测类别或分数猜。

### 阶段 B：揭示 mapper 选择

阶段 A 保存后才显示系统选中的候选与当时分数。标注者再回答：

1. 被选节点在关联前是否干净、已污染或无法判断；
2. 若 A–E 都不匹配，完整的 `t^-` 地图中是否另有同一实例节点；
3. 证据是否足够、置信度和缺失证据。

程序根据两阶段答案自动推导 `KEEP/REASSIGN/NEW/DEFER`；标注者不能直接输入想要的动作。

## 4. 物理实例与 mask 质量定义

`CLEAN_SINGLE_INSTANCE`：mask 主体是一个持续物理实例，少量边界泄漏不影响身份。  
`BORDERLINE_SINGLE_INSTANCE`：仍能确认一个主实例，但边界泄漏或局部遮挡明显；只进入敏感性集合。  
`MIXED_MULTIPLE_INSTANCES`：两个或更多可分物体共同构成 mask。  
`BACKGROUND_OR_FRAGMENT`：背景薄片、相减残片或不存在的实体。  
`DUPLICATE_PROPOSAL_SAME_FRAME`：同帧重复 proposal，本问题不是跨时关联。  
`DYNAMIC_POSE_DEPTH_ERROR`：动态变化、pose 或深度投影使身份比较不成立。  
`GRANULARITY_AMBIGUOUS`：part-whole 边界无法按当前 object 粒度可靠决定。  
`INSUFFICIENT`：图片/3D/历史缺失或完全看不清。

物理实例按场景中持续存在、机器人可再次观察的 object 区分；语义名称不是实例。两个相同椅子必须不同，椅子在不同视角仍是同一实例。书和桌子、枕头和沙发是不同实例。无法稳定决定的部件关系必须标为粒度歧义。

## 5. root error 与 cascade

只有下列条件同时成立，事件才是 root false attach：

- 当前 observation 为 `CLEAN`（主集合）或 `BORDERLINE`（扩展集合）；
- 系统所选节点的 `t^-` 状态可确认是单一实例；
- 当前 observation 与所选节点不是同一物理实例；
- 这是该节点/lineage 从干净状态首次被不同实例污染。

已经污染节点上的后续错误记为 `CASCADE_OR_PRECONTAMINATED`，不再当独立 root 正例。对象 merge 导致的污染传播归入原 episode。最终统计同时报告原始错误事件、root episode 和每个 episode 的 cascade 大小，三者不能混算。

## 6. 自动 GT 的角色

Replica 官方 instance GT 可在独立 evaluator 中：

- 给所有事件生成可扩展的候选身份与 purity；
- 检查人工标注一致性；
- 帮助定位正确节点是否在原始 top-K 中；
- 自动追踪后续可再观测视角。

GT、自动错误标签和抽样 strata 在标注页面全部隐藏。人工判断负责确认 processed mask 是否是可行动的真实物体、GT 粒度是否合理，以及正确修复动作是否真的明确。若二者冲突，进入专家裁决，不以自动 GT 强行覆盖人类判断。

## 7. 抽样与工作量

### 7.1 工具校准，不做论文统计

先在已经使用过的 room0 生成 20 个基础包：8 个自动 GT 疑似错关联、8 个自动 GT 正确关联、4 个 mixed/低纯度排除例；另插入 4 个证据完全相同但 case ID 不同的暗重复包。

校准只检查：

- 页面证据是否足够；
- 选项定义是否会被稳定理解；
- 暗重复的一致性；
- 每例用时和主要证据缺口。

若证据不足率超过 20%，或 eligible/error 与动作的暗重复一致率低于 90%，先修 packet/定义，不进入未见场景。

### 7.2 最小正式 pilot

校准通过后，先只解封 `room2`：

- 从空图运行完整 400 个处理帧，固定 stride=5；
- `evidence_top_k=5`，保存 observation PCD、processed mask、完整 similarity matrix 和 object version；
- 在线按 event UID 哈希做与风险分数无关的 Bernoulli prevalence 抽样，目标约 150–200 个可评 MERGE 事件；
- 另建高召回 case-harvest 队列用于找正例，但不得用它报告自然发生率；
- 先看 evidence sufficiency、重复一致性和 root error 的数量级，再决定是否扩展 office2–4。

不采用“30 个 root error 才算 GO”的硬门。若 room2 只有少量正例，结论只能是单场景可行性/发生率区间较宽，不能据此声称问题不存在或跨场景普遍。

## 8. 必须保存的字段

每个 case 至少保存：run/scene/event/obs UID，source frame，事件时 mapper 最新闭合帧，目标 `t^-` version，完整候选版本和 top-K，所有展示资产 SHA-256，盲标答案、揭示后答案、自动推导动作、置信度、两阶段用时、标注版本和重复组。

统一时间线：`s=错误关联帧`；`d=人工首次可确认/提交帧或 wall time`；实验 0 不调用 VLM，因此 `h=null,c=null`。同时记录提交时 canonical mapper 的最新闭合帧，避免把异步进度混成事件时状态。

## 9. 实验 0 最终统计（标注完成后）

- eligible MERGE 分母、人工排除与 DEFER 原因；
- root false-attach 数量、按概率权重估计的发生率和场景级区间；
- `KEEP/REASSIGN/NEW`；若正确节点不在展示 top-K，单列；
- same-class/different-class/null-label 仅作描述，不作为人工实例判断依据；
- root 后两个相关实例各自的未来独立视角数；
- cascade size、下游污染和最终是否仍有害；
- 标注一致性、证据充分率、用时和全部限制。

校准、prevalence probability sample、case harvest 和自动 GT 全量结果必须分开报告。

