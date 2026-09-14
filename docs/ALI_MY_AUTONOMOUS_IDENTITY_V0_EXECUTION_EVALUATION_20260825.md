# ALI-MY Autonomous Identity Repair V0：服务器实现与评估总结

日期：2026-08-25
执行位置：远程服务器 `server-3048-out`，未在本地运行代码或实验
基线：`publish/ali-my-v2-10h-20260824 @ 79a3d5f`
实验分支：`exp/ali-my-autonomous-repair-v0-20260825`

## 1. 结论先行

本阶段已经实现了一个可运行、可审计、无人工答案泄漏的 identity 自主闭环：

```text
机器 checker/worklist 发现可疑 incident
→ 依据 executor 能力枚举有限候选（始终包含 NO-OP）
→ 对每个候选执行真实 sparse causal replay
→ 按实体 ID 无关的成员分区去重
→ 使用隔离的未来证据比较匿名结果
→ 单一校准阈值决定 COMMIT 或 DEFER
```

但目前只能称为 **Autonomous Identity Shadow V0**，不能称为已经完成生产级自主修复：

- 已做到自主发现、候选生成、真实执行、结果检查和安全延期；
- 已在 1 个开发 false merge 上正确推荐修复；
- 新增机器盲选案例中，1 个真实错误与 1 个真实干净样本都被安全延期；
- 干净样本未被误修，但真实错误也未修复；
- 校准 artifact 明确标记为未就绪，因此生产自动提交数为 0。

最重要的新结论是：当前主要瓶颈已不再是“能否生成修复”，而是“是否存在能区分反事实结果的独立观测”。对后续成员分配不发生分叉的案例，增加相同未来帧的普通 crop/wide context 没有信息增益。

## 2. 两份参考方案的合并解释

本阶段同时依据以下两份文档，而不是只执行其中一份：

1. `ALI_MY_V2_TO_ICRA_FINAL_FRAMEWORK_AND_EXPERIMENT_PLAN_20260825.md`
2. `ALI_MY_V2_TO_ICRA_PLAN_REVISED_MINIMAL_RULES_VLM_ROLES_20260825 (1).md`

采用的合并原则是：

- 第一份文档给出完整的 Trace → Hypothesize → Replay → Verify 框架、实验层次和论文主张边界；
- 第二份文档对“最少手工规则”和 VLM 的职责进行收紧；
- 第二份不是替代第一份，而是约束实现方式：VLM 可提出、排序、批判和请求证据，但不得直接修改图，不得把自报 confidence 当成提交阈值；
- 硬规则只保留确定性 invariant、schema、provenance 和 evidence-isolation 条件；
- 噪声证据保持连续量，最终只允许一个经过冻结校准的语义 commit threshold；
- NO-OP 必须与修复候选一起比较，证据不足必须 fail closed。

## 3. 实现范围

### 3.1 有限动作空间与执行语义

identity 路径现在从真实 executor capability 枚举动作：

- `NO_OP`
- `SAME_INSTANCE`
- `SEPARATE_MEMBER_GROUPS`

其中：

- 原生 `CREATE` 的反事实合并编译为精确绑定的 `ASSIGN_OBSERVATION`；
- 原生 `CREATE` 的保持分离复用原生创建 identity；
- 原生 `ASSOCIATE` 的真正反事实分离编译为 `CREATE_INSTANCE`；
- 新 identity 使用 `revision-lineage:<obs_uid>` 确定性生成；
- `separate_from_identity_uids` 显式绑定被错误关联的目标身份；
- executor 仍负责根据 `obs_uid` 确定性生成新 entity UUID。

这次实际实验暴露并修复了一个重要缺口：此前原生 `ASSOCIATE` 只枚举 `SAME_INSTANCE`，执行后通常与 NO-OP 得到相同分区。现在会同时枚举真正的 `CREATE_INSTANCE + identity boundary` 反事实。

### 3.2 结果去重与 identity/provenance 正式化

候选结果不使用任意 entity UUID 比较，而使用成员 observation 集合的规范分区：

- 每个 entity 的成员 observation 排序；
- 空组移除；
- 所有组排序；
- 对规范分区计算 hash；
- 与 NO-OP 分区相同的候选不进入结果判别；
- 多个动作落到同一分区时只保留一个行为结果。

如果没有与 NO-OP 不同的可执行分区，系统输出：

`NO_DISTINCT_EXECUTABLE_REPAIR → SHADOW_NO_DISTINCT_REPAIR → DEFER`

而不是伪造 critic 请求。

### 3.3 提议证据与验证证据隔离

每个 incident 的 evidence split 满足：

- 提议证据只用于构造候选上下文；
- 验证证据来自 anchor 之后至少 3 帧；
- proposal 与 verification 的 observation UID 交集必须为空；
- proposal 与 verification 的内容 SHA-256 交集必须为空；
- critic 看不到 proposal evidence；
- 两个顺序互换请求使用完全相同的验证图像集合；
- critic 只看到匿名 `STATE_A/STATE_B`，看不到动作名和 gold；
- evidence 从所有 replay state 的 union 中选取，再按共同细化分区平衡采样，避免某个候选轨迹更长而占据更多图像。

### 3.4 VLM 的受限角色

VLM 仅担任 outcome critic：

- 比较匿名执行结果；
- 引用冻结 evidence ID；
- 主动寻找反证；
- 可以输出 `DEFER` 和需要的证据；
- 两个请求交换 state 顺序，只有两次都支持同一物理分区才算完整支持；
- confidence 只保留为诊断字段，不参与阈值比较；
- VLM 不生成最终约束、不读取 human label、不直接 commit。

### 3.5 单一选择性提交

生产判定只使用一个冻结 calibration artifact：

- 连续特征包括 primary advantage 与 pairwise critic preference；
- primary mapper likelihood 明确标注为非独立真值；
- 只比较一次 `benefit_probability < commit_threshold`；
- 当前 calibration 为 `ready_for_automatic_commit: false`；
- 因此本阶段所有 production 决策必须为 `DEFER`。

## 4. 自我修正记录

### 4.1 原生 ASSOCIATE 的“伪修复”问题

初次机器 intake 重放中，两条 `ASSOCIATE` 案例只生成 `SAME_INSTANCE`。动作名虽然不同，成员分区却与 NO-OP 相同。

采取的修正：

1. 先加入实体 ID 无关的分区去重；
2. 对无不同分区的案例 fail closed，避免无意义 API 调用；
3. 继续追查后确认不是案例问题，而是有限动作语义不完整；
4. 为原生 `ASSOCIATE` 增加 `SEPARATE_MEMBER_GROUPS`；
5. 编译为确定性 `CREATE_INSTANCE` 并建立持久 identity boundary；
6. 用 office0 和 room0 两个真实全局重放验证，均得到 2 个 unique partitions，其中 1 个是不同于 NO-OP 的修复结果。

### 4.2 不用提示词强迫 outlet 案例匹配人工答案

开发案例 `identity_dev_002` 的人工答案是 `CREATE_INSTANCE`，但 outcome critic 持续偏向 NO-OP。

深入审计发现：

- source observation 原始 mask 点约 450，处理后约 94，损失约 79%；
- pre-DBSCAN 存在两个空间簇，中心相距约 1.299 m；
- anchor outlet 中心与其中一部分仅相距约 0.22 m；
- 后续 lineage 观测跨越相邻 fixture，不能作为干净物体轨迹。

因此没有通过修改 ontology prompt 强迫模型选择人工答案，而是将该案例归因到 observation geometry / `PARTITION_OBSERVATION` 设计缺口。当前自动 compiler 对 `PARTITION_OBSERVATION` 继续明确延期，避免在 point assignment 尚未 hash-bound、pre-association executor 尚未集成时产生不可审计修复。

### 4.3 证据调度停止条件

两个新机器案例的 critic 都要求更多上下文。但结构审计表明，NO-OP 和修复态对全部 8 个 held-out observation 的 owner assignment 完全相同。

调度器因此返回：

`NO_SCHEDULABLE_ADDITIONAL_EVIDENCE`

两个案例均：

- 追加 case 数：0
- 追加 critic request 数：0
- unschedulable case 数：1

这阻止了“证据不足 → 无限增加同类图像 → 重复调用 API”的错误循环。

## 5. 实验结果

### 5.1 三个已确认 identity 错误（开发集）

使用 3 个已有 human-confirmed identity 案例，仅在事后 evaluator 中读取人工约束。

| 指标 | 结果 |
|---|---:|
| 案例数 | 3 |
| 有限候选包含正确执行目标 | 3/3 |
| Shadow 推荐修复 | 1/3 |
| 正确 Shadow 修复 | 1 |
| 错误 Shadow 修复 | 0 |
| 推荐条件下 precision | 1.00 |
| 已确认错误 repair recall | 1/3 |
| Production commit | 0 |
| Production defer | 3 |

逐例结论：

- `identity_dev_001`：office false split。增加多尺度上下文和通用组合物体 ontology 后，两次顺序互换结果不一致，最终 `SHADOW_INCONCLUSIVE`；
- `identity_dev_002`：room outlet false merge。现有 identity replay 证据指向 observation partition/geometry 问题，保持 `DEFER`；
- `identity_dev_003`：room book/coffee-table false merge。双顺序均支持 `CREATE_INSTANCE`，正确 Shadow 推荐修复。

### 5.2 两个干净 NO-OP 控制

| 指标 | 结果 |
|---|---:|
| 干净案例数 | 2 |
| 严格 NO-OP preferred | 1 |
| Inconclusive 但未推荐修复 | 1 |
| Shadow repair false positive | 0 |
| Shadow no-repair rate | 1.00 |
| Production commit | 0 |

更新后的通用 ontology 没有在这两个控制上引入修复型 false positive。

### 5.3 两个新机器 intake 案例

选择过程只读取 machine `case_selection.json`、provenance、binding 完整性、未来可见性和机器 review score；排除了 3 个开发 incident 与 2 个干净控制 incident。运行时 manifest 不包含 human label。

#### Replay 与证据协议

| 项目 | office0 | room0 |
|---|---:|---:|
| 原生动作 | ASSOCIATE | ASSOCIATE |
| Feasible actions | 3 | 3 |
| Finite constraints | 2 | 2 |
| Unique executed partitions | 2 | 2 |
| Distinct repair partitions | 1 | 1 |
| NO-OP-equivalent actions | 1 | 1 |
| Proposal images | 3 | 3 |
| Verification images | 8 | 8 |
| Proposal/verification obs overlap | 0 | 0 |
| Proposal/verification hash overlap | 0 | 0 |
| Order-swapped critic requests | 2 | 2 |

#### 冻结 critic 结果

- office0：两个顺序均 `DEFER`；
- room0：两个顺序均 `DEFER`；
- 两例的 pairwise preference 均为 0；
- primary mapper likelihood 对修复均没有独立支持：
  - office advantage：约 `-4.27e-05`
  - room advantage：约 `-1.62e-03`
- scheduler 对两例都判定普通 context 无可调度信息增益。

#### 事后人工标签评估

运行时决策全部冻结后才读取标签：

| 指标 | 结果 |
|---|---:|
| 机器 intake 案例 | 2 |
| 已有人工标签 | 2 |
| 真实错误 | 1 |
| 真实干净 | 1 |
| Shadow repair recommendation | 0 |
| Shadow inconclusive | 2 |
| Shadow repair false positive | 0 |
| 正确修复真实错误 | 0 |
| 干净样本被 production abstention 保护 | 1/1 |
| 真实错误仍未修复 | 1/1 |
| Production commit | 0 |
| Unsafe production commit | 0 |

标签详情：

- office blind case `identity_machine_a890b680c8e921372fec`
  - source incident：`incident_999b3ff28e75f169ec3c`
  - 事后标签：`WRONG / SEMANTIC_IDENTITY_ERROR`
  - 结果：`SHADOW_INCONCLUSIVE / DEFER`
- room blind case `identity_machine_23aa8531e0c832a7e359`
  - source incident：`incident_49fa635065712e5f560d`
  - 事后标签：`CORRECT / NOT_APPLICABLE`
  - 结果：`SHADOW_INCONCLUSIVE / DEFER`

这组结果说明 fail-closed 行为有效，但 repair recall 仍不足，不能用“零误提交”替代修复能力。

## 6. Timing instrumentation

| 阶段 | office0 | room0 |
|---|---:|---:|
| Freeze protocol wall time | 121.37 s | 173.09 s |
| Case total | 112.01 s | 157.57 s |
| Snapshot + closure | 78.27 s | 18.06 s |
| NO-OP replay | 11.61 s | 44.85 s |
| Candidate replay 1 | 11.04 s | 46.86 s |
| Candidate replay 2 | 10.71 s | 46.84 s |
| Shared context build | 9.35 s | 15.52 s |
| 双路 VLM wall time | 9.89 s | 13.95 s |

当前主要计算开销不是 VLM，而是 closure/snapshot 和重复 replay。后续优化应优先做 common-prefix snapshot 复用、原生动作等价候选的 preflight 去重，以及只从约束事件处分支重放。

## 7. 测试与审计

最终服务器回归：

- 全部 revision 测试：`145 passed, 1 skipped`
- 自主 identity 聚焦测试：`43 passed`
- 额外 no-distinct orchestration 回归：`2 passed`
- Python compileall：通过
- `git diff --check`：通过
- 规则复杂度审计：通过

规则审计结果：

- production 语义 commit threshold 数：1
- legacy 多 gate stack 不在 production path：通过
- oracle 字段不在 production source：通过
- VLM self-confidence 不与 threshold 比较：通过

## 8. 当前框架状态判断

| 层次 | 状态 | 结论 |
|---|---|---|
| Machine incident discovery | 已实现 | 可从 checker/worklist 自动选入 |
| Finite candidate recall | 已实现于当前 identity action family | 3 个开发错误为 3/3 |
| Exact identity/provenance binding | 已实现 | 约束绑定 observation、event、sequence、identity boundary |
| Counterfactual replay | 已实现 | office0/room0 均执行出不同分区 |
| Oracle-free evidence isolation | 已实现 | hash 与 observation 双重隔离 |
| Outcome selection | 部分实现 | 可正确推荐 1 个开发修复，也可安全延期 |
| Counterfactual observability | 未解决 | 两个 machine intake 案例未来 owner assignment 不分叉 |
| Calibrated automatic commit | 未完成 | calibration 明确未就绪 |
| PARTITION_OBSERVATION executor | 未完成 | 当前只 fail closed |
| Production autonomous repair | 未达到 | 本阶段 0 commit |

## 9. 下一步最优路线

### P0：先实现 Counterfactual Observability Contract

不应继续简单增加 crop 或 wide frame。应在调用 VLM 前计算“反事实是否可观测”：

1. 对 NO-OP 与每个候选计算成员分区 symmetric difference；
2. 找到约束事件之后第一个 owner assignment 不同的 observation；
3. 只把真正分叉的 observation 放入 verification；
4. 如果直到序列结束都没有分叉，标记 `COUNTERFACTUAL_UNOBSERVABLE_FROM_ASSOCIATION_TRACE`；
5. 对该状态不调用 VLM，直接安全延期并记录所缺少的观测类型。

然后增加一种独立证据，而不是复用 mapper association：

`STATE_DIFFERENCE_REPROJECTION`

建议契约：

- 输入绑定 RGB-D、camera pose、source point hash 和 candidate state hash；
- 只投影两个状态 symmetric-difference 的 3D support；
- 输出每个未来帧的 per-state reprojection mask、可见点比例、遮挡关系和边界残差；
- 不读取 native match score、human label 或 endpoint owner；
- 两个状态使用相同帧与相同渲染参数；
- artifact 全部 hash-frozen；
- 若可见 support 仍不足，则判定物理不可识别，而不是让 VLM猜测。

第一轮只验证：

- 当前 office 真实错误；
- 当前 room 干净样本。

目标不是立刻 commit，而是证明新 evidence 能让前者与后者产生不同 outcome preference，同时不依赖人工答案调参。

### P0：冻结单阈值校准协议

当前 3 个错误 + 2 个干净控制远不足以拟合 production threshold。

建议：

1. 使用现有 40 个 human-confirmed errors 构造 development/calibration pool；
2. 为每个错误按 scene、stage、native action 匹配 clean controls；
3. group split 必须按 scene/incident family，禁止相邻 finding 跨 train/holdout；
4. 先冻结 feature schema，再拟合一个 benefit probability；
5. 目标 harm rate 与 commit threshold 在 holdout 之前冻结；
6. fresh holdout 只报告 selective precision、coverage、repair recall 和 harm，不再改阈值。

### P1：实现受限的 PARTITION_OBSERVATION

从 outlet 案例出发，先做最小可审计版本：

- contract 绑定 raw mask hash、processed point hash、pre-voxel point indices 和相机参数；
- point assignment 在 association 之前执行；
- 每个 partition 必须非空、互斥、并集等于绑定输入；
- replay 记录 split 前后点数、cluster geometry 和 provenance；
- 不允许 VLM直接输出 point indices；
- VLM最多提出“需要 partition”及区域语义，实际 point assignment 由确定性几何程序生成；
- 第一阶段只做 1 个真实错误 + 1 个干净负例，不全量跑。

### P1：Relation gold 与自动 constraint generation

identity observability 稳定后，再接 relation：

- 小规模冻结 relation gold，优先 support / inside / attached / adjacent 中最容易混淆的对；
- proposal 与 outcome evidence 继续隔离；
- VLM只提出 relation hypothesis 和证据请求；
- deterministic relation invariant 才能作为 hard veto；
- 自动 constraint generation 必须由 executor capability 决定动作空间，不能从 human error type 路由；
- 每个新 action family先通过 no-op、合法动作负例和 local/global parity。

## 10. 主要代码与 artifact

核心模块：

- `conceptgraph/revision/capabilities.py`
- `conceptgraph/revision/auto_constraints.py`
- `conceptgraph/revision/autonomous_identity.py`
- `conceptgraph/revision/evidence_split.py`
- `conceptgraph/revision/candidate_verifier.py`
- `conceptgraph/revision/shadow_critic.py`
- `conceptgraph/revision/selective_commit.py`

主要命令：

- `scripts/build_revision_identity_machine_intake.py`
- `scripts/freeze_revision_identity_selective_v0.py`
- `scripts/run_revision_shadow_critic_requests.py`
- `scripts/schedule_revision_identity_context_evidence.py`
- `scripts/finalize_revision_identity_selective_v0.py`
- `scripts/evaluate_revision_identity_machine_intake_posthoc.py`
- `scripts/audit_revision_rule_complexity.py`

实验根目录：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_autonomous_identity_v0_20260825`

关键结果：

- `posthoc_development_evaluation_multiscale_ontology.json`
- `clean_noop_ontology_posthoc_evaluation.json`
- `machine_intake_office_separate_freeze/freeze_protocol.json`
- `machine_intake_room_separate_freeze/freeze_protocol.json`
- `machine_intake_office_separate_final_decision.json`
- `machine_intake_room_separate_final_decision.json`
- `machine_intake_separate_posthoc_evaluation.json`
- `rule_complexity_audit_final_v2.json`

## 11. 最终主张边界

本阶段可以严谨支持：

> 系统能够从机器发现的 identity incident 出发，在不读取人工答案的条件下，枚举 executor-grounded 的有限反事实动作，执行 identity/provenance 一致的 sparse replay，用隔离的未来证据进行匿名结果审查，并在证据或校准不足时安全延期。

本阶段不能支持：

- 已实现可靠的全自动生产修复；
- 已达到统计显著的 repair recall；
- VLM outcome critic 已在所有 identity 类型上泛化；
- `PARTITION_OBSERVATION` 已可自动执行；
- 0 unsafe commit 等价于方法有效，因为当前 commit coverage 为 0。

下一阶段的首要问题不是继续扩写 prompt，而是建立真正能观察 counterfactual state difference 的验证证据。
