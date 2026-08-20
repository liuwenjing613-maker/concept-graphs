# `ali-my` 下一步最优路线：从“证据与查错器已搭好”转向“有效性决策门”

> 评审对象：`liuwenjing613-maker/concept-graphs` 的 `ali-my` 分支
> 代码快照：分支头 `66c109dc042fdf21936e933aa80f4e8307ddd4d6`，相对 `ali-dev@72f5962` 前进 4 个提交
> 评审时间：2026-08-20
> 本文只规划**紧接着的一步**，不展开完整自动纠错、回滚和最终论文系统。

> **执行状态更新（2026-08-20）**：本文原定的建图、证据门和双场景正式审计均已完成。逐条 finding 的复杂问卷在 **0/160** 时停止；第一次 trigger-incident 去重又被实测发现 147/160 仍重复同一 final-owner set，同样在 0/160 时停用。当前正式流程是 **final endpoint census v2.1**：5587 条 findings 归并为 98 个不同 final-object endpoints，1 个证据阻断，剩余 room0 69 + office0 28 = **97 个不同 final objects 全部纳入**，每个 object 只判断一次。用户只选 `CORRECT / WRONG / UNCLEAR` 及 WRONG 的主错误类型。本页后文保留为历史计划和设计依据，**不要再按其中的 finding 级 160 例、32 例 R2、人工猜根因/修复步骤操作**。当前入口与逐项解释以 [`ALI_MY_EVIDENCE_AUDIT_METHOD_GUIDE.md`](./ALI_MY_EVIDENCE_AUDIT_METHOD_GUIDE.md) 第 34 节为准。

---

## 0. 一句话结论

**不要继续做大规模逻辑重构，也不要直接进入自动回滚。最优下一步是用 1 到 2 天完成一次“查错器有效性决策门”：先修 2 个会污染实验结论的小型 P0 问题，再用固定阈值、双队列抽样和人工标注，回答查错器是否真的能发现“正确、因果定位合理、且值得修”的建图错误。**

当前代码没有暴露出必须推倒重来的致命架构错误。真正危险的不是“代码不够优雅”，而是继续增加规则后，得到几千条候选，却仍然不知道其中多少是真的、多少会伤害最终地图、多少值得调用 VLM 或触发回溯。人类很擅长把庞大的输出量误认成进展，机器也没必要陪着一起犯这个毛病。

---

# 1. 我对当前进度的判断

## 1.1 已经完成得比较扎实的部分

### A. 统一证据链已经基本形成闭环

当前 `EvidenceRecorder` 已覆盖：

- run、代码版本、配置、模型和 prompt；
- frame、RGB/depth、位姿和内参；
- raw/kept/rejected observation；
- 原始与处理后 mask、过滤 gate、深度统计、DBSCAN 前后统计；
- observation PCD、图像特征与文本特征引用；
- 每帧完整 similarity matrix、Top-K 候选、阈值、margin 和最终关联决策；
- object version、mapping event、merge candidate 与接受/拒绝原因；
- VLM 请求、响应、图像哈希、耗时和解析状态；
- final membership、evidence summary 和完整性审计结果。

这已经不只是“日志更多了”，而是形成了从 observation 到 association、fusion、object identity 的可追溯账本。对于后续做反事实重放、回滚或 VLM 复核，这是正确地基。

### B. 查错器的分层思路是对的

`layered_audit.py` 已把问题拆成：

1. system integrity；
2. detection；
3. segmentation；
4. projection geometry；
5. association；
6. fusion；
7. object identity；
8. caption；
9. relation。

目前前 7 层已实现，caption/relation 暂未启用。这个取舍是正确的，因为你现在的核心论文问题仍是多视角实例关联与融合错误，不该为了“看起来完整”把 caption、edge、VLM 全塞进来，然后得到一台结构宏伟但没有任何一项被验证的机器。

### C. “事实、假设、否决条件、缺失证据”被明确区分

Finding 中已经分开保存：

- `proven_facts`
- `hypotheses`
- `vetoes`
- `missing_evidence`
- `certainty`
- `route`

并且默认 `repair_allowed=false`。这一点非常重要。规则只能把案例筛出来，不能把启发式阈值包装成真理。

### D. 最新的双队列抽样方向是科学的

当前 `v1.1.0` 已加入：

- `calibration_random`
- `diagnostic_priority`
- selection probability
- sampling weight
- checker 配额
- entity cap
- deterministic seed

这比只看 Top-K 高分样本强得多：

- 随机校准队列用于估计真实 precision；
- 高优先级队列用于快速发现典型根因、检验上限；
- 两者不能混在一起报告一个“准确率”。

这一步明显是在从“查错脚本”转向“可以写进论文的方法评估”。

---

## 1.2 当前已经有的实际运行证据

仓库内 `room0` 全量审计产物显示：

| 项目 | 当前仓库产物 |
|---|---:|
| Evidence Gate | PASS |
| 地图被审计器修改 | 否 |
| Findings | 2455 |
| Root-cause candidates | 574 |
| Likely mapping conflicts | 502 |
| Ambiguous mapping risks | 1944 |
| Evidence packets | 200 |
| 审计耗时 | 约 206 s |

2 帧 evidence smoke 还验证了：

- 70 个 raw observations；
- 47 个 kept observations；
- 最终 19 个对象；
- missing reference 为 0；
- logging error 为 0；
- evidence 开关下 canonical object/edge 输出和几何、颜色、bbox、CLIP 数组一致。

这些结果足以说明：

> **证据系统能跑，查错器能产生可复核案例，且目前没有看到审计器直接修改地图的证据。**

但它们还不能说明：

> **查错器有效。**

原因很简单：2455 是候选数量，不是错误数量；574 是启发式根因聚合，不是经过反事实证明的真实因果链；200 个 packet 已生成，但仓库中还没有对应人工标签和 precision。

---

## 1.3 当前成熟度判断

| 模块 | 判断 | 是否继续扩展 |
|---|---|---|
| 统一证据字段覆盖 | 基本够用 | 暂停加字段 |
| 证据完整性门禁 | 已可用，但有一个 P0 细节 | 小修 |
| 分层筛查规则 | 已达到可验证 MVP | 暂停加规则 |
| Evidence packet | 已具备人工复核条件 | 直接使用 |
| 抽样机制 | 思路正确，仍受上游截断影响 | 小修 |
| 阈值可靠性 | 未校准 | 必须验证 |
| Finding precision | 未知 | 下一步核心 |
| 根因定位准确性 | 未知 | 下一步核心 |
| Findings 的下游危害 | 未知 | 下一步核心 |
| 自动修复/回滚 | 尚不应进入 | 暂停 |

---

# 2. 是否存在“致命错误”

## 2.1 总判断

**未发现需要推倒重来的致命架构错误。**

但存在两个会直接污染下一轮实验结论的 P0 问题，以及两个必须补齐的实验门禁。它们都可以小范围修复，不值得借机重构五千行代码。

---

## 2.2 P0-1：similarity shape 异常时使用 `np.empty`

当前 `record_associations()` 中，如果 spatial、visual 或 aggregate matrix 的 shape 不符合预期，会执行类似：

```python
if aggregate.shape != expected_shape:
    aggregate = np.empty(expected_shape, dtype=np.float32)
```

`np.empty` 不是“空值”，而是未初始化内存。后续代码仍会：

- 排序；
- 生成 Top-K；
- 计算 top1/top2/margin；
- 写入 association evidence。

一旦 shape 异常，证据账本可能生成看似正常、实际随机的候选分数。它不会直接改变原始 mapping，但会污染查错结果和人工标注。

### 必须修改为

```python
def validate_similarity_matrix(name, value, expected_shape):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != expected_shape:
        return (
            np.full(expected_shape, np.nan, dtype=np.float32),
            {
                "valid": False,
                "error": "SHAPE_MISMATCH",
                "name": name,
                "actual_shape": list(arr.shape),
                "expected_shape": list(expected_shape),
            },
        )
    return arr, {"valid": True}
```

同时做到：

1. `associations.jsonl` 写入 `similarity_evidence_valid=false`；
2. 不生成 Top-K、margin 和候选排名；
3. Evidence Gate 将该 frame 标记为 FAIL；
4. semantic checker 不得继续消费该 frame；
5. formal run 使用 `evidence_mode=strict`。

这只是一处局部安全修复，不需要改整体设计。

---

## 2.3 P0-2：每条规则 500 条上限会让“随机校准样本”仍然有顺序偏差

当前全量结果中：

- `DET-001 = 500`
- `SEG-004 = 500`
- `ASSOC-003 = 500`
- `ASSOC-004 = 500`

它们都碰到了 `max_findings_per_rule=500`。

当前 v1.1 的双队列抽样会从**已输出 findings**中随机抽样。但是，如果某条规则真实产生了 1500 个候选，代码只保留执行顺序最早的 500 个，那么后续的 `calibration_random` 只是“在前 500 个中随机”，并不是“在全部 1500 个中随机”。

这会破坏 selection probability 和 sampling weight 的统计含义。

### 最快修法

本轮验证不必马上实现复杂 reservoir sampler。采用两步即可：

1. 为 validation config 把 `max_findings_per_rule` 提高到 5000 或 10000；
2. 增加：
   - `attempted_count`
   - `emitted_count`
   - `suppressed_count`
   - `population_censored`
3. 只要任一规则 `suppressed_count > 0`，本轮 calibration 结果不得进入论文结论，自动 FAIL。

建议：

```yaml
limits:
  max_findings_per_rule: 10000
  fail_if_population_censored: true
```

后续如果内存或磁盘真的成为瓶颈，再实现：

- calibration 的分层 reservoir sampling；
- diagnostic 的 bounded top-K heap；
- 全量只保留 population count。

现在不要为了一个尚未发生的规模问题，先设计一座航天发射中心。

---

## 2.4 实验门禁 A：最新代码和已提交全量产物版本不一致

当前代码和配置是 `schema/config v1.1.0`，但仓库中的 room0 全量 `audit_summary.json` 仍是 `v1.0.0`。

因此：

- v1.1 的候选角色补全；
- object-version fallback；
- 双队列抽样；
- selection probability 与 sampling weight；
- 新的 case packet 视图选择；

虽然有单元测试，但尚未由仓库中的 full-room0 产物证明端到端可用。

### 处理

修完 P0-1、P0-2 后，必须重新生成一份：

```text
room0 + schema 1.1.0 + full audit + case_selection.json
```

所有文件中的 schema/config/commit 必须一致。

---

## 2.5 实验门禁 B：非干扰性只在 2 帧 smoke 上证明

2 帧 smoke 很有价值，但仍未覆盖：

- 长时间连续 association；
- 同一对象多次更新；
- 多次 denoise/filter/merge；
- merge chain；
- object index 重排；
- 边更新；
- inactive object version 回溯；
- 更长时间的累计数值漂移。

此外，evidence 并非完全“零侵入”：

- 给 raw gobs 增加 `raw_det_idx/obs_uid`；
- 给 detection/object 增加 `obs_uids`；
- merge 时扩展 `obs_uids`。

这不代表设计错误，但意味着“完全不影响 mapping”必须由更长回归证明，而不能只依赖意图。

### 最小非干扰回归

使用缓存 detection、相同随机种子、相同配置，运行：

- A：`save_evidence=false`
- B：`save_evidence=true, evidence_mode=strict`

至少覆盖 80 到 120 帧，并确保触发 denoise、filter、merge。

比较以下 canonical signature：

```text
每个最终对象：
  sorted[(image_idx, mask_idx)]
  class_id histogram
  num_detections
  bbox center/extent
  point count
  point cloud hash 或逐点容差
  CLIP feature
  active status

最终图：
  object count
  edge count
  按 endpoint canonical signature 对齐后的 edge relation/support
```

通过条件：

- observation membership 完全一致；
- object/edge 拓扑完全一致；
- bbox/PCD/CLIP 在预设数值容差内一致；
- 每帧 object count 与 merge/filter 次数一致；
- 无 evidence logging error。

---

# 3. 为什么现在不应该继续重构

当前“逻辑繁杂”的感觉是真的，但这不等于现在应该重构。

## 3.1 现在重构的代价

在没有人工标签前，你不知道：

- 哪些 checker 真有用；
- 哪些规则可以删除；
- 哪些字段真正被 reviewer 使用；
- 哪些 Evidence Packet 图片是冗余的；
- 哪些阈值需要 scene-adaptive；
- 哪些根因值得进入最终方法。

现在重构，等于在不知道哪些器官有用时，先给整个系统做整形手术。代码会更漂亮，但研究风险不会下降。

## 3.2 最优原则

本轮只允许三类改动：

1. **防止实验结论错误的改动**；
2. **保证结果可复现的改动**；
3. **让人工标注和指标计算能完成的改动**。

以下内容全部延后：

- 重写 `EvidenceRecorder`；
- 合并 `evidence_audit.py` 与 `layered_audit.py`；
- 扩展 caption/relation checker；
- 加更多 part-whole 规则；
- 上 VLM 自动判断；
- 自动 detach/merge/delete；
- 完整 rollback engine；
- 全面性能优化；
- 把所有阈值改成自适应；
- 重新设计 3D 表示。

---

# 4. 下一步的唯一研究目标

## 4.1 阶段名称

**Audit Validity Gate v1：查错器真实性、因果性与可修复性验证**

## 4.2 本轮只回答四个问题

### Q1. Evidence 是否可信且不影响原始建图

这是系统门禁。

### Q2. Checker 报出的案例有多少是真异常

这是 finding precision。

### Q3. 真异常中有多少会污染最终对象图

这是 downstream harm，不是“看起来不太对”。

### Q4. 真且有害的异常中，有多少能映射到明确修复动作

这是 actionability，决定下一步应该先做哪一种回滚。

---

# 5. 本轮必须验证的四个假设

| 假设 | 内容 | 失败意味着什么 |
|---|---|---|
| H1 非干扰性 | evidence 开关不改变 mapping | 先修基础设施，不能继续 |
| H2 有效筛查 | 高优先级 findings 中有较高比例是真异常 | 否则需要删规则或重新定义信号 |
| H3 因果定位 | 上游 checker 能正确指出主要错误阶段 | 否则不能据此回滚 |
| H4 可操作性 | 相当一部分真异常有明确且局部的修复动作 | 否则“诊断”难转化为论文方法 |

---

# 6. 实验设计：最快但仍然可信的方案

## 6.1 场景选择

建议使用两个 Replica 场景：

### Development scene：`room0`

理由：

- 当前已有完整证据与审计产物；
- 便于对比 v1.0 与 v1.1；
- 你已经熟悉可视结果；
- 可以快速发现代码回归。

### Held-out scene：`office0`

理由：

- 与 room 场景类型不同；
- 原始 ConceptGraphs 论文中，office0 的 node precision 低于 room0，具有更高挑战性；
- 用同一套阈值直接测试，可以判断规则是否只适配 room0。

如果 office0 尚未准备好，可暂用 `room2`，但优先 office0。

## 6.2 主实验范围

本轮主实验：

```text
make_edges=false
caption checker=false
relation checker=false
```

只验证：

- detection；
- segmentation；
- geometry；
- association；
- fusion；
- object identity。

另外单独做一个 10 到 20 帧的：

```text
make_edges=true
```

只用于验证 VLM evidence capture 没断，不把 caption/edge precision 混入本轮核心结论。

## 6.3 配置冻结原则

1. 复制当前 `v1.yaml` 为 `v1_validation.yaml`；
2. 只允许修改：
   - finding cap；
   - case 数量；
   - 输出目录；
3. checker 阈值不改；
4. room0 结果出来后，office0 仍使用同一配置；
5. sample 生成后，不得回头改阈值再报告同一批结果。

---

# 7. Case 抽样：保留当前双队列，但重新定义用途

## 7.1 推荐样本量

每个场景生成 80 个待标案例：

| 场景 | Calibration random | Diagnostic priority | 合计 |
|---|---:|---:|---:|
| room0 | 40 | 40 | 80 |
| office0 | 40 | 40 | 80 |
| 总计 | 80 | 80 | 160 |

160 个案例通常可以在半天内完成首轮标注，同时足够回答“方向是否值得继续”。

如果单个 association packet 平均审查时间超过 2 分钟，可先完成 120 个：

- room0：80；
- office0：40。

但不要只审查 priority cases。

## 7.2 两个 cohort 的用途

### Calibration random

用于估计：

- overall finding precision；
- actionable precision；
- evidence sufficiency；
- 不同 stage 的粗略 precision；
- 跨场景稳定性。

计算时使用 selection probability 与 sampling weight。

### Diagnostic priority

用于估计：

- Precision@K；
- 最强案例是否真的强；
- 哪类错误最适合做第一种修复；
- 规则是否能提供可理解的根因；
- Evidence Packet 是否足够支持判断。

**禁止把两个 cohort 直接混合成一个“总体准确率”。**

---

# 8. 人工标注不只标“对/错”

这是本轮最关键的设计。

一个 `DET-001` 重复 proposal 可能确实重复，但如果两个 proposal 最终被吸收到同一节点，且没有导致节点重复、错误特征加权或后续关联变化，它可能是“真实异常但暂时无害”。

如果只标 true/false，会让查错器看起来很准，却无法证明研究价值。

## 8.1 推荐标签结构

每个 case 保存一行 JSON：

```json
{
  "case_uid": "finding_000123",
  "scene_id": "room0",
  "reviewer_id": "R1",

  "evidence_sufficient": "YES",
  "finding_correct": "YES",
  "root_stage_correct": "YES",

  "physical_interpretation": "same_physical_instance_duplicate_proposals",

  "downstream_harm": "LOCAL_WEIGHTING_BIAS",
  "harm_confidence": 4,

  "repair_action": "DROP_DUPLICATE_OBSERVATION",
  "repair_locality": "LOCAL",
  "repair_confidence": 5,

  "alternative_explanation": null,
  "review_seconds": 52,
  "notes": ""
}
```

## 8.2 枚举建议

### `evidence_sufficient`

- `YES`
- `NO`
- `PARTIAL`

### `finding_correct`

- `YES`
- `NO`
- `UNCERTAIN`

### `root_stage_correct`

- `YES`
- `NO`
- `UNCERTAIN`
- `NOT_APPLICABLE`

### `downstream_harm`

- `NONE`
- `LOCAL_WEIGHTING_BIAS`
- `WRONG_OBSERVATION_MEMBERSHIP`
- `FALSE_SPLIT_DUPLICATE_NODE`
- `FALSE_MERGE_IDENTITY_POLLUTION`
- `GEOMETRY_CORRUPTION`
- `RELATION_POLLUTION`
- `UNKNOWN`

### `repair_action`

- `NONE`
- `DROP_OBSERVATION`
- `REASSIGN_OBSERVATION`
- `MERGE_OBJECTS`
- `SPLIT_OBJECT`
- `RECOMPUTE_GEOMETRY`
- `DOWNWEIGHT_EVIDENCE`
- `NEED_MORE_VIEW`
- `UNKNOWN`

## 8.3 “可行动真错误”的定义

```text
finding_correct == YES
AND downstream_harm != NONE
AND repair_action not in {NONE, UNKNOWN, NEED_MORE_VIEW}
```

这会产生真正重要的指标：

> **Actionable Precision**

它比普通 checker precision 更接近论文价值。

---

# 9. 标注流程

## 9.1 首轮

- 由你完成全部 160 个；
- packet 默认显示 RGB、mask、depth、3D overlay、时间线和候选对象视图；
- 不先看 `certainty` 和 `review_score`，避免被规则结论暗示；
- 标注时记录 review time。

## 9.2 复核

从 160 个中随机抽 32 个，由队友独立标注：

- 16 个 calibration；
- 16 个 priority；
- 覆盖 detection、association、fusion、object identity。

报告：

- finding_correct 原始一致率；
- downstream_harm 一致率；
- repair_action 一致率；
- 分歧案例的最终 adjudicated label。

下一步只是方向决策，不必为了一个漂亮的 Cohen’s kappa 再耗两天。原始一致率和分歧原因已经足够。

---

# 10. 必须计算的指标

## 10.1 系统层

### Evidence Gate Pass Rate

```text
通过的 formal runs / formal runs 总数
```

目标：100%。

### Mapping Non-interference

```text
canonical object membership difference = 0
canonical edge difference = 0
geometry/feature difference within tolerance
```

目标：全部通过。

### Population Censoring

```text
suppressed findings per checker
```

目标：全部为 0。

---

## 10.2 查错有效性

### A. Weighted Finding Precision

仅使用 `calibration_random`：

\[
\hat{P}_{finding}
=
\frac{\sum_i w_i \cdot \mathbf{1}[finding\_correct_i=YES]}
{\sum_i w_i}
,\quad
w_i=\frac{1}{\pi_i}
\]

其中 \(\pi_i\) 为 case selection 中记录的 inclusion probability。

### B. Weighted Actionable Precision

\[
\hat{P}_{actionable}
=
\frac{\sum_i w_i \cdot \mathbf{1}[case_i\text{ 为可行动真错误}]}
{\sum_i w_i}
\]

### C. Priority Precision@K

仅使用 `diagnostic_priority`，分别报告：

- P@10；
- P@20；
- 每个 checker 的 top-case precision；
- actionable P@K。

它表示“优先队列是否节省人工检查时间”，不表示总体 prevalence。

### D. Root-stage Accuracy

在 `finding_correct=YES` 的案例中：

```text
root_stage_correct == YES 的比例
```

### E. Evidence Sufficiency

```text
evidence_sufficient == YES 的比例
```

若这一项低，下一步不是调阈值，而是补 packet。

### F. Actionable Error Yield

```text
确认的可行动真错误数 / 人工审查小时
```

它直接回答查错器是否比人从头浏览所有帧更省时间。

### G. Cross-scene Drop

```text
room0 actionable precision - office0 actionable precision
```

绝对下降过大说明规则过拟合 room0。

---

# 11. 本轮决策标准

这些不是学界通用常数，而是为了在时间受限情况下做研究决策的工程门槛。

## 11.1 GO：进入单一错误类型的最小修复实验

同时满足：

1. Evidence Gate 全通过；
2. evidence ON/OFF 非干扰回归通过；
3. 无 candidate population censoring；
4. priority actionable P@20 ≥ 0.70；
5. calibration weighted finding precision ≥ 0.50；
6. calibration weighted actionable precision ≥ 0.35；
7. root-stage accuracy ≥ 0.65；
8. evidence sufficiency ≥ 0.80；
9. office0 相比 room0 的 actionable precision 绝对下降不超过 0.20；
10. 至少有两类 checker 各发现 8 个以上可行动真错误，或一类 checker 发现 20 个以上。

通过后，下一步只选**一类**错误实现修复，依据标签中最多、危害最大、动作最局部的一类决定。

## 11.2 CONDITIONAL GO：规则有价值，但需要剪枝

满足：

- priority actionable P@20 在 0.50 到 0.70；
- 或 finding precision 尚可，但 actionable precision 低；
- 或仅 1 到 2 个 checker 有明显价值。

处理：

- 禁用低 precision checker；
- 合并重复规则；
- 对有效 checker 校准阈值；
- 不增加新规则；
- 再做一轮小规模验证。

## 11.3 STOP / REDESIGN

任一情况成立：

- evidence 改变 mapping；
- Evidence Gate 仍可能接受随机/伪造 similarity；
- priority actionable P@20 < 0.50；
- 大多数“真异常”没有下游危害；
- 根因定位经常错误；
- office0 上结果明显崩溃；
- reviewer 大量选择 evidence insufficient。

这时应重新思考问题定义，而不是继续堆查错规则。

---

# 12. 本轮具体执行顺序

## Step 1：冻结当前版本，开验证分支

```bash
git checkout ali-my
git branch freeze/ali-my-audit-v1.1-20260820
git checkout -b exp/audit-validity-gate-v1
```

冻结分支只用于回溯，不再修改。

## Step 2：修两个 P0

### P0-A

- 删除 similarity shape mismatch 的 `np.empty` fallback；
- 写 invalid status；
- Evidence Gate 阻断。

### P0-B

- 增加 attempted/emitted/suppressed counts；
- validation config 提高 finding cap；
- 任一 suppressed count 非 0 时阻断 calibration 结论。

### 顺手修改一个术语

当前 priority score 中的 `independent_evidence` 实际按 hypothesis support 字符串数量计算，不能严格代表统计独立证据。建议改名为：

```text
support_signal_diversity
```

这不影响算法，只避免论文中过度声称。

## Step 3：运行定向测试

```bash
PYTHONPATH=. pytest -q \
  tests/test_evidence.py \
  tests/test_evidence_audit.py \
  tests/test_layered_audit.py
```

新增至少 3 个测试：

1. matrix shape mismatch 必须 fail gate；
2. cap suppression 必须在 summary 中可见；
3. calibration 遇到 censored population 必须拒绝给 weighted precision。

## Step 4：做 80 到 120 帧 evidence ON/OFF 回归

建议复制现有 smoke 脚本，改为：

```text
start=0
end=600
stride=5
make_edges=false
save_json=true
evidence_mode=strict
```

实际帧数以 80 到 120 为准，并保证触发多次 postprocess。

固定：

```bash
PYTHONHASHSEED=0
```

并在 Python 入口固定：

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

输出 `parity_report.json`，不要只在终端打印“看起来一样”。

## Step 5：重跑 v1.1 room0 full audit

要求：

- manifest、findings、summary、case_selection 全部为 1.1.0；
- commit hash 指向验证分支；
- Evidence Gate PASS；
- population uncensored；
- case selection 中 40 random + 40 priority。

## Step 6：冻结阈值，运行 office0

不允许根据 room0 修改 checker threshold。

office0 运行后同样生成 80 个案例。

## Step 7：完成 160 个标签

产出：

```text
validation_gate/
├── config/
│   └── v1_validation.yaml
├── runs/
│   ├── room0/
│   └── office0/
├── parity/
│   └── parity_report.json
├── cases/
├── labels/
│   ├── labels_r1.jsonl
│   ├── labels_r2_subset.jsonl
│   └── labels_adjudicated.jsonl
├── metrics/
│   ├── overall_metrics.json
│   ├── metrics_by_checker.csv
│   ├── metrics_by_stage.csv
│   └── metrics_by_scene.csv
└── decision.md
```

## Step 8：只做 GO / CONDITIONAL GO / STOP 决策

本轮结束时不实现回滚。`decision.md` 只回答：

1. 哪些 checker 保留；
2. 哪些 checker 删除；
3. 最值得修的错误类型是什么；
4. 是否值得进入修复闭环；
5. 当前方法需要补证据、调阈值，还是改问题定义。

---

# 13. 推荐时间预算

| 时间 | 工作 |
|---|---|
| 第 1 天上午 2 到 3 小时 | P0 修复、新增测试、冻结配置 |
| 第 1 天下午 | parity run、room0 v1.1 full audit |
| 第 1 天晚上或并行服务器 | office0 audit、生成 packets |
| 第 2 天上午 3 到 5 小时 | 160 个案例首轮标注 |
| 第 2 天下午 1 到 2 小时 | 队友复核 32 个、分歧裁决 |
| 第 2 天下午 2 小时 | 计算指标、写 decision.md |

如果生成 full evidence 很慢，优先减少 frame 数，不要减少 calibration random 的比例。

---

# 14. 如何优化“逻辑繁杂”，但不做重构

本轮只做概念边界整理，不移动大量代码。

在文档和输出中固定五个层次：

```text
1. Evidence Ledger
   只记录事实与引用

2. Evidence Gate
   判断账本是否完整可信

3. Screeners
   产生风险候选，不宣称错误

4. Case Sampler
   产生 calibration 与 diagnostic 两个 cohort

5. Adjudicator / Evaluator
   人工或未来 VLM 给标签，并计算有效性
```

把当前所有 checker 对外统一称为 `screener`，把 `root_causes` 称为 `root-cause hypotheses`。

这样即使代码暂时仍在两个大文件里，论文逻辑也会立刻清晰很多。先让概念边界正确，再让文件结构漂亮。

---

# 15. 本轮明确不做的事情

- 不增加新的 checker；
- 不做 caption/relation 审计；
- 不调用 VLM 自动判错；
- 不设计完整 rollback transaction；
- 不自动删除、重分配或合并；
- 不进行全量参数搜索；
- 不为了速度重写 pairwise search；
- 不把 finding 数量当作方法效果；
- 不从 priority cohort 估计总体准确率；
- 不在看到 room0 结果后再调阈值测试 office0。

---

# 16. 为什么这条路线最适合当前论文目标

## 16.1 与 ConceptGraphs 原始问题直接对齐

ConceptGraphs 本身就承认：

- node caption 会错；
- 小或细物体会漏；
- 会产生 duplicate detections；
- 这些错误会影响 downstream planning。

它对 scene graph 质量也采用了人工判断的 node/edge precision，而不是只报告规则触发次数。因此，你现在最需要的不是第三批规则，而是把查错器候选转化为可报告的 precision、根因准确率和 downstream harm。

## 16.2 与后续领域趋势对齐

近期在线开放词汇 3D 方法越来越强调：

- 置信度建模；
- 多视角一致性；
- 语义与几何联合关联；
- temporal memory；
- local-to-global refinement；
- 对不可靠观测进行选择性更新。

这说明你的潜在论文价值不在“写一个规则报警器”，而在：

> **用可追溯证据估计错误可靠性，并让系统只对高置信、可行动的错误执行受控修正。**

但要走到这一步，必须先证明现有 screener 能稳定找到高价值错误。

## 16.3 与高质量系统论文的验证方式对齐

LM-Nav 不只说明三个模块能拼起来，还通过组件消融和真实导航结果判断：

- VLM grounding 是否有效；
- graph search 是否有效；
- traversability 是否有效；
- 哪一模块导致最终失败。

你现在也需要同样的逻辑：

```text
证据完整
→ 筛查准确
→ 根因定位正确
→ 错误确实有害
→ 修复动作明确
```

而不是：

```text
规则很多
→ findings 很多
→ 所以方法有效
```

后者很热闹，审稿人也会很热闹，只不过热闹的是拒稿理由。

---

# 17. 最终建议

当前最优决策不是“速度优先”或“逻辑优先”二选一，而是：

> **冻结当前复杂逻辑，只修会让实验结论失真的两处问题，然后用最小但科学的标注实验决定哪些逻辑值得保留。**

这会同时实现：

- 加快进度：不再继续扩展和重构；
- 优化逻辑：通过数据删掉无效 checker，而不是凭感觉整理；
- 降低选题风险：尽快知道诊断是否能转化为有害错误和修复动作；
- 为下一步选方法：由标签决定先做 drop、reassign、merge 还是 split；
- 为论文积累真实证据：precision、actionability、root-cause accuracy、cross-scene transfer、review efficiency。

**本轮完成的标志不是代码更漂亮，而是你能用一页 `decision.md` 明确回答：这个查错器是否值得继续，以及第一种最值得实现的修复是什么。**

---

# 参考依据

## 当前仓库

- `conceptgraph/utils/evidence.py`
- `conceptgraph/audit/evidence_audit.py`
- `conceptgraph/audit/layered_audit.py`
- `conceptgraph/audit/configs/v1.yaml`
- `conceptgraph/slam/rerun_realtime_mapping.py`
- `conceptgraph/slam/mapping.py`
- `conceptgraph/slam/utils.py`
- `docs/ALI_MY_EVIDENCE.md`
- `docs/audit_v1_artifacts/README.md`
- `docs/audit_v1_artifacts/room0_20260819/full_audit/audit_summary.json`
- `tests/test_evidence.py`
- `tests/test_evidence_audit.py`
- `tests/test_layered_audit.py`

## 论文与方法论

1. Gu et al., **ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning**, arXiv:2309.16650.
2. Huang et al., **Visual Language Maps for Robot Navigation**, arXiv:2210.05714.
3. Shah et al., **LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action**, arXiv:2207.04429.
4. Kossen et al., **Active Testing: Sample-Efficient Model Evaluation**, ICML 2021, arXiv:2103.05331.
5. Nguyen et al., **Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking**, CVPR 2025.
6. Tang et al., **OnlineAnySeg: Online Zero-Shot 3D Segmentation by Visual Foundation Model Guided 2D Mask Merging**, CVPR 2025.
7. Zhu et al., **OGScene3D: Incremental Open-Vocabulary 3D Gaussian Scene Graph Mapping for Scene Understanding**, arXiv:2603.16301.
8. Li et al., **Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding**, CVPR 2025.

---

## 评审边界说明

本评审基于远程 `ali-my` 源码、分支差异、仓库内测试和已提交运行产物。服务器中未提交到 Git 的完整二进制 observation PCD、全部 200 个可视 packet 和原始 Replica 数据无法在本次评审中逐个复核。因此，本文对“代码结构与实验设计”的判断置信度高，对“每条 checker 的真实 precision”不作未经标注的猜测，这正是下一步必须完成的内容。
