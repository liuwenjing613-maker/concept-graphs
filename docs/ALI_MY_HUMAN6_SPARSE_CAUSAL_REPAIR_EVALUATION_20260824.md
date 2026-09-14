# ALI-MY Human-Confirmed Sparse Causal Repair 实现与评估总结

> 日期：2026-08-24
> 执行位置：服务器 `server-3048-out`
> 仓库：`/home/chenkejun/beauty/conceptgraphs/code/official/ali-my-revision`
> 分支：`exp/ali-my-revision-kernel-v1`
> 正式结果：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_human6_causal_pilot_v2_20260824`
> 保留的首次失败结果：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_human6_causal_pilot_20260824`

## 1. 最终结论

本轮完成了 `CREATE_INSTANCE`、阈值语义和分阶段计时的实现与真实人工错误验证。冻结的 6 个案例来自现有 40 个 human-confirmed endpoint errors，严格保持 3 个 false merge + 3 个 false split，没有根据运行结果换案例。

六例经过逐案因果审计后分为：

- 3 个 `REPLAYABLE_ASSOCIATION_CAUSE`：2 个 false merge、1 个 false split；
- 3 个 `DEFER_NON_ASSOCIATION_ROOT`：错误根因位于 segmentation/geometry，association sparse primitive 无法诚实修复。

对 3 个可重放案例，V2 的严格对照结果为：

| 指标 | 结果 |
|---|---:|
| Native endpoint wrong | 3/3 |
| Natural replay still wrong | 3/3 |
| Sparse causal repair endpoint correct | 3/3 |
| Strict contrast pass | 3/3 |
| Conditional replayable repair rate | 100% |
| 全 6 个人工错误上的 repair yield | 50% |
| 诚实延期 | 3/6 |
| 作用域外观测变化 | 0 |
| Sparse runtime invariant pass | 3/3 |
| Source hashes unchanged | 是 |

因此，本轮结论是：

> `GO`：谱系级 persistent sparse causal repair 在这 3 个已审计、可由 association 修复的真实错误上成立，而且 Native/Natural replay 均不能自行修正。

但本轮不支持以下更强结论：

- 不泛化到全部 40 个 human-confirmed errors；
- 不把 3 个 segmentation/geometry 根因的延期计作成功；
- 不声称 office0 的关系正确性，因为没有非空 relation stream；
- 不声称物理世界几何真值已经修复，因为没有 corrected geometry gold；
- 不声称 blind root-cause localization、自动 VLM 约束生成或冷启动端到端加速已经完成。

## 2. 为什么本轮只跑 6 例和 2 个场景

本轮目标是先验证方法语义，而不是尽快扩大实验规模。一次跑完 40 例或更多场景会在方法仍可能调整时产生大量重复计算。

实际采用的最小判别性设计是：

| 维度 | 实际范围 | 原因 |
|---|---|---|
| 人工错误 | 6/40 | 满足 3 false merge + 3 false split |
| 场景 | room0、office0 | 复用完整 evidence 与已有基线 |
| 可重放执行 | 3 | 只对 association-rooted 错误运行 sparse replay |
| 因果延期 | 3 | 不用错误 primitive 伪造修复 |
| 正式重跑 | V1、V2 各一次 | V1 保留失败证据；V2 使用同一冻结 cohort |

这符合“先跑一两个场景、先取得方法级信息，再决定是否扩展”的资源策略。

## 3. 冻结来源与选择完整性

正式输入来源：

| 来源 | canonical-LF SHA-256 | 验证 |
|---|---|---|
| expert confirmed queue | `a8445b5acb8b7f1cac72aedde1948ea7c7edd9a72eae375608ba089a57666feb` | PASS |
| R1 frozen labels | `f7db781367e6343fe01fc81a1fbcf48cc92917847dae4fb329eae19e4ff0861a` | PASS |
| R2 frozen labels | `83de2b09a8d3022a555465e81dbb61e6d1ed4360915bfe745af43f020de9671b` | PASS |

冻结 cohort 校验：

- case count = 6；
- false merge = 3，false split = 3；
- replayable association cause = 3；
- deferred non-association root = 3；
- R2-confirmed selected cases = 3；
- `selection_is_purposive=true`；
- `population_inference_to_all_confirmed_errors=false`。

冻结清单：

`docs/revision_v1_audits/HUMAN_CONFIRMED_SIX_CAUSAL_PILOT_MANIFEST_20260824.json`

## 4. 六例因果审计

| Case | 类型 | 处置 | 最早可解释根因 | Sparse 动作 |
|---|---|---|---|---|
| `human6_office0_false_split_51aaf9ba` | false split | replayable | anchor CREATE 产生重复 sofa 谱系 | `ASSIGN_OBSERVATION` + persistent lineage redirect |
| `human6_room0_false_merge_06525b4b` | false merge | replayable | 新 outlet 后续跨实例关联/后处理合并 | `CREATE_INSTANCE` persistent boundary |
| `human6_room0_false_merge_9727f850` | false merge | replayable | 新 table-part 被后处理并入既有对象 | `CREATE_INSTANCE` persistent boundary |
| `human6_room0_false_merge_f74cb76c` | false merge | deferred | 单个 mask 已同时含 table/floor | segmentation undersegmentation |
| `human6_room0_false_split_05c2ca82` | false split | deferred | 两 proposal 已正确并入同一 cabinet；无合法第二 target | segmentation/geometry fragmentation |
| `human6_room0_false_split_4adafe73` | false split | deferred | 两 sofa proposal 已并入同一 owner；邻近节点语义不合法 | containment/segmentation fragmentation |

延期理由不是执行失败：

1. table/floor 案例的唯一观测在 association 前已经混合几何，重新指定 owner 无法拆分一个 mask 内部的点；
2. cabinet 案例的两个 reviewed proposals 已以 `1.897626 > 1.2` 并入同一 lineage，不存在第二个可审计 cabinet target；
3. sofa 案例的两个 proposals 已以 `1.575825/1.577569 > 1.2` 并入同一 owner，强制到邻近节点会制造类别错误。

## 5. 实现语义

### 5.1 `CREATE_INSTANCE` 是持久实例边界，不是一次 `CREATE_OBJECT`

V1 已能在 anchor 强制创建，并能阻止 object-object postprocess merge，但真实 outlet 案例证明这还不够：未来观测可先自然关联到边界另一侧，污染对象的推断谱系，之后 postprocess guard 会误以为两侧属于同一 lineage。

V2 同时在两个边界执行保护：

1. detection-object association；
2. object-object postprocess merge。

关联阶段规则：

- 用 immutable provenance lineage 标记 incoming observation；
- 用当前对象的 revision/provenance lineage 标记 candidate；
- 对每个 protected lineage，若 observation 与 candidate 恰好只有一侧包含该 lineage，则 candidate 禁止；
- 默认 target 被禁止时，选择分数最高且严格满足 threshold 的同侧 candidate；
- 没有 eligible 同侧 candidate 时，确定性地创建同 lineage object；
- observation provenance 未知时不把“未知”解释成“非 protected”，避免用缺证据制造否决；
- 显式直接约束优先于派生边界规则。

这没有降低 `sim_threshold`，而是在相似度判定之外增加显式 instance identity constraint。

### 5.2 `ASSIGN_OBSERVATION` 的 persistent lineage redirect

false split 的 anchor 修正后，Natural replay 在 frame 35 再次出现 `0.803134 < 1.2`，因此重新 CREATE duplicate。仅修 anchor 不能修最终 endpoint。

V2 只在以下条件成立时推导 persistent redirect：

- primitive 是 `ASSIGN_OBSERVATION` 或 `MUST_LINK`；
- anchor 的 recorded association 是 `CREATE_OBJECT`；
- anchor 可解析出 immutable created/source lineage；
- 当前模式是 `PERSISTENT_SPARSE_CONSTRAINT_REPLAY`。

从此以后：

- 携带 source lineage 的 observation 必须解析到唯一 active target；
- target 不存在、target 模糊或多个 redirect 冲突时硬 `DEFER`；
- 不读取 clean final membership，不枚举未来 trajectory members；
- target lineage 写入 revision lineage metadata，防止被 redirected observation 的旧 provenance 污染；
- `ANCHOR_ONLY_REPAIR` 不启用这一派生规则。

冻结 primitive 的 `active_from/active_until` 只限制直接 anchor action。persistent derived closure 的生命周期从 anchor 后开始，到 suffix 结束；否则把 anchor interval 当作 propagation lifetime 会让 persistent mode 退化为 anchor-only。

### 5.3 Threshold semantics

统一语义为：

`score > sim_threshold` 才允许 MERGE；`score == sim_threshold` 必须 CREATE。

实现与 trace 均使用：

- comparator：`STRICT_GREATER_THAN`；
- threshold：`1.2`；
- equality decision：`CREATE_OBJECT`；
- 每个决策记录 top-1 score、`score-threshold`、eligible 和 native decision。

`cfslam_pipeline_batch.py`、`mapping.py`、replay candidate ranking 和评估 trace 使用相同规则，不再混用 `>=` 与 `>`。

### 5.4 Timing instrumentation

每个案例/分支分别记录：

- snapshot amortized wall time；
- snapshot cold upper-bound wall time；
- suffix execute wall time；
- suffix overlay wall time；
- suffix orchestration wall time；
- relation rebuild wall time；
- endpoint evaluation wall time；
- runtime invariant verification wall time；
- case total 和 run total wall time。

兼容字段 `runtime_ms` 保留，但报告中不把 suffix-only 时间包装成 cold end-to-end 时间。

## 6. 首次失败被保留，而不是被覆盖

V1 root：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_human6_causal_pilot_20260824`

V1 结果：

| 指标 | V1 |
|---|---:|
| Strict contrast pass | 1/3 |
| Sparse endpoint correct | 1/3 |
| Conditional replayable repair rate | 33.33% |
| Overall 6-case repair yield | 16.67% |
| Pilot status | FAIL |

两个真实失败：

1. office0 false split：anchor `FORCE_TARGET` 正确，但 frame 35 再次 CREATE，说明 positive repair 未持久；
2. room0 outlet false merge：frame 139 第一次 postprocess veto 正确，但后续 association 污染 protected lineage，最终仍合并。

V2 没有替换案例，只修正这两个语义缺口后在新 root 重跑同一 cohort：

| 指标 | V1 | V2 |
|---|---:|---:|
| Native wrong | 3/3 | 3/3 |
| Natural still wrong | 3/3 | 3/3 |
| Sparse correct | 1/3 | 3/3 |
| Strict contrast pass | 1/3 | 3/3 |
| Conditional rate | 33.33% | 100% |
| Overall 6-case yield | 16.67% | 50% |
| Status | FAIL | PASS |

详细失败链条已追加到：

`docs/revision_v1_audits/FAILED_RUNS_LEDGER.md` 的 F35、F36。

## 7. 三个可重放案例的详细结果

| Case | Gold group sizes | Native | Natural | Sparse | 持久机制 | Affected obs | Outside obs / changed |
|---|---:|---|---|---|---|---:|---:|
| office sofa false split | 131 + 119 | wrong | wrong | correct | 130 lineage activations；1 次真实 override | 250 | 1310 / 0 |
| room outlet false merge | 3 + 15 | wrong | wrong | correct | 6 association veto；24 postprocess veto | 34 | 3745 / 0 |
| room table-part false merge | 10 + 51 | wrong | wrong | correct | 0 association veto；6 postprocess veto | 71 | 3708 / 0 |

进一步解释：

- office sofa：Sparse 将两个完整、atomic group 归到同一 owner；`group_owners_pairwise_disjoint=false` 在该 `SAME_OWNER` 任务中是正确结果。
- outlet 和 table-part：Sparse 保持两组各自 atomic 且 owner pairwise disjoint。
- 三例均无 missing/extra/duplicate observation；
- 三例 `outside_partition_exact_to_native=true`；
- 三例 runtime invariants 全部通过；
- Sparse 分支没有任何作用域外 observation 改变。

## 8. 关键阈值与因果证据

| 事件 | top-1 | threshold delta | Native/Natural | Sparse |
|---|---:|---:|---|---|
| office split anchor f32 r1 | 0.772956 | -0.427044 | CREATE | FORCE_TARGET 到既有 sofa |
| outlet anchor f122 r15 | 0.901358 | -0.298642 | CREATE | FORCE_CREATE + protected lineage |
| table-part anchor f144 r19 | 0.982718 | -0.217282 | CREATE | FORCE_CREATE + protected lineage |
| office successor f35 r0 | 0.803134 | -0.396866 | 再次 CREATE，重建 false split | persistent redirect 到 target |
| outlet successor f140 r14 | 1.909036 | +0.709036 | MERGE 到跨边界 outlet | boundary veto，因无 eligible 同侧 target 而 CREATE |

这组证据同时说明：

- 两个 false merge 的 anchor action 表面上与 Natural 一样都是 CREATE；真正差异是 persistent instance semantics；
- false split 的 anchor 修复仍不足，必须观察后续 Natural replay；
- boundary guard 没有修改 threshold；即使相似度很高，只要显式 instance constraint 证明跨边界，也必须拒绝；
- 相等边界已有独立单测，`score == 1.2` 不会 merge。

## 9. Strict contrast 的通过条件

一个 replayable case 只有同时满足下列条件才计作 PASS：

1. Native endpoint wrong；
2. Natural replay endpoint wrong；
3. Sparse endpoint correct；
4. 评估 group 均 atomic；
5. SAME_OWNER/DIFFERENT_OWNER 条件正确；
6. collateral safe；
7. runtime invariants pass；
8. sparse mechanism trace verified；
9. pre-anchor snapshot validation pass；
10. relation rebuild PASS 或明确 `NOT_REQUESTED`；
11. affected scope 外的 relation signature 与 Native exact；
12. source hashes unchanged。

因此，V2 的 3/3 不是只看 anchor action，也不是只看最终对象数。

## 10. Relation、geometry 与 source safety

### 10.1 Relation

room0 复用已有非空 relation stream：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_edges_make_true_room0_20260823`

两个 room0 replayable case：

- Native/Natural/Sparse relation rebuild 均通过 backend validation；
- Sparse affected scope 外 relation signature 与 Native exact；
- relation rebuild 单分支约 6.1–6.7 ms。

office0 没有找到非空 relation stream，因此明确为 `NOT_REQUESTED`，没有用 empty relation 假装关系验证成功。

本轮只证明 relation pipeline structural consistency，不在缺少 human relation labels 时声称 relation semantic correctness。

### 10.2 Geometry

报告保存 affected owner 的 bbox/point diagnostics 和有限几何不变量，但没有 corrected geometry gold。因此只声明：

- committed object geometry 有限且结构有效；
- sparse repair 没有作用域外 membership collateral；
- 不声明物理几何真值已恢复。

### 10.3 Immutable sources

正式运行前后，expert queue、R1、R2 和 provenance source hashes 均未变化。`source_hashes_unchanged=true`。

## 11. Timing 结果

正式 V2 总 wall time：

- `543,614.46 ms`，约 `9.06 min`。

| Case | Natural suffix | Sparse suffix | Sparse - Natural | Snapshot amortized |
|---|---:|---:|---:|---:|
| office sofa split | 51.503 s | 53.891 s | +2.388 s | 7.245 s |
| room outlet merge | 81.791 s | 81.436 s | -0.355 s | 1.516 s |
| room table-part merge | 64.145 s | 63.842 s | -0.303 s | 1.894 s |
| Mean | 65.813 s | 66.390 s | +0.577 s | 3.552 s |

在这 3 例上，Sparse suffix mean 比 Natural 高约 `0.88%`。样本太小，且运行顺序/缓存共享会影响 wall time，因此不把这一差异解释为稳定性能回归或加速。

Snapshot cold upper-bound 范围为 `7.182–117.971 s`，明显高于部分 cached/amortized 计时。当前可以报告分阶段成本，但仍不能声称 cold end-to-end acceleration。

Sparse 分支其他阶段：

- endpoint evaluation：约 3.7–10.2 ms；
- invariant verification：约 22.1–68.1 ms；
- room0 relation rebuild：约 6.1–6.3 ms；
- suffix replay attempt：每例 1 次，没有隐藏 retry。

## 12. 测试与静态验证

服务器最终执行：

`79 passed, 3 warnings in 2.79s`

覆盖：

- 全部 `tests/test_revision*.py`；
- `tests/test_mapping_threshold_semantics.py`；
- 新增 association-time `CREATE_INSTANCE` boundary regression；
- 新增 false-split mechanism 必须具有 persistent redirect override 的回归；
- constraint/compiler/runtime verifier/snapshot/scope/relation/corruption/parity 等既有测试。

另执行：

`python -m compileall -q conceptgraph/revision conceptgraph/slam scripts/run_revision_human_error_pilot.py tests`

结果通过。

3 个 warning 均来自既有依赖的 deprecation 信息（faiss/distutils、supervision），不影响本轮结果。

## 13. 主要代码位置

- `conceptgraph/revision/constraints.py`：`CREATE_INSTANCE`、constraint type/action/conflict semantics；
- `conceptgraph/revision/sparse_replay.py`：threshold trace、persistent association boundary、lineage redirect、postprocess guard、mechanism counters；
- `conceptgraph/slam/mapping.py`：严格大于 threshold 的 canonical matcher semantics；
- `conceptgraph/slam/cfslam_pipeline_batch.py`：生产 batch path 的同一 threshold 语义；
- `conceptgraph/slam/utils.py`：revision lineage 传播和 postprocess merge guard；
- `conceptgraph/revision/runtime_verify.py`：constraint execution 与 created-lineage runtime invariants；
- `conceptgraph/revision/snapshot.py`、`evaluate.py`、`benchmark/experiment_v1.py`：分阶段 timing；
- `conceptgraph/revision/benchmark/human_error_pilot.py`：冻结 cohort、因果验证、strict contrast、collateral 与 mechanism evaluation；
- `scripts/run_revision_human_error_pilot.py`：正式服务器入口；
- `tests/test_revision_constraints_v1.py`、`tests/test_revision_human_error_pilot.py`、`tests/test_mapping_threshold_semantics.py`：关键回归。

## 14. 可复现实验路径

### 正式 V2

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_human6_causal_pilot_v2_20260824`

关键文件：

- `aggregate_metrics.json`；
- `cohort_validation.json`；
- `source_verification.json`；
- 每例 `case_manifest.json`、`causal_validation.json`、`constraint.json`、`dependency.json`、`pre_anchor_snapshot.json`、`relation_rebuild.json`；
- 每例 `branches/native.json`、`branches/natural.json`、`branches/sparse.json`；
- 每例 `metrics.json`。

### 保留的 V1 失败

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_human6_causal_pilot_20260824`

两个根目录并存，最终 aggregate 不扫描额外目录，也不会把 V1 失败混入 V2。

## 15. GO / FIX / 下一步

| 能力 | 决策 | 当前证据 |
|---|---|---|
| `CREATE_INSTANCE` anchor semantics | GO | 两个真实 false merge anchor trace verified |
| association + postprocess persistent boundary | GO（3-case scope） | 6 association veto + 30 postprocess veto；2/2 FM 修正 |
| positive lineage redirect | GO（1-case scope） | f35 Natural CREATE 被唯一必要 override 修正 |
| strict threshold semantics | GO | production/replay 统一；equality 单测 |
| collateral/runtime safety | GO（3-case scope） | outside changed 0；invariants 3/3 |
| segmentation/geometry repair | DEFER | 3 个根因超出 association primitive 能力 |
| relation semantic correctness | 未证明 | room0 structural pass；无 human relation gold |
| 全 40 泛化 | 未证明 | purposive 6-case pilot，不做总体推断 |
| cold end-to-end acceleration | 未证明 | 只有 stage timing 与 cold upper bound |
| blind/automatic repair generation | 未运行 | 使用 frozen human-confirmed causal evidence |

下一步最有信息增益的顺序不是立即跑完全部 40 例：

1. 先冻结 1–2 个 clean-negative / no-op 案例，检验 persistent rules 是否产生 false mutation；
2. 再从剩余 confirmed errors 中冻结 1 个新的 association-rooted false merge 和 1 个新的 false split，优先选择不同场景；
3. 只有上述小 gate 通过且语义不再变化，再扩展一个 holdout 场景；
4. segmentation/geometry 根因应进入新的 primitive 设计，不应继续用 association action 硬修。

本轮已经达到当前最合理的停止点：核心语义在真实错误上有严格对照证据，同时保留了失败、延期和不支持的结论。
