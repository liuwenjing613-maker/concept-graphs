# ALI-MY Revision Kernel V2：Identity / Provenance 正式化与验证报告

日期：2026-08-24
执行环境：远程服务器 ubun；GPU 验证全部在服务器完成
代码工作树：/home/chenkejun/beauty/conceptgraphs/code/official/ali-my-revision-human6-publish-20260824
实验根目录：/home/chenkejun/beauty/conceptgraphs/experiments/revision_identity_provenance_v2_20260824

## 1. 最终结论

本轮已经把 revision kernel 从“依赖对象当前归属的补丁式回放”推进到“显式区分不可变来源与可修订身份、并以成对身份边界约束后续关联和合并”的 V2 语义。

在当前唯一可精确回放的 3 个 human-confirmed identity errors 上：

- Native replay：3/3 仍错误。
- Natural replay：3/3 仍错误。
- Sparse causal repair：3/3 修正确。
- 严格反事实对照，即 Native 错、Natural 错、Sparse 对：3/3。
- 对 6 个选定 human-confirmed cases 的总体修复产出率：3/6 = 50%。
- 其余 3/6 因缺少可执行的观测级或几何级干预而显式 DEFER，没有伪装成成功。

这说明 identity/provenance 正式化已经解决当前 3 个真实可回放 FM/FS 的核心语义问题，但不能据此声称解决了全部 40 个错误，也不能声称已经获得跨场景总体泛化能力。

自动 constraint generation 已接成完整的 fail-closed 原型。它在 5 个盲测案例上没有产生任何不安全提交，但 identity 案例 0/3 可提交，说明生成器尚未达到生产可用水平。当前正确策略是保留严格门槛，提升证据，而不是降低 gate。

## 2. 本轮要求与完成状态

| 要求 | 完成情况 | 结论 |
|---|---:|---|
| identity / provenance 正式化 | 完成 | 不可变 provenance 与 effective identity 分离 |
| 2 个 no-op 负例 | 完成 | 2/2 精确不变 |
| 1 个合法 merge 负例 | 完成 | 15/15 检查通过，合法 merge 未被误拦截 |
| 现有 3 例组件消融 | 完成 | 找到 FS 与 FM 各自必要组件 |
| 1 FM + 1 FS local/global parity | 完成 | 两例均 9/9 parity 通过 |
| 2 个新 holdout | 完成 | 几何恢复与语义重标注能力探针；均保持隔离 |
| 小规模 relation gold | 完成 | 44 个判定，限定域 F1 为 0.952381 |
| PARTITION_OBSERVATION 设计 | 完成设计与纯函数执行器 | 真实 f74cb 因无 point gold 而 DEFER |
| 自动 constraint generation | 完成原型与盲测 | 0 unsafe commit，但 identity 生成能力不足 |
| GPU 正式回放与指标 | 完成 | 只在远程服务器执行 |
| beauty 目录总结 | 完成 | 本文同时复制到 /home/chenkejun/beauty |

ali-dev 与 ali-my 的 make_edges=true 已由此前实验完成，本轮直接复用其产物，没有浪费 GPU 重跑。

## 3. Identity / provenance 正式语义

### 3.1 两个不能再混用的量

V2 明确分开：

1. Provenance lineage
   表示对象由哪些原始 observation / detection / object lineage 产生。它是审计事实，必须保持不可变，不能因为 revision redirect 或 merge 而被重写。

2. Effective identity
   表示 revision 之后对象在当前世界模型中的身份归属。它可以被 ASSIGN、CREATE_INSTANCE 或合法 merge 更新。

因此：

- FORCE_TARGET 只能改变 effective identity，不能覆盖 provenance。
- CREATE_INSTANCE 必须创建新 identity，同时继承观测来源的 provenance。
- 合法 merge 合并 effective identity 的成员关系与审计元数据，但不能抹掉两个来源 lineage。
- 判断“是否禁止合并”必须针对一对 identity，而不是对某个对象设置永久的全局禁止标记。

### 3.2 CREATE_INSTANCE 的精确定义

CREATE_INSTANCE 现在包含：

- created_identity_uid：新建实体的明确身份。
- separate_from_identity_uids：必须与哪些身份保持分离。
- source observation 的不可变 provenance。
- identity contract completeness：信息不完整时不允许静默降级。

修复 false merge 的关键不是“创建了一个对象”本身，而是创建之后把“新 identity 与错误吸附目标 identity 之间的边界”传播到：

- association-time gate；
- postprocess merge gate；
- global replay 的后续帧。

这正是旧实现中 Natural replay 仍会重新合并的原因。

### 3.3 Threshold semantics

阈值语义已固定为单调、可解释的决策：

- 候选分数达到接受阈值才允许正向关联。
- 身份边界属于硬约束，不能被高相似度覆盖。
- 不完整或冲突的 identity contract 必须 DEFER。
- pair-specific boundary 只阻止指定身份对，不能演变成“该对象永远不允许任何 merge”。

这使合法 merge 负例能够通过，同时错误 merge 被阻止。

## 4. Human-confirmed six 正式回放

正式输出：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_identity_provenance_v2_20260824/human6_formal_identity_v2

聚合文件：

aggregate.json
SHA-256：4e56c880edb08c97eb9f2fc499c5be54f44210e65d5ea7401836f686b244ad51

### 4.1 三个可回放真实错误

| 案例 | 类型 | Native | Natural | Sparse | 结果 |
|---|---|---:|---:|---:|---|
| office0 51aaf9ba | false split | 错 | 错 | 对 | PASS |
| room0 06525b4b | false merge，outlet | 错 | 错 | 对 | PASS |
| room0 9727f850 | false merge，table part | 错 | 错 | 对 | PASS |

关键点：本轮使用新的 pair-boundary 语义完整重跑，并非把旧失败运行与新局部结果拼接。早期 human3_pair_boundary_smoke 的 1/3 失败产物被保留作为迭代证据。

### 4.2 三个明确延迟的错误

- room0 f74cb：table / floor undersegmentation，需要 observation point partition。
- room0 05c2：cabinet fragmentation，需要几何或观测级恢复能力。
- room0 4ada：sofa fragmentation，需要几何或观测级恢复能力。

这三例不能由当前 ASSIGN / CREATE_INSTANCE 的 identity 层可靠表达，因此选择 DEFER 是正确行为，不计作修复成功。

## 5. 负例与组件消融

### 5.1 两个 no-op

输出：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_identity_provenance_v2_20260824/negative_controls/aggregate.json

SHA-256：0ac4a0d6cbb5ea7415cf8d934e92331515231273048adf548b2a14cdae832bd0

结果：

- 2/2 通过。
- endpoint partition 完全不变。
- geometry 完全不变。
- 没有额外 identity mutation。

这排除了“框架开启后即使无有效 constraint 也会扰动地图”的风险。

### 5.2 一个合法 merge

输出：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_identity_provenance_v2_20260824/legal_merge_global/negative_room0_human_correct_merge_e00007175/legal_merge_result.json

SHA-256：8418ddad3fad1ad65d42a597b176b7cb93098d51c03c28d98ee34eacc69c0d42

结果：15/15 检查通过，运行时间 257992.750 ms。

审计时发现 trace 的合法语义不是 decision=MERGE，而是：

- operation=OBJECT_MERGE_CANDIDATE
- decision=ACCEPT
- reject reasons 为空

因此修正了评估器的错误假设，并保留 failed_evaluator_v1.json 作为审计记录。最终结论来自重新审计已保存状态，不是篡改运行结果。

### 5.3 三例组件消融

输出：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_identity_provenance_v2_20260824/component_ablation/aggregate.json

SHA-256：57f2de1bca7d0a4b5fe1202259dc20893eedc2727e1657d9eb25f8e5559eaf55

结论：

- False split：关闭 persistent positive-lineage redirect 后 endpoint 再次错误，因此 redirect 是必要组件。
- Outlet false merge：
  - 关闭 association guard 仍正确，postprocess 共 veto 13 次。
  - 关闭 postprocess guard 后错误，association 共 veto 9 次。
  - 两者都关闭后错误。
- Table-part false merge：
  - 关闭 association guard 仍正确，postprocess 共 veto 6 次。
  - 关闭 postprocess guard 后错误，association 共 veto 7 次。
  - 两者都关闭后错误。

严格解释：

- 当前两个真实 FM 中，postprocess identity guard 对最终 endpoint 是必要的。
- association guard 是有价值的 defense-in-depth，但单独不足以保证 endpoint，也不是这两个案例上的 endpoint 必要条件。
- 不能把这个 2-case 结果外推成 association guard 永远不必要。

## 6. Local / global parity

输出目录：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_identity_provenance_v2_20260824/local_global_parity

### 6.1 False split parity

案例：human6_office0_false_split_51aaf9ba
结果：9/9 检查通过
SHA-256：f48564588590164c4edb79050550f0f78a6c9ed76c90468f3b551c11c98736a4
global runtime：129686.527 ms

### 6.2 False merge parity

案例：human6_room0_false_merge_06525b4b
结果：9/9 检查通过
SHA-256：e940160adf5bb3de7235a0147275c1494c3f2cec435a7a38e445251c919941e4
global runtime：265083.146 ms

两例均满足 local 与 global 在：

- endpoint ownership；
- effective identity；
- provenance preservation；
- boundary activation；
- constraint decision；
- invariant status

上的一致性。尚未完成第二个 table-part FM 的 global parity，因此它仍是下一轮优先验证项。

## 7. Timing instrumentation

Human6 正式运行总耗时：590380.613 ms。

三个 replayable cases 的 suffix timing：

| 模式 | Mean | Median |
|---|---:|---:|
| Natural | 68.938 s | 66.971 s |
| Sparse | 73.359 s | 67.326 s |
| 差值 | +4.421 s，+6.41% | +0.355 s，+0.53% |

解释：

- 中位数开销接近持平。
- 均值增幅受个别运行波动影响。
- 样本只有 3 个，不能把它作为稳定吞吐结论。
- instrumentation 已能分离 load、association、postprocess、constraint 与总运行阶段，下一轮应增加重复运行和置信区间，而不是扩大到全部场景。

## 8. 两个 capability holdout

冻结清单：

docs/revision_v2_audits/CAPABILITY_HOLDOUT_MANIFEST_20260824.json
SHA-256：1775150f449256b4c4b8da98db21c1e8f59023d68c70385e501660bcfa912551

### 8.1 Geometry holdout

- room0 incident 38d。
- raw mask 正确，processed mask 被破坏。
- 期望 action family：RESTORE_OBSERVATION_GEOMETRY。
- 当前没有完整 executor 与反事实 evaluator，因此只能 DEFER。

### 8.2 Semantic holdout

- office0 incident bf85。
- human gold：whiteboard。
- 期望 action family：RELABEL。
- 当前没有完成 semantic relabel 的执行与保持性评估，因此只能 DEFER。

这两个 holdout 是新能力的盲测探针，不是跨数据集泛化估计，也不能与已使用过的 human6 混作测试集。自动生成评估后它们已经被消费；正式实现对应能力后必须再冻结新的未见 holdout。

## 9. 小规模 relation gold

输出：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_identity_provenance_v2_20260824/relation_gold/evaluation.json

SHA-256：28e42528ff65e69e5d95a9e0d0268573a28517dcd1546d89fe3c0178be0a8102

数据：

- 11 个映射后的显式 on triples。
- 44 个限定域标签：22 positive，22 negative。
- TP=20，FN=2，FP=0，TN=22。
- Accuracy=0.954545。
- Balanced accuracy=0.954545。
- Scoped precision=1.0。
- Scoped recall=0.909091。
- Specificity=1.0。
- F1=0.952381。
- 另有 29 个预测位于 gold 范围之外，统一标为 unknown。

因此 precision=1.0 只对当前限定 gold 域成立；不能把 29 个 unknown 当作真阴性，也不能宣称全图 relation precision 为 1.0。

## 10. PARTITION_OBSERVATION

新增了 observation partition schema 与纯函数执行器，核心约束为：

1. Source binding
   绑定 source observation UID 与完整 observation payload SHA-256，防止对错帧或错版本执行。

2. Exhaustive
   每个原始 point 必须且只能被分配一次。

3. Disjoint
   子实例不能共享同一点索引。

4. Non-empty
   每个输出 part 至少包含一个点。

5. Canonical assignment
   使用规范 uint16 assignment 与 assignment hash，保证可复现。

6. Atomicity
   所有验证先通过才生成任何子对象；失败时不允许半执行。

7. Provenance preservation
   子对象继承 parent observation provenance，但拥有不同 effective identities。

真实 f74cb 设计审计：

- source points：1990 × 3，float64。
- source colors：1990 × 3，float64。
- payload SHA-256：16bc732c4608950806dbcec19679fcf8247edf76d3c478beed6d1c31e5dd7d7cd。
- Native CREATE 前 DBSCAN 显示 2 个 cluster，但 cluster labels 未被保存。
- 当前没有 human point assignment / point gold。

最终状态：DEFER，不生成真实 constraint。

DBSCAN 的“2 clusters”只能作为标注候选，不能代替 gold；否则会把模型自己的假设当成监督真值，破坏因果验证。

设计文件：

docs/revision_v2_audits/PARTITION_OBSERVATION_F74CB_DESIGN_20260824.json
SHA-256：473e7f25665a5c13041617631d08a8947451cf40de58f4ced04f9da1133d63f6

## 11. 自动 constraint generation

### 11.1 采用的四阶段框架

自动生成实现为严格的 candidate → normalize → shadow gate → commit/defer：

1. Candidate generation
   只接收盲化后的观测证据与结构摘要，生成候选 action、target、evidence citation 和 confidence。

2. Normalize and identity contract
   将自由输出规范化成 ASSIGN、CREATE_INSTANCE、RELABEL、RESTORE 或 PARTITION；解析 identity aliases；检查 provenance、target completeness、pair boundary 与 action schema。缺字段或冲突即 DEFER。

3. Counterfactual shadow gate
   候选必须在隔离副本中通过 endpoint、collateral、invariants、source immutability、no-op、legal-merge、component ablation、local/global parity 和 independence 检查。

4. Commit or defer
   只有所有 gate 精确通过才暴露可提交 constraint。任何未知、能力缺失、vote 分歧或 payload 不完整均 DEFER。当前实验没有修改真实 baseline。

### 11.2 盲测协议

- 5 cases，每例 3 votes，共 15 个有效调用。
- 每轮最多 5 个并发 API slot，并轮换 credential slot。
- 这是同一推理服务的多次独立采样，不声称是五个独立模型。
- prompt 中没有 error type、human label、expected action 或最终 ownership。
- API key 只在进程内存使用，没有写入输出、文档或 git。
- 第一次批量 TTY 输入因 framing 混合出现一次 401 INVALID_API_KEY；未生成结果、未计入 15 次有效调用。之后改为逐条发送，15/15 得到有效结构化结果。

协议文件：

inference_protocol.json
SHA-256：96cde2baaaca57f6081fa303c2c181c54c5ddb984f6590c8e73cbf87a9f6fe0a

blind_generation.json
SHA-256：e7664cd6f910e13ab5fde99821708bec218e724a00acf0a58ed6fbcc26d06656

posthoc_evaluation.json
SHA-256：61e5fcfe68f7d5933fa3e57d4f4b2a8e9792428768aa02004001e17c8ec63e9e

### 11.3 逐例结果

| Case | 三次 vote | Strict aggregate |
|---|---|---|
| office FS | 3 × DEFER | DEFER |
| room outlet FM | 3 × DEFER | DEFER |
| room table-part FM | 3 × DEFER | DEFER |
| geometry holdout | 3 × RESTORE_OBSERVATION_GEOMETRY | 因无 executor/evaluator 而 DEFER |
| semantic holdout | 2 × RELABEL refrigerator，1 × DEFER | 分歧且标签错误，DEFER |

### 11.4 指标

- vote-level action-family accuracy：5/15 = 0.3333。
- strict aggregate action accuracy：1/5 = 0.20。
- relaxed majority action accuracy：2/5 = 0.40。
- identity strict accuracy：0/3。
- capability strict accuracy：1/2。
- strict payload correct：1/5。
- semantic wrong-label votes：2。
- commit eligible：0。
- unsafe commits：0。
- final defers：5。
- repairable identity commits：0。

最重要的双重结论：

- 安全性方向正确：错误语义标签没有进入 commit，0 unsafe commit。
- 有效性明显不足：identity 案例没有生成可执行修复，不能用于自动生产。

## 12. 代码、测试与审计

主要新增或修改：

- conceptgraph/revision/identity.py
- conceptgraph/revision/constraints.py
- conceptgraph/revision/sparse_replay.py
- conceptgraph/revision/partition.py
- conceptgraph/revision/auto_constraints.py
- conceptgraph/revision/vlm.py
- conceptgraph/slam/utils.py
- tests/test_revision_identity_v2.py
- tests/test_revision_partition_observation.py
- tests/test_revision_auto_constraints.py
- scripts/formalize_revision_identity_contracts.py
- scripts/run_revision_identity_v2_global_validation.py
- scripts/freeze_revision_v2_holdouts.py
- scripts/build_revision_relation_gold.py
- scripts/build_revision_v2_blind_generation_manifest.py
- scripts/run_revision_v2_auto_constraint_generation.py
- scripts/evaluate_revision_v2_auto_constraint_generation.py
- scripts/design_partition_observation_candidate.py

验证结果：

- revision 相关测试：98 passed，2 warnings。
- threshold semantics 的两个直接测试：2/2 PASS。
- threshold 测试未能在混合 pytest 环境中收集，是因为 cg-main 的旧 OpenAI 依赖遮蔽 cg-ali，且 /opt 环境缺少 open3d；在真实 cg-ali runtime 中直接执行通过。该问题属于测试 harness 依赖冲突，不应伪装成测试失败或逻辑通过的额外证据。
- runtime oracle-leakage audit：0 violations。
- git diff check：PASS。
- API-like secret 扫描：未发现 key 被写入源码、测试或文档。

## 13. 当前框架成熟度

已经具备：

- 可审计的 provenance。
- 可修订但不污染来源的 effective identity。
- pair-specific negative identity boundary。
- FS persistent redirect。
- FM association 与 postprocess 双层保护。
- no-op、合法 merge、消融、parity 负控。
- observation partition 的可验证数据契约。
- fail-closed 自动生成和 shadow gate 骨架。
- 基于真实 human error 的条件性因果证据。

仍不具备：

- 对全部 40 个错误类型的统一执行能力。
- RELABEL、RESTORE_OBSERVATION_GEOMETRY 的正式 executor 与 evaluator。
- f74cb 的人类 point-level partition gold。
- 足够规模的 relation gold。
- 跨场景、跨数据集、跨阈值的统计泛化证据。
- 可用的自动 identity constraint generator。
- 大规模重复运行后的稳定延迟与显存统计。

## 14. 最优下一步

### P0：不要降低自动生成 gate，先提升 identity evidence

当前 3 个 identity cases 全部 DEFER，根因不是 gate 太严，而是盲输入只提供局部结构摘要，缺少可判别身份的注册多视图、3D 邻域、时间连续性和冲突轨迹。

下一步应为每个候选 pair 构造：

- registered multi-view crops；
- 3D overlap / separation 与尺度变化；
- temporal co-visibility；
- merge-chain provenance；
- competing target 排名；
- 能定位到 frame / object / point 的证据引用。

然后仍以 strict unanimity 和 shadow replay 为提交条件。

### P0：优先实现 RESTORE_OBSERVATION_GEOMETRY

几何 holdout 的 3/3 votes 都识别出正确 action family，这是当前自动生成中最强的可利用信号。应先实现：

- raw mask / processed mask 双版本绑定；
- observation payload hash；
- restoration executor；
- source immutability；
- endpoint 与 collateral geometry evaluator；
- 新的未见 holdout。

已有 geometry holdout 已被本轮分析消费，不能继续当最终测试集。

### P0：为 f74cb 获取真实 point gold

最小标注是 1990 个点的二分 assignment，或可无损映射到这些点的 mask。标注完成后：

- 冻结 assignment hash；
- 执行 PARTITION_OBSERVATION；
- 验证 exhaustive/disjoint；
- 做 Natural / Sparse endpoint 对照；
- 检查子对象 provenance 与后续 merge boundary；
- 再选新 partition holdout。

不应使用 DBSCAN 输出直接作为 gold。

### P1：补 table-part FM global parity

现有 1 FM + 1 FS 已满足本轮要求，但第二种真实 FM 具有不同几何和 merge chain，应补一次 global parity，确认不是 outlet 特例。

### P1：扩 relation gold

从当前 44 个判定扩到：

- 多 relation type；
- hard negative；
- predicted-only triples 的人工抽样；
- object identity 修复前后对照；
- macro 与 per-type 指标。

### P1：新增真正外部 holdout

在完成相应 executor 后，从未用于设计或 prompt 调整的新场景冻结 holdout。当前 2 个 capability holdout 只能保留为开发集。

### P2：性能与规模

在方法稳定后再扩到更多场景。建议先做每个代表 case 3 次重复，报告 median、p90、GPU peak memory 与 constraint-stage overhead；不应现在一次性跑全量，避免方法仍调整时浪费 GPU。

## 15. 可复现审计清单

Tracked manifests：

- docs/revision_v2_audits/HUMAN_CONFIRMED_SIX_IDENTITY_CONTRACT_MANIFEST_20260824.json
- docs/revision_v2_audits/IDENTITY_PROVENANCE_NEGATIVE_CONTROLS_20260824.json
- docs/revision_v2_audits/CAPABILITY_HOLDOUT_MANIFEST_20260824.json
- docs/revision_v2_audits/ROOM0_SMALL_RELATION_GOLD_20260824.json
- docs/revision_v2_audits/AUTO_CONSTRAINT_BLIND_INPUT_MANIFEST_20260824.json
- docs/revision_v2_audits/PARTITION_OBSERVATION_F74CB_DESIGN_20260824.json

主要实验：

- human6_formal_identity_v2
- component_ablation
- local_global_parity
- legal_merge_global
- negative_controls
- relation_gold
- auto_constraint_generation

## 16. 最终判断

V2 identity/provenance kernel 已经通过当前最关键的最小因果闭环：

- 真实错误上的 Native / Natural / Sparse 严格对照；
- 必要组件消融；
- no-op 与合法 merge 反例；
- local/global parity；
- provenance 不变性；
- GPU 全局回放；
- fail-closed 自动生成门控。

因此可以把它视为“当前 3 个可回放 human-confirmed identity errors 上成立的研究原型”。

但自动 constraint generation 仍应标记为“安全但低召回、不可生产”。接下来最有价值的投入不是扩大盲目全量运行，而是增强 identity evidence、实现 geometry restoration、取得 point-level partition gold，并在方法冻结后用新的外部 holdout 做一次真正的泛化评估。
