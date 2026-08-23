# ALI-MY Revision Kernel 实施与验证总结（2026-08-22）

## 1. 最终结论

本轮在服务器上完成了 Revision Kernel v0 的实现、真实受控注入、oracle 局部回放、全量时间回放参考、final-member refusion 消融、baseline relation closure、事务安全验证、跨场景局部稳健性与五路 VLM 盲测。

总体结论为：**核心 Revision Kernel 通过，VLM 自动修复能力尚未达到生产可用标准，真实关系恢复的统计结论受空边数据限制。**

具体分级如下：

| 模块 | 结论 | 说明 |
|---|---|---|
| revision-disabled parity | PASS | 关闭功能后对象、成员、对象 JSON、边拓扑和逐帧计数均与冻结基线一致 |
| 证据索引与 lineage | PASS | 3,779 个 association observation、6,095 个 mapping event、6,119 个 object version 可闭环解析 |
| observation 精确物化 | PASS | 3,779/3,779 通过，零近似、零异常 |
| 三类 oracle local replay | PASS | room0 与 office0 共 6 个案例均从受损状态恢复到成员 F1=1.0 |
| local vs global reference | PASS | room0 三例成员 F1 和 bbox IoU 逐例一致；局部耗时仅为全局的 0.22%–1.25% |
| V1–V9 / shadow transaction | PASS | 所有 room0、office0 事务均提交，闭包外变化为 0，源账本 hash 不变 |
| baseline relation closure | STRUCTURAL PASS / EMPIRICAL N/A | 真实 200 帧边输入为 0；非空合成测试通过原 `process_edges` 路径 |
| VLM constraint generator | NOT READY | 原始严格动作 1/3；语义编译后 2/3，另 1/3 安全 DEFER；盲门控后错误自动提交为 0 |

因此，当前可以支持的严谨表述是：

> 在冻结 evidence/version ledger 上，已证明“因果追踪 → dependency-local counterfactual replay → baseline edge closure → V1–V9 shadow commit”可实现、可重复，并在三类 controlled corruption 上与昂贵全局时间回放参考一致。

当前不能支持的表述是：

> VLM 已能稳定、自动地给出全部正确修复约束；或真实 relation 恢复已经在非空关系数据上得到统计验证。

---

## 2. 服务器、代码与实验基线

### 2.1 服务器资源

- CPU：224 logical CPUs
- RAM：约 503 GiB
- GPU：8 × NVIDIA RTX 5880 Ada 48 GiB
- 数据盘：约 23 TiB 可用
- 本轮仅使用空闲的物理 GPU 5；没有中断或占用其他用户的 GPU 作业
- 三条全局参考曾并行使用约 100 个 CPU 核；发现服务器总 load 有一定超额后，没有继续叠加新计算进程

### 2.2 独立工作树

- 工作树：`/home/chenkejun/beauty/conceptgraphs/code/official/ali-my-revision`
- 分支：`exp/ali-my-revision-kernel-v0`
- 基线：`bff233ff004939d2ecf4ac5546f87cb7b7b16e60`
- 实施提交：`ec18ce3`（`feat: add evidence-backed revision kernel validation`）

原 `ali-dev`、`ali-my`、`ali-my-VLM` 工作树未被原地修改。本轮采用当前 `bff233f` 作为代码基线，因为它包含计划所引用旧提交之后的 audit/parity/frozen-evaluation 修正；随后用冻结输出完成 revision-disabled parity，防止“换基线后悄悄改变算法”。

### 2.3 主数据

room0 正式 evidence run：

`/home/chenkejun/beauty/conceptgraphs/data/Replica/room0/exps/ali_my_validity_room0_full_200f_e6b0f17_20260820`

关键规模：

| 项目 | 数量 |
|---|---:|
| raw observations | 6,303 |
| kept / associated observations | 3,779 |
| mapping events | 6,095 |
| object versions | 6,119 |
| final objects | 72 |
| recorded postprocess merges | 24 |

office0 次级稳健性 run：

`/home/chenkejun/beauty/conceptgraphs/data/Replica/office0/exps/ali_my_validity_office0_full_200f_20260820`

所有 clean evidence 只读；live corruption 写入了新的独立输出目录。

---

## 3. 实施内容

本轮新增约 4,612 行代码与测试，核心模块如下：

1. `ProvenanceIndex` / `LineageIndex`
   - 校验并索引 observation、association、mapping event、object version、pair decision 和 final membership；
   - 支持 version parent/child DAG、ancestor/descendant、current entity 与 revision redirect；
   - 对六个核心源文件计算 SHA-256。

2. `ControlledCorruptionController`
   - 支持 `FORCE_CREATE`、`FORCE_ASSOCIATE` 和 postprocess merge 接口；
   - 注入位置位于原 matcher 决策之后、evidence 记录和 merge 之前；
   - 必须恰好命中一次，否则 run 终止；
   - 新增跨 run 稳定 observation key 和 origin-observation selector，避免时间戳 run ID、随机 object UUID 改变后计划失效。

3. `ObservationMaterializer`
   - 从 evidence ref 精确恢复 processed mask、点云 points/colors、CLIP feature、类别、2D/3D bbox 与 observation UID；
   - 缺失 payload 时硬失败，不进行近似补偿；
   - 使用原 `merge_obj2_into_obj1` 重建对象。

4. `CausalTracer` / `DependencyClosure`
   - 从注入 association event 找到 mapping event、version descendants、受影响 observation、后续依赖事件和 incident edges；
   - 闭包以 correctness 优先，允许保守扩张；
   - 支持闭包外对象前后 hash 比较。

5. `CounterfactualReplayEngine`
   - dependency-local replay：仅物化受影响 identity 的 observation，按历史时间顺序运行原融合/去噪路径并遵守 repair constraint；
   - full temporal global reference：从头处理 3,779 个 observation，调用原 spatial/visual similarity、aggregate、matcher、merge、denoise、filter 和 periodic merge；
   - `FINAL_MEMBER_REFUSION_BASELINE`：只按最终成员重新融合，不重放历史，用于消融。

6. `AliDevBaselineRelationBackend`
   - 仅包装未修改的 `ali-dev process_edges`；
   - node replay 为 local，edge replay v0 为 global baseline reconstruction；
   - 校验 endpoint、self-loop、tuple 结构和 relation type，不引入新关系算法。

7. `StructuralVerifier` / `ShadowTransactionManager`
   - 实现 V1 membership ownership、V2 obs 可解析、V3 PCD 有限非空、V4 bbox 有限非退化、V5 version source 有效、V6 edge endpoint active、V7 无意外 self-loop、V8 闭包外不变、V9 源 artifact 不变；
   - 只允许 `baseline → shadow → derived`；任一硬约束失败即 ABORT。

8. Maturity / conflict / ticket / VLM
   - 保留 raw maturity signals；
   - 实现四类 online conflict 与有限 rebase；
   - 实现 RepairTicket 状态机；
   - VLM 只生成类型化 constraint，不直接 patch final map；API key 只读入内存，不写入参数、环境文件或结果。

---

## 4. 硬门槛验证

### 4.1 revision-disabled parity

120 帧 room0 新 run 与冻结 `revision off` 基线比较：

| 检查 | 结果 |
|---|---|
| object count equal | PASS |
| canonical membership equal | PASS |
| object fields within tolerance | PASS |
| full object JSON equal | PASS |
| edge topology equal | PASS |
| parity trace present | PASS |
| per-frame counts equal | PASS |

结果：`PASS_REVISION_DISABLED_PARITY`。

结果文件：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/parity/revision_disabled_parity.json`

### 4.2 全量 observation 物化

| 指标 | 结果 |
|---|---:|
| observation count | 3,779 |
| passed | 3,779 |
| failed | 0 |
| elapsed | 27.40 s |

逐 observation 检查 point count、PCD points exact、bbox center/extent tolerance、CLIP exact、class 与 obs UID。一项不满足即失败。

结果文件：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/materialization_fidelity.json`

### 4.3 live controlled corruption

运行了独立 200 帧 live false-merge run：

`/home/chenkejun/beauty/conceptgraphs/data/Replica/room0/exps/ali_my_revision_live_false_merge_200f_bff233f_20260822`

结果：

| 项目 | 结果 |
|---|---:|
| planned observation | old run `f000108_r0024` |
| actual observation | new run `f000108_r0024` |
| injection count | 1 |
| original decision | CREATE |
| corrupted decision | FORCE_ASSOCIATE to active object index 66 |
| clean/corrupt evidence integrity | both PASS |
| affected observations | 75 |
| live corrupted member F1 | 0.973864 |
| over-merge | 1 |
| over-split | 1 |
| duplicate | 0 |

这证明注入钩子不仅写日志，而且实际改变最终结构；跨 run ID 和随机 UUID selector 也真实通过。

结果文件：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/live_false_merge_comparison.json`

---

## 5. room0 核心实验

### 5.1 Membership 与效率

| Case | Corrupted F1 | Local F1 | Global F1 | Local ms | Global ms | Local/Global | Event fraction | Transaction |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| false merge | 0.669250 | 1.000000 | 1.000000 | 1,912.03 | 543,078.62 | 0.3521% | 1.9847% | COMMITTED |
| false split | 0.980648 | 1.000000 | 1.000000 | 1,260.45 | 564,834.76 | 0.2232% | 2.0111% | COMMITTED |
| wrong membership | 0.983699 | 1.000000 | 1.000000 | 6,972.92 | 559,976.19 | 1.2452% | 3.2019% | COMMITTED |

汇总：

- 三例全部 PASS；
- mean corrupted F1：0.877866；
- mean repaired F1：1.000000；
- mean local/global runtime ratio：0.006068，即局部平均约使用全局 0.61% 的时间；
- 三例 speedup 约为 284×、448×、80×；
- local 与 global 的受影响 membership F1 差均为 0；
- local 与 global 的受影响 bbox IoU 差均为 0；
- 所有 V1–V9 检查通过，outside-closure changed entity 为 0。

主结果：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/revision_metrics.json`

### 5.2 Geometry 与 refusion 消融

| Case | Local bbox IoU | Global bbox IoU | Local center error | Refusion center error | Global center error |
|---|---:|---:|---:|---:|---:|
| false merge | 0.982722 | 0.982722 | 0.000559 | 0.002111 | 0.000559 |
| false split | 1.000000 | 1.000000 | 0.000000 | 0.000046 | 0.000000 |
| wrong membership | 1.000000 | 1.000000 | 0.000000 | 0.000514 | 0.000000 |

解释：

- final-member refusion 在这三例的 membership 上也达到 1.0，因此论文不能夸大“只有历史 replay 才能修回成员关系”；
- 但 refusion 的几何中心误差逐例高于历史 replay；false merge 中约高 3.8 倍；
- local replay 与 global temporal replay 的几何逐例一致，支持历史顺序与周期处理的计算价值；
- 当前差异量级较小，论文应将其报告为几何一致性增益，而非夸张为数量级精度提升。

### 5.3 可重复性

false-merge local replay 连续运行两次：

| 检查 | 结果 |
|---|---|
| membership equal | PASS |
| point digest / bbox state equal | PASS |
| decision trace equal | PASS |

两次耗时分别为 2,587.80 ms 与 1,515.51 ms，差异来自缓存热身；状态完全一致。

结果文件：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/determinism_false_merge.json`

---

## 6. office0 跨场景局部稳健性

office0 仅运行 local/refusion，不运行昂贵 global reference，因此全局耗时必须标为 N/A。

| Case | Corrupted F1 | Local F1 | Local ms | Transaction |
|---|---:|---:|---:|---|
| false merge | 0.667305 | 1.000000 | 10,767.29 | COMMITTED |
| false split | 0.990164 | 1.000000 | 2,889.38 | COMMITTED |
| wrong membership | 0.992048 | 1.000000 | 22,682.73 | COMMITTED |

汇总：

- 3/3 PASS；
- mean corrupted F1：0.883172；
- mean repaired F1：1.000000；
- 三例 outside-closure change 均为 0；
- 三个事务均 COMMITTED。

结果目录：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822_office0`

该结果说明局部内核并非只对 room0 单一案例有效，但由于没有 office0 global reference，不能用它扩展 local-vs-global 的效率/等价性结论。

---

## 7. Relation closure 结果与限制

room0 正式 run 的配置为 `make_edges=false`，且 200 帧 cached edge 输入均为空：

| 指标 | 结果 |
|---|---:|
| frames replayed | 200 |
| input edge observations | 0 |
| output edges | 0 |
| dangling edges | 0 |
| self loops | 0 |
| novel relation types | 0 |
| structural validation | PASS |
| statistically informative | No |

因此 empty→empty 的 precision/recall=1 不能作为 relation accuracy 证据。

为验证代码路径，另做非空合成回归：两帧 `A -on→ B` 输入均通过未修改的 `process_edges`，最终 relation 为 `on`、support=2，endpoint 有效，无 self-loop、无新 relation type。该测试证明包装器语义没有被新方法替换，但不替代真实非空关系数据评估。

准确表述：

> Node replay: dependency-local. Edge replay v0: global baseline reconstruction using unchanged ali-dev logic. Real relation stream in this validation run is empty and non-informative.

---

## 8. 五路 VLM 盲测

### 8.1 协议

- 五个 API key 通过无回显输入进入内存；未写入命令参数、代码、环境文件或结果；
- 五路请求并行；
- 模型：`gpt-5.6-sol`；
- prompt 不包含 failure type、final membership、source identity UUID 或 oracle constraint；
- oracle 仅在模型返回之后做评分；
- 共约 27,888 tokens；单请求 8.39–14.65 s；
- repo 与实验结果目录执行 `sk-` 搜索为零命中。

### 8.2 原始动作

| Case | Votes | Raw action | Strict expected | Raw exact |
|---|---:|---|---|---|
| false merge | 2 | DEFER | SEPARATE_MEMBER_GROUPS | No，但为安全 abstention |
| false split | 1 | SAME_INSTANCE | SAME_INSTANCE | Yes |
| wrong membership | 2 | SAME_INSTANCE(ANCHOR, CANDIDATE_1) | MOVE_OBSERVATION | No（类型不直接一致） |

原始严格动作准确率：1/3。

### 8.3 不看 oracle 的类型编译

规则：若 observation 当前已归属 `CURRENT_ENTITY_CONTEXT`，而模型声明 `ANCHOR` 与某个 alternate candidate 为 `SAME_INSTANCE`，则机器可执行语义是把 anchor 从 current 移到该 candidate，即编译为 `MOVE_OBSERVATION`。该规则只使用当前状态与模型 aliases，不使用 oracle。

编译后：

| Case | Compiled action | Correct | Blind auto-commit gate |
|---|---|---|---|
| false merge | DEFER | 安全不动作 | blocked |
| false split | SAME_INSTANCE | Yes | allowed |
| wrong membership | MOVE_OBSERVATION | Yes | allowed |

- compiled action accuracy：2/3；
- safe abstention rate：1/3；
- blind gate 下事后错误自动提交数：0。

结论：

- 当前 VLM 能处理部分视觉明确的 identity incident；
- false merge 证据不足时能正确选择保守 DEFER，这是安全优点；
- 但 3 个案例远不足以证明泛化，且 raw action schema 仍需更严格；
- 当前不能宣称 VLM constraint generator 已达到自动修复所需的召回率与校准水平。

结果文件：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/vlm_constraint_results.json`

---

## 9. 测试与代码审计

### 9.1 测试

- `/opt/anaconda3`：`78 passed, 1 skipped`；
- skip 项为依赖 Open3D 的非空 relation 测试，已在 `cg-ali` 环境中以相同断言直接通过；
- `test_general_utils.py` 的 2 项测试因 system Python 缺 `supervision` 无法在同一 pytest collection 中运行，已在含完整 mapping 依赖的 `cg-ali` 环境直接通过；
- 有效总计：81 项验证通过。

这是服务器已有“pytest 在 system env、mapping 依赖在 cg-ali env”的环境拆分，不是本次实现引入的测试失败。没有为得到绿色结果而删除或弱化测试。

### 9.2 静态检查

- `git diff --cached --check`：PASS；
- Python compileall：PASS；
- revision-disabled live parity：PASS；
- 源 artifact SHA-256 在每个事务前后相同；
- API key 搜索：零命中。

### 9.3 提交后工作树

提交后无 tracked diff。仅保留两个运行生成且未纳入提交的 untracked 项：

- `conceptgraph/scannet200_classes_colors.json`
- `latest_pcd_save`

未删除它们，以避免误删可能被后续实验复用的生成资产。

---

## 10. 已知限制

1. 核心定量矩阵仍是 controlled corruption，不是人工标注的真实线上错误分布。
2. room0 只有每类 1 个 global reference；office0 只有 local，不足以统计尾部失败率。
3. local replay 的 constraint-partition policy 在 oracle 实验中使用已知正确约束；VLM 与 oracle 必须继续分开报告。
4. 真实 relation 输入为空，不能给出 relation precision/recall 的有效统计结论。
5. final-member refusion 在 membership 上与 local replay 同样达到 1.0；历史 replay 的优势主要体现在与 global reference 一致的几何状态，而非这批案例的成员 F1。
6. VLM 样本数仅 3 个 incident、5 个 votes；2/3 compiled accuracy 不能外推为生产准确率。
7. online conflict/rebase 与 RepairTicket 已做单元测试，但尚未进行长时并发 live stress test。
8. 尚未加入队友的新 relation backend，符合本阶段“不换整体方向、不引入新关系算法”的边界。

---

## 11. 下一步建议

按优先级建议：

1. 生成或选择至少一个 `make_edges=true` 且真实非空的 formal evidence run，补做 relation recovery 的非空定量验证。
2. 把 controlled matrix 扩到至少 5 个场景、每类 20–50 个案例，并按对象大小、anchor 时刻、margin、遮挡程度分层。
3. 保留 oracle local replay 作为 executor upper bound；VLM 只比较 constraint quality，不混用 oracle grouping。
4. 强化 VLM schema：区分 observation-to-entity 与 entity-to-entity 的 `SAME_INSTANCE`，把类型编译规则写入正式 JSON Schema。
5. 对 VLM 做校准曲线、abstention-risk 曲线与错误自动提交率评估；在样本扩大前保持 `DEFER` 优先。
6. 对 conflict/rebase 做多线程或事件流 stress test，验证 max-rebase、lineage redirect、hypothesis invalidation 和 source-hash gate。
7. 论文中把 refusion 作为强消融保留，并把几何增益按真实数值报告，不夸大。

---

## 12. 主要结果路径

代码：

- `/home/chenkejun/beauty/conceptgraphs/code/official/ali-my-revision`
- branch：`exp/ali-my-revision-kernel-v0`
- commit：`ec18ce3`

room0：

- `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/revision_metrics.json`
- `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/revision_report.md`
- `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/materialization_fidelity.json`
- `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/live_false_merge_comparison.json`
- `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/determinism_false_merge.json`
- `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822/vlm_constraint_results.json`

office0：

- `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822_office0/revision_metrics.json`
- `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822_office0/revision_report.md`

每个 room0 case 目录均包含：

- `case.json`
- `trace.json`
- `dependency.json`
- `corruption.json`
- `constraint.json`
- `transaction.json`
- `verification.json`
- `before_after_summary.json`
- `edge_rebuild_summary.json`
- `metrics.json`
- `branches/{clean,corrupted,final_member_refusion,local_replay,global_replay}.json`

最终总结服务器路径：

`/home/chenkejun/beauty/ALI_MY_REVISION_KERNEL_VALIDATION_SUMMARY_20260822.md`

---

## 13. 最终判定

### 可以进入下一阶段

- Revision Kernel executor：是；
- controlled-validation 扩样本：是；
- 非空 baseline relation run：是；
- conflict/rebase live stress：是。

### 暂不应直接上线

- VLM 无人工/门控直接自动提交：否；
- 对真实 relation accuracy 做强结论：否；
- 用当前 3+3 个案例宣称广泛泛化：否。

本轮最重要的结果不是“模型看起来会修”，而是把修订过程从不可验证的 final-map patch，变成了可追踪、可回放、可比较全局参考、可被硬约束拒绝、且不改写源证据的事务化执行路径。
