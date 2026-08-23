# ALI-MY Revision Kernel V1 实现与验证总结

> 日期：2026-08-23
> 分支：`exp/ali-my-revision-kernel-v1`
> 基线：`900f117557b9fea2e0924165b5e98917bc88afd9`
> 开发场景：Replica `room0`、`office0`
> 服务器实验根：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823`

## 1. 最终结论

本轮完成了 Revision Kernel V1 的 Phase 0–3 核心链路：将 V0 的“按 clean final membership 强制整条轨迹”改成“单个历史错误事件 + 稀疏约束 + anchor 前快照 + 自然 suffix replay”，并把生产运行时与 benchmark oracle 严格隔离。

结论是 **有条件 GO**：

- 冻结的两场景主矩阵共 60 例全部完成；其中 49 例在一次历史干预后产生了可测最终损伤，49/49 均由稀疏 replay 恢复到参考成员划分和几何状态，room0 的 24 个受损案例同时恢复了有信息量的关系状态。
- 另外 11 例在自然 suffix 演化中自行恢复，保留在冻结总体中，但没有被计作修复成功。
- 60/60 最终状态均通过全观测并集上的对称 collateral gate，没有候选侧新增、遗漏、重复归属或错误重分区。
- 预先冻结的同约束 global reference 为 6/6 精确；独立 live mapper 与 temporal simulator fidelity 为 6/6 精确。
- room0、office0 的自然 global clean replay 均逐项复现冻结基线；room0 的 49 条有向关系边也精确一致。
- 生产运行时五个文件的 clean/GT/oracle 禁止字段静态审计为 0 违规；服务器最终回归为 `128 passed, 1 skipped`，Revision 专项为 `65 passed, 1 skipped`。

但当前证据**不支持**以下更强结论：blind root-cause localization、VLM 自动约束质量、clean-negative harm rate、在线 rebase/真实提交、六场景 holdout 泛化，以及冷启动端到端加速。这些均没有被本轮结果包装成已完成能力。

## 2. 为什么没有把所有场景和昂贵案例一次跑完

本轮采用“先完成最有判别力的最小矩阵，再由分歧触发扩展”的策略：

| 层级 | 实际规模 | 选择理由 |
|---|---:|---|
| Primary controlled | 2 场景 × 3 类 × 10 = 60 | 完成指导中最高优先级的 room0/office0 每类至少 10 例，足以检验去 oracle 后的核心方法 |
| Global same-constraint | 2 场景 × 3 类 × 1 = 6 | 每类每场景冻结 1 例；仅出现真实 local/global 分歧时才扩展 |
| Live fidelity | room0 共 6 例：4 merge、1 split、1 wrong-membership | 全部来自原始 outcome-blind 10 例清单；在 fidelity 已逐项精确后停止继续烧算力 |
| Outcome-selected diagnostic | 1 | 只用于定位 F33，明确排除在主结果和 runtime 汇总外 |
| Holdout / VLM / online / clean negative | 0 | 当前先稳定 executor 语义；过早扩张会在后续方法调整时重复昂贵实验 |

Primary、global 和 live 的正式清单都在看到对应结果前冻结。F33 诊断例是结果驱动的调试，不替换任何正式案例，也不进入 headline 计数。

## 3. 主要实现

### 3.1 Production 与 benchmark 隔离

- `conceptgraph/revision/constraints.py`：定义稀疏约束、冲突语义与显式 replay mode。
- `conceptgraph/revision/sparse_replay.py`：执行 temporal corruption、natural replay、anchor-only 和 persistent sparse replay；运行路径不读取 clean final owner。
- `conceptgraph/revision/benchmark/`：集中放置 oracle sparse constraint 编译、冻结采样和 clean-reference evaluator。
- `conceptgraph/revision/runtime_verify.py`：只检查运行时可获得的不变量；clean-reference 恢复质量不参与 runtime commit gate。
- `scripts/audit_revision_oracle_leakage.py`：以 AST 检查变量、属性、参数、关键字参数和 mapping string key，防止重构后绕过隔离。

### 3.2 真实 one-event temporal corruption

Primary corruption 不再直接篡改 final membership，而是在一个真实 association event 上只覆盖一次动作：

- `FALSE_SPLIT`：原 merge 改为 create；
- `FALSE_MERGE`：原 create/非目标动作改为错误 target；
- `WRONG_MEMBERSHIP`：原 target 改为可行但错误的 target；
- 此后所有 observation 继续使用 mapper 的自然 suffix 演化。

每个案例保留原生动作、注入历史动作、约束动作和实际动作四个层次，避免把“记录过的错误历史”误当成“native natural decision”。

### 3.3 Anchor 前快照与严格 suffix replay

- `conceptgraph/revision/snapshot.py` 重建 anchor watermark 之前的局部状态；同一帧内只包含 anchor 前的 detection/event。
- 每个 dependency seed 必须解析成功，禁止缺失 evidence、跳过 version 或近似补全。
- `conceptgraph/revision/dependency_graph.py` 使用显式 typed node/edge，不再通过 JSON 字符串子串推断依赖。
- suffix replay 从快照继续；当局部写入会破坏当前 head 的对象原子性时，执行有记录的 closure expansion，而不是静默写入不完整对象。
- first-divergence 工具在 local/global 或 clean parity 不一致时输出第一处分歧，而不是只看最终 pickle 猜原因。

### 3.4 评估器硬化

本轮针对“看起来全 1.0”主动增加了以下反例门：

- 成员划分使用双方 observation 并集，检测两侧遗漏和跨对象重复；
- 同一对象内部重复 observation 也会失败；
- relation row 重复不会再被字典索引静默覆盖；
- bbox 使用声明的绝对容差，不因近 1.0 浮点误差误判；
- 空 relation 明确标记为 noninformative，不能当作关系恢复证据；
- frozen manifest 复用时校验参数、路径、顺序和 case ID，禁止静默改变实验规模；
- same-frame audit 直接比较不可变 target-origin，不依赖截断的 top-10 调试候选。

## 4. 冻结 Primary 结果

### 4.1 总体

| 场景 | 冻结案例 | 产生最终损伤 | 受损案例精确修复 | 自愈且不计功劳 | 最终 collateral-safe | 有信息关系案例 |
|---|---:|---:|---:|---:|---:|---:|
| room0 | 30 | 24 | 24/24 | 6 | 30/30 | 30 |
| office0 | 30 | 25 | 25/25 | 5 | 30/30 | 0 |
| 合计 | 60 | 49 | 49/49 | 11 | 60/60 | 30 |

“精确修复”要求：corruption 确实造成至少一个声明维度的损伤、稀疏约束实际覆盖了注入的错误历史动作、runtime invariants 通过、成员/几何恢复，并通过全 observation 并集上的 collateral gate。自愈案例不进入 49 个成功分母。

### 4.2 按错误类型

| 类型 | 冻结案例 | 产生损伤 | 精确修复 | 自愈 |
|---|---:|---:|---:|---:|
| FALSE_MERGE | 20 | 20 | 20 | 0 |
| WRONG_MEMBERSHIP | 20 | 20 | 20 | 0 |
| FALSE_SPLIT | 20 | 9 | 9 | 11 |

FALSE_SPLIT 的 11 个非成功均是注入后自然 suffix 自愈，不是 crash、snapshot mismatch、invariant failure、constraint defer 或 collateral damage。

### 4.3 方法状态质量

| 场景 / 方法 | member F1 mean | bbox IoU mean | relation exact | relation informative |
|---|---:|---:|---:|---:|
| room0 corrupted history | 0.924899 | 0.917732 | 26.67% | 30/30 |
| room0 sparse local | 1.000000 | 1.000000 | 100% | 30/30 |
| office0 corrupted history | 0.956262 | 0.921212 | 100%* | 0/30 |
| office0 sparse local | 1.000000 | 1.000000 | 100%* | 0/30 |

`*` office0 当前 relation stream 为空，100% 只是 empty-set structural equality，不能作为关系恢复质量证据。room0 的 relation stream 非空，因此 room0 的 exact 才具有信息量。

按受损维度计数：

- room0：membership 22、geometry 17、relation 22；三个维度的 improved count 分别也是 22、17、22。
- office0：membership 20、geometry 20、relation 0；improved count 分别为 20、20、0。
- 两场景 damaging case 的 member/geometry recovery 中位数均为 1.0；room0 damaged relation recovery 均值为 1.0。

### 4.4 最重要的 ablation 限制

`natural_recompute_ablation` 在 60/60 案例中与 sparse replay 最终状态等价。这说明：

> 在已知 anchor、原始 evidence 和 matcher 本身没有被破坏的外部 override benchmark 中，稀疏约束足以纠正“保留错误历史动作”的分支，但当前实验没有证明该约束在重新运行 native matcher 后仍然是必要条件。

因此本轮可支持“sparse correction is sufficient relative to the corrupted historical branch”，不能写成“没有该 constraint 就无法恢复”或“系统已完成盲诊断”。

## 5. Locality 与 runtime

### 5.1 Primary locality

| 场景 | effective obs fraction p50 | p95 | max | closure event p50 | p95 | max | expanded cases / obs |
|---|---:|---:|---:|---:|---:|---:|---:|
| room0 | 3.96% | 8.46% | 10.19% | 2.19% | 5.63% | 6.47% | 2 / 31 |
| office0 | 8.69% | 16.03% | 16.15% | 5.05% | 12.42% | 13.03% | 1 / 131 |

closure expansion 没有被隐藏：room0 两例、office0 一例。扩展是为了保持当前 head entity 的原子性，不代表依赖回退为全局 replay。

### 5.2 Primary 运行时间

| 场景 | suffix p50 | p95 | max | snapshot cumulative p50 | cold upper-bound p50 |
|---|---:|---:|---:|---:|---:|
| room0 | 80.89 s | 141.54 s | 160.05 s | 104.89 s | 228.96 s |
| office0 | 46.59 s | 66.63 s | 73.69 s | 31.23 s | 94.18 s |

`snapshot cumulative` 是增量前缀缓存的累计构建时间；当前 artifact 没有分离每例新增 cache cost 与 same-frame cost，所以不能从它推出精确 amortized runtime。

### 5.3 六个正式 global reference

| 场景 | 例数 | member / geometry / relation exact | 有信息关系 | suffix local/global p50 / p95 / max | cold local/global p50 / p95 / max |
|---|---:|---:|---:|---:|---:|---:|
| room0 | 3 | 3 / 3 / 3 | 3 | 0.522 / 0.533 / 0.534 | 1.186 / 1.237 / 1.243 |
| office0 | 3 | 3 / 3 / 3* | 0 | 0.386 / 0.479 / 0.489 | 0.962 / 1.041 / 1.050 |

结论：suffix compute 确实低于 global replay；但计入当前非摊销 snapshot 构建后，room0 反而更慢，office0 也没有稳定的端到端优势。因此当前只能声称“dependency-bounded suffix work reduction”，不能声称“cold end-to-end acceleration”。下一次性能工作应先增加精确增量 cache timer，再决定是否值得扩场景。

## 6. Clean parity、materialization 与 relation 回归

### 6.1 Global clean replay

| 场景 | observations | objects | membership / payload / postprocess / decisions | relation | runtime |
|---|---:|---:|---|---|---:|
| room0 | 3,779 | 72 | 全部 exact | 49 条有向边 exact | 217.83 s |
| office0 | 1,560 | 29 | 全部 exact | empty/noninformative exact | 168.60 s |

room0 最终清洁检查还同时通过 bbox IoU ≥ 0.999、source hashes equal，`payload_mismatch_count=0`、`first_decision_divergence=null`。

### 6.2 Evidence materialization 与 make_edges 回归

- room0：3,779/3,779 observation 精确 materialize；
- office0：1,560/1,560 observation 精确 materialize；
- room0 非空关系流：200/200 frames、2,425 frame-level relation observations、49 final directed edges；
- ali-dev 与 ali-my 的 edge identity/support，以及 detection/process-edge/map-edge 三类来源均 exact。

这部分复用了已经完成 `make_edges=true` 的 ali-dev/ali-my 结果，没有重复跑无信息的边生成工作。

## 7. Live-vs-simulator fidelity

正式 staged-six 在以下五类 gate 上全部 6/6 通过：

1. 注入 event 与次数精确，且每例只有一次；
2. suffix 的 create/merge decision kind 精确；
3. target immutable origin 精确，mismatch 为 0；
4. UUID-independent final membership partition 与 raw object payload 精确；
5. denoise/filter/merge postprocess schedule 精确。

六例 member F1 均为 1.0，raw payload 比较覆盖 points、bbox、features、class histogram 和 member partition。另行 uniqueness audit 在六个 live map 中均发现：每个 observation 只有一个 PCD、对象内重复 0、跨对象重复 0。

限制：这六例仅来自 room0，且 `make_edges=false`，所以它验证 one-event mapper/simulator 语义，不提供 live relation fidelity 证据；关系正确性由独立 room0 make-edges 回归和 global reference gate 支撑。

## 8. F33：全绿之前真正发现并修复的语义错误

冻结案例 `room0/false_merge_f000015_r0011` 在最初实现中只到 member F1 `0.993216`，关系状态也不 exact；严格 collateral gate 正确地把它判为失败，而没有因接近 1.0 放行。

追踪结果：

- native anchor action 是 `CREATE_OBJECT`（target `None`）；
- 注入的错误历史 target 是对象 7；
- `CANNOT_LINK` 排除了对象 7；
- 旧实现却把“注入历史 target”作为 natural match，再选择了另一个 eligible 对象 12；
- 根因是 `KEEP_NATURAL` 与 `NO_CONSTRAINT` 的语义被混为一谈。

修复后：

- ConstraintEngine 接收真实 native match；
- `KEEP_NATURAL` 使用 native decision；
- `NO_CONSTRAINT` 才保留历史 branch；
- 只重跑受影响的冻结 F33 案例，其余 19 个 false merge 已是 `FORCE_CREATE` 且 reference state hash exact，不做无信息重复计算；
- 修复后 member、geometry、room0 informative relation 和全 3,779 observation partition 均 exact；
- 额外 outcome-selected global diagnostic 也 exact，但明确不计入正式六例。

旧错误 artifact 保留在：

`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/room0_primary_diagnostics_pre_f33/false_merge_f000015_r0011`

该案例说明本轮的 evaluator/audit 确实能够打破“接近完美但语义错误”的实现，而不是让所有结果机械变绿。

## 9. 审计与可复现性

- Primary selection：seed `20260823`；两个场景各 30，manifest missing=0、unexpected=0。
- Global selection：每场景每类型 1 例；六例均 outcome-blind，missing=0、unexpected=0。
- Live selection：六例全部可追溯到真正 pre-live 的原始十例 manifest；没有按 fidelity outcome 换案例。
- Same-frame batching audit：room0 比较 43 个决策、office0 比较 46 个决策，共 89；watermark crossing=0、decision/origin mismatch=0。
- Snapshot gate：60/60 通过；每个 requested seed 都解析，无 skipped version、无 approximation fallback。
- Runtime leakage audit：5 个 production runtime 文件，7 个禁止 identifier，0 violation。
- 人工 artifact review：每场景随机抽 5 个成功 + 5 个失败/自愈，共 20；逐项查看 incident、constraint、snapshot、dependency、corruption/replay trace、relation、runtime 和 benchmark metrics。
- 失败没有删除：34 项实现/审计问题保留在 `docs/revision_v1_audits/FAILED_RUNS_LEDGER.md`，包括被停止的过载 precheck、被缩减的 live scheduler 和 F33/F34。

## 10. GO / FIX / 未运行边界

| 能力 | 决策 | 当前证据 |
|---|---|---|
| 无 final-owner trajectory 的 sparse executor | GO | runtime AST 0 违规；49/49 damaging repair exact |
| one-event temporal simulator | GO（限 staged scope） | 独立 live 6/6 五类 gate exact |
| anchor 前 snapshot + suffix-local replay | GO | snapshot 60/60；same-frame 89/89；global 6/6 |
| 成员/几何恢复与 collateral safety | GO（两开发场景） | 49/49 damaging exact；60/60 collateral-safe |
| baseline relation state consistency | GO（room0） | 24 damaging primary + 3 formal global informative exact |
| 冷启动端到端加速 | FIX / 不成立 | room0 cold ratio p50 1.186；精确 amortized cost 缺失 |
| dependency-local relation replay | FIX / 未实现 | 当前关系重建仍扫描完整 frame stream |
| blind anchor localization | 未运行 | 使用 injected anchor；没有 Top-k/MRR 结果 |
| clean-negative harm / false mutation | 未运行 | 不能报告自动提交安全率 |
| 六场景 holdout 泛化 | 未运行 | office1–4、room1–2 保持未触碰 |
| online rebase / frame-boundary live commit | 未运行 | 当前是 event-stream executor，不是 live atomic commit |
| VLM constraint generation | 未运行 | API 未用于本轮 executor 隔离实验 |
| physical-world GT correctness | 未证明 | clean branch 是未注入 corruption 的 mapper counterfactual |

## 11. 测试口径

服务器最终执行：

```text
/opt/anaconda3/bin/python -m pytest -q tests --ignore=tests/test_general_utils.py
128 passed, 1 skipped

/opt/anaconda3/bin/python -m pytest -q tests/test_revision_*.py
65 passed, 1 skipped

/opt/anaconda3/bin/python scripts/audit_revision_oracle_leakage.py
pass=true, runtime_file_count=5, violations=[]
```

无 scope 的仓库根 `pytest` 会在收集阶段命中两个可执行 legacy utility（文件名符合 pytest pattern）及轻量环境中缺失 `supervision` 的 `test_general_utils.py`。这不是 V1 通过结果，失败命令已作为 F19 保留；没有通过安装无关重依赖来掩盖该限制。

## 12. 后续最合适的顺序

现在不建议立即跑更多场景。信息增益最高的下一步是：

1. 给 snapshot cache 增加“每例新增构建时间”和“same-frame prefix 时间”两个独立计时；先在 room0、office0 各选 1 个 damaging case 验证真实摊销成本。
2. 若方法语义不再变化，再做一个小 clean-negative gate，优先测 false mutation，而不是直接扩 150–192 个 positive case。
3. 只有前两项通过后再解冻 1–2 个 holdout 场景；不需要一次解冻六个。
4. blind anchor、online rebase 和 VLM 分别作为独立变量逐层加入，避免 executor、diagnosis 和 VLM 错误混在同一张结果表中。

## 13. 关键证据路径

- room0 primary：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/room0_primary`
- office0 primary：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/office0_primary`
- room0 global：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/room0_global_reference`
- office0 global：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/office0_global_reference_staged3`
- live staged-six：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/live_fidelity_room0_staged6`
- room0 clean parity：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/global_clean_parity_room0_final.json`
- office0 clean parity：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/global_clean_parity_office0.json`
- relation backend parity：`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/phase0_relation_backend_parity/backend_parity.json`
- 最终服务器报告：`/home/chenkejun/beauty/ALI_MY_REVISION_KERNEL_V1_IMPLEMENTATION_EVALUATION_20260823.md`
