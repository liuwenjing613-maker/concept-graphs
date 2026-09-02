# 实验 0 标注数据的多维结果分析与新修复机制挖掘

日期：2026-09-02
性质：只读、探索性分析；未修改生产默认逻辑，未启动完整场景重跑。
范围：Replica/room0 的 174 个独立标注事件、14 个 compiled error episodes、7507 条 association/校正 observation GT、对象版本历史，以及现有 B0/B0R/B1/B2/B3 回放结果。

## 1. 先给结论

1. **实验 0 暴露的并不只是“路由动作选错”，而是两级故障：上游 observation mask 把多个实例混在一起，下游 identity routing 又把这种污染持续吸收到对象中。** 一旦候选对象已经混合，单独执行 `FORCE_NEW` 或 `FORCE_TARGET` 只能改当前锚点，不能恢复完整物理实例。

2. **14 个错误锚点的当前 mask 看起来很好，不代表它们的因果根也干净。** 人工标注中 13/14 为 `CLEAN_SINGLE_INSTANCE`、1/14 为 `BORDERLINE_SINGLE_INSTANCE`，14/14 都足以判断当前身份；校正 GT 也没有把这 14 张锚点 mask 判成 mixed。但在 9 个“原动作涉及 ATTACH”的错误中，按成员身份与像素混合联合审计，原目标 9/9 都已经污染。人工 D 阶段在其中 5 例写成 clean，说明页面对历史对象完整性的支持不足，而不是当前锚点不可判。

3. **v2_r2_013 的真正起点不是 frame253。** 当前证据能定位到：

   - frame131：首张含两个 blinds 实例的像素混合 mask 被并入目标，是像素级污染起点；
   - frame138：混合 mask 中 GT19 约 48.98%、GT15 约 46.06%，其 observation-level top GT 首次从 15 翻成 19，是身份分叉点；
   - frame139：第二个 top-GT19 混合 observation 进入，形成持续的多 GT 成员污染；
   - frame253：只是 6323 个 association 事件之后被抽到并标出的晚期症状。

   因此 frame138 **没有单一的 NEW/ATTACH 正确答案**。正确修复语义是 `SPLIT_OBSERVATION_MASK` 或 `QUARANTINE_AND_RESEGMENT`：GT15 部分附回原目标，GT19 部分新建或进入临时身份簇。把整张混合 mask 直接 NEW 会同时污染新对象并错误搬走 GT15。

4. **v2_r2_013 的 B2/B3 endpoint 虽通过，但完整实例修复失败。** 新对象 16/16 observation 为 GT19，precision=1.0；然而 room0 中 GT19 共 115 条，它只收回 16 条，global observation recall=13.91%、F1=0.244。原目标仍有 236 条：129 GT15、99 GT19、8 GT21，按 observation 成员的残余污染率为 45.34%；像素联合审计仍判为污染。它是“高精度、低召回的新子簇”，不是整实例恢复。

5. **正确候选本身也经常不可直接接入。** 在 7 个正确动作应为 ATTACH 的错误中，只有 3 例存在某个 clean 合法候选；2 例所有合法候选都已污染，2 例合法候选仍不确定。即使“存在 clean 候选”，room0_large_r1_0089 的同一实例主要成员仍埋在另一个污染候选中，只连到那一个 clean 小片段也不能完成实例重聚。

6. **“一个 observation 标多个合法候选”确实是对象图过分割的重要信号，但不能无条件等同于过分割。** 154 个正确动作应为 ATTACH 的事件中，7 个标了 2–3 个合法候选；其中 5 个事件的至少两个候选都含当前 corrected-GT 实例，构成同一物理实例跨多个 object UID 的直接碎片化信号，对应 4 组唯一候选集合。另 2 个事件只有一个候选得到离线同 GT 支持，更像同类外观混淆、part-whole 歧义或人工多选，不应自动 MERGE。

7. **应把错误拆成至少五种修复原语，而不是统一 FORCE：**

   - 上游混合 mask：拆 observation、隔离、重分割；
   - 历史目标已污染且当前实例应 NEW：回溯最早根错误、拆污染目标、重放依赖闭包，并设置防重并边界；
   - 正确候选已污染：先拆候选中的目标身份 lineage，再重定向/重聚；
   - 正确候选不确定：抽取已有同身份成员到临时簇，延迟最终提交；
   - 目标确实 clean 且没有历史同身份残留：才允许直接 redirect/attach。

8. **优先级最高的不是立刻改生产阈值，而是先修评价契约和因果标注。** mixed observation 继续不计入“纯路由错误率”是合理的，但不能从因果账本中消失。应新增独立 `SEGMENTATION_ORIGIN` ticket，并让晚期 routing symptom 指向它；同时把 endpoint 正确升级为“端点 + 全实例 precision/recall + 残余污染 + 碎片化 + 闭包副作用”的联合门槛。

## 2. 证据、推断与设想的边界

- **[现有证据]** 来自冻结的 174 条人工标注、14 条 compiled episodes、校正 observation GT、7507 条 association、12142 个 object versions、final membership 与已经完成的回放产物。
- **[分析定义]** 本报告新增的“像素联合污染状态”、污染时间层、完整实例指标和修复原语分类，是对现有数据的只读派生，不是人工标签，也不是生产规则。
- **[推断]** “页面为何误判 clean”“哪种原语更合适”等，是由数据支持的机制推断，仍需最小实验验证。
- **[尚未验证设想]** 在线混合检测、自动重分割、临时身份簇和历史 rollback 尚未接入 mapper。下文所有阈值都是可证伪实验门槛，不是建议直接上线的默认值。
- 校正 GT、最终 owner 和未来视角只用于离线诊断或 oracle 上界；不得作为锚点时刻的在线输入。

## 3. 数据与可复现性

### 3.1 主要输入

- 标注协议：`github_error_label_worktree/experiments/experiment0_manual_annotation_20260901/PROTOCOL_V2_CN.md` 及同目录最新修订/执行文档。
- 174 个独立事件：`results/experiments/experiment0_manual_annotation_20260901/v2_large_room0_r1/analysis_20260902/episodes/combined_human_routing_events.jsonl`
- 14 个错误 episode：`.../episodes/human_error_episodes.jsonl`
- 正确离线 GT：`results/experiments/experiment0_manual_annotation_20260901/corrected_gt_audit_room0/observation_gt.jsonl`
- association、observation、object version 与最终成员：`results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/online_label_trigger_v1_room0_dev_pcd/evidence/`
- 路由审计：`results/experiments/experiment0_manual_annotation_20260901/identity_routing_v2_audit_room0/`
- 抽样清单：`results/experiments/experiment0_manual_annotation_20260901/v2_large_room0_r1/worklist_manifest.json`
- 回放：`.../analysis_20260902/oracle_minimal_replay_*` 与 `oracle_create_partition_audit/create_partition_audit.json`

旧 Habitat sidecar 与本次校正 GT 定义不一致，本分析没有混用。

### 3.2 复现命令与产物

服务器项目根目录：`/home/chenkejun/beauty/conceptgraphs`

```bash
python code/experiments/experiment0_multidimensional_analysis_20260902/analyze_experiment0_multidimensional.py
```

最终运行状态：`OK`；174 events、14 errors；只读统计耗时 4.055 秒。

机器可读输出目录：

`/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_multidimensional_analysis_20260902/`

其中：

- `cross_statistics.json`：交叉统计；
- `event_multidimensional_table.jsonl`：174 条事件级明细；
- `error_multidimensional_table.jsonl/.csv`：14 条错误明细；
- `error_candidate_compositions.jsonl`：候选成员、GT、类别、空间、时间和像素混合组成；
- `replay_full_instance_audit.json`：B0/B0R/B1/B2/B3 与全实例指标；
- `run_manifest.json`：输入路径、行数、mtime、定义和告警。按要求没有重复做无意义哈希检查。

## 4. 状态定义：不能只看 observation-level top GT

### 4.1 两层候选状态

| 状态层 | CLEAN | ALREADY_CONTAMINATED | UNCERTAIN |
|---|---|---|---|
| 成员层 | 有效成员只出现一个 observation-level top GT | 第二 GT 至少 2 条且跨至少 2 帧 | GT 覆盖不足或证据不完整 |
| 像素联合层 | 成员层 clean，且历史中没有 corrected-GT mixed/two-foreground mask | 成员层已污染，或至少 2 张混合 mask 跨至少 2 帧且累计非主导 top-2 GT 像素至少 5% | 存在像素混合迹象但未达到持续/占比门槛，或覆盖不足 |

5%、2 observation/2 frame 是本次探索性审计阈值，只用于把“12/12 top GT15 但每张都夹着 GT19”这种情况从 clean 中分离出来，尚未验证为生产阈值。

### 4.2 四个因果时间点

对污染对象至少要记录：

1. `first_pixel_mixed_mask`：第一张像素混合 mask 进入；
2. `first_persistent_pixel_contamination`：混合连续出现并达到材料占比；
3. `first_strict_multi_gt`：成员 top GT 首次分叉；
4. `first_persistent_multi_gt`：第二身份至少两条、跨两帧。

只找第 3 点会把 frame138 当根；加入像素层后，v2_r2_013 的根向前移到 frame131。对线上 ticket 仍需遵循 `s<=d<=h<=c`：例如该例可记 `s=131`，`d` 为后续发现/复核时刻，`h` 为冻结快照时刻，`c` 为结论落盘时刻。

## 5. 174 条与 14 条的联合统计

### 5.1 原动作 × 正确动作

| 原动作 | 正确 ATTACH | 正确 NEW | 不适用 | 合计 |
|---|---:|---:|---:|---:|
| ATTACH_EXISTING | 149 | 7 | 3 | 159 |
| NEW | 5 | 7 | 3 | 15 |
| 合计 | 154 | 14 | 6 | 174 |

14 个 compiled routing errors 为：`SHOULD_HAVE_BEEN_NEW` 7、`WRONG_ATTACH_EXISTING` 2、`WRONG_NEW_FALSE_SPLIT` 5。compiled 因果角色为 ROOT 7、CASCADE 6、PENDING 1。这里的 ROOT 只是在 174 条采样事件中的根，不保证是全 association ledger 的真实根。

### 5.2 当前 mask 与身份可判定性

| 指标 | 14 个错误 |
|---|---:|
| 人工 CLEAN_SINGLE_INSTANCE | 13 |
| 人工 BORDERLINE_SINGLE_INSTANCE | 1 |
| 人工 SUFFICIENT_FOR_IDENTITY | 14 |
| 校正 GT purity >=0.95 | 12 |
| 校正 GT purity 0.80–0.95 | 2 |
| 锚点 mask corrected-GT mixed/two-foreground | 0 |

结论只适用于“被抽中的当前锚点”。它不能反证更早历史 mask 是否混合，v2_r2_013 和 v2_r2_011 正是反例。

### 5.3 被选目标在错误前是否已经污染

对全部 159 个原 ATTACH 事件：

| 状态 | 只看成员 top GT | 像素联合 |
|---|---:|---:|
| CLEAN_SINGLE_INSTANCE | 109 | 75 |
| ALREADY_CONTAMINATED | 43 | 53 |
| UNCERTAIN | 7 | 31 |

该样本混合了概率样本、错误 harvest 和复核样本，不能当作全流 prevalence。

对 9 个原 ATTACH 且最终判错的事件：

- 成员层：8 污染、1 clean；
- 像素联合层：9/9 污染；
- 人工 D 阶段：4 污染、5 clean；这 5 个“人工 clean”在像素联合审计中全部已污染。

这是 error-conditioned 小样本，不能直接报告人工标注总体准确率；它足以证明现有 D 页面会系统性漏掉某类历史混合。

污染回溯进一步显示：9 个 ATTACH 错误中，5 个 case row 能沿当前 object lineage 直接找到早于锚点、且不在 174 条标注内的首次严格多 GT 事件，对应 4 个唯一根；锚点与根的 association event gap 分别为 3540、210、150、5877、6323（其中 210/150 指向同一个根 e00009856）。另有 3 例的污染经 `OBJECT_MERGE` 进入当前对象，必须走 parent-version DAG；剩余 room0_large_r1_0072 是成员 top GT 尚未分叉、但像素历史已持续混合的 pixel-only case。也就是说，只按“第二个 top GT 何时出现”回溯仍会漏一类根。

### 5.4 正确合法候选是否可直接接入

7 个正确动作应为 ATTACH 的错误中：

| 合法候选状态 | 例数 | 代表病例 |
|---|---:|---|
| 至少有一个 clean 候选 | 3 | room0_large_r1_0033、room0_large_r1_0089、v2_r2_001 |
| 所有可用合法候选都污染 | 2 | room0_large_r1_0169、v2_r2_007 |
| 合法候选不确定 | 2 | v2_r2_005、v2_r2_010 |

关键细节：

- room0_large_r1_0169：合法候选 20/20 成员 top GT86，但有 3 张混合 mask，累计非主导像素 6.44%，像素联合判污染。
- v2_r2_007：合法候选 28/28 top GT77，但 8 张混合 mask、跨 6 帧，累计非主导像素 7.35%，判污染。
- v2_r2_005/v2_r2_010：合法候选 33/33 top GT86，但 3 张混合 mask、跨 3 帧，累计非主导像素 2.87%，不足以证明 clean，故判 uncertain。
- room0_large_r1_0089：一个合法候选含 93 条成员（88 GT9、3 GT18、2 GT71），另有一个仅 1 条 GT9 的 clean 小候选。把当前 observation 只接到 clean 小候选仍会把同一 GT9 的大部分历史留在污染对象中，完整修复需要“拆污染候选 + 重聚 GT9”。
- 真正适合直接重定向的窄例是 room0_large_r1_0033 与 v2_r2_001；后者的正确候选排第 12，当前 top-5 页面看不到。

### 5.5 多合法候选与对象图过分割

现有协议允许一个 observation 标多个 `matching_candidate_codes/legal_target_uids`，这一设计应保留；如果强迫人工单选，会把“同一实例已分散在多个图对象中”隐藏成一次普通 ATTACH。

154 个正确动作应为 ATTACH 的独立事件中，合法候选数分布为：1 个候选 147 例、2 个候选 6 例、3 个候选 1 例。多候选事件共 7/154=4.55%，其中只有 room0_large_r1_0089 本身属于 14 个 routing errors；其余 6 个虽然路由动作可判为正确，仍可能暴露结构性过分割。因此只分析 14 个错误会漏掉多数此类信号。

用当前 observation 的 corrected GT 反查每个候选历史成员：

| case | 当前 GT | 多候选中的同 GT 成员数 | 同 GT 候选质心距离 | 判断 |
|---|---:|---|---|---|
| room0_large_r1_0020 | 13 | 152 + 4 | 0.170 m | 两个 clean object 分担同一 cushion，强过分割信号 |
| room0_large_r1_0073 | 13 | 134 + 4 | 0.171 m | 与 0020 是同一候选对的后续版本，不是新的独立 split root |
| room0_large_r1_0089 | 9 | 88 + 1 | 1.062 m | 同一 sofa 跨污染主对象与 clean 单成员碎片；需先拆污染再重聚 |
| room0_large_r1_0136 | 79 | 3 + 1 | 0.004 m | 几乎共点的两个小 object，强 duplicate/过分割信号，但两者像素状态污染/不确定 |
| room0_large_r1_0156 | 69 | 1 + 224 | 0.184 m | 同一 cushion 的单成员 orphan + 主对象；主对象仍 uncertain |
| room0_large_r1_0140 | 30 | 0 + 36 + 0 | 不适用 | 三个人工匹配中只有一个含 GT30；其他候选是远处不同 blinds GT，未获离线支持 |
| room0_large_r1_0142 | 70 | 0 + 210 | 不适用 | 只有一个候选含 GT70；另一选择可能是 cushion/sofa part-whole 或视觉歧义 |

因此有 5 个 event-level 同 GT 多对象信号，但 room0_large_r1_0020/0073 复用了同一候选对，折合 4 组唯一 fragmentation candidate sets。这个统计是“结构性风险信号”，不是总体过分割发生率；两个人工多选未获离线 GT 支持也说明多选本身不能直接触发对象 MERGE。

应新增独立标签/审计字段：

- `MULTI_LEGAL_SAME_INSTANCE`：人工确认多个 candidate 属同一物理实例；
- `OFFLINE_SAME_GT_SUPPORT_COUNT`：离线审计有多少候选真正含当前 GT；
- `FRAGMENTATION_STATUS`：clean-clean、clean-contaminated、uncertain 或 part-whole；
- `CANONICAL_OWNER_PROPOSAL`：只输出建议，不在标注时自动合并。

对应修复也要分层：两个 clean 碎片可做 `REUNIFY_CLEAN_FRAGMENTS`；一侧污染时先 `EXTRACT_CLEAN_LINEAGE` 再重聚；证据不足时建立临时 equivalence set，不能直接不可逆 MERGE。

### 5.6 排名、分差与 NEW 被覆盖

- 7 个正确 existing target 的最佳排名：rank1×3、rank2×2、rank12×1、rank19×1。
- “top-5 候选 + NEW”覆盖 12/14；另外 2 个错误不是判断原则问题，而是候选召回不足。
- 7 个本应 NEW 的错误全部被 existing top1 覆盖；top1 相对 NEW threshold 的中位超额为 0.6311。
- 14 个错误的 top1-top2 margin 中位数为 0.2187。

高分和大 margin 只表示“与当前已经形成的对象表征很相似”，不表示该对象对应单一物理实例。v2_r2_013 在污染形成阶段持续给出 1.65–1.82 的高分，正是反例。

### 5.7 候选池、类别、空间与时间

- 14 个错误页面共出现 71 个唯一冻结候选。成员层判 52 clean/17 污染/2 uncertain；像素联合后变为 27 clean/28 污染/16 uncertain。
- 14/14 错误页面至少出现一个像素联合污染候选。这是错误条件下候选池描述，不是一般候选污染率。
- 当前 observation 与原目标的离线类别关系：异类 4、同类不同实例 1、未知 9。同类例就是 v2_r2_013 的相邻 blinds；异类例证明仅靠类别 gate 也不够。
- v2_r2_013 在 frame253 前两个主要成员簇质心相距约 0.290 m，只占对象 3D bbox 对角线的 6.9%，有 7 个同帧共现、16 次 top-GT 时间切换；空间接近、同类、交替可见共同制造了“像一个对象”的假象。
- v2_r2_011 的 book/table 两簇质心只差约 0.0478 m，有 14 个同帧共现、32 次 top-GT 切换。即使异类，边界相贴时也会被单 mask 和对象历史吞并。

### 5.8 后续视角与原系统自愈

- 13/14 有可用未来视角诊断；frame+30 内至少 2 个独立视角的有 8 例，至少 3 个的有 6 例，中位数为 2；全后缀独立视角中位数为 8。
- 最终 owner 在后缀上看似稳定的有 10/13，但这是 post-run 结果，不是锚点时在线证据。
- 按“当前 observation 最终进入合法 lineage”计算，3/14 有结构性自愈；但这 3 个最终 owner 全部仍是污染或不确定。因此真正达到 identity-clean 的自愈为 0/14，另外 11 例未自愈。

“最终又并到一起”不能当作自愈，因为它可能只是回到一个混合对象。

## 6. 典型病例一：v2_r2_013 的完整污染时间线

目标：`30c6ac88-91e0-44bd-8667-6fd8df4c12a6`；两个物理实例都是 blinds（GT15、GT19）。

### 6.1 像素污染先于成员身份翻转

| frame | observation | top-2 corrected-GT 像素 | association top1 | top1-top2 margin | 含义 |
|---:|---|---|---:|---:|---|
| 131 | `...f000131_r0013` | GT15 80.51%、GT19 10.93% | 1.8204 | 0.8969 | 首张已知混合 mask 进入；association e00005346，version e00005347/@v000011 |
| 135 | `...f000135_r0020` | GT15 61.92%、GT19 31.40% | 1.6506 | 0.7798 | 混合扩大；到 @v000013 形成持续像素污染 |
| 136 | `...f000136_r0015` | GT15 56.24%、GT19 37.09% | 1.7894 | 0.9445 | 边界继续漂移 |
| 137 | `...f000137_r0017` | GT15 51.13%、GT19 43.16% | 1.7985 | 0.9539 | 两实例已近均分 |
| 138 | `...f000138_r0016` | **GT19 48.98%、GT15 46.06%** | 1.8001 | 0.9518 | 同一混合 mask 的 top GT 翻转；association e00005670，生成 @v000016 |
| 139 | `...f000139_r0017` | GT19 54.26%、GT15 40.98% | 1.7918 | 0.9550 | 第二条 top-GT19 成员进入，@v000017 后成为持续多身份污染 |

threshold 为 1.2。错误不是“低分犹豫后误并”，而是上游边界逐帧漂移、下游以高置信度连续吸收。只挖低 margin 或 threshold 附近样本会漏掉它。

在 frame138 事件前的 @v000015：

- 12 条成员在 observation-level 全部是 GT15；
- 但其中 4 张 mask 已 mixed；
- 聚合像素为 GT15 84.40%、GT19 14.26%、GT17 1.34%。

所以“12/12 GT15”只是 top-label 一致，不是单实例纯净。

### 6.2 frame253 是晚期症状

frame253 锚点 association e00011993 的当前 mask 是 clean GT19；系统以 top1=1.8501、threshold=1.2、margin=0.9714 继续 ATTACH。

精确 t-minus 目标 @v000158 已有 132 条成员：108 GT15、24 GT19。像素累计约 GT15 55.75%、GT19 42.63%、GT21 1.49%、GT17 0.12%；117/132 历史 mask 为 mixed。冻结页面使用 @v000157，131 条成员（107 GT15、24 GT19），比精确 t-minus 少同帧先到的一条成员。

compiled episode 把 frame253 记为 ROOT，只能解释为“174 个采样标签内没有更早已标根”。全 ledger 中，严格身份分叉 association e00005670 不在 174 条里，距锚点 6323 个 association 事件；首个像素污染 association e00005346 还比它早 324 个事件。

### 6.3 页面为何把历史误判为 clean

[现有证据]

- 页面只展示 6/131 条历史成员；本例展示样本的 top GT 构成为 5 条 GT15、1 条 GT19，虽未完全漏掉某个身份类别，却没有展示真实占比、混合像素、首次分叉或时间切换。
- 当前 history selector 以首/尾、最大 mask、高置信和均匀时间采样为主，最大仅 6 张，不以“身份多样性/污染转折”取样。
- 冻结版本比精确 t-minus 少 1 条，159 个 ATTACH 事件中这种 packet/exact mismatch 有 38 个；9 个 ATTACH 错误中有 7 个。多数可由同帧更早事件解释，不自动等于页面 bug，但页面必须显式标明版本边界。

[推断]

- 同类相邻 blinds 在单张缩略图上本就难区分；六张视觉上连贯的历史图会让“对象很稳定”的先验压过一张 GT19 历史图。
- D 阶段把“当前 mask 是否单实例”与“候选历史是否单实例”压缩成相近视觉判断，没有成员组成、空间双峰、同帧共现和时间转折证据。
- 页面展示的是稀疏样本而不是对象的 identity history，因此 human `CLEAN_SINGLE_INSTANCE` 实际只证明“所见证据像 clean”，不能证明 131 条完整历史 clean。

建议在后续标注协议中把两个问题拆开：A/B 判断当前 observation；D 单独判断候选完整性，并提供历史成员数、疑似混合比例、空间双峰、共现/切换、首次异常帧和版本精确性。人工盲标页面不能直接暴露离线 GT；这些 GT 字段用于标后审计，线上页面使用不含 GT 的代理信号。

### 6.4 为什么 B2/B3 endpoint pass 仍不够

v2_r2_013 的 anchor-only CREATE 回放：

| 指标 | B2/B3 结果 |
|---|---:|
| endpoint/root action | 通过 |
| 新 owner 成员 | 16 |
| 新 owner GT19 | 16（precision 100%） |
| room0 全部 GT19 observations | 115 |
| 新 owner global recall | 13.91% |
| 新 owner F1 | 0.244 |
| 原目标成员 | 236 |
| 原目标组成 | 129 GT15、99 GT19、8 GT21 |
| GT19 残留在原目标 | 99 |
| 原目标 observation 残余污染率 | 45.34% |
| 原目标 mixed mask 数 | 220 |
| 原目标累计非主导 top-2 像素 | 51.64% |

B2 重放 120 observations、闭包 252、约 203.9 秒；B3 重放 2739、闭包 7004、约 301.8 秒。两者 invariants 通过且定义外 observation 改动为 0。它们证明“从 clean frame253 新建并维持一个精确 GT19 子簇”可行；它们没有证明“历史 GT19 已从混合目标中找回”。

按照本报告的全实例门槛，这次结果应判 **局部成功、整体未修复**。

## 7. 典型病例二：v2_r2_011 证明不是 blinds 特例

frame299 的当前锚点是 clean GT10 table，系统 ATTACH，人工应 NEW。锚点前目标有 94 条成员，校正成员为 50 GT92 book、43 GT10 table（另 1 条不在有效 top-GT 统计），其中 72 张 mask mixed；像素约 51.80% GT92、45.53% GT10。两簇质心只相距 4.78 cm，有 14 个同帧共现、32 次时间切换。

对象从 frame155 的首个版本开始就是 book+table 混合 mask；早期多条成员的 observation-level top GT 都是 92，top-ID 账本因此表面 clean。到 frame186，混合 mask 的 GT10 约 46.79%、GT92 约 40.76%，association e00008322 仍以 top1=1.687、threshold=1.2、margin≈0.956 并入，首次产生 top-ID 分叉；frame189 后成为持续多 GT 污染。

这与 v2_r2_013 形成两种互补证据：

- 同类不同实例：类别过滤无能为力；
- 异类但边界相贴：类别信息可能报警，却仍需要 observation split，而不是把整张 mask 路由给任一对象。

## 8. 其他病例揭示的修复差异

### 8.1 历史污染后才出现的 clean symptom

room0_large_r1_0072/0083/0085/0114/0143 当前锚点本应 NEW，但被并进已经污染的目标。当前 clean observation 做 NEW 可以阻止继续污染，却不能抽回历史上已被该目标吸收的同身份成员。

- room0_large_r1_0072 的目标在成员 top-GT 上仍像单身份，但历史有持续像素混合，是“像素污染先于身份分叉”的另一例。
- room0_large_r1_0083 的污染状态经 object merge transition e00014063 继承；
- room0_large_r1_0085 与 room0_large_r1_0143 共享 object merge transition e00008523。仅沿当前 object UID 线性回溯会断在 merge 后，必须沿 parent-version DAG 找根。
- room0_large_r1_0143 的增强 CREATE 回放虽然 B2/B3 endpoint 通过，新 owner 26/26 为 GT8，但全局 GT8 共 235 条，recall 仅 11.06%、F1=0.199，仍有 208 条 GT8 留在原目标，原目标 observation 残余污染 36.72%。这是第二个“endpoint 通过但完整实例失败”的实证。

### 8.2 错 ATTACH 且正确候选不确定

v2_r2_010 与 v2_r2_005 共用更早的污染 association e00009856：前者锚点时错误目标约 89 GT32+2 GT86，后者演化为约 90 GT32+3 GT86。合法 GT86 候选虽然 33/33 top GT86，却已有 3 张像素混合 mask，因此只判 uncertain；在两个页面中分别排 rank2 和 rank19。

若只把当前 observation redirect，先前 2/3 条 GT86 仍残留错误目标；若直接把它们并到 uncertain candidate，又可能扩大另一处污染。更合理的最小原语是：先把错误目标中的已知 GT86 lineage 抽到临时簇，再用后续独立视角确认是否与合法候选合并。

### 8.3 错 NEW 但合法候选已污染

room0_large_r1_0169 与 v2_r2_007 的合法候选在成员 top-GT 上都“全纯”，像素历史却已污染。两例 B0 endpoint 本来就为 true，说明原系统后续把锚点放回某个合法 lineage；但最终 owner 仍污染/不确定，只是 topology 自愈。直接以 endpoint 判修好会把问题隐藏。

### 8.4 部分遮挡与粒度不能和多实例混合共用一个动作

[现有证据] 14 个错误锚点只有 v2_r2_007 被人工判为 borderline，其余当前锚点都 clean；因此没有证据说“当前锚点遮挡”是这 14 例的主因。相反，明确证据集中在历史 mask：v2_r2_013 连续出现两个同类 foreground，v2_r2_011 从对象出生起就混入贴邻的 book/table。检测类别/文本在历史中也有 blinds/window 等变化，但仅凭这些字符串不能证明 part-whole 粒度错误。

[推断] 摄像机运动下边界占比逐帧变化，可能同时受到遮挡、mask under-segmentation 和投影误差影响；当前 ledger 能证明“两个 GT 像素进入同一 mask”，不能把三者的贡献精确拆开。因此不能见到 partial mask 就一律 `SPLIT`：

- 一个物理实例因遮挡只露出局部，应该保留 fragment hypothesis，等待同一实例的视角补全；
- 两个可分物理实例进入同一 mask，才执行 observation split；
- 真正的 part-whole（例如整体家具与其可动部件）需要 `PART_OF`/层级关系，而不是把 part 和 whole 当同一 identity，也不是强制 NEW；
- 无法判别时进入 quarantine/temp cluster，避免把不确定性永久写入对象。

后续页面和实验应增加 occlusion ratio、2D/3D connected components、深度断裂、跨帧 split 稳定性与 part-whole 候选关系；在有专门标注前，本报告不为这几类给出发生率。

## 9. 为什么现有采样与协议漏掉最早高分误并

### 9.1 协议做对了什么

`PROTOCOL_V2_CN.md` 把 `MIXED_MULTIPLE_INSTANCES` 放在路由判断之前，并明确 mixed 不能伪装成 NEW。这一原则应保留：frame138 的整张 mask 确实不应被标成 `FORCE_NEW`。

### 9.2 盲点在哪里

当前 private routing audit 先要求当前 observation 有可靠 GT（purity>=0.90、support>=0.90、top pixels>=25），再产生可自动判路由的记录。7507 条中编译出 5565 条，1942 条因 observation GT 不可靠被排除。v2_r2_013 的 frame138 purity=0.489；v2_r2_011 的关键混合帧也低于门槛，因此 segmentation-origin 根不会进入路由错误 harvest。

大样本 worklist 从 7465 条概率总体均匀抽 150 条，单事件纳入概率约 2.009%，另有 4 条 error harvest 与 8 条 control。被 purity/mixed gate 排除的根若要进入人工视野，主要只能依赖约 2% 的概率队列。当前查到的 4 个唯一、可直接沿成员加入追到的最早严格污染根中，3 个不在 private routing audit，4 个都不在 174 条标注；其中 2 个就是明确的 mixed/two-foreground 根 e00005670、e00008322。

此外，高分 mining 也会漏：v2_r2_013 的混合阶段 top1 持续 1.65–1.82、margin 0.78–0.96；它不像“阈值边缘错误”，反而像系统最确信的匹配。

### 9.3 概率样本错误率应怎样解释

150 条概率样本中 148 条是路由可判定 clean observation，发现 5 条 routing error，条件错误率为 5/148=3.38%，Wilson 95% CI 约 [1.45%, 7.66%]。另外 2 条为非路由问题。

这个数只回答“当前 mask clean/可判时，路由动作错多少”，不能回答 end-to-end identity integrity。mixed segmentation roots 被排除后，直接把 3.38% 当系统身份错误率会向下偏。174 条中的 14/174 也不能当 prevalence，因为合并数据包含定向 harvest/复核。

### 9.4 建议的双账本，而不是把 MIXED 硬塞回 NEW

保留两个互不混淆的分母：

1. `ROUTING_TICKET`：只收 clean/borderline 且身份可判的 observation，继续统计 NEW/ATTACH 错误率；
2. `SEGMENTATION_ORIGIN_TICKET`：记录 mixed mask、可分组件、当前目标、首次像素污染、后续身份分叉与派生 routing symptoms，不进入纯路由错误率，但进入端到端 identity failure 率和因果图。

下游标注事件增加 `caused_by_event_uid`/`caused_by_ticket_uid`，允许 frame253 指向 frame131/138。若根经 OBJECT_MERGE 继承，再记录 `parent_version_uids`，避免只沿单 UID 回溯。

## 10. 真正需要不同修复原语的类型

下表的 14 例计数来自本次分析推断，不是人工 ground-truth repair 标签。

| 推断类型 | 例数 | 最小修复原语 | 适用条件 | 主要失败风险 | 在线是否用未来信息 |
|---|---:|---|---|---|---|
| 已污染目标中的 clean 当前实例，本应 NEW | 5 | `BACKTRACK_OR_SPLIT_CONTAMINATED_TARGET` + 为新 lineage 建持久边界 | 当前 mask clean；旧目标含当前 GT 的历史成员 | 只新建当前帧会低召回；拆错会伤及主实例；闭包过大 | 回溯只用已发生历史，不需要未来；离线 GT 仅评测 |
| 已确认 segmentation-origin 根 | 2 | `SPLIT_OBSERVATION_MASK` / `QUARANTINE_AND_RESEGMENT`，随后 closure replay | 根 observation 本身有两个可分实例 | 过分割、深度边界不稳、两个碎片再次被合并 | 当前帧重分割不需未来；oracle GT split 使用离线信息，不能上线 |
| 合法候选 clean、无明显历史残留 | 2 | `DIRECT_REDIRECT_TO_VERIFIED_CLEAN_TARGET` | 候选联合状态 clean，且错误对象中没有同身份历史 | 候选漏召回；“clean”判错 | 不用未来 |
| 合法候选已污染/实例被分散 | 3 | `SPLIT_CONTAMINATED_CANDIDATE` + `EXTRACT_LINEAGE` + redirect/reunify | 正确身份的一部分在污染候选内 | 拆断同一实例、漏掉像素级混合成员、产生碎片 | 可只用当前与过去；oracle 成分标签只作上界 |
| 合法候选不确定且错误目标已有同身份残留 | 2 | `EXTRACT_TO_TEMP_ID_CLUSTER` + `DELAY_COMMIT` | 立即 ATTACH/NEW 都有较高污染风险，未来存在独立视角 | 临时簇膨胀、确认延迟、长期悬而未决 | 只有未来视角实际到达后才能用；预读 suffix 属泄漏 |

这些类型对应脚本中的推断计数：5、2、2、3、2，总计 14。

此外还有一个跨越 routing-error 分类的结构型原语：`REUNIFY_SAME_INSTANCE_FRAGMENTS`。它由 7 个多候选 ATTACH 标注触发，但只有 5 个得到“至少两个候选含同一 corrected GT”的支持，且其中 4 个不是 routing error。该原语不能盲目调用：clean-clean 可直接验证重聚；clean-contaminated 要先拆污染 lineage；只有一个候选获同 GT 支持时不得自动合并。

### 10.1 当正确候选本身已混合时的决策顺序

1. **先禁止整对象 attach/merge。** 候选状态若为 contaminated/uncertain，`FORCE_TARGET(candidate_uid)` 不是安全原语。
2. **判断污染发生在哪一层。** 当前 mask mixed 用 `SPLIT_OBSERVATION_MASK`；历史候选 mixed 用 `SPLIT_CANDIDATE/EXTRACT_LINEAGE`；两者同时存在则先拆 observation，再拆候选。
3. **若最早污染可由过去 ledger 定位，优先在最早根修。** 对 observation associate 沿 version predecessor 回溯；对 object merge 沿 parent DAG 回溯。根处修完后只重放受影响依赖闭包。
4. **若身份组件可被当前/过去证据稳定分离，重聚组件。** 例如从污染候选中抽出同一物理实例的历史 observation/像素片段，再把当前 clean observation 接入该组件。
5. **若分离仍不确定，建立临时身份簇并延迟。** 后续视角到达时逐步确认；不要在锚点时预读未来，也不要把临时簇当永久新实例。
6. **最后设置防重并边界。** split 后若关联/后处理立刻把两簇重新并回去，rollback 没有意义。边界应可撤销、有证据期限，避免永久阻止真正同一实例的重聚。

对 v2_r2_013 的具体答案是：**回到 frame131/138，先 quarantine+resegment 混合 mask；GT15 片段 ATTACH 原 blinds，GT19 片段进入 NEW/临时簇；再抽回 f138 之后属于 GT19 的历史 lineage，重放局部闭包，并加可审计的防重并约束。** frame253 anchor-only CREATE 只能作为立即止血或精确 seed，不能作为最终修复。

## 11. 回放结果与闭包副作用

### 11.1 现有 B0/B0R/B1/B2/B3 能说明什么

- B0：原生结果；root action 可能错但 endpoint 已因后续合并而 true，故不能单独诊断。
- B0R：无修复重放，现有四组均保持 exact partition parity，可作为重放基线。
- B1：只改 membership，root action 可正确，但 geometry 标记为 invalid 是设计限制，不应解读成生产算法失败。
- B2：typed dependency closure；较适合最小干预评估。
- B3：完整 post-anchor suffix；是 oracle 压力测试，闭包和运行成本显著更高，也更容易把未来行为混入诊断。

代表性成本：

| case/variant | B2 replay / closure / runtime | B3 replay / closure / runtime | endpoint 解释 |
|---|---|---|---|
| v2_r2_013 | 120 / 252 / 203.9s | 2739 / 7004 / 301.8s | B2/B3 通过，但 full-instance recall 13.91% |
| room0_large_r1_0143 增强变体 | 59 / 608 / 116.2s | 1454 / 6960 / 173.5s | 通过，但 recall 11.06%；基础变体 endpoint 未通过 |
| room0_large_r1_0169 | 61 / 81 / 240.7s | 4209 / 7009 / 411.3s | B0 本就 endpoint true，不能证明根修复 |
| v2_r2_007 | 560 / 589 / 214.7s | 5010 / 7018 / 490.5s | B0 本就 endpoint true，最终 owner 仍非 clean |

现有 B2/B3 的 runtime invariants 均通过，定义外 observation 改动为 0。这是重要的安全证据，但只保证闭包外不变；闭包内是否把整个物理实例修好，仍需下列完整指标。

### 11.2 新的完整实例评价门槛

每个 repair 至少同时报告：

1. root action 是否满足复合语义，而不只是 NEW/ATTACH；
2. 新/目标 owner 的 observation precision；
3. 物理实例在全 ledger 中的 observation recall 与 F1；
4. 像素/点级 fragment precision、recall；
5. 同身份仍残留在其他 owner 的数量与比例；
6. 外来身份仍残留在目标 owner 的 observation 与像素比例；
7. 同一 GT 被拆成多少 owner，避免以过度碎片化换 purity；
8. closure 大小、实际 replay 数、运行时间；
9. closure 外改动、对象版本不变量、几何有效性；
10. 延迟决策的等待帧数与 unresolved ticket 数。

对混合 observation，单个 top GT 会把 49/46 的 mask 整体记给 GT19 或 GT15，因此 observation-level precision/recall仍偏乐观；根修复必须补充像素或 3D point 级分片指标。

## 12. 优先级排序

### P0：先修标注与评价契约，不动生产 mapper

- mixed 继续排除出 routing error denominator，但进入独立 segmentation-origin 因果账本；
- 保留人工多候选标注，并增加 offline same-GT 支持数、唯一 fragment set 和 candidate cleanliness；把 object over-segmentation 与 observation under-segmentation 分开统计；
- 为晚期 symptom 添加 root link、四层污染时间、object-merge parent DAG；
- 页面把当前 mask 质量与候选历史完整性分开；显示 exact t-minus/frozen version 差异和污染转折摘要；
- 回放通过条件从 endpoint 升级为全实例指标。

理由：无需假设某个新模型有效，却能立刻防止“晚期 ROOT”“clean candidate”“endpoint pass”三种错误结论继续扩散。

### P1：做两个 oracle 上界实验

- v2_r2_013：root mask split + 历史 lineage 抽取 + B2 closure replay；
- v2_r2_011：不同类 mixed root 的同样实验。

目的不是上线 GT oracle，而是先证明“即使给正确 split，现有对象/闭包机制能否恢复完整实例”。若 oracle 都失败，就不应先投入在线 mixed detector。

### P1：实现只读 root backtrace 与 candidate partition proposal

- 同时走 linear version predecessor 和 object-merge parent DAG；
- 输出 proposal，不自动改图；
- 先覆盖 e00005346/e00005670、e00008322，以及 merge roots e00008523/e00014063。

### P2：验证在线 quarantine/resegment 与临时身份簇

- 当前/过去证据足够时 split；
- 不足时临时簇 + 延迟确认；
- 所有未来证据必须在时间上真实到达后使用。

### P2：验证可撤销的 persistent boundary

只在 split 后防止立即重并，不把它当作恢复历史 recall 的替代品。

### P3：再考虑更全局的图优化

只有局部 root/split/closure 原语证明不足，才考虑全局多假设优化；否则计算量、解释成本和未来泄漏风险都过高。

## 13. 最小验证实验与可证伪判据

### E1：v2_r2_013 segmentation-root oracle 上界（最高价值）

- 样本：正例 f131–f139；负例选相邻 blinds 但单实例 clean 的历史窗口。
- 处理：在 f131/f138 对 GT15/GT19 做 oracle pixel/point split；GT15 fragment ATTACH 原目标，GT19 fragment NEW/临时簇；抽取随后 GT19 lineage；先跑 B2，达到继续判据后才跑 B3。
- 对照：B0、frame253 anchor-only CREATE；整张 f138 mask NEW 仅作为明确的失败对照，不作为候选方案。
- 预注册继续门槛：两个 owner observation/fragment precision 均>=95%；GT19 global recall>=80%；原目标 GT19 残留<=5%；目标非主导像素<=5%；closure 外改动=0；invariants 通过。
- 可证伪：oracle split 后仍无法提高 recall，或为提高 recall 大量吞入 GT15/GT21，或闭包外发生变化，则“局部 root split+replay 足以修复”被否证。
- 未来信息：oracle GT 用于离线上界，不能作为生产输入。

### E2：无未来信息的 mixed-mask 发现与重分割

- 样本：v2_r2_013 f131–f139、v2_r2_011 f155–f189；加入同类相邻但 clean、单实例部分遮挡、单实例碎片化、以及 part-whole 接触的 holdout 负例。
- 输入：仅当前及过去 RGB/depth/3D/已有对象，不读未来帧和最终 GT owner。
- 目标：最迟在 f138/e00005670 前后把 v2_r2_013 送入 quarantine；输出两个稳定 fragment，而非一个二选一路由。
- 继续门槛：正例根召回 2/2；fragment pixel precision/recall 各>=90%；holdout clean mask 误报警<=5%；不降低 clean routing endpoint。
- 可证伪：高分混合帧仍不报警，或通过大量切碎 clean/occluded single-instance masks 才达到召回，则在线方案不成立；若 part-whole 样本只能被迫二选一，也说明原语集合不完整。

### E3：从晚期错误回溯到最早分叉

- 样本：direct lineage 的 v2_r2_013/v2_r2_011/v2_r2_005/v2_r2_010，以及 parent-DAG 的 room0_large_r1_0083/0085/0143。
- 方法：从锚点 target version 反向检查每次 member add、pixel mixture transition 和 object merge parents，分别给出 first-pixel、persistent-pixel、first-top-ID、persistent-top-ID。
- 继续门槛：复现 v2_r2_013 f131/f135/f138/f139；复现 v2_r2_011 的 birth-mixed 与 f186/f189；对 e00008523/e00014063 能穿过 merge 找到父 lineage。
- 可证伪：只能返回 merge 后节点、返回时间晚于已知污染、或漏掉被污染成员，则线性/当前实现不够。
- 未来信息：回溯只使用发现时刻之前已落盘的历史；不读取 anchor 后 suffix。

### E4：污染合法候选的 split-before-attach

- 样本：v2_r2_007、room0_large_r1_0169、room0_large_r1_0089；clean 对照 room0_large_r1_0033、v2_r2_001。
- 对照：直接 ATTACH、只重定向当前 observation、split candidate 后抽 lineage+重聚。
- 继续门槛：目标身份 recall 明显提高，owner 非主导像素下降到<=5%，同一 GT owner 数不增加，clean 对照不被误拆。
- 可证伪：split 只提高 purity 却进一步降低 recall，或 clean 对照频繁碎裂。
- 未来信息：第一阶段用 GT oracle 评估上界；通过后再换成只用当前/过去特征的 proposal。

### E5：不确定候选的临时身份簇

- 样本：v2_r2_005、v2_r2_010；必要负例为后续视角不足的 case。
- 处理：把错误目标中的 2/3 条 GT86 历史 lineage 与当前 observation 先置入 temp cluster，不立即并入 uncertain 合法候选；每个独立新视角到达后更新假设。
- 对照：立即 attach 合法候选、立即 NEW、只改当前 observation。
- 继续门槛：最终污染率低于三种对照，GT86 recall 不下降；中位确认延迟、最大 unresolved 数有界。
- 可证伪：在现有视角密度下多数 ticket 无法收敛，或临时簇数持续增长。
- 未来信息：允许“到达后的未来视角”；离线一次性读取整个 suffix 决定锚点动作属于泄漏，禁止。

### E6：完整实例 metric gate 回归

- 样本：v2_r2_013、room0_large_r1_0143，并加入 B0 endpoint 本来就 true 的 v2_r2_007/0169。
- 处理：不改任何回放结果，只重算 endpoint、precision、global recall/F1、残余污染、pixel impurity、owner fragmentation 和 closure cost 的联合 pass/fail。
- 预期：现有四例不能仅凭 endpoint 宣称完整成功；v2_r2_013、0143 必须因低 recall/高残余污染 fail。
- 可证伪：若联合 gate 仍让这些明显不完整的分区 pass，指标定义本身需要重写。
- 未来信息：纯离线评价，可以用全序列，但不得反哺同一次在线决策。

### E7：双流采样能否抓住高分根

- 方法：保持原 150 条 routing probability queue 不变，新增等预算 segmentation-risk queue；风险信号关注 mask 多组件、深度断裂、跨帧边界漂移、同帧共现、对象内部空间/时间双峰，而不是低 association margin。
- 继续门槛：在固定预算下捕获 e00005346/e00005670 与 e00008322；holdout clean 报警率有界；不改变纯 routing error denominator。
- 可证伪：只有使用 corrected GT 或未来帧才能抓住根，或 fixed-budget 下相对均匀采样没有增益。

### E8：多候选标注能否作为对象过分割触发器

- 正例：room0_large_r1_0020/0073、0089、0136、0156；其中 0020/0073 按同一候选集合只计一个 split pattern。
- 歧义/负例：room0_large_r1_0140、0142，以及单候选但同类相邻的 clean 事件。
- 处理：先离线计算每个候选对当前 GT 的成员支持、候选纯净度、同 GT 质心距离和成员质量分布；clean-clean 运行 canonical reunification proposal，污染候选先抽取同 GT lineage，uncertain 只建立临时 equivalence set。
- 对照：保持多 object 不变、直接全对象 MERGE、只把当前 observation attach 到其中一个候选。
- 继续门槛：同一 GT owner 数下降、global recall 不降、canonical owner 非主导像素<=5%、其他 GT owner/member 不变；0020/0073 不得重复生成两个独立修复 ticket。
- 可证伪：多候选触发器在 0140/0142 上自动误并不同 GT，或在 0089 上把 GT18/GT71 一起并入 canonical sofa，则“多选即合并”假设被否证。
- 未来信息：corrected GT 只用于离线验证；在线版本只能使用人工多选与当前/过去候选证据，后续视角必须到达后再更新 equivalence set。

## 14. 局限

- 只有 room0，一个 dev ledger；不能外推到其他场景和生产总体。
- 14 个错误经过概率抽样、harvest 与复核混合，除了 150 条概率子样本外不能估 prevalence。
- 7 个多候选事件只能估计本批标注中的信号比例；5 个同 GT 多对象事件还包含一个重复候选集合，不能当作 5 个独立过分割根。
- 像素联合状态依赖 corrected-GT projection，仍可能有边界投影噪声；本次阈值只是诊断性定义。
- observation-level top GT 会丢失混合 mask 内的次实例，因而完整实例 recall 仍需 pixel/point 级实验确认。
- 未来视角和 native final ownership 是诊断数据，不是因果在线证据。
- 现有可比回放只覆盖少量 case；B2/B3 证明 closure 可运行，不证明 proposed split/backtrack 机制已实现。
- 本报告没有改变标注协议文件或 mapper，仅给出下一轮最小实验和停止条件。

## 15. 最终判断

实验 0 当前最重要的新发现，是 **“路由错误的候选对象并非静态、纯净的选择项；候选本身可能已经是历史错误与混合 mask 的产物。”** 因此修复单位必须从“当前 observation 的单次动作”提升为“observation fragment + identity lineage + object-version DAG + 受影响闭包”。

短期应先建立 segmentation-origin 因果账本与全实例 metric gate；随后用 v2_r2_013、v2_r2_011 做 oracle root-split 上界。只有 oracle 能把 recall 和残余污染同时修好，才值得继续训练无未来信息的在线 resegment/quarantine。对于 clean 候选保留简单 direct redirect；对于污染候选先 split；对于不确定候选用临时簇和延迟决策。这样既不把所有错误强行统一，也不会再把“高精度小子簇”误当成“整个实例已修复”。
