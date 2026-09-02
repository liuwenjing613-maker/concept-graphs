# 实验 0：room0 主论文范围净化审计

## 最终判断

当前标注仍然有效，但它支持的是**事件级身份路由判断**，不能把 14 个错误直接当成主论文的 14 个独立 root false-attach。按最新冻结定义逐层复核后：

- 人工错误事件：14；
- false split / 非 ATTACH 错误：5；
- 当前 observation 通过 strict purity/质量门：13；
- ATTACH 错误中，精确 t-minus 目标已预污染：9；
- 当前 14 例里可作为主论文严格 root 正例：**0**。

这不是说人工标注错了。人工标签正确回答了“当前 observation 应该 ATTACH 哪个节点或 NEW”；问题在于后续目标历史审计发现，这些页面上的错误多数已经是更早污染的症状。主论文只允许把第一次从干净状态发生的错误 ATTACH 算作 root。

## 1. 为什么原来的 14 例不能直接计数

| case | 路由事实 | 人工因果角色 | 精确 t-minus 目标 | 主论文处置 |
|---|---|---|---|---|
| room0_large_r1_0033 | WRONG_NEW_FALSE_SPLIT | PENDING | N/A | OUT_FALSE_SPLIT_OR_NON_ATTACH |
| room0_large_r1_0072 | SHOULD_HAVE_BEEN_NEW | CASCADE | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |
| room0_large_r1_0083 | SHOULD_HAVE_BEEN_NEW | CASCADE | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |
| room0_large_r1_0085 | SHOULD_HAVE_BEEN_NEW | CASCADE | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |
| room0_large_r1_0089 | WRONG_NEW_FALSE_SPLIT | CASCADE | N/A | OUT_FALSE_SPLIT_OR_NON_ATTACH |
| room0_large_r1_0114 | SHOULD_HAVE_BEEN_NEW | CASCADE | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |
| room0_large_r1_0143 | SHOULD_HAVE_BEEN_NEW | ROOT | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |
| room0_large_r1_0169 | WRONG_NEW_FALSE_SPLIT | ROOT | N/A | OUT_FALSE_SPLIT_OR_NON_ATTACH |
| v2_r2_001 | WRONG_NEW_FALSE_SPLIT | ROOT | N/A | OUT_FALSE_SPLIT_OR_NON_ATTACH |
| v2_r2_005 | WRONG_ATTACH_EXISTING | CASCADE | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |
| v2_r2_007 | WRONG_NEW_FALSE_SPLIT | ROOT | N/A | OUT_FALSE_SPLIT_OR_NON_ATTACH |
| v2_r2_010 | WRONG_ATTACH_EXISTING | ROOT | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |
| v2_r2_011 | SHOULD_HAVE_BEEN_NEW | ROOT | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |
| v2_r2_013 | SHOULD_HAVE_BEEN_NEW | ROOT | ALREADY_CONTAMINATED | OUT_CASCADE_OR_PRECONTAMINATED_TARGET |

最重要的修正是：root/cascade 不能只依据当前截图或目标 top-GT 多数来定，必须读取事件发生前的**精确 object version**，并沿 observation association 与 object-merge parent DAG 查历史。冻结页面版本和精确 t-minus 版本不一致的案例也必须以精确版本为统计依据。

## 2. 概率样本重新解释

150 个概率样本中，原先有 148 个可做一般路由判断，事件级错误为 5/148=3.38%。该数仍可报告为‘clean observation 条件下的路由动作错误率’，但不是主论文 root false-attach 发生率。

按主论文 strict 条件过滤后，概率样本中有 69 个‘当前 observation 纯净 + 精确 t-minus 目标因果干净 + 至少两条目标历史’的 ATTACH 事件；人工确认 root false-attach 为 0，Wilson 95% 区间为 [0.00%, 5.27%]。

这个 0 不能解释为问题不存在：这里只是 room0 开发场景，严格分母较小，而且完整流自动审计仍发现待人工复核的候选。它能说明的是：**当前标注尚未给主论文提供已确认的自然 root 正例。**

## 3. 完整流自动候选的深审计

旧的 top-GT 自动逻辑在完整流中给出 6 个 main-scope root 候选。加入精确 t-minus 混合历史后，严格干净候选 0 个，目标清洁度边界候选 1 个，明确预污染并排除 5 个。自动结果只是选例依据，不能替代人工标签。

| event | frame | 自动动作真值 | t-minus 状态 | 是否已人工标注 | 处置 |
|---|---:|---|---|---|---|
| room0_20260831T111035Z_5c9d86fa_e00007863 | 178 | SHOULD_HAVE_BEEN_NEW | ALREADY_CONTAMINATED | 否 | AUTO_OUT_PRECONTAMINATED_TARGET |
| room0_20260831T111035Z_5c9d86fa_e00009856 | 214 | WRONG_ATTACH_EXISTING | UNCERTAIN | 否 | AUTO_TARGET_CLEANLINESS_UNCERTAIN_NEEDS_HUMAN_REVIEW |
| room0_20260831T111035Z_5c9d86fa_e00010853 | 231 | SHOULD_HAVE_BEEN_NEW | ALREADY_CONTAMINATED | 否 | AUTO_OUT_PRECONTAMINATED_TARGET |
| room0_20260831T111035Z_5c9d86fa_e00011993 | 253 | SHOULD_HAVE_BEEN_NEW | ALREADY_CONTAMINATED | 是 | AUTO_OUT_PRECONTAMINATED_TARGET |
| room0_20260831T111035Z_5c9d86fa_e00014199 | 299 | SHOULD_HAVE_BEEN_NEW | ALREADY_CONTAMINATED | 是 | AUTO_OUT_PRECONTAMINATED_TARGET |
| room0_20260831T111035Z_5c9d86fa_e00016809 | 342 | SHOULD_HAVE_BEEN_NEW | ALREADY_CONTAMINATED | 是 | AUTO_OUT_PRECONTAMINATED_TARGET |

## 4. 当前能得出的结论

1. 标注 schema 能稳定表达 ATTACH/NEW 身份事实，标注数据可以继续用于候选覆盖、困难负例和事件级错误分析。
2. 现有 Q2–Q4 mixed-root 回放只能作为旁支/上界分析；mixed mask 不进入主论文 false-attach 正例。
3. 当前不能声称‘方法可行’，也不能进入自动 trigger 或 VLM 主实验；实验 0 的核心自然 root 数仍未建立。
4. 下一步只复核深审计保留下来的自然 root 候选，并在 office0 与未见场景从空图严格在线运行后重复同一协议。
5. baseline 同配置整场重复确定性仍需单独完成；现有局部 B0R parity 不能替代整场双跑。

## 5. 后续判定顺序

- 先让人工盲审保留的自动候选，不能直接把自动 GT 当答案；
- 对通过者编译未来独立视角、top-K+NEW 覆盖和 cascade descendants；
- room0/office0 只做开发，不调未见场景阈值；
- 至少 4 个未见场景完成后，再判断自然 root 数、场景覆盖和未来证据是否足以继续；
- Experiment 0 未成立前，不把 mixed-mask detector、VLM 或复杂修复器接回主线。

## 6. 可复现产物

- `metrics.json`：所有计数与 Wilson 区间；
- `human_error_scope_rows.jsonl/.csv`：14 例逐例范围裁决；
- `probability_scope_rows.jsonl`：150 个概率样本的严格分母归类；
- `auto_root_candidate_rows.jsonl`：完整流自动 root 候选的精确 t-minus 深审计。
