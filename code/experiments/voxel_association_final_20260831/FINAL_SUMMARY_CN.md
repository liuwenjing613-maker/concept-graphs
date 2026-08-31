# 体素持续区域 × 历史关联不确定性：最后一次验证总结

日期：2026-08-31  
最终判决：**STOP——停止把 voxel 或 association score 继续开发成自动错误触发器。**

## 1. 一句话结论

本轮严格按预注册顺序，先验证 association event uncertainty，再决定是否构造 `V→A`。DEV 只有“决策分数离阈值有多近”在两个场景勉强同向；冻结到 HOLDOUT 后，room1 仍很弱，office1 没有任何严格可评的错误 MERGE 正例，无法评价排序。因此触发早停，**没有继续实现或运行 V→A**。

这不是证明“体素存储无用”，而是证明：

> 在当前四个 Replica 在线场景和可靠 GT 门下，没有足够、稳定、跨场景的证据支持“历史关联不确定性＋后续体素持续区域”成为自动修复 trigger。

## 2. 开始前如何检查方法合理性

### 2.1 没有未来泄漏

四个场景均从 `frame=0` 空图开始，处理400帧（原始帧0–1995，stride=5）。每次 association 当时已经保存：

- 完整 spatial / visual / aggregate 矩阵；
- top-K 候选、top1/top2、阈值和 mapper 动作；
- 候选 object 的当时版本和成员；
- observation UID、frame 和 processed mask。

评测没有在终态 object 上重新计算历史分数。候选 object GT 身份只由严格早于当前帧的 observation 建立，同帧 observation 被排除。

### 2.2 GT 与生产分数隔离

GT 只在离线 evaluator 中读取。可靠 observation 要求：官方 Replica instance GT 支持比例和 purity 均不低于0.9，top实例至少25像素，且必须是前景。

候选 object 至少有3个可靠历史 observation、跨3帧，dominant GT instance ratio至少0.8。身份不成熟或已经混乱的候选被排除，而不是强行贴标签。

### 2.3 不重新堆总分

只验证五个预注册相对指标：

1. top1-top2 小 margin；
2. top1 离阈值近；
3. 空间 winner 与语义 winner 不一致；
4. chosen target 的语义 residual；
5. chosen target 的空间 residual。

没有训练分类器、没有加权求和、没有再取多个百分位最大值。

预注册文件 SHA256：`bb358e1776ac84f77e96db2a8764760f0f5eda644e8fe0a9fd26c77c908bc41ee`。

## 3. Event GT 的含义

MERGE：当前 observation 的 GT instance 与 chosen object 的可靠历史 dominant GT 不同，记为错误关联；相同记为正确。

CREATE：仅当 top-1 aggregate candidate 已有可靠历史身份时可评；若它与 observation 是同一 GT instance，记为 false-split birth，否则是正确创建。

MERGE 与 CREATE 从未混成一个统一目标。

## 4. DEV 结果与冻结规则

DEV 数据门通过：

| 场景 | 可评 MERGE | 错误 | 正确 | 错误 object clusters | 正确 clusters |
|---|---:|---:|---:|---:|---:|
| room0 | 796 | 20 | 776 | 8 | 18 |
| office0 | 497 | 95 | 402 | 15 | 19 |

五个指标中，只有 `risk_near_threshold = -abs(top1-threshold)` 在两个 DEV 场景同时满足 AUROC>0.55、AP lift>0：

| 场景 | AUROC（cluster bootstrap 95% CI） | AP lift（95% CI） |
|---|---:|---:|
| room0 | 0.735（0.462–0.891） | +0.041（+0.005–+0.187） |
| office0 | 0.557（0.436–0.688） | +0.004（-0.032–+0.074） |

因此它只是在预注册门上“允许继续”，并没有形成强 DEV 证据。较差场景几乎等于随机，AP 增益接近零。

冻结规则 SHA256：`afbba9a6d841987ee27cf624aa9edc3e99d44ff290b691d26f49625cb76371475`。

## 5. HOLDOUT 结果与早停

冻结规则原样应用于 room1、office1：

| 场景 | 可评 MERGE | 错误/正确 | AUROC | AP lift | 判定 |
|---|---:|---:|---:|---:|---|
| room1 | 297 | 12/285 | 0.611（0.257–0.840） | +0.026（-0.013–+0.256） | 远低于GO |
| office1 | 109 | 0/109 | 不可计算 | 不可计算 | 无自然正例 |

room1 的冻结指标没有反转，但区间很宽并跨随机水平；AUROC 0.611、AP lift +0.026 都远低于预注册 GO（0.70、+0.10）。office1 的109个严格可评 MERGE 全部正确，因此无法评价任何错误排序器。

协议规定 HOLDOUT 不可评时为 `STOP_HOLDOUT_UNEVALUABLE`，立即停止。因此没有继续构造 object-level `A`、`V`、`V→A`，也没有在 HOLDOUT 上更换成看起来更好的指标。

## 6. False split / CREATE 支线为什么也停止

严格可评 CREATE event 数量及其中错误数：

| room0 | office0 | room1 | office1 |
|---:|---:|---:|---:|
| 0/0 | 4/10 | 0/1 | 0/1 |

样本远不足以验证 threshold slack 是否能发现 false-split birth。降低 object maturity 或 GT purity 会直接削弱标签可信度，所以没有为凑样本改变门槛。

## 7. 完整性和覆盖率

| 场景 | 全部 association | 矩阵行核验 | 最终可评 MERGE | 因 observation GT 不可靠而排除 |
|---|---:|---:|---:|---:|
| room0 | 7,507 | 937 | 796 | 6,570 |
| office0 | 3,106 | 863 | 497 | 2,234 |
| room1 | 4,473 | 519 | 297 | 3,953 |
| office1 | 3,272 | 131 | 109 | 3,140 |

矩阵 observation/object 顺序、top1 重构、candidate version 对齐均未发现错误。大量 observation 被排除的主要原因是 mask 的 top-instance purity 不足0.9；这也说明很多现有地图错误首先来自混合/边界不可靠 mask，而不是“干净 observation 被 matcher 选错 object”。association uncertainty 无法替代 mask 内部分区证据。

## 8. 最终应该保留和停止什么

### 保留

- 简单稀疏 voxel ledger，作为空间索引、repair scope、证据 provenance 和可视化底座；
- AssociationTrace，作为历史审计和 counterfactual replay 输入；
- 已有第二标签连续区域结果，作为探索性现象记录。

### 停止

- 不把原始 spatial/semantic score 或相对 uncertainty 接入在线 ticket；
- 不继续开发统一 voxel trigger；
- 不继续在这四个场景上调阈值、加特征或训练分类器；
- 不声称解决 false merge、false split 或 mask repair。

如果未来获得新的、包含足够自然 association errors 的场景和独立人工 actionable 标签，可以重新提出一个新研究问题；那不属于本轮方案的继续调参。

## 9. Timing 与产物

- DEV evaluator：34.40秒（包含2,000次 object-cluster bootstrap）。
- HOLDOUT evaluator：12.63秒。
- mapper 没有重跑：复用了此前严格在线、决策时保存且完整性通过的 causal ledger；这不改变因果性，也避免重复约30分钟建图。

关键原始产物：

- `dev/event_records.jsonl`、`holdout/event_records.jsonl`；
- `dev/event_metrics.csv`、`holdout/event_metrics.csv`；
- `dev/integrity_audit.json`、`holdout/integrity_audit.json`；
- `dev/frozen_rule.json`；
- `holdout/holdout_decision.json`；
- `final/01_final_decision.png`。

本结论属于严格但小规模的机制验证，不是跨数据集 SOTA 结论。自动 instance GT 也不等于人工 actionable repair truth；然而现有结果已经不足以支持继续投入这个 trigger 方向。
