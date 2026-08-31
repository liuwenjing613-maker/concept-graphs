# 体素持续区域 × 历史关联不确定性：最后一次验证预注册

冻结日期：2026-08-31。本文在读取本方法的 DEV/HOLDOUT 指标前冻结。

## 1. 研究问题与范围

只验证一个窄假设：

> 决策发生时保存的相对 association uncertainty，能否识别自然产生的错误关联；若能，是否能补充后续形成的 `second-label largest 3D connected-region fraction`。

不再验证或抢救旧的统一体素分数，不声称覆盖所有 mask、false split、same-class false merge 或语义标签错误。

## 2. 数据与因果约束

- DEV：room0、office0。
- HOLDOUT：room1、office1。
- 四个场景均为 `frame=0` 空图开始、`start=0,end=2000,stride=5` 的400帧严格在线建图。
- 只读取 association 决策发生时已经保存的空间、视觉、aggregate 矩阵、候选 object 版本和阈值。
- 禁止在终态 object 上重算当时分数。
- 候选 object 身份只用严格早于当前帧的历史 observation 建立；同帧 observation 不进入身份判断。
- GT 只进入离线 evaluator，不进入 mapper、分数或候选生成。

## 3. Event GT

单 observation 可评条件：

- 官方 Replica instance GT 支持比例 `>=0.9`；
- top-instance purity `>=0.9`；
- top-instance 像素 `>=25`；
- top instance 为前景，排除 wall/floor/ceiling/unknown/undefined。

候选 object 的历史身份可评条件：

- 至少3个可靠历史 observation；
- 来自至少3个不同处理帧；
- dominant GT instance ratio `>=0.8`。

MERGE 事件：选中 object 的历史 GT instance 与当前 observation 不同，记为错误关联；相同记为正确。

CREATE 事件：只评价 top-1 aggregate candidate 身份可评的事件。top-1 candidate 与 observation GT instance 相同，记为 false split birth；不同记为正确创建。CREATE 与 MERGE 分开报告，绝不合成统一标签。

## 4. 固定候选指标

风险方向统一为“越大越可疑”：

1. `risk_low_margin = -(top1_aggregate-top2_aggregate)`；
2. `risk_near_threshold = -abs(top1_aggregate-threshold)`；
3. `risk_modality_disagreement = 1[argmax(spatial) != argmax(visual)]`；
4. `risk_semantic_residual = max(visual)-visual(chosen target)`，仅 MERGE；
5. `risk_spatial_residual = max(spatial)-spatial(chosen target)`，仅 MERGE。

不训练分类器，不加权求和，不取多个百分位最大值。

## 5. DEV 选择与早停

数据可评门（每个 DEV 场景、MERGE 主目标）：

- 至少50个可评事件；
- 至少10个错误和10个正确事件；
- 正负事件各至少来自5个 object cluster。

若不满足，判为 `STOP_DATA_UNSUPPORTED`，停止，不进入体素组合。

在固定5个指标中，只允许选择同时满足以下条件者：

- room0、office0 AUROC 均 `>0.55`；
- room0、office0 AP lift 均 `>0`。

合格指标按“最大化较差场景 AUROC，再最大化较差场景 AP lift，再按上面固定顺序”选择。若无指标合格，判为 `STOP_EVENT_SIGNAL`。

CREATE/false-split 作为独立支线使用相同方向门；其失败不改变 MERGE 主线，但必须明确报告。

## 6. HOLDOUT 门

HOLDOUT 只评价 DEV 冻结的一个 MERGE 指标，禁止调参。

- 任一场景 AUROC `<=0.5` 或 AP lift `<=0`：`STOP_HOLDOUT_REVERSAL`，立即停止，不做 V→A。
- 两场景 AUROC 均 `>=0.70` 且 AP lift 均 `>=0.10`：event signal `GO`。
- 其余正向结果：`MODIFY_EVENT_WEAK`，允许完成一次预注册的 object-level 互补性检查，但不能单独上线。

置信区间使用 object-cluster bootstrap，避免把同一 object 的大量重复事件当成独立样本。

## 7. Object-level（仅在未早停时执行）

比较：

- `N`：observation 数量基线；
- `A`：冻结 association 风险的 object/region 聚合；
- `V`：5cm、按帧平衡的第二标签最大连续区域比例；
- `V→A`：先取 V 前 `2K`，再用 A 重排，`K=ceil(20% objects)`。

所有 object 至少5个独立帧支持。对 observation 数和 voxel 数进行 DEV 拟合、HOLDOUT 冻结应用的线性去混杂，并同时报告原始值与去混杂值。

最终 `GO` 必须同时满足：

- room1、office1 的 V→A AUROC 均 `>=0.70`；
- AP lift 均 `>=0.10`；
- Top-20% / Bottom-20% error rate 均 `>=2`；
- V→A 相对最佳单项 A 或 V 至少有约 `0.05` 的 AP 或 AUROC 绝对提升，且 paired cluster bootstrap 增益大部分为正。

否则停止把 voxel 当作 trigger。体素仅保留为空间索引、可视化、repair scope 和 provenance。

## 8. 必须报告

- 全部成功、失败、排除原因、样本量、错误率和 timing；
- event-level 与 object-level 分开；MERGE 与 CREATE 分开；
- AUROC、AP、AP lift、Top/Bottom、cluster-bootstrap 95% CI；
- GT 可评覆盖率和不可靠/混合 observation 排除量；
- association matrix/top-K/版本对齐完整性；
- 自动 instance GT 不是人工 actionable truth，最终只给机制可行性结论。
