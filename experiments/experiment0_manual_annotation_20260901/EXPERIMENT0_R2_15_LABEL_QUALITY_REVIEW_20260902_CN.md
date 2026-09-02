# 实验 0 R2：15 例人工标注质量复核

日期：2026-09-02
范围：`room0`，12 个基础案例 + 3 个隐藏重复
目的：判断现有标注是否能支持实验 0 的身份路由判断，以及是否可以开始扩量；不使用机械 Go 门槛。

## 1. 直接结论

标注 schema、证据包和两阶段路由推导机制已经被验证可用，可以支持实验 0；但当前 15 条原始答案不能不经裁决直接作为大规模标注模板或正式统计输入。

原因不是流程失效，而是存在两类清晰、可修正的系统性定义误用：

1. 把仍可绑定到稳定物理实例的局部可见 observation 错标为 `BACKGROUND_OR_FRAGMENT`；
2. 把同类别、同材质或同一窗户系统中空间上不同的实例错标为同一实例。

建议保留原始标签只读，为 5 条记录写 adjudication overlay（4 个基础现象，其中 1 个有隐藏重复）。完成这一步后即可开始扩量，不需要重新做一轮固定 Go 校准集。

## 2. 完整性与可复现性

- 15/15 已完成，0 草稿；15 个 case UID 唯一。
- mapper 完成到 frame 399。
- schema、事件绑定、冻结 `t^-` snapshot、展示资产和盲标锁定检查：`PASS`，0 错误。
- 原始状态：11 条 `COMPLETED`，4 条 `EXCLUDED`。
- observation quality：11 `CLEAN`、3 `BACKGROUND_OR_FRAGMENT`、1 `MIXED`。
- 身份证据状态：15/15 均为 `SUFFICIENT_FOR_IDENTITY`。

## 3. 隐藏重复

3/3 隐藏重复在以下字段全部完全一致：quality、候选集合、身份证据状态、目标预状态、完整地图状态、outside UID 和派生路由。

这说明标注操作具有很高的内部稳定性。但 `v2_r2_013` 的 blinds 判断及其重复 `v2_r2_002` 同时重复了相同错误，因此“稳定”不能替代“正确”。该结果指向规则理解偏差，而不是偶然点击错误。

## 4. 必须裁决的记录

| case | 原标注 | 视觉与冻结参考裁决 | 建议 overlay | 对实验 0 的影响 |
|---|---|---|---|---|
| `v2_r2_007` | quality=`BACKGROUND_OR_FRAGMENT`；候选 B | 当前是可绑定到既有 sofa 的局部观察，B 身份判断正确；局部可见不等于背景碎片 | quality=`BORDERLINE_SINGLE_INSTANCE`，保留 B | 从排除项恢复为 `WRONG_NEW_FALSE_SPLIT` |
| `v2_r2_008` | quality=`BACKGROUND_OR_FRAGMENT`；候选 B | 当前是 lamp 的极小局部观察，B 的多帧历史和空间位置可确认身份 | quality=`BORDERLINE_SINGLE_INSTANCE`，保留 B | 从排除项恢复为 `CORRECT_ATTACH` |
| `v2_r2_011` | quality=`CLEAN`；候选 A、D | 当前 table 与 A/D 是同类、外观近似但空间位置不同的实例；3D 不连续 | `NONE_SHOWN`；完整 `t^-` 地图=`NO_MATCHING_NODE_EXISTS` | 从 `CORRECT_ATTACH` 改为 `SHOULD_HAVE_BEEN_NEW` |
| `v2_r2_013` | quality=`CLEAN`；候选 B、C、D、E | 当前是中央独立 blinds panel；相邻面板属于同一窗户系统，但不是同一物理实例 | `NONE_SHOWN`；完整 `t^-` 地图=`NO_MATCHING_NODE_EXISTS` | 从 `CORRECT_ATTACH` 改为 `SHOULD_HAVE_BEEN_NEW` |
| `v2_r2_002` | 与 `013` 相同 | `013` 的隐藏重复，应保持同一裁决 | 与 `013` 相同 | 同上，仅用于一致性，不算独立样本 |

私有自动参考只作为审计线索；上述结论已结合人工页面中的 current context/crop、候选多帧历史和 3D 空间关系逐例复核。

## 5. 裁决后的覆盖

建议 overlay 后，15 条记录应为：

- 13 条 `COMPLETED`，2 条真实排除（1 background、1 mixed）；
- `CORRECT_ATTACH` 3；
- `CORRECT_NEW` 2；
- `SHOULD_HAVE_BEEN_NEW` 3；
- `WRONG_ATTACH_EXISTING` 2；
- `WRONG_NEW_FALSE_SPLIT` 3；
- `OUT_OF_SCOPE_BACKGROUND_OR_FRAGMENT` 1；
- `OUT_OF_SCOPE_MIXED_MULTIPLE_INSTANCES` 1。

去掉 3 个隐藏重复后，12 个基础案例正好是五个路由单元各 2 条，加 1 条真实 background 和 1 条真实 mixed。因此标注设计能表达实验 0 所需的五类 ATTACH/NEW 事实，并能把非路由问题排除在主统计之外。

## 6. 时间与局限

- 盲标中位数 18.8 秒；揭示后中位数 5.5 秒；单例总中位数 27.8 秒。
- `v2_r2_001` 的盲标计时为 40394.7 秒，是页面长时间停留造成的计时伪影，必须保留原始值但不能进入效率结论。
- 本轮只有 `room0`，是刻意平衡的 schema calibration，不能估计自然错误率。
- 15/15 都给出充分证据且置信度 5，但高置信度记录中仍有系统性错误；后续不能用置信度替代抽查。

## 7. 扩量建议

完成 adjudication overlay 后可以开始大规模标注。扩量时不设新的固定 Go 门槛，但必须执行滚动质量控制：

1. 对 `BACKGROUND_OR_FRAGMENT` 全量复核，防止把可识别的局部实例错误排除；
2. 对同类别、同材质、同一窗户/家具系统候选进行重点复核，身份必须由位置和 3D 连续性决定；
3. `NONE_SHOWN` 必须绑定完整冻结 `t^-` 地图检查；
4. 随机抽查至少 10%，同时保留隐藏重复；
5. 原始标签不可覆盖，所有更正写入独立 overlay，并分别报告原始结果和裁决后结果。

最终判断：**机制可用，原始 15 条需先做小范围裁决；裁决后可以扩量，不需要推倒重来。**
