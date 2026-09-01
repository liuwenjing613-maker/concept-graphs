# 实验 0 v2 试标复核与全面标注决策

日期：2026-09-01  
场景：Replica `room0`  
结论：`HOLD_FULL_ANNOTATION / GO_CALIBRATION_R2`

## 1. 结论

当前 15 条原始标注**不能直接用于后续路由统计，也不能立刻开始全面标注**。数据文件本身完整、绑定正确、页面逻辑有效；主要问题是标注规则被系统性误解：把“独立物体因视角、遮挡或出画而只看到一部分”标成了 `GRANULARITY_AMBIGUOUS`。

这不是整个标注方案失效。逐图复核表明：去掉该误用后，多数候选身份判断正确，问题集中在少量百叶窗实例混淆和一次页面误选。应先冻结更清楚的操作定义，保留原始答案只读，做一轮小型新鲜校准集 R2；R2 通过后再进入全面标注。

## 2. 数据与完整性

- 12 个基础案例 + 3 个隐藏重复，共 15 条；15/15 均有盲标草稿和最终标签，case UID 唯一。
- mapper 从空图在线运行至 frame 399；本轮案例均绑定事件时 `t^-` snapshot。
- event UID、原动作/目标、public/private packet、展示资产 SHA256 和最终派生字段复算全部通过，0 个结构或哈希错误。
- 盲标答案与揭示动作后的最终文件完全一致，未发现揭示后篡改盲标身份答案。
- 本试标是五类路由的平衡校准集，不能估计自然错误率。

## 3. 原始标注结果

| 指标 | 结果 | 含义 |
|---|---:|---|
| 完成文件 | 15/15 | 标注流程完整 |
| 可进入五类路由的 `COMPLETED` | 4/15 | 远不足以验证五类路由 |
| 被排除 | 11/15 | 其中 9 条为 `GRANULARITY_AMBIGUOUS`，2 条为预设排除例 |
| 身份证据 `SUFFICIENT` | 15/15 | 不是证据缺失造成的排除 |
| 原始五类/排除标签与私有参考相同 | 4/15 | 主要被错误的 quality 排除 |
| 临时把 `GRANULARITY_AMBIGUOUS` 视为单实例后相同 | 11/15 | 说明系统性问题主要在 quality 定义 |
| 合法候选集合与私有参考相同 | 9/13 | 另 2 条是预设排除例，不计身份集合 |
| 隐藏重复：候选身份核心一致 | 3/3 | 身份判断有重复性 |
| 隐藏重复：全部核心字段一致 | 2/3 | 1 组仅 observation quality 和派生路由不一致 |
| 单例总标注用时中位数 | 33.5 秒 | 1 条 750.2 秒为明显会话停留异常值 |

私有自动分层只作为审核提示，不直接替代人工真值。下面所有分歧均已结合当前 mask、场景位置、候选多视角历史、3D 对齐和校正 GT 逐图复核。

## 4. 根因判断

### 4.1 `GRANULARITY_AMBIGUOUS` 被误用于“局部可见”

正确边界应为：

- `CLEAN_SINGLE_INSTANCE`：mask 中可确认是一个独立物理实例；物体不必完整出现在画面内。
- `BORDERLINE_SINGLE_INSTANCE`：仍能确认一个实例，但严重遮挡、出画、边界泄漏或极小可见区域降低了可靠性。
- `GRANULARITY_AMBIGUOUS`：真正无法稳定决定 part–whole 单位，例如同一 mask 究竟表示柜门还是整柜、桌腿还是整桌。它不是“只看见物体一部分”的同义词。

当前 9 条 `GRANULARITY_AMBIGUOUS` 中，逐图均能确认一个物理实例；其中较完整者应为 `CLEAN`，严重局部可见者应为 `BORDERLINE`。

### 4.2 同类别或同一组合系统不等于同一物理实例

`v2_trial_001/009` 和 `v2_trial_002` 的当前 observation 是特定位置的百叶窗面板；所选候选虽然同属 blinds，外观相似，甚至属于同一房间的窗户系统，但空间位置和面板边界不同，不能作为同一物理实例。三条应为 `NONE_SHOWN`，完整 `t^-` 地图也没有同实例节点，因此原 ATTACH 应改判为 `SHOULD_HAVE_BEEN_NEW`。

### 4.3 一次页面误选

`v2_trial_013` 实际提交选择 A，但备注已写明“候选选错了，应该是 C”。图像、3D 和校正 GT 均支持 C；这是提交错误，不是定义争议。

## 5. 逐例修订建议

原始 `event_labels.jsonl` 必须保持只读；下列内容只能写入单独 adjudication overlay，不能覆盖原答案。

| 案例 | 当前关键答案 | 建议答案 | 建议派生路由 | 依据 |
|---|---|---|---|---|
| `001` / `009` | `CLEAN` 或 `GRANULARITY`; A,B,C | `CLEAN`; `NONE_SHOWN`; full map=`NO_MATCHING_NODE_EXISTS` | `SHOULD_HAVE_BEEN_NEW` | 同一隐藏重复；当前 blinds#22 与候选面板位置不同 |
| `002` | `CLEAN`; B,C | `CLEAN`; `NONE_SHOWN`; full map=`NO_MATCHING_NODE_EXISTS` | `SHOULD_HAVE_BEEN_NEW` | 当前 blinds#30 与候选面板位置不同 |
| `003` | `GRANULARITY`; `NONE_SHOWN` | `CLEAN`; `NONE_SHOWN` | `CORRECT_NEW` | 独立 cabinet#2，frame 0 空图；只是视角下只显示该单元 |
| `004` | `GRANULARITY`; A | `BORDERLINE`; A | `WRONG_ATTACH_EXISTING` | blanket#86 严重局部可见，但身份 A 清楚 |
| `006` / `007` | `GRANULARITY`; B | `BORDERLINE`; B | `WRONG_ATTACH_EXISTING` | 同一隐藏重复；blanket#86 局部可见，身份 B 清楚 |
| `008` | `GRANULARITY`; D | `BORDERLINE`; D | `WRONG_NEW_FALSE_SPLIT` | sofa#9 在画面边缘仅露一部分，D 为同实例 |
| `012` / `015` | `GRANULARITY`; C,E | `BORDERLINE`; C,E | `WRONG_NEW_FALSE_SPLIT` | 同一隐藏重复；sofa#9 被历史拆为两个合法节点 |
| `013` | `GRANULARITY`; A；备注应为 C | `CLEAN`; C | `CORRECT_ATTACH` | cushion#69；备注、历史和 3D 均支持 C |

`005`、`010`、`011`、`014` 无需修改。`010` 的 background/fragment 和 `011` 的 mixed 排除判断正确。

若按上述建议建立覆盖层，15 条会覆盖五个路由单元和两个排除单元，3 组隐藏重复也会在 quality、身份集合与路由上全部一致。但这只是使用审核信息后的**仲裁结果**，不能反过来当作独立通过率。

## 6. R2 校准与全面标注门槛

### 6.1 先做的最小修订

1. 在页面 quality 选择前加入三问：
   1. mask 是否混入两个可分物体？是则 `MIXED`。
   2. 即使只看到局部，是否仍能确认一个独立实例？是则 `CLEAN/BORDERLINE`。
   3. 是否只有 part–whole 单位本身无法确定？只有此时才用 `GRANULARITY_AMBIGUOUS`。
2. 明示负例：同类别、同材质、同一窗户/家具组合、相邻摆放都不能证明同一实例；必须同时看位置、形状和多视角历史。
3. 提交时若备注中出现“选错/应为 X”但勾选不是 X，页面应阻止提交或二次确认。

### 6.2 新鲜 R2 校准集

不能只在已经看过的 15 条上重标后宣布通过。建议再取 12 个新基础案例 + 3 个隐藏重复：

- 五个路由单元各 2 条，共 10 条；
- 1 条 mixed、1 条 background/fragment；
- 3 条隐藏重复；
- 至少 4 条刻意包含遮挡/出画的单实例；至少 2 条为“同类别但空间位置不同”的身份负例；
- 尽量来自不同 causal group，避免把同一 cascade 当成多个独立证据。

R2 不采用机械百分比门槛，采用逻辑门槛：

- 不再把“仅局部可见”当成粒度歧义；
- 五个路由单元和两个排除单元均至少有一个可用、经仲裁确认的案例；
- 同类别不同实例不再系统性误合；
- 选择与备注无矛盾；
- 隐藏重复在 quality 和候选身份集合上一致；若不一致，必须能归因于证据不足，而不是规则摇摆。

全部满足后，状态改为 `GO_FULL_ANNOTATION`。

## 7. 全面标注的范围

全面标注不等于立刻人工遍历 room0 的 7,507 个路由事件。R2 通过后应分两部分：

1. 对高召回错误候选和 root/cascade 组做全量人工确认；
2. 对普通 ATTACH/NEW 按动作、时间段、历史长度和 observation quality 做概率抽样，才能估计自然错误率。

room0 只能作为机制与流程证据；正式结论仍需在一个从 frame 0 空图在线运行的未见场景上复验。当前试标不能用于错误率、论文主结果或跨场景泛化结论。

## 8. 产物

- 原始标签：`v2_schema_trial_room0/labels/event_labels.jsonl`（只读）
- 可复现复核指标：`v2_schema_trial_room0/annotation_review_metrics.json`
- 逐例证据拼图：`v2_schema_trial_room0/review_contact_sheets/`
- 复核脚本：`review_v2_trial_annotations.py`
- 拼图脚本：`make_v2_review_contact_sheets.py`
