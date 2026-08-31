# 简单体素触发 V0 复审：最终结论

日期：2026-08-31  
复审状态：**MODIFY——保留简单体素账本；否决原统一分数；只保留 association 空间分区信号作为待验证假设。**

## 1. 最终回答

不能完全否决体素判断。

上一轮的 `STOP` 结论过强，因为它把三件事混在了一起：体素账本、A/B/C 统计方法、自动 GT 评测。复审后应拆成以下结论：

| 层级 | 复审结论 |
|---|---|
| 体素坐标、observation 去重、`seen_count/label_hist/obs_ids` 序列化 | 实现正确，全部结构不变量通过 |
| 每个 observation 直接投一票 | 不够合理；同一帧重叠 mask 造成相关票重复 |
| 原 `semantic/fragment/pair` 百分位取最大值 | 否决；高分饱和且掩盖有效单项 |
| 原统一 GT 标签 | room0 明显受背景/unknown 和 observation 数量偏差影响 |
| 体素统一检测 mask + association + split | 当前不成立 |
| 体素对 association 的局部空间冲突提示 | 有探索性信号，不能否决，但尚未达到上线证据 |

因此准确表述不是“体素无用”，而是：

> **当前简单体素账本值得保留；原统一触发器无效；第二标签形成连续3D区域可能提示一部分跨类别 association/merge 错误，需要更可靠 GT 和 holdout 验证。**

## 2. 复审做了什么

在原 room0 / office0 两个400帧冻结结果上新增：

1. 逐数组核验体素 key、坐标、offset、`seen_count`、`label_hist`、`obs_ids`；
2. 将同一帧在同一体素中的多 observation 总权重归一为1，重算 frame-balanced 统计；
3. 将 `wall/floor/ceiling/unknown/undefined` 从实例 association GT 中排除；
4. 增加更保守 GT 阈值敏感性；
5. 去除 object 体素数和 observation 数带来的规模混杂；
6. 固定最小 observation 支持量后重新评价；
7. 用独占最近 GT 的3D归因和官方 O3 一对一匹配做独立交叉检查；
8. 对选定指标做5000次分层 bootstrap 和10000次置换检验。

复审脚本总执行时间约64.9秒，不包含后续统计 bootstrap。

## 3. 体素构造审计

### 3.1 正确的部分

三种尺度、两个场景全部满足：

- `seen_count == obs_offsets` 对应长度；
- 每个 observation 在一个体素中最多登记一次；
- `label_hist` 与 `obs_ids -> observation label` 精确一致；
- packed voxel key 与三维坐标精确一致；
- observation、最终 object membership、mask 和 label 闭环仍然成立。

所以没有发现坐标错位、同 observation 在同体素重复计数或 label histogram 写错等实现 bug。

### 3.2 不合理的部分：同帧票并不独立

5 cm 下：

| 构造问题 | room0 | office0 |
|---|---:|---:|
| 同帧额外 observation link / 全部 link | **19.8%** | **21.1%** |
| 至少有一次同帧多 observation 的体素 | 43.2% | 48.9% |
| 同帧出现多个不同 label 的体素 | **42.6%** | **48.7%** |

原设计中一个 observation 一票在代码上正确，但多个高度相关、相互重叠的 mask 可以在同一帧重复影响同一体素，违反“多视角长期证据”的独立性假设。

建议仍保持简单 schema，但改变聚合规则：

```text
obs_ids     继续完整保存
seen_count  改为 unique frame count
label_hist  每帧总权重固定为1；同帧多个 label 平分这一票
```

这不需要增加体素字段。

## 4. GT 评测审计

### 4.1 背景和 unknown 污染

只保留 GT 支持率至少80%、top GT 为可靠前景实例的 observation 后：

| 统一错误标签 | room0 | office0 |
|---|---:|---:|
| 原标签 | 45/69 | 24/33 |
| 严格前景标签 | **30/69** | **24/33** |

room0 有15个原“错误”在排除结构表面和 unknown 后消失，说明原 AUROC 0.331 的标签确实受到明显污染。office0 基本不变。

### 4.2 repeated 定义带来 observation 数偏差

原标签要求错误 observation 至少出现2次，所以 observation 少的 object 结构上不可能成为正例。仅用 `num_detections` 排序严格统一错误已经达到：

| 场景 | AUROC |
|---|---:|
| room0 | **0.794** |
| office0 | **1.000** |

因此上一轮“非体素 object label 熵很强”的结论也被高估：

| 严格统一错误 | room0 | office0 |
|---|---:|---:|
| 非体素 label 熵原 AUROC | 0.692 | 0.938 |
| 去除 observation 数和 object 大小后 | **0.566** | **0.657** |

它仍可能有信息，但不能再声称已得到强而稳定的 trigger。

### 4.3 mixed-mask 自动 GT 失去负例区分力

在至少5个可靠 observation 的 object 中，以 mixed fraction ≥10% 标记 mask 错误：

| 场景 | 正例 / 可评 object |
|---|---:|
| room0 | **22/23** |
| office0 | **19/22** |

几乎所有充分观察的 object 都被判为 mixed-mask 正例，说明当前像素 purity 阈值同时吸收了边界、遮挡和数据对齐误差，不能作为评测 mask trigger 的最终 GT。

### 4.4 独立3D GT 覆盖不足

独占最近 GT 归因只有 room0 8个、office0 5个可靠前景 object。官方 O3 的2 cm一对一 Hungarian IoU≥0.25匹配也只有：

- room0：8/72；
- office0：1/35。

因此独立3D评测中出现的高 AUROC 是极小样本现象，已排除出最终支持证据。

语义标签评测同样病态：可靠且 canonical label 可评的 room0 12个 object 全部语义错误；office0 为5/8，无法形成稳定正负集。

## 5. 统计方法审计

### 5.1 原统一分数仍然应否决

`max(semantic percentile, fragmentation percentile, pair percentile)` 会让近40%的 object 得分超过0.9，正确 object 只要任一噪声项很高就被推到前列。原统一分数的失败不是体素账本本身的充分反证。

### 5.2 mask / unified 的表面信号主要是规模混杂

按帧平衡后的平均体素 disagreement 对严格统一错误：

| 状态 | room0 | office0 |
|---|---:|---:|
| 原始 AUROC | 0.638 | 0.833 |
| 去除 object 大小和 observation 数后 | **0.547** | **0.546** |

对保守 mixed-mask：

| 状态 | room0 | office0 |
|---|---:|---:|
| 原始 AUROC | 0.630 | 0.861 |
| 规模去混杂后 | **0.541** | **0.611** |

所以目前没有证据证明体素比“这个 object 被看了多少次”更准确地触发 mask 错误。

### 5.3 association 存在值得保留的空间信号

最稳定的候选是：

```text
second-label largest 3D connected-region fraction
```

含义是：不是看 object 总体 label 熵，而是检查第二种稳定 label 是否在 object 内形成了一整块连续三维区域。

在去除 object 大小和 observation 数后，保守 association GT 上：

| 场景 | N / 正例 | AUROC | 95% bootstrap区间 | AP增益 |
|---|---:|---:|---:|---:|
| room0 | 16 / 6 | **0.683** | 0.400–0.933 | +0.131 |
| office0 | 22 / 15 | **0.810** | 0.600–0.971 | +0.226 |

三个尺度方向一致：

| 尺度 | room0 | office0 |
|---|---:|---:|
| 2.5 cm | 0.700 | 0.810 |
| 5 cm | 0.683 | 0.810 |
| 10 cm | 0.750 | 0.867 |

固定最小支持量、取消“至少2次错误”的定义性偏差后：

| 协议 | room0 | office0 |
|---|---:|---:|
| ≥3 pure obs，wrong fraction≥0.10 | 0.697 | 0.810 |
| ≥3 pure obs，wrong fraction≥0.20 | 0.697 | 0.692 |
| ≥5 pure obs，wrong fraction≥0.10 | 0.660 | 0.783 |
| ≥5 pure obs，wrong fraction≥0.20 | 0.660 | 0.631 |

这说明体素中可能确实存在 association 空间分区信号。但是：

- room0 置信区间仍跨过0.5；
- continuous wrong-fraction 的 Spearman 相关不显著；
- 样本只有16–22个；
- 当前信号主要适用于第二 label 与主 label 不同的跨类别错误；
- 两个同类实例错误融合时仍可能完全看不出来。

所以它是“值得做下一次最小验证的候选”，不是已经成立的方法。

`mean owner entropy` 在部分设置下有效，但 office0 对阈值敏感，暂不冻结为主指标。

## 6. 更新后的最终判定

### 被否决

- 原 A/B/C 百分位最大值统一触发器；
- 用当前自动 GT 声称统一覆盖 mask、association、split；
- 用当前两个场景声称非体素 label 熵已是可靠 trigger；
- 直接将体素分数接入在线 ticket / VLM。

### 被保留

- `seen_count + label_hist + obs_ids` 简单账本结构；
- 按帧平衡投票；
- 第二标签最大连续3D区域，作为跨类别 association/merge 候选；
- `obs_ids` 回查当前 owner，从而分析体素中的多 owner 冲突。

### 仍未证明

- mask trigger 的独立体素增益；
- same-class false merge；
- false split 的可靠候选召回；
- 对真实人工 actionable error 的精度；
- 在线发现帧 `d` 和触发时延。

## 7. 下一步最小实验

不要继续堆更多体素字段。只验证一个假设：

```text
按帧平衡的简单体素账本
        ↓
第二标签最大连续3D区域
        ↓
跨类别 association / false-merge review
```

建议协议：

1. 最少5个独立 frame 支持后才可触发；
2. 开发场景冻结阈值，再使用一个新 holdout 场景；
3. 高分、低分各人工审核20个，并按 observation 数匹配；
4. 单独加入已知 same-class merge，明确测量盲区；
5. 或在严格在线流中注入已知 association corruption，记录 `s <= d`；
6. 只有两场景和 holdout 都达到 AUROC≥0.70、AP lift≥0.10、Top/Bottom≥2倍，才进入在线 revision ticket。

这里存在一个必须由用户确认的评测选择：下一轮应以“人工 actionable association 标签”为主，还是以“受控在线 corruption”为主。两者回答的问题不同，不能擅自混用。

## 8. 产物

服务器：

```text
/home/chenkejun/beauty/conceptgraphs/results/experiments/voxel_trigger_reaudit_20260831/output_v3
/home/chenkejun/beauty/conceptgraphs/code/experiments/voxel_trigger_reaudit_20260831
```

本地同步：

```text
results/voxel_trigger_reaudit_20260831
```

主要文件：

- `construction_audit.csv`
- `metrics.csv`
- `statistical/selected_uncertainty.csv`
- `support_controlled/support_controlled_metrics.csv`
- `statistical/02_selected_uncertainty.png`

