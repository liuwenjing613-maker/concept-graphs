# ConceptGraphs：`ali-dev` / `main` 复现指标与论文对比

更新时间：2026-08-22  
论文：*ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning*（arXiv:2309.16650）  
论文链接：https://arxiv.org/pdf/2309.16650

## 1. 结论先行

| 问题 | 结论 | 证据强度 |
|---|---|---|
| `main` 是否复现了论文语义分割结果？ | **是，而且非常接近。** mAcc 仅差 **+0.02** 个百分点，F-mIoU 仅差 **+0.52** 个百分点。 | 强：同为 Replica 8 场景、同一官方指标口径，且两个论文指标同时接近。 |
| `ali-dev` 是否比论文 `main` 更好？ | 严格说，当前 8 场景数值来自 **`ali-dev` 的后续分支 `ali-my`**，不是严格 `ali-dev`。该派生结果相对论文 ConceptGraphs 的 mAcc / F-mIoU 分别高 **+4.46 / +4.79** 个百分点。 | 中强：指标和数据口径对齐，但代码不是严格 `ali-dev` HEAD。 |
| 关系边是否达到论文的 88% edge precision？ | **现有结果不能这样比较。** 论文的 88% 是 AMT 人工判断预测边是否合理；当前 3.26% 是用稀疏 ReplicaSSG GT 计算的闭世界 precision 下界，任务、GT、边集合和模型都不同。 | 不可直接比较。 |
| 当前关系边本身表现如何？ | 8 场景 paper 词表排名为 R@1 **55.03%**、R@3 **73.15%**、mR@1 **19.18%**、mR@3 **32.96%**；端点可匹配后 top-1 条件召回为 **67.77%**。 | 对 ReplicaSSG 协议有效，但论文没有报告这些指标。 |
| 对象识别是否优于 `main`？ | **不全面。** `ali-dev` 派生结果 mR@1 / mR@5 略高，但覆盖率和总体 R@1 / R@5 明显低于 `main`。 | 强：两者使用相同 ReplicaSSG 对象评测器。论文无对应数字。 |

最稳妥的总判断是：

> `main` 的论文语义分割结果已被高精度复现；`ali-dev` 基础上的 `ali-my` 在语义分割上取得明确提升，但对象覆盖与总体对象召回下降。关系边在 ReplicaSSG 排名指标上已有有效结果，但尚未完成论文 AMT 协议，不能宣称已复现或超过论文 88% edge precision。

## 2. 版本边界：哪些结果可以叫 `ali-dev`

| 名称 | 提交 / 来源 | 当前可用结果 | 正确称呼 |
|---|---|---|---|
| `main` | 仓库官方 post-map | Replica 8 场景语义、ReplicaSSG 对象 | `main` 复现结果 |
| 严格 `ali-dev` | `72f5962822b5e8678a446f367a06df1a977d2a4d` | 现存 room0 对象图；关系边已重映射并用严格 PCD 复评 | 严格 `ali-dev` room0 |
| `ali-my` | `bff233ff004939d2ecf4ac5546f87cb7b7b16e60` | Replica 8 场景语义、对象；8 场景关系推理的底图 | `ali-dev` 派生 / `ali-my` 结果 |

`ali-dev` 是 `ali-my` 的直接祖先，Git 关系为 `0 / 14`，没有分叉；新增提交主要涉及证据、审计、确定性和边界帧健壮性。基础算法一致，但 8 场景数值不能无条件改名为“严格 `ali-dev`”。

room0 上已完成严格同步门：72/72 对象和类别一致、bbox 最大差 1.82 mm、CLIP 最小余弦 0.999917、归一化帧集合和 896 个候选对完全一致。因此 room0 关系结果可以归入严格 `ali-dev`；其余 7 场景仍应标成 `ali-dev` 派生结果。

## 3. 可严格对比的论文指标：Replica 语义分割

### 3.1 指标映射

| 论文名称 | 本地评测名称 | 定义 | 是否可直接对比 |
|---|---|---|---|
| mAcc | mRecall | 各语义类别 `TP / (TP + FN)` 的宏平均 | 是 |
| F-mIoU | fmiou / fwIoU | 按 GT 类别频率加权的 IoU | 是 |
| — | mIoU、mPrecision、mF1、point accuracy | 本地额外输出 | 论文 Table II 未报告，只能在复现分支之间比较 |

本次 `main` 和 `ali-my` 均使用 Replica 全部 8 个标准场景、同一 Semantic GT、同一类别与提示词、同一精确 CPU `cKDTree k=1` 点匹配，并采用 `n_exclude=6`。`n_exclude=6` 排除 other、floor、wall、ceiling、door、window，与本仓库 `main` 的论文汇总口径一致。

### 3.2 论文值、`main` 复现值和 `ali-dev` 派生值

| 方法 | mAcc / mRecall | 相对论文 ConceptGraphs | F-mIoU / fwIoU | 相对论文 ConceptGraphs |
|---|---:|---:|---:|---:|
| 论文 ConceptGraphs | **40.63%** | 基准 | **35.95%** | 基准 |
| 本次复现 `main` | **40.65%** | **+0.02 pp** | **36.47%** | **+0.52 pp** |
| `ali-dev` 派生 `ali-my` | **45.09%** | **+4.46 pp** | **40.74%** | **+4.79 pp** |
| 派生结果相对复现 `main` | **+4.45 pp** | — | **+4.27 pp** | — |

复现误差的相对比例为：

- mAcc：`0.0191 / 40.63 = 0.047%`；
- F-mIoU：`0.5205 / 35.95 = 1.45%`。

这说明 `main` 的复现不是“趋势接近”，而是两个论文主指标都数值对齐；0.52 pp 的 F-mIoU 差异处于很小范围，可能来自具体 post-map 快照、依赖版本、浮点/近邻后端或数据落盘差异。

### 3.3 放回论文 Table II 的完整上下文

| 方法 | mAcc | F-mIoU | 来源 |
|---|---:|---:|---|
| CLIPSeg | 28.21% | 39.84% | 论文 |
| LSeg | 33.39% | 51.54% | 论文 |
| OpenSeg | 41.19% | **53.74%** | 论文 |
| MaskCLIP | 4.53% | 0.94% | 论文 |
| Mask2Former + Global CLIP | 10.42% | 13.11% | 论文 |
| ConceptFusion | 24.16% | 31.31% | 论文 |
| ConceptFusion + SAM | 31.53% | 38.70% | 论文 |
| ConceptGraphs | 40.63% | 35.95% | 论文 |
| ConceptGraphs-Detector | 38.72% | 35.82% | 论文 |
| **本次复现 `main`** | **40.65%** | **36.47%** | 本次复现 |
| **`ali-dev` 派生 `ali-my`** | **45.09%** | **40.74%** | 本次复现 |

解释：

- `main` 复现值与论文 ConceptGraphs 基本重合，验证了评测链路。
- `ali-dev` 派生结果的 mAcc 高于论文表中所有方法，较论文 OpenSeg 高 3.90 pp。
- `ali-dev` 派生结果的 F-mIoU 高于论文 ConceptGraphs 4.79 pp，也高于 ConceptFusion + SAM 2.04 pp；但仍低于 LSeg 10.80 pp、低于 OpenSeg 13.00 pp。
- 因而不能只用 mAcc 宣称“全面 SOTA”；它提升了类别平均准确率和本方法自身的频率加权 IoU，但论文中的 OpenSeg/LSeg 仍有更高 F-mIoU。

### 3.4 本地额外语义指标：派生结果 vs 复现 `main`

| 指标 | `ali-dev` 派生 `ali-my` | 复现 `main` | 差值 |
|---|---:|---:|---:|
| mIoU | **27.64%** | 24.85% | **+2.78 pp** |
| mRecall / mAcc | **45.09%** | 40.65% | **+4.45 pp** |
| mPrecision | **42.60%** | 36.31% | **+6.29 pp** |
| mF1 | **34.22%** | 30.41% | **+3.80 pp** |
| fwIoU / F-mIoU | **40.74%** | 36.47% | **+4.27 pp** |
| 点准确率 | **51.11%** | 43.26% | **+7.85 pp** |

8 个场景中，派生结果在 6 个场景的 mIoU 高于 `main`，仅 room0（-1.62 pp）和 room1（-1.79 pp）较低。最大提升来自 office2（+5.41 pp）和 office0（+4.71 pp）。这表明提升不是单一场景偶然值，但仍存在 room0/room1 回退。

## 4. 论文没有对应数字的对象指标

ConceptGraphs 论文 Table I 的 “Node Precision” 是人工判断节点描述是否准确，Table II 是点级语义分割；论文没有报告 ReplicaSSG 的对象 R@1、R@5、mR@1、mR@5 或几何覆盖率。因此下表只能比较两个本地复现分支，不能与论文直接做数值差。

| 指标 | `ali-dev` 派生 `ali-my` | 复现 `main` | 差值 | 判断 |
|---|---:|---:|---:|---|
| 预测对象数 | 381 | 612 | -231 | 派生地图更紧凑 |
| 几何匹配数 / 248 GT | 138 | 175 | -37 | `main` 更好 |
| 几何覆盖率 | 55.65% | **70.56%** | **-14.92 pp** | 明显回退 |
| R@1 | 17.74% | **20.56%** | **-2.82 pp** | 回退 |
| R@5 | 43.95% | **55.65%** | **-11.69 pp** | 明显回退 |
| mR@1 | **32.32%** | 31.59% | **+0.74 pp** | 基本持平、略高 |
| mR@5 | **55.30%** | 54.60% | **+0.70 pp** | 基本持平、略高 |

对象结果说明：派生地图用更少节点获得了接近甚至略高的类别均衡召回，但漏掉了更多 GT 实例，总体覆盖和总体召回不如 `main`。因此合适的表述是“更紧凑、长尾均衡性持平略高，但实例覆盖不足”，而不是“对象识别全面提升”。

## 5. 关系边：为什么不能把 3.26% 与论文 88% 相减

### 5.1 论文 Table I 报告的是什么

论文对 7 个 Replica 场景的节点和边做 Amazon Mechanical Turk 人工评估，每项由 3 名标注者判断并多数投票。论文 ConceptGraphs 的平均 Node Precision 为 **0.71**，Edge Precision 为 **0.88**；ConceptGraphs-Detector 分别为 **0.61 / 0.91**。论文没有使用 office4，也没有报告关系 recall、mRecall 或 F1。

| 场景 | 论文 CG Node Precision | 有效对象 | 重复对象 | 论文 CG Edge Precision |
|---|---:|---:|---:|---:|
| room0 | 0.78 | 54 | 3 | 0.91 |
| room1 | 0.77 | 43 | 4 | 0.93 |
| room2 | 0.66 | 47 | 4 | 1.00 |
| office0 | 0.65 | 44 | 2 | 0.88 |
| office1 | 0.65 | 23 | 0 | 0.90 |
| office2 | 0.75 | 44 | 3 | 0.82 |
| office3 | 0.68 | 60 | 5 | 0.79 |
| **平均** | **0.71** | — | — | **0.88** |

### 5.2 当前关系复现报告的是什么

当前 8 场景关系评测使用 ReplicaSSG 的 149 条稀疏关系 GT，固定 7 类谓词：`on / in / near / above / under / attached to / with`。其排名指标为：

此前 `main` 8 场景建图设置了 `make_edges=false`，没有运行关系推理；评测器中出现的 predicate 零值只是空关系输入产生的结构性零，**不是 `main` 的关系性能**。因此本节只能报告 `ali-dev` 派生关系系统及严格 room0，当前不存在有效的“`ali-dev` 关系 vs `main` 关系”数值对照。

| 范围 | GT 关系 | 端点匹配 | 候选覆盖 | R@1 | R@3 | mR@1 | mR@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 场景全部 | 149 | 121 | 121 | **55.03%** | **73.15%** | **19.18%** | **32.96%** |
| 给定端点/候选后的条件结果 | 121 | 121 | 121 | **67.77%** | **90.08%** | — | — |

0.99 阈值导出的实际边为：

| 输出头 | 边数 | TP | 闭世界 Precision | 映射 GT Recall | F1 |
|---|---:|---:|---:|---:|---:|
| paper 7 类词表 | 399 | 13 | **3.26%** | **10.74%** | **5.00%** |
| `ali-dev` 原生兼容头 | 150 | 11 | **7.33%** | **9.09%** | **8.12%** |

这里的低 precision 不能解释为“绝大多数边在人类看来是错的”。ReplicaSSG 的关系标注非常稀疏：只要预测边没有出现在封闭 GT 中，就会被计作 FP，即使它可能是合理但未标注的关系。因此它是闭世界下界和阈值诊断，不是论文人工 precision 的复现。

### 5.3 两种关系评测的协议差异

| 维度 | 论文 Table I | 当前 ReplicaSSG 关系评测 | 影响 |
|---|---|---|---|
| 判断者 | 3 名 AMT 人工，多数票 | 稀疏封闭 GT 自动匹配 | 论文允许合理开放词汇边；当前未标即负例 |
| 场景 | 7 场景，无 office4 | 8 场景 | 样本范围不同 |
| 边生成 | 3D 位置建图，最小生成树剪枝后交给 LLM | 4,013 个候选对并集，阈值导出 | 边数量和先验完全不同 |
| 关系词表 | 开放词汇自然语言 | 固定 7 类 + none | 输出空间不同 |
| 主模型 | LLaVA + GPT-4 `gpt-4-0613` | `gpt-5.6-sol` | 模型不同 |
| 指标 | Edge Precision | R@1/R@3/mR + 闭世界 P/R/F1 | 数学定义不同 |
| 召回 | 不报告 | 报告 | 论文的 88% 不代表高覆盖率 |

所以以下写法是错误的：

> “ali-dev edge precision 为 3.26%，比论文 88% 低 84.74 个百分点。”

正确写法是：

> “论文在 7 场景 AMT 开放词汇人工评估中报告 88% Edge Precision；我们的 8 场景 ReplicaSSG 评测报告 R@1 55.03%、R@3 73.15%，以及 0.99 阈值下 3.26% 的稀疏 GT 闭世界 precision。两者协议不同，后者不能作为前者的复现值。”

### 5.4 严格 `ali-dev` room0

只有 room0 同时具备严格 `ali-dev` PCD 和已同步关系边：

| 指标 | 严格 `ali-dev` room0 |
|---|---:|
| GT 对象 / 预测对象 / 几何匹配 | 59 / 72 / 31 |
| GT 关系 / 端点匹配 / 候选覆盖 | 25 / 16 / 16 |
| paper 头 R@1 / R@3 | **56.00% / 64.00%** |
| paper 头 mR@1 / mR@3 | **37.50% / 45.83%** |
| paper 头 0.99 边 / TP / P / 映射 R / F1 | 55 / 5 / **9.09% / 31.25% / 14.08%** |
| `ali-dev` 兼容头边 / TP / P / 映射 R / F1 | 16 / 3 / **18.75% / 18.75% / 18.75%** |

论文 room0 的人工 Edge Precision 是 91%，但它与上表 9.09% / 18.75% 仍不可直接比较。严格 room0 的价值在于证明节点同步和 ReplicaSSG 自动评测可复现，不代表已经复现论文的人评。

### 5.5 当前关系模型的真实强弱项

| 谓词 | GT | R@1 | R@3 | 判断 |
|---|---:|---:|---:|---|
| near | 79 | **74.68%** | **84.81%** | 当前主力 |
| on | 40 | **55.00%** | **55.00%** | 中等，top-3 无额外收益 |
| with | 22 | 4.55% | **90.91%** | 排名校准明显不足，常在 top-3 但不在 top-1 |
| in | 3 | 0% | 0% | 未解决 |
| above | 2 | 0% | 0% | 未解决 |
| under | 2 | 0% | 0% | 未解决 |
| attached to | 1 | 0% | 0% | 端点也未匹配 |

整体 R@3 73.15% 主要由 `near`、`on` 和 `with` 支撑；mR@3 只有 32.96%，说明长尾谓词仍是主要短板。若论文目标强调关系多样性，应优先改善 `with` 的 top-1 校准以及 `in/above/under/attached to`，而不是继续只调全局阈值。

## 6. 什么已经复现、什么还没有

| 项目 | 状态 | 可以对外使用的结论 |
|---|---|---|
| 论文 Table II：`main` mAcc | 已高精度复现 | 40.65% vs 40.63%，+0.02 pp |
| 论文 Table II：`main` F-mIoU | 已高精度复现 | 36.47% vs 35.95%，+0.52 pp |
| `ali-dev` 派生语义提升 | 已测 | mAcc / F-mIoU 相对论文 +4.46 / +4.79 pp |
| ReplicaSSG 对象指标 | 已测，但论文无对应值 | 只能比较派生分支与复现 `main` |
| ReplicaSSG 关系 R/mR | 已测，但论文无对应值 | 作为新增 closed-vocabulary benchmark 报告 |
| `main` 关系边 | 未运行 | `make_edges=false` 的零值无效，不能拿来与 `ali-dev` 比较 |
| 论文 7 场景 Node Precision | 未复现 | 需要同口径 3 人人工审核节点描述 |
| 论文 7 场景 Edge Precision | 未复现 | 需要 MST/论文边集和同口径 3 人人工审核 |
| 8 场景严格 `ali-dev` | 未完成 | 当前只有严格 room0；其他 7 场景来自 `ali-my` |

## 7. 论文或答辩中建议采用的表述

推荐主结论：

> 在 Replica 8 场景和官方语义评测口径下，我们将 ConceptGraphs `main` 的 mAcc / F-mIoU 复现为 40.65% / 36.47%，与论文报告的 40.63% / 35.95% 分别相差 +0.02 / +0.52 个百分点。基于 `ali-dev` 的后续 `ali-my` 分支达到 45.09% / 40.74%，相对论文值提高 +4.46 / +4.79 个百分点。与此同时，ReplicaSSG 对象覆盖率由 `main` 的 70.56% 降至 55.65%，表明语义分割提升并未转化为更高的实例覆盖。

推荐关系结论：

> 在独立的 ReplicaSSG 8 场景关系评测中，模型取得 R@1 55.03%、R@3 73.15%、mR@1 19.18% 和 mR@3 32.96%。这些指标不是 ConceptGraphs 论文 AMT Edge Precision 的复现；论文 88% 需要按开放词汇、论文边生成和三人多数票协议另行评估。

不建议写：

- “8 场景严格 `ali-dev` 达到 45.09 / 40.74”——目前严格 `ali-dev` 只有 room0。
- “关系边精度只有 3.26%，远低于论文 88%”——指标口径不等价。
- “全面超过 `main`”——对象覆盖率、R@1、R@5 均回退。
- “关系 R@3 73.15% 接近论文 88%”——R@3 与人工 precision 不是同一个量。

## 8. 若要完成真正的论文关系边复现

最小闭环应为：

1. 固定论文 Table I 的 7 场景，office4 不进入论文主表。
2. 在严格 `ali-dev` 上补齐另外 6 个场景，避免用派生分支冒充严格分支。
3. 按论文逻辑生成边：对象 caption + 3D 位置、MST 剪枝、开放词汇关系描述；同时记录与当前候选并集方案的差异。
4. 尽可能使用论文模型 `gpt-4-0613`；不可用时把模型替换列为明确实验变量，不能称完全复现。
5. 对全部预测节点和边做 3 人独立人工标注，多数票汇总 Node Precision、Valid Objects、Duplicates、Edge Precision。
6. ReplicaSSG R/mR 继续保留为补充实验，用于揭示论文 precision 无法反映的召回与长尾谓词能力。
7. 主表只放同口径值；不同协议放在补充表，并标注“not directly comparable”。

## 9. 本地证据入口

- `main` / `ali-my` 8 场景对齐评测：`docs/ALI_MY_PAPER_MAIN_ALIGNED_8SCENE_EVALUATION_20260821.md`
- `ali-dev` 关系边完整评测：`docs/ALI_DEV_RELATION_EDGES_EVALUATION_20260822.md`
- `main` 语义原始 JSON：`staging_remote/ali_my_paper_main_aligned_20260821/semseg_main_8scene_cdtree/semseg_results.json`
- `ali-my` 语义原始 JSON：`staging_remote/ali_my_paper_main_aligned_20260821/semseg_8scene_cdtree/semseg_results.json`
- 对象汇总 JSON：`staging_remote/ali_my_paper_main_aligned_20260821/summary/replicassg_8scene_summary.json`
- 关系 8 场景汇总：`staging_remote/evaluation_summary_threshold_0p99.json`
- 严格 `ali-dev` room0：`staging_remote/evaluation_room0_threshold_0p99_strict_ali_dev_map.json`

## 10. 最终判断

1. **`main` 论文语义指标：复现成功。**
2. **`ali-dev` 派生分支：语义显著优于论文 `main`，但实例覆盖不足。**
3. **严格 `ali-dev`：当前只能对 room0 的关系结果作严格归属。**
4. **关系边：ReplicaSSG 自动评测已完成，论文 AMT Edge Precision 尚未复现。**
5. **当前最值得补的实验不是再调 0.99 阈值，而是严格 7 场景 `ali-dev` + 论文式人工节点/边审核。**
