# 简单体素错误触发 V0：room0 / office0 最终实验总结

日期：2026-08-31  
最终判定：**STOP 当前统一体素触发器；保留体素账本实现，不进入在线修复集成。**

## 1. 最重要结论

本轮已经把讨论中的简单设计完整实现并在两个 400 帧 ali-dev 冻结场景上跑通：

- 每个体素的逻辑内容只有 `seen_count + label_hist + obs_ids`；
- room0 和 office0 均重建了逐帧 observation 到 3D 体素的证据；
- 计算了 object 内部语义冲突、3D 碎裂、邻近 object 重复三类指标；
- 使用 2.5、5、10 cm 三种体素尺度；
- 同时比较了不做体素化的 object 成员 label 熵基线；
- 使用 Replica 实例 sidecar 只做事后错误归因，GT 没有进入体素内容或异常分数。

技术实现是可行的，数据也很轻量；但当前异常指标不能稳定提高错误触发精度。5 cm 主尺度上：

| 任务 / 分数 | room0 | office0 |
|---|---:|---:|
| 统一重复错误 / combined AUROC | **0.331** | **0.451** |
| 统一重复错误 / AP - 错误率 | **-0.097** | **-0.031** |
| mixed-mask / 体素语义 AUROC | 0.524 | 0.450 |
| association / 体素语义 AUROC | 0.363 | 0.644 |
| association / 3D 碎裂 AUROC | 0.529 | 0.583 |
| false-split pair / label×邻近 AUROC | 无法定义¹ | 0.396 |

¹ room0 只有 12 个 GT 可靠的邻近候选，且 12 个全是正例，没有可用于排序评测的可靠负例。

因此不能声称“简单体素统计能够统一兼顾 mask 与 association 错误”。当前版本的正式结论是：

> **体素账本可以建立，但当前 A/B/C 指标没有把账本中的信息转化成可靠触发；不应继续把三项分数堆叠或直接接入修复系统。**

一个有价值的反向结果是：不做体素化的 object 成员 label 熵，对“重复 mixed-mask”明显更强：

| repeated mixed-mask | room0 | office0 |
|---|---:|---:|
| 错误率 | 31/69 = 0.449 | 22/33 = 0.667 |
| 非体素 label 熵 AUROC | **0.702** | **0.884** |
| 非体素 label 熵 AP | **0.600** | **0.908** |
| AP - 错误率 | **+0.150** | **+0.241** |
| Top-20% / Bottom-20% | **4.50×** | Bottom-20% 为 0 |
| 体素语义 AUROC | 0.524 | 0.450 |

这说明下一步更合理的最小实验不是继续增加体素字段，而是先验证一个更简单的 **object observation-label consistency trigger**。

## 2. 实验回答了什么

本轮只回答以下问题：

1. 能否从 ali-dev 已跑完的 room0 / office0 结果恢复所有原始 observation 并建立简单全局体素账本？
2. 简单体素统计能否比 object 成员 label 统计更准确地触发 mask / association / split 错误？
3. 结果对 2.5、5、10 cm 是否稳定？
4. 计算量、存储量和可视现象是什么？

本轮不是在线 mapper 集成实验，也没有执行任何自动修复。它是冻结地图上的离线 V0 机制筛选，符合“先用 1–2 场景做高信息量最小实验，再决定是否扩大”的目标。

## 3. 输入与完整性校验

| 项目 | room0 | office0 |
|---|---:|---:|
| 原始帧 | 400 | 400 |
| B0 object 数 | 72 | 35 |
| 前景 object 数 | 69 | 33 |
| 原始 detections | 12,514 | 6,111 |
| 2D 过滤后 observations | 7,524 | 3,106 |
| 3D 接受 observations | 7,507 | 3,106 |
| 最终成员唯一 observations | 7,507 | 3,106 |
| mask 精确恢复 | 7,507 / 7,507 | 3,106 / 3,106 |
| label 精确恢复 | 7,507 / 7,507 | 3,106 / 3,106 |
| 缺失 final member | 0 | 0 |
| 总执行时间² | 343.8 s | 182.8 s |

² 包含从 detection cache 重建 3D observation、GT 事后归因和六份体素聚合，不是在线单帧延迟。

room0 的原始 B0 序列化中有 8 条同一 owner、同一 mask、同一 label 的重复成员登记；构建账本时按 `(frame, filtered_mask_idx)` 去重，避免同一个 observation 重复投票。office0 没有重复。

## 4. 简单体素内容

逻辑上每个体素只保存：

```text
Voxel
├─ seen_count    覆盖该体素的不同 observation 数
├─ label_hist    原始 observation label 的计数
└─ obs_ids       覆盖该体素的 observation ID
```

当前 object map 另外保存：

```text
Object -> voxel_ids
```

NPZ 为了高效存取，额外使用坐标和 CSR offset 数组；这些只是索引结构，不是新的证据字段。

一个真实 5 cm 体素示例：

```text
scene: room0
voxel_coord: [117, 4, -16]
seen_count: 623
label_hist:
  pillow: 218
  armchair: 217
  sofa chair: 119
  chair: 63
  folded chair: 5
  cushion: 1
obs_ids: 共 623 个，账本中完整保留
```

`seen_count` 是 observation 数，不是帧数；同一帧中不同 mask 可以覆盖同一体素，所以可能大于 400。

## 5. 5 cm 主尺度账本统计

| 指标 | room0 | office0 |
|---|---:|---:|
| 全局体素数 | 40,416 | 20,431 |
| voxel-observation links | 2,656,932 | 1,020,198 |
| `seen_count` 中位数 | 22 | 25 |
| `seen_count` 均值 | 65.7 | 49.9 |
| 多 label 体素比例 | 45.8% | 55.6% |
| disagreement ≥ 0.25 体素比例 | 25.3% | 40.7% |
| 压缩 NPZ | 1.07 MB | 0.40 MB |
| 数组展开内存 | 12.23 MB | 5.09 MB |
| `obs_ids` 占展开内存 | 82.9% | 76.5% |

结论：表示本身并不重。真正的主要开销来自 `obs_ids`，而不是 label histogram。

三种尺度的体素数：

| 尺度 | room0 | office0 |
|---|---:|---:|
| 2.5 cm | 150,048 | 73,037 |
| 5 cm | 40,416 | 20,431 |
| 10 cm | 10,396 | 5,427 |

尺度变粗后体素数和文件大小快速下降，但语义冲突的触发性能没有获得跨场景稳定提升。

## 6. 指标定义

### A. object 内部体素语义冲突

从体素 `label_hist` 推导：

- 体素多数 label 的空间熵；
- 每个体素的非多数票比例；
- 第二 label 最大连续区域占 object 的比例。

三者先在场景内转为百分位，取最大值作为 `semantic_conflict_score`。

### B. object 3D 碎裂

在 object 的体素集合上做 26 邻域连通分量，主要使用：

```text
fragmentation = 1 - largest_component_voxels / object_voxels
```

再转为场景内百分位。

### C. 邻近 object 重复 / false split

先保留 bbox 间隔不超过 2 个体素的 object 对，再计算：

- 精确体素 IoU；
- 1 / 2 邻域接触比例；
- 两个 object observation-label histogram 的 cosine similarity。

当前 pair 分数是：

```text
label similarity × spatial contact
```

### Unified

```text
combined = max(semantic percentile,
               fragmentation percentile,
               duplicate-pair percentile)
```

### 非体素基线

直接对最终 object 的 observation `class_id` histogram 求熵，再转为百分位。它不使用空间体素。

## 7. GT 与评测修正

### 7.1 被排除的粗 GT 体素交集诊断

第一次冻结运行中，预测 object 与每个 GT object 分别体素化后逐个求交。同一个粗体素可能同时包含两个相邻 GT 实例的点，因此一个预测体素会被多个 GT object 重复计数。

该问题在 5 cm 下把冻结的 identity-error 比例夸大到 room0 61/69、office0 25/33。它只影响 GT 事后标签，不影响体素内容和异常分数。

这套 many-to-many GT voxel overlap 结果已完整保留用于审计，但明确排除出最终性能结论。

### 7.2 最终采用的 observation-sidecar 归因

每个原始 2D mask 与同帧 Replica 实例图求交：

- `mask_mixed`：top GT purity < 0.8，或第二 GT 占 mask ≥ 0.1；
- repeated mask error：同一 object 至少 2 个 mixed masks，且占 GT 可评 observation ≥ 5%；
- pure observation：非 mixed 且 purity ≥ 0.8；
- repeated association error：至少 2 个 pure observations 指向 object 主 GT 之外的实例，且占 pure observations ≥ 5%；
- false split：两个 object 均有至少 2 个 pure observations、主 GT 占比 ≥ 0.5，并指向同一非背景 GT 实例。

GT 只在所有分数生成后加载，用于评价。

### 7.3 pair 未知样本不当负例

如果一个 pair 的任一端没有可靠前景 GT 归属，该 pair 标记为 `UNRESOLVED`，不计入 AUROC/AP 分母。5 cm 下：

| pair 覆盖 | room0 | office0 |
|---|---:|---:|
| 全部邻近候选 | 122 | 60 |
| GT 可靠可评候选 | 12 | 20 |
| GT 未决候选 | 110 | 40 |
| 可评 false-split 正例 | 12 | 7 |

room0 的 12 个可评候选全部是正例，所以无法计算排序 AUROC；不能把其余 110 个未知候选强行当作负例。

## 8. 错误类型数量

5 cm 下 observation-sidecar 归因：

| object 级错误证据 | room0（69） | office0（33） |
|---|---:|---:|
| 至少一个 mixed mask | 44 | 25 |
| repeated mixed mask | 31 | 22 |
| repeated association | 27 | 18 |
| false-split incident | 16 | 12 |
| 主 GT 为 wall/floor/ceiling 的疑似 spurious object | 28 | 4 |
| mask / association / split 并集 | 45 | 24 |
| 再加入 spurious | 53 | 24 |

这些是可重复的自动 GT 诊断，不等同于人工确认的“可行动真错误”。尤其 room0 有较多 observation 主 GT 为背景，说明边界、遮挡和 sidecar 覆盖本身也会影响自动标签。

## 9. 详细性能

### 9.1 Mask 错误

| 5 cm repeated mixed-mask | room0 | office0 |
|---|---:|---:|
| 体素语义 AUROC | 0.524 | 0.450 |
| 体素语义 AP / 错误率 | 0.467 / 0.449 | 0.619 / 0.667 |
| 体素 Top-20% / Bottom-20% | 1.00× | 1.00× |
| 非体素 label 熵 AUROC | **0.702** | **0.884** |
| 非体素 AP / 错误率 | **0.600 / 0.449** | **0.908 / 0.667** |
| 非体素 Top-20% / Bottom-20% | **4.50×** | **Bottom=0** |

体素投票没有提高 mask 错误触发；它反而把不同 observation 的差异压成局部多数票，损失了 object 成员 label 分布中的信号。

### 9.2 Association 错误

| 5 cm repeated association | room0 | office0 |
|---|---:|---:|
| 可评 object | 44 | 24 |
| 正例 | 27 | 18 |
| 体素语义 AUROC | 0.363 | 0.644 |
| 体素语义 AP - 错误率 | -0.088 | +0.063 |
| 碎裂 AUROC | 0.529 | 0.583 |
| 碎裂 AP - 错误率 | +0.043 | +0.042 |
| 非体素 label 熵 AUROC | 0.327 | 0.838 |

结果存在明显场景反转：office0 有信号，room0 相反。不能冻结统一 association trigger。

### 9.3 False split

可靠 GT false-split 对总数及邻近候选召回：

| 尺度 | room0 | office0 |
|---|---:|---:|
| 2.5 cm | 6/39 = 15.4% | 5/25 = 20.0% |
| 5 cm | 12/39 = 30.8% | 7/25 = 28.0% |
| 10 cm | 17/39 = 43.6% | 9/25 = 36.0% |

5 cm 下 office0 可评 pair 排序：

- N=20，正例=7，错误率=0.350；
- `label × r1 contact`：AP=0.329，AUROC=0.396；
- `label × r2 contact`：AP=0.324，AUROC=0.253。

room0 可评候选全为正例，无法得到 AUROC。当前 pair 设计同时存在“候选召回不足”和“候选内排序不足”。

### 9.4 Unified

| 5 cm combined | room0 | office0 |
|---|---:|---:|
| mask/association/split 错误率 | 0.652 | 0.727 |
| AP | 0.555 | 0.696 |
| AP - 错误率 | **-0.097** | **-0.031** |
| AUROC | **0.331** | **0.451** |
| Top-20% / Bottom-20% | **0.455×** | **1.00×** |

加入疑似 spurious object 后，room0 的 combined AUROC 进一步为 0.321；office0 不变。

## 10. 为什么失败

### 10.1 类别 label 不是实例 identity

同类实例误融合时，两边都可能稳定是 `chair`；体素语义熵仍然很低。仅靠 label histogram 无法覆盖 same-class association 错误。

### 10.2 体素多数投票会抹平 observation 差异

mixed-mask 错误的有效信号常常是“同一 object 的不同 observation 给出不同 label”。投到 3D 后，局部多数票会把少数但重要的 observation 压掉。实测非体素 label 熵明显优于体素语义冲突。

### 10.3 false-split 分数的假设不成立

当前分数假设：同一个真实物体被拆成两个 object 后，两边的 label 仍然相似。但真实案例中，一个 GT 实例的两个碎片可能分别被叫成 `potted plant` 和 `refrigerator`，使真正 split pair 得到低分；相邻的同名独立 object 或 GT 未决 object 反而得到高分。

### 10.4 3D 碎裂不等于错误

遮挡、稀疏深度、细杆结构和视角覆盖都会产生多个体素连通块；错误 object 也可能在融合后非常连通。因此碎裂指标只能弱提示，不能单独判错。

### 10.5 `max(percentile)` 造成高分饱和

5 cm combined 分数：

| 分布 | room0 | office0 |
|---|---:|---:|
| 最小值 | 0.457 | 0.470 |
| 中位数 | 0.855 | 0.848 |
| 分数 ≥ 0.9 | 30/69 = 43.5% | 13/33 = 39.4% |
| 唯一分值数 | 31 | 16 |

三个百分位取最大值，天然让大量 object 接近 1，排序区分度不足。

### 10.6 邻近 gate 漏掉多数 split pair

5 cm 只召回约 28%–31% 的可靠 false-split 对。即使后续排序完美，总召回上限仍然太低。

## 11. 成功、失败和限制

### 成功

- 两场景 10,613 个唯一 observation 的 mask/label/final membership 精确闭环；
- 简单体素 schema、三尺度统计、object / pair 指标和可视化全部实现；
- 5 cm 存储很小，技术上可在线维护；
- 找到了一个比体素更强的 mask-trigger 候选：object observation-label entropy；
- 发现并修正了 GT 粗体素重复计数和 pair 未知负例两个评测问题。

### 失败

- unified trigger 未过预注册 GO 条件；
- voxel semantic 没有稳定优于非体素基线；
- fragmentation 没有跨场景强分离；
- label×proximity false-split 分数不成立；
- `all_history` 和 `final_members_only` 在这两个 run 中完全相同，无法验证未归属 observation 的增益。

### 限制

- 只有 room0 / office0 两个开发场景，不能给出广泛泛化结论；
- 使用冻结 B0 结果做离线分析，不是 frame-by-frame 在线触发时延实验；
- Replica sidecar 是自动 GT 归因，不是逐 object 人工 actionable 标签；
- room0 的 pair 可靠负例覆盖不足；
- 未覆盖从未进入任何 mask 的漏检、pose drift、动态变化和关系语义错误；
- 当前 label 是 detector 的 200 类字符串，存在同义词和粒度不一致。

## 12. 下一步建议

### 建议 1：停止当前统一体素 trigger

不继续调 `max`、权重或阈值，也不扩大到更多场景。当前机制没有通过两场景最小门。

### 建议 2：单独验证非体素 mask trigger

下一轮只做一个更小的实验：

```text
Object 的原始 observation label 分布
            ↓
label entropy / dominant-ratio / temporal persistence
            ↓
只触发 repeated mixed-mask review
```

先在现有两场景冻结规则，再在新的 holdout 场景验证。不要同时声称它能解决 association。

### 建议 3：association 必须使用实例划分证据

false merge / false split 尤其是 same-class 情况，需要的不是更多类别统计，而是：

- 哪些 3D 区域在多个视角中经常被同一个 mask 一起包含；
- 哪些区域在清晰视角中经常被两个 mask 分开；
- observation 到 object 的 membership 是否随新视角反复改变。

这些信息可以通过现有 `obs_ids` 回查 source mask 后在 object/pair 层计算，不必增加每体素字段。

### 建议 4：机制通过离线门后再做在线集成

只有当单一错误家族在开发场景和 holdout 场景都显示稳定 top-vs-bottom 分离，再把统计改成逐帧增量更新并记录 `d` 发现帧。当前不应直接接到 revision ticket 或 VLM。

## 13. 产物位置

服务器完整原始结果：

```text
/home/chenkejun/beauty/conceptgraphs/results/experiments/voxel_trigger_v0_20260831/full
```

服务器后分析：

```text
/home/chenkejun/beauty/conceptgraphs/results/experiments/voxel_trigger_v0_20260831/analysis
```

实现：

```text
/home/chenkejun/beauty/conceptgraphs/code/experiments/voxel_trigger_v0_20260831/voxel_trigger_v0.py
/home/chenkejun/beauty/conceptgraphs/code/experiments/voxel_trigger_v0_20260831/analyze_voxel_trigger_v0.py
/home/chenkejun/beauty/conceptgraphs/code/experiments/voxel_trigger_v0_20260831/test_voxel_trigger_v0.py
```

本地同步目录：

```text
D:\Users\刘雯静\Downloads\Conceptgraphs\results\voxel_trigger_v0_20260831
```

主要图：

- `01_gate_and_family_diagnostics.png`
- `02_voxel_map_scale_storage.png`
- `03_ranked_objects_5cm.png`
- `04_room0_voxel_object_cases.png`
- `04_office0_voxel_object_cases.png`
- `05_false_split_pair_cases.png`

