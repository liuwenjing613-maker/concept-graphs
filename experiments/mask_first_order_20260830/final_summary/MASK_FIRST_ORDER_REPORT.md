# Mask-first 累积顺序实验（room0 + office0）

## 最核心结果

把 mask partition/去污染放在前面是正确的，但它不能替代关联修复。完整 OM_all 在原生关联下已稳定改善结构；随后加 OA 又产生最大单步提升，说明高质量 mask 之后 association 仍是主要限制。OP 单独恢复被拒绝观测不稳定，不能作为独立主方向。

## 实验顺序与实现

所有新建条件均对 400 帧从空图按时间顺序在线构建。阶段按能力累积，但不重复叠加同一几何：

1. `B0`：冻结基线。
2. `MF_OP`：保留原处理后观测，并恢复可确定 owner 的 rejected raw observation；仍用原生在线关联。
3. `MF_OM_pure`：吸收 OP 能力，将同一 owner 的 raw proposal 先裁剪并合成一个干净观测；仍用原生在线关联。
4. `MF_OM_all_native`：吸收前两步，对所有相交 raw proposal 做最大化 GT partition、去污染和 FP 抑制；仍用原生在线关联。
5. `MF_OM_all_OA`：观测流与上一步完全相同，只把关联换成 GT identity。该定义与此前正式 `OM_all` 相同，smoke parity 通过后直接复用，避免重复全量运行。
6. `OG`：可观测 GT 上限。

## 5 cm 主评测

AP 是 AP25/AP50 均值；F1 是 IoU≥0.25 的一对一节点 F1。二者均为类别无关结构指标。

| stage | room0 AP | office0 AP | avg AP | room0 F1 | office0 F1 | avg F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.0223 | 0.0045 | 0.0134 | 0.1104 | 0.0690 | 0.0897 |
| MF_OP | 0.0257 | 0.0173 | 0.0215 | 0.1350 | 0.1333 | 0.1342 |
| MF_OM_pure | 0.0423 | 0.0122 | 0.0273 | 0.2041 | 0.0860 | 0.1451 |
| MF_OM_all_native | 0.1901 | 0.1073 | 0.1487 | 0.4205 | 0.2692 | 0.3448 |
| MF_OM_all_OA | 0.6770 | 0.6358 | 0.6564 | 0.8352 | 0.8627 | 0.8490 |
| OG | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

两场景平均的 AP 单步增量：

- `B0->MF_OP`：+0.0080
- `MF_OP->MF_OM_pure`：+0.0058
- `MF_OM_pure->MF_OM_all_native`：+0.1214
- `MF_OM_all_native->MF_OM_all_OA`：+0.5077
- `MF_OM_all_OA->OG`：+0.3436

`MF_OM_all_native` 恢复了 B0→OG 结构 AP 差距的 13.7%；加 OA 后累计恢复 65.2%。因此 mask 修复有明确价值，但仅靠 mask 不够。

## 与原先 OA-first 顺序的直接对照（5 cm）

| stage | room0 AP | office0 AP | avg AP | room0 F1 | office0 F1 | avg F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.0223 | 0.0045 | 0.0134 | 0.1104 | 0.0690 | 0.0897 |
| FWD_OA | 0.0500 | 0.0136 | 0.0318 | 0.2179 | 0.0930 | 0.1555 |
| FWD_OP | 0.0571 | 0.0203 | 0.0387 | 0.2209 | 0.1573 | 0.1891 |
| FWD_OM_pure | 0.0601 | 0.0455 | 0.0528 | 0.2222 | 0.2535 | 0.2379 |
| MF_OM_all_OA | 0.6770 | 0.6358 | 0.6564 | 0.8352 | 0.8627 | 0.8490 |
| OG | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

顺序并非可交换的独立加法。OA-first 的早期提升较小；mask-first 做到 maximal partition 后结构显著改善，但最大的交互增益仍来自在干净观测上做 OA。

## FROSS-style 敏感性（5 mm 去重、0.1 m 对应）

| stage | room0 F1 | office0 F1 | avg F1 |
|---|---:|---:|---:|
| B0 | 0.1840 | 0.1379 | 0.1610 |
| MF_OP | 0.1472 | 0.1778 | 0.1625 |
| MF_OM_pure | 0.2449 | 0.2581 | 0.2515 |
| MF_OM_all_native | 0.3636 | 0.3462 | 0.3549 |
| MF_OM_all_OA | 0.6044 | 0.5882 | 0.5963 |
| OG | 0.9890 | 1.0000 | 0.9945 |

该敏感性下 `MF_OM_all_native → MF_OM_all_OA` 仍在两场景显著增加。反例也被保留：room0 的 `MF_OP` 比 B0 下降，进一步说明 OP 单独不稳定。三档 2.5/5/10 cm 主评测对 `OM_all` 和 `+OA` 的方向一致。

## 标签与数据集控制

- Replica/ReplicaSSG 两场景配对审计通过。
- 主结论只使用类别无关 AP/F1，因此 `desk lamp` 与 `lamp` 等别名不会改变 mask/association 顺序判断。
- 严格官方映射是基线；`desk lamp→lamp` 等仅作为预先定义的敏感性规则，不改写 GT。本轮两个场景预测中没有 `desk lamp`，room0 仅实际出现 `ceiling light`，office0 没有灯具扩展别名命中。
- 语义分母小且本体敏感，因此本轮不把 semantic accuracy 用作顺序结论证据。

## 质量门与资源

- 新建正式在线图：6/6 READY；INCOMPLETE：0。
- 统一 runner SHA256：`43c5e9c105140d3d237a23ca50b6f6bdcde0fd53e9c121266c8328857e1de29d`；manifest 与文件一致：True。
- OP 溯源缺失/意外：0/0。
- 复用最终 +OA 的两场景 10 帧几何/对象数/观测数精确 parity：True。
- GPU0/1/2 用于在线构图；GPU3 CUDA 初始化失败后未继续使用，其失败目录已删除。评测为 CPU 几何计算。

## 最终方向

推荐顺序冻结为：**observation-level mask partition/去污染 → association/replay → semantic verification**。不要把 OP 当成主修复器，也不要认为先修 mask 就能消除 association 问题。下一步应实现真实的 targeted partition，并用 replay/verify/rollback 检验它能否逼近本 Oracle，而不是继续扩展 final-level merge/split 规则。

## 局限

- 仅两个场景，不能外推总体置信区间。
- OM_pure、OM_all、OA、OG 均使用 GT 提供 Oracle 能力，不是可部署方法。
- mask 与 association 存在强交互，不能把各阶段增量解释成互相独立的因果主效应。
- FROSS-style 结果是敏感性审计，不是官方 FROSS benchmark。

## 产物

- 汇总 JSON：`/home/chenkejun/beauty/conceptgraphs/results/experiments/mask_first_order_20260830/final_summary/mask_first_order_summary.json`
- 原始实验：`/home/chenkejun/beauty/conceptgraphs/results/experiments/mask_first_order_20260830`
