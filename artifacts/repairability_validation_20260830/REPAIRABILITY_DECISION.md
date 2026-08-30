# 两场景可修性验证与最终方向

## 最终判断

冻结问题方向：**历史 observation 的 mask partition/去污染/重组 + transactional replay + verify/rollback**。
仅修 final false split/false merge/label 不能成为主线；稳定结构后，语义确认是第二阶段。当前只冻结方向，不宣称 partition 方法已经实现成功。

## 方法自检

- 旧 O1 对 unmatched observation 使用最终 B0 lineage，存在未来 lineage 依赖；新版 OA/OP/OM 从空图按时间顺序构建，且 manifest 明确 `future_final_lineage_used_for_mapping=false`。
- 旧 O1 只以 purity 判定，未同时要求 visible-instance recall；新版漏斗同时报告 0.3/0.5/0.7 的 purity+recall。
- 旧 O3 对自身评测得到 1.0 是定义上限，不是模型结果；不再压成单一 rho。
- 关系结果因模型/候选集不统一被排除。
- Replica/ReplicaSSG 数据配对和标签本体单独审计；严格官方映射、原评测别名、灯具复合词扩展三档并列，不按样本事后挑映射。

## 数据集配对

| scene | 原始 RGB/depth | 在线帧 | schedule | GT depth 对齐最差 median | 最低 5 cm 内比例 | pass |
|---|---:|---:|---|---:|---:|---|
| room0 | 2000/2000 | 400 | 0..1995 / 5 | 0.003374 m | 0.994371 | True |
| office0 | 2000/2000 | 400 | 0..1995 / 5 | 0.001845 m | 0.995349 | True |

两个场景均从各自 Replica 序列读取 2000 帧，按 stride=5 在线处理 400 帧；轨迹哈希、scene alias、GT sidecar 帧号与 B0 schedule 完全一致。office0 配置中的 room0 render-camera 路径只在 `vis_render=true` 时使用，本实验为 false。

## 5 cm 实例 F1

| scene | B0 | OA | OP | OM_pure | OM_all | OG |
|---|---:|---:|---:|---:|---:|---:|
| room0 | 0.110 | 0.218 | 0.221 | 0.222 | 0.835 | 1.000 |
| office0 | 0.069 | 0.093 | 0.157 | 0.254 | 0.863 | 1.000 |

最大边际在两场景均为 `OM_all - OM_pure`，即利用现有 raw proposal 支撑做理想 partition/cleanup。2.5/5/10 cm 方向一致。

## 5 mm 去重、0.1 m 点对应敏感性

| scene | B0 | OA | OP | OM_pure | OM_all | OG |
|---|---:|---:|---:|---:|---:|---:|
| room0 | 0.184 | 0.192 | 0.209 | 0.317 | 0.604 | 0.989 |
| office0 | 0.138 | 0.186 | 0.180 | 0.310 | 0.588 | 1.000 |

该敏感性定义下仍是 OM_all 最大；粗 2.5 cm 去重使 OG 自一致性失败，已判为方法失败，不参与结论。

## 下游代理（5 cm）

| scene/condition | class F1 | R@1 | R@3 | 1 m | count MAE |
|---|---:|---:|---:|---:|---:|
| room0/B0 | 0.034 | 0/8 | 0/8 | 0/8 | 2.444 |
| room0/OM_all | 0.073 | 0/8 | 0/8 | 0/8 | 4.400 |
| room0/OG | 1.000 | 8/8 | 8/8 | 8/8 | 0.000 |
| office0/B0 | 0.062 | 1/5 | 1/5 | 1/5 | 1.143 |
| office0/OM_all | 0.205 | 0/5 | 1/5 | 1/5 | 2.636 |
| office0/OG | 1.000 | 5/5 | 5/5 | 5/5 | 0.000 |

OM_all 大幅改善结构，却没有同步改善类别查询，说明语义是结构稳定后的第二瓶颈。

## 标签本体敏感性（5 cm 下游代理）

| scene/mode | B0 class F1 | OM_all class F1 | OG class F1 | OM_all-B0 |
|---|---:|---:|---:|---:|
| room0/official_only | 0.040 | 0.026 | 1.000 | -0.014 |
| room0/current_aliases | 0.034 | 0.073 | 1.000 | +0.039 |
| room0/reviewed_lamp_aliases | 0.034 | 0.073 | 1.000 | +0.039 |
| office0/official_only | 0.000 | 0.000 | 1.000 | +0.000 |
| office0/current_aliases | 0.062 | 0.205 | 1.000 | +0.143 |
| office0/reviewed_lamp_aliases | 0.062 | 0.205 | 1.000 | +0.143 |

扩展别名在 B0 中的实际命中：`{"office0": [], "room0": ["ceiling light"]}`。`desk lamp` 本轮没有出现；仍显式纳入规则以防后续模型输出。三档结果只用于检验结论是否依赖词表。
官方映射下 OM_all-B0 为 room0 负、office0 零，因此当前类别 F1 的增量不具本体稳健性，不用于定量证明语义收益；三档均显示 OM_all 远低于 OG。

## 真实 replay

- room0_strict: build/local/global=True/True/True，points 17→2716 (159.8×)，association changed=False，closure=3779 observations。
- office0_relaxed_exploratory: build/local/global=True/True/True，points 168→26223 (156.1×)，association changed=False，closure=1560 observations。

这证明 geometry overlay/replay/parity 机制可运行，不证明目标对象或 scene 指标已改善。最高收益的 PARTITION_OBSERVATION 仍未接入 pre-association sparse replay。

## GPT 与 LLaVA

正式调用共 898576 tokens，含 smoke 共 910045 tokens；endpoint 未提供价格，不能换算金额。两场景 GPT vision 与 LLaVA-caption→Terra 的 model-covered accuracy 都为 0，直接替换不受支持。

## 局限

- Only two scenes were run by explicit user constraint; no population confidence interval or scene bootstrap is valid.
- OM_pure/OM_all/OG use GT only as Oracle capabilities and are not deployable methods.
- The strict semantic cohort is empty in both scenes, so pure semantic benefit is unidentified.
- The real micro-pilot has one strict room0 positive and one relaxed office0 exploratory case, not a repair success-rate estimate.
- Both real micro-pilot dependency closures cover their whole scene, making outside-closure safety vacuous for both cases.
- Clean objects were not run through a reliable automatic diagnosis path; exact no-op executor controls pass, but diagnostic false-mutation rate remains unmeasured.
- Relation metrics were excluded because the prior relation comparison mixed models/candidate sets.
- FROSS-style correspondence is a sensitivity audit adapted to dense points, not the official FROSS benchmark pipeline.
- ReplicaSSG-to-Visual-Genome mapping intentionally collapses some classes (for example sofa to chair); expanded compound aliases are sensitivity results, not annotation truth.
- The apparent OM_all-minus-B0 class-F1 delta is not ontology-robust: it is positive under the frozen aliases but zero/negative under official-only mapping, so it is not used as quantitative evidence for semantic benefit.
- The frozen office0 config contains a room0 render-camera metadata path, but vis_render=false; RGB/depth/pose selection and depth alignment independently verify office0 data use.

## 产物

- 汇总 JSON：`/home/chenkejun/beauty/conceptgraphs/results/experiments/repairability_validation_20260830/final_summary/repairability_decision.json`
- 原始实验根目录：`/home/chenkejun/beauty/conceptgraphs/results/experiments/repairability_validation_20260830`
