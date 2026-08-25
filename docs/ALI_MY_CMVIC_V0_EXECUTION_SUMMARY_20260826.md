# ALI-MY CMVIC V0 远端执行总结（2026-08-26）

## 最终决策

**FIX。方向保留，V0 主统计量暂不进入 calibration 或自动提交。**

本轮没有基础设施级 STOP：真实投影、因果证据隔离、raw replay parity、identity/provenance、schema guard 和 source immutability 全部通过，production commit 始终为 0。

必须 FIX 的原因：

1. 已知正确 CREATE_INSTANCE 正控被 CMVIC 错误偏向 NOOP，ΔCMVIC=-0.0463227。
2. 机器盲选 3 例事后全部是 endpoint-correct，只形成 4 个负候选，无法做双类别排序验证。
3. 5 个候选中有 2 个 VLM 顺序交换不一致，必须 fail closed。

## 已完成实现

- leave-one-verification-frame-out causal raw-object replay；
- 历史 state-only API 兼容；
- room0 RGB-D projection、pose inverse、depth visibility、z-buffer；
- counterfactual projection overlay 与 evidence hash；
- CMVIC 连续分数和 observability audit；
- assignment likelihood 作为 diagnostic；
- primary statistic / evidence policy calibration guard；
- machine-only blind selector；
- positive/clean controls；
- 3 个机器盲例、4 个候选；
- 10 次匿名顺序交换 VLM critic；
- posthoc risk–coverage、failure cases 和 timing；
- 完整单元/回归测试。

## 审计结果

真实 room0 roundtrip：

- 状态 PASS_TRANSFORM_PROVEN；
- minimum IoU 0.6439117；
- mean IoU 0.8383685；
- maximum inverse median depth error 0.0008393 m。

raw-object replay parity：

- state hash exact；
- membership exact；
- object summary exact；
- source hashes unchanged；
- snapshot validation pass。

评估隔离：

- runtime human/gold loaded=false；
- proposal/verification intersection 为空；
- 每个 protocol 只有一个 evidence policy UID；
- calibration_ready=false；
- production_commit_count=0。

## 实验结果

| case | role | label | ΔCMVIC | Δassignment | VLM |
|---|---|---:|---:|---:|---:|
| identity_dev_003 | 正控 | 1 | -0.0463227 | -0.0134056 | null（[-1, 0]） |
| identity_machine_23aa... | 负例 | 0 | -0.1247102 | -0.0029208 | -1.0 |
| identity_machine_f104... | 负例 | 0 | -0.0784253 | +0.00000010 | null（[0, -1]） |
| identity_machine_8758... c1 | 负例 | 0 | -0.0360382 | -0.0000858 | -1.0 |
| identity_machine_8758... c2 | 负例 | 0 | -0.0001991 | -0.0000712 | 0.0 |

VLM：+1 支持候选，-1 支持 NOOP，0 DEFER。只有两种匿名顺序一致时才输出数值；否则为 null，括号内保留两次物理映射。

机器负例 4/4 没有 CMVIC false positive，这是有效的安全信号。但所有机器标签均为 0，所以 AURC 均输出 null，状态 ONE_CLASS_ONLY；禁止宣称 CMVIC 排序优于 baseline。

干净 no-op 控制没有产生不同的可执行分区，正确 fail closed 为 DEFER_NO_DISTINCT_EXECUTABLE_SEPARATION。

## 失败机制

正控四帧 147、156、179、188 中：

- 前两帧两个状态投影一致；
- 后两帧候选多出一个位于画面边缘、严重遮挡的低质量分区；
- 全 ROI Hungarian 分数把它作为 unmatched projected instance 处罚；
- 一个 VLM 顺序支持 NOOP，交换顺序后 DEFER。

所以 projected difference pixels 非零只代表数组不同，不代表观测足以判断哪个 partition 更正确。当前 observability 过于宽松。

## 最优下一步

实现 Counterfactual Partition Observation V1：

1. 只在两状态真正不同的 changed support 上评分。
2. 用像素对 same-instance / different-instance 共分区关系，而不是全 ROI 实例数平均。
3. 将 observability 分为：
   - GEOMETRIC_DIFFERENCE；
   - MEASUREMENT_SUPPORTED；
   - PARTITION_IDENTIFIABLE。
4. edge-clipped、单视角、遮挡差异必须 DEFER。
5. VLM 只做解释与顺序稳定性诊断，不做 primary statistic。
6. 旧正控只用于设计回归；验证改用新的 1 FM + 1 FS 正例和 2 个可执行负例。
7. 机制门通过前不扩大量样本，不做 calibration。

## 性能

5 个 case 串行计算量合计约 47.56 分钟，已在远端并行执行。

- causal prefix replay：约 38.1%；
- candidate replay：约 27.3%；
- snapshot：约 19.9%；
- NOOP replay：约 13.7%；
- projection：约 0.16%。

因此下一步优先做 prefix/snapshot/NOOP 缓存；当前语义失败与主要瓶颈都不应靠强行使用 GPU 解决。

## 关键产物

实验根目录：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_counterfactual_observability_v0_20260825

统一结果：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_counterfactual_observability_v0_20260825/posthoc/CMVIC_PILOT_RESULTS.json

报告：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_counterfactual_observability_v0_20260825/posthoc/CMVIC_PILOT_REPORT.md

详细 postmortem：

docs/CMVIC_V0_POSTMORTEM_20260825.md

## 验证状态

- focused CMVIC/replay/selector tests：通过；
- revision + projection regression：157 passed，1 skipped；
- formatting、compile 和 diff checks：最终提交前再次执行；
- production graph：未修改。
