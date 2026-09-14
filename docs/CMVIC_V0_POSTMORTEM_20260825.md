# CMVIC V0 Postmortem（2026-08-25—26）

## 结论

本轮决策为 **FIX，不是 GO，也不是 STOP**。

基础设施与因果审计全部通过：真实 room0 投影 roundtrip、raw-object replay parity、proposal/verification 隔离、统一 evidence policy、calibration schema guard、源快照不可变和 production commit=0 均成立。因此没有理由停止 counterfactual observability 方向。

但 V0 主统计量不能进入 calibration 或自动提交：人工确认正确的 CREATE_INSTANCE 正控被 CMVIC、旧 assignment diagnostic 和匿名 VLM 同时偏向 NOOP/DEFER。与此同时，3 个机器盲选 endpoint-correct 负例共 4 个候选全部被 CMVIC 排在 NOOP 后，说明 V0 有负控安全价值，但正负识别机制尚未成立。

## 1. 执行范围

- 远端分支：exp/ali-my-counterfactual-observability-v1-20260825
- 基线提交：7987fec87677543395694959ed6db0111cd75903
- 场景：Replica room0，200 帧 run
- 正控：identity_dev_003，人工确认 CREATE_INSTANCE
- 干净负控：identity_clean_noop_room0_001
- 机器盲选：3 个 case、4 个不同候选
- held-out 策略：每例最多 4 个 evenly-spaced future frames
- VLM：匿名状态、顺序交换，共 10 次 shadow calls
- 自动提交：0
- calibration：未就绪

所有 replay、投影、VLM 和评估均在远端服务器执行。

## 2. 实现结果

新增或正式化的能力包括：

- causal prefix raw-object replay：验证帧自身严格不进入该帧的几何；
- state-only 历史 API 保持兼容，同时暴露受影响 raw replay objects；
- camera-to-world pose 反演、depth visibility、z-buffer、物理 footprint 投影；
- 共同 ROI 下的 Hungarian IoU CMVIC；
- counterfactual projection observability 与匿名 overlay；
- evidence policy UID、frame evidence hash、mask selection audit；
- CMVIC 作为可选 primary statistic，assignment likelihood 降为 diagnostic；
- calibration primary statistic / evidence policy schema guard，旧标定对 CMVIC fail closed；
- 机器盲选 selector、冻结器、VLM runner 对接、posthoc evaluator；
- 单类别评估显式标为 ONE_CLASS_ONLY，AURC 输出 null；
- projection、occlusion、assignment、permutation、schema、selector、replay parity 测试。

## 3. 关键审计

### 3.1 真实投影 roundtrip

状态：PASS_TRANSFORM_PROVEN。

- pose 约定：camera-to-world，投影前取逆；
- depth scale：6553.5；
- voxel size：0.01 m；
- depth tolerance：0.02 m；
- minimum projection IoU：0.6439117176413756；
- mean projection IoU：0.8383685174306421；
- minimum source-mask coverage：0.6484538903584786；
- maximum inverse median depth error：0.0008392512452705114 m。

产物：
/home/chenkejun/beauty/conceptgraphs/experiments/revision_counterfactual_observability_v0_20260825/roundtrip/roundtrip_projection_audit.json

SHA-256：
c947872191ade00f00db7b6a40c9a934b1b5e19fa0164cd23ae87e3589e332c5

### 3.2 replay 新旧接口 parity

状态：PASS。

- legacy state hash 与 raw-object API state hash 完全一致；
- membership 完全一致；
- object summary 完全一致；
- raw object members 均存在于 endpoint state；
- source hashes 未改变；
- snapshot validation 通过。

产物：
/home/chenkejun/beauty/conceptgraphs/experiments/revision_counterfactual_observability_v0_20260825/roundtrip/local_replay_raw_object_parity.json

SHA-256：
ace506c1c1f9167d4c62e228cc2612c8432eb649d6d6827e3d656cd5aa01ee05

### 3.3 盲选与标签隔离

最终冻结 selector 使用固定种子、frame quartile round-robin 和 executor feasibility；没有使用 review score、人工 final state、CMVIC 或 VLM 分数。

冻结清单包含 12 个 deterministic prefilter candidates；达到目标 3 例后停止继续 replay：

- identity_machine_23aa8531e0c832a7e359：anchor 38，Q1；
- identity_machine_f104e60c903a44f1a570：anchor 92，Q2；
- identity_machine_8758b7d9784e8e8619bf：anchor 172，Q4。

事后加载冻结 endpoint labels 后，3 例均为 evidence-sufficient CORRECT；其中 1 例有 R1/R2 一致复核。因此 4 个候选全部只能标为 non-beneficial，不能形成双类别排序实验。

选择清单 SHA-256：
d46535f6523b2a16410f70cdf4c65034f9fef7412f78c812fe7370f335d8a9fc

## 4. 结果表

| 角色 | case | 候选数 | posthoc label | ΔCMVIC | Δassignment | VLM 倾向 |
|---|---|---:|---:|---:|---:|---:|
| 正控 | identity_dev_003 | 1 | 1 | -0.0463227 | -0.0134056 | null（[-1, 0]） |
| 机器负例 | identity_machine_23aa... | 1 | 0 | -0.1247102 | -0.0029208 | -1.0 |
| 机器负例 | identity_machine_f104... | 1 | 0 | -0.0784253 | +0.00000010 | null（[0, -1]） |
| 机器负例 | identity_machine_8758... candidate 1 | 1 | 0 | -0.0360382 | -0.0000858 | -1.0 |
| 机器负例 | identity_machine_8758... candidate 2 | 1 | 0 | -0.0001991 | -0.0000712 | 0.0 |

VLM 倾向编码：+1 候选，-1 NOOP，0 DEFER。只有两种顺序映射回同一物理偏好时才保留数值；否则 fail closed 为 null，括号内记录原始映射。

干净负控没有生成不同的可执行 endpoint partition，正确结果为 DEFER_NO_DISTINCT_EXECUTABLE_SEPARATION，没有进入候选分数表。

统一结果：

- decision：FIX；
- STOP reasons：空；
- FIX reasons：POSITIVE_CONTROL_NOT_FAVORED、MACHINE_HOLDOUT_LABELS_ONE_CLASS_ONLY、VLM_ORDER_SWAP_INCONSISTENT；
- distinct candidate observability：5/5；
- machine candidate observability：4/4；
- machine labels：0 positive / 4 negative；
- CMVIC、assignment、VLM、CMVIC+VLM 的 AURC：均为 null，状态 ONE_CLASS_ONLY；
- calibration_ready：false；
- production_commit_count：0。

最终结果 SHA-256：
5af02bf8b0864991f95c080c18fac9e7fd32408e03c52ba245fff216f0f21d34

## 5. 正控失败的逐帧机制

identity_dev_003 的 4 个 future frames 为 147、156、179、188。

| frame | candidate instances | NOOP instances | observed masks | per-frame ΔCMVIC |
|---:|---:|---:|---:|---:|
| 147 | 3 | 3 | 5 | 0 |
| 156 | 3 | 3 | 2 | 0 |
| 179 | 3 | 2 | 1 | -0.0603752 |
| 188 | 3 | 2 | 2 | -0.1249156 |

前两帧两个状态在可见投影上没有差异。后两帧候选多出一个可见分区，但该分区位于右侧边界且严重遮挡；V0 的 max-instance-count normalization 把它作为 unmatched projected instance 处罚。

顺序交换 VLM 的结果也不是候选证据：

- 一个顺序选择 NOOP；
- 交换顺序后选择 DEFER；
- critic 明确指出差异只出现在 tiny、edge-clipped、heavily occluded region，缺乏独立视角确认。

因此，V0 的 “difference_pixels != 0” 只证明两个投影数组不同，不证明未来观测有足够信息判断哪个分区更正确。当前 5/5 observability coverage 是几何差异覆盖率，不是 epistemic identifiability coverage。

## 6. 为什么不能调阈值补救

本轮失败不能通过降低提交阈值、忽略负 ΔCMVIC 或挑掉 frame 179/188 修复：

- 正控方向错误发生在主统计量内部，而不是 commit threshold；
- 删除失败帧属于 outcome-conditioned frame selection；
- 加一个根据该正控调出的面积阈值会把设计样本泄漏进下一次验证；
- 机器样本只有负类，无法证明新阈值保留正类召回；
- VLM 同样没有提供独立的正向证据。

正确动作是改变 partition observation 的统计对象，并在新 holdout 上验证。

## 7. 最优下一版：Counterfactual Partition Observation V1

### 7.1 从“全 ROI 实例平均”改为“变化支持上的共分区一致性”

对每个 held-out frame t：

1. 只使用 causal prefix 产生的受影响对象几何。
2. 构造两状态都通过深度支持的共同像素域 U_t。
3. 构造 counterfactual difference support D_t：两状态对像素对的 same-instance / different-instance 关系不同的区域。
4. 将 observed masks 转成观测共分区关系；重叠、无标签和深度冲突像素记为 unknown，不强行匹配。
5. 对 D_t 上的像素对计算每个状态与观测的 co-assignment agreement，而不是对整个 ROI 的实例数做惩罚。
6. 用 contingency matrix 计算 pairwise agreement，避免显式 O(n²) 枚举。
7. 候选优势只来自真正能区分两状态的观测关系；公共背景和无关 context objects 不稀释分数。

这个统计量同时适用于 false merge 与 false split：

- false merge：NOOP 预测 same，观测与修复预测 different；
- false split：NOOP 预测 different，观测与修复预测 same。

### 7.2 将 observability 分成三层

- GEOMETRIC_DIFFERENCE：投影数组是否不同；
- MEASUREMENT_SUPPORTED：差异是否由深度支持、远离边界且不是单像素/单 footprint artifact；
- PARTITION_IDENTIFIABLE：观测 mask 在差异支持上是否提供 same/different 关系。

只有第三层才能进入 repair ranking。前两层不足时必须 DEFER。

测量质量下限必须由体素 footprint、深度噪声和图像边界推导并预先冻结；不能用旧正控结果拟合。连续质量值完整记录，避免把多个启发式规则伪装成语义判决。

### 7.3 多视角聚合

- 要求至少两个非冗余 view clusters 提供同方向 partition evidence；
- frame selection 仍只按时间/几何可用性预声明，不按分数挑帧；
- frame eligibility 必须由 raw frame 可用性与 pre-anchor geometry 的视锥相交决定，不能依赖候选/NOOP 的 full-endpoint membership；
- evidence policy identity 必须绑定 frame eligibility、minimum gap、maximum frames 和聚合策略，之后才能用于 calibration compatibility；
- 聚合采用预声明的 robust median 或 measurement-quality-weighted mean；
- 单帧 edge-clipped 差异不能把案例从 DEFER 推到可提交；
- 记录 leave-one-frame-out sign stability。

### 7.4 VLM 的最小角色

VLM 不作为 primary statistic，不使用 confidence 作为概率。它只做：

- 解释 changed-support 是否对应同一物体；
- 检查边界裁切、遮挡和 mask conflict；
- 给出 DEFER 原因；
- 在匿名顺序交换后检查结论一致性。

任何顺序不一致均 fail closed。

## 8. 下一轮验证设计

旧 identity_dev_003 只能作为 V1 设计/回归案例，不得再次作为唯一确认正控。

下一轮最小但有效的冻结样本：

- 1 个未用于 V1 设计的人工确认 false merge 正修复；
- 1 个未用于 V1 设计的人工确认 false split 正修复；
- 2 个 endpoint-correct 且能生成不同分区的负例；
- 每例最多 4 个固定 future frames；
- 正负标签均冻结后再统一揭盲。

先做机制门：

- 两个新正例至少不被稳定偏向 NOOP；
- 两个负例不被候选稳定占优；
- 顺序交换 VLM 无相反结论；
- leave-one-frame-out 方向稳定；
- 若任一不满足，停止扩样本。

机制门通过后，才在未使用的 human-confirmed errors 中扩到平衡 holdout，并进入 risk–coverage/calibration。不能用本轮 4 个负例与旧正控同时做设计和确认。

## 9. 资源与性能

5 个 case 的串行计算量合计约 2,853,615 ms（47.56 分钟），实际通过 case 并行缩短墙钟时间。

分项合计：

- causal prefix replay：1,087,703 ms，约 38.1%；
- candidate replay：780,216 ms，约 27.3%；
- snapshot：566,977 ms，约 19.9%；
- NOOP replay：390,824 ms，约 13.7%；
- projection verifier：4,508 ms，约 0.16%。

因此当前优化优先级不是 GPU 化 Hungarian 或投影，而是：

1. 同一 case 的 causal prefix 增量缓存；
2. NOOP prefix 在候选间复用；
3. snapshot 一次构建、多候选共享；
4. 只在机制门通过后扩大 case 数。

当前路径主要是 Python/Open3D replay 和 provenance I/O，并非 GPU tensor workload；强行占用 GPU 不会修复统计语义，也不会触及主要瓶颈。

## 10. 最终判断

Counterfactual observability 方向保留，CMVIC V0 主统计量退回研究诊断状态。

可以保留并复用的部分：

- causal evidence split；
- raw-object replay API；
- projection/roundtrip；
- evidence identity/hash；
- schema guard；
- blind selector；
- timing instrumentation；
- anonymous order-swapped critic harness。

必须替换或加强的部分：

- exact-nonzero observability；
- 全 ROI Hungarian IoU 平均；
- unmatched instance count 对低可见分区的直接处罚；
- 单类别下的排序评价；
- 将 VLM 平均偏好当作稳定证据。

下一步应实现 Counterfactual Partition Observation V1 的 target-local co-assignment statistic 与三层 observability gate，然后只跑 1 FM + 1 FS 新正例和 2 个可执行负例。
