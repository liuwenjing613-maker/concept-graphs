# CMVIC V0 Premortem（2026-08-25）
> 协议状态：实现期冻结协议说明。原文件正文因写入缺陷未落盘，本正文于 2026-08-26 根据已冻结代码、选择清单和实验产物恢复。它不得被表述为独立、结果前时间戳预注册；可审计的结果前约束以代码哈希、盲选清单哈希、evidence split 与 freeze protocol 为准。

## 1. 目的与边界

本轮只回答一个问题：在不读取人工结论、不把 held-out 帧回灌进 replay、且不改生产图的前提下，反事实 3D 分区能否通过未来 RGB-D 观测被区分，并为候选修复提供比现有 assignment likelihood 更接近因果结果的证据。

本轮不是阈值标定、统计显著性实验或论文主结果。所有输出均为 shadow artifact，production commit 必须保持关闭。

## 2. 核心假设

- H1：若候选与 NOOP 在 held-out 未来视角上的可见投影分区完全相同，则证据对该候选不可观测，必须 DEFER。
- H2：若两状态投影不同，基于共同观测掩码的多视角实例一致性分数 CMVIC 可以提供连续的候选相对优势。
- H3：状态顺序匿名且交换后的 VLM 只能充当诊断 critic；顺序不一致必须视为不稳定证据，不能自动提交。
- H4：已知合法 CREATE_INSTANCE 正控应至少不被稳定地错误偏向 NOOP；干净 no-op 负控不应产生可执行修复优势。
- H5：CMVIC 的 projection/assignment 计算不是主要时间瓶颈，snapshot 与 causal replay 才是主要成本。

## 3. 不可变实验契约

- proposal evidence 只能来自 anchor 及以前。
- verification evidence 只能来自 anchor + minimum_frame_gap 之后。
- 每个验证帧的几何只能由该帧之前的因果 prefix replay 生成；该帧自身的点云不得用于解释自身。
- NOOP 与候选必须使用同一帧、同一深度、同一观测掩码集合和同一 evidence policy。
- 变换约定、深度尺度、体素尺寸和遮挡容差必须先经真实 room0 roundtrip 证明。
- 人工标签与 gold 只能在 freeze 完成后用于 posthoc 评估。
- calibration 不得复用不同 primary statistic 或不同 evidence policy；不匹配时 fail closed。
- 本轮不得更新 production graph，不得执行自动 commit。

## 4. V0 主统计量

COUNTERFACTUAL_MULTI_VIEW_INSTANCE_CONSISTENCY：

1. 将每个因果状态的受影响实例点云投影到冻结验证帧。
2. 用真实深度和 z-buffer 进行可见性裁剪。
3. 在两状态投影联合 ROI 内选择同一组观测实例掩码。
4. 每帧以 Hungarian 最大 IoU 匹配 projected instances 与 observed masks。
5. 每帧分数为匹配 IoU 和除以两侧实例数最大值；跨帧等权平均。
6. 候选优势定义为 score(candidate) - score(NOOP)。
7. 无序可见分区像素差严格为零时标记 COUNTERFACTUAL_UNOBSERVABLE。

V0 不引入根据 pilot 结果调出的语义阈值。深度容差只由传感/体素分辨率定义，不作为效果调参项。

## 5. 选择与样本

机器样本选择只能使用：

- 场景、anchor frame、机器 checker/stage/subtype；
- executor 能否生成有限且不同的分区；
- 固定随机种子与预声明的 anchor quartile 覆盖。

禁止使用 review score、人工 final state、人工 error type、CMVIC 得分、VLM 得分或 posthoc 标签选择样本。目标 6–8 例只是上限；若 executor 屏幕后不足，报告真实可执行数，不补入不合格样本。

机制控制：

- 1 个已有人工确认 CREATE_INSTANCE 正控；
- 1 个干净 no-op 负控；
- 所有验证帧按固定 evenly-spaced 时间策略选择，最多 4 帧，不根据得分挑帧。

## 6. 评价与标签规则

- 人工约束 manifest 中与候选 fingerprint 完全一致的 CREATE_INSTANCE 才标为 beneficial。
- 干净 no-op 候选标为 non-beneficial。
- 冻结 endpoint 标签一致认为原 endpoint CORRECT 时，机器提出的改变保守标为 non-beneficial。
- endpoint WRONG 但没有候选级修复 gold 时不得反推出候选 beneficial。
- 只有同时含正负两类且样本数满足最低要求时才允许比较 risk–coverage/AURC。
- 单类别样本只能评估 false-positive safety，状态必须写为 ONE_CLASS_ONLY，禁止宣称排序优越性或总体泛化。

## 7. GO / FIX / STOP

STOP：

- room0 projection roundtrip 未通过；
- proposal/verification evidence 交叉；
- runtime 加载人工/gold；
- 同一 protocol 内 evidence policy 不唯一；
- production commit 非零；
- 源快照被 replay 修改。

FIX：

- 正控被 CMVIC 稳定偏向 NOOP；
- 干净负控产生修复优势；
- 所有候选均反事实不可观测；
- VLM 顺序交换不稳定；
- 机器 holdout 只有单类别或标签不足；
- CMVIC 在可比较样本上不优于 assignment diagnostic；
- 运行成本无法支持计划内小规模验证。

GO 只在没有 STOP/FIX 条件、机制正负控方向正确、证据隔离和 schema guard 全部通过时成立。GO 仍不授权 production commit，只允许进入独立 calibration/holdout 阶段。

## 8. 自我调整规则

允许调整：

- 缩小样本数以避免在机制未证实时浪费 replay；
- 修复实现 bug、泄漏风险、哈希/身份契约和计时遗漏；
- 在不查看结局的前提下修复 executor feasibility；
- 根据资源瓶颈并行不同 case。

不允许在本轮正式比较中调整：

- 根据正控/盲例结果挑选验证帧；
- 根据标签调 IoU、可见面积或提交阈值；
- 删除失败样本；
- 将 VLM confidence 当校准概率；
- 将投影差异本身等同于支持修复的证据。

若 V0 失败，下一版必须先从 failure mechanism 推导新的 partition observation，再在新的正控和未使用 holdout 上验证；旧 pilot 只用于设计，不得重复充当确认集。

## 9. 必须产物

- projection roundtrip audit；
- raw-object replay parity audit；
- machine-only blind selection manifest 与私有溯源；
- 每例 evidence split、freeze protocol、CMVIC comparison、匿名顺序交换 critic；
- posthoc candidate table、risk–coverage 状态、失败样本清单和 timing；
- CMVIC_V0_POSTMORTEM_20260825.md；
- /home/chenkejun/beauty/ 下的最终执行总结。
