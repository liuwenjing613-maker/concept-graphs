# ali-my 可测指标评测汇总（2026-08-21）

> **更新说明：** 本文以下内容是早期 `room0 + office0`、200 帧/stride 10 的两场景审计评测。现已完成与论文 main 对齐的 Replica 全 8 场景、400 帧/stride 5 正式评测；请以 [`ALI_MY_PAPER_MAIN_ALIGNED_8SCENE_EVALUATION_20260821.md`](ALI_MY_PAPER_MAIN_ALIGNED_8SCENE_EVALUATION_20260821.md) 作为跨方法主结果。本文仍保留用于 R1/R2 证据审计和历史复现。
>
> **方法版本说明：** 2026-08-21 新增的 `ali-my-VLM-only-repair-v1` 是一个完全隔离的方法版本：它只使用已固化的可追溯图像证据调用 VLM 进行检查、诊断和最小修复，不使用 R1/R2 人工标签参与推理。其结果只用于评估“纯 VLM”路线的能力上限，不并入上述 8 场景跨方法主结果。

## 结论

服务器上可定位到的 ali-my 正式冻结地图只有 `room0` 和 `office0`。本次已对这两个场景完成：

- Replica 语义分割：`n_exclude = 1 / 4 / 6` 全部口径；
- ReplicaSSG 物体识别：Object R@1、R@5、mR@1、mR@5；
- 几何匹配覆盖率；
- 关系输出结构检查；
- R1 证据审计与 R2 重测一致性复算；
- 相关自动化测试、输入/输出哈希和 main 协议回归。

主结果采用 main 报告使用的 `n_exclude=6` 语义口径。两场景合并后，ali-my 的语义分割结果为：**mIoU 26.70%、mRecall 43.48%、mPrecision 33.19%、mF1 32.63%、fwIoU 49.74%、点准确率 61.37%**。物体识别合并结果为：**R@1 20.27%、R@5 36.49%、mR@1 33.36%、mR@5 48.57%**。

关系相关的 0 分不能解释为关系模型性能：这两次 ali-my 冻结运行均配置为 `make_edges=false`，边文件均为空数组，实际没有执行关系推理。

在同一批 97 个唯一 endpoint 上，隔离的 `ali-my-VLM-only-repair-v1` 完成了全量标签盲推理。事后与冻结 R1 对齐后，对象状态准确率为 **65/97 = 67.01%**；WRONG 检测的 precision/recall/F1 分别为 **58.49% / 77.50% / 66.67%**。系统通过双阶段门控选出并在隔离派生图中执行了 19 个修复，其中 17 个为重标，2 个为合并；但按 R1 事后核对，只有 **11/19 = 57.89%** 的候选对应人工判定的 WRONG，其余 8 个是高置信误修。因此，该版本可作为高召回的候选查错器，但尚不能用于自动覆盖原图。

## 1. 评测对象与完整性

| 场景 | ali-my 地图对象数 | 地图 SHA-256 | 帧设置 | 边输出 |
|---|---:|---|---|---|
| room0 | 72 | `9550113b42c42a83d640241f038d06b3f040f7deb237314e4550b757a08bd23e` | 200 帧，stride 10 | `[]` |
| office0 | 29 | `b63788b3c675b31ca086429ea9995048d814466429b442ac82fcc3a9aa86aa2d` | 200 帧，stride 10 | `[]` |

两份地图都包含 1024 维原生 `clip_ft`，因此物体开放词汇分类可直接测试。服务器上的 ReplicaSSG GT、Replica Semantic GT、OpenCLIP ViT-H-14 缓存权重均完整可用，未临时下载或替换权重。

## 2. Replica 语义分割

### 2.1 主口径：n_exclude=6

`n_exclude=6` 排除 other、floor、wall、ceiling、door、window，与 main 汇总主口径一致。

| 范围 | mIoU | mRecall | mPrecision | mF1 | fwIoU | 点准确率 |
|---|---:|---:|---:|---:|---:|---:|
| room0 | 23.55% | 41.85% | 26.34% | 28.05% | 49.92% | 62.09% |
| office0 | 27.42% | 46.58% | 30.91% | 32.78% | 49.67% | 60.64% |
| 两场景合并 | **26.70%** | **43.48%** | **33.19%** | **32.63%** | **49.74%** | **61.37%** |

### 2.2 全部排除口径

| n_exclude | 范围 | mIoU | mRecall | mPrecision | mF1 | fwIoU | 点准确率 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | room0 | 16.77% | 37.81% | 18.57% | 21.96% | 25.57% | 41.86% |
| 1 | office0 | 12.57% | 40.24% | 12.64% | 17.96% | 11.78% | 26.39% |
| 1 | 两场景合并 | 15.53% | 39.11% | 17.68% | 21.46% | 18.65% | 33.82% |
| 4 | room0 | 23.46% | 42.50% | 26.03% | 28.44% | 45.33% | 60.05% |
| 4 | office0 | 28.67% | 49.24% | 31.99% | 34.62% | 49.73% | 61.79% |
| 4 | 两场景合并 | 26.56% | 43.87% | 32.46% | 32.68% | 47.83% | 60.90% |
| 6 | room0 | 23.55% | 41.85% | 26.34% | 28.05% | 49.92% | 62.09% |
| 6 | office0 | 27.42% | 46.58% | 30.91% | 32.78% | 49.67% | 60.64% |
| 6 | 两场景合并 | 26.70% | 43.48% | 33.19% | 32.63% | 49.74% | 61.37% |

语义评测复用了 main 为同一物理场景保存的 `rgb_cloud`，因为 ali-my 冻结目录中没有单独保存该中间文件。分类提示词、类别、GT 与指标定义均与官方脚本一致；点最近邻改为 CPU 上的 SciPy `cKDTree` 精确 k=1。room0 对 main 的回归值完全复现；office0 因最近邻后端的等距/数值细节，与原 CUDA Chamfer KNN 有约 1.09 个百分点差异。因此跨分支比较应使用同一 cKDTree 后端。

同一 CPU 最近邻后端、相同两场景的描述性比较为：ali-my 合并 mIoU 26.70%，main post-map 合并 mIoU 22.68%，差值 +4.02 个百分点。两者帧设置不同（ali-my 200 帧/stride 10，main 400 帧/stride 5），不能据此作因果性优劣结论。main 已发布的 8 场景汇总也不能与这里的 2 场景结果直接横比。

## 3. ReplicaSSG 物体识别与几何覆盖

使用地图中原生 1024 维 `clip_ft`，模型/权重为 `ViT-H-14 / laion2b_s32b_b79k`，提示词为 `a photo of a {class}`。几何匹配阈值和召回定义与 main evaluator 相同。

| 范围 | GT 物体 | 预测物体 | 几何匹配 | 几何覆盖率 | R@1 | R@5 | mR@1 | mR@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| room0 | 59 | 72 | 29 | 49.15% | 22.03% | 32.20% | 28.36% | 40.80% |
| office0 | 15 | 29 | 10 | 66.67% | 13.33% | 53.33% | 22.22% | 50.00% |
| 两场景合并 | 74 | 101 | 39 | 52.70% | **20.27%** | **36.49%** | **33.36%** | **48.57%** |

合并 mR 是在两场景并集的 17 个实际出现类别上重新计算，不是两个场景 mR 的算术平均。

协议回归使用 main room0 的原始输入重新运行，精确复现已发布结果：R@1 28.81%、R@5 47.46%、mR@1 39.24%、mR@5 55.53%。这说明 ali-my 的物体结果与 main 使用的是同一评测实现与缓存权重。

## 4. 关系输出

| 范围 | GT 关系数 | 正预测边 | evaluator Predicate R@1 | evaluator Predicate mR@1 | 解释 |
|---|---:|---:|---:|---:|---|
| room0 | 25 | 0 | 0% | 0% | 未运行关系推理 |
| office0 | 16 | 0 | 0% | 0% | 未运行关系推理 |
| 合并 | 41 | 0 | 0% | 0% | 结构性零值，不是模型性能 |

scene-graph object tag 与 possible-tag 分类也会被 evaluator 填成 0，但当前不存在场景图标签或候选标签，因此这些行不属于有效可测方法。

## 5. R1 证据审计与 R2 重测

### 5.1 R1

- 共 97 个 endpoint，证据充分 95 个；正确 55、错误 40、不明确 2。
- 条件错误率 42.11%，证据边界敏感范围 41.24%–43.30%。
- room0：67 个证据充分 endpoint 中错误 27，错误率 40.30%。
- office0：28 个 endpoint 均证据充分，错误 13，错误率 46.43%。
- 错误类型：semantic identity 17、geometry 11、spurious 6、false split 3、false merge 3。
- review score：AUC 0.4205，AP 0.3656；Top-5/10/20/40 precision 分别为 20.0%/20.0%/30.0%/32.5%，没有优于错误基准率。

### 5.2 R2

- 24 个重测案例全部完成。
- final-state agreement：20/24 = 83.33%，Cohen's kappa 0.7055。
- 三字段完全一致：19/24 = 79.17%。
- 两轮均判错的 10 例中，错误类型一致 9 例，agreement 90.00%，kappa 0.8611。

R1 系统门禁复算通过：projection PASS，检查 5069 个资产，97 个 incident、40 个错误，决策为 `PROCEED_TO_EXPERT_TRACE`。R2 复算与冻结 JSON 完全一致。

## 6. ali-my-VLM-only 检查与派生修复实验

### 6.1 评测边界与实现

本实验的输入是 R1 工作清单中的 **97 个唯一、可评审 endpoint**，不是“97 个已确认错误”。对应的冻结 R1 真值是正确 55、错误 40、不明确 2。所有 R1/R2 人工字段在推理前均被排除；标签只在 97 个结果全部落盘后加载，用于事后评估。

| 阶段 | 模型 | 职责 | 输出约束 |
|---|---|---|---|
| 证据审计 | `gpt-5.6-terra` | 对真实物体、保存标签、单物体性、几何完整性、成员一致性和背景/噪声进行强制分项审计 | 固定 JSON schema；必须覆盖所有检查项 |
| 最终判定 | `gpt-5.6-sol` | 给出 CORRECT/WRONG/UNCLEAR、错误类型和最小修复动作 | 与 endpoint 别名、对象 UID 和允许动作精确绑定 |
| 修复复核 | `gpt-5.5` | 独立复核诊断是否有证据支持、动作是否为最小安全修复 | 只有显式 approve 且达到置信门槛才能进入派生图 |

每个 endpoint 最多向 VLM 提供 10 张经 SHA-256 绑定的证据图：2 张最终几何图、最多 2 张 trigger panel、1 张 timeline，余位用于 endpoint 上下文和 mask crop。只接受 `TRACEABLE`、资产哈希正确且与最终地图精确连接的证据包。

自动执行门槛为：RELABEL 要求诊断置信度不低于 0.85、复核置信度不低于 0.80；DELETE 为 0.97/0.95；MERGE_WITH 为 0.95/0.90。SPLIT_OBJECT、TRIM_GEOMETRY 以及需要成员重分配的结构修复在本版本中只产生计划，不直接修改地图。

推理调用使用 OpenAI-compatible `chat/completions` 接口，API 密钥只注入运行进程环境，未写入代码、结果、命令历史或派生图。该方法不运行本地模型，未使用任何 GPU，GPU3 未使用。

### 6.2 97 个 endpoint 正式结果

| R1 真值（行）/ VLM 判定（列） | CORRECT | UNCLEAR | WRONG | 合计 |
|---|---:|---:|---:|---:|
| CORRECT | 33 | 0 | 22 | 55 |
| UNCLEAR | 0 | 1 | 1 | 2 |
| WRONG | 8 | 1 | 31 | 40 |
| 合计 | 41 | 2 | 54 | 97 |

| 指标 | 结果 |
|---|---:|
| 状态完全一致 | 65/97 = **67.01%** |
| WRONG 检测 TP / FP / FN / TN | 31 / 22 / 9 / 33 |
| WRONG 检测 Precision | **58.49%** |
| WRONG 检测 Recall | **77.50%** |
| WRONG 检测 F1 | **66.67%** |
| WRONG 概率 Brier score | 0.2662 |
| 两者均判为 WRONG 时的错误类型正确率 | 15/31 = **48.39%** |

WRONG 检测的二分类指标排除 2 个 R1 UNCLEAR 对象，因此 TP+FP+FN+TN=95。状态完全一致则保留全部 97 个对象。

| 执行状态 | 数量 | 含义 |
|---|---:|---|
| `APPROVED_FOR_DERIVED_MAP` | 19 | 诊断和 verifier 均达到对应动作门槛 |
| `BELOW_DIAGNOSIS_THRESHOLD` | 14 | 诊断置信度不足 |
| `BELOW_VERIFICATION_THRESHOLD` | 2 | verifier 同意方向但置信度不足 |
| `NEEDS_FULL_MEMBER_PASS` | 12 | 需要完整成员级重建，本版本不自动执行 |
| `VERIFIER_REJECTED` | 2 | verifier 拒绝诊断或修复动作 |
| `NO_MUTATION` | 48 | 判定正确/不明确，或没有允许的可执行修复 |

### 6.3 隔离派生图结果

| 场景 | 原图对象 | RELABEL | MERGE_WITH | 实际应用 | 派生图对象 |
|---|---:|---:|---:|---:|---:|
| room0 | 72 | 11 | 2 | 13 | 70 |
| office0 | 29 | 6 | 0 | 6 | 29 |
| 合计 | 101 | 17 | 2 | 19 | 99 |

19 个通过门控的动作全部在新派生文件中执行，原 pickle、原 membership 和冻结证据均未原地修改。派生 pickle 的顶层 `objects`、`final_membership` 和 `edges["objects"]` 对象快照已逐对象验证一致；两场景实际边集仍为空。所有派生输出哈希、源 pickle 哈希和源 membership 哈希均复算通过。

事后加载 R1 标签后，19 个已应用候选中 11 个对应 R1 WRONG，8 个对应 R1 CORRECT。这些人工标签没有用于过滤派生图，因为过滤会泄漏真值并破坏“纯 VLM”实验定义。因此派生图是完整保留方法成功与失败的研究产物，不是可替换原图的生产结果。

### 6.4 耗时口径

代表性正式分片的清单记录为 12 个 endpoint 耗时 662.26 秒，端到端平均为 **55.19 秒/endpoint**。这是单 endpoint 自身的检查延迟，不会因多密钥并行而变成 11 秒。使用 5 个 API 分组后，理想吞吐量约为 **11.04 秒完成一个 endpoint**。

按 55.19 秒线性折算，97 个 endpoint 完全串行约需 89.2 分钟，理想五路并行约需 17.8 分钟，实际还会有分组、网络和格式重试开销。如果把扫描全部 97 个 endpoint 的成本均摊到 R1 的 40 个真错误，则约为 **134 秒/真错误**，即约 2 分 14 秒。

### 6.5 结论与可复现位置

这个版本的有效信息是：纯 VLM 对明显语义身份错误具有一定的候选召回能力，但在“保存标签是否必须修改”和结构性错误上过度自信。已知案例 `incident_05c2ca82e74170f33cb1` 在 R1/R2 均为 FALSE_SPLIT，但 VLM 在获得完整 10 张证据图后仍判为 CORRECT。因此，本版本应定位为一个可复现的 VLM-only baseline，用于识别后续需要补强的证据表示、结构推理和保守门控，而不是最终修复策略。

- 隔离工作树：`/home/chenkejun/beauty/conceptgraphs/code/official/ali-my-VLM`
- 分支：`ali-my-VLM`
- 实现提交：`4e4a3b7 feat: add isolated VLM-only endpoint repair pipeline`
- 派生图一致性修复提交：`02bf0e9 fix: keep derived map graph snapshots coherent`
- 97 对象正式结果：`/home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1`
- R1 事后评估：`full_97_locked_v1/evaluation_r1_frozen.json`
- 隔离派生图：`/home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1_derived_maps`
- 派生图清单：`full_97_locked_v1_derived_maps/derived_map_manifest.json`
- 服务器方法文档：`docs/ALI_MY_VLM_ONLY_REPAIR_V1.md`
- 回归测试：`12 passed`

## 7. 当前不能有效测试的指标及原因

| 指标 | 原因 |
|---|---|
| 真正的 Predicate R@1 / mR@1 | 两个正式运行均 `make_edges=false`，边 JSON 为空，没有关系模型输出；当前 0 分只能视为管线结构性结果。 |
| scene-graph object tag / possible tags | 没有生成 scene graph、VLM caption 或候选标签，evaluator 的 0 行无有效输入。 |
| ali-my 的 Replica 8 场景总指标 | 服务器数据中只有 room0、office0 两份可审计的 ali-my 正式冻结地图，其余场景没有 ali-my 预测地图。 |
| 全图 recall、specificity、FPR | R1 标注对象是被规则触发的 97 个 endpoint；未触发节点没有全量人工真值，不能从该样本估计这些总体指标。 |
| 独立标注者一致性 | R2 是同一复核者重测，不是两名独立标注者。 |
| root-stage 准确率、修复后下游增益 | VLM-only 派生图已生成，但它未通过 expert trace 的因果归因，且尚未重跑语义分割、物体识别和 replay。因此不能把 19 个派生修复解释为已证明的指标增益。 |

## 8. 运行与复现说明

- 原 ali-my 证据审计/指标代码：服务器分支 `exp/audit-validity-gate-v1`，HEAD `bff233ff004939d2ecf4ac5546f87cb7b7b16e60`。
- VLM-only 版本：独立工作树 `ali-my-VLM`，独立分支 `ali-my-VLM`，远程 HEAD `02bf0e9671b4256d32bed43598391857b90a2812`。原 `ali-my` 工作树和分支未被该版本修改。
- 指标评测 GPU：只短时使用物理 GPU1 做 OpenCLIP 文本编码，约增加 4 GiB 显存；点云最近邻在 CPU 上运行。任务结束后 GPU1 回到原有占用。GPU3 未使用。
- VLM-only GPU：所有视觉推理通过第三方 API 完成，未占用任何本地 GPU，GPU3 未使用。
- 自动化验证：原指标/审计测试 `63 passed`；VLM-only 审计、schema、门控、UUID 序列化和派生图对象快照一致性测试 `12 passed`。
- 物体 evaluator 的 main room0 回归精确一致；语义 room0 回归精确一致，office0 的最近邻后端差异已单独披露。
- 下载到本地的关键输出 SHA-256 与服务器一致。

## 9. 原始结果位置

- 语义完整 JSON：`staging_remote/ali_my_evaluation_20260821/semseg_gpu1_text_cpu_knn/semseg_results.json`
- 语义汇总 CSV：`staging_remote/ali_my_evaluation_20260821/semseg_gpu1_text_cpu_knn/semseg_results.csv`
- 语义混淆矩阵：`staging_remote/ali_my_evaluation_20260821/semseg_gpu1_text_cpu_knn/semseg_conf_matrices.npz`
- room0 物体完整结果：`staging_remote/ali_my_evaluation_20260821/replicassg_gpu1/room_0/results.json`
- office0 物体完整结果：`staging_remote/ali_my_evaluation_20260821/replicassg_gpu1/office_0/results.json`
- R1/R2 冻结证据：`staging_remote/github_results/`
- 审计方法说明：`docs/ALI_MY_EVIDENCE_AUDIT_METHOD_GUIDE.md`
- VLM-only 97 对象结果：`/home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1/`
- VLM-only R1 事后评估：`/home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1/evaluation_r1_frozen.json`
- VLM-only 派生图：`/home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1_derived_maps/`
- VLM-only 派生图清单：`/home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1_derived_maps/derived_map_manifest.json`
