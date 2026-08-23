# ConceptGraphs 服务器实验与关系边进度总览

> 与论文及复现 `main` 的统一指标对比见：[`ALI_DEV_MAIN_REPRODUCTION_VS_PAPER_20260822.md`](ALI_DEV_MAIN_REPRODUCTION_VS_PAPER_20260822.md)。该报告区分了可直接对齐的语义指标、论文未报告的 ReplicaSSG 指标，以及不可直接相减的两种关系边 precision。

## 最新：`ali-dev` 关系边与 8 场景同步（2026-08-22）

更新时间：2026-08-22 19:25（Asia/Shanghai）  
服务器：`chenkejun@frp-van.com:64906`  
实验根：`/home/chenkejun/beauty/conceptgraphs/experiments/ali-dev-relations/synced_8scene_gpt56sol_v1`

### 先看结论

**本轮已完成，不再有后台关系任务。**

- 8 个 Replica 场景共建立 4,013 个无序候选对，生成 13,375 张几何/视觉证据，共约 841 MB。候选和推理不读取 GT。
- 5 个 API key 同时使用，最终 4,013/4,013 响应有效，0 个遗留失败。密钥仅位于运行进程环境，未写入任何产物。
- 原模型 `gpt-4o-2024-05-13` 不在网关模型列表；本轮统一使用可用的 `gpt-5.6-sol`，实际返回模型 4,013/4,013 均为 `gpt-5.6-sol`。
- 仅用 `room0 + office0` 开发集冻结实际边阈值 0.99；其余 6 场景不参与调参。完整排名始终保留，可在不重跑 API 的情况下重新设阈值。
- 0.99 阈值下，8 场景实际导出 399 条论文词表有向边和 150 条 `ali-dev` 原生 `on top of/under` 兼容边。
- ReplicaSSG 全量排名指标：**R@1 55.03%、R@3 73.15%、mR@1 19.18%、mR@3 32.96%**。
- 0.99 实际边的闭世界指标：paper 头 P=3.26%、映射 GT R=10.74%、F1=5.00%；`ali-dev` 兼容头 P=7.33%、R=9.09%、F1=8.12%。ReplicaSSG 关系 GT 稀疏，这组 precision 是严格闭世界下界，不是论文主指标。
- `ali-my` 8 场景已同步完成。严格 `ali-dev` 只找到 room0 对象图；经对象数、类别、bbox、CLIP、帧名和候选对六重校验 PASS 后，已将 room0 关系节点字段重映射到 `ali-dev` ID/UUID/tag。
- 关系层是只读后处理；原 `ali-my`/`ali-dev` 对象图 SHA-256 与运行前一致，没有改动任何 PCD 地图。
- 本轮关系候选、证据、API 推理、导出和评测全部使用 CPU/API，没有使用任何 GPU，明确未使用 GPU3。

### 版本、地图和模型选择

| 项目 | 冻结值 |
|---|---|
| `ali-dev` 工作树 | `/home/chenkejun/beauty/conceptgraphs/code/official/ali-dev` |
| `ali-dev` 提交 | `72f5962822b5e8678a446f367a06df1a977d2a4d`，detached HEAD，tracked tree clean |
| `ali-my` 对齐地图提交 | `bff233f...`，8 场景各 400 帧、stride=5 |
| 关系模型 | 请求与实际返回均为 `gpt-5.6-sol` |
| 提示词 SHA-256 | `1ca792e9d6147c658cfbc23c712a8e86269007c0a9f4289721abac8de66f029c8` |
| 最终脚本 SHA-256 | `4ce4d98d067c02b094323d24e5f941319248c4587a3bd62e2ab26623a16a7f62` |
| `ali-my` room0 源图 SHA-256 | `64ccdb3b6e77f93fe776b0c62673ac41fe86c831a8937f3284d54decf8004dcb` |
| `ali-dev` room0 目标图 SHA-256 | `8a94dbafb8b7617b61d3de029dcd15c09128daed258238d47e4c2c788f01821f` |

网关 `/v1/models` 中没有原始 `gpt-4o-2024-05-13`。为避免场景间模型混用，不用更快但较弱的模型中途接续，而是全程使用同一 `gpt-5.6-sol`。

### 关系方法与证据约束

本轮没有盲目全连接，也没有使用 GT 挑候选。冻结候选集是以下无序对的并集：

- AABB 相交；
- 3D 表面间隙 ≤ 1.75 m；
- 共视且中心距离 ≤ 3.0 m；
- 每个非背景对象最多 8 个近邻，同时保留 2.625 m 表面间隙上限。

每对最多附带 3 张共视 mask overlay 和 1 张 3D 几何图。A 为红色、B 为青色；图像、源图和提示词均以 SHA-256 绑定到单个响应。一次 API 返回两个方向的 paper top-3 以及限制为 `on top of/under` 的 `ali-dev` 兼容头。

paper 词表与 ReplicaSSG 对齐：`on / in / near / above / under / attached to / with`，另有 `none` 用于拒绝弱证据。排名 R/mR 不设置信心阈值；只有实际边导出和 P/R/F1 使用 0.99 阈值。

### 候选、证据与完整性总账

| 场景 | 候选对 | 证据图 |
|---|---:|---:|
| office0 | 296 | 985 |
| office1 | 209 | 766 |
| office2 | 395 | 1,327 |
| office3 | 669 | 2,211 |
| office4 | 356 | 1,236 |
| room0 | 896 | 2,938 |
| room1 | 632 | 2,033 |
| room2 | 560 | 1,879 |
| **合计** | **4,013** | **13,375** |

`integrity_audit.json` 总状态为 **PASS**：

- 4,013 个候选、4,013 个有效响应、4,013 个唯一 case ID、4,013 个唯一无序对；
- 自环 0，证据/地图/提示词/响应身份绑定问题 0；
- 5 个密钥槽最终分担 803 / 803 / 803 / 802 / 802 个响应；
- evidence quality：`good=2018`、`partial=1946`、`poor=49`；
- 模型标记为 same physical object 的候选 83 对，仅作诊断，没有自动合并对象；
- API 用量：prompt 30,563,536 tokens，completion 1,596,018 tokens，总计 32,159,554 tokens；
- 单候选延迟均值 12.391 s，中位数 11.605 s，p95 18.678 s，最大 38.048 s。

中途在 office2 遇到一次短时网关波动，主轮出现连续失败。操作为：立即暂停新请求、单例验证网关恢复、按缺失文件精确补跑，再执行全量断点闭环。最终 manifest 为 900 cached + 3,113 completed + 0 errors，没有把失败当作空关系。

### 开发集阈值选择

阈值仅使用 room0 + office0 的 41 条 GT 关系。31 条的两个端点能被几何匹配，且 31/31 均在候选集中；候选召回上限为全部 GT 的 75.61%，给定端点匹配后为 100%。

| 阈值 | paper 命中 | paper 边 | 闭世界 P | 映射 GT R | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 20 | 2,019 | 0.99% | 64.52% | 1.95% |
| 0.70 | 20 | 1,876 | 1.07% | 64.52% | 2.10% |
| 0.80 | 20 | 1,640 | 1.22% | 64.52% | 2.39% |
| 0.90 | 20 | 1,303 | 1.53% | 64.52% | 3.00% |
| 0.95 | 19 | 851 | 2.23% | 61.29% | 4.31% |
| 0.975 | 14 | 334 | 4.19% | 45.16% | 7.67% |
| **0.99** | **5** | **88** | **5.68%** | **16.13%** | **8.40%** |
| 0.992–1.000 | 0 | 0 | 无定义 | 0% | 无定义 |

模型最高实际信心度为 0.99，因此 0.99 是可用分数网格上闭世界 F1 的真实峰值。选定 0.99 为实际 sidecar 的主阈值，同时保留完整 `ranked_relations.json`；需要更高召回时可直接改用 0.95，无需再请求 API。

### ReplicaSSG 全量指标

几何匹配严格沿用仓库 `scripts/eval_replicassg_main.py` 的协议：单向 KD-tree 0.1 m，预测 segment 最佳 GT 覆盖率至少 50%，second/best 不高于 0.75，每个 GT 选最大匹配预测对象。

| 集合 | GT 关系 | 端点匹配 | 候选覆盖 | R@1 | R@3 | mR@1 | mR@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部 8 场景 | 149 | 121 | 121 | **55.03%** | **73.15%** | **19.18%** | **32.96%** |
| 开发集 room0+office0 | 41 | 31 | 31 | 48.78% | 58.54% | 17.86% | 33.93% |
| 留出 6 场景 | 108 | 90 | 90 | 57.41% | 78.70% | 33.19% | 56.83% |

全部 8 场景的候选上限为 121/149=81.21%；在端点能匹配的关系中，候选覆盖为 121/121=100%。排名条件 R@1（给定端点/候选）为 82/121=67.77%。

| 谓词 | GT | 端点匹配 | R@1 命中 | R@3 命中 | R@1 | R@3 |
|---|---:|---:|---:|---:|---:|---:|
| near | 79 | 69 | 59 | 67 | 74.68% | 84.81% |
| on | 40 | 24 | 22 | 22 | 55.00% | 55.00% |
| with | 22 | 22 | 1 | 20 | 4.55% | 90.91% |
| in | 3 | 2 | 0 | 0 | 0% | 0% |
| above | 2 | 2 | 0 | 0 | 0% | 0% |
| under | 2 | 2 | 0 | 0 | 0% | 0% |
| attached to | 1 | 0 | 0 | 0 | 0% | 0% |

不应只看 73.15% 的 R@3：它主要来自 `near`、`on` 以及 `with` 在 top-3 中的覆盖；`in/above/under/attached to` 仍然是明确短板。

0.99 阈值实际边总计：

| 头 | 导出边 | 映射 GT 真阳性 | 闭世界 P | 映射 GT R | F1 |
|---|---:|---:|---:|---:|---:|
| paper | 399 | 13 | 3.26% | 10.74% | 5.00% |
| `ali-dev` compatible | 150 | 11 | 7.33% | 9.09% | 8.12% |

### 每场景结果（0.99 实际边）

| 场景 | GT | 端点/候选 | R@1 | R@3 | mR@1 | mR@3 | paper 边/TP/F1 | 兼容边/TP/F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| office0 | 16 | 15/15 | 37.50% | 50.00% | 12.50% | 29.17% | 33 / 0 / 0% | 11 / 0 / 0% |
| office1 | 7 | 7/7 | 57.14% | 85.71% | 33.33% | 50.00% | 33 / 0 / 0% | 10 / 0 / 0% |
| office2 | 17 | 16/16 | 70.59% | 94.12% | 60.00% | 93.33% | 55 / 2 / 5.63% | 21 / 2 / 10.81% |
| office3 | 19 | 17/17 | 47.37% | 78.95% | 57.14% | 82.14% | 29 / 4 / 17.39% | 11 / 4 / 28.57% |
| office4 | 18 | 12/12 | 66.67% | 66.67% | 66.67% | 66.67% | 54 / 0 / 0% | 25 / 0 / 0% |
| room0 | 25 | 16/16 | 56.00% | 64.00% | 37.50% | 45.83% | 55 / 5 / 14.08% | 16 / 3 / 18.75% |
| room1 | 16 | 9/9 | 43.75% | 43.75% | 37.04% | 37.04% | 67 / 0 / 0% | 34 / 0 / 0% |
| room2 | 31 | 29/29 | 58.06% | 93.55% | 50.09% | 86.67% | 73 / 2 / 3.92% | 22 / 2 / 7.84% |

### `ali-dev` room0 同步门

源是 `ali-my` 400 帧 stride-5 room0 图，目标是服务器现有严格 `ali-dev` room0 图。不直接复制节点 ID，而是先做以下 parity gate：

| 门 | 结果 |
|---|---:|
| 对象数 | 72 = 72 |
| 按索引类别一致 | 72/72 |
| 按索引背景标记一致 | 72/72 |
| bbox 最大绝对差 | 0.0018206 m |
| CLIP 余弦最小/平均 | 0.999917 / 0.999999 |
| 完整路径 Jaccard | 0（两侧运行根不同，预期值） |
| 归一化帧文件名 Jaccard 最小/平均 | 1.0 / 1.0 |
| 候选对 | 896 = 896，Jaccard=1.0 |
| 最终决策 | **PASS** |

PASS 后仅把 55 条 paper 边和 16 条兼容边的 `object_id/uuid/tag` 重写为目标 `ali-dev` 节点字段。两张图均未修改。

另外用严格 `ali-dev` PCD 重新执行 ReplicaSSG 几何匹配，与 `ali-my` 源图指标完全一致：31 个 GT 对象匹配、16 条关系端点/候选匹配、paper 命中 R@1=14、R@3=16、0.99 实际边命中 5、F1=14.08%；兼容头命中 3、F1=18.75%。

### 服务器产物入口

| 路径 | 内容 |
|---|---|
| `/home/chenkejun/beauty/conceptgraphs/experiments/ali-dev-relations/synced_8scene_gpt56sol_v1` | 主实验：candidates、cases、predictions、exports、阈值扫描、全量/严格评测、完整性审计 |
| `/home/chenkejun/beauty/conceptgraphs/results/ali-my/paper_main_aligned_20260821/relations_gpt56sol_v1` | `ali-my` 8 场景同步 sidecar，每场景 4 个导出 JSON + sync manifest，`_evaluation/` 保留阈值和审计文件 |
| `/home/chenkejun/beauty/conceptgraphs/results/ali-dev/replica_stride5/Replica/room0/relations_gpt56sol_synced_v1` | 严格 `ali-dev` room0 节点重映射边、parity report、全量上下文与目标图独立评测 |
| `/home/chenkejun/beauty/conceptgraphs/scripts/relation_pipeline.py` | 可复现的 build / infer / export / evaluate / audit / sync / remap 脚本 |

每个场景的核心文件：

- `edge_json_paper_aligned.json`：0.99 阈值实际 paper 边；
- `edge_json_ali_dev_compatible.json`：0.99 阈值 `on top of/under` 兼容边；
- `ranked_relations.json`：每个候选、两个方向的完整 top-3 和信心度；
- `export_manifest.json` / `sync_manifest.json`：数量、阈值、文件哈希和源图绑定。

### 运行资源与边界

- 最终检查时没有残留 `relation_pipeline.py` 进程。
- 关系任务不需要 GPU，未使用 GPU3，也未占用当时可用的 GPU1/GPU5。他人或既有 GPU 进程均未触碰。
- 最终根盘约剩 291 GB，未删除他人文件、旧实验或失败现场。
- 严格 `ali-dev` 输入只找到 room0；其他 7 场景是经验证与前期协议对齐的 `ali-my` 400 帧 stride-5 地图，不应在论文中误称为 8 场景严格 `ali-dev` 重建。
- 本轮没有人工逐边标注。ReplicaSSG 已标注关系很稀疏，因此闭世界 FP 包含“证据上可能合理但 GT 未标”的边；主结论应以论文排名 R/mR、端点匹配率和候选上限为主，P/R/F1 作严格诊断。
- `with` 在 top-3 表现好但 top-1 很弱；`in/above/under/attached to` 未解决。如后续要提升论文 mR，应优先做谓词定向校准/少样本规则，不应根据留出场景再调阈值。

---

## 既有：`ali-my` 有效性门（2026-08-20）

更新时间：2026-08-20（Asia/Shanghai）  
服务器：`chenkejun@frp-van.com:64906`  
代码目录：`/home/chenkejun/beauty/conceptgraphs/code/official/ali-my`  
验证总目录：`/home/chenkejun/beauty/conceptgraphs/validation_gate`

这份文档只做一件事：让接手的人不用翻聊天记录和终端日志，也能理解这一轮为什么做、实际做了什么、哪些门已经通过、哪里遇到过真实边界，以及现在为什么还不能宣布查错器“有效”。

## 先看结论

**机器侧验证已经完成并全部通过；查错器的研究有效性仍待真实人工标注。**

具体来说：

- 代码冻结、两个 P0 修复、定向测试、120 帧证据开/关一致性、room0 与 office0 两个 200 帧正式包均已完成。
- 两个正式场景的 Evidence Gate、审计有效性门和 population uncensored 门全部 PASS。
- room0 与 office0 各生成 80 个可视案例，共 160 个；无案例构建警告。
- 160 例 R1 工作清单和 32 例独立 R2 复核清单已经准备好，指标工具也已完成并测试。
- **没有用模型判断冒充人工标签。** 因此 weighted precision、actionable precision、P@20 和最终 GO / CONDITIONAL GO / STOP 目前都应写成 `PENDING HUMAN LABELS`。
- 本轮没有自动删除、拆分、重关联或合并任何地图对象。

最准确的当前表述是：

> 证据系统已经证明“能完整运行、能自检、不会改变原建图、不会因为规则上限而截断总体”；查错规则是否“报得准、定位对、确实有害、值得修”，必须由 160+32 个人工标签回答。

## 整个流程一眼看懂

```text
旧版 v1.1 快照 66c109d
    ↓ 冻结分支
修 P0：伪 similarity + finding population 截断
    ↓ 20 个原审计测试通过
120 帧 evidence OFF / ON 严格对照
    ↓ 七项一致性全部 PASS
room0 200 帧正式建图 + v1.1 审计
    ↓ Gate PASS，80 cases
office0 正式运行遇到“过滤后零检测”真实边界
    ↓ 修复空检测链，增加 2 个边界测试
office0 200 帧正式建图 + v1.1 审计
    ↓ Gate PASS，80 cases
在最终建图提交 e6b0f17 上重跑 parity 与 room0
    ↓ 所有正式运行版本对齐
生成 160 例 R1 清单 + 32 例 R2 清单 + 指标工具
    ↓
当前停在：等待真实 R1、独立 R2 与分歧裁决
```

## 这一轮为什么必须做

上一阶段已经能留下证据、生成几千条 finding 和可视案例，但“finding 很多”不等于“查错器有效”。真正缺的是四个答案：

1. 留证功能会不会悄悄改变原建图？
2. 规则报出的案例有多少是真异常？
3. 真异常中有多少会污染最终对象图？
4. 真且有害的异常中，有多少能对应明确、局部的修复动作？

因此本轮没有继续堆规则，也没有开始自动回滚，而是执行 `docs/ali_my_next_step_validation_gate_plan.md` 中的有效性门。

## 版本与分支

| 用途 | 分支 / 提交 | 说明 |
|---|---|---|
| v1.1 冻结点 | `freeze/ali-my-audit-v1.1-20260820` / `66c109d` | 只用于回溯，不再修改 |
| P0 门禁修复 | `4bcc68f` | similarity 异常与 population censoring 防护 |
| 一致性比较器 | `97207eb` | 生成可落盘的 parity report |
| 正式建图快照 | `e6b0f17` | 增加过滤后零检测和空 CLIP batch 防护；所有最终正式运行均指向它 |
| 当前验证分支头 | `exp/audit-validity-gate-v1` / `70e1146` | 在正式建图快照之后只增加标注抽样、指标工具和带逐项解释的盲审页面，不参与建图 |

验证分支与冻结分支都已推送到 `liuwenjing613-maker/concept-graphs`。正式运行 manifest 显示 `git_dirty=true`，原因是服务器存在已知的未跟踪运行缓存和权重软链接；没有未提交的 tracked source diff，运行快照和源文件哈希均已写入证据。

## 实际修复了什么

### 1. similarity shape 异常不再生成伪证据

旧逻辑在矩阵形状错误时可能用未初始化的 `np.empty` 继续排序，进而写出看似正常、实际随机的 Top-K 和 margin。现在：

- 异常矩阵明确标为 invalid；
- 不再生成 Top-K、排名或 margin；
- semantic checker 不消费这类帧；
- strict Evidence Gate 直接 FAIL；
- runner 用失败退出码阻断正式实验。

这项修复不改变 mapping 决策，只阻止错误证据污染审计结论。

### 2. 不再把“前 500 条”误当作总体

validation config 将单规则 cap 提高到 10000，并记录：

- `attempted_count`
- `emitted_count`
- `suppressed_count`
- `population_censored`

只要任一规则 `suppressed_count > 0`，校准随机样本就不能计算 weighted precision，validation gate 自动 FAIL。最终 room0、office0 的所有规则 suppressed 均为 0。

优先级评分中的 `independent_evidence` 同时更名为更准确的 `support_signal_diversity`，避免把启发式支持信号误称为统计独立证据。

### 3. office0 暴露的零检测边界

office0 的源帧 `001680` 合法地产生了 0 个检测。第一次正式运行因此在第 168 个采样帧失败：过滤函数把空结果组织成了错误形状，后续 CLIP 也不应处理空 crop batch。

最终修复为：

- 过滤后为空时返回结构完整的空 `sv.Detections`：`xyxy=(0,4)`、`mask=(0,H,W)`、confidence/class 均为 `(0,)`；
- CLIP 空 batch 不再预处理图片，直接返回 `crops=[]`、`text=[]` 和 `(0, output_dim)` 特征；
- 该帧在正式 evidence 中记为 `processed=true`，`skip_reason=no_kept_2d_observations`，不会被伪造成错误帧，也不会中断建图。

失败现场没有删除，保存在：

`/home/chenkejun/beauty/conceptgraphs/validation_gate/aborted_runs/office0_full_aborted_empty_filter_frame1680_before_5a56f48`

它是一次有价值的边界发现，不应在总结里被抹掉。

### 4. 补齐人工盲审、指标与决策入口

新增：

- `scripts/generate_validation_gate_r2_subset.py`
- `scripts/compute_validation_gate_metrics.py`
- `scripts/serve_validation_gate_r1.py`

前者稳定生成 16 calibration + 16 priority 的独立复核子集，并保证覆盖两个场景和 detection、association、fusion、object identity。后者会：

- 拒绝缺失、重复、占位、非法枚举或被篡改抽样元数据的标签；
- 将 adjudicated label 作为最终权威；
- 分开计算 calibration weighted precision 与 priority P@K；
- 输出 overall、checker、stage、scene 四类指标；
- 检查冻结的 GO 阈值并生成 `decision.md`。

当前实际执行返回 `NOT_READY`，原因是 `labels_r1.jsonl` 尚不存在。这是正确行为，不是报错遗漏。

R1 页面只绑定回环地址，通过 SSH 本地通道访问 `http://127.0.0.1:8765/`。页面把 160 例稳定打散，隐藏 cohort、certainty、sampling weight 和 review score，自动计时并原子保存；复核者只需查看证据和填写人工判断。

## 测试与非干扰性验证

### 代码测试

- 服务器系统测试环境：`28 passed`。
- 其中包含原 evidence/audit/layered audit 回归、6 个指标与 R2 抽样测试，以及 2 个 R1 页面标签校验测试。
- 映射环境另外直接通过 2 个零检测链测试：空过滤结果结构正确、空 detection batch 绕过 CLIP 并返回正确维度。
- `git diff --check` 通过。

测试环境被刻意分开：系统环境有 pytest 但没有 `supervision`；mapping 环境有真实映射依赖但没有 pytest。没有为追求“一条命令全绿”而改动服务器公共环境。

### 120 帧 evidence ON/OFF 对照

固定同一 detection cache、配置、随机种子和 `PYTHONHASHSEED=0`，分别运行：

- OFF：`ali_my_validity_parity_off_120f_e6b0f17_20260820`
- ON：`ali_my_validity_parity_on_120f_e6b0f17_20260820`

最终比较报告：`/home/chenkejun/beauty/conceptgraphs/validation_gate/parity/parity_report.json`

| 比较项 | 结果 |
|---|---:|
| 最终对象数量 | 一致 |
| canonical observation membership | 一致 |
| bbox / PCD / feature 数值字段 | 容差内一致 |
| 对象 JSON | 一致 |
| edge topology | 一致 |
| 逐帧对象、merge、filter 计数 | 一致 |
| parity trace | 完整存在 |

总状态：`PASS`。

证据开启侧共处理 120 帧、3934 个 raw detection、2263 个 kept observation、1671 个 rejected observation，最终 68 个对象；缺失引用、重复 membership 和 logging error 均为 0。

这证明的是“开启留证旁路没有改变本次映射结果”，不是“地图本身完全正确”。

## 两个正式场景结果

两场景使用同一份冻结配置 `v1_validation.yaml`，没有看完 room0 后再为 office0 改阈值。

### 建图与证据完整性

| 项目 | room0 | office0 |
|---|---:|---:|
| 正式运行提交 | `e6b0f17` | `e6b0f17` |
| 帧数 | 200 | 200 |
| raw detections | 6303 | 3052 |
| kept observations | 3779 | 1560 |
| rejected observations | 2524 | 1492 |
| CREATE decisions | 96 | 35 |
| ASSOCIATE decisions | 3683 | 1525 |
| object merges | 24 | 6 |
| final objects | 72 | 29 |
| missing references | 0 | 0 |
| duplicate memberships | 0 | 0 |
| logging errors | 0 | 0 |
| Evidence Gate | PASS | PASS |

### 分层审计

| 项目 | room0 | office0 |
|---|---:|---:|
| findings | 3720 | 1867 |
| root-cause candidates | 857 | 436 |
| likely / ambiguous / insufficient | 571 / 3144 / 5 | 355 / 1507 / 5 |
| high / medium / low | 1217 / 2435 / 68 | 633 / 1196 / 38 |
| detection | 624 | 388 |
| segmentation | 876 | 452 |
| geometry | 261 | 77 |
| association | 1897 | 927 |
| fusion | 42 | 8 |
| object identity | 20 | 15 |
| population censored | 否 | 否 |
| weighted precision allowed | 是 | 是 |
| cases | 40 random + 40 priority | 40 random + 40 priority |
| case warnings | 0 | 0 |
| 核心审计耗时 | 187.556 s | 81.755 s |

findings 与 root-cause candidates 都只是复核候选。比如 association 数量高，说明当前规则把大量关联决策列为风险点；在人工标签完成前，不能据此声称 office0 比 room0 更准、更差，或已经发现了 927 个真实关联错误。

## 模型权重和 detection cache 是怎么处理的

按“优先复用服务器已有资产”的原则，没有重新下载完整模型，也没有占用 GPU 重新生成可以复用的 room0 detection。

实际复用：

| 模型 | 服务器现有位置 / 处理 |
|---|---|
| YOLO-World | `models/runtime/yolov8l-world.pt`，仓库根软链接 `yolov8l-world.pt` |
| SAM-L | `models/runtime/sam_l.pt`，仓库根软链接 `sam_l.pt` |
| OpenAI CLIP ViT-B/32 | `/home/chenkejun/beauty/weights/clip/ViT-B-32.pt`，项目 weights 目录软链接 |
| OpenCLIP ViT-H | `/home/chenkejun/beauty/conceptgraphs/models/huggingface`，离线 HF cache |

三个中断下载的 partial 文件被保留在 `validation_gate/weights/`，没有冒充完整权重，也没有覆盖已有模型。

旧 office0 detection cache 没有直接复用，因为其类别维度是旧的动态格式（逐帧/全局类别数不固定），而本轮正式配置要求固定 Scannet200 的 200 类。正式 office0 cache 重新生成在：

`/home/chenkejun/beauty/conceptgraphs/data/Replica/office0/exps/office0_detections_stride10_validation_20260820/detections`

它包含 200 个帧目录，类别维度统一为 200；`frame001680` 是合法空检测帧，不是缓存损坏。

## GPU、磁盘和他人任务保护

- 全程明确禁用 GPU3。
- 每次启动前重新检查所有 GPU；只在 GPU2 显示约 27 MiB、0% utilization 时启动本轮任务。
- GPU5–7 等已有大显存进程未触碰，也没有杀进程、迁移进程或接管显卡。
- detection、mapping 结束后 GPU2 已释放；审计和案例渲染使用 CPU/磁盘，不长期占 GPU。
- 根盘最终约剩 89G，使用率 95%。没有删除他人文件或旧运行；新增大产物均放在本轮独立目录。

## 服务器产物索引

验证总入口：

`/home/chenkejun/beauty/conceptgraphs/validation_gate`

| 入口 | 内容 |
|---|---|
| `config/v1_validation.yaml` | 本轮冻结审计配置 |
| `runs/room0/formal` | 指向 room0 最终正式运行 |
| `runs/office0/formal` | 指向 office0 最终正式运行 |
| `parity/parity_report.json` | 最终提交上的 evidence ON/OFF 比较报告 |
| `parity/parity_report_97207eb.json` | 修空检测前提交的历史 parity 报告 |
| `cases/room0` | room0 的 80 个案例 |
| `cases/office0` | office0 的 80 个案例 |
| `labels/r1_worklist.jsonl` | 160 例 R1 空白工作清单 |
| `labels/r2_subset_manifest.jsonl` | 32 例独立 R2 空白清单 |
| `labels/README.md` | 面向复核者的中文填写说明 |
| `metrics/` | 人工标签完成后生成指标；当前保持空白 |
| `decision.md` | 当前明确写为 `PENDING HUMAN LABELS` |
| `aborted_runs/...` | office0 零检测修复前的失败现场与日志 |

正式场景原始目录：

- room0：`/home/chenkejun/beauty/conceptgraphs/data/Replica/room0/exps/ali_my_validity_room0_full_200f_e6b0f17_20260820`
- office0：`/home/chenkejun/beauty/conceptgraphs/data/Replica/office0/exps/ali_my_validity_office0_full_200f_20260820`

每个正式目录里重点看：

| 文件 | 人类应该怎么理解 |
|---|---|
| `evidence/manifest.json` | 谁、哪次代码、什么配置和模型跑出的结果 |
| `evidence/evidence_summary.json` | 帧、检测、关联、合并和完整性总账 |
| `audit/validation.json` | Evidence Gate 是否通过 |
| `audit_validity_gate_v1/audit_summary.json` | findings、根因、截断、案例数和 Gate 汇总 |
| `audit_validity_gate_v1/case_selection.json` | 为什么这 80 例被选中、抽样概率与权重 |
| `audit_validity_gate_v1/cases/<finding_uid>/` | 人工真正要看的可视证据包 |

## 160+32 个人工环节怎么接

### R1 首轮

- 由一位真实复核者完成全部 160 例。
- room0、office0 各 80；每场景都是 40 calibration random + 40 diagnostic priority。
- 推荐直接打开已经启动的 `http://127.0.0.1:8765/` 页面，不需要编辑 JSON。
- 页面隐藏 cohort、certainty 和 review score，填写后自动生成 `labels_r1.jsonl`，抽样元数据不会被人工修改。
- 先看图像、mask/depth、3D overlay、时间线和候选对象，再填写 finding 是否正确、根因阶段、下游危害与修复动作。
- 不应先根据 checker certainty 或 review score 猜答案。

### R2 独立复核

- 另一位真实复核者独立完成 32 例，不能先看 R1。
- 16 calibration + 16 priority；两个场景和关键阶段均已覆盖。
- 对 `finding_correct`、`downstream_harm`、`repair_action` 报告原始一致率。
- 分歧案例共同裁决后写入 `labels_adjudicated.jsonl`。

### 指标和最终决策

标签完成后，在仓库根执行：

```bash
/opt/anaconda3/bin/python scripts/compute_validation_gate_metrics.py \
  --validation-root /home/chenkejun/beauty/conceptgraphs/validation_gate
```

工具将生成：

- `metrics/overall_metrics.json`
- `metrics/metrics_by_checker.csv`
- `metrics/metrics_by_stage.csv`
- `metrics/metrics_by_scene.csv`
- 最终 `decision.md`

冻结的 GO 核心门槛包括：priority actionable P@20 ≥ 0.70、calibration weighted finding precision ≥ 0.50、weighted actionable precision ≥ 0.35、root-stage accuracy ≥ 0.65、evidence sufficiency ≥ 0.80、office0 相对 room0 的 actionable precision 绝对下降不超过 0.20，以及足够集中的可行动真错误数量。

标签不完整时工具退出码为 2 并报告 `NOT_READY`，不会提前制造结论。

## 三个最容易误读的概念

### Evidence Gate PASS

表示证据文件可读、UID/引用/哈希/shape/事件链一致，账本没有明显自相矛盾。它不表示地图语义正确。

### Finding

表示某条规则发现了值得复核的风险信号。它不是已确认错误，也不代表应该自动修。

### Root-cause candidate

表示跨阶段 findings 被聚合后，系统认为更像上游来源的假设。它仍是可解释的因果候选，不是反事实证明。

## 当前明确不做什么

- 不因为 findings 多就继续增加 checker。
- 不把 calibration 与 priority 混成一个总体准确率。
- 不用 AI 自评替代 R1/R2 人工真值。
- 不根据 room0 结果临时调整 office0 阈值。
- 不自动 drop、reassign、merge、split 或 rollback。
- 不删除失败运行、旧缓存或他人 GPU 进程来“清理现场”。

## 下一位接手者只需要做什么

机器侧没有遗留的重跑任务。现在只需确定：

1. 谁完成 160 例 R1；
2. 谁作为独立第二复核者完成 32 例 R2；
3. 分歧由谁裁决。

完成后运行现成指标工具，再根据 `decision.md` 选择 GO、CONDITIONAL GO 或 STOP。若最终 GO，也只选一种最集中、危害最大、修复最局部的错误类型进入下一轮，不在本轮同时实现多种自动修复。

更完整的方法原理可继续阅读：`docs/ALI_MY_EVIDENCE_AUDIT_METHOD_GUIDE.md`。本文件记录的是 2026-08-20 这次有效性门的真实执行状态与交接边界。
