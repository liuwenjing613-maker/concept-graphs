# ConceptGraphs `ali-my` 有效性门执行总览：final endpoint census v2.1

更新时间：2026-08-21（Asia/Shanghai）
服务器：`chenkejun@frp-van.com:64906`
代码目录：`/home/chenkejun/beauty/conceptgraphs/code/official/ali-my`
当前验证目录：`/home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1`
历史目录：`validation_gate`（finding v1）与 `validation_gate_incident_v2`（trigger-incident 中间版）均完整保留，只追溯，不继续标

这份文档只做一件事：让接手的人不用翻聊天记录和终端日志，就能理解为什么人工方案被简化、R1 得到了什么结论、screeners 哪部分有效或无效、R2 正在检验什么，以及下一步为什么必须做专家因果追踪与 replay。

## 当前结论：R1 与 R2 均已完成，40 个确认错误进入专家追踪，5 个分歧单独待裁决

旧版把一条 checker finding 当成一个人工案例，同时问证据、规则真假、根因阶段、最终危害、修复动作与修复范围。它在 **0/160** 标签时被主动停止。第一次 trigger-incident 去重后又发现 147/160 仍重复同一 final-owner set，同一个 object 最多判断 11 次，因此中间版也在 0/160 时停用。两次都没有无效标签需要迁移。

深入复核后确认旧方案有三个方法问题：

1. 同一物理事件会被 detection、segmentation、association、fusion 等多个 checker 重复报警，逐条标注会把一个错误算多次。
2. 上游暂态异常可能被后续 merge、denoise 或谱系收敛解决；如果最终地图已正确，再追究这个暂态 finding 对当前终态有效性没有意义。
3. 人看到最终 object 时不能可靠反推一个未保存的历史阶段，也不能凭图猜哪种修复重跑后一定有效。

当前正式协议因此改成：

```text
冻结的双场景 evidence ledger
        ↓ Evidence Gate / parity PASS
分层 checker 产生 findings
        ↓
所有阶段、checker、trigger observations
        ↓ 按完全相同的 active final-object set 聚合
唯一 final endpoint
        ↓ 没有可复核终态且缺阶段忠实证据 → 机器阻断
98 个不同 final endpoints
        ↓ 1 个阻断；其余全部纳入，不抽样
97 个 endpoint evidence packets（room0 69 + office0 28）
        ↓ exact final map / 完整 membership / 同源 observation / 哈希绑定
R1 只判断最终状态
        ↓
CORRECT → 计入终态正确
UNCLEAR → 计入证据覆盖缺口，不猜
WRONG   → 进入专家因果追踪
        ↓
局部 intervention + replay
        ↓
只有重跑真实改善才叫 repair verified
```

R1 已完成 97/97，并冻结为：

```text
55 CORRECT
40 WRONG
 2 UNCLEAR
95 / 97 evidence sufficient
```

正式决策是 `PROCEED_TO_EXPERT_TRACE`，40 例专家队列已生成。当前 `review_score` 的 ROC AUC 只有 0.420，Top 20 仅命中 6 个错误，低于 42.11% 的可判 endpoint 错误基线，因此不能把该分数当作错误概率或有效优先级。完整解释见主方法文档第 35 节。

R2 也已完成 24/24：证据充分性一致 23/24，最终状态一致 20/24（83.33%，κ=0.706），三字段完全一致 19/24。R1 的 10 个 `WRONG` 在 R2 全部仍为 `WRONG`，两轮都判错时错误类型 9/10 一致；分歧主要是 3 个 `CORRECT` 和 1 个 `UNCLEAR` 在第二轮改判 `WRONG`。这些结果是同一复核者、短间隔 test-retest，不是 inter-rater reliability。

## 实际完成量

| 项目 | room0 | office0 | 合计 |
|---|---:|---:|---:|
| 原始 findings | 3720 | 1867 | 5587 |
| 通过基础资格的 findings | 3715 | 1862 | 5577 |
| 阻断前 final endpoints | 70 | 28 | 98 |
| 被合并掉的重复 findings | 3645 | 1834 | 5479 |
| 同时触发多个 checker 的 endpoints | 57 | 23 | 80 |
| 因无忠实可复核终态而 fully blocked | 1 | 0 | 1 |
| 可复核 final endpoints | 69 | 28 | 97 |
| R1 全量纳入 | 69 | 28 | 97 |

两个场景都满足 Evidence Gate PASS、population uncensored、case build 无 warning。97 个 endpoints 正好对应 97 个不同 final objects，任何 object 都没有跨 endpoint 重复。97/97 页面包均通过 worklist 绑定、artifact 哈希、final object UID、完整 membership、点数和 final pickle linkage 核对；共检查 5069 个展示资产。

阻断的不是“系统觉得难”，而是 endpoint 若只剩依赖正式运行未保存历史 PCD 的信号，就不能让人用别的图代替。若同一 final object 还有其他可复核信号，缺口信号只保留在内部历史，不再单独制造人工案例。

## 人工复核已经完成；若要统一最终真值，只需裁决 5 个分歧

页面先展示最终对象，再展示成员代表视图；只有仍有疑问时才展开代表性 trigger 和当时的 association。待判对象固定为 `O1 [ENDPOINT]`，其他对象明确写 `[context]`。checker 名、阶段、规则 subtype、抽样队列和 review score 均不显示，避免暗示答案。

1. `evidence_sufficient`
   - `YES`：现有最终对象、成员与视图足以明确判断对错。
   - `NO`：关键视角、对象或几何不足，不能可靠判断。
2. `final_state`
   - `CORRECT`：即使中间出现过重复 proposal、低 margin 或临时 CREATE，最终节点数量、身份、成员与几何已经正确。
   - `WRONG`：错误仍保留在最终地图。
   - `UNCLEAR`：证据不够；`evidence_sufficient=NO` 时固定选它。
3. 仅在 `WRONG` 时选一个可见终态错误类型：`FALSE_MERGE`、`FALSE_SPLIT`、`SPURIOUS_OBJECT`、`MISSING_OBJECT`、`WRONG_MEMBERSHIP`、`GEOMETRY_CORRUPTION`、`SEMANTIC_IDENTITY_ERROR` 或 `OTHER`。其他情况固定为 `NOT_APPLICABLE`。

R1 97 例和 R2 24 例都已完成，不需要继续打开标注页。R2 页面和 worklist 均未包含 R1 答案。5 个任一字段不一致的 endpoint 已写入 `expert/r2_disagreement_queue.jsonl`，未混入 40 个确认错误。若论文需要一份最终统一真值，最小的后续人工任务是让另一位不知道两轮答案的人只裁决这 5 例；否则直接把它们作为重复稳定性限制报告。

## 当前入口

R1 与 R2 都已冻结，`8765` 和 `8766` 服务均已关闭。下面的隧道只保留作历史操作记录，当前不需要再运行：

```bash
ssh -N -L 8766:127.0.0.1:8766 -p 64906 chenkejun@frp-van.com
```

服务运行时打开 `http://127.0.0.1:8766/`。协议为 `final_endpoint_r2_v2_1`，共 24 例。冻结标签为：

`/home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1/labels/labels_r2_frozen_20260821.jsonl`

不要编辑冻结的 R1/R2 worklist 或 R1 标签，不要回到历史 finding v1 页面。

## R1 后机器已经完成的工作

机器已严格校验 97 条标签并计算完整 endpoint 普查的 coverage、confirmed error count、条件错误率、全样本 confirmed yield、上下界、错误类型、linked checker/stage、review_score 排序和复核时长。当前没有抽样缺口，因此 headline 不使用 calibration 权重。

只有 `evidence_sufficient=YES + final_state=WRONG` 的 40 个 incidents 已进入 expert trace queue。专家阶段只建立根因假设与候选干预；随后执行真实 intervention/replay，对比对象图是否改善。没有 replay 改善就不能写成“修复有效”。本轮 R2 确由同一人完成，所以只报告 intra-rater/test-retest 稳定性；没有伪造成 inter-rater agreement。

## 已完成的工程核验

- 新协议相关测试与底层 evidence/audit 回归共 63 项通过。
- 额外纳入 `test_general_utils.py` 时，当前基础环境因既有的 `supervision` 缺失在收集阶段停止；它与本次实现无关。
- 正式新审计、组包、服务与指标预检均使用 CPU，`CUDA_VISIBLE_DEVICES` 为空；未占用任何人的 GPU，GPU3 也完全未使用。
- 历史 root、失败现场和现有未跟踪权重/缓存均保留，没有清理或覆盖他人文件。

更完整的方法逻辑、每个选项的解释与真实 R1/R2 评估，见 `docs/ALI_MY_EVIDENCE_AUDIT_METHOD_GUIDE.md` 第 34～35 节。

---

# 以下是 finding v1 与 trigger-incident 中间版的历史执行记录

下面内容保留用于理解机器侧门禁、修复历史和这次协议纠偏的来源。凡是出现“finding 级 R1”“32 例 R2”“人工判断根因/修复”的地方，都只描述已经退役的旧协议，不再照做。

## 先看结论

**机器侧的建图、证据账本和“人类看到的证据是否忠实于系统”三层门禁均已完成；查错器的研究有效性仍待真实人工标注。**

具体来说：

- 代码冻结、两个 P0 修复、定向测试、120 帧证据开/关一致性、room0 与 office0 两个 200 帧正式包均已完成。
- 两个正式场景的 Evidence Gate、审计有效性门和 population uncensored 门全部 PASS。
- room0 与 office0 各生成 80 个可视案例，共 160 个；无案例构建警告。
- 160 例 R1 工作清单和 32 例独立 R2 复核清单已经准备好，指标工具也已完成并测试。
- 原 R1 页面曾被暂停：它把抽样 observation 的 `pcd_overlay.png` 放在图库中，却没有直接展示最终 object；这不足以支撑“最终地图危害”判断。暂停时标签是 **0/160**，因此没有无效旧标签需要清理。
- 现在 160/160 例都新增了可追溯的人类证据投影：系统触发记录、原始 RGB/mask/depth、决策时 object version、代表视图覆盖率，以及直接从最终 map pickle 读取的完整 object 成员与几何。
- 全量复核结果为：134 例 `TRACEABLE`；26 例 `TRACEABLE_WITH_CRITICAL_GAP`。这 26 例不是文件损坏，而是正式运行当时没有保留某个历史状态的完整 PCD 快照，页面会如实声明，绝不拿别的点云冒充。
- **没有用模型判断冒充人工标签。** 因此 weighted precision、actionable precision、P@20 和最终 GO / CONDITIONAL GO / STOP 目前都应写成 `PENDING HUMAN LABELS`。
- 本轮没有自动删除、拆分、重关联或合并任何地图对象。

最准确的当前表述是：

> 证据系统已经证明“能完整运行、能自检、不会改变原建图、不会因为规则上限而截断总体”；新版 R1 还证明“人看到的触发 observation、对象身份和最终 object 与系统账本及最终 pickle 一致”。查错规则是否“报得准、定位对、确实有害、值得修”，仍必须由 160+32 个人工标签回答。

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
发现旧 R1 只展示抽样 observation PCD，无法可靠判断 final object
    ↓ R1 在 0/160 暂停
为 160 例建立“系统记录 → 人类页面 → 最终 pickle”证据投影
    ↓ 160/160 哈希、成员集合、点数核对通过
修正指标：证据不足是覆盖缺口，不再暗算成 finding=NO
    ↓
当前停在：新版 R1 已可开始，之后做独立 R2 与分歧裁决
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
| 当前验证分支头 | `exp/audit-validity-gate-v1` / `55fc8bc` | 在正式建图快照之后增加盲审抽样、人类—系统证据投影、final object 页面、条件化指标和 R2/裁决硬门；不参与建图 |

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

### 5. 修复“人看到的证据”和“系统使用的证据”不一致

这是本次追加检查中最重要的修正。

旧页面虽然有 RGB、crop、mask 和 `pcd_overlay.png`，但它们只是 packet 选中的 observation 视图。系统 checker 实际还会读取 association 候选分数、object versions、完整 final membership 等结构化记录。尤其是旧 `pcd_overlay.png`，它只是若干 observation PCD 的叠加，**不是最终地图 object**。如果让人凭它填写 `downstream_harm`，页面问题和系统问题并不等价，标签即使填满也没有可靠含义。

新版页面把每例证据固定分成六步：

1. **先看本例真正要回答的问题**：判断现实中是否有建图错误，而不是确认“阈值是否触发”。例如 `LOW_MARGIN` 的 margin 数值当然是真的，但低 margin 本身不等于关联错误。
2. **看系统当时做了什么**：直接展示 association 决策、Top-1/Top-2、每个候选的空间/视觉/综合分数和决策时 object version。
3. **看触发 observation 的同源证据**：六联图中的 RGB、raw mask、processed mask、mask 变化、depth 和 observation PCD 都来自同一条 ledger 记录所引用的 artifact。
4. **看对象身份和代表视图**：对象用稳定别名串起来；每组视图明确写出“选中几张 / 完整成员共几张”，代表视图不再冒充完整历史。
5. **看最终地图 object**：直接读取 manifest 用 SHA-256 锁定的最终 map pickle；页面同时显示完整成员数、帧范围、类别统计、bbox、点数和完整 PCD 哈希，并生成统一世界坐标图与逐对象放大图。
6. **最后才看旧 packet 材料**：旧图仍保留作追溯，但页面明确标出 `pcd_overlay.png` 不是 final object。

页面展示现在有三种明确身份：

| 页面内容 | 能否当成系统精确证据 | 人应该怎么用 |
|---|---|---|
| checker 数值、ledger 引用的 RGB/mask/depth/PCD、object version、final pickle | 是 | 可直接支撑对应事实 |
| 从完整成员中抽出的代表视图 | 不是全部，但来源可追溯 | 用于理解物理关系，同时查看 `selected / total` 覆盖率 |
| 正式运行没有保存的历史 PCD 快照 | 不存在 | 页面必须声明缺口；若它阻断判断，选 `PARTIAL` 或 `NO` |

全量回填和核对结果：

| 核对项 | 结果 |
|---|---:|
| R1 案例 | 160/160 |
| 可追溯且没有已知关键视觉缺口 | 134 |
| 可追溯但有明确关键视觉缺口 | 26 |
| 源 artifact 哈希一致 | 160/160 |
| 可用最终 object 的 UID、完整成员集合与点数和最终 pickle 一致 | 160/160 |
| 能追溯到至少一个 active final object | 160/160 |
| 最终指标门重新读取 review JSON | 160/160 |
| 最终指标门重新计算页面图片哈希 | 9918/9918 |

26 例缺口分布如下：

| Checker | 例数 | 真正缺少什么 |
|---|---:|---|
| `SEG-002` | 11 | DBSCAN 前点坐标未保存；只有精确聚类统计和 DBSCAN 后 PCD |
| `GEO-003` | 10 | 同上，不能目视还原触发时的多簇几何 |
| `GEO-005` | 1 | 去噪前后有精确版本统计，但没有两个时刻的完整 object PCD |
| `FUSE-007` | 4 | 融合前后有中心、尺度、点数和成员变化，但没有两个版本的完整 object PCD |

这些缺口不自动等于 `evidence_sufficient=NO`。如果其余证据已经足以排除合理反例，可以继续判断；如果缺失状态正是结论所依赖的关键环节，就必须选 `PARTIAL` 或 `NO`。这种诚实的不确定性比填一个看似完整但不可复现的 YES/NO 更有价值。

## 测试与非干扰性验证

### 代码测试

- 原有效性门服务器测试：`28 passed`，覆盖 evidence/audit/layered audit、指标、R2 抽样和旧 R1 标签校验。
- 本次修改直接相关回归：`33 passed`；再加入 evidence recorder 与 Evidence Gate 回归后，系统环境可收集的完整相关集合为 `42 passed`。覆盖 layered audit、R2 子集、R1 服务、标签逻辑、指标条件化、顶层人类证据 manifest 与逐图片复核门禁，以及 R2/分歧裁决硬状态门。
- 新服务对真实 160 例逐例执行 `case_payload`，重新核验 review JSON、case JSON 和所有页面图片哈希：`ALL_CASE_PAYLOADS_OK=160`。
- 无头浏览器实际渲染检查通过：普通案例显示 3 张 final object 卡和 2 张最终几何图；带缺口的 `FUSE-007` 案例正确显示缺口声明；两次检查均无页面脚本错误。
- 映射环境另外直接通过 2 个零检测链测试：空过滤结果结构正确、空 detection batch 绕过 CLIP 并返回正确维度。
- `git diff --check` 通过。

测试环境被刻意分开：系统环境有 pytest 但没有 `supervision`，因此直接运行整个 `tests/` 会在收集 `test_general_utils.py` 时报告缺少该包；排除这一项后的 6 个测试文件共 42 项全部通过。mapping 环境有真实映射依赖但没有 pytest，并已直接通过 2 个零检测链测试。没有为追求“一条命令全绿”而改动服务器公共环境。

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
- 本次 160 例人类证据回填、哈希核对、测试和网页渲染全部显式禁用 CUDA，只用 CPU；没有占用 GPU3，也没有挤占任何其他 GPU。
- 根盘最终约剩 88G，使用率 95%。没有删除他人文件或旧运行；新增大产物均放在本轮独立目录。

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
| `review_evidence_manifest.json` | 160 例人类证据投影总清单、worklist 哈希、完整性状态和缺口分布 |
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
| `.../cases/<finding_uid>/review_evidence.json` | 该例系统触发记录、对象角色、代表视图覆盖、final object 和证据缺口 |
| `.../cases/<finding_uid>/review_observation_Q*.png` | 同一 observation 的 RGB/raw mask/processed mask/depth/3D 六联图 |
| `.../cases/<finding_uid>/review_final_objects_relative.png` | 相关 final objects 的统一世界坐标图，用于判断重合或分离 |
| `.../cases/<finding_uid>/review_final_objects_detail.png` | 每个 final object 独立放大图，用于判断自身几何 |

## 160+32 个人工环节怎么接

### R1 首轮

- 由一位真实复核者完成全部 160 例。
- room0、office0 各 80；每场景都是 40 calibration random + 40 diagnostic priority。
- 推荐直接打开已经启动的 `http://127.0.0.1:8765/` 页面，不需要编辑 JSON。
- 页面隐藏 cohort、certainty 和 review score，填写后自动生成 `labels_r1.jsonl`，抽样元数据不会被人工修改。
- 当前服务状态为 `READY`，进度 `0/160`。从第 1 例开始即可；旧页面没有留下任何标签。
- 每例固定按顺序看：证据对齐状态 → 系统问题 → 当时决策 → 触发 observation → 对象代表视图和覆盖率 → final object → 再填标签。
- `finding_correct` 回答“这是不是现实中的建图错误”，不是“规则阈值有没有被触发”。
- `downstream_harm` 必须看完 final object 再填，不能用旧 `pcd_overlay.png` 代替最终对象。
- `evidence_sufficient=YES` 表示 finding、根因、最终危害和修复都能落定，不能再搭配 `UNCERTAIN`、`UNKNOWN` 或 `NEED_MORE_VIEW`。
- `evidence_sufficient=PARTIAL` 必须在备注中写清缺哪一环；`NO` 时固定使用 `finding=UNCERTAIN`、`root=UNCERTAIN`、`harm=UNKNOWN`、`repair=NEED_MORE_VIEW`、`locality=NOT_APPLICABLE`。
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

指标口径已经修正为：

- 只有 `evidence_sufficient=YES` 的案例进入 finding/actionable/root-stage 准确率分母。
- `PARTIAL` 和 `NO` 是**证据覆盖缺口**，不是 finding 误报，不能暗中按 false 计数。
- calibration 同时报告条件 weighted precision、全样本保守下界和上界。下界只把证据充分且确认的案例计为正；上界假设所有证据不足案例都可能为正。
- priority 同时报告“证据充分案例上的条件 P@10/P@20”“原 Top-K 中证据充分的覆盖率”和“全 K 例上的 confirmed yield”，避免低覆盖制造虚高 P@K。
- root-stage accuracy 只在证据充分且 `finding_correct=YES` 的案例中计算。

冻结的业务门槛仍包括：priority actionable P@20 ≥ 0.70、calibration weighted finding precision ≥ 0.50、weighted actionable precision ≥ 0.35、root-stage accuracy ≥ 0.65、office0 相对 room0 的 actionable precision 绝对下降不超过 0.20，以及足够集中的可行动真错误数量。除此之外新增不可绕过的证据覆盖门：全部案例充分率、calibration 加权充分率、priority Top-20 覆盖率，以及 room0/office0 各自的 calibration 加权充分率都必须 ≥ 0.80。任一未通过，结论固定为 `STOP_OR_REDESIGN_EVIDENCE`，不能用一个场景的高覆盖掩盖另一个场景，也不能用低覆盖下的高 precision 放行。

标签不完整时工具退出码为 2 并报告 `NOT_READY`，不会提前制造结论。

## 四个最容易误读的概念

### Evidence Gate PASS

表示证据文件可读、UID/引用/哈希/shape/事件链一致，账本没有明显自相矛盾。它不表示地图语义正确。

### Human–system evidence projection PASS

表示网页展示的结构化记录、派生图和 final object 能追溯到 checker ledger 与最终 pickle，且文件在生成后没有被改动。它仍不表示证据对每个具体问题都充分；26 例历史快照缺口必须由人判断是否阻断结论。

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

机器侧没有遗留的建图、证据回填或页面修复任务。当前明确分工是：用户只做真实人工判断，其余文件校验、统计和报告由现成工具完成。

1. 用户通过 `http://127.0.0.1:8765/` 完成 160 例 R1；每例只有点击“保存本例”才计入进度。
2. R1 完成后先做标签覆盖和逻辑一致性检查；不要手改 worklist 或抽样权重。
3. 32 例 R2 必须由另一位真实复核者独立完成，不能由 AI 冒充，也不能先看 R1。
4. R1/R2 在 finding、harm、repair 任一字段有分歧时，由人共同裁决并写入 `labels_adjudicated.jsonl`。
5. 之后运行指标工具；它会重新检查系统门、人类证据投影门、证据覆盖、R2 完成状态和裁决状态，再生成最终 `decision.md`。

如果 R2 未完成，状态只能是 `PENDING_INDEPENDENT_R2`；有分歧未裁决，状态只能是 `PENDING_ADJUDICATION`；证据覆盖不足，状态是 `STOP_OR_REDESIGN_EVIDENCE`。只有这些门都通过后，才根据 `decision.md` 选择 GO、CONDITIONAL GO 或 STOP。若最终 GO，也只选一种最集中、危害最大、修复最局部的错误类型进入下一轮，不在本轮同时实现多种自动修复。

更完整的方法原理可继续阅读：`docs/ALI_MY_EVIDENCE_AUDIT_METHOD_GUIDE.md`。本文件记录的是 2026-08-20 这次有效性门的真实执行状态与交接边界。
