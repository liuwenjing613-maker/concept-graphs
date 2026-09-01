# 实验 0：ATTACH/NEW 身份路由标注协议 v2

日期：2026-09-01  
状态：`R2_SCHEMA_CALIBRATION`。本轮只验证定义和证据，不估计自然错误率，不解封未见场景。

## 1. 唯一主问题

每个保留 observation 的原始路由动作统一写成：

```text
a_t ∈ {ATTACH(o_1), ..., ATTACH(o_n), NEW}
```

人工真值回答：在事件发生前冻结的 `t^-` 地图中，当前 observation 应附着到哪些合法已有节点，还是应新建节点。物理实例由位置、形状、多视角历史和场景上下文决定，不能由检测类别或相似度分数替代。

五个互斥路由结果为：

| 原始动作 | 正确动作 | 路由标签 |
|---|---|---|
| `ATTACH(A)` | `ATTACH(A)`，A 属于合法目标集合 | `CORRECT_ATTACH` |
| `ATTACH(A)` | `ATTACH(B)`，A 不合法 | `WRONG_ATTACH_EXISTING` |
| `ATTACH(A)` | `NEW` | `SHOULD_HAVE_BEEN_NEW` |
| `NEW` | `ATTACH(B)` | `WRONG_NEW_FALSE_SPLIT` |
| `NEW` | `NEW` | `CORRECT_NEW` |

同一实例若已被历史错误拆成多个干净节点，允许多个合法目标。人工只标合法目标集合；重放时使用哪个节点由冻结策略决定。

## 2. 由表及里的标注顺序

1. **证据绑定**：事件 UID、原 observation、`t^-` object version、相似度矩阵和资产哈希必须完整。
2. **观测有效性**：先判断 mask 是否代表一个可持续追踪的物理实例。
3. **实例身份**：隐藏 mapper 动作、目标、分数和自动 GT，判断展示候选中哪些属于同一物体。
4. **动作真值**：揭示原始 `ATTACH/NEW` 后，由程序根据身份集合推导五类路由标签。
5. **因果位置**：目标预污染、root 和 cascade 单独记录，不能覆盖路由事实。
6. **下游表现**：假合并、假分裂、离散类别错误、CLIP 语义错误和级联在 episode 复核中多选。
7. **修复结果**：B1/B2/B3 与语义 2×2 由重放程序计算，不让人工猜测。

本轮试标只完成前四层，并记录目标预状态供后续 episode compiler 使用。

## 3. 在线捕获、延迟裁决

- mapper 必须从 frame 0 空图严格在线运行。
- 每个 ATTACH 和 NEW 事件发生时冻结 `t^-` 节点 UID/version、完整候选顺序、矩阵和 observation 特征。
- 建图时可以同步处理已经闭合的事件包；证据不足时保留 `PENDING/DEFERRED`，不能猜。
- 后续视角可以离线帮助确认现实身份，但不能进入事件时 mapper，也不能改变 `t^-` 候选集合。
- 人工标签不反馈当前 canonical map。

统一记录 `s≤d≤h≤c`。实验 0 不调用 VLM，所以 `h/c=null`；同时保存事件时和提交时 mapper 最新帧。

## 4. 两阶段页面

### 阶段 A：mapper 盲标

展示当前 RGB/mask、候选历史和三视图 3D，不展示原始动作、目标、UID、排名、分数、GT 或抽样组。依次填写：

1. observation quality；
2. 同一物理实例候选，可多选；
3. 身份证据状态；
4. 简短实例描述。

盲标提交前，页面会再次显示 quality、候选集合和证据状态。确认后立即锁定，不能在看到 mapper 动作后回改；若确有误操作，只保留原答案并另写裁决覆盖层。

身份证据状态只有：

- `SUFFICIENT_FOR_IDENTITY`
- `PARTIAL`
- `INSUFFICIENT`

`UNCERTAIN` 不能与 `SUFFICIENT_FOR_IDENTITY` 同时出现。

### 阶段 B：揭示原始动作

- 原动作是 ATTACH 时，判断原目标在事件前为干净、已污染或不确定。
- 原动作是 NEW 时，目标前状态固定为 `NOT_APPLICABLE`。
- 已在页面选出匹配节点时，完整地图状态选 `NOT_NEEDED_MATCH_SHOWN`。
- 页面没有匹配节点时，只能在完成事件时地图检查后选择 `MATCH_EXISTS_OUTSIDE` 或 `NO_MATCHING_NODE_EXISTS`；没有检查就选 `UNCHECKED`。
- `MATCH_EXISTS_OUTSIDE` 必须绑定事件时节点 UID，不能只写“应该有”。

人工不直接输入五类路由标签。

## 5. observation quality

必须按以下三问顺序判断：

1. mask 是否同时包含两个或更多可分物体？是则选 `MIXED_MULTIPLE_INSTANCES`。
2. 若现实中仍是一个完整物理实例，即使画面只看到一部分、被遮挡、出画或被其他物体截断，仍应在 `CLEAN_SINGLE_INSTANCE` 与 `BORDERLINE_SINGLE_INSTANCE` 中选择。
3. 只有无法稳定决定“该区域本身是独立物体，还是另一物体的一部分”时，才选 `GRANULARITY_AMBIGUOUS`。选择它必须写明具体 part-whole 边界。

因此，**局部可见不等于粒度歧义**。同类别、同材质、属于同一窗/家具系统或彼此相邻，也都不等于同一实例；身份必须结合位置、形状、多视角历史和场景上下文。

- `CLEAN_SINGLE_INSTANCE`：mask 只表示一个持续物理实例，边界足以支持身份判断；允许只看到该实例的一部分。
- `BORDERLINE_SINGLE_INSTANCE`：仍能确认是一个物理实例，但边界泄漏、遮挡或有效点不足明显影响稳定性；仅进敏感性集合。不能仅因“只看到一部分”就自动选它。
- `MIXED_MULTIPLE_INSTANCES`：两个或更多可分物理实例共同构成一个 mask；接触、承托或相邻不把它们变成同一实例。
- `BACKGROUND_OR_FRAGMENT`：墙、地面、天花板、薄片或不存在的实体。
- `DUPLICATE_PROPOSAL_SAME_FRAME`：同帧重复 proposal。
- `DYNAMIC_POSE_DEPTH_ERROR`：动态、位姿或深度问题使身份比较不成立。
- `GRANULARITY_AMBIGUOUS`：part-whole 粒度无法稳定决定，例如不能判断是独立坐垫还是沙发本体的一部分；不是“完整实例只露出局部”的代称。
- `INSUFFICIENT`：图片、3D 或历史证据缺失。

后六类优先作为非路由问题排除，不能伪装成 NEW。

## 6. 自动 GT 的严格边界

只允许使用由当前 depth/pose 投影到 ReplicaSSG 标注网格、3 cm 最近邻门限生成的校正 sidecar。旧 Habitat sidecar 禁止使用。

校正 GT 可以私下用于：

- 检查完整 `t^-` 地图中是否存在同实例节点；
- 让小型校准覆盖五个路由单元；
- 抽查人工标签和自动谱系。

GT 分层、实例 ID、类别和“预期答案”不能在盲标页面显示。GT 只作为高精度选例和审核提示，最终标签仍需人工确认；冲突进入 adjudication overlay。

## 7. root/cascade 与表现层

五类路由标签描述单次动作事实，不等于 root/cascade：

- 若目标在本事件前已污染，仍保留本事件的路由标签，同时标记需要因果复核。
- episode compiler 按时间和 lineage 找到首次污染或首次错误 NEW，后续相关事件标为 cascade candidate。
- `false_merge`、`false_split`、`discrete_label_error`、`clip_semantic_error`、`downstream_cascade` 是可多选表现，不取代根路由类型。

## 8. v1 数据处置

- 原 24 条标签永久只读，保留其原始哈希和时间。
- 旧自动 GT 分层、旧派生错误标签和 `1/11` 比例不得进入 v2 统计。
- `calibration_016/024` 保留为身份分歧，另写仲裁覆盖层。
- `calibration_020` 在覆盖层修正为 `BACKGROUND_OR_FRAGMENT`，不覆盖原答案。
- v1 与 v2 的错误率和一致率禁止混算。

## 9. 当前试标集的解释

R2 schema calibration 为五个路由单元各选 2 条，并加入 1 条清晰 mixed、1 条清晰 background 和 3 条隐藏重复；12 个基础案例中 8 个专门覆盖单实例局部可见。所有基础事件均未在 R1 或旧 v1 页面出现。某一类若只有同一 episode 的少量事件，manifest 必须明确独立 episode 数，不得把 cascade 当成新的 root 证据。

`WRONG_ATTACH_EXISTING` 的两条基础事件是 room0 中仅有的 fresh 候选，来自同一 blanket 因果组的相邻帧，只用于检验该路由单元能否被正确表达，不能当作两份独立 root 证据。隐藏重复只评估标注一致性，也不能作为独立样本计数。

这是刻意平衡的工具校准集，只验证：定义是否清楚、证据是否充分、五类动作能否稳定区分。它不能估计自然错误发生率。
