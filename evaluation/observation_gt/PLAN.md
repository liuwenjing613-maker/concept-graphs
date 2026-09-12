# Observation–GT 关联与 VLM 决策评测设计

## 1. 文档状态

- 目标代码库：`/root/autodl-tmp/concept-graphs_v7_CLIP`
- 计划实现目录：`evaluation/observation_gt/`
- 文档性质：实现前设计方案
- 评测类型：离线、事件级、基于已有建图 evidence
- 与现有评测的关系：作为 `evaluation/unified` 最终地图评测的补充，不替代 v2 指标

## 2. 背景与目标

现有 `unified_fullmesh_v2_20260910` 评测最终建图结果，可衡量最终地图的实例分割、语义识别和表面覆盖质量，但无法回答以下问题：

1. 某一次 observation 本应关联到哪个真实 GT 实例；
2. baseline 关联决策是否正确；
3. VLM 是否修正了错误融合；
4. VLM 是否反而引入了新的错误融合、错误丢弃或重复实例；
5. 当同一个 GT 实例被拆成多个预测实例时，关联到较小碎片应该如何评价。

本模块将每个 observation 的 processed mask 结合深度、内参和相机位姿反投影到三维，并与固定 GT mesh 匹配，得到 observation 的 GT 身份。随后，利用关联事件发生前的历史 observation 成员关系，为已有预测实例建立因果、无未来信息泄漏的 GT 身份，最终分别评价 baseline 与 VLM 决策。

## 3. 设计原则

### 3.1 因果性

评价某个关联事件时，只允许使用该事件发生前已经出现的数据。不得使用最终地图的 best-IoU 匹配反推早期预测实例身份，否则后续的合并、删除和纠错结果会泄漏到早期决策评价中。

### 3.2 身份正确与碎片归并分开评价

同一 GT 实例可能对应多个预测实例。当前 observation 被关联到同 GT 的非主预测实例时：

- 身份关联正确；
- 不属于跨 GT 错误融合；
- 但没有完成向主预测实例的归并，仍存在碎片化。

因此，不能仅用“是否进入最大预测实例”作为唯一正确性标准。

### 3.3 不强迫低质量掩码获得唯一身份

主指标固定以全部 VLM gate events 为分母：CLEAN 身份动作正确或 MIXED 被正确 `DISCARD` 统一记为 `SUCCESS`；明确错误记为 `FAILURE`；身份不足以判定和 GT/深度不足分别保留为 `AMBIGUOUS`、`UNSCORABLE`，不得从主分母删除。CLEAN 身份与 MIXED discard 另以各自固定 GT cohort 给出专项诊断。

### 3.4 使用 Replica 官方实例 GT

reference 直接由 Replica v1 官方 `habitat/mesh_semantic.ply` 和 `habitat/info_semantic.json` 生成，保留 `info_semantic.json.objects` 中声明的原生 object ID 与 class name。mesh face 使用但 JSON 未声明的 object ID（包括非负 ID）明确记为 unlabeled，不猜测其类别。旧 `evaluation/unified` reference 仅作为向后兼容输入，不再作为正式 observation-GT 结果的数据源。观察级评测的协议、输入哈希和阈值必须写入输出，确保可复现。

## 4. 输入

### 4.1 运行 evidence

从一次完整建图运行目录中读取：

- `config_params.json`；
- `evidence/frames.jsonl`；
- `evidence/observations.jsonl`；
- processed mask 文件；
- RGB 和 depth 文件；
- association 事件；
- blocking association gate / VLM gate 事件；
- 事件前对象 UID 和 observation 成员关系。

帧、深度、位姿、内参和 processed mask 的加载优先复用：

```text
conceptgraph/revision/counterfactual_projection.py
ProjectionEvidenceLoader
```

### 4.2 GT reference

从 `reference_root/<scene>/` 读取：

```text
reference.npz
manifest.json
```

需要的数组包括：

```text
xyz       GT mesh 顶点坐标
semantic  GT 语义标签
instance  GT 实例标签
```

### 4.3 不设置忽略类别

所有 detection class 都进入同一分类流程。只要官方 reference 为区域提供 object ID，`wall`、`ceiling`、`floor` 和 `other` 都作为正常实例参与 `CLEAN/MIXED` 判断；仅没有官方 object 映射的区域记为 unlabeled，并按证据充分程度归入 `MIXED` 或 `UNSCORABLE`。

## 5. 阶段一：Observation 掩码反投影

### 5.1 有效像素

对 processed mask 中每个像素，只有同时满足以下条件才进入反投影：

1. mask 值为真；
2. 深度值有限且大于 0；
3. 深度处于配置允许范围内；
4. 像素坐标位于深度图范围内。

### 5.2 相机坐标与世界坐标

利用相机内参将像素 `(u, v, d)` 反投影为相机坐标点，再根据当前代码库规定的 pose convention 转换到世界坐标。

必须增加坐标系诊断：输出每个场景反投影点到 GT mesh 的最近距离分布，包括中位数、P90、P95 和 5 cm 内比例。如果绝大多数点无法落在 GT mesh 附近，应立即停止评测并提示检查：

- pose 是 camera-to-world 还是 world-to-camera；
- 深度单位和缩放系数；
- GT mesh 与数据集轨迹是否处于相同坐标系。

### 5.3 与 GT mesh 匹配

为 GT `xyz` 建立 `scipy.spatial.cKDTree`。每个反投影点查询最近 GT 顶点，距离严格小于配置阈值时接受匹配，默认：

```text
gt_match_distance_m = 0.05
```

匹配后得到该点的 GT semantic 和 GT instance。

## 6. 阶段二：Observation 掩码归类

### 6.1 统计量

对每个 observation 记录：

```text
obs_uid
frame_uid
class_name
mask_area
valid_depth_pixels
valid_depth_ratio
matched_gt_pixels
gt_support_ratio
instance_support_ratio
unlabeled_gt_ratio
top1_gt_instance
top1_gt_class
top1_purity
top2_gt_instance
top2_purity
purity_margin
median_gt_distance_m
p95_gt_distance_m
core_top1_gt_instance
core_top1_purity
core_agrees_with_full_mask
quality_status
quality_reason
```

若距离阈值内全部 GT 点数为 `N`，其中属于有效实例 `g` 的数量为 `n_g`，则：

```text
top1_purity = max_g(n_g) / N
top2_purity = second_max_g(n_g) / N
purity_margin = top1_purity - top2_purity
```

unlabeled 点包含在 `N` 中，因此实例边界或未知区域污染会降低 purity。

### 6.2 边界鲁棒性

对原始 mask 进行可配置的二值腐蚀，默认半径 2 像素，得到核心区域 mask。原 mask 和核心 mask 应独立计算 dominant GT。只有二者 dominant GT 一致时，才允许将 observation 标记为高置信 `CLEAN`。

如果腐蚀后核心区域为空，不直接判错，而是记录原因，并根据其余条件归为 `MIXED` 或 `UNSCORABLE`。

### 6.3 三类输出

#### CLEAN

具有充分深度和 GT 支持，且能稳定指向唯一物体 GT 实例。用于身份关联主指标。

#### MIXED

具有充分 GT 支持，但明显覆盖多个物体 GT 实例，或者原 mask 与核心 mask 的 dominant GT 不一致。用于评价 VLM 是否正确选择 `DISCARD`，不强制赋予唯一身份。

#### UNSCORABLE

深度、GT 支持、位姿或有效点数量不足，无法可靠评价。必须保存排除原因。

### 6.4 初始阈值

初始试跑使用：

```text
min_valid_depth_ratio = 0.70
min_gt_support_ratio  = 0.80
min_matched_gt_pixels = 50
min_top1_purity       = 0.80
max_top2_purity       = 0.15
min_purity_margin     = 0.60
mask_erosion_radius   = 2
gt_match_distance_m   = 0.05
```

这些是开发阈值，不直接视为最终论文阈值。正式评测前应在开发场景上人工检查分层样本，然后冻结阈值，并执行：

```text
top1 purity：0.70 / 0.80 / 0.90
最少 GT 点：25 / 50 / 100
距离阈值：0.03 / 0.05 / 0.07 m
```

的敏感性分析。

## 7. 阶段三：事件前预测实例的 GT 身份

### 7.1 为什么需要这一阶段

阶段一和阶段二只回答：

> 当前 observation 对应哪个 GT 实例？

但关联事件提供的是若干预测实例候选。为了判断系统选得是否正确，还需要回答：

> 事件发生前，每个候选预测实例分别对应哪个 GT 实例？

例如：

```text
当前 observation → GT sofa_1

预测实例 A 的历史 observations → sofa_1, sofa_1, sofa_1
预测实例 B 的历史 observations → sofa_1
预测实例 C 的历史 observations → chair_2, chair_2
```

由此可知：

```text
A → sofa_1 的主预测实例
B → sofa_1 的较小碎片
C → chair_2
```

选择 A 是正确主实例关联，选择 B 是身份正确但碎片未消除，选择 C 才是错误融合。

### 7.2 因果成员集合

对关联事件 `e`，读取 `object_uids_before` 以及与其按位置对齐的 `candidate_object_version_uids`，再从 `evidence/object_versions.jsonl` 取得每个候选在该快照版本中的 `member_observation_uids`。这是事件记录直接冻结的成员快照，比从最终地图或仅按事件重放推测成员更可靠。只能使用当前候选版本中已经归属该对象的历史 observation。

不得使用：

- 当前 observation 自身；
- 当前事件之后才加入的 observation；
- 最终地图的成员关系；
- 最终地图 best-IoU 匹配。

### 7.3 GT 身份投票

只使用历史成员中已归类为 `CLEAN` 的 observation。对预测实例 `m` 和 GT 实例 `g` 定义：

```text
weight(o) = top1_purity(o) * gt_support_ratio(o)
score(m,g) = sum_o weight(o) * I[gt(o) = g]
```

预测实例身份：

```text
identity(m) = argmax_g score(m,g)
identity_purity(m) = max_g score(m,g) / sum_g score(m,g)
```

不使用 mask 像素数作为主要权重，避免近距离大掩码完全压过多个独立历史观察。像素支持量仅作为诊断字段和完全相同时的确定性 tie-break。

### 7.4 预测实例身份状态

```text
RELIABLE:
    至少存在一个 CLEAN 历史 observation
    且 identity_purity >= 0.80

MIXED_PREDICTION:
    存在 CLEAN 历史 observation
    但 identity_purity < 0.80

UNKNOWN_PREDICTION:
    没有可用的 CLEAN 历史 observation
```

只有一个 CLEAN 历史 observation 时允许得到身份，以覆盖建图早期事件，但记录：

```text
identity_support_count = 1
low_support_identity = true
```

并在结果中额外报告排除单观察支持后的敏感性结果。

### 7.5 同一 GT 的主预测实例

若同一 GT `g` 对应多个可靠预测实例，则定义事件时刻的主预测实例：

```text
canonical_prediction(g) = argmax_m score(m,g)
```

确定性 tie-break 顺序为：

1. `score(m,g)` 更高；
2. CLEAN 历史 observation 数量更多；
3. 对 GT `g` 的累计有效匹配点更多；
4. object UID 字典序更小。

这里的“主预测实例”是当前事件前历史证据最强的实例，不是最终地图中 IoU 最大的实例。

## 8. 阶段四：关联动作分类

将 baseline 和 VLM 最终动作统一归一化为：

```text
ATTACH(object_uid)
NEW
DISCARD
UNRESOLVED
```

对 `CLEAN` observation：

| 动作结果 | 分类 | 身份是否正确 | 是否完成主实例归并 |
|---|---|---:|---:|
| ATTACH 到同 GT 主预测实例 | `CORRECT_CANONICAL` | 是 | 是 |
| ATTACH 到同 GT 非主预测实例 | `CORRECT_FRAGMENT` | 是 | 否 |
| ATTACH 到其他 GT | `WRONG_FUSION` | 否 | 否 |
| ATTACH 到 MIXED/UNKNOWN 候选 | `AMBIGUOUS_TARGET` | 不确定 | 不确定 |
| 存在可靠同 GT 实例但选择 NEW | `DUPLICATE_CREATION` | 未错融 | 否 |
| 不存在可靠同 GT 实例，且展示候选含 MIXED/UNKNOWN 实例时选择 NEW | `AMBIGUOUS_NEW` | 不确定 | 不确定 |
| 不存在可靠同 GT 实例，且全部展示候选均可靠属于其他 GT 时选择 NEW | `CORRECT_NEW` | 是 | 是 |
| 选择 DISCARD | `FALSE_DISCARD` | 否 | 否 |

若 V7 冻结的高层决策为 `MERGE_REVIEW`，上表中的执行器 `DISCARD` 只是等待候选对
合并证书时的保守回退，不代表 VLM 认为当前 observation 应被丢弃。此时 CLEAN 事件
根据 `same_aliases` 的事件前 GT 身份改记为 `CORRECT_MERGE_REVIEW`、
`WRONG_MERGE_REVIEW` 或 `AMBIGUOUS_MERGE_REVIEW`；实际回退动作和后续候选对
投票/执行状态作为并行诊断保留，不与 VLM 高层判断混为一个标签。

`CLEAN + NEW` 的歧义范围只覆盖本次 gate 实际展示给决策器的候选。地图中未进入
候选集的 MIXED/UNKNOWN 实例不构成“当前 NEW 动作存在可选旧目标”的证据，只作为
全局身份质量诊断保留。对于 `DUPLICATE_CREATION`，额外记录可靠同 GT 实例位于
展示候选内还是候选外，以区分选择错误与候选召回错误。

对 `MIXED` observation：

| 动作结果 | 分类 |
|---|---|
| DISCARD | `CORRECT_DISCARD` |
| ATTACH | `MIXED_MASK_ATTACHMENT` |
| NEW | `MIXED_MASK_NEW` |

`UNSCORABLE` 进入统一的全部事件主分母，并作为独立 outcome 保留；它既不强行记为成功，也不强行记为失败。

## 9. 阶段五：Baseline 与 VLM 转换评价

从 gate event 中解析：

```text
current_observation_uid
baseline_match_index
final_match_index
candidate_alias_to_object_index
changed
decision_source
```

先将候选索引解析为稳定 object UID，再分别对 baseline 和 final 动作执行第 8 节分类，形成转换矩阵。

重点转换包括：

```text
WRONG_FUSION      → CORRECT_CANONICAL   修正错误融合并归入主实例
WRONG_FUSION      → CORRECT_FRAGMENT    修正跨GT身份，但仍有碎片
CORRECT_FRAGMENT  → CORRECT_CANONICAL   改善碎片归并
CORRECT_CANONICAL → WRONG_FUSION        VLM引入错误融合
CORRECT_*         → FALSE_DISCARD       VLM错误丢弃干净观察
DUPLICATE_CREATION→ CORRECT_*           VLM避免重复实例
```

结果必须按 `decision_source` 分组，至少区分：

- VLM 直接决策；
- geometry containment 覆盖；
- 规则回退；
- 完整系统最终决策。

这样可以避免将几何规则的收益误归因于 VLM。

## 10. 指标

### 10.1 Observation 可评测性

```text
Scorable Observation Rate
CLEAN Rate
MIXED Rate
UNSCORABLE Rate
Unlabeled GT Point Rate
```

### 10.2 统一全事件主指标

```text
Overall Event Success Rate
= (CLEAN身份正确 + MIXED正确DISCARD)
  / 全部VLM gate events
```

同一固定分母同时报告 `SUCCESS / FAILURE / AMBIGUOUS / UNSCORABLE / UNRESOLVED` 的数量和比例。baseline 与 final 的事件集合必须完全一致。

### 10.3 CLEAN 身份专项指标

```text
Identity Success Rate
= (CORRECT_CANONICAL + CORRECT_FRAGMENT + CORRECT_NEW)
  / 全部CLEAN gate events

Canonical Success Rate
= (CORRECT_CANONICAL + CORRECT_NEW)
  / 全部CLEAN gate events
```

`AMBIGUOUS_TARGET/NEW` 留在 CLEAN 固定分母中单列，避免通过改变可判定覆盖率制造分母漂移。另保留 resolved-only accuracy 作为诊断，不作为主结论。

### 10.4 MIXED 丢弃与 DISCARD 二分类

```text
TP = MIXED + DISCARD
FP = CLEAN + DISCARD
FN = MIXED + ATTACH/NEW
TN = CLEAN + 非DISCARD
```

报告 precision、recall、F1、scorable binary accuracy、CLEAN false-discard rate，并继续报告 wrong fusion、fragment attachment、duplicate creation 等条件诊断。

### 10.5 VLM 增益指标

```text
VLM Correction Count
VLM Harm Count
VLM Correction Rate
VLM Harm Rate
```

修正定义为 `非SUCCESS → SUCCESS`，伤害定义为 `SUCCESS → 非SUCCESS`。净收益使用与主指标相同的全部事件固定分母：

```text
Net VLM Gain = (VLM Correction Count - VLM Harm Count)
               / 全部VLM gate events
```

同时报告 `AMBIGUOUS/UNSCORABLE → FAILURE` 等 `new_failure_count`，避免修正/伤害二元统计隐藏新出现的明确错误。

此外分别报告：

- 跨 GT 身份修正数量；
- 跨 GT 错误引入数量；
- 碎片到主实例的改善数量；
- 主实例退化到碎片的数量；
- 对 CLEAN/MIXED mask 的 DISCARD 行为。

## 11. 输出文件

每次评测写入新的输出目录，不覆盖已有结果：

```text
observation_gt_metrics/
├── protocol.json
├── observation_labels.jsonl
├── prediction_identities.jsonl
├── event_decisions.jsonl
├── metrics.json
├── transition_matrix.csv
├── threshold_sensitivity.json
└── diagnostics/
    ├── clean_examples/
    ├── mixed_examples/
    └── unscorable_examples/
```

`protocol.json` 至少保存：

- 协议版本；
- evaluator 代码哈希；
- GT reference 和 manifest 哈希；
- evidence 输入哈希；
- 所有阈值；
- 原生 instance ID 到 semantic ID/class name 的映射；
- unlabeled 区域策略；
- pose convention；
- 深度缩放；
- GT 匹配距离；
- 场景及运行目录。

## 12. 计划代码结构

```text
evaluation/observation_gt/
├── __init__.py
├── evaluate_observation_gt.py
├── generate_gate_review.py
├── observation_labeler.py
├── instance_identity.py
├── association_metrics.py
├── build_replica_official_reference.py
├── config.example.json
├── README.md
├── test_build_replica_official_reference.py
├── test_observation_labeler.py
└── test_association_metrics.py
```

职责如下：

- `evaluate_observation_gt.py`：命令行入口、输入发现、流程调度、输出及指纹；
- `generate_gate_review.py`：第三个独立 CLI 入口；读取冻结路径一结果生成静态人工复核网页，不参与指标计算；
- `observation_labeler.py`：反投影、GT 最近邻匹配、mask 核心腐蚀和三分类；
- `build_replica_official_reference.py`：从 Replica 官方语义 mesh 与实例元数据生成 reference；
- `instance_identity.py`：重建事件前对象成员、GT 投票和主预测实例选择；
- `association_metrics.py`：动作归一化、结果分类、转换矩阵和指标；
- `config.example.json`：阈值和诊断设置；
- `README.md`：运行命令、输入结构、协议和结果解释；
- 测试文件：覆盖边界、错误融合、同 GT 碎片、NEW 和 DISCARD。

## 13. 单元测试与集成测试

至少覆盖：

1. 单一 GT 纯掩码归为 `CLEAN`；
2. 两个 GT 接近各半归为 `MIXED`；
3. 深度不足归为 `UNSCORABLE`；
4. `wall/ceiling/floor/other` 具有官方 object ID 时可归为 `CLEAN/MIXED`；
5. 5 cm 严格距离边界；
6. 原 mask 与核心 mask 不一致时不能归为 `CLEAN`；
7. 位姿方向错误可由距离诊断发现；
8. 同 GT 非主实例归为 `CORRECT_FRAGMENT`；
9. 不同 GT 目标归为 `WRONG_FUSION`；
10. 已有同 GT 实例时 NEW 归为 `DUPLICATE_CREATION`；
11. 没有同 GT 实例时 NEW 归为 `CORRECT_NEW`；
12. CLEAN observation 的 DISCARD 归为 `FALSE_DISCARD`；
13. MIXED observation 的 DISCARD 归为 `CORRECT_DISCARD`；
14. baseline 错、VLM 对计为 correction；
15. baseline 对、VLM 错计为 harm；
16. 当前事件及未来 observations 不得参与候选实例投票；
17. 相同输入和配置重复运行产生相同结果；
18. 输入哈希改变时禁止静默复用旧结果。

## 14. 实施顺序

### 第一步：输入与协议校验

实现配置解析、reference/evidence 哈希、场景识别、pose/depth 诊断和输出目录保护。

### 第二步：Observation GT 标注

实现 mask 反投影、GT 最近邻匹配、三分类和 `observation_labels.jsonl`。

### 第三步：room0 人工抽查

从 CLEAN、MIXED、UNSCORABLE 中分层抽样，生成 RGB mask overlay 和 GT 统计，人工检查坐标系及阈值。

### 第四步：事件前预测实例身份

使用 `object_uids_before`、`candidate_object_version_uids` 与 `object_versions.jsonl` 恢复精确的事件前候选成员快照，计算 GT 投票、实例身份纯度和同 GT 主实例。`mapping_events.jsonl` 仅用于完整性审计，不作为候选快照的替代来源。

### 第五步：Baseline/VLM 决策打分

解析 baseline 与 final 动作，生成逐事件分类、转换矩阵和 VLM correction/harm 统计。

### 第六步：测试与敏感性分析

运行单元测试、room0 完整试跑及阈值敏感性分析，确认规则稳定后冻结正式协议版本。

### 第七步：批量场景评测

在 room0–2、office0–4 或论文实际采用的场景集合上运行，生成 pooled 指标、逐场景指标和论文表格输入。

## 15. 验收标准

实现完成需满足：

1. 能直接读取已有 v7_CLIP 运行 evidence，无须重新调用 VLM；
2. 不修改在线建图结果和原始 evidence；
3. observation 分类结果可通过可视化样本人工核验；
4. 所有事件判断只使用事件前信息；
5. 同 GT 非主实例不会被误记为跨 GT 错误融合；
6. `other/wall/ceiling/floor` 的官方实例正常参与评测，不存在隐藏分支；
7. baseline 与 VLM final 使用完全相同的 GT 标签和候选身份口径；
8. 能分别报告 VLM 修正、VLM 伤害和几何规则覆盖的结果；
9. 输出包含完整协议、输入哈希和排除原因；
10. 单元测试全部通过，重复运行结果确定一致。

## 16. 与论文实验部分的衔接

最终论文应将本模块定义为 observation-level causal association evaluation，并与最终地图指标并列：

- 最终地图指标回答 VLM 是否改善最终 3D scene graph；
- observation 级 IAA/Wrong Fusion Rate 回答 VLM 是否避免跨实例错误融合；
- CAA/Fragment Attachment Rate 回答 VLM 是否改善同一 GT 的碎片归并；
- transition matrix 和 Net VLM Gain 回答 VLM 修正了多少错误、又引入了多少新错误。

主文中应报告固定阈值、可评测 observation 比例和 VLM/几何规则分层结果；详细阈值敏感性、排除原因分布和可视化案例可放入附录。
