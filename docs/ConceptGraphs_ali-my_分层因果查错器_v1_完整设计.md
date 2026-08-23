# ConceptGraphs `ali-my` 分层因果查错器 v1：完整设计与实现规格

> **文档目标**：基于当前 `ali-my` 分支的在线建图逻辑与统一证据留存，设计一套能够真正落地的、分阶段、分确定性、可追溯的查错器。  
> **当前范围**：只负责发现、定位、分类和解释错误，不执行自动回滚、删除、合并或重建。  
> **后续接口**：高确定性错误可进入规则修复候选；存在现实歧义的错误生成多模态证据包，再交由 VLM 或人工裁决。

---

## 0. 文档依据与当前代码基线

本设计以当前仓库中的 `ali-my` 分支为准，重点依据以下代码：

```text
conceptgraph/slam/rerun_realtime_mapping.py
conceptgraph/slam/mapping.py
conceptgraph/slam/utils.py
conceptgraph/utils/evidence.py
conceptgraph/utils/ious.py
conceptgraph/hydra_configs/base_mapping.yaml
conceptgraph/hydra_configs/rerun_realtime_mapping.yaml
tests/test_evidence.py
docs/ALI_MY_EVIDENCE.md
```

当前已知分支提交：

```text
branch: ali-my
commit: fda44745c846b00a2e030c58ea990834c1cd1401
```

当前在线建图主流程为：

```text
RGB-D Frame
    ↓
YOLO-World 产生 bbox / class / confidence
    ↓
SAM 根据 bbox 产生 mask
    ↓
resize_gobs
    ↓
filter_gobs
    ├── 小 mask 过滤
    ├── 背景类过滤
    ├── 超大 bbox 过滤
    └── 低 confidence 过滤
    ↓
mask_subtract_contained
    ↓
Depth + mask 反投影为 observation point cloud
    ↓
初始化点云处理、bbox 计算
    ↓
构造 detection_list
    ↓
计算 spatial similarity
    ↓
计算 CLIP visual similarity
    ↓
aggregate_similarities(sim_sum)
    ↓
每个 observation 独立执行 greedy argmax
    ├── max_score > threshold：关联到已有 object
    └── max_score <= threshold：创建新 object
    ↓
merge_obj_matches
    ↓
在线 relation 更新
    ↓
周期性 denoise / filter / merge
    ↓
最终 object nodes、captions、relations
```

当前 unified evidence sidecar 已保存：

```text
manifest.json
frames.jsonl
observations.jsonl
similarities/frame_*.npz
associations.jsonl
observation_pcd/*.npz
mapping_events.jsonl
vlm_events.jsonl
final_membership.json
evidence_summary.json
```

这些证据已经足以构建第一版查错器，但在进入严格判断前仍需补齐若干关键字段，见第 5 节。

---

# 第一部分：总体设计原则

## 1. 查错器真正要回答什么

查错器不能仅凭一个阈值就宣称：

> “现实世界一定不是这样。”

它真正能够严谨判断的是：

> **当前地图假设是否与程序定义、历史观测、几何证据、语义证据、时序证据和关系约束发生冲突。**

因此，必须把三件事情分开：

### 1.1 程序或账本是否自洽

例如：

```text
矩阵 argmax 是 obj_17
日志 target 却写成 obj_31
```

这是当前算法下的确定性错误，不需要 VLM。

### 1.2 地图内容是否存在强冲突

例如：

```text
一个 observation 与目标节点的历史成员在语义上和几何上都严重冲突
```

这很可能是错误关联，但仍应检查遮挡、新视角、part-whole 等合理解释。

### 1.3 现实世界究竟是什么

例如：

```text
两个节点是重复 sofa，还是两张外观相同且紧挨着的 sofa？
```

内部指标可能不足以最终判断，需要：

```text
多视角图片
3D 对比
时序共现
VLM
人工
GT
```

---

## 2. 三维设计框架

本查错器采用：

\[
\boxed{
\text{按错误产生阶段分类}
\times
\text{按判定确定性分级}
\times
\text{跨阶段因果归因}
}
\]

### 2.1 按错误产生阶段分类

```text
System / Evidence
Detection
Segmentation
Projection / Geometry
Association
Fusion / Postprocess
Object Identity
Caption
Relation
Temporal Change（预留）
```

### 2.2 按判定确定性分级

```text
CONFIRMED_SYSTEM_ERROR
LIKELY_MAPPING_CONFLICT
AMBIGUOUS_MAPPING_RISK
INSUFFICIENT_EVIDENCE
NO_CONFLICT_FOUND
```

### 2.3 跨阶段因果归因

一个最终 duplicate node 可能由不同原因造成：

```text
重复 detection proposal
→ 同帧产生两个 observation
→ 阈值略高，均创建新节点
→ 后续未合并
→ 最终 duplicate object
```

也可能是：

```text
同一物体在不同视角几何 overlap 太低
→ association 失败
→ 新建节点
→ 最终 duplicate object
```

查错器必须同时输出：

```text
症状是什么
最早可见异常在哪里
可能根因是什么
还有哪些合理解释未排除
缺少哪些证据
```

---

## 3. 统一判定等级

### 3.1 `CONFIRMED_SYSTEM_ERROR`

含义：

> 在当前代码、配置和 policy 下，该状态不可能合法出现。

典型例子：

```text
保存的 margin 与 similarity matrix 重算不一致
CREATE 决策违反严格阈值条件
已 inactive 的 merge source 再次被消费
同一 event_uid 对应两种不同事件
edge 指向不存在的 object UID
```

处理：

```text
无需 VLM
阻止该 run 被视为有效实验
修代码或修证据链
```

---

### 3.2 `LIKELY_MAPPING_CONFLICT`

含义：

> 至少两个相互独立的证据家族支持错误，且没有强 veto。

例如：

```text
semantic outlier 很高
AND geometric support 很低
AND 历史 association margin 很小
AND 不存在新视角 / part-whole 解释
```

处理：

```text
优先进入人工核验
未来可进入 sandbox 修复候选
当前 v1 不直接修改地图
```

---

### 3.3 `AMBIGUOUS_MAPPING_RISK`

含义：

> 出现异常，但存在合理现实解释。

例如：

```text
两个 object 的 CLIP 很相似且位置很近
```

它们可能是：

```text
重复节点
两把同款椅子
整体与部件
容器与内部物体
```

处理：

```text
生成多模态 Evidence Packet
送入 VLM 或人工
```

---

### 3.4 `INSUFFICIENT_EVIDENCE`

含义：

> 当前统一证据不足以完成该规则判断。

例如：

```text
没有 processed mask
没有 target-before object state
没有完整 VLM 输入图像
```

处理：

```text
明确记录缺少什么
绝不能把 UNKNOWN 当成 PASS
```

---

### 3.5 `NO_CONFLICT_FOUND`

含义：

> 在当前规则、证据和阈值下没有发现冲突。

它不等于：

```text
现实一定正确
```

只能解释为：

```text
本规则没有发现异常
```

---

## 4. 确定性和严重性必须分离

每条 finding 同时保存：

```text
certainty
severity
```

严重性建议：

```text
CRITICAL
HIGH
MEDIUM
LOW
```

例如：

```text
SYS-002 margin 记录错
certainty = CONFIRMED_SYSTEM_ERROR
severity = MEDIUM
```

而：

```text
一个疑似 false merge 污染 12 个成员和 8 条边
certainty = LIKELY_MAPPING_CONFLICT
severity = HIGH
```

---

# 第二部分：当前代码 policy 与合法行为

## 5. 必须写入 manifest 的 audit policy

查错器不能把当前算法假设永久硬编码，应在 `manifest.json` 中显式保存：

```yaml
audit_policy:
  observation_ownership: exclusive_single_target
  same_frame_many_to_one: allowed
  relation_cardinality: many_to_many

  environment_mode: static
  object_granularity: instance_with_part_whole_ambiguity

  association_rule:
    type: independent_greedy_argmax
    threshold_comparison: strict_greater_than
    max_score_equal_threshold: create_object

  postprocess_merge:
    source_single_consumption: true
    target_must_be_active: true
    source_must_be_active: true

  missing_evidence_policy: unknown_not_pass
```

### 5.1 当前 association policy 的准确含义

当前代码逐 observation 独立执行：

```python
if max_score <= sim_threshold:
    CREATE_OBJECT
else:
    MERGE_TO_ARGMAX_OBJECT
```

因此：

```text
obs_A → obj_17
obs_B → obj_17
```

合法。

但：

```text
同一个 obs_A
同时成为 obj_17 和 obj_31 的正式成员
```

在当前 exclusive policy 下不合法。

### 5.2 relation 与 membership 不同

以下关系完全可能：

```text
board --ON--> stool_A
board --ON--> stool_B
```

所以：

```text
一个 object 同时与多个 object 建立 ON / NEAR / TOUCHING 等关系
```

不能判错。

---

# 第三部分：当前统一证据需要补齐的最小内容

## 6. 补充原则

只补查错器真正需要的证据，不继续建设“日志博物馆”。

---

## 6.1 保存 processed mask

### 当前问题

实际建图使用：

```text
resize 后 mask
→ filter 后 mask
→ mask_subtract_contained 后 mask
```

但 `observations.jsonl` 中的 `mask_ref` 主要指向 detector cache 中的原始 mask。

因此：

```text
查错器看到的 mask
≠
真正用于生成 observation PCD 的 mask
```

### 必须新增

```json
{
  "raw_mask_ref": "...",
  "resized_mask_ref": "...",
  "processed_mask_ref": "evidence/processed_masks/<obs_uid>.npz",
  "raw_mask_area": 15322,
  "processed_mask_area": 12105,
  "mask_operations": [
    "resize_nearest",
    "filter_keep",
    "subtract_contained"
  ]
}
```

### 修改位置

在：

```text
rerun_realtime_mapping.py
gobs['mask'] = mask_subtract_contained(...)
```

之后保存 processed masks。

### 验收

```text
processed mask 重新反投影
→ 得到的 observation PCD 与实际使用结果一致
```

---

## 6.2 真实过滤轨迹

### 当前问题

`EvidenceRecorder._filter_reason()` 复制了一套 `filter_gobs()` 判断逻辑。

未来一旦主逻辑改了，日志可能继续使用旧逻辑，产生：

```text
程序实际因为 A 过滤
日志却声称因为 B 过滤
```

### 最优修改

让 `filter_gobs()` 原生返回：

```python
filtered_gobs, filter_trace
```

其中：

```json
{
  "obs_uid": "...",
  "decision": "REJECT",
  "rule_id": "FILTER_LOW_CONFIDENCE",
  "threshold": 0.25,
  "value": 0.18
}
```

不要让 recorder 再推断。

---

## 6.3 结构化 ArtifactRef

每个重型文件引用统一为：

```json
{
  "path": "evidence/processed_masks/obs_x.npz",
  "format": "npz",
  "key": "mask",
  "index": null,
  "shape": [480, 640],
  "dtype": "bool",
  "sha256": "..."
}
```

替代模糊字符串：

```text
mask.npz#arr_0[17]
```

ArtifactResolver 必须验证：

```text
文件存在
key 存在
index 合法
shape 一致
dtype 一致
hash 一致
```

---

## 6.4 Object Version Ledger

新增：

```text
object_versions.jsonl
```

每次 object 状态变化后记录：

```json
{
  "event_uid": "...",
  "object_uid": "obj_17",
  "version_before": 5,
  "version_after": 6,
  "trigger_type": "OBS_ASSOCIATE",
  "trigger_obs_uid": "obs_73",

  "before": {
    "member_observation_uids": [],
    "num_detections": 5,
    "n_points": 3210,
    "bbox_center": [],
    "bbox_extent": [],
    "clip_feature_ref": "..."
  },

  "after": {
    "member_observation_uids": [],
    "num_detections": 6,
    "n_points": 3528,
    "bbox_center": [],
    "bbox_extent": [],
    "clip_feature_ref": "..."
  },

  "delta": {
    "clip_cosine_change": 0.16,
    "center_shift_norm": 0.08,
    "extent_growth_ratio": 1.32,
    "point_count_ratio": 1.10
  }
}
```

它是：

```text
fusion shock
历史反事实
局部回放
```

的基础。

---

## 6.5 完整 merge transaction

当前 merge 事件需要保存：

```text
source_before
target_before
target_after
```

而不是只有：

```text
source_before
target_after
```

建议：

```json
{
  "event_type": "OBJECT_MERGE",
  "source_object_uid": "...",
  "target_object_uid": "...",

  "source_before": {},
  "target_before": {},
  "target_after": {},

  "candidate": {
    "overlap_forward": 0.91,
    "overlap_backward": 0.43,
    "visual_similarity": 0.87,
    "text_similarity": null,
    "text_similarity_source": "not_available",
    "decision": "ACCEPT"
  }
}
```

---

## 6.6 独立 text semantic evidence

当前 postprocess 中：

```python
text_sim = visual_sim
```

这不能作为两份独立证据。

查错器必须保存：

```json
{
  "visual_similarity": 0.88,
  "text_similarity": null,
  "text_similarity_source": "visual_proxy",
  "independent_semantic_evidence_count": 1
}
```

后续若加入真正 CLIP text embedding，再更新为独立证据。

---

## 6.7 深度与聚类统计

每个 observation 建议保存：

```json
{
  "valid_depth_ratio": 0.84,
  "depth_quantiles": {
    "q05": 1.12,
    "q25": 1.18,
    "q50": 1.24,
    "q75": 1.32,
    "q95": 2.61
  },
  "pre_dbscan": {
    "cluster_count": 3,
    "largest_cluster_ratio": 0.71,
    "second_cluster_ratio": 0.21,
    "largest_centers_distance": 0.84
  },
  "post_dbscan": {
    "n_points": 423
  }
}
```

这直接支持：

```text
背景泄漏
欠分割
多表面
深度异常
```

---

## 6.8 精确 VLM 输入记录

当前只保存 prompt fingerprint 和响应，不足以重建实际多模态输入。

需要保存：

```json
{
  "prompt_text_ref": "...",
  "prompt_sha256": "...",
  "input_image_refs": [],
  "input_image_sha256": [],
  "model_name": "...",
  "model_version": "...",
  "temperature": 0,
  "raw_response": "...",
  "parsed_output": {}
}
```

---

# 第四部分：查错器总体架构

## 7. 执行顺序

```text
Evidence Readiness Gate
          ↓
System Integrity Checker
          ↓
Detection Checker
          ↓
Segmentation Checker
          ↓
Projection / Geometry Checker
          ↓
Association Checker
          ↓
Fusion / Postprocess Checker
          ↓
Object Identity Checker
          ↓
Caption Checker
          ↓
Relation Checker
          ↓
Cross-stage Root Cause Resolver
          ↓
findings.jsonl
root_causes.jsonl
Evidence Packets
audit_summary.json
```

原则：

```text
所有 checker 只读 evidence
所有 checker 只输出 finding
任何 checker 不修改 map
```

---

## 8. 统一数据结构

### 8.1 `AuditContext`

```python
@dataclass
class AuditContext:
    run_id: str
    scene_id: str
    manifest: dict
    config: dict
    policy: dict
    evidence_root: Path
    environment_mode: str
```

### 8.2 `FactStore`

缓存预计算事实：

```text
frame facts
observation facts
association facts
object-version facts
object-pair facts
temporal facts
edge facts
```

### 8.3 `Finding`

```python
@dataclass
class Finding:
    finding_uid: str
    checker_id: str
    stage: str
    subtype: str
    scope: dict

    certainty: str
    severity: str

    proven_facts: list[str]
    hypotheses: list[dict]
    vetoes: list[str]
    missing_evidence: list[str]

    evidence_refs: dict
    route: str
    repair_allowed: bool = False
```

---

# 第五部分：Checker 0：证据与系统一致性

## 9. SYS-001：Artifact 与 UID 可解析

### 检查

```text
manifest 可读取
JSONL 每行合法
frame_uid 唯一
obs_uid 唯一
event_uid 唯一
object_uid 合法
所有引用文件存在
所有 NPZ key/index 合法
matrix axes 与 UID 列表一致
```

### 判定

任何确定性违反：

```text
CONFIRMED_SYSTEM_ERROR
severity = CRITICAL / HIGH
route = BLOCK_RUN
```

---

## 10. SYS-002：Top-K、top1、top2、margin 与矩阵一致

对每条 association：

```python
row = aggregate_sim[obs_index]
order = argsort(row)[::-1]
expected_top1 = order[0]
expected_top2 = order[1]
expected_margin = row[top1] - row[top2]
```

检查：

```text
保存顺序
保存分数
保存 margin
保存 object UID
```

都必须一致。

---

## 11. SYS-003：CREATE / ASSOCIATE 决策符合当前 policy

当前规则：

```text
max_score <= threshold
→ CREATE_OBJECT

max_score > threshold
→ MERGE_TO_ARGMAX_OBJECT
```

注意：

```text
等于 threshold 时也是 CREATE
```

检查：

```text
decision
target UID
threshold
argmax
```

---

## 12. SYS-004：Observation ownership

仅在：

```yaml
observation_ownership: exclusive_single_target
```

下执行。

如果同一个 `obs_uid` 同时正式属于多个 active object：

```text
POLICY VIOLATION
```

但必须明确：

```text
它违反的是当前关联 policy
不是现实世界物理规律
```

---

## 13. SYS-005：对象成员计数守恒

检查：

```text
num_detections
member occurrence 数量
unique member 数量
```

当前 exclusive membership 下应满足：

```text
num_detections
=
len(member_occurrences)
=
len(unique_members)
```

如系统允许 shared evidence，则改为 policy-dependent。

---

## 14. SYS-006：Merge source 生命周期

检查：

```text
source != target
source active
target active
source 未在当前 transaction 中被消费
source 未被再次使用
merge DAG 无环
```

当前代码已知风险：

```text
merge_overlap_objects 中检查 kept_objects[j]
但没有在每轮开始检查 kept_objects[i]
```

因此同一个 source 可能被多次消费。

该问题应作为固定 regression test。

---

## 15. SYS-007：Merge 成员守恒

需要：

```text
source_before
target_before
target_after
```

检查：

\[
Members_{after}
=
Members_{source-before}
\cup
Members_{target-before}
\]

不允许：

```text
成员丢失
成员凭空出现
成员重复
```

缺 target-before 时输出：

```text
INSUFFICIENT_EVIDENCE
```

---

## 16. SYS-008：事件重放与 final membership 一致

从：

```text
OBJECT_CREATE
OBS_ASSOCIATE
OBJECT_MERGE
OBJECT_FILTER
```

重放 object membership。

要求：

```text
replayed active objects
=
final_membership active objects
```

---

## 17. SYS-009：Edge endpoint 合法

检查：

```text
source object 存在
target object 存在
endpoint active
endpoint canonical
edge UID 与端点一致
```

---

# 第六部分：Checker 1：Detection Proposal

## 18. DET-001：同帧重复 proposal

### 定义

两个 detection 可能指向同一个现实物体。

### 指标

Mask IoU：

\[
IoU_M(A,B)
=
\frac{|M_A\cap M_B|}
{|M_A\cup M_B|}
\]

Mask containment：

\[
Contain_M(A,B)
=
\frac{|M_A\cap M_B|}
{\min(|M_A|,|M_B|)}
\]

还需：

```text
bbox IoU
CLIP similarity
depth distribution similarity
symmetric 3D support
label semantic compatibility
```

### 冷启动候选条件

```text
same_frame
AND (
  mask_iou > 0.85
  OR mask_containment > 0.95
)
AND clip_similarity > 0.90
AND symmetric_3d_support > 0.70
```

### 强 veto

```text
part-whole possible
container-content possible
support relation possible
two masks separated in depth/3D
```

### 输出

无 veto：

```text
LIKELY_MAPPING_CONFLICT
hypothesis = duplicate_proposal
```

有 veto：

```text
AMBIGUOUS_MAPPING_RISK
```

---

## 19. DET-002：疑似假阳性 detection

证据：

```text
低 detector confidence
极小 mask
低 valid_depth_ratio
PCD 点数很少
bbox 退化
单帧出现
无后续支持
与背景纹理高度一致
```

判定：

```text
单一低支持信号
→ AMBIGUOUS

低置信 + 低深度 + 无时序支持
→ LIKELY
```

单帧出现绝不能直接判错。

---

## 20. DET-003：类别标签不稳定

对比：

```text
detector class
CLIP image feature nearest labels
历史 class histogram
caption
```

只输出：

```text
class_label_noise risk
```

不要直接等同于 object identity error。

---

## 21. DET-004：漏检接口

当前 evidence 无法证明从未检测的物体存在。

规则只输出：

```text
NOT_SUPPORTED_IN_V1
requires:
  GT
  full-frame VLM
  visibility / coverage
```

---

# 第七部分：Checker 2：Segmentation

## 22. SEG-001：非法或退化 mask

确定性检查：

```text
processed mask 为空但 observation 被 kept
mask shape 与图像不一致
mask/bbox 越界
非法数值
processed area 与记录不一致
mask 有像素但所有 depth 无效
```

输出：

```text
CONFIRMED_SYSTEM_ERROR
```

---

## 23. SEG-002：背景泄漏

典型情况：

```text
cup mask 带入大量 table
chair mask 带入 wall
```

证据：

```text
depth 多峰
PCD 多大簇
mask 边界与 RGB/depth 边界不一致
bbox/extent 异常大
caption 经常描述背景主体
跨视角 geometry 变化剧烈
```

升级为 `LIKELY`：

```text
depth/3D 多簇
AND 语义不稳定
AND 不存在长物体/遮挡合理解释
```

---

## 24. SEG-003：欠分割

一个 mask 覆盖多个现实实例。

信号：

```text
两个以上稳定深度层
两个大 3D 簇
bbox 被异常拉长
caption 同时描述两个主体
后续帧中两个簇分别形成独立节点
```

强反例：

```text
长桌
沙发
椅背和椅腿
同一物体被遮挡后的多个可见区域
```

默认：

```text
AMBIGUOUS_MAPPING_RISK
```

多帧持续支持后：

```text
LIKELY_MAPPING_CONFLICT
```

---

## 25. SEG-004：过分割

同一现实物体被切成多个 mask。

信号：

```text
同一帧 mask 相邻
深度连续
3D 表面连续
CLIP 相似
后续反复关联到同一个 object
mask 并集形成稳定整体
```

注意：

```text
多个 observation → 同一个 object
```

在当前系统中合法。

此规则只说明：

```text
upstream segmentation fragmentation
```

---

## 26. SEG-005：Containment subtraction 损伤

比较 raw mask 与 processed mask：

\[
loss\_ratio
=
1-
\frac{|M_{processed}|}
{|M_{raw}|}
\]

检查：

```text
面积损失过高
processed mask 碎裂
PCD 点数骤降
bbox 过度缩小
被扣除对象不具有合理 part-whole 关系
```

输出：

```text
CONTAINMENT_SUBTRACTION_DAMAGE
```

通常进入 VLM，因为需要判断：

```text
合理的 pillow-on-sofa
还是错误的 mask subtraction
```

---

## 27. SEG-006：遮挡与低质量观测标签

不直接判错，只给 observation 增加质量标签：

```text
occluded
truncated_at_boundary
extreme_zoom
small_visible_fraction
low_valid_depth
high_background_ratio
```

Association checker 后续应降低其证据权重。

---

## 28. SEG-007：同帧 mask 交叠冲突图

建立同一帧 mask graph：

```text
node = observation
edge = overlap / containment / adjacency
```

将组件分类：

```text
duplicate-like
part-whole-like
adjacent-parts-like
multi-object-underseg-like
ambiguous
```

它是后续 root cause resolver 的关键输入。

---

# 第八部分：Checker 3：Projection / Geometry

## 29. GEO-001：非法 PCD / bbox

确定性检查：

```text
PCD 为空却 kept
非有限坐标
bbox volume 非有限
extent 负值
点数与 artifact 不一致
pose/intrinsics 不可解析
```

---

## 30. GEO-002：2D↔3D 重投影一致性

将 observation PCD 投回原帧：

\[
P_{3D}\rightarrow\hat M_{2D}
\]

计算：

\[
IoU(\hat M_{2D},M_{processed})
\]

严重不一致可能说明：

```text
pose 错
intrinsics 错
RGB-depth 对齐错
mask 索引错
PCD artifact 错
```

---

## 31. GEO-003：多簇异常

记录：

```text
cluster_count
largest_cluster_ratio
second_cluster_ratio
cluster center distance
depth modes
```

多簇事实只说明：

```text
可能欠分割
可能背景泄漏
可能深度噪声
```

不能直接决定根因。

---

## 32. GEO-004：跨帧几何冲突

对同 object 历史 observation：

```text
center trajectory
bbox extent trajectory
orientation trajectory
```

若出现异常跳变，输出：

```text
POSE_OR_ASSOCIATION_OR_CHANGE
```

在 Replica static mode 中：

```text
world change prior 很低
```

但仍不能把 pose error 与 false association 混为一谈。

---

## 33. GEO-005：Denoise 破坏

检查：

```text
point_count drop
bbox volume change
center shift
main cluster loss
后续 association score 是否恶化
```

升级条件：

```text
点数骤降
AND bbox/中心剧烈变化
AND 后续匹配明显变差
```

---

## 34. GEO-006：对象尺度异常

相对于同类或场景统计：

```text
bbox volume 极端
extent ratio 极端
point density 极端
```

只作为风险，不直接判错。

---

# 第九部分：Checker 4：Association

## 35. 基础指标

Top-1 / Top-2 margin：

\[
m=s_1-s_2
\]

Threshold slack：

\[
q=s_1-\tau
\]

Candidate entropy：

\[
p_j=
\frac{\exp(s_j/T)}
{\sum_k\exp(s_k/T)}
\]

\[
H=-\sum_jp_j\log p_j
\]

此外计算：

```text
spatial rank
visual rank
aggregate rank
target geometry support
target semantic medoid distance
co-visibility
observation quality
```

---

## 36. ASSOC-001：执行与日志一致性

对应 SYS-002 / SYS-003。

这是 hard invariant。

---

## 37. ASSOC-002：低 margin 风险

```text
margin 很小
```

只代表：

```text
系统当时犹豫
```

输出：

```text
AMBIGUOUS_MAPPING_RISK
```

不能直接说关联错误。

---

## 38. ASSOC-003：空间和语义排序冲突

例如：

```text
spatial top1 = obj_17
visual top1 = obj_31
aggregate top1 = obj_17
```

可能原因：

```text
同款相邻对象
pose/geometry 不准
mask 泄漏
object feature 污染
真实错误关联
```

输出：

```text
SPATIAL_SEMANTIC_DISAGREEMENT
```

---

## 39. ASSOC-004：同帧 many-to-one 分流

```text
obs_A → obj_17
obs_B → obj_17
```

分三类：

### A. 两个 observation 高度重合

倾向：

```text
duplicate proposal / duplicate segmentation
```

### B. 两个 observation 空间分离、类别不兼容

倾向：

```text
false association / false merge
```

### C. 两个 observation 是同一物体部件

倾向：

```text
part-whole / over-segmentation
```

所以 many-to-one 是：

```text
触发器
不是错误结论
```

---

## 40. ASSOC-005：成员语义离群

使用 leave-one-out robust medoid：

\[
D_{sem}(o_i,O)
=
1-\cos
\left(
f_i,
Medoid(F_{O\setminus i})
\right)
\]

使用 median/MAD 标准化：

\[
z_i=
\frac{D_i-median(D)}
{1.4826\cdot MAD(D)+\epsilon}
\]

冷启动：

```text
robust_z > 3
→ semantic outlier candidate
```

---

## 41. ASSOC-006：成员几何支持不足

\[
S_{geo}(o_i,O\setminus i)
=
\frac{
|\{p\in P_i:d(p,P_{O\setminus i})<r\}|
}
{|P_i|}
\]

解释：

```text
低 support 可能是错误关联
也可能是新视角首次看到物体另一侧
```

升级为 `LIKELY`：

```text
semantic outlier 高
AND geometry support 低
AND 无新视角/遮挡解释
```

---

## 42. ASSOC-007：高语义、低几何远距离关联

信号：

```text
spatial≈0
visual 很高
aggregate 刚过阈值
center distance 大
threshold slack 小
```

常见于：

```text
相同椅子
重复家具
纹理别名
pose error
```

只能高风险提示。

---

## 43. ASSOC-008：疑似漏关联 / false split 前兆

一个 observation 被 CREATE，但：

```text
top1 略低于 threshold
后续新节点与旧节点高度相似
最终 3D overlap 高
没有分离 co-visibility
时间连续
```

推断：

```text
threshold 过严
→ 本应关联却新建
→ duplicate node
```

---

## 44. ASSOC-009：Leave-One-Observation-Out 反事实

步骤：

```text
1. 临时移除可疑 observation
2. 用剩余成员重建 target core
3. 重新计算 observation 对 target core 和其他候选的分数
4. 比较新排名
```

若：

```text
原 target 分数显著下降
另一个 candidate 明显更高
另一个 candidate 同时获得语义和几何支持
```

则为强 false association 证据。

当前 v1 可先做：

```text
final-map approximate leave-one-out
```

严格 event-time 反事实需 object version ledger。

---

## 45. ASSOC-010：ID Switch

连续轨迹：

```text
frame10 → obj17
frame20 → obj31
frame30 → obj17
```

同时：

```text
obj17 与 obj31 最终高度重合
```

可能是 ID switch。

强 veto：

```text
两对象反复同帧空间分离共现
```

---

## 46. ASSOC-011：Candidate set 异常

检查：

```text
target 不在 candidate axis
矩阵列 UID 重复
候选对象在事件时已 inactive
candidate 数与 object snapshot 不一致
```

这是 system-level error。

---

# 第十部分：Checker 5：Fusion / Postprocess

## 47. FUSE-001～006：确定性生命周期规则

```text
FUSE-001 source != target
FUSE-002 source 单次消费
FUSE-003 source/target merge 前 active
FUSE-004 exclusive policy 下成员无交集
FUSE-005 target-after = source-before ∪ target-before
FUSE-006 merge graph 无环
```

---

## 48. FUSE-007：Fusion Shock

语义漂移：

\[
\Delta_{sem}
=
1-\cos(f_{before},f_{after})
\]

中心漂移：

\[
\Delta_{center}
=
\frac{
\|c_{after}-c_{before}\|
}
{diag(B_{before})+\epsilon}
\]

体积增长：

\[
\Delta_{volume}
=
\frac{V_{after}}
{V_{before}+\epsilon}
\]

点数增长：

\[
\Delta_{points}
=
\frac{N_{after}}
{N_{before}+\epsilon}
\]

触发：

```text
feature 突变
bbox 爆炸
中心跳变
新增独立大簇
class histogram 高熵化
```

Fusion shock 只定位：

```text
异常从哪次融合开始
```

不能单独证明 observation 错误。

---

## 49. FUSE-008：非对称 overlap 风险

计算：

```text
support(A→B)
support(B→A)
volume ratio
```

典型风险：

```text
small → large 很高
large → small 很低
```

可能是：

```text
pillow / sofa
cup / table
monitor / wall
```

所以 merge 需要：

```text
双向 support
语义兼容
尺度比
part-whole veto
co-visibility
```

---

## 50. FUSE-009：伪 text evidence

当前：

```text
text_sim = visual_sim
```

所以：

```text
visual + text
```

只能算一个证据家族。

---

## 51. FUSE-010：Postprocess merge candidate 审计

建议让 merge 代码输出所有 candidate：

```json
{
  "source_uid": "...",
  "target_uid": "...",
  "overlap_forward": 0.91,
  "overlap_backward": 0.42,
  "visual": 0.88,
  "text": null,
  "decision": "ACCEPT",
  "reject_reasons": []
}
```

这样查错器能判断：

```text
成功 merge 是否有 ACCEPT candidate
REJECT candidate 是否被错误执行
```

---

## 52. FUSE-011：Filter 误删风险

对象被 filter 时记录：

```text
实际触发条件
成员数
unique frame 数
n_points
下游边数
是否刚发生 fusion shock
```

确定性错误：

```text
实际不满足 filter 条件却被删除
```

现实风险：

```text
满足过滤条件但可能是真实小物体
```

后者不能自动判错。

---

# 第十一部分：Checker 6：Object Identity

## 53. OBJ-001：疑似错误融合

一个 object 内部形成两个稳定成员群：

```text
Cluster A 内部语义和几何一致
Cluster B 内部语义和几何一致
A 与 B 彼此冲突
```

强信号：

```text
语义双峰
3D 双簇
重复同帧空间分离共现
低 margin 历史关联
Fusion shock
```

升级为 `LIKELY`：

```text
重复分离 co-visibility
AND 两类独立证据
AND 无 part-whole veto
```

---

## 54. OBJ-002：疑似错误拆分 / duplicate object

对象对 \(A,B\)：

```text
CLIP similarity 高
双向 3D support 高
中心与 bbox 接近
时间轨迹互补
无分离 co-visibility
类别兼容
```

强 veto：

```text
反复同帧同时出现
processed mask 不重合
3D 中心稳定分离
```

这通常说明是两个真实同类实例。

---

## 55. OBJ-003：弱节点 / 噪声节点

指标：

```text
unique_frame_count
temporal_span
median_confidence
valid_depth_ratio
median_mask_area
n_points
bbox stability
feature consistency
edge dependencies
```

当前配置允许：

```text
obj_min_detections = 1
```

所以弱节点会大量保留。

第一版只输出：

```text
LOW_SUPPORT_OBJECT
```

不自动删除。

---

## 56. OBJ-004：Part-Whole 歧义

候选：

```text
chair / chair back
table / table top
sofa / pillow
cabinet / drawer
```

part-whole 是 duplicate merge 的强 veto。

需要：

```text
类别词义
mask containment
尺度比
空间关系
多视角
```

---

## 57. OBJ-005：Identity Instability

对象长期：

```text
class histogram 高熵
CLIP feature 多峰
bbox extent 多峰
center 多峰
caption 冲突
```

输出：

```text
IDENTITY_UNSTABLE
```

可能根因：

```text
false merge
background leakage
遮挡
粒度混乱
```

---

## 58. OBJ-006：Missing Object

v1 不实现内部自动判断。

接口要求：

```text
GT
full-frame VLM
active re-observation
visibility coverage
```

---

# 第十二部分：Checker 7：Caption

## 59. v1 范围

先列入框架，但优先级低于 node identity。

---

## 60. CAP-001：多视角 caption 冲突

检查：

```text
同一 object 的 caption 是否聚成多个语义簇
是否存在单一显著背景物反复抢占
```

---

## 61. CAP-002：Consolidated caption 与多数视角不一致

对：

```text
rough captions
consolidated caption
```

做语义一致性检查。

---

## 62. CAP-003：Caption 与几何/类别冲突

例如：

```text
caption = wall
bbox 体积很小且悬空
```

只作为风险，因为 caption 可能描述的是部件或背景。

---

## 63. CAP-004：输入视角质量

选择视角时考虑：

```text
mask area
valid depth
occlusion
background ratio
camera diversity
contributed point count
```

---

# 第十三部分：Checker 8：Relation

## 64. REL-001：Endpoint 合法性

对应 SYS-009。

---

## 65. REL-002：Relation ontology 归一化

将：

```text
on
on top of
resting on
supported by
```

归一化为：

```text
ON / SUPPORTS
```

其他：

```text
INSIDE / CONTAINS
ABOVE / BELOW
LEFT_OF / RIGHT_OF
FRONT_OF / BEHIND
NEAR
TOUCHING
```

---

## 66. REL-003：ON / SUPPORTS 几何一致性

检查：

```text
subject bottom 接近 support top
水平投影重叠
垂直间隙合理
```

允许：

```text
一个物体同时由多个物体支撑
```

---

## 67. REL-004：INSIDE / CONTAINS

不能只看 bbox overlap。

至少检查：

```text
subject point cloud 多数位于 container 内
```

---

## 68. REL-005：方向关系坐标系

LEFT/RIGHT/FRONT/BEHIND 必须记录：

```text
world frame
camera frame
object-centric frame
```

坐标系缺失：

```text
INSUFFICIENT_EVIDENCE
```

---

## 69. REL-006：严格互斥冲突

只在：

```text
同一对象对
同一时间
同一坐标系
```

下检查严格互斥关系。

以下不冲突：

```text
cup ON table
cup NEAR plate
cup IN tray
```

---

# 第十四部分：Temporal / World Change 预留

## 70. environment_mode

```yaml
environment_mode:
  static
  semi_static
  dynamic
```

同一现象：

```text
cup 从 table 到 desk
```

在 static Replica 中更可能：

```text
association / pose / map error
```

在真实家庭环境中可能：

```text
world change
```

v1 只输出：

```text
WORLD_CHANGE_OR_MAPPING_ERROR
```

---

# 第十五部分：独立证据家族

## 71. 六类证据

1. **程序与谱系**
   ```text
   UID、event、membership、merge lifecycle
   ```

2. **2D 视觉**
   ```text
   bbox、raw/processed mask、RGB crop
   ```

3. **深度与 3D**
   ```text
   depth、PCD、bbox、cluster、reprojection
   ```

4. **语义**
   ```text
   CLIP、类别分布、caption
   ```

5. **时序**
   ```text
   多帧连续性、co-visibility、ID switch
   ```

6. **关系与任务**
   ```text
   part-whole、container、support、下游依赖
   ```

升级为 `LIKELY_MAPPING_CONFLICT` 通常要求：

```text
至少两个独立证据家族
AND 无强 veto
```

---

# 第十六部分：Veto 机制

## 72. 常见强 veto

### 72.1 Part-whole

```text
高 overlap 不等于 duplicate
```

### 72.2 Repeated separated co-visibility

两个节点反复同帧空间分离共现：

```text
强烈反对 duplicate merge
```

### 72.3 New viewpoint

新视角看到此前未覆盖表面：

```text
低 geometry support 未必错误
```

### 72.4 Occlusion / truncation

低语义或几何一致性可能来自遮挡。

### 72.5 Dynamic world

真实物体可能移动。

---

# 第十七部分：Root Cause Resolver

## 73. 原则

多个 checker finding 不能简单堆叠。

优先寻找：

```text
最早出现的异常
能解释最多下游症状的异常
```

### 优先级

```text
System
→ Detection
→ Segmentation
→ Geometry
→ Association
→ Fusion
→ Identity
→ Caption
→ Relation
```

---

## 74. 典型因果模板

### 74.1 重复节点

```text
DET-001 duplicate proposal
→ ASSOC-008 borderline CREATE
→ OBJ-002 duplicate pair
```

### 74.2 错误融合

```text
SEG-002 background leakage
→ ASSOC-003 spatial-semantic disagreement
→ ASSOC-006 low geometry support
→ FUSE-007 fusion shock
→ OBJ-001 identity bimodality
```

### 74.3 位姿错误

```text
同一帧多个 GEO-004
→ 多个 ASSOC-003
→ 多对象同时 identity unstable
```

优先根因：

```text
pose / transform anomaly
```

而不是几十个独立 false association。

---

## 75. Root cause 输出

```json
{
  "root_cause_uid": "rc_0001",
  "primary_hypothesis": "SEGMENTATION_BACKGROUND_LEAKAGE",
  "supporting_findings": [
    "finding_001",
    "finding_017",
    "finding_031"
  ],
  "explains": [
    "false_association",
    "fusion_shock",
    "identity_instability"
  ],
  "alternative_hypotheses": [
    "pose_error"
  ],
  "missing_evidence": [
    "processed_mask",
    "target_before_version"
  ],
  "certainty": "LIKELY_MAPPING_CONFLICT"
}
```

---

# 第十八部分：Evidence Packet

## 76. 生成原则

只对被规则触发的 suspect case 生成，不给每个 observation 存一座图片仓库。

---

## 77. Duplicate Proposal Packet

```text
full RGB frame
A/B mask overlay
A masked crop
B masked crop
A/B context crop
depth overlay
3D PCD overlay
metric table
```

---

## 78. False Association Packet

```text
suspect observation masked crop
suspect context crop
top1 object 代表视角
top2 object 代表视角
top1/top2 score table
observation vs target-core 3D overlay
timeline
fusion shock before/after
```

---

## 79. False Split Packet

```text
object A best views
object B best views
A/B 3D overlay
same-frame co-visibility frames
timeline
pair metrics
part-whole warning
```

---

## 80. False Merge Packet

```text
object member montage
semantic cluster A views
semantic cluster B views
3D cluster overlay
critical association event
critical fusion event
before/after bbox
```

---

## 81. 代表视角选择

每个 object 最多 4～6 张：

```text
最高 contributed points
最高 confidence
最大 camera viewpoint diversity
最强冲突视角
最早创建视角
最近一次视角
```

---

# 第十九部分：Finding Schema

## 82. 完整示例

```json
{
  "finding_uid": "finding_000123",
  "checker_id": "ASSOC-006",
  "stage": "association",
  "subtype": "LOW_GEOMETRIC_SUPPORT",

  "scope": {
    "frame_uid": "...",
    "obs_uid": "...",
    "event_uid": "...",
    "object_uid": "..."
  },

  "certainty": "LIKELY_MAPPING_CONFLICT",
  "severity": "HIGH",

  "policy_context": {
    "observation_ownership": "exclusive_single_target",
    "environment_mode": "static"
  },

  "proven_facts": [
    {
      "name": "semantic_outlier_robust_z",
      "value": 4.1
    },
    {
      "name": "geometric_support",
      "value": 0.08
    },
    {
      "name": "association_margin",
      "value": 0.03
    }
  ],

  "hypotheses": [
    {
      "name": "false_association",
      "support": [
        "semantic_outlier",
        "low_geometric_support"
      ]
    },
    {
      "name": "segmentation_background_leakage",
      "support": [
        "multi_depth_modes"
      ]
    }
  ],

  "vetoes": [
    "part_whole_not_excluded"
  ],

  "missing_evidence": [
    "target_object_version_before_association"
  ],

  "evidence_refs": {
    "association_event": "...",
    "processed_mask": "...",
    "observation_pcd": "...",
    "case_packet": "cases/finding_000123/"
  },

  "route": "VLM_REVIEW",
  "repair_allowed": false
}
```

必须严格分开：

```text
proven_facts
hypotheses
vetoes
missing_evidence
```

---

# 第二十部分：不要设计万能异常总分

## 83. 错误做法

\[
R=w_1R_{geo}+w_2R_{sem}+w_3R_{margin}
\]

然后：

```text
R > 0.7 → 错
```

问题：

```text
part-whole 得高 overlap
同款椅子得高 visual
新视角得低 geometry support
小物体得低 frame count
```

一个总分会把不同含义的证据乱加。

---

## 84. 正确做法

```text
逻辑组合
+ 独立证据家族
+ 强 veto
+ 确定性等级
```

总分只允许用于：

```text
review_priority
```

不能用于：

```text
真值判决
```

---

# 第二十一部分：阈值策略

## 85. 冷启动阈值

### Duplicate proposal

```text
mask_iou > 0.85
OR containment > 0.95

clip_similarity > 0.90
symmetric_3d_support > 0.70
```

### False association

```text
semantic_outlier_robust_z > 3
geometric_support < 0.15
```

margin 只作为辅助：

```text
margin < 0.03～0.05
threshold_slack < 0.05
```

### Duplicate object

```text
clip_similarity > 0.90
symmetric_3d_support > 0.75
normalized_center_distance < 0.20
no repeated separated co-visibility
```

---

## 86. 正式阈值校准

优先使用：

```text
median
MAD
percentile
object-scale normalized distance
scene-specific calibration
```

而不是固定米制阈值。

---

# 第二十二部分：代码结构

## 87. 推荐目录

```text
conceptgraph/
└── audit/
    ├── __init__.py
    ├── context.py
    ├── evidence_index.py
    ├── artifact_resolver.py
    ├── facts.py
    ├── finding.py
    ├── runner.py
    │
    ├── checkers/
    │   ├── base.py
    │   ├── system_integrity.py
    │   ├── detection.py
    │   ├── segmentation.py
    │   ├── projection_geometry.py
    │   ├── association.py
    │   ├── fusion.py
    │   ├── object_identity.py
    │   ├── caption.py
    │   └── relation.py
    │
    ├── triage/
    │   ├── root_cause.py
    │   ├── certainty.py
    │   ├── vetoes.py
    │   └── thresholds.py
    │
    └── cases/
        ├── evidence_packet.py
        ├── image_render.py
        └── pcd_render.py
```

---

## 88. Checker 接口

```python
class Checker:
    checker_id: str
    required_evidence: set[str]

    def run(
        self,
        context: AuditContext,
        facts: FactStore,
    ) -> list[Finding]:
        raise NotImplementedError
```

规则返回：

```text
Finding[]
```

不能直接 mutate map。

---

## 89. Fact Builder

集中预计算：

```text
mask pair matrix
object-pair 3D support
CLIP pair matrix
co-visibility
semantic medoid
robust outlier
object version delta
edge endpoint map
```

避免每个 checker 重复计算。

---

# 第二十三部分：运行方式

## 90. CLI

```bash
python -m conceptgraph.audit.runner \
  --experiment_dir /path/to/exps/run_name \
  --audit_config conceptgraph/audit/configs/v1.yaml \
  --build_cases true
```

---

## 91. 输出目录

```text
experiment/
├── evidence/
└── audit/
    ├── audit_manifest.json
    ├── evidence_validation.json
    ├── findings.jsonl
    ├── root_causes.jsonl
    ├── audit_summary.json
    ├── metrics_cache/
    └── cases/
        ├── finding_000001/
        ├── finding_000002/
        └── ...
```

---

## 92. Audit config 示例

```yaml
version: 1

strict_evidence: true
environment_mode: static

enabled_checkers:
  system_integrity: true
  detection: true
  segmentation: true
  projection_geometry: true
  association: true
  fusion: true
  object_identity: true
  caption: false
  relation: false

thresholds:
  duplicate:
    mask_iou: 0.85
    containment: 0.95
    clip_similarity: 0.90
    symmetric_3d_support: 0.70

  association:
    semantic_outlier_z: 3.0
    geometric_support_low: 0.15
    low_margin: 0.05
    low_threshold_slack: 0.05

  object_pair:
    duplicate_clip_similarity: 0.90
    duplicate_symmetric_support: 0.75
    center_distance_normalized: 0.20

case_builder:
  enabled: true
  max_images_per_object: 6
  save_3d_overlay: true
  save_depth_overlay: true
```

---

# 第二十四部分：单元测试与回归测试

## 93. System tests

### SYS test 1

篡改 margin：

```text
矩阵真实 margin = 0.08
日志 margin = 0.30
```

应触发：

```text
SYS-002
```

### SYS test 2

同一 merge source 被消费两次：

```text
obj_A → obj_B
obj_A → obj_C
```

应触发：

```text
SYS-006 / FUSE-002
```

### SYS test 3

edge 指向 inactive object：

应触发：

```text
SYS-009
```

---

## 94. Real-conflict synthetic tests

### Duplicate proposal

两个几乎相同 mask / PCD：

```text
应触发 DET-001
```

### Part-whole veto

```text
sofa + pillow
```

高 overlap 但应降级为：

```text
AMBIGUOUS
```

### Two identical chairs veto

两把外观相同椅子反复同帧分离出现：

```text
不应判 duplicate
```

### False association

可疑 observation：

```text
semantic outlier 高
geometry support 低
alternate candidate 更强
```

应触发：

```text
ASSOC-006 / ASSOC-009
```

### New viewpoint legal case

低 geometry support，但语义稳定且 viewpoint coverage 明显新增：

```text
不应升级为 LIKELY
```

---

## 95. 当前 smoke snapshot 回归

固定检查：

```text
missing_reference_count
duplicate_membership_occurrences
merge_source_reuse
association matrix consistency
```

当前已知 source 重复消费问题必须能被稳定检测。

---

# 第二十五部分：评估方案

## 96. 第一阶段数据

先在：

```text
Replica room0 完整序列
```

上运行。

两帧 smoke 只能测代码，不适合调阈值。

随后扩展：

```text
room1
room2
office0～office4
```

---

## 97. 人工标注单位

每条 finding 标注：

```text
true error
valid unusual case
insufficient evidence
wrong root cause
```

同时标记：

```text
error stage
error subtype
recommended action
```

---

## 98. 查错器指标

每一类单独报告：

```text
precision
recall（有完整标注时）
unknown rate
VLM escalation rate
平均 case 构建成本
运行时开销
```

特别关注：

```text
LIKELY_MAPPING_CONFLICT precision
```

第一版建议目标：

```text
LIKELY precision ≥ 90%～95%
```

宁可召回低，也不要把合法现实状态大量误判为错误。

---

## 99. 不允许只报告一个总分

必须分：

```text
Detection
Segmentation
Association
Fusion
Object Identity
```

否则不知道到底哪部分有用。

---

# 第二十六部分：实施顺序

## 100. P0：证据补齐

1. processed mask  
2. 真实 filter trace  
3. structured ArtifactRef  
4. object version ledger  
5. merge 三方状态  
6. depth/cluster stats  
7. audit policy  

---

## 101. P1：System Integrity Checker

优先完成：

```text
SYS-001～009
```

目标：

```text
先保证用于查错的账本可信
```

---

## 102. P2：Detection + Segmentation + Geometry

实现：

```text
DET-001～003
SEG-001～007
GEO-001～006
```

---

## 103. P3：Association + Fusion

实现：

```text
ASSOC-001～011
FUSE-001～011
```

这是第一版最关键部分。

---

## 104. P4：Object Identity + Root Cause

实现：

```text
OBJ-001～005
RootCauseResolver
```

---

## 105. P5：Evidence Packet 与校准

```text
生成图片包
人工标注 Top-K
调整阈值
建立 case bank
```

---

# 第二十七部分：v1 验收标准

## 106. 必须满足

1. Audit 开关不改变 mapping 输出。  
2. 缺证据输出 `INSUFFICIENT_EVIDENCE`，不输出假 PASS。  
3. 每条 hard invariant 都有单元测试。  
4. 当前 merge source 重复消费问题可稳定检测。  
5. 每条 finding 定位到 frame / obs / event / object。  
6. 每条现实冲突 finding 有事实、假设、veto、缺失证据。  
7. 每个高风险 case 可生成可读 Evidence Packet。  
8. v1 不执行 detach / merge / delete。  
9. 完整 room0 上完成 Top-K 人工核验。  
10. 分类别报告 precision、unknown rate、VLM escalation rate。  

---

# 第二十八部分：明确不在 v1 做的事情

```text
自动回滚
自动提交修复
动态世界变化判断
完整漏检检测
最终 caption factuality
最终 relation correctness
任务相关修复优先级
```

这些都建立在 v1 查错器可信之后。

---

# 第二十九部分：最终架构总结

```text
统一证据
   ↓
Evidence Readiness
   ↓
确定性系统检查
   ↓
Detection / Segmentation / Geometry
   ↓
Association / Fusion
   ↓
Object Identity
   ↓
Root Cause Resolver
   ↓
┌──────────────────────────────┐
│ CONFIRMED_SYSTEM_ERROR       │ → 修代码 / 阻止实验
│ LIKELY_MAPPING_CONFLICT      │ → 高优先级人工核验
│ AMBIGUOUS_MAPPING_RISK       │ → VLM Evidence Packet
│ INSUFFICIENT_EVIDENCE        │ → 补证据
│ NO_CONFLICT_FOUND            │ → 无异常，但不等于 GT 正确
└──────────────────────────────┘
```

本版本最重要的原则是：

> **硬规则只确认系统内部逻辑不可能同时成立的错误；涉及现实物体身份、归属、语义和关系的规则，必须作为分层证据驱动的异常诊断，而不是一条阈值直接宣判。**

最终得到的不是一个“看到异常就判刑”的规则堆，而是一套：

\[
\boxed{
\text{Provenance-Grounded}
+
\text{Stage-Aware}
+
\text{Uncertainty-Aware}
+
\text{Causal Audit Engine}
}
\]

它可以成为后续：

```text
规则直接修复
VLM 歧义裁决
局部回放
sandbox verification
self-healing scene graph
```

的稳定底座。
