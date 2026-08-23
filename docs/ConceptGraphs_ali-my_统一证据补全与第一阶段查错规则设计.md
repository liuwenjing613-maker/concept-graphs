# ConceptGraphs `ali-my`：统一证据补全与第一阶段查错规则设计

> 适用范围：`liuwenjing613-maker/concept-graphs` 仓库 `ali-my` 分支，重点针对 `rerun_realtime_mapping.py` 与 `conceptgraph/utils/evidence.py`。  
> 当前目标：先补齐能够支撑可靠追溯与自动排查的证据，再建立一套只读、可解释、可复现的第一阶段查错规则。  
> 当前不做：自动回滚执行、直接修改地图、动态世界变化判断、完整 caption/关系边纠错、在线 VLM 决策闭环。

---

## 总体结论

`ali-my` 当前已经具备较完整的基础证据链：

```text
Run
  → Frame
  → Observation
  → Association
  → Object lifecycle
  → Final membership
  → VLM / Edge events
```

已有内容包括：

- `manifest.json`
- `frames.jsonl`
- `observations.jsonl`
- `similarities/frame_*.npz`
- `associations.jsonl`
- `observation_pcd/*.npz`
- `mapping_events.jsonl`
- `vlm_events.jsonl`
- `final_membership.json`
- `evidence_summary.json`

这些内容已经足以开始开发查错机制，不需要推倒重做。但当前系统仍有两类不足：

1. **证据虽然能“追到哪里”，但部分证据并不是建图时真正使用的最终输入，或者无法精确恢复当时状态。**
2. **当前只记录了部分成功决策，缺少对象版本、后处理候选、精确图像证据和严格完整性约束，尚不足以支撑可靠自动判断。**

因此下一步应按以下顺序推进：

```text
补齐关键证据
  → Evidence Readiness Validator
  → 硬性不变量检查
  → 四类高频错误检查
  → 生成 findings 与图片证据包
  → 后续再接确定性修正或 VLM 判断
```

---

# 第一部分：目前统一证据缺乏的内容及补充方案

## 1. 当前证据能力与边界

### 1.1 已经满足的能力

当前证据可以回答：

- 某个最终对象由哪些 `obs_uid` 构成；
- 每个 observation 来自哪一帧、哪个原始检测；
- observation 的原始 bbox、mask、置信度、类别、点云和特征在哪里；
- 一次在线关联的全部候选对象、空间分数、视觉分数、总分、阈值和 margin；
- 一次对象后处理合并的源对象、目标对象和合并分数；
- 对象是否经历过滤、去噪、合并和边变化；
- 最终对象 UID 与当前数组下标之间的对应关系；
- 证据开关是否改变 baseline 输出。

### 1.2 仍然不能可靠回答的问题

当前证据还不能完整回答：

- 生成 observation 点云时真正使用的是哪一版 mask；
- 过滤记录是否来自实际执行路径，还是事后重新推断；
- 某个对象在一次 observation 融合前后分别是什么状态；
- 后处理阶段有哪些候选对象对被检查但没有合并；
- 同一个对象被连续修改时，每次修改对应哪个版本；
- 当前所谓 `text_similarity` 是否真的是独立文字证据；
- 一个文件引用不仅“路径存在”，而且数组 key、index、shape 是否有效；
- 当前运行能否在另一台机器上精确复现；
- VLM 实际看到了哪些图片、完整提示词是什么、参数是什么；
- 如何直接生成供人工或 VLM 判断的自包含图片证据包。

下面按优先级逐项补齐。

---

## 2. P0：必须立即补齐的证据

## 2.1 保存真正用于 3D 建图的 `processed_mask`

### 当前问题

当前 `observations.jsonl` 中的 `mask_ref` 指向检测阶段保存的原始 mask，但实际建图流程为：

```text
raw mask
  → resize
  → filter
  → mask_subtract_contained
  → depth back-projection
  → observation point cloud
```

因此当前日志中的 mask 与生成 `observation_pcd` 的 mask 可能不一致。

### 风险

这会影响：

- 2D 重复检测判断；
- mask 泄漏和分割碎片判断；
- 2D/3D 一致性检查；
- observation 点云重建；
- VLM 对分割是否合理的判断。

### 必须补充

在 `mask_subtract_contained()` 之后、生成 3D 点云之前保存：

```text
evidence/processed_masks/<obs_uid>.npz
```

并在 `observations.jsonl` 增加：

```json
{
  "raw_mask_ref": {"...": "..."},
  "processed_mask_ref": {"...": "..."},
  "raw_mask_area": 0,
  "processed_mask_area": 0,
  "removed_pixel_count": 0,
  "mask_operations": [
    "resize_nearest",
    "filter",
    "subtract_contained"
  ],
  "subtract_source_obs_uids": []
}
```

### 验收标准

- 每个 kept observation 必须有 `processed_mask_ref`；
- `processed_mask_area` 必须等于文件中 mask 的像素和；
- 使用 processed mask、depth、intrinsics、pose 重建的原始 3D 点数应与日志记录一致；
- raw mask 和 processed mask 必须能够同时查看和对比。

---

## 2.2 将过滤证据改为“执行时记录”，不能事后重新推断

### 当前问题

当前 `EvidenceRecorder._filter_reason()` 根据配置和 observation 信息重新计算过滤原因。它与当前 `filter_gobs()` 大体一致，但本质上是两套逻辑。

一旦未来修改：

- 过滤顺序；
- 阈值；
- 特殊类别；
- mask 操作；
- 置信度策略；

证据记录可能与真实执行路径不一致。

### 必须补充

让 `filter_gobs()` 直接返回或通过 callback 写出 `filter_trace`：

```json
{
  "obs_uid": "...",
  "decision": "KEEP",
  "evaluated_gates": [
    {
      "gate": "mask_area",
      "value": 1234,
      "operator": ">=",
      "threshold": 25,
      "passed": true
    },
    {
      "gate": "confidence",
      "value": 0.82,
      "operator": ">=",
      "threshold": 0.25,
      "passed": true
    }
  ],
  "first_failed_gate": null
}
```

被过滤 observation 必须记录第一个真实失败 gate，而不是用事后规则猜测。

### 建议接口

```python
filtered_gobs, filter_trace = filter_gobs(..., return_trace=True)
```

### 验收标准

- `filter_trace` 中 KEEP 的 observation 集合必须与实际 filtered observation 集合完全一致；
- 每个 rejected observation 必须有唯一、明确的 `first_failed_gate`；
- 修改过滤规则后，证据记录不需要同步维护第二套判断逻辑。

---

## 2.3 将字符串文件引用升级为可验证的结构化 `ArtifactRef`

### 当前问题

当前引用常写成：

```text
../detections/frame000020/mask.npz#arr_0[3]
```

这种格式可读，但不能保证：

- 文件存在；
- `arr_0` 存在；
- index 3 合法；
- shape 与预期一致；
- 文件没有被覆盖；
- 拷贝到另一台机器后仍可解析。

### 必须补充

统一使用结构化引用：

```json
{
  "path": "../detections/frame000020/mask.npz",
  "format": "npz",
  "key": "arr_0",
  "index": 3,
  "sha256": "...",
  "shape": [36, 680, 1200],
  "dtype": "bool"
}
```

RGB、depth、crop、feature、mask、PCD、similarity matrix 均使用统一格式。

### 新增组件

```text
conceptgraph/audit/evidence_resolver.py
```

至少提供：

```python
resolve_artifact(ref)
validate_artifact(ref)
load_artifact(ref)
```

### 验收标准

- 正式实验中所有引用必须可解析；
- 不仅校验文件，还要校验 key、index、shape、dtype；
- `missing_reference_count` 应扩展为详细分类，而不是单个总数；
- 任一引用失效时，run 标记为 `EVIDENCE_INVALID`，不能进入语义查错阶段。

---

## 2.4 补齐 observation 点云的确定性与完整性信息

### 当前问题

当前已经保存 observation PCD，但仍存在三个风险：

1. PCD 可配置采样上限，日志未明确说明是否采样；
2. 3D 点生成代码存在随机扰动，未保存随机种子；
3. 未区分原始反投影、采样、voxel downsample、DBSCAN 后各阶段点数。

### 必须补充

在 observation 记录中增加：

```json
{
  "pcd_ref": {"...": "..."},
  "pcd_stage": "after_init_process",
  "pcd_is_sampled": false,
  "pcd_raw_valid_depth_points": 0,
  "pcd_before_downsample_points": 0,
  "pcd_after_downsample_points": 0,
  "pcd_after_dbscan_points": 0,
  "pcd_stored_points": 0,
  "pcd_sample_indices_ref": null,
  "pcd_random_seed": 0,
  "voxel_size": 0.01,
  "dbscan_eps": 0.1,
  "dbscan_min_points": 10,
  "points_sha256": "..."
}
```

### 两种可接受方案

#### 方案 A：保存完整、实际参与融合的 observation PCD

优点：最直接，后续重建稳定。  
缺点：磁盘占用更大。

#### 方案 B：保存完整 processed mask、depth、pose、intrinsics、随机种子和确定性采样信息

优点：磁盘更省。  
缺点：重建路径更复杂，需要保证所有几何操作确定性。

当前 Replica 实验规模下，优先推荐方案 A。

### 验收标准

- 从证据重载 PCD 后，点数和 hash 与日志一致；
- 日志明确区分“完整 PCD”和“仅用于可视化的采样 PCD”；
- 任何用于后续反事实检查的 PCD 不允许是不透明采样结果。

---

## 2.5 增加对象版本链 `object_versions.jsonl`

### 当前问题

当前可以知道一次 observation 被关联到哪个 object，但不知道目标对象在此次融合前后具体发生了什么变化。

例如无法直接回答：

- 加入某个 observation 后 feature 是否突然漂移；
- bbox 是否异常扩大；
- 中心是否突然移动；
- 类别直方图是否开始冲突；
- 点云是否出现新的孤立部分。

### 必须新增

```text
evidence/object_versions.jsonl
```

对象版本格式：

```text
<object_uid>@v000001
<object_uid>@v000002
```

每次以下操作后版本加一：

- `OBJECT_CREATE`
- `OBS_ASSOCIATE`
- `OBJECT_DENOISE`
- `OBJECT_FILTER`
- `OBJECT_MERGE`
- 以后新增的 `OBS_DETACH`、`OBJECT_REBUILD`

### 每个版本必须记录

```json
{
  "object_version_uid": "obj_uuid@v000007",
  "object_uid": "obj_uuid",
  "version": 7,
  "trigger_event_uid": "event_uid",
  "status": "active",

  "member_observation_uids": [],
  "num_unique_observations": 0,
  "num_detections": 0,
  "unique_frame_count": 0,

  "n_points": 0,
  "bbox_center": [],
  "bbox_extent": [],
  "bbox_volume": 0,

  "class_histogram": {},
  "dominant_class": "...",
  "dominant_class_ratio": 0.0,

  "clip_feature_ref": {"...": "..."},
  "clip_feature_sha256": "...",

  "parent_version_uids": []
}
```

### 关联事件同步增加

```json
{
  "target_object_version_before": "obj@v6",
  "target_object_version_after": "obj@v7",
  "candidate_object_version_uids": ["..."]
}
```

### 验收标准

- 每个 active object 的版本号严格递增；
- 每个版本必须有唯一触发事件；
- 当前版本成员集合必须能够由上一版本与触发事件推导；
- 最终版本必须与 `final_membership.json` 完全一致。

---

## 2.6 完整记录后处理 merge 的候选、拒绝原因和三方状态

### 当前问题

当前主要记录成功的 `OBJECT_MERGE`，但对于错误拆分检查，真正需要知道：

- 哪些对象对被比较；
- 哪些接近阈值但没有合并；
- 是 overlap、visual 还是 text gate 拒绝；
- 遍历顺序是否影响结果；
- source 是否已经在同一事务中被消费。

当前合并事件还缺少目标对象合并前状态。现有 `before_summary` 主要对应 source，`after_summary` 对应合并后的 target。

### 必须新增

```text
evidence/object_pair_decisions.jsonl
```

每次 merge interval 创建一个事务：

```json
{
  "merge_transaction_uid": "merge_tx_frame_000100_01",
  "frame_uid": "...",
  "candidate_rank": 1,

  "source_object_version_uid": "objA@v3",
  "target_object_version_uid": "objB@v5",

  "overlap_a_to_b": 0.91,
  "overlap_b_to_a": 0.76,
  "visual_similarity": 0.87,
  "text_similarity": null,
  "text_similarity_source": "unavailable",

  "thresholds": {
    "overlap": 0.7,
    "visual": 0.7,
    "text": 0.7
  },

  "decision": "ACCEPT",
  "reject_reason": null,
  "source_active_before": true,
  "target_active_before": true,
  "source_consumed_after": true
}
```

成功合并事件必须保存：

```json
{
  "source_before": {},
  "target_before": {},
  "target_after": {}
}
```

### 必须增加的 merge 不变量字段

- `source_consumed_in_transaction`
- `source_member_set_before`
- `target_member_set_before`
- `member_intersection_before`
- `member_union_after`

### 验收标准

- 每个成功 merge 必须对应一个 `ACCEPT` candidate；
- 同一 merge transaction 中，一个 source 只能被消费一次；
- source 和 target 合并前成员集合必须不相交；
- target-after 成员集合必须等于两者集合并集；
- rejected near-threshold candidate 必须可查询。

---

## 2.7 修正 text feature 和 `text_similarity` 的证据语义

### 当前问题

当前 batched CLIP 路径可能返回空 `text_feats`，但 observation 仍可能写出看似有效的 `text_feat_ref`。同时后处理 merge 中：

```python
text_sim = visual_sim
```

这意味着 visual 与 text 并非两个独立证据。

### 必须补充

#### 无 text feature 时

```json
{
  "text_feat_ref": null,
  "text_feature_status": "NOT_COMPUTED"
}
```

#### 使用 visual 作为代理时

```json
{
  "text_similarity": 0.91,
  "text_similarity_source": "VISUAL_PROXY",
  "independent_evidence_group": "image_clip"
}
```

#### 真正计算 label/text feature 时

```json
{
  "text_similarity_source": "CLIP_TEXT_LABEL",
  "independent_evidence_group": "label_semantics"
}
```

### 规则要求

查错系统不得把同一特征复制出的 visual 与 text score 当作两个独立证据。

### 验收标准

- 不存在指向空数组或越界元素的 `text_feat_ref`；
- 每个语义分数都明确标注来源；
- 所有规则按 `independent_evidence_group` 统计独立证据数量。

---

## 2.8 增加因果链接、事务 ID 和跨分支 lineage

### 当前问题

当前 event UID 能表示顺序，但尚未形成完整的因果依赖图。后续需要知道：

- 某个对象版本由哪些事件产生；
- 某次后处理 merge 依赖哪些对象版本；
- 某条 edge 依赖哪些节点版本和 observation；
- repair branch 中的新对象与 baseline 对象是什么关系。

### 必须补充

所有 event 统一增加：

```json
{
  "transaction_uid": "...",
  "parent_event_uids": [],
  "input_object_version_uids": [],
  "output_object_version_uids": [],
  "branch_id": "baseline",
  "event_sequence": 123
}
```

对象增加：

```json
{
  "lineage_uid": "origin_<first_obs_uid>",
  "origin_observation_uid": "...",
  "branch_object_uid": "..."
}
```

### 设计原则

- 当前 UUID 继续作为 run 内稳定对象 ID；
- `lineage_uid` 用于 baseline 与后续 repair branch 对齐；
- 数组下标和 `curr_obj_num` 只用于显示，不能参与因果追踪。

### 验收标准

- event graph 无环；
- 每个对象版本都能追到一个或多个原始 observation；
- final object 的全部上游事件可通过 parent link 查询；
- 后续分支地图可以与 baseline 做对象级 diff。

---

## 3. P1：在接入 VLM 或正式实验前补齐

## 3.1 保存 VLM 的精确输入，而不只保存 fingerprint

### 当前问题

当前 VLM 证据已经保存模型、原始响应、耗时、状态、解析输出和 prompt fingerprint，但 fingerprint 不能恢复完整调用。

还缺少：

- 完整规范化 prompt 文本；
- 实际输入图片列表；
- 图片 hash；
- temperature、max tokens 等参数；
- parser 版本；
- request ID；
- 图片与 observation/object UID 的对应关系。

### 必须补充

```json
{
  "call_uid": "...",
  "call_type": "FRAME_EDGE",
  "model_name": "...",
  "model_revision": "...",

  "prompt_template_version": "...",
  "prompt_text": "...",
  "prompt_fingerprint": "...",

  "image_inputs": [
    {
      "artifact_ref": {},
      "sha256": "...",
      "linked_obs_uids": []
    }
  ],

  "generation_params": {
    "temperature": 0,
    "max_tokens": 0,
    "seed": null
  },

  "raw_response": "...",
  "parsed_output": {},
  "parser_version": "...",
  "status": "ok",
  "latency_ms": 0
}
```

### 验收标准

- 能够根据 VLM event 重建完全相同的输入 packet；
- `make_edges=true` 至少完成一次真实 smoke；
- 图片缺失时，VLM evidence run 标记为无效；
- 不在 JSON 中重复写 base64 图片，只保存 artifact ref 与 hash。

---

## 3.2 增加代码、环境、模型权重和最终输出 hash

### 当前问题

当前 smoke manifest 记录过 `git_dirty=true`，这意味着仅靠 commit 无法恢复运行时代码。

### 必须补充

`manifest.json` 增加：

```json
{
  "git_head": "...",
  "git_dirty": false,
  "git_diff_sha256": null,
  "git_patch_ref": null,

  "python_version": "...",
  "cuda_version": "...",
  "torch_version": "...",
  "open3d_version": "...",
  "dependency_lock_ref": "...",

  "random_seeds": {
    "python": 0,
    "numpy": 0,
    "torch": 0
  },

  "model_weights": {
    "detector": {"name": "...", "sha256": "..."},
    "segmenter": {"name": "...", "sha256": "..."},
    "clip": {"name": "...", "sha256": "..."}
  },

  "final_outputs": {
    "pcd_ref": {},
    "pcd_sha256": "...",
    "object_json_sha256": "...",
    "edge_json_sha256": "..."
  }
}
```

### 验收标准

- 正式实验优先要求 clean commit；
- dirty run 必须自动保存 patch；
- evidence on/off 的最终地图 hash 对照仍保持一致；
- 同一输入、同一 seed 的重复运行应在允许容差内一致。

---

## 3.3 增加 `best_effort` 与 `strict` 两种证据模式

### 当前问题

当前 sidecar 采用“日志失败不能改变建图结果”的设计，这是正确的 baseline 保护策略。但正式论文实验不能在证据已经损坏的情况下继续被视为有效 run。

### 建议配置

```yaml
evidence_mode: best_effort   # 开发期
evidence_mode: strict        # 正式实验
```

### 状态定义

```text
MAP_COMPLETED_EVIDENCE_VALID
MAP_COMPLETED_EVIDENCE_INVALID
MAP_FAILED
```

### strict 模式失败条件

- 引用缺失；
- 数组 selector 越界；
- UID 重复；
- membership 不一致；
- event chain 断裂；
- object version 不连续；
- 日志写入异常；
- VLM 图片或 prompt 不完整；
- schema 校验失败。

### 验收标准

正式实验只有 `MAP_COMPLETED_EVIDENCE_VALID` 才能进入查错与论文统计。

---

## 3.4 按需生成自包含图片证据包，不要无差别复制所有图片

### 当前问题

Rerun 可以显示 RGB、depth、点云、bbox 和边，但它不是稳定、自动保存、可直接交给 VLM 的证据包。当前 bbox crop 还会包含较多背景，容易导致显著背景物体抢占判断。

### 推荐目录

```text
audit/cases/<finding_uid>/
├── case.json
├── overview.jpg
├── masked_crop_<obs_uid>.jpg
├── context_crop_<obs_uid>.jpg
├── candidate_compare.jpg
├── pcd_overlay.png
└── metrics.json
```

### 每个 observation 生成两种裁剪

#### Masked crop

- 使用 `processed_mask`；
- 背景设置为中性灰或模糊；
- 主要判断“这个区域是什么”。

#### Context crop

- bbox 扩大 1.5～2 倍；
- 显示 mask 轮廓和稳定 UID；
- 主要判断“它和周围对象是什么关系”。

### 生成原则

- 只为 finding 生成，不为所有 observation 永久生成；
- 图片必须带 `obs_uid/object_uid/finding_uid`；
- 图片包必须自包含，离开原服务器目录仍能打开；
- 图像包是派生证据，不替代原始 RGB、mask、depth 和 PCD。

---

## 4. 不需要额外保存的内容

为避免证据系统无限膨胀，当前不建议：

- 每帧复制一份完整对象地图；
- 每帧保存所有对象对的图片 montage；
- 将点云、mask、feature 直接写入 JSON；
- 重复复制整个 Replica RGB-D 数据集；
- 将 Rerun viewer 作为唯一永久证据；
- 为所有 observation 立即调用 VLM；
- 在查错规则尚未验证前保存自动修复结果。

核心原则是：

> 建图阶段保存不可替代的原始证据和状态变化；能够离线计算的统计量、对比图和 VLM packet 按需生成。

---

## 5. 第一部分完成后的证据验收清单

补齐后，一个正式 run 必须满足：

```text
[ ] processed mask 全覆盖 kept observations
[ ] filter trace 与真实保留集合完全一致
[ ] artifact ref 文件、key、index、shape、hash 全部有效
[ ] observation PCD 明确完整/采样状态并可确定性重载
[ ] association 前后对象版本完整
[ ] postprocess merge 候选与拒绝原因可查询
[ ] source/target/after 三方状态完整
[ ] 不存在虚假的 text feature 引用
[ ] 每个语义分数标明真实来源与独立证据组
[ ] event、transaction、object version 因果链连通且无环
[ ] final membership 可由事件链重建
[ ] VLM 精确输入可恢复
[ ] 正式运行代码、环境、模型、seed 和输出 hash 完整
[ ] evidence strict validator 通过
```

---

# 第二部分：基于统一证据的第一阶段详细排查查错规则

## 1. 第一阶段检查范围

第一阶段只检查以下内容：

1. **证据和映射状态完整性错误**；
2. **同帧重复检测与重复 observation**；
3. **错误融合，即一个节点混入其他真实物体**；
4. **错误拆分，即一个真实物体保留成多个节点**；
5. **弱节点、噪声节点和残片节点**。

第一阶段暂不正式检查：

- 漏检对象；
- caption 事实错误；
- 关系边语义错误；
- 动态物体变化；
- 地图错误与真实世界变化的区分。

原因是这些问题需要额外负证据、时间维度或 VLM 推理。在节点身份尚未稳定前检查 caption 和边，容易把上游污染误判成下游模型错误。

---

## 2. 查错系统总体架构

```text
Evidence files
   │
   ▼
Evidence Readiness Gate
   │
   ├── schema / UID / artifact
   ├── matrix / event / membership
   └── object version chain
   │
   ▼
Fact Builder
   │
   ├── observation facts
   ├── association facts
   ├── object facts
   ├── object-pair facts
   └── temporal / co-visibility facts
   │
   ▼
Rule Engine
   │
   ├── hard invariants
   ├── duplicate observation
   ├── false merge
   ├── false split
   └── weak object
   │
   ▼
findings.jsonl
   │
   ▼
Case Builder
   │
   ├── structured metrics
   ├── 2D evidence
   ├── 3D evidence
   └── VLM-ready packet
```

第一版必须是**只读离线审计器**，不得修改 baseline map。

推荐代码结构：

```text
conceptgraph/audit/
├── evidence_resolver.py
├── validator.py
├── facts.py
├── metrics.py
├── runner.py
├── case_builder.py
└── rules/
    ├── evidence_integrity.py
    ├── mapping_invariants.py
    ├── duplicate_observation.py
    ├── false_merge.py
    ├── false_split.py
    └── weak_object.py
```

输出结构：

```text
<experiment>/audit/
├── validation.json
├── findings.jsonl
├── audit_summary.json
└── cases/
```

---

## 3. 统一指标定义

## 3.1 2D mask 指标

### Mask IoU

\[
IoU_{2D}(A,B)=\frac{|M_A\cap M_B|}{|M_A\cup M_B|}
\]

### Mask containment

\[
Contain_{2D}(A,B)=\frac{|M_A\cap M_B|}{\min(|M_A|,|M_B|)}
\]

`Contain2D` 更适合发现同一物体被不同类别重复检测的情况。

### Mask fill ratio

\[
Fill(M)=\frac{|M|}{Area(BBox(M))}
\]

### Boundary touch ratio

processed mask 位于图像边界的像素比例，用于判断截断观测。

---

## 3.2 3D 几何指标

### 单向点云支持

使用与 mapping 一致的最近邻半径 `r`：

\[
Overlap(A\rightarrow B)=
\frac{|
\{p\in P_A:\min_{q\in P_B}\|p-q\|<r\}
|}{|P_A|}
\]

### 对称重叠

\[
Overlap_{sym}(A,B)=
\min(Overlap(A\rightarrow B),Overlap(B\rightarrow A))
\]

使用 `min` 而不是 `max`，可避免一个小物体完全落在大物体中时被误判为同一实例。

### 归一化中心距离

\[
D_{center}(A,B)=
\frac{\|c_A-c_B\|_2}
{0.5(\|e_A\|_2+\|e_B\|_2)+\epsilon}
\]

其中 `c` 为 bbox center，`e` 为 bbox extent。

---

## 3.3 语义指标

### 图像特征相似度

\[
S_{img}(A,B)=\cos(f_A,f_B)
\]

### 成员 observation 语义支持

对对象成员 observation 选择 feature medoid：

\[
f_{medoid}=\arg\max_{f_i}\sum_{j\ne i}\cos(f_i,f_j)
\]

定义：

\[
SemSupport(o_i,O)=\cos(f_i,f_{medoid}(O\setminus o_i))
\]

使用 medoid 或 trimmed mean，不使用易被异常 observation 污染的普通平均。

### 类别主导比例

\[
DominantRatio(O)=\frac{\max_k count(class_k)}{N_O}
\]

类别差异只能作为风险证据，不能单独判错，因为 `couch/sofa/sofa chair` 等标签可能是同义或粒度差异。

---

## 3.4 关联决策指标

```text
top1 = s1
top2 = s2
margin = s1 - s2
threshold_slack = s1 - sim_threshold
```

其中：

- margin 小：候选之间难以区分；
- threshold slack 小：决策刚刚越过阈值；
- 两者都只能表示风险，不能单独证明错误。

---

## 3.5 对象变化指标

### Feature drift

\[
Drift_{feat}=1-\cos(f_{before},f_{after})
\]

### Bbox volume growth

\[
Growth_{vol}=\frac{V_{after}}{\max(V_{before},\epsilon)}
\]

### Center shift

\[
Shift_{center}=
\frac{\|c_{after}-c_{before}\|_2}
{\|e_{before}\|_2+\epsilon}
\]

这些指标用于判断一次融合是否造成异常“冲击”。

---

## 4. Finding 分级与分流

每条 finding 必须分为：

| 等级 | 含义 | 后续分流 |
|---|---|---|
| `CERTAIN` | 违反程序或数据不变量，确定错误 | `DIRECT_CODE_FIX` 或未来确定性修复 |
| `HIGH_CONFIDENCE` | 至少两类独立证据支持，无 veto | `DETERMINISTIC_REPAIR_CANDIDATE` |
| `AMBIGUOUS` | 有明显冲突，但证据不足 | `VLM_REVIEW` |
| `RISK_ONLY` | 仅风险信号，不能判错 | `LOG_ONLY` |

独立证据家族限定为：

```text
2D mask
3D geometry
image semantics
label/text semantics
association history
object state change
multi-view temporal evidence
```

同一个 CLIP feature 复制成 visual/text 两个分数，只能算一个家族。

---

## 5. Gate 0：Evidence Readiness 检查

只要 Gate 0 失败，就停止后续语义查错。否则查错结果本身不可信。

## EVI-001：Schema 与 manifest 完整性

### 条件

- schema version 不支持；
- manifest 缺少 run、scene、config、commit、status；
- mapping/detection config 不可解析；
- run 尚未正常 close。

### 输出

```text
certainty = CERTAIN
triage = DIRECT_CODE_FIX
```

---

## EVI-002：UID 唯一性

### 检查

- `frame_uid` 全局唯一；
- `obs_uid` 全局唯一；
- `event_uid` 全局唯一；
- `object_version_uid` 全局唯一；
- 同一 `object_uid + version` 不得重复。

任何重复均为确定性错误。

---

## EVI-003：Artifact 引用有效性

### 检查

对每个 ArtifactRef 验证：

```text
path exists
format supported
key exists
index in range
shape matches
dtype matches
sha256 matches
```

### 输出

列出具体文件、字段和失效类型，不只输出一个计数。

---

## EVI-004：Similarity matrix 完整性

### 检查

对于每帧：

```text
spatial.shape == (num_observations, num_objects_before)
visual.shape  == (num_observations, num_objects_before)
aggregate.shape == same
observation_uids == matrix row axis
object_uids_before == matrix column axis
```

同时检查 NaN、异常 Inf 和非法数值范围。

---

## EVI-005：Observation 生命周期完整性

每个 kept observation 必须满足以下之一：

```text
A. 最终属于一个 active object
B. 有明确 OBJECT_FILTER / DELETE / INVALID tombstone
```

不得：

- 无归属且无删除事件；
- 同时属于多个 active object；
- 同一个 active object 中重复出现。

---

## EVI-006：对象版本链完整性

### 检查

- version 从 1 连续递增；
- 每个版本只有一个 trigger event；
- parent version 存在；
- active object 只有一个 current version；
- merged/filtered object 不得无事件重新变 active。

---

## EVI-007：Final membership 可重建

从 `OBJECT_CREATE`、`OBS_ASSOCIATE`、`OBJECT_MERGE`、`OBJECT_FILTER` 事件重建最终成员关系。

必须满足：

```text
replayed_membership == final_membership
num_detections == member occurrence count
unique member count == occurrence count
```

第一阶段不允许一个 observation 被重复计数。

---

## EVI-008：VLM evidence 完整性

仅在 `make_edges=true` 时检查：

- prompt text 存在；
- 图片引用可解析；
- model 与生成参数存在；
- raw response 与 parsed output 对应；
- parser version 存在。

当前不基于 VLM 输出做地图纠错，但必须保证证据本身可靠。

---

## 6. Mapping Integrity 硬性不变量

这些规则完全不需要 VLM 或图片判断。

## MAP-001：Association 决策与矩阵不一致

### 重新计算

```python
best_idx = argmax(aggregate_row)
best_score = aggregate_row[best_idx]
```

### 正确逻辑

```text
best_score <= sim_threshold → CREATE_OBJECT
best_score > sim_threshold  → MERGE_TO_OBJECT(best_idx)
```

### 判错条件

- 记录的 decision 不符合阈值；
- target UID 不是 argmax 对应 UID；
- target 不在 `object_uids_before` 中；
- CREATE_OBJECT 却指向旧对象 UID。

### 级别

`CERTAIN`

---

## MAP-002：Top-K 与 margin 记录不一致

从完整矩阵重新计算：

- top1；
- top2；
- candidate order；
- margin。

任何不一致均为证据实现错误。

---

## MAP-003：重复 active ownership

### 判错条件

同一个 `obs_uid`：

- 出现在两个 active object；或
- 在同一 active object 中出现两次以上。

### 级别

`CERTAIN`

---

## MAP-004：`num_detections` 与成员数量不一致

### 正确条件

```text
num_detections == len(member_observation_uids)
len(member_observation_uids) == len(set(member_observation_uids))
```

当前第一阶段不接受重复 observation 权重。

---

## MAP-005：Merge source 被重复消费

### 判错条件

同一 `merge_transaction_uid` 中，同一个 source object 被 ACCEPT 两次以上。

### 级别

`CERTAIN`

### 当前回归案例

现有两帧 smoke 已经出现过同一个 source object 先合并到一个 target，随后又被再次作为 source 合并，最终造成两个 observation 重复计入 cabinet 节点。该案例应永久保留为 validator 回归测试。

---

## MAP-006：Merge 前成员集合已相交

### 判错条件

```text
set(source_members) ∩ set(target_members) != empty
```

这表示相同证据被重复融合，属于确定性错误。

---

## MAP-007：Merge graph 非法

### 检查

- source == target；
- source 在 merge 前已经 inactive；
- target 在 merge 前已经 inactive；
- merge lineage 出现环；
- source 消失但无 merge/filter tombstone。

任一情况均为 `CERTAIN`。

---

## MAP-008：Merge 后状态不是集合并集

必须满足：

```text
target_after.members
  == unique(source_before.members ∪ target_before.members)
```

同时：

```text
target_after.num_detections
  == unique member count
```

---

## MAP-009：Edge 指向失效对象

仅在有 edge 时检查：

- edge source/target UID 存在；
- 指向 active object；
- object merge/filter 后 edge 已正确迁移或删除；
- edge UID 与节点 UID、relation 一致。

---

## 7. 错误类别一：同帧重复检测

## 7.1 目标

检测同一帧中，一个真实物体被多个类别或多个框重复检测，导致：

```text
重复 observation
  → 重复对象初始化
  → 后续关联摇摆
  → 依赖 postprocess merge 补救
```

当前 smoke 中 `sofa chair` 与 `couch` 的 mask、bbox 和 3D 区域近乎相同，是标准案例。

---

## DUP-OBS-001：近乎完全重复 observation

### 冷启动保守条件

同一 frame 内 observation A、B 同时满足：

```text
Contain2D >= 0.95
IoU2D >= 0.85
S_img >= 0.95
且满足以下之一：
    Overlap3D_sym >= 0.80
    D_center <= 0.15
```

### 结果

```text
error_type = DUPLICATE_OBSERVATION
certainty = HIGH_CONFIDENCE
triage = DETERMINISTIC_REPAIR_CANDIDATE
```

### 注意

当前阶段只输出 candidate，不直接删除。

---

## DUP-OBS-002：几何完全重合但标签冲突

### 条件

- 2D/3D 高度重合；
- image feature 高相似；
- class name 差异较大或 label text compatibility 低。

### 结果

```text
certainty = AMBIGUOUS
triage = VLM_REVIEW
```

原因可能是：

- 同一物体的同义标签；
- 大类与细类差异；
- 容器与内部物体；
- detector 误标。

---

## DUP-OBS-003：同帧多个 observation 关联到同一 target

对每帧按 `target_object_uid` 分组。

### 分流

#### 若 observation 之间近乎完全重合

归入 `DUPLICATE_OBSERVATION`。

#### 若 observation 之间明显分离

归入后续 `FALSE_MERGE` 高风险。

---

## 重复检测自动判断 veto

以下情况禁止自动判为重复：

- mask 明显嵌套但对称 3D overlap 低；
- 一个 observation 是另一个的局部部件；
- 同帧中存在稳定不同中心；
- context crop 显示两个独立实例；
- label pair 具有明显 part-whole 可能性。

触发 veto 后转 `VLM_REVIEW`。

---

## 8. 错误类别二：错误融合 `False Merge`

## 8.1 目标

识别最终对象中混入其他真实物体 observation 的情况。

不能依赖单一指标。建议至少使用以下三类证据：

```text
成员一致性
关联历史
融合冲击 / 反事实
```

---

## FM-001：同一对象含有同帧、明显分离的成员

### 条件

同一个 final object 中存在同一 frame 的 observation A、B，且：

```text
IoU2D < 0.05
Overlap3D_sym < 0.10
D_center > 0.50
```

并且不是明确 part-whole 情况。

### 解释

同一帧中两个明显分离的可见实例被融合为一个 object，通常是错误融合。

### 结果

```text
certainty = HIGH_CONFIDENCE
triage = DETERMINISTIC_REPAIR_CANDIDATE
```

---

## FM-002：成员语义离群

### 计算

对每个成员 observation 计算：

```text
SemSupport(obs, object_without_obs)
```

同时计算 robust z-score：

\[
z_i=\frac{d_i-median(d)}{1.4826\cdot MAD(d)+\epsilon}
\]

其中 `d_i = 1 - SemSupport_i`。

### 冷启动条件

```text
SemSupport < 0.75
AND robust_z > 3.5
```

### 结果

单独触发时：

```text
certainty = RISK_ONLY
```

必须再结合几何、关联或反事实证据。

---

## FM-003：成员几何离群

### 计算

将 observation PCD 与对象剩余成员重建 PCD 比较：

```text
GeoSupport_forward = Overlap(obs → object_without_obs)
GeoSupport_reverse = Overlap(object_without_obs → obs)
```

### 冷启动条件

```text
GeoSupport_forward < 0.10
AND GeoSupport_reverse < 0.10
```

### 结果

单独触发为 `RISK_ONLY`。

---

## FM-004：类别组成异常

### 指标

- dominant class ratio；
- normalized class entropy；
- label compatibility matrix。

### 冷启动风险条件

```text
num_unique_classes >= 3
AND dominant_class_ratio < 0.60
AND incompatible_label_group_count >= 1
```

### 注意

`sofa/couch/sofa chair` 不能简单视为冲突。类别规则只做补充证据。

---

## FM-005：Fusion shock

一次 `OBS_ASSOCIATE` 后，若同时出现：

```text
feature_drift > 0.15
AND 至少一个几何冲击：
    bbox_volume_growth > 1.5
    center_shift > 0.30
    significant_second_3d_component = true
```

则标记：

```text
error_type = FUSION_SHOCK
certainty = AMBIGUOUS
triage = VLM_REVIEW
```

阈值必须后续在完整 room0 上校准。

---

## FM-006：低 margin + 语义/几何冲突

### 条件

```text
margin <= min(0.03, scene_margin_q05)
OR threshold_slack <= 0.05
```

同时：

```text
语义离群 OR 几何离群
```

### 结果

`AMBIGUOUS`，进入 VLM，而不是直接判错。

---

## FM-007：留一 observation 反事实关联

这是第一阶段最重要的高价值规则。

### 过程

对于可疑 observation `o_i`：

1. 从当前 target object 中暂时移除 `o_i`；
2. 使用其余成员重建 `target_without_i`；
3. 重新计算 `o_i` 对全部候选对象的 spatial、visual、aggregate score；
4. 比较原目标和最佳替代目标。

### 指标

```text
s_target_without_i
s_alt_best
counterfactual_gain = s_alt_best - s_target_without_i
```

### 高置信度条件

```text
s_target_without_i <= sim_threshold
AND s_alt_best > sim_threshold
AND counterfactual_gain >= 0.10
AND 至少一个独立证据支持：
    semantic outlier
    geometric outlier
    fusion shock
```

### 结果

```text
certainty = HIGH_CONFIDENCE
recommended_action = DETACH_AND_REASSOCIATE
triage = DETERMINISTIC_REPAIR_CANDIDATE
```

当前阶段只输出建议，不执行。

---

## False Merge 综合判定

### HIGH_CONFIDENCE

满足任一：

```text
FM-001
FM-007 + 一类独立支持证据
```

### AMBIGUOUS

满足：

```text
至少两类独立风险证据
但反事实结果不明确
```

### RISK_ONLY

只有：

```text
低 margin
低 dominant ratio
单一语义离群
单一几何离群
```

---

## 9. 错误类别三：错误拆分 / 重复节点 `False Split`

## 9.1 目标

识别一个真实物体长期保留成多个 active object 的情况。

对所有 active object pair 先做粗筛：

```text
bbox proximity
OR directional 3D overlap > low threshold
OR image feature similarity > high threshold
```

避免全量昂贵比较。

---

## FS-001：active object pair 近乎完全重合

### 冷启动条件

对象 A、B 满足：

```text
Overlap3D_sym >= 0.85
S_img >= 0.95
D_center <= 0.20
label_compatible = true
co_visible_separate_count == 0
```

### 结果

```text
error_type = FALSE_SPLIT
certainty = HIGH_CONFIDENCE
recommended_action = MERGE_OBJECTS
triage = DETERMINISTIC_REPAIR_CANDIDATE
```

---

## FS-002：互补时间段占据同一空间

### 条件

- 两对象的 3D 区域高度重合；
- observation 时间段互补；
- 从未在同一帧以两个分离 mask 同时出现；
- feature 和 label 高度兼容；
- 一个对象创建时存在接近阈值的旧对象候选。

### 结果

`AMBIGUOUS` 或 `HIGH_CONFIDENCE`，取决于反事实结果。

---

## FS-003：近阈值 CREATE 导致碎片化

### 条件

一次 `CREATE_OBJECT` 决策满足：

```text
0 <= sim_threshold - top1_score <= 0.05
```

且最终新对象与当时 top1 对象满足：

```text
high final overlap
high image similarity
no co-visible separation
```

### 结果

```text
error_type = NEAR_THRESHOLD_FRAGMENTATION
certainty = AMBIGUOUS
triage = VLM_REVIEW
```

---

## FS-004：后处理近阈值拒绝

从 `object_pair_decisions.jsonl` 找到：

- overlap、visual 或 text 中只有一个 gate 略低于阈值；
- 其他证据均高度一致；
- pair 最终仍同时 active；
- 无 co-visibility veto。

该类用于分析 merge threshold 是否过严或语义 gate 是否异常。

---

## FS-005：重复实例反事实合并收益

将对象 A、B 临时合并，在 sandbox 中计算：

```text
member semantic consistency change
3D compactness change
duplicate observation count change
class dominant ratio change
```

若合并后：

```text
semantic consistency ↑
geometry consistency 不下降
无同帧分离冲突
```

则提高 duplicate confidence。

当前阶段只计算候选收益，不提交地图。

---

## False Split 强 veto

满足任一条件，禁止自动合并：

```text
同帧分离共现 >= 2 次
且 processed masks 不重叠
且 3D centers 稳定分离
```

或：

```text
两节点分别拥有独立、多视角、稳定轨迹
```

或：

```text
明显 part-whole / container-content 关系
```

或：

```text
合并后几何出现双峰或大幅降低紧致度
```

触发 veto 后只能 `VLM_REVIEW` 或 `KEEP_SEPARATE`。

---

## 10. 错误类别四：弱节点、噪声节点和残片

## 10.1 原则

不能使用：

```text
num_detections == 1 → 删除
```

因为插座、开关、薄物体、小装饰物可能只出现一次。

弱节点必须由多项质量证据共同判断。

---

## WN-001：无效几何节点

### 硬性条件

- PCD 为空；
- 点坐标存在 NaN/Inf；
- bbox center/extent 非有限；
- extent 为负；
- object 无成员 observation；
- 所有 artifact ref 失效。

### 结果

`CERTAIN`。

---

## WN-002：单视角低质量节点

### 风险信号

```text
unique_frame_count == 1
confidence <= scene_q10
processed_mask_area <= scene_q10
valid_depth_ratio < 0.50
n_points <= scene_q10
boundary_touch_ratio > 0.50
denoise_keep_ratio < 0.30
```

### 判定

- 触发 1～2 项：`RISK_ONLY`；
- 触发至少 3 项：`AMBIGUOUS`；
- 再加无效几何或明显重复证据：`HIGH_CONFIDENCE`。

---

## WN-003：2D mask 碎片化

对 processed mask 做 connected components。

### 风险条件

```text
component_count >= 2
AND second_component_area / total_area >= 0.20
```

若对应 3D 点也形成明显分离 component，则提高风险。

该规则常对应：

- segmentation 泄漏；
- 背景混入；
- 过度包含多个物体。

---

## WN-004：深度不支持

### 条件

```text
valid_depth_ratio < 0.30
OR pcd_raw_valid_depth_points < min_points_threshold
```

如果 observation 仍被保留，则可能存在 evidence 或 mapping 实现不一致；若只是刚过阈值，则记为风险。

---

## WN-005：多视角几何不稳定

对于有多个 observation 的节点，计算：

- observation center dispersion；
- extent ratio dispersion；
- pairwise 3D support；
- view-to-view semantic consistency。

若：

```text
center dispersion 高
AND pairwise geometry support 低
AND semantic consistency 低
```

则更可能是错误融合，不应简单归为弱节点。该规则应自动转入 `FALSE_MERGE` 检查。

---

## 弱节点保留 veto

满足以下多数条件时，即使单视角也应保留：

```text
high confidence
high valid depth ratio
mask not truncated
PCD geometry finite and compact
distinct spatial location
crop visually clear
not duplicate with another node
```

---

## 11. Association 风险规则

这些规则不直接定义错误，只用于排序和触发更深检查。

## AR-001：低 margin

冷启动：

```text
margin <= min(0.03, scene_margin_q05)
```

输出 `RISK_ONLY`。

---

## AR-002：刚过阈值

```text
0 < threshold_slack <= 0.05
```

输出 `RISK_ONLY`。

---

## AR-003：语义与几何分数强冲突

示例冷启动条件：

```text
spatial >= 0.70 AND visual <= 0.60
```

或：

```text
visual >= 0.85 AND spatial <= 0.05
```

这可能表示：

- 同位置不同物体；
- 相似外观不同位置；
- 遮挡；
- 2D mask 泄漏；
- pose/depth 错误。

只进入 `VLM_REVIEW` 或 false merge/split 深查。

---

## AR-004：Top-1 与 Top-2 几乎并列

```text
margin <= 0.01
```

应自动将：

- observation；
- top1 object 当时版本；
- top2 object 当时版本；
- 2D/3D 对比；

打包为候选案例。

---

## AR-005：CREATE near threshold

```text
0 <= sim_threshold - top1_score <= 0.05
```

作为 false split 候选入口。

---

## 12. 规则组合与 VLM 分流

## 12.1 直接确定，不调用 VLM

以下类型无需 VLM：

- UID 重复；
- artifact 越界或丢失；
- association 与 argmax/threshold 不一致；
- observation 多 active ownership；
- duplicate membership；
- merge source 重复消费；
- merge member overlap；
- merge lineage 环；
- final membership 无法重放；
- 无效几何。

---

## 12.2 高置信度候选，暂不自动执行

满足：

- 至少两类独立证据支持；
- 没有 veto；
- 反事实检查支持；
- 操作对象和受影响范围明确。

输出：

```text
DETERMINISTIC_REPAIR_CANDIDATE
```

但第一阶段仍只记录，不修改地图。

---

## 12.3 有歧义时调用 VLM

典型情况：

- 几何相同但类别冲突；
- 高视觉相似但可能是两件同款物体；
- mask 嵌套，可能是部件与整体；
- 反事实 top1/top2 仍接近；
- 弱节点可能是真实小物体；
- 语义与几何证据方向相反。

VLM 只接收规则已经定位好的少量 Evidence Packet，不负责扫描全图。

---

## 13. `findings.jsonl` 统一格式

```json
{
  "schema_version": "0.1.0",
  "finding_uid": "finding_000012",
  "run_id": "...",

  "rule_ids": ["FM-002", "FM-003", "AR-001"],
  "error_type": "FALSE_MERGE_RISK",
  "certainty": "AMBIGUOUS",
  "severity": "HIGH",

  "scope": {
    "obs_uid": "...",
    "object_uid": "...",
    "object_version_uid": "...",
    "alternate_object_uids": ["..."]
  },

  "metrics": {
    "margin": 0.02,
    "threshold_slack": 0.03,
    "semantic_support": 0.61,
    "geometric_support": 0.08,
    "counterfactual_gain": 0.12
  },

  "independent_evidence_groups": [
    "association_history",
    "image_semantics",
    "3d_geometry"
  ],

  "vetoes_triggered": [],

  "evidence_refs": {
    "association_event_uid": "...",
    "processed_mask_ref": {},
    "observation_pcd_ref": {},
    "target_before_version": "...",
    "target_after_version": "...",
    "case_packet_ref": "audit/cases/finding_000012"
  },

  "triage": "VLM_REVIEW",
  "recommended_action": "DETACH_AND_REASSOCIATE",
  "action_executed": false
}
```

每条 finding 必须回答：

1. 哪个实体可疑；
2. 哪些规则触发；
3. 数值是多少；
4. 哪些证据是独立的；
5. 是否有 veto；
6. 应直接判断、交给 VLM，还是仅记录；
7. 后续可能执行什么操作。

---

## 14. 图片证据包规则

## 14.1 重复 observation

必须包含：

```text
原始 full frame
A/B mask 叠加图
A masked crop
B masked crop
A/B context crop
A/B 3D overlay
指标表
```

## 14.2 False merge

必须包含：

```text
可疑 observation crop
目标对象历史代表视图 3～5 张
目标对象冲突视图
top2 candidate 代表视图
observation vs target-without-observation 3D overlay
关联 top-k 分数表
反事实分数表
```

## 14.3 False split

必须包含：

```text
object A 代表视图
object B 代表视图
A/B 3D overlay
A/B observation timeline
同帧共现证据
postprocess candidate gate 表
```

## 14.4 弱节点

必须包含：

```text
全部有效视角，最多 4 张
full-frame 位置
masked crop
depth crop
point cloud render
质量指标表
```

---

## 15. 冷启动阈值与校准原则

本文给出的数值只作为第一版保守起点，不能直接写成最终论文阈值。

### 校准流程

1. 修复并补齐证据后，完整运行 Replica `room0`；
2. 先运行 Gate 0 和 hard invariants；
3. 每类规则导出 Top-100 candidate；
4. 人工标注 `correct / incorrect / ambiguous`；
5. 将 deterministic 分支阈值调到 precision 优先；
6. 使用 `room1`、`office0` 等场景验证泛化；
7. 最终锁定阈值并记录版本。

### 目标

确定性候选分支优先追求：

\[
Precision \ge 95\%
\]

歧义分支负责召回，不要求直接自动修正。

### 阈值版本

```json
{
  "rule_config_version": "audit-rules-v0.1",
  "calibration_scenes": ["room0"],
  "validation_scenes": ["room1", "office0"],
  "thresholds": {}
}
```

---

## 16. 必须建立的测试

## 16.1 当前 smoke 回归测试

当前两帧 smoke 应能够检测：

```text
MAP-005 MERGE_SOURCE_REUSED
MAP-003 DUPLICATE_MEMBERSHIP
MAP-004 NUM_DETECTIONS_MISMATCH
```

修复 merge 实现后，重新运行应全部归零。

---

## 16.2 合成单元测试

至少构造：

1. 正常 association；
2. argmax target 写错；
3. threshold decision 写错；
4. observation 重复 ownership；
5. merge source 重复消费；
6. 同帧近乎完全重复 mask；
7. 同一对象混入一个远距离 observation；
8. 一个对象拆成两个近乎重合节点；
9. 单视角高质量小物体；
10. 单视角低质量噪声物体。

---

## 16.3 证据开关透明性

保持当前测试：

```text
save_evidence=true/false
最终 object JSON hash 相同
最终 edge JSON hash 相同
```

同时新增：

- strict/best-effort 模式；
- ArtifactRef selector 校验；
- object version chain；
- processed mask；
-真实 `make_edges=true` VLM smoke。

---

## 17. 第一阶段完成标准

第一阶段结束时，系统应能够对任意 finding 生成：

```text
错误类型
可疑对象/观测
触发规则
关键数值
原始证据引用
对象历史版本
图片证据包
确定性/VLM 分流结果
建议动作
```

同时必须满足：

```text
[ ] 查错器不修改 baseline map
[ ] Evidence Gate 失败时停止语义判断
[ ] 硬性不变量能够发现当前 merge 重复消费问题
[ ] 四类错误均有独立 rule module
[ ] 每条 finding 都可追溯到原始 observation
[ ] 每条 HIGH_CONFIDENCE finding 至少有两类独立证据
[ ] 所有自动候选均检查 veto
[ ] VLM 只处理 AMBIGUOUS finding
[ ] 阈值有版本、有校准集和验证集
[ ] 完整 room0 可批量输出 audit report
```

---

## 18. 推荐实施顺序

```text
Step 1  保存 processed mask 和真实 filter trace
Step 2  实现 ArtifactRef 与 strict validator
Step 3  增加 object_versions 和 association before/after
Step 4  增加 postprocess candidate/merge transaction 记录
Step 5  修正 text evidence 语义与空引用
Step 6  用当前 smoke 实现 MAP hard invariants
Step 7  修复 merge source 重复消费并做回归测试
Step 8  实现 duplicate observation 规则
Step 9  实现 false merge 规则与 leave-one-out 反事实
Step 10 实现 false split 和 weak object 规则
Step 11 按 finding 生成自包含图片证据包
Step 12 完整 room0 人工标注并校准阈值
```

这一阶段完成后，再进入下一层：

```text
CERTAIN / HIGH_CONFIDENCE
  → 后续确定性局部修正候选

AMBIGUOUS
  → VLM 多模态判断

RISK_ONLY
  → 保留日志，等待更多观测或后续任务触发
```

当前最重要的不是立刻“让系统自己改图”，而是先保证：

> 每一个错误判断都有可靠证据、明确规则、可复算数值和可定位的历史决策来源。
