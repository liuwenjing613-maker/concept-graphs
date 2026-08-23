# ConceptGraphs `ali-dev` 第一阶段统一证据留存设计

> 适用范围：基于 `ali-dev` 分支的 `rerun_realtime_mapping.py`，先建立统一、可查询、可追踪的证据留存体系。  
> 当前目标：方便后续查看“某个节点由哪些观测形成、某次关联为什么发生、错误可能在哪一步产生”。  
> 当前不做：自动错误检测、VLM 自动纠错、回滚执行、分支地图、完整历史重放。

---

## 1. 设计目标

统一证据留存系统只需要解决四个问题：

1. **一次实验是怎么运行的**：使用了什么数据、配置、模型和代码版本。
2. **每个观测是什么**：来自哪一帧、哪个原始检测，包含什么 2D、3D 和语义证据。
3. **每次关联为什么发生**：候选对象有哪些，各自分数是多少，最终创建新节点还是融合到已有节点。
4. **最终节点是怎么形成的**：由哪些观测组成，中间是否经历过节点合并、过滤、去噪和关系更新。

统一证据链应当满足：

```text
实验 Run
  ↓
帧 Frame
  ↓
观测 Observation
  ↓
关联决策 Association
  ↓
对象节点 Object
  ↓
描述与关系 VLM / Edge
```

所有记录围绕这条链组织，不保存与当前研究无关的信息。

---

## 2. `ali-dev` 当前已经保存了什么

`ali-dev` 已经具备较好的实验输出基础，现有内容应直接复用，不要重复保存。

| 现有内容 | 当前保存情况 | 作用 | 是否足以支持追溯 |
|---|---|---|---|
| 检测实验配置 | `config_params_detections.json` | 记录检测相关 Hydra 配置 | 基本足够 |
| 建图实验配置 | `config_params.json` | 记录建图相关 Hydra 配置 | 基本足够 |
| 每帧原始检测 | `detections/<frame>/` 下的 `.npz` 和 `.pkl.gz` | 保存框、置信度、类别、mask、crop、特征、caption、2D edge 等 | 证据较完整，但缺统一观测 ID |
| 检测可视化 | `vis/` 中的 RGB、depth、VLM 标注图 | 人工检查检测、mask、caption 和关系 | 可直接复用 |
| 最终对象地图 | `pcd_<exp_suffix>.pkl.gz` | 保存最终对象、配置、类别、颜色及边 | 保存了最终状态，但缺关联决策过程 |
| 最终节点 JSON | `obj_json_<exp_suffix>.json` | 快速查看节点标签、caption、bbox | 仅是结果摘要 |
| 最终关系 JSON | `edge_json_<exp_suffix>.json` | 快速查看关系边及累计次数 | 仅是结果摘要 |
| Rerun 可视化 | `.rrd` 或 Viewer 中的时间序列 | 查看相机、点云、对象和关系随时间变化 | 适合观察，不适合作为唯一证据源 |
| 每帧对象快照 | `saved_obj_all_frames/` | 查看每帧对象地图 | 当前会删掉大部分观测来源字段，只适合可视化 |

### 当前真正缺少的内容

`ali-dev` 已经保存了“检测结果”和“最终地图”，但没有统一保存以下关键信息：

- 原始检测经过过滤后，对应关系是否仍然可追踪；
- 每个有效观测的永久 ID；
- 关联时全部候选对象及其空间、视觉、总相似度；
- 新建节点或融合节点的明确决策记录；
- 节点合并、过滤、去噪导致的对象身份变化；
- 最终对象与所有来源观测之间的稳定成员关系；
- VLM 调用所使用的模型、提示词版本、输入证据和原始输出；
- 关系边从哪一帧、哪两个 2D 观测映射而来。

因此当前需要补充的不是另一份“大而全的日志”，而是一套连接现有产物的**轻量证据索引**。

---

## 3. 统一身份设计

后续所有文件必须使用稳定 ID 互相引用，不能只依赖数组下标。

| ID | 建议格式 | 含义 |
|---|---|---|
| `run_id` | `room0_20260817_101530` | 一次完整实验 |
| `frame_uid` | `<run_id>_f000120` | 一次被处理的输入帧 |
| `obs_uid` | `<run_id>_f000120_r0007` | 第 120 帧的第 7 个原始检测 |
| `object_uid` | 将 `ali-dev` 当前对象 `id` 的 UUID 转成字符串 | 一个地图节点的稳定身份 |
| `event_uid` | `<run_id>_e00008421` | 一次关联、合并、过滤或关系更新事件 |

### 必须遵守的规则

1. `object index` 只表示对象在当前列表中的位置，不能作为永久身份。
2. `curr_obj_num` 可以继续用于界面显示，但不能作为唯一追溯键。
3. `obs_uid` 必须在 `filter_gobs()` 之前创建，确保被过滤的检测也能找到原始位置。
4. 对象融合时保留目标对象的 `object_uid`，同时记录被融合观测的 `obs_uid`。
5. 对象间后处理合并时，必须同时记录源对象 UID 和目标对象 UID。

---

## 4. 推荐目录结构

在每次建图实验目录下新增一个 `evidence/` 文件夹，其他 `ali-dev` 原始输出保持不变。

```text
<scene>/exps/<mapping_exp_suffix>/
├── config_params.json
├── config_params_detections.json
├── pcd_<exp_suffix>.pkl.gz
├── obj_json_<exp_suffix>.json
├── edge_json_<exp_suffix>.json
├── saved_obj_all_frames/
├── evidence/
│   ├── manifest.json
│   ├── frames.jsonl
│   ├── observations.jsonl
│   ├── associations.jsonl
│   ├── mapping_events.jsonl
│   ├── vlm_events.jsonl
│   ├── final_membership.json
│   ├── similarities/
│   │   └── frame_000120.npz
│   ├── observation_pcd/
│   │   └── <obs_uid>.npz
│   └── evidence_summary.json
└── rerun_<exp_suffix>.rrd
```

设计原则：

- JSON/JSONL 只保存标识、数值、状态和文件引用；
- mask、特征、相似度矩阵、点云等大数组使用现有文件或压缩 NPZ；
- 原始 RGB、depth、检测结果不重复复制，只记录相对路径；
- 每处理完一帧立即追加写入，程序中断时已有记录仍然有效。

---

## 5. 必须统一留存的证据

## 5.1 `manifest.json`：实验总信息

一场实验只保存一份。

### 必须字段

| 字段 | 内容 |
|---|---|
| `schema_version` | 证据格式版本，例如 `0.1` |
| `run_id` | 本次实验唯一 ID |
| `scene_id`、`dataset` | 场景与数据集名称 |
| `branch`、`git_commit` | `ali-dev` 及具体代码版本 |
| `start_time`、`end_time` | 实验开始和结束时间 |
| `status` | `running`、`completed`、`early_exit`、`failed` |
| `mapping_config_ref` | 指向 `config_params.json` |
| `detection_config_ref` | 指向 `config_params_detections.json` |
| `detection_exp_suffix` | 使用的检测实验 |
| `mapping_exp_suffix` | 当前建图实验 |
| `model_versions` | 检测、SAM、CLIP、VLM 模型名称或权重版本 |
| `prompt_versions` | caption、edge、caption consolidation 的提示词版本 |

### 目的

保证以后看到任何一个错误案例时，能够确定它属于哪次实验，以及当时使用了什么配置和模型。

---

## 5.2 `frames.jsonl`：帧级输入索引

每处理一帧写一行。

### 必须字段

| 字段 | 内容 |
|---|---|
| `frame_uid`、`frame_idx` | 稳定帧 ID 与循环索引 |
| `source_frame_id` | 原始文件名或数据集中的真实帧编号 |
| `rgb_path`、`depth_path` | RGB 和深度文件相对路径 |
| `pose` | 当前使用的 4×4 相机位姿 |
| `intrinsics` | 当前相机内参 |
| `processed` | 是否进入检测和建图流程 |
| `skip_reason` | 无检测、提前退出、数据无效等原因 |
| `num_raw_detections` | 原始检测数量 |
| `num_kept_observations` | 过滤后进入 3D 建图的观测数量 |

### 目的

将数据集文件、检测结果、相机位姿和后续对象观测统一到同一帧上。

---

## 5.3 `observations.jsonl`：观测证据表

每一个原始检测写一行，包括后来被过滤的检测。

### 必须字段

| 类别 | 字段 | 内容 |
|---|---|---|
| 身份 | `obs_uid`、`frame_uid` | 永久观测 ID 与来源帧 |
| 索引 | `raw_det_idx`、`filtered_det_idx` | 过滤前、过滤后的检测位置 |
| 过滤 | `status`、`filter_reason` | `kept` 或 `rejected`，以及具体原因 |
| 2D 几何 | `bbox_2d`、`mask_area` | 检测框与 mask 面积 |
| 视觉 | `confidence`、`class_id`、`class_name` | 检测模型输出 |
| 文件引用 | `mask_ref`、`crop_ref` | 指向已有检测保存目录 |
| 特征引用 | `image_feat_ref`、`text_feat_ref` | 指向现有特征文件及数组位置 |
| 3D 几何 | `pcd_ref`、`n_points`、`bbox_3d_center`、`bbox_3d_extent` | 该观测独立形成的 3D 证据 |
| 语义 | `raw_caption`、`detection_label` | 当前帧对应的 VLM 描述和显示标签 |

### 关键建议

- 在原始检测结果中新增 `raw_det_idx`，再执行 `filter_gobs()`；
- `mask_idx` 继续保留，但只表示过滤后的局部索引；
- 对进入建图的观测保存一份压缩、下采样的独立 3D 点云；
- 被过滤观测无需保存 3D 点云，但必须保存其过滤原因和 2D 证据引用。

### 目的

以后可以直接回答：

- 一个节点里的某次观测来自哪一帧；
- 原始检测是否合理；
- 它是否因为过滤规则被丢弃；
- 2D mask、3D 点云或语义描述究竟从哪一步开始异常。

---

## 5.4 `associations.jsonl`：观测到对象的关联证据

每个进入 3D 建图的观测保存一条关联记录。

### 必须字段

| 字段 | 内容 |
|---|---|
| `event_uid`、`frame_uid`、`obs_uid` | 本次关联事件及来源 |
| `object_uids_before` | 关联计算时地图中对象 UID 的顺序 |
| `spatial_sim_ref` | 该帧空间相似度矩阵引用 |
| `visual_sim_ref` | 该帧视觉相似度矩阵引用 |
| `aggregate_sim_ref` | 该帧总相似度矩阵引用 |
| `top_candidates` | 至少保存 Top-3 的对象 UID 和三类分数 |
| `top1_score`、`top2_score`、`margin` | 最优、次优与差值 |
| `sim_threshold`、`match_method`、`phys_bias` | 当时使用的决策参数 |
| `decision` | `CREATE_OBJECT` 或 `MERGE_TO_OBJECT` |
| `target_object_uid` | 最终对象 UID；新建时为新节点 UID |

### 相似度矩阵保存方式

每帧统一保存：

```text
similarities/frame_000120.npz
```

其中至少包含：

```text
observation_uids
object_uids
spatial_sim
visual_sim
aggregate_sim
```

### 目的

以后不需要重新运行检测和 CLIP，就能分析：

- 该次融合是否接近阈值；
- Top-1 与 Top-2 是否过于接近；
- 空间相似度和视觉相似度是否互相冲突；
- 某个错误是在线关联造成，还是后续对象合并造成。

---

## 5.5 `mapping_events.jsonl`：对象与关系生命周期

只记录真正改变对象集合、对象成员或关系图的事件。

### 当前需要的事件类型

| 事件 | 需要记录的内容 |
|---|---|
| `OBJECT_CREATE` | 新对象 UID、来源 `obs_uid`、创建帧 |
| `OBS_ASSOCIATE` | `obs_uid`、目标对象 UID、关联事件 ID |
| `OBJECT_MERGE` | 源对象 UID、目标对象 UID、overlap、视觉相似度、发生帧 |
| `OBJECT_FILTER` | 被删除对象 UID、成员观测、点数、检测次数、删除原因 |
| `OBJECT_DENOISE` | 对象 UID、处理前后点数和 bbox 摘要 |
| `EDGE_ADD` | 两端对象 UID、关系、来源帧与来源 2D 观测 |
| `EDGE_UPDATE` | 关系累计次数变化、当前关系类型 |
| `EDGE_DELETE` | 两端对象 UID、删除原因 |

### 每条事件的公共字段

```text
event_uid
frame_uid
event_type
object_uid / source_object_uid / target_object_uid
before_summary
after_summary
reason
```

### 目的

最终对象列表经过 `filter_objects()` 和 `merge_objects()` 后，数组索引会改变。事件表用于保留这些变化，使最终节点仍然能追踪到原来的对象和观测。

---

## 5.6 `vlm_events.jsonl`：描述与关系推理证据

`ali-dev` 会从帧级标注图生成 caption 和关系，最后还会合并多个 caption。因此每次 VLM 调用必须单独记录。

### 当前需要的调用类型

1. `FRAME_CAPTION`
2. `FRAME_EDGE`
3. `OBJECT_CAPTION_CONSOLIDATION`

### 必须字段

| 字段 | 内容 |
|---|---|
| `event_uid`、`frame_uid` 或 `object_uid` | 调用对应的帧或对象 |
| `model_name`、`model_version` | 实际使用的 VLM/LLM |
| `prompt_version` | 提示词版本，不必每行重复完整提示词 |
| `input_image_ref` | 输入给 VLM 的标注图 |
| `input_labels` | 图中使用的检测编号和类别 |
| `input_observation_uids` | 输入图里的观测 UID |
| `input_captions` | 合并对象 caption 时使用的原始 caption 列表 |
| `raw_response` | 模型原始返回结果 |
| `parsed_output` | 程序解析后的 caption 或 edge |
| `latency_ms`、`status`、`error` | 调用耗时与异常情况 |

### 目的

以后可以区分：

- 检测本身错了；
- 标注图把背景或邻近物体带入了输入；
- VLM 原始输出错误；
- 模型输出正确，但解析或对象映射错误；
- 多视角 caption 合并阶段产生了错误概括。

---

## 5.7 `final_membership.json`：最终节点成员索引

实验结束后生成一份面向人工查看的最终索引。

每个最终节点至少保存：

```text
object_uid
current_object_index
curr_obj_num
status
class_name
class_histogram
member_observation_uids
num_detections
bbox_center
bbox_extent
n_points
consolidated_caption
parent_or_merged_from_object_uids
outgoing_edge_uids
incoming_edge_uids
```

### 目的

点击任意最终节点时，可以立即看到：

- 它由哪些帧、哪些检测组成；
- 是否由其他节点后处理合并而来；
- 哪次观测最早创建它；
- 哪些 caption 和关系依赖这个节点。

---

## 5.8 `evidence_summary.json`：完整性检查结果

实验结束后自动生成，至少包括：

```text
num_frames
num_raw_detections
num_kept_observations
num_rejected_observations
num_create_decisions
num_associate_decisions
num_object_merges
num_filtered_objects
num_final_objects
num_vlm_calls
num_edges
missing_reference_count
logging_errors
```

它不是论文评估指标，只用于确认本次证据是否保存完整。

---

## 6. 现有 `ali-dev` 产物如何处理

| 现有产物 | 处理方式 |
|---|---|
| 原始 RGB、depth、pose | 不复制，在 `frames.jsonl` 中记录相对路径与实际使用的 pose、intrinsics |
| 每帧检测目录 | 原样保留，`observations.jsonl` 通过路径和数组索引引用 |
| 检测可视化图 | 原样保留，作为人工证据和 VLM 输入引用 |
| `config_params*.json` | 原样保留，由 `manifest.json` 引用 |
| 最终 `pcd_*.pkl.gz` | 原样保留，作为最终完整地图状态 |
| `obj_json`、`edge_json` | 原样保留，用于快速浏览结果 |
| Rerun 文件 | 推荐保留，用于时间轴可视化，但不承担唯一证据功能 |
| `saved_obj_all_frames` | 可继续用于可视化，不作为对象来源和关联决策的正式记录 |

---

## 7. 建议的代码插入位置

只需要在现有流程的六个位置增加记录，不改变原有算法。

| 位置 | 需要记录什么 |
|---|---|
| 读取帧和原始检测之后 | `frames.jsonl`，给原始检测分配 `raw_det_idx` 和 `obs_uid` |
| `filter_gobs()` 之后 | 每个检测的保留或过滤结果及原因 |
| 独立 3D 观测生成之后 | 观测点云、3D bbox、点数和特征引用 |
| 三类相似度计算之后、匹配之前 | 完整相似度矩阵、Top-K 候选、阈值和关联决策 |
| `merge_obj_matches()`、`filter_objects()`、`merge_objects()`、`process_edges()` 前后 | 对象和关系生命周期事件 |
| VLM 调用及最终 caption consolidation 前后 | 模型、提示词版本、输入证据、原始输出和解析结果 |

最重要的原则是：

> **先记录决策依据，再执行会改变地图的操作。**

---

## 8. 保存格式与工程规则

### 8.1 文件格式

- `JSON`：实验总信息、最终成员索引、汇总统计；
- `JSONL`：帧、观测、关联和生命周期事件；
- `NPZ`：相似度矩阵、观测点云、mask 和高维特征；
- `PKL.GZ`：继续保存 `ali-dev` 原有最终地图和可视化快照；
- `RRD`：Rerun 可视化时间线。

### 8.2 写入规则

1. 所有路径优先保存为相对于实验目录的相对路径。
2. 每帧处理完成后立即 `flush`，避免异常退出导致整场日志丢失。
3. `manifest.json` 开始时写入 `running`，结束时更新为最终状态。
4. 所有 JSONL 记录必须包含 `run_id` 和 `schema_version`。
5. 大数组不得直接写进 JSON。
6. 证据记录开关开启后，最终地图结果必须与原始 `ali-dev` 保持一致。
7. 日志代码异常不能静默改变关联、过滤或融合结果。

---

## 9. 当前阶段必须做与暂时不做

## 9.1 当前必须做

- 稳定的 `run_id`、`frame_uid`、`obs_uid`、`object_uid`；
- 原始检测到过滤后观测的对应关系；
- 每个有效观测的 2D、3D、语义证据引用；
- 每次关联的完整候选分数和最终选择；
- 对象创建、观测融合、对象合并、过滤和去噪事件；
- VLM caption、关系和最终 caption 合并的输入输出；
- 最终对象到全部来源观测的成员索引；
- 一份证据完整性汇总。

## 9.2 当前暂时不做

- 自动判断哪次关联错误；
- 人工或 VLM 触发的 detach、split、merge 修复；
- 地图版本分支和回滚提交；
- 每帧保存一份完整对象地图；
- 从任意历史帧重新播放整个流程；
- 任务相关错误优先级和下游导航评估；
- 动态环境中的出现、消失和移动状态机。

这些内容以后都可以基于当前证据层增加，不需要现在把系统做得像一个小型数据库公司。

---

## 10. 最小验收标准

完成后应通过以下检查：

1. 开启或关闭证据留存，最终对象数量、对象点云和关联结果一致。
2. 每个原始检测都有唯一 `obs_uid`，包括被过滤的检测。
3. 每个进入建图的观测都有且只有一条关联决策记录。
4. 每个关联记录中的候选对象顺序与相似度矩阵列顺序一致。
5. 每个最终对象都能解析出完整的 `member_observation_uids`。
6. 每次对象后处理合并和删除都能在 `mapping_events.jsonl` 中找到。
7. 每个最终 caption 能追踪到原始 caption、输入观测和 VLM 调用记录。
8. 每条最终关系边能追踪到两端对象 UID及其来源帧。
9. `evidence_summary.json` 中 `missing_reference_count = 0`。
10. 程序提前退出时，已处理帧的日志仍可正常读取。

---

## 11. 推荐实现顺序

### 第一步：统一身份和实验信息

实现 `manifest.json`、`frames.jsonl`，增加 `raw_det_idx`、`obs_uid`，将当前对象 UUID 统一序列化为 `object_uid`。

### 第二步：完成核心建图证据

实现 `observations.jsonl`、每帧相似度矩阵和 `associations.jsonl`。完成后，应能查看“任意观测当时为什么被分给某个对象”。

### 第三步：补齐生命周期与 VLM 证据

实现 `mapping_events.jsonl`、`vlm_events.jsonl`、`final_membership.json` 和 `evidence_summary.json`。完成后，应能查看“任意最终节点是怎样逐步形成的”。

当前阶段做到第三步即可停止，不需要立即实现修复。

---

## 12. 最终应达到的查看效果

选择任意最终节点，系统能够展示：

```text
Object UID
├── 当前类别、caption、bbox、点数
├── 所有来源 Observation
│   ├── RGB / depth / mask / crop
│   ├── 独立 3D 点云
│   ├── 检测类别、置信度、原始 caption
│   └── 来源帧、相机位姿
├── 每次 Association
│   ├── 空间相似度
│   ├── 视觉相似度
│   ├── 总相似度
│   ├── Top-K 候选
│   └── 最终决策
├── 对象生命周期
│   ├── 创建
│   ├── 观测融合
│   ├── 对象合并
│   ├── 去噪
│   └── 过滤
└── VLM 与关系证据
    ├── 原始 caption
    ├── consolidated caption
    ├── 关系边来源帧
    └── 模型、提示词版本和原始输出
```

达到这一状态后，后续无论做人工错误标注、VLM 诊断、局部重建还是动态更新，都不需要重新设计底层证据格式。

---

## 13. 核心结论

第一阶段统一保存的重点不是“把所有中间变量都存下来”，而是保存以下五类不可替代证据：

```text
输入证据
+ 观测身份
+ 关联依据
+ 对象成员与生命周期
+ VLM 输入输出
```

`ali-dev` 现有检测文件、最终点云、JSON 和 Rerun 可视化继续保留；新增的 `evidence/` 只负责把这些分散产物用稳定 ID 和事件记录串成一条完整证据链。
