# `ali-my` 统一证据留存 + 分层查错器：完整方法理解文档

> **用途**：这不是“文件说明书”，也不是“规则清单”。
> 它的目标是让你真正理解目前 `ali-my` 已经做成了什么、为什么这样设计、每份证据在解决什么问题、查错器到底如何判断、哪些结论能说到什么程度，以及未来如果把这一部分写进论文，应当怎样把代码实现抽象成一个清晰的方法。
>
> **当前代码基准**：`ali-my`，Evidence schema `0.2.0`，基础 Layered Audit schema `1.1.0`；当前人工有效性门使用 Final Endpoint Validation config `2.1.0`、Review Evidence schema `2.1.0` 与 endpoint metrics `2.1.0`。
> **当前可复核的完整实测产物**：2026-08-20 已在同一冻结配置上完成 Replica `room0`、`office0` 两个 200 帧正式运行，并在不重跑检测/建图的前提下完成 final-endpoint 级重新审计。5587 条 checker findings 最终归并为 98 个不同的 final-object endpoints；1 个证据阻断，剩余 97 个全部进入普查，而不是再抽取 160 条重复案例。2026-08-21，R1 已完成 97/97：55 个最终正确、40 个确认仍错、2 个证据不足；正式决策是进入 40 个确认错误的专家因果追踪，不是直接宣称已经修复。隐藏 R1 答案的 24 例 R2 也已完成：最终状态一致率 83.33%，Cohen’s κ=0.706；它是同一复核者的重复稳定性，不是独立评审者一致性。完整结果见第 35 节。
>
> **最重要的一句话**：
>
> **我们做的不是“多记一些日志 + 写一堆阈值规则”，而是把原本不可追溯的增量 3D 建图过程，改造成一个“可回放、可验证、可定位风险、可交给人/VLM复核”的证据化状态演化系统。**

> **2026-08-20 人工协议纠偏**：逐条 finding 的复杂 R1 在 0/160 时停止。第一次改成“同一触发 observations + 同一谱系”的 incident 后，我们又主动检查了人工重复度，发现其中 147/160 仍落在重复的 final-object 集合上，同一个 object 最多会判断 11 次，因此也没有投入人工。最终 `v2.1` 直接以“完全相同的 active final-object set”为 R1 单位；当前两场景的每个 endpoint 都是一个不同的 final object，没有重复。人只判断自己看得到的最终状态；阶段根因留给已确认错误后的专家追踪，修复必须由 replay 验证。详细实测数字、操作方法与服务器入口见第 34 节。

---

# 0. 先只建立一个总的脑图

如果后面所有细节你一时记不住，只要先记住下面这条链：

```text
ConceptGraphs 原始增量建图
        │ 每一步旁路记录
        ▼
① Unified Evidence Ledger
        │
        ▼
② Evidence Gate：先证明证据链可信
        │ PASS
        ▼
③ Layered Screeners：DET→SEG→GEO→ASSOC→FUSE→OBJ
        │
        ▼
④ Findings：facts / hypotheses / veto / missing evidence
        │
        │ 所有阶段/observation 报警按 active final-object set 聚合
        ▼
⑤ Final Endpoint Builder
   ├─ 同一最终对象只形成一个 R1 单元
   ├─ 多 checker / 多 observation 报警只保留为内部证据
   └─ 没有可复核终态且缺阶段忠实快照 → 阻断，不给人猜
        │
        ▼
⑥ Endpoint Evidence Packet
   exact final map + 完整成员 + 代表视图 + 同源上下文
        │ 哈希锁定并核对人/系统看到的是同一对象状态
        ▼
⑦ R1 人工最终状态复核
   evidence sufficient? → final CORRECT / WRONG / UNCLEAR
        │
        ├─ CORRECT / UNCLEAR：不做无意义根因和修复猜测
        │
        └─ WRONG
             ▼
   专家因果追踪 → intervention / replay → 修复验证
```

目前真正已经实现的是 **①到⑦**、完整 R1 指标、筛查排序诊断、24 例 R2 重复稳定性评估，以及“只把确认的最终错误生成专家队列”的出口。R1 已完成 97/97，并生成 40 例专家队列；R2 已完成 24/24，另有 5 例分歧单独进入待裁决队列，没有污染 40 例确认错误。

目前**还没有**实现的是最后一段：

```text
确认错误后的完整专家因果结论
→ 实际回滚/重关联/拆分/合并
→ 对照 replay
→ 证明最终对象图确实改善
```

这个边界一定要分清。否则很容易把“已经能发现风险”说成“已经能自动纠错”，论文审稿人看到这种跨越式叙事通常会迅速恢复清醒。

---

# 1. 为什么需要这一套东西：先从 ConceptGraphs 原始流程说起

## 1.1 原始 ConceptGraphs 实际在做什么

ConceptGraphs 的对象建图核心可以简化成：

```text
第 t 帧 RGB-D
   ↓
2D instance proposals
   ↓
每个 proposal 提取：
  mask
  CLIP/DINO 语义特征
  3D point cloud
   ↓
与当前地图中的每个 object 比较
   ↓
几何相似度 + 语义相似度
   ↓
greedy association
   ├─ 分数够高 → 融入已有 object
   └─ 分数不够 → 创建新 object
   ↓
周期性 denoise / filter / merge
   ↓
最终 object map
   ↓
caption + relation
   ↓
3D Scene Graph
```

ConceptGraphs 论文中的关联，本质上是：

\[
\phi(i,j)=\phi_{\text{geo}}(i,j)+\phi_{\text{sem}}(i,j)
\]

其中：

- \(\phi_{\text{geo}}\)：新 observation 与已有 object 的 3D 几何重叠；
- \(\phi_{\text{sem}}\)：视觉语义特征相似度；
- 找到综合分数最大的 object；
- 如果最大分数大于阈值，就关联并融合；
- 否则创建一个新 object。

因此每一帧都在做一个不可逆倾向很强的决策：

```text
“这个新看到的东西，到底是不是地图里已经存在的那个东西？”
```

如果这一步错了，后面可能出现：

```text
错误 observation
   ↓
融入错误 object
   ↓
object 几何/语义被污染
   ↓
后续新 observation 又拿这个已污染 object 做匹配
   ↓
错误继续传播
   ↓
最后形成：
  false merge
  duplicate object
  identity instability
  错误 caption
  错误 relation
```

ConceptGraphs 自己也报告了小/细物体漏检、duplicate detections、node caption 错误，并指出关键错误会影响后续规划。

---

## 1.2 原始输出为什么不足以“查错”

假设最终地图中出现：

```text
Object 17:
  看起来像 sofa
  点云形状很怪
  包含 11 个 observations
```

仅看最终 object，你不知道：

1. 这 11 个 observation 分别来自哪一帧？
2. 每一个 observation 当时原始 mask 是什么？
3. 它有没有经历 containment subtraction？
4. 2D mask 投影成 3D 后是否出现多个 cluster？
5. 它当时有哪些候选 object？
6. spatial score、visual score、aggregate score 分别是多少？
7. 第一名和第二名只差 0.001，还是差 0.8？
8. 它为什么被分给 Object 17？
9. Object 17 在它加入之前是什么样？
10. 加入以后 bbox、point count、class composition 发生了什么？
11. Object 17 后面是否又和另一个 object merge？
12. 最终的异常到底源于 segmentation、association，还是后处理 merge？

这就是原始系统最致命的研究障碍：

> **最终状态能告诉你“现在错了”，却很难告诉你“错误是怎么形成的”。**

所以我们第一步没有直接做 VLM 纠错，而是先把整个过程变成可追溯的。

---

# 2. 目前方法真正的核心思想

可以把当前方法概括成两个大模块：

## 模块 A：Unified Evidence Ledger

目的：

> **把每一次影响地图状态的观测、比较、决策和状态变化，都以统一 UID 串起来。**

它解决的是：

```text
“我能不能从最终错误节点，一路倒着找到当时发生了什么？”
```

---

## 模块 B：Stage-aware Layered Audit

目的：

> **不直接让一个大模型盯着最终地图猜哪里错，而是按照建图因果顺序，在不同阶段利用对应证据筛出异常。**

它解决的是：

```text
“有了证据以后，哪些地方最值得怀疑？
怀疑的依据是什么？
是否存在反例？
还缺什么证据？
应该直接阻断、人工看、VLM 看，还是只记录？”
```

所以二者关系不是：

```text
证据模块      查错模块
两个平行功能
```

而是：

```text
证据模块 = 查错器的感官和记忆
查错器   = 对证据进行结构化推理
```

没有前者，后者只能猜。

---

# 3. 先彻底搞懂 6 个核心概念

这是整套系统最值得先吃透的部分。

---

## 3.1 Frame：这一时刻机器人看到了什么

一帧记为：

```text
frame_uid
```

例如：

```text
room0_..._f000120
```

Frame 记录：

- RGB；
- depth；
- camera pose；
- intrinsics；
- 原始 detection 数；
- 最终保留 observation 数；
- 是否处理；
- 如果跳过，为什么跳过。

它回答：

> **“这件事发生在哪个时刻、哪个视角、使用了什么相机几何？”**

---

## 3.2 Observation：这一帧中的“一个候选物体观测”

这是整个系统最关键的粒度。

假设第 120 帧 SAM/GroundingDINO 给出 20 个 proposal，那么就是 20 个 raw observations。

每个 observation 有永久 UID：

```text
obs_uid = frame_uid + raw_det_idx
```

例如：

```text
..._f000120_r0007
```

它代表的不是“真实世界物体”，而是：

> **“模型在某一帧提出的一个候选实例。”**

这点极其重要：

```text
Observation ≠ Object
```

例如同一张 sofa：

```text
frame 10 → obs_A
frame 20 → obs_B
frame 30 → obs_C
```

如果关联正确：

```text
obs_A
obs_B ───→ Object sofa_1
obs_C
```

---

## 3.3 Object：地图认为存在的一个持续实体

Object 是经过跨帧 association 后形成的地图节点。

例如：

```text
Object 17
```

可能由：

```text
obs_A + obs_B + obs_C + obs_D + ...
```

共同组成。

所以 object 是一种**身份假设**：

> “这些来自不同时间、不同视角的 observations，属于同一个现实物体。”

而你的论文潜在核心问题恰恰就在这里：

> **这个身份假设有可能在在线建图过程中逐渐犯错。**

---

## 3.4 Object Version：Object 在某一时刻的状态

只存最终 Object 仍然不够。

因为：

```text
Object 17
最初只有 obs_A
后来加入 obs_B
后来加入 obs_C
后来 denoise
后来又 merge 了 Object 31
```

如果只看最后：

```text
Object 17 = A+B+C+...
```

还是不知道哪一步把它搞坏了。

所以现在为对象维护：

```text
object_uid@v000001
object_uid@v000002
object_uid@v000003
...
```

每个 version 可以记录：

- member observations；
- point count；
- bbox；
- class histogram；
- dominant class；
- 当前 active / filtered / merged 状态；
- 由哪个 event 产生；
- parent version 是谁。

因此：

```text
Object 不再是一张静态照片
而是一条状态演化轨迹
```

这是未来做真正 rollback 的关键地基。

---

## 3.5 Event：是什么操作让状态发生变化

系统把重要操作显式记录成 event，例如：

```text
OBJECT_CREATE
OBS_ASSOCIATE
OBJECT_DENOISE
OBJECT_FILTER
OBJECT_MERGE
EDGE_ADD
EDGE_UPDATE
EDGE_DELETE
```

于是可以建立：

```text
obs_7
  ↓
association event
  ↓
obj_17@v3
  ↓
denoise event
  ↓
obj_17@v4
  ↓
merge event
  ↓
obj_17@v5
```

所以 Version 回答：

> “对象变成什么样了？”

Event 回答：

> “为什么从上一个状态变成这个状态？”

---

## 3.6 Finding：查错器发现的“风险案例”

Finding 也不是“确认错误”。

一个 Finding 包含：

```text
finding_uid
checker_id
stage
subtype

scope
proven_facts
hypotheses
vetoes
missing_evidence

certainty
severity
route

repair_allowed = false
```

这几个字段是整套查错思想最重要的地方。

### `proven_facts`

机器能直接从证据中计算出的事实。

例如：

```text
mask_iou = 0.9999
clip_similarity = 1.0
3d_support = 0.9888
```

### `hypotheses`

对这些事实的一种解释。

例如：

```text
duplicate_proposal
```

### `vetoes`

为什么还不能直接断言。

例如：

```text
part_whole_not_excluded
occlusion_not_excluded
many_to_one_is_legal_policy
```

### `missing_evidence`

如果想进一步确认，目前还缺什么。

例如：

```text
event_time_candidate_object_versions
GT visibility coverage
2D-to-3D reprojection renderer
```

这形成一个非常重要的原则：

> **事实和解释分开。**

不是：

```text
“两个 mask 很像，所以一定重复。”
```

而是：

```text
事实：
2D overlap 高
CLIP 高
3D overlap 高

假设：
可能是 duplicate proposal

反例：
也可能是真实的 part-whole

因此：
交给人工/VLM复核，而不是直接删除
```

这比“一堆 if 然后自动修图”严谨得多。

---

# 4. 统一证据留存到底保存了什么

不要按文件名背。最容易理解的方法是按“建图生命周期”看。

---

# 4.1 Run 级：我到底跑的是哪一次实验？

文件：

```text
manifest.json
```

主要记录：

- schema version；
- run_id；
- scene_id；
- dataset；
- Git branch；
- Git commit；
- 当前代码是否 dirty；
- Git diff hash；
- 如果有未提交修改，保存 runtime patch；
- 开始/结束时间；
- mapping config；
- detection config；
- model versions；
- prompt versions；
- runtime 环境；
- Python / NumPy / PyTorch / Open3D / CUDA 版本；
- random seeds；
- audit policy；
- 源代码快照；
- make_edges；
- run status。

它解决的是科研中非常现实的问题：

> “三天后看到一个异常案例，我还能不能知道它到底是哪版代码、哪组参数跑出来的？”

否则你会得到一种颇具人类传统特色的科研产物：

```text
room0_final_new2_reallyfinal_改完了.pkl
```

然后无人知道它怎么来的。

---

# 4.2 Frame 级：这个 observation 出现在哪个视角？

文件：

```text
frames.jsonl
```

保存：

```text
frame_uid
frame_idx
source_frame_id

RGB ref
depth ref

camera pose
intrinsics

processed
skip_reason

num_raw_detections
num_kept_observations
```

用途：

- 重建当时机器人视角；
- 做 2D↔3D 检查；
- 判断跨视角不一致；
- 给未来 VLM 提供完整场景上下文。

---

# 4.3 Raw Observation 级：检测器到底提出了什么？

文件：

```text
observations.jsonl
```

在 detection 进入后续 mapping 之前，就先为每个 proposal 固定：

```text
raw_det_idx
obs_uid
bbox_2d
raw mask
mask area
confidence
class_id
class_name
raw caption
image feature ref
text feature ref
crop ref
```

这里的关键不是数据多，而是：

> **即使这个 proposal 后来被 filter 掉，它仍然有身份。**

所以以后可以分析：

```text
“这个真实物体为什么没进入地图？”
```

究竟是：

```text
根本没检测到
还是检测到了但被 filter 掉
还是生成 3D 时失败
还是后来 object 被 filter
```

这四种完全不同。

---

# 4.4 Filter 级：为什么这个 proposal 被保留/拒绝？

文件：

```text
filter_trace.jsonl
```

以及 `observations.jsonl` 中的：

```text
status
filter_reason
filter_trace
```

重点不是只记：

```text
rejected
```

而是记录实际执行过的 gate：

```text
gate
value
operator
threshold
passed
```

例如：

```text
mask_area >= threshold          PASS
not_background                  PASS
bbox_area_ratio <= threshold    PASS
confidence >= threshold         PASS
valid_3d_observation            FAIL
```

最终：

```text
decision = REJECT
first_failed_gate = insufficient_3d_points
```

这叫：

> **execution trace，而不是事后猜原因。**

---

# 4.5 Mask 处理级：原始 mask 后来被改成什么样？

现在不仅保存原始 mask，还保存：

```text
processed_mask_ref
raw_mask_area
pre_subtract_mask_area
processed_mask_area
removed_pixel_count
subtract_source_obs_uids
mask_operations
```

当前处理链中会涉及：

```text
resize
filter
subtract_contained
```

因此后面可以检测：

```text
原来一个完整物体
↓
因为 containment subtraction
被扣掉了 70%
↓
剩下一堆碎片
```

否则只看最终 mask，你会把这个错误错怪给 SAM。

---

# 4.6 2D→3D Projection 级：这个 observation 变成了怎样的 3D 几何？

对于被保留的 observation，会记录：

```text
observation_pcd/*.npz
processed_mask
bbox_3d_center
bbox_3d_extent

valid_depth_ratio
depth_quantiles
boundary_touch_ratio

pre-DBSCAN:
  cluster_count
  largest_cluster_ratio
  second_cluster_ratio
  cluster center distance
  point count

post-DBSCAN:
  point count

voxel size
DBSCAN eps
DBSCAN min points
```

这类信息解决：

```text
“2D 看起来还行，为什么变成 3D 后这么怪？”
```

典型可能性：

```text
mask 吃进背景
depth 大量无效
一个 mask 实际包含两个 3D cluster
DBSCAN 把主体删掉
相机姿态导致跨帧几何漂移
```

---

# 4.7 Association 级：为什么它当时被分给了这个 Object？

这是最核心的证据之一。

完整相似度矩阵保存在：

```text
similarities/frame_xxxxxx.npz
```

里面保存：

```text
observation_uids
object_uids

spatial_sim
visual_sim
aggregate_sim
```

因此保留的是：

> **当时完整候选空间。**

不是只保留最后胜者。

同时：

```text
associations.jsonl
```

对每个 observation 记录：

```text
top_candidates

每个 candidate：
  object_uid
  spatial_score
  visual_score
  aggregate_score

top1_score
top2_score
margin

sim_threshold
match_method
phys_bias

decision:
  CREATE_OBJECT
  或 MERGE_TO_OBJECT

target_object_uid

target_object_version_before
target_object_version_after

candidate_object_version_uids
mapping_event_uid
transaction_uid
```

于是你终于能回答：

```text
“它为什么进了 Object 17？”
```

例如：

```text
Object 17:
  spatial = 0.08
  visual  = 0.92
  total   = 1.00

Object 23:
  spatial = 0.54
  visual  = 0.48
  total   = 1.02

最终：
Object 23 只领先 0.02
```

这与：

```text
Object 23 = 1.70
Object 17 = 0.40
```

显然不是同一种可信程度。

---

# 4.8 Object evolution 级：融合以后 Object 怎么变了？

文件：

```text
object_versions.jsonl
mapping_events.jsonl
```

每次重要更新会形成新的 object version。

可以跟踪：

```text
member observations
n_points
num_detections
bbox
class histogram
dominant class
active status
parent versions
trigger event
```

于是可以画出：

```text
v1：正常
 ↓ obs_8 加入
v2：中心移动 3cm
 ↓ obs_19 加入
v3：bbox 突然扩大 2.4×
 ↓ merge object_31
v4：class composition 从 chair 90%
   变成 chair 45% + sofa 35% + table 20%
```

这时“哪一步开始坏”就可以分析，而不是看最终点云猜。

---

# 4.9 Post-process Merge 级：为什么两个 object 又被合并？

文件：

```text
object_pair_decisions.jsonl
mapping_events.jsonl
```

merge candidate 保存：

```text
source object
target object

source/target version

overlap
visual similarity
text similarity

thresholds

ACCEPT / REJECT
reject_reason

source_active_before
target_active_before
source_consumed_after

source members before
target members before
member intersection before
```

如果接受 merge，还会记录：

```text
source_before
target_before
target_after

member union after
input versions
output versions
```

这使得未来可以做：

```text
撤销某次 merge
```

因为你知道：

```text
merge 前两个 object 是谁
merge 后是谁
成员集合如何变化
```

---

# 4.10 VLM / Edge 级：以后语义关系判断也能回溯

当前 recorder 已经具备：

```text
vlm_events.jsonl
```

可以记录：

- VLM call type；
- prompt text / fingerprint；
- image input；
- image hash；
- linked observations；
- model；
- generation params；
- raw response；
- parsed output；
- parser version；
- latency；
- request ID；
- error/status。

Edge 也会记录：

```text
EDGE_ADD
EDGE_UPDATE
EDGE_DELETE
```

但**当前分层查错器 v1 中 caption 和 relation checker 仍然关闭**。

也就是说：

```text
证据接口已经预留
≠
caption/relation 查错已经完成
```

---

# 4.11 Final membership：最后每个 Object 到底由谁组成

文件：

```text
final_membership.json
```

它本质上是最终 ownership table：

```text
Object 17
  ├─ obs_A
  ├─ obs_B
  └─ obs_C

Object 18
  ├─ obs_D
  └─ obs_E
```

当前 policy 是：

```text
一个 observation 最终只能属于一个 active object
```

即：

```text
exclusive_single_target
```

但：

```text
同一帧多个 observations
→ 可以合法地被关联到同一个 object
```

即：

```text
same_frame_many_to_one = allowed
```

这一区分很重要。后面的 `ASSOC-004` 只会把 many-to-one 当作**风险信号**，不会直接当作系统错误。

---

# 4.12 Evidence summary：这次账本自己有没有明显问题

文件：

```text
evidence_summary.json
```

汇总：

- frame 数；
- raw / kept / rejected observations；
- create / associate；
- object merge；
- missing references；
- duplicate members；
- logging errors；
- 等。

它是快速体检，不替代 Evidence Gate。

---

# 5. 为什么还需要 Evidence Gate

有了账本之后，不能立刻跑语义查错器。

因为可能发生：

```text
similarity NPZ 丢了
object version 链断了
obs_uid 重复了
mask 引用指向错误文件
filter_trace 和 observation status 对不上
final membership 根本无法从 event history 重放出来
```

如果账本本身就是错的，然后查错器还一本正经地分析“association 有问题”，那就属于自动化胡说八道。

所以必须先问：

> **“我即将用来判断地图的证据，本身可靠吗？”**

这就是 Evidence Gate。

---

# 5.1 Gate 主要检查什么

当前 `evidence_audit.py` 主要检查：

### EVI-001：结构化文件能否正常读取

例如：

```text
JSON / JSONL parse error
```

---

### EVI-002：UID 是否唯一、完整

检查：

```text
frame_uid
obs_uid
event_uid
object_version_uid
```

是否缺失或重复。

---

### EVI-003：ArtifactRef 是否真的有效

一个 artifact ref 不只是字符串路径，还可以包含：

```text
path
format
key
index
sha256
shape
dtype
```

Gate 会检查：

```text
文件是否存在
hash 是否一致
NPZ key 是否存在
index 是否越界
shape 是否一致
dtype 是否一致
文件是否可读
```

因此 artifact ref 的意义不是：

```text
“告诉你文件在哪”
```

而是：

> **“告诉你是哪一个可验证的具体证据。”**

---

### EVI-004：similarity matrix 是否可信

检查：

```text
frame 对应 NPZ 是否存在
observation axis 是否一致
object axis 是否一致

spatial / visual / aggregate
shape 是否正确
是否全为 finite
```

这是 association 审计的前置条件。

---

### EVI-005：Observation 与真实过滤轨迹是否一致

例如：

```text
observations.jsonl 说 KEEP
但 filter_trace 说 REJECT
```

或者 kept observation：

```text
没有 processed mask
没有 PCD
PCD 是抽样版而非真实融合版
mask area 与实际 artifact 不一致
point 数与实际 artifact 不一致
```

都会被视为证据问题。

---

### EVI-006：Object version / event dependency 是否成链

检查：

```text
version 的 trigger event 是否存在
parent version 是否存在
version 号是否连续
最终 active version 是否和 final membership 一致
event 输出的 version 是否真的存在
event parent graph 是否有环
```

---

### EVI-007：最终地图能不能从事件历史重放出来

这个检查非常重要。

系统尝试：

```text
OBJECT_CREATE
OBS_ASSOCIATE
OBJECT_FILTER
OBJECT_MERGE
...
```

重新 replay。

最终必须得到：

```text
与 final_membership 相同的 active object membership
```

并检查：

```text
一个 kept observation 是否被多个 active object 同时占有
```

---

### EVI-008：开启 VLM/edge 时，VLM 证据是否完整

如果：

```text
make_edges=true
```

则要求：

```text
prompt
images
model
generation params
raw response
parsed output
parser version
```

等证据存在。

---

# 5.2 Mapping invariants：不是所有错误都是“感知错误”

Evidence Gate 之后还检查若干 mapping invariant，例如：

```text
一个 merge source 是否被重复消费
source/target 是否处于合法 active 状态
merge 前成员是否非法重叠
merge 后成员集合是否真的是 union
是否出现没有 ACCEPT 决策却实际 merge
一个 observation 是否同时属于两个 active objects
num_detections 是否和成员数量一致
```

这些问题不是：

```text
“现实世界里这个是不是 chair？”
```

而是：

```text
“代码执行是否违反自己声明的规则？”
```

因此它们可以被更强地判定为：

```text
CONFIRMED_SYSTEM_ERROR
```

---

# 5.3 最重要的理解：Gate PASS 不等于地图正确

必须牢牢记住：

```text
Evidence Gate PASS
```

只意味着：

> **证据链内部基本自洽，可以放心使用它做后续诊断。**

不意味着：

```text
所有 detection 都对
所有 segmentation 都对
所有 association 都对
最终 scene graph 都对
```

这是两个完全不同的层次。

---

# 6. 分层查错器为什么要“分层”

原始建图的因果顺序是：

```text
Detection
   ↓
Segmentation
   ↓
2D→3D Geometry
   ↓
Association
   ↓
Fusion / Postprocess
   ↓
Object Identity
   ↓
Caption
   ↓
Relation
```

如果最终 object 错了：

```text
Object 18 很怪
```

可能根本不是 object-level merge 的问题。

可能是：

```text
一开始 detector 就重复提了两个几乎一样的 proposal
```

然后后面所有异常都是它的下游症状。

所以当前查错器按 stage 分开：

```text
system
detection
segmentation
geometry
association
fusion
object_identity
caption      [当前关闭]
relation     [当前关闭]
```

这就是所谓：

> **stage-aware audit**

它并不只是为了代码整齐，而是在利用 pipeline 本身的因果顺序。

---

# 7. 一个 Finding 到底如何表达“判断”

当前 Finding 不是：

```text
error = true
```

而是一种更接近科学推理的结构：

```text
{
  "proven_facts": [...],
  "hypotheses": [...],
  "vetoes": [...],
  "missing_evidence": [...],
  "certainty": "...",
  "severity": "...",
  "route": "...",
  "repair_allowed": false
}
```

---

# 7.1 Certainty：我有多确定

当前主要分四档。

### ① `CONFIRMED_SYSTEM_ERROR`

只用于能够从程序 invariant 直接证明的问题。

例如：

```text
一个 observation 同时属于两个 active objects
```

这种不需要 VLM 判断。

---

### ② `LIKELY_MAPPING_CONFLICT`

多种证据较一致地支持一个 mapping conflict。

但它仍然不等于 ground-truth confirmed。

例如：

```text
两个 observation：
  2D 几乎完全重合
  CLIP 几乎一致
  3D 几乎完全重合
```

强烈怀疑 duplicate proposal。

---

### ③ `AMBIGUOUS_MAPPING_RISK`

存在异常信号，但还有合理替代解释。

例如：

```text
spatial top candidate
和
semantic top candidate
不一致
```

可能是 false association，也可能只是：

```text
同类物体挨得很近
遮挡
视角变化
```

---

### ④ `INSUFFICIENT_EVIDENCE`

当前证据不足以做判断。

例如：

```text
没有 calibrated reprojection renderer
```

那就明确说缺，而不是凭感觉“应该没问题”。

---

# 7.2 Severity：如果是真的，影响可能有多大

大致分：

```text
CRITICAL
HIGH
MEDIUM
LOW
```

注意：

```text
certainty ≠ severity
```

可以：

```text
高度确定的小问题
```

也可以：

```text
不太确定但一旦成立就很严重的问题
```

---

# 7.3 Route：这个 case 下一步应该去哪

例如：

```text
BLOCK_RUN
HUMAN_REVIEW
VLM_REVIEW
LOG_ONLY
DOWNWEIGHT_EVIDENCE
SUPPLEMENT_EVIDENCE
```

因此 checker 不只是：

```text
“报错”
```

而是在做初步 triage。

---

# 8. 当前每一层到底查什么

下面是当前 `v1.1` 真正已经实现的核心规则。

这部分是未来你写方法章节时最重要的技术内容之一。

---

# 8.1 Detection 层

## DET-001：`DUPLICATE_PROPOSAL`

### 问题

同一帧 detector/segmenter 可能对同一个真实物体提出两个几乎一样的实例。

例如：

```text
sofa proposal A
sofa proposal B
```

如果两个都进入 mapping，可能：

```text
重复增加 observation
污染 association
造成重复 object
使 feature / point cloud 权重失衡
```

### 当前判定

对于同帧两个 kept observations：

\[
(\text{IoU}>0.85 \;\lor\; \text{containment}>0.95)
\]

同时：

\[
\text{CLIP sim}>0.90
\]

同时：

\[
\text{symmetric 3D support}>0.70
\]

才会触发。

### 为什么需要三种证据

只有 2D overlap：

```text
可能只是遮挡
```

加入 semantic：

```text
至少视觉上很像同一个东西
```

再加入 3D：

```text
它们在空间中也覆盖同一片结构
```

证据更强。

### 但为什么仍不自动删

因为：

```text
sofa + pillow
cabinet + drawer
table + cup
```

也可能天然重合。

因此存在 part-whole veto。

---

## DET-002：`POSSIBLE_FALSE_POSITIVE`

不是单看 detector confidence。

当前会组合多个信号：

```text
低 confidence
小 mask
少 3D points
低 valid depth ratio
缺乏跨帧 temporal support
```

其中 confidence、mask area、point count 的“低”使用当前 scene 中 kept observations 的低分位统计。

当同时出现至少 3 个弱信号时才进入风险候选。

思路是：

> **单一指标弱，不足以证明 false positive；多个相互补充的弱信号一起出现，才值得审查。**

但仍保留 veto：

```text
真实的小物体也可能只出现一次
```

---

## DET-004：当前无法可靠检查 False Negative

为什么？

如果 detector 根本没有提出 observation：

```text
账本里就没有那个 object 的候选
```

你需要：

```text
GT
或
完整 full-frame visibility coverage
```

才能知道：

```text
“这里本来应该有东西，但 detector 没看到。”
```

因此当前系统明确输出：

```text
缺少证据
```

而不是装作已经能查漏检。

---

# 8.2 Segmentation 层

## SEG-001：`INVALID_OR_DEGENERATE_MASK`

这是系统一致性检查：

```text
实际 processed mask 为空
或
artifact mask area 与账本记录不一致
```

可以直接视为 system error。

---

## SEG-005：`CONTAINMENT_SUBTRACTION_DAMAGE`

ConceptGraphs 中会进行 contained-mask subtraction。

假设：

```text
一个大 mask
里面包含一个小 mask
```

为了避免重复，可能从大 mask 中减去小 mask。

问题是：

```text
真实 part-whole
或者强遮挡
```

情况下，这个操作可能把主体错误切碎。

当前会看：

```text
subtraction 后面积损失 > 50%
并且
第二连通区域占比 >= 20%
```

如果同时发生：

```text
大幅被扣掉 + 结果明显碎裂
```

则认为存在高风险。

---

## SEG-002：`BACKGROUND_LEAKAGE_OR_UNDERSEGMENTATION`

一个 observation 在投影到 3D、DBSCAN 前，如果出现明显第二 cluster：

```text
second_cluster_ratio >= 0.20
```

说明一个 2D mask 可能对应了多个空间结构。

两种常见解释：

```text
A. mask 吃进了背景
B. 两个物体被 segment 成了一个
```

所以这里明确保留两个 hypothesis。

---

## SEG-006：`LOW_QUALITY_OBSERVATION`

质量信号包括：

```text
valid depth 过低
mask 大量触碰图像边界
mask 明显碎裂
```

如果其中至少两个同时出现，就标为 low-quality observation。

这类 case 的合理动作未必是删除，更可能是：

```text
DOWNWEIGHT_EVIDENCE
```

这是一个很值得保留的思想：

> **并非所有异常 observation 都要二元地“留/删”，也可以降低可信权重。**

---

## SEG-004：`POSSIBLE_OVERSEGMENTATION`

如果同一个 final object 内：

```text
同一帧存在两个 member observations
```

而且：

```text
CLIP similarity > 0.90
3D normalized center distance <= 0.50
```

说明可能：

```text
同一个真实实例在这一帧被切成多个 proposal
```

但也可能是：

```text
真实物体的合法 parts
```

所以只是 ambiguous risk。

---

# 8.3 Geometry 层

## GEO-001：`INVALID_PCD_OR_BBOX`

检查：

```text
PCD 缺失/为空
NaN / Inf
bbox center/extent 非法
extent 为负
```

属于确定的系统问题。

---

## GEO-003：`MULTI_CLUSTER_GEOMETRY`

与 SEG-002 有联系，但角度不同。

SEG 角度问：

```text
mask 是不是混了背景/多个物体？
```

GEO 角度问：

```text
这个 observation 的 3D 几何是不是明显多峰？
```

当 pre-DBSCAN：

```text
second cluster ratio >= 0.20
```

触发。

同一事实可以在不同 stage 支持不同解释，这恰恰是后面 root-cause aggregation 的基础。

---

## GEO-004：`CROSS_FRAME_GEOMETRY_CONFLICT`

同一个 object 的连续 member observations：

```text
obs_t
obs_t+k
```

比较它们 3D bbox center。

如果 normalized center jump：

```text
> 1.0
```

就说明同一个“身份”在不同帧中的几何位置跳得很大。

可能原因：

```text
association 错了
pose 有误
视角/遮挡导致 bbox 变化
环境真的动态变化
```

当前环境 policy 默认 static，所以动态变化主要作为 veto / alternative explanation。

---

## GEO-005：`DENOISE_DESTRUCTIVE_CHANGE`

如果一次 object denoise 后：

\[
\frac{N_{\text{after}}}{N_{\text{before}}}<0.30
\]

即 70% 以上点被清掉，就认为 denoise 可能破坏了主体。

这类规则的优点是：

> **直接检查“操作前后状态变化”，而不是只检查最后几何。**

---

## GEO-006：`OBJECT_SCALE_OUTLIER`

对同类 object 的 bbox volume 做 robust statistics。

当同类数量足够时，用 log-volume + median/MAD 检查极端 scale outlier。

它目前只是低优先级风险：

```text
LOG_ONLY
```

---

## GEO-002：目前尚缺完整 2D→3D reprojection checker

配置中已经有：

```text
reprojection IoU thresholds
```

但当前代码明确标记：

```text
calibrated 2D-to-3D reprojection renderer
```

尚缺。

因此目前不能在论文里说：

```text
“我们已经完整验证 2D mask 和 3D projection 的重投影一致性。”
```

还没有。

---

# 8.4 Association 层

这是目前最值得重点理解的一层。

---

## ASSOC-002：`LOW_MARGIN`

如果：

\[
\text{margin}=s_1-s_2\le 0.05
\]

说明第一名和第二名候选差距很小。

例如：

```text
Object A = 1.203
Object B = 1.197
```

虽然算法必须选一个，但这种决策天然不稳。

所以它目前：

```text
AMBIGUOUS_MAPPING_RISK
LOG_ONLY
```

它不是说：

```text
“选错了”
```

而是说：

> **“当时这个决策本来就不确定。”**

---

## ASSOC-003：`SPATIAL_SEMANTIC_DISAGREEMENT`

对于同一个 observation：

```text
spatial score 第一名 = Object A
visual score 第一名  = Object B
```

说明几何和语义对“它是谁”意见不一致。

这可能暗示：

```text
同类物体相邻
pose / geometry 有问题
视觉 embedding 混淆
association 可能选错
```

因此送：

```text
VLM_REVIEW
```

而不是直接改。

---

## ASSOC-007：`HIGH_SEMANTIC_LOW_GEOMETRY_ASSOCIATION`

当前典型条件：

```text
已经执行 MERGE_TO_OBJECT

visual score >= 0.85
spatial score <= 0.05

并且 aggregate score
只比 association threshold 高 <= 0.05
```

这类 case 很值得警惕：

```text
“视觉上很像，
空间上几乎不支持，
最后只是勉强越过总阈值。”
```

现实中非常容易是：

```text
两把长得一样的 chair
两个相似 cabinet
两个同款 pillow
```

被语义强行吸到一起。

---

## ASSOC-004：`MANY_TO_ONE`

同一 frame：

```text
obs_A ─┐
       ├→ Object 17
obs_B ─┘
```

当前 policy 明确允许这种行为。

所以 checker 不会说：

```text
“many-to-one 本身非法”
```

它会进一步看两 observation：

### 如果 2D 高度重合：

```text
MANY_TO_ONE_DUPLICATE_LIKE
```

可能是 duplicate proposals。

### 如果二者明显分离：

```text
MANY_TO_ONE_SEPARATED
```

更值得怀疑：

```text
是不是两个真实实例被吸进一个 object？
```

但仍保留：

```text
many_to_one_is_legal_policy
```

作为 veto。

这体现了一个很关键的审计原则：

> **“违反统计直觉”不等于“违反算法 policy”。**

---

## ASSOC-005：`SEMANTIC_MEMBER_OUTLIER`

对于一个已经形成的 object，如果至少有多个 member observations：

```text
obs1
obs2
obs3
obs4
...
```

先在 image feature 中找一个 medoid，再计算每个 member 到主体语义簇的距离。

使用 robust median/MAD。

若某个 observation：

```text
robust z > 3
```

说明：

```text
它在这个 object 的多视角语义历史中很另类
```

可能是：

```text
错误 association
极端视角
遮挡
视觉 embedding 失真
```

单独这条只是风险。

---

## ASSOC-006：`LOW_GEOMETRIC_MEMBER_SUPPORT`

只对上面的 semantic outlier 继续深挖。

把可疑 observation 从 object 中拿掉：

```text
target core = 其余 observations
```

再计算：

```text
可疑 observation PCD
和
其余 object core
```

的几何支持。

如果：

```text
geometric support < 0.15
```

就形成：

```text
语义上不像
+
几何上也不像
```

这比 ASSOC-005 强很多。

因此更接近：

```text
false association
```

---

## ASSOC-009：近似 Leave-One-Out 反事实重关联

对已经满足：

```text
语义异常
+
几何低支持
```

的 observation，系统继续问：

> “如果它当时不属于当前 Object，还有没有另一个 Object 比现在这个更合理？”

当前实现会在**最终地图**中比较其他 object：

```text
alternate semantic score
alternate geometry score
```

如果另一个 object：

```text
语义更好
几何也更好
总分比当前 target 至少高 0.10
```

则产生：

```text
APPROXIMATE_LEAVE_ONE_OUT_REASSOCIATION
LIKELY_MAPPING_CONFLICT
```

这是目前最接近“反事实”的一条规则。

但要注意两个限制：

```text
1. 比较的是 final map
2. 不是 association 发生当时的候选 object state
```

因此 Finding 会显式记录：

```text
approximation = final_map
missing_evidence = event_time_candidate_object_versions
```

论文里如果以后使用，应该称：

```text
approximate counterfactual reassociation
```

不能写成严格 counterfactual replay。

---

## ASSOC-010：trajectory-level identity alignment 当前缺失

意味着目前还没有完整建立类似：

```text
全局轨迹级 ground-truth identity
```

来直接算 ID-switch / association accuracy。

---

# 8.5 Fusion / Postprocess 层

## FUSE-007：`FUSION_SHOCK`

比较一次：

```text
OBS_ASSOCIATE
或
OBJECT_MERGE
```

前后的 object version。

当前看三个主要变化：

```text
normalized center shift > 0.30
bbox volume growth > 1.50×
point count growth > 1.50×
```

如果至少两个同时异常：

```text
FUSION_SHOCK
```

思路非常直观：

> 正常加入一个新视角，object 应该渐进变化；
> 如果一次 update 让它在多个维度突然“爆炸”，这一步值得查。

但仍有 veto：

```text
一个全新视角确实可能补充大量之前没看到的几何
```

---

## FUSE-008：`ASYMMETRIC_OVERLAP_RISK`

如果能够同时得到：

```text
A→B overlap
B→A overlap
```

且二者相差：

```text
> 0.50
```

说明两个 object 很可能：

```text
大小差异极大
part-whole
只是一方被另一方包含
```

这种情况下仅依赖单向 overlap merge 会有风险。

当前实际 recorder 并不总能提供完整双向 overlap，所以这条规则的可用性受证据限制。

---

## FUSE-009：`TEXT_SIMILARITY_IS_VISUAL_PROXY`

这是一个很值得注意的“诚实规则”。

当前 merge 记录中所谓：

```text
text_similarity
```

实际来源仍可能是：

```text
VISUAL_PROXY
```

也就是并没有真正独立的 text semantic evidence。

因此系统会主动产生：

```text
INSUFFICIENT_EVIDENCE
```

提醒：

> 不能把 visual 和所谓 text 两个数，当成两个独立语义证据。

---

## FUSE-011：`FILTER_POLICY_MISMATCH`

如果程序执行了：

```text
OBJECT_FILTER
```

但它实际上并不满足：

```text
obj_min_points
或
obj_min_detections
```

规定的过滤条件，则可以直接判定：

```text
系统执行与 policy 不一致
```

属于 confirmed system error。

---

# 8.6 Object Identity 层

这一层不是看某一次 observation，而是问：

> **“最终形成的 object identity 本身合理吗？”**

---

## OBJ-002：`POSSIBLE_DUPLICATE_OBJECT`

两个 active final objects：

```text
CLIP similarity > 0.90
normalized center distance < 0.20
symmetric 3D support > 0.75
```

高度相似且空间几乎重合。

可能是：

```text
false split
```

即一个真实物体被建成两个节点。

但系统还会检查：

```text
两个 object 是否在多个 frame 中
明确同时、分离地出现
```

如果：

```text
separated co-visibility >= 2
```

就是一个很强的 veto：

> “它们真的曾同时作为两个分开的东西出现过。”

这能降低把同类相邻物体误合并的风险。

---

## OBJ-003：`LOW_SUPPORT_OBJECT`

如果一个 final object 同时满足至少 3 个弱支持信号：

```text
只在一个 view 出现
只有一个 observation
median confidence 很低
3D points 极少
```

则可能是：

```text
noise node / weak object
```

但仍保留：

```text
real small object
```

这个反例。

---

## OBJ-005：`IDENTITY_INSTABILITY`

统计一个 object 所有 member observations 的 class composition。

如果：

```text
出现 >= 3 种 class
并且
dominant class ratio < 0.60
```

说明这个 object 的身份长期不稳定。

可能是：

```text
false merge
或者
2D semantic label instability
```

因此这条不能独立证明 false merge，但很适合和 association/fusion findings 联合解释。

---

## OBJ-006：缺少 GT / 完整可见性覆盖

所以当前 object-level false negative / recall 仍不能靠 audit 直接得到。

---

# 9. 最关键的一点：查错器不是“一条规则判一个错”

真正有价值的是跨阶段组合。

来看一个**假想但完全符合当前规则逻辑**的例子。

---

## 9.1 一个 false association 可能怎样被逐步发现

假设：

```text
obs_X
```

被关联到了：

```text
Object A
```

### 第一步：当时决策就很犹豫

```text
ASSOC-002
margin = 0.01
```

说明 A 和第二候选几乎打平。

---

### 第二步：几何和语义意见不一致

```text
ASSOC-003

spatial top = Object B
visual top  = Object A
```

此时开始怀疑：

```text
是不是语义把一个相似物体吸错了？
```

---

### 第三步：随着 Object A 越来越完整，obs_X 显得语义离群

```text
ASSOC-005
robust z = 4.2
```

---

### 第四步：把 obs_X 从 A 中拿掉后，它和 A 剩余几何也几乎不重合

```text
ASSOC-006
geometry support = 0.08
```

---

### 第五步：反而另一个 Object B 同时语义、几何都更好

```text
ASSOC-009

A score = 0.82
B score = 1.04
gain = 0.22
```

于是风险强度越来越高：

```text
低 margin
+
spatial/semantic disagreement
+
semantic outlier
+
low geometric support
+
stronger alternate
```

这就比一个简单规则：

```text
if CLIP < 0.7:
    error
```

强得多。

---

# 10. Root-cause aggregation：为什么还要从 Findings 聚合成“根因候选”

同一个原始问题可能产生一串下游症状：

```text
DET-001 duplicate proposal
        ↓
ASSOC-004 same-frame many-to-one
        ↓
FUSE-007 fusion shock
        ↓
OBJ-005 identity instability
```

如果把这四条当成四个独立错误：

```text
错误数 = 4
```

会严重重复计算。

于是当前系统会按共享实体：

```text
object
object pair
observation
observation pair
```

把相关 findings 聚在一起。

然后按：

```text
更早 stage
→ 更高 severity
→ 更高 certainty
```

选一个 primary hypothesis。

例如：

```text
primary root-cause hypothesis:
duplicate_proposal

supporting findings:
DET-001
ASSOC-004
FUSE-007
OBJ-005
```

---

## 10.1 这是不是“真正的因果证明”？

不是。

当前的 root-cause resolver 本质是：

> **利用 pipeline 时间/阶段顺序做启发式因果归因。**

它非常适合：

```text
缩小人工/VLM审查范围
提出最早的可疑源头
```

但还不能严格声称：

```text
“我们已经证明 DET-001 导致了 OBJ-005。”
```

真正更强的 causal claim 需要以后做：

```text
intervention / rollback
```

例如：

```text
撤销 duplicate observation
重新 replay 后续 mapping
↓
后续 fusion shock 消失
identity 恢复稳定
```

这时才能更有底气地说：

```text
这是因果根因
```

因此当前论文安全术语建议是：

```text
root-cause hypothesis
stage-aware root-cause candidate
provenance-guided diagnosis
```

暂时不要把“causal diagnosis”写得过满。

---

# 11. Evidence Packet：为什么最后还要生成可视案例包

结构化 JSON 很适合程序，不适合人和 VLM。

所以对于一个 Finding，系统会把相关证据重新打包成：

```text
Evidence Packet
```

典型包括：

```text
case.json
metrics.json

overview.jpg
context crops
masked crops

mask overlay
depth visualization
3D PCD overlay
timeline / multi-view context
```

并尽量把：

```text
chosen target
spatial top candidate
visual top candidate
aggregate top candidates
counterfactual alternate
```

对应 object 的关键 views 一起带出来。

所以 Packet 的作用是：

> **把机器筛出的“抽象异常”，转换成一个人/VLM真正能审查的病例。**

## 11.1 但“有 Packet”不等于“人工证据有效”

这一点是在真正准备 R1 时发现的，也是整套方法必须补上的一层。

旧 packet 主要回答：

```text
这个 finding 周围有哪些图、mask、depth 和 observation PCD？
```

但人工标签还要回答：

```text
系统当时究竟做了什么 identity 决策？
决策时比较的是哪些 object version？
触发 observation 最后归到哪里？
最终地图里相关 object 的完整成员和几何是什么？
```

如果网页只展示几张抽样 crop 和 `pcd_overlay.png`，而 checker 实际读取的是完整 association row、object versions、final membership 和数值数组，那么人和系统面对的不是同一个问题。此时标签即使形式上完整，也不能证明规则有效。

因此有效人工证据必须满足：

```text
同一个 case
  ↕ UID / hash / membership 可追溯
checker 真正读取的 ledger record
  ↕ 确定性投影
人类页面实际看到的图和数值
  ↕ final membership / final pickle
最终地图中的真实 object
```

这里的核心不是让人读取机器用的每个数组，而是保证页面上的每个结论都能说清来源，并且不会把一种状态的图冒充另一种状态。

### 新 R1 页面如何组织证据

每例固定分六层：

1. **审查问题**：明确问“是否对应真实建图错误”，而不是问某个阈值是否触发。
2. **系统决策**：直接显示 CREATE/MERGE、Top 候选、空间/视觉/综合分数、margin、阈值和决策时版本。
3. **触发 observation**：同一 ledger record 引用的 RGB、raw mask、processed mask、mask change、depth 和保存的 observation PCD。
4. **对象身份**：为相关对象分配页面别名，串联其角色、决策时版本、最终去向和代表视图覆盖率。
5. **最终 object**：从 manifest 哈希锁定的 final pickle 直接读取完整 PCD，并核对 UID、完整 observation membership 和点数。
6. **旧 packet 材料**：保留作追溯，但明确警告 `pcd_overlay.png` 只是抽样 observation 叠加，不是 final object。

### 三种证据必须在页面上分开

| 类型 | 例子 | 能说明什么 |
|---|---|---|
| 系统精确记录 | association 分数、mask artifact、object version、final membership、final pickle PCD | 可直接支持对应系统事实 |
| 人类解释视图 | 从完整成员中确定性抽取的多视角 crop | 帮助理解物理关系；必须同时显示 `selected / total`，不能冒充全部成员 |
| 未保留状态 | DBSCAN 前点坐标、去噪前后完整 object PCD、融合前后完整版本 PCD | 不能通过事后拼图恢复；必须声明缺口 |

### 为什么 final object 不能省略

以真实的 `room0/finding_002041`（`ASSOC-002 LOW_MARGIN`）为例：

- 触发 observation 的 Top-1 / Top-2 分数约为 `0.871 / 0.825`，margin 约 `0.046`；
- 系统阈值为 `1.2`，所以最终动作不是归入 Top-1，而是 `CREATE_OBJECT`；
- 新对象最终只有 1 个 observation、16 个点；
- 两个主要候选却分别是 220 个 observation 和 195 个 observation 的 active armchair objects。

只看“margin 很低”或一张 observation overlay，人很容易把任务误解成“确认低 margin 是否发生”。看完系统决定和最终对象后，真正的问题才变成：

> 这个 1-view / 16-point 新节点是正确的新物体，还是本应属于两个已有椅子之一，从而造成 false split？

这才是能够产生有效标签的问题。

### 历史 finding v1 的 160 例投影结果（已退役）

下面这组数字记录了我们为什么必须纠偏，便于追溯旧方案，但**不再是当前 R1 队列的状态**：

```text
160 / 160：源 artifact 哈希一致
160 / 160：相关 final object UID、完整成员集合和点数与 final pickle 一致
160 / 160：页面 review JSON 与 worklist / case JSON 绑定一致
9918 / 9918：最终指标门重新计算页面图片哈希一致
134 / 160：TRACEABLE
 26 / 160：TRACEABLE_WITH_CRITICAL_GAP
```

26 例关键视觉缺口来自正式运行没有保存历史 PCD 状态：

```text
SEG-002   11  DBSCAN 前点坐标未保存
GEO-003   10  DBSCAN 前多簇点坐标未保存
GEO-005    1  去噪前后完整 object PCD 未保存
FUSE-007   4  融合前后完整 object-version PCD 未保存
```

数值统计、成员变化和最终 object 仍然是精确的；缺的是某个历史阶段的直接视觉复核。旧方案曾打算让人再选 `PARTIAL/NO`，深入复查后认为这仍是在把系统未保存证据的责任推给复核者，因此已废止。

中间版 `v2.0` 先按精确 trigger 集合识别出 221 个 fully blocked incidents；进一步按 final endpoint 聚合后，大多数缺口信号都与同一最终对象上的可复核信号汇合，不再单独制造人工案例。最终 `v2.1` 的 98 个 endpoint 中只有 1 个完全没有忠实复核依据而被阻断；其余 97 个各对应一个不同 final object。这个变化正好体现了“阶段性缺口不能跨阶段重复惩罚，也不能让人猜”。详见 34.3～34.4。

### 当前协议中，证据不足不能被统计成“最终状态正确”

新版只有 `evidence_sufficient=YES/NO`。`NO` 表示页面不足以判断最终状态，必须与 `final_state=UNCLEAR`、`final_error_type=NOT_APPLICABLE` 配对；它既不是 checker false positive，也不是 final map 正确。最终指标因此分开报告：

- `evidence_sufficient=YES` 中最终 `WRONG` 的 endpoint 比例；
- 97 个 flagged final endpoints 的完整普查覆盖率、确认错误产出、保守下界和上界；
- 当前 cap 没有约束任何场景，因此 headline 是普查实测率，不再使用 calibration 权重冒充必要的估计；
- 证据不足时保持 `UNCLEAR`，绝不暗算成 `CORRECT` 或 `WRONG`。

当前不再设置一轮与论文主问题无关的 32 例复杂 R2。R1 完成后，机器先计算 incident endpoint 指标；只有 `evidence_sufficient=YES + final_state=WRONG` 的案例进入专家因果追踪。专家填写根因和候选干预，真正的修复有效性只能由 intervention/replay 证明。单复核者带来的主观性会作为本轮限制如实报告，不伪造一致率。

---

# 12. 一个真实案例：`finding_000002`

当前 GitHub 中保存了一份真实 Evidence Packet：

```text
DET-001
DUPLICATE_PROPOSAL
```

对应同一帧两个 observations。

实测事实：

```text
mask IoU             = 0.9999279
mask containment     = 1.0
CLIP similarity      = 1.0
symmetric 3D support = 0.9887698
```

几乎可以理解成：

```text
2D：
两块 mask 几乎完全一样

语义：
视觉 embedding 几乎完全一样

3D：
投影出的结构也几乎完全一样
```

因此系统给：

```text
certainty = LIKELY_MAPPING_CONFLICT
severity  = HIGH
hypothesis = duplicate_proposal
vetoes = []
route = HUMAN_REVIEW
repair_allowed = false
```

这里最重要的不是“它看起来很像 bug”。

而是方法逻辑：

```text
机器已经把：
  事实
  假设
  证据
  置信等级
  审查路径
明确分开

但最后仍不自动删除。
```

这就是当前系统的保守性。

---

# 13. 双队列抽样为什么仍保留，但当前实际是 endpoint 普查

如果 audit 一跑得到两千多条 Findings，人不可能全看。

最直觉的做法是：

```text
只看 review score 最高的 200 个
```

但这样有一个统计学问题：

> 你看到的全是系统自己觉得最明显的 case。

于是你不能拿这 200 个的准确率说：

```text
“我们的 checker precision = 90%”
```

因为样本已经被偏置筛选过。

所以基础版 v1.1 把 Evidence Packet 分成两个 cohort；final-endpoint v2.1 仍保留它作为“未来 endpoint 数超过人工上限时”的回退策略，但必须先按最终对象去重。

当前真实情况更简单：room0 只有 69 个可复核 endpoints，office0 只有 28 个，都没有触及每场景 80 的上限，因此 97 个全部进入 R1。现有 `calibration_random / diagnostic_priority` 字段只是构建器留下的可追溯分区，不影响 headline；当前结果按**完整 flagged-endpoint census**计算，不使用抽样权重。

---

## 13.1 `calibration_random`

这一节描述的是发生抽样时的统计回退，不是当前 97 例必须依赖的估计。

目标：

```text
估计 checker 到底准不准
```

做法：

按：

```text
checker_id × certainty
```

分层，然后随机抽样。

每个 case 会记录：

```text
selection_probability
sampling_weight = 1 / probability
```

因此以后可做加权 precision 估计。

---

## 13.2 `diagnostic_priority`

当前 room0 的 29 个 priority endpoints 也已经和其余 endpoints 一起全部纳入普查；该排序只帮助未来优先查看，不会改变分母。

目标不是统计总体准确率，而是：

```text
最快看到最有价值的错
找最典型 root cause
决定下一步修什么
```

会综合：

```text
severity
certainty
超过阈值的强度
support signal 数量
下游相关 findings 数量

扣除：
veto
missing evidence
重复实体/重复规则
```

同时限制：

```text
单 checker 最大案例数
单实体最大案例数
重复 scope
```

避免 200 个 packet 全是同一张 sofa。

---

## 13.3 最重要的纪律

当前代码已经明确写入 policy：

```text
只有 calibration_random
可以用于 weighted precision

diagnostic_priority
只能用于 root-cause discovery
不能当概率样本
```

这一点以后写论文必须保留。

---

# 14. Review Score 到底是什么

它只负责排列 `diagnostic_priority`，不是正确概率，也不会显示给 R1。incident v2 会在同一 incident 的可复核 findings 中选择代表信号并形成 incident 级优先级；人只面对最终对象证据，不根据分数猜答案。

当前 priority cohort 的 review score 大致考虑：

```text
+ severity
+ certainty
+ threshold exceedance
+ support signal diversity
+ downstream pollution

- veto penalty
- missing evidence penalty
- redundancy penalty
```

这里有一个术语需要你自己理解清楚。

代码字段现在叫：

```text
independent_evidence
```

但实际计算方式主要是：

```text
不同 hypothesis support 字符串的数量
```

它并没有从统计意义证明这些 evidence 真的 independent。

所以以后论文里更安全的表述是：

```text
support-signal diversity
multi-cue support
```

不要直接宣称：

```text
statistically independent evidence
```

除非后面重新设计。

---

# 15. 历史 room0 v1.0 全量结果应该怎么看

2026-08-19 保存的 `room0` full audit 是 `schema/config 1.0.0`。它只用于说明方法演进，不是当前 incident v2 正式结果；当前双场景实测见 34.4。

结果：

| 指标 | 数量 |
|---|---:|
| Evidence Gate | PASS |
| mapping_mutated | false |
| Findings | 2455 |
| Root-cause candidates | 574 |
| LIKELY_MAPPING_CONFLICT | 502 |
| AMBIGUOUS_MAPPING_RISK | 1944 |
| INSUFFICIENT_EVIDENCE | 9 |
| Evidence Packets | 200 |

阶段分布：

| Stage | Findings |
|---|---:|
| system | 1 |
| detection | 555 |
| segmentation | 519 |
| geometry | 50 |
| association | 1268 |
| fusion | 42 |
| object identity | 20 |

---

# 15.1 这些数字能说明什么

可以说：

```text
1. Evidence Gate 成功通过
2. audit 是只读的，没有修改地图
3. 当前规则能筛出大量候选风险
4. association 是候选最密集的阶段
5. 多个下游 finding 能被聚合成较少 root-cause candidates
6. 系统能够生成可人工复核的 packet
```

---

# 15.2 这些数字绝对不能说明什么

不能说：

```text
“room0 有 2455 个错误”
```

因为：

```text
Findings = 风险候选
```

不是 ground truth error。

也不能说：

```text
“502 个错误已经确认”
```

因为：

```text
LIKELY_MAPPING_CONFLICT
```

仍然是 evidence-supported hypothesis。

---

# 15.3 为什么 500 特别危险

当前保存的 1.0.0 结果中：

```text
DET-001   = 500
SEG-004   = 500
ASSOC-003 = 500
ASSOC-004 = 500
```

恰好碰到：

```text
max_findings_per_rule = 500
```

因此：

```text
500
```

不是“真实发生量”，只是“输出被截到了上限”。

该风险后来已经通过提高 cap、记录 attempted/emitted/suppressed counts，并在出现 suppression 时让 validity gate 失败而解决。incident v2 的两个正式场景均为 uncensored population。

---

# 16. 证据模块是否真的没有改变原建图

这件事已经做过一个很有价值的 smoke test。

Replica room0：

```text
2 frames
70 raw detections
47 kept observations
23 rejected

25 create
22 associate
7 object merges

final objects = 19
missing references = 0
logging errors = 0
```

并且 evidence：

```text
ON / OFF
```

对照时：

```text
canonical object JSON hash 一致
edge JSON hash 一致
```

进一步检查：

```text
PCD
color
bbox
CLIP array
```

也一致。

所以目前有证据支持：

> **在该 smoke 范围内，证据旁路没有改变原始 mapping 结果。**

但要注意：

```text
2-frame smoke
≠
完整长序列形式化证明
```

未来正式实验仍应做更长 parity test。

---

# 17. 当前系统的代码结构应该怎么理解

不要按 10 万行代码读。

只记住下面这张职责表。

| 文件 | 角色 | 一句话理解 |
|---|---|---|
| `conceptgraph/utils/evidence.py` | Producer | 建图运行时把事实写进证据账本 |
| `conceptgraph/audit/evidence_audit.py` | Gate | 检查证据账本和 mapping invariant 是否可信 |
| `conceptgraph/audit/layered_audit.py` | Screener | 分阶段根据证据产生风险 Findings |
| `conceptgraph/audit/configs/v1.yaml` | Policy | 定义环境假设、算法 policy、阈值和 case sampling |
| `conceptgraph/audit/runner.py` | Orchestrator | 运行整个 audit |
| `tests/test_evidence.py` | Regression | 验证证据 recorder |
| `tests/test_evidence_audit.py` | Regression | 验证 gate |
| `tests/test_layered_audit.py` | Regression | 验证 layered screener / case builder |
| `docs/audit_v1_artifacts/...` | Evidence | 保存真实 room0 审计产物 |

逻辑上：

```text
mapping
  ↓ calls
evidence.py
  ↓ produces ledger
evidence_audit.py
  ↓ PASS
layered_audit.py
  ↓
findings / root causes / cases
```

---

# 18. 当前配置中的 Policy 为什么很重要

查错不是只看数字。

同样一个现象，在不同 mapping policy 下含义可能完全不同。

当前显式声明：

```text
observation ownership:
exclusive_single_target

same-frame many-to-one:
allowed

relation:
many-to-many

object granularity:
instance_with_part_whole_ambiguity

environment:
static

missing evidence:
unknown_not_pass
```

以及 association：

```text
independent greedy argmax

max score > threshold:
associate

max score == threshold:
create object
```

merge：

```text
source can only be consumed once
source must be active
target must be active
```

因此审计器不是凭自己的“常识”判断：

```text
“many-to-one 看起来怪，所以一定错。”
```

而是先问：

```text
“它违反了什么 policy？”
```

如果没有违反：

```text
只能记作 risk
```

这其实是当前系统最像正式科研方法的部分之一。

---

# 19. Evidence Recorder 的 sidecar 到底是什么意思

当前设计的原则是：

```text
Evidence 不参与：
detection decision
filter decision
association score
mapping decision
merge decision
edge decision
```

它是旁路 observer。

但有两种运行模式需要区分。

---

## `best_effort`

如果 evidence 写失败：

```text
记录 logging error
然后 mapping 继续
```

适合开发调试。

---

## `strict`

如果 evidence 写失败：

```text
直接抛异常终止 formal run
```

这不是 evidence 改变了建图算法，而是：

> **正式实验不允许“地图跑完了但证据坏了”。**

所以论文实验更应该用 strict。

---

# 20. 当前已经做好的“创新基础”究竟在哪里

如果只说：

```text
“我们记录日志，然后写规则查错。”
```

确实毫无论文味。

更准确的抽象是：

---

## 20.1 从“最终地图”提升到“状态演化账本”

以前：

```text
Final Map
```

现在：

```text
Observation
→ Decision
→ Object Version
→ Event
→ New Object Version
→ ...
→ Final Map
```

所以地图不只是一个结果，而是一条 provenance chain。

---

## 20.2 从“一次阈值判错”提升到“事实-假设-否决结构”

以前：

```text
score < threshold
→ error
```

现在：

```text
Proven facts
   ↓
Hypothesis
   ↔
Veto / alternative explanation
   ↓
Missing evidence
   ↓
Certainty + Route
```

更适合后面接 VLM reasoning。

---

## 20.3 从“最终异常”向“上游根因”回溯

把：

```text
identity instability
```

追溯到：

```text
fusion shock
association conflict
duplicate proposal
```

至少形成 stage-aware root-cause hypothesis。

---

## 20.4 从“全量盲查”变为“受控病例选择”

```text
calibration random
+
diagnostic priority
```

兼顾：

```text
科学评估
和
高效率根因发现
```

---

# 21. 但哪些东西目前还不能叫创新贡献

必须克制。

当前还没有证据支持直接宣称：

```text
“我们的 checker 准确地发现了 mapping errors”
```

因为还没完成正式人工标注 precision。

也不能宣称：

```text
“我们实现了自动 error correction”
```

因为没有 repair loop。

也不能宣称：

```text
“VLM 驱动了在线自纠错”
```

因为当前 `VLM_REVIEW` 主要是 route，VLM adjudicator 尚未成为完整自动方法。

更不能说：

```text
“我们实现了严格 causal diagnosis”
```

因为当前 root cause 主要依赖 stage ordering 与关联 findings 聚合。

这些都不是坏事。

现在正处于：

> **把研究基础设施转化成可验证方法的中间节点。**

---

# 22. 当前已知边界和工程风险

这部分以后你自己做实验时要很清楚。

---

## 22.1 Caption / Relation checker 暂时没有启用

当前：

```yaml
caption: false
relation: false
```

所以现阶段查错器主要覆盖：

```text
object mapping lifecycle
```

而不是完整 scene graph semantics。

---

## 22.2 False negative 不能靠现有 provenance 自己证明

因为：

```text
没被观察到的东西
不会自然出现在 observation ledger
```

需要 GT / visibility coverage / external detector。

---

## 22.3 2D→3D calibrated reprojection 还没完成

`GEO-002` 会明确指出缺失。

---

## 22.4 ASSOC-009 是近似反事实

当前使用 final map 做 alternate comparison，不是 event-time exact replay。

---

## 22.5 Merge 的 text similarity 当前可能只是 visual proxy

因此 `FUSE-009` 会主动标记证据不独立。

---

## 22.6 当前 `record_associations()` 有一个必须修的工程问题

当前代码在 similarity matrix shape 不符合预期时会使用：

```python
np.empty(expected_shape)
```

然后后续仍可能计算 Top-K 和 margin。

`np.empty` 是未初始化内存，不代表无效值。

因此 formal validation 前应改成：

```text
明确 invalid
→ NaN / status
→ Gate FAIL
→ 不继续产生 association evidence
```

这个问题影响的是：

```text
异常情况下证据可信性
```

不是整个 provenance 架构。

---

## 22.7 `max_findings_per_rule=500` 会截断 population

当前 full room0 已经实际撞到。

正式计算 calibration precision 前，需要避免：

```text
先截取前 500
再从这 500 中随机抽样
```

否则那不是完整 population 的随机样本。

---

# 23. 如果未来写成论文，应该怎样抽象，而不是照着代码文件写

论文中绝对不要写成：

```text
我们创建了：
manifest.json
frames.jsonl
observations.jsonl
...
```

那是 implementation details，不是方法。

更好的方法结构如下。

---

# 23.1 Problem formulation：增量 object-centric mapping

给定 posed RGB-D sequence：

\[
I_{1:T}
\]

在时间 \(t\)，得到一组 object observations：

\[
Z_t=\{z_{t,i}\}_{i=1}^{N_t}
\]

其中一个 observation 可抽象为：

\[
z_{t,i}=
(m_{t,i},p_{t,i},f_{t,i})
\]

分别代表：

```text
2D mask
3D geometry
semantic feature
```

当前地图：

\[
M_t=(O_t,E_t)
\]

每个 observation 根据 association function：

\[
a_{t,i}
=
\arg\max_j
\phi(z_{t,i},o_j)
\]

选择：

```text
existing object
或
new object
```

问题在于：

> **错误 association / segmentation / fusion 会被写入 persistent object state，并可能进一步污染后续决策。**

---

# 23.2 Provenance-aware Evidence Ledger

可以把当前实现抽象成一个 provenance graph：

\[
G^P=(V^P,E^P)
\]

其中节点不是 scene graph object，而是：

```text
Frame
Observation
Artifact
Decision Event
Object Version
Merge Transaction
VLM Event
```

边表示：

```text
observed-in
derived-from
associated-to
updated-by
merged-from
produced-version
supported-by
```

因此对于最终 object \(o_j^T\)，可以查询：

\[
\mathcal{P}(o_j^T)
\]

得到其完整 provenance：

```text
最终 object
← object versions
← mapping events
← association decisions
← observations
← frame / mask / depth / feature
```

论文中这一层真正的核心不是 JSON 格式，而是：

> **persistent identity + versioned state + typed event provenance**

---

# 23.3 Evidence Validity Gate

定义一个 evidence validity function：

\[
V(G^P)\in\{0,1\}
\]

检查：

```text
referential integrity
UID uniqueness
artifact integrity
decision trace consistency
object-version consistency
event replayability
mapping invariants
```

只有：

\[
V(G^P)=1
\]

才运行后续 semantic diagnosis。

这个设计能避免：

> **用坏证据诊断好坏地图。**

---

# 23.4 Stage-aware Screening

对于 pipeline stage \(s\) 的 checker \(C_k\)：

\[
F_k=C_k(G^P;\theta_k)
\]

它不直接输出：

```text
error / correct
```

而输出：

\[
F_k=
(\mathcal{F},
\mathcal{H},
\mathcal{V},
\mathcal{M},
c,
r)
\]

其中：

- \(\mathcal{F}\)：proven facts；
- \(\mathcal{H}\)：hypotheses；
- \(\mathcal{V}\)：vetoes；
- \(\mathcal{M}\)：missing evidence；
- \(c\)：certainty；
- \(r\)：review route。

这是比“一堆 threshold”更值得写进论文的统一形式。

---

# 23.5 Cross-stage Root-Cause Hypothesis

对于共享同一 observation/object provenance 的 findings：

\[
\mathcal{C}_e
=
\{F_k\mid e\in scope(F_k)\}
\]

按 pipeline stage order 聚合上游与下游症状。

当前版本采用：

```text
earlier stage
+ severity
+ certainty
```

作为 primary root-cause hypothesis 排序。

以后如果加入 actual rollback/intervention，可以升级为：

```text
hypothesis
→ intervention
→ replay
→ downstream error disappearance
```

那时才真正向 causal repair 靠近。

---

# 23.6 Evidence-guided Review Sampling

将风险案例分为：

\[
\mathcal D_{\text{cal}}
\]

和：

\[
\mathcal D_{\text{diag}}
\]

分别用于：

```text
Calibration：
无偏/可加权地估计 checker precision

Diagnostic：
高效发现典型、高危、可行动错误
```

这是当前 v1.1 case builder 最适合论文抽象的部分。

---

# 24. 如果最终方法真的采用这一套，论文 Methods 可以怎么排

推荐结构：

```text
3. Method

3.1 Base Incremental Object-centric Mapping
    简述 ConceptGraphs association/fusion

3.2 Provenance-aware Mapping Ledger
    Frame
    Observation
    Object Version
    Decision Event
    Artifact Reference
    统一 lineage

3.3 Evidence Validity and Replay Consistency
    Evidence Gate
    mapping invariants

3.4 Stage-aware Error Screening
    Detection
    Segmentation / geometry
    Association
    Fusion
    Identity

3.5 Cross-stage Root-cause Aggregation
    共享实体
    上游/下游症状
    hypothesis / veto

3.6 Evidence-guided Verification
    Calibration cohort
    Diagnostic cohort
    Evidence Packet
    Human/VLM adjudication

3.7 Corrective Update
    [未来完成后再写]
```

这样逻辑是：

```text
怎么建图
↓
怎么记住建图历史
↓
怎么保证历史可信
↓
怎么从历史中发现异常
↓
怎么找到最早可疑根因
↓
怎么选择病例进一步确认
↓
怎么修
```

比：

```text
我们有 20 个 checker
规则 1 是……
规则 2 是……
```

强得太多。

---

# 25. 一段未来可以转化成论文语言的中文方法概述

> 我们在增量对象级 3D 建图流程上引入一个旁路的 provenance-aware evidence ledger。不同于仅保存最终对象地图，该账本为每个 2D/3D observation 分配稳定身份，并显式记录过滤轨迹、完整关联候选分数、关联决策、对象版本以及后处理合并事件，从而将最终对象与产生它的历史观测和状态更新建立可回放的依赖关系。在此基础上，我们首先通过 evidence validity gate 检查 UID、artifact reference、对象版本链、事件重放和成员归属等系统不变量，仅在证据自洽时执行后续诊断。随后，我们设计 stage-aware screeners，按照 detection、segmentation、geometry、association、fusion 和 object identity 的顺序，从多模态证据中提取风险案例。每个案例均显式区分可观测事实、错误假设、否决条件和缺失证据，而非直接将启发式阈值触发等价为真实错误。最后，我们依据共享 observation/object provenance 聚合跨阶段症状形成 root-cause hypotheses，并通过随机校准队列与高优先级诊断队列构建 Evidence Packets，为后续人工或 VLM 复核以及受控回滚提供依据。

这段可以作为未来 Method overview 的骨架。

但在正式论文中，必须根据后续实验结果决定：

```text
哪些 checker 最终保留
哪些只是工程工具
哪些能够成为真正贡献
```

---

# 26. 现在你自己应该怎样理解这套东西，而不是死记文件

以后遇到任何一个最终错误 object，只按下面 9 个问题倒着查。

---

## Q1. 最终谁组成了它？

看：

```text
final_membership
```

---

## Q2. 这些 observation 分别从哪来？

看：

```text
observations
frames
```

---

## Q3. 原始 proposal 本身靠谱吗？

看：

```text
bbox
mask
confidence
duplicate proposal
```

---

## Q4. 它有没有在 filter / mask subtraction 时被搞坏？

看：

```text
filter_trace
raw vs processed mask
removed pixels
```

---

## Q5. 2D→3D 是否合理？

看：

```text
depth
PCD
cluster stats
bbox
```

---

## Q6. 当时为什么关联到这个 object？

看：

```text
full similarity matrix
top candidates
margin
spatial vs visual
threshold
```

---

## Q7. 加进去以后 object 有没有突然变坏？

看：

```text
object_versions
FUSION_SHOCK
```

---

## Q8. 后处理有没有把两个 object 错 merge？

看：

```text
object_pair_decisions
OBJECT_MERGE
source / target versions
```

---

## Q9. 最后身份是否稳定？

看：

```text
duplicate final objects
weak support
class instability
```

如果这 9 个问题能顺畅回答，你就真正掌握了当前系统。

---

# 27. 如果老师让你两分钟解释“你这几天到底做了什么”

可以直接按这个逻辑讲：

> ConceptGraphs 的一个关键问题是增量关联和融合一旦出错，最终地图只留下结果，很难知道错误是从哪一步产生的。所以我第一步不是直接让 VLM 判断，而是先把在线建图过程做成可追溯的。现在每个 observation 都有永久 UID，并保存原始/处理后 mask、depth/3D 点云、完整 association candidate scores、object version 和 merge event，因此一个最终节点可以完整追溯到形成它的每一步。
>
> 在这个统一证据账本上，我又做了一层只读的 stage-aware audit。它先通过 Evidence Gate 检查证据链本身是否完整和可重放，然后分别在 detection、segmentation、geometry、association、fusion 和 object identity 阶段找风险。每条 finding 不直接说“这是错的”，而是明确区分事实、假设、反例和缺失证据。多个阶段在同一个 observation/object 上出现的异常再聚合成 root-cause hypothesis。最后系统把高价值案例打包成 RGB、mask、depth、3D 和历史上下文 Evidence Packet，供人工以及下一步 VLM 判断。
>
> 所以目前完成的并不是自动修复，而是“可追溯 + 可筛查 + 可复核”的诊断基础。下一步要验证这些筛查到底有多准、哪些错误真正影响最终地图，再决定第一类值得实现的回滚动作。

---

# 28. 最后用一句真正准确的话定义当前方法

如果需要给现在这部分取一个不夸张、又足够研究化的名字，我最推荐：

> **Provenance-aware, Stage-wise Auditing for Incremental Open-Vocabulary 3D Object Mapping**

中文：

> **面向增量开放词汇 3D 对象建图的可追溯分阶段审计框架**

它比：

```text
日志系统
查错器
规则系统
```

更准确。

也比现在就叫：

```text
Autonomous Causal Error Correction
```

诚实得多。

等以后真正加入：

```text
VLM adjudication
+
intervention
+
rollback/replay
+
repair verification
```

再升级成：

> **Evidence-Guided Self-Correcting 3D Scene Graph Mapping**

那时这个名字才配得上系统，而不是让标题替实验先完成科研任务。

---

# 29. 当前阶段最应该牢记的五个结论

1. **Evidence 的核心不是“保存更多字段”，而是把 Observation → Decision → Object Version → Final Object 串成 lineage。**

2. **Checker 的核心不是“阈值判错”，而是把事实、假设、veto 和缺失证据分开，并根据 stage 形成结构化风险。**

3. **Root-cause 目前是 provenance + stage-order 支持的 hypothesis，不是严格因果证明。真正因果性要靠后续 rollback/intervention 验证。**

4. **当前完成的是诊断基础设施和 incident endpoint 人工门。它能不能成为论文核心方法，不由 5587 条 Findings 决定，而由真实 endpoint error yield、专家因果追踪和 replay 修复收益决定。**

5. **人工标签只有在人和系统面对同一份 observation、decision state 与 final object 时才有效。代表视图必须声明覆盖率，缺失历史状态必须声明缺口；证据不足是 coverage failure，不是 checker false positive。**

---

# 30. 代码与结果索引

## 核心实现

```text
conceptgraph/utils/evidence.py
conceptgraph/audit/evidence_audit.py
conceptgraph/audit/layered_audit.py
conceptgraph/audit/configs/v1.yaml
conceptgraph/audit/configs/v2_validation.yaml
conceptgraph/audit/runner.py
scripts/build_validation_gate_review_packets.py
scripts/assemble_validation_gate_incidents.py
scripts/serve_validation_gate_incident_r1.py
scripts/compute_validation_gate_incident_metrics.py
scripts/generate_validation_gate_expert_queue.py

旧版追溯入口（不再作为当前人工协议）：
scripts/serve_validation_gate_r1.py
scripts/compute_validation_gate_metrics.py
```

## 测试

```text
tests/test_evidence.py
tests/test_evidence_audit.py
tests/test_layered_audit.py
tests/test_assemble_validation_gate_incidents.py
tests/test_serve_validation_gate_incident_r1.py
tests/test_validation_gate_incident_metrics.py
tests/test_generate_validation_gate_expert_queue.py

旧版兼容测试：
tests/test_serve_validation_gate_r1.py
tests/test_validation_gate_metrics.py
```

## 证据说明

```text
docs/ALI_MY_EVIDENCE.md
```

## room0 审计产物

```text
docs/audit_v1_artifacts/room0_20260819/
```

其中重点：

```text
full_audit/audit_summary.json
full_audit/findings.jsonl
full_audit/root_causes.jsonl
full_audit/evidence_validation.json
full_audit/audit_manifest.json
full_audit/audit_config.yaml

sample_case_finding_000002/
```

---

# 31. 参考论文

## 直接方法基础

**Gu et al., ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning.**

重点读：

```text
Object-based 3D Mapping
Object Association
Object Fusion
Scene Graph Construction
Limitations
```

当前 evidence/audit 的主要对象就是这条 mapping lifecycle。

---

## 相关系统化思维参考

**Shah et al., LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action.**

值得借鉴的不是它的导航算法，而是：

```text
系统模块边界清楚
完整系统效果和组件效果分开验证
失败归因到具体组件
```

这也是后续你验证 evidence/checker/repair 时应采用的论文表达方式。

---

## 开放词汇空间表示背景

**Huang et al., Visual Language Maps for Robot Navigation.**

它同样展示了：

```text
3D reconstruction noise
odometry drift
多视角 feature fusion
视觉语义歧义
```

如何影响空间语义表示。它不是当前查错器的直接实现来源，但有助于理解为什么 geometry、multi-view consistency 和 semantic ambiguity 都值得成为证据。

---

# 32. 版本说明

截至本文对应的当前 `ali-my`：

```text
Evidence schema:
0.2.0

Layered Audit base:
1.1.0

Final Endpoint Validation config:
2.1.0

Review Evidence:
2.1.0

Final Endpoint Metrics:
2.1.0
```

`Review Evidence 1.0.0`、finding 级 R1/R2 和旧 validation metrics 都保留作历史追溯，但已经退出正式有效性门。2026-08-19 保存的 room0 full audit 同样只是历史 `v1.0.0`，可用于理解方法演进，不能代表当前结果。

原文曾写“后续必须用 current code + v1.1 config + current sampler 重跑”。机器侧重跑、finding 去重和 final-endpoint 二次纠偏均已在 2026-08-20 完成：

```text
room0    200 帧正式运行，Evidence Gate PASS
office0  200 帧正式运行，Evidence Gate PASS
parity   evidence OFF / ON 非干扰比较 PASS

raw findings                         5587
eligible findings                    5577
final endpoint sets before block       98
duplicate findings collapsed         5479
fully blocked endpoints                 1
reviewable final endpoints             97（room0 69 + office0 28）
distinct final objects                 97
max reviews per final object            1
selection mode             endpoint_census（全部纳入，不是抽样）
packet / hash / final linkage       97 / 97 PASS
displayed assets checked                5069
人工标签                              0 / 97
```

当前尚未完成的不是机器重跑，而是：

```text
97 例简化 R1 最终状态复核
final endpoint census error / coverage / bounds 计算
仅对已确认 WRONG 的 expert causal trace
选定局部干预后的真实 replay 验证
```

因此旧版 room0 的 2455 findings、双场景的 5587 findings，乃至中间版的 2998 trigger-level incidents，都不能当成独立真实错误。当前统计单位是 97 个不同 final objects 构成的 flagged endpoint 普查；在人工标签完成前，它们仍不能叫真实错误。

---

# 33. 文档阅读建议

如果第一次看：

```text
先读：
0 → 34
```

先知道当前真正执行到哪里、你只需要回答什么。然后读：

```text
1 → 2 → 3 → 4 → 5 → 6
```

形成底层证据系统脑图。

第二遍重点：

```text
8 → 9 → 10 → 11 → 11.1 → 13 → 34.2～34.7
```

理解 checker 怎么产生候选，以及为什么最终必须先聚合 incident、再判断 endpoint。

准备写论文时：

```text
20 → 23 → 24 → 25 → 28
```

把代码实现转换成研究方法语言。

准备自己排错时：

```text
26
```

直接按 9 个问题查。

---

**最终目标不是让你记住 `observations.jsonl` 有多少字段，而是看到任意一个错误 object 时，你能清楚回答：它由哪些 observation 形成、每个 observation 经历了什么、当时为什么关联、哪一次状态更新开始异常、查错器依据什么怀疑它、还有什么反例，以及下一步需要什么证据才能决定是否修。**

做到这一点，这套系统才真正从“Codex 写出来的代码”变成你的方法。

---

# 34. 当前正式方案：final endpoint census v2.1

这一节是现在真正应执行的协议。先给结论：**用户不再标 160 条 findings 或 trigger incidents，而是复核 97 个不同的 final objects；每个 object 只出现一次。**

## 34.1 为什么 incident v2.0 仍然不够简化

逐条 finding 的旧 R1 在 0/160 时停止后，我们先做了第一轮去重：同一触发 observation 集合、同一最终谱系合成一个 incident。它确实把 5577 条 eligible findings 降成 2998 个 trigger-level incidents，并阻断了 221 个缺阶段快照的 incidents。

但在真正交给用户前，我们又检查了 160 条队列的终态重叠：

```text
160 个 trigger-level incidents
只有 55 个不同的非空 final-owner sets
147 / 160 落在重复 final-owner set 中
同一个 final object 最多重复出现 11 次
```

这说明“跨 checker 去重”还不等于“人工终态去重”。如果 R1 的问题是最终对象是否正确，那么让人因不同触发帧反复判断同一个 final object，仍然没有意义。中间版也在 0/160 时停止，没有标签需要迁移。

最终 `v2.1` 因而把身份键改成：

```text
同一 scene
+ 完全相同的 active final-object UID set
= 一个 final endpoint review unit
```

只有完全没有 active final owner 的孤立事件，才回退到 trigger observation 身份。当前正式 97 例全部都有 active final owner。

## 34.2 人、机器与统计现在面对的是同一个单位

一个 endpoint unit 内可以包含：

```text
多个 checker
多个阶段
多个触发 observations
多个历史 findings
```

但这些只作为内部证据，不再增加 R1 样本数。R1 的分母、网页待判对象和后续专家队列都使用同一个稳定 endpoint UID。

当前两个场景的结果还有一个很重要的性质：所有 endpoint sets 都是单个 final object，且不存在 object 跨 endpoint 重复：

```text
reviewable endpoints            97
distinct scene + final object   97
object 出现在多个 endpoints       0
每个 object 最大复核次数           1
```

因此这次不是“抽 97 条代表报警”，而是对全部 flagged final objects 做一次普查。

## 34.3 两个正式场景的真实归并结果

| 场景 | 原始 findings | eligible findings | final endpoints（阻断前） | 合并掉的重复 findings | 跨 checker endpoints | fully blocked | 可复核 endpoints | R1 全量纳入 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| room0 | 3720 | 3715 | 70 | 3645 | 57 | 1 | 69 | 69 |
| office0 | 1867 | 1862 | 28 | 1834 | 23 | 0 | 28 | 28 |
| 合计 | 5587 | 5577 | 98 | 5479 | 80 | 1 | 97 | 97 |

如何理解：

- 5587 是规则信号数量，不是错误数量。
- 5577 条 eligible findings 最终只指向 98 个不同 final endpoints，5479 条只是同一最终对象上的重复阶段/触发描述。
- 80 个 endpoints 同时被多个 checker 报警；checker 重合是内部支持信息，不是多个样本。
- 只有 1 个 endpoint 完全依赖未保留的阶段忠实状态，因此机器侧阻断。
- room0 69、office0 28 都低于每场景 80 的上限，所以 97 个可复核 endpoints 全部纳入。当前 `selection_mode=endpoint_census`，headline 不使用抽样权重。

## 34.4 页面怎样保证你明确知道“判断谁”

页面固定先展示最终对象，再展示辅助上下文：

```text
1. exact final-map geometry
2. 完整 membership、点数、bbox、帧跨度和类别统计
3. 完整成员中确定性抽取的代表视图
4. 必要时展开一条精确的代表性 trigger context
5. 必要时展开该 trigger 的 association record
```

最终图中：

```text
O1 [ENDPOINT]  = 当前真正要判断的 final object
O2/O3 [context] = 用来比较空间、身份或候选关系的上下文对象
```

ENDPOINT 固定排第一，图例、逐对象标题和页面文字卡片都明确写出角色，不需要靠猜颜色判断。

页面不会把一条代表 trigger 冒充全部历史：97 个 endpoints 内部共关联 3205 个 trigger observations，页面只显示 141 个精确代表 triggers，并明确显示 `displayed / linked_total`。完整 trigger UID 集合仍冻结在 worklist，供确认错误后的专家追踪。

全量投影核验：

```text
97 / 97  TRACEABLE
97 / 97  source case、worklist、review JSON 绑定一致
97 / 97  final UID、完整 membership、点数与 final pickle 一致
97 / 97  artifact hashes 一致
critical visual gaps = 0
displayed assets checked = 5069
```

R1 API 仍隐藏 checker、stage、subtype、linked findings、review score 和 cohort，避免规则名称暗示人工答案。

## 34.5 你只回答三个问题

### 1. `evidence_sufficient`

- `YES`：O1 的最终点云、成员、代表视图与上下文足以让你明确选 `CORRECT` 或 `WRONG`。
- `NO`：关键视角、对象边界或几何不足，仍有两种同样合理的解释。此时固定选择 `UNCLEAR`，不能猜。

### 2. `final_state`

- `CORRECT`：O1 最终身份、节点数量、成员和几何没有可见错误。即使中间出现重复 proposal、低 margin、暂态 CREATE 或其他报警，只要最后已经正确，就选它。
- `WRONG`：可见错误仍保留在 O1 对应的最终地图状态中。
- `UNCLEAR`：现有证据无法可靠区分对错。它不是 `CORRECT`，也不是 checker false positive。

### 3. `final_error_type`（仅 `WRONG` 时）

| 选项 | 怎么判断 |
|---|---|
| `FALSE_MERGE` | O1 把两个或更多现实物体错保存在同一节点中 |
| `FALSE_SPLIT` | 同一现实物体仍被拆成 O1 与另一个 final 节点 |
| `SPURIOUS_OBJECT` | O1 本身只是噪声、背景或无意义残片 |
| `MISSING_OBJECT` | 页面上下文清楚显示应有真实物体，但没有对应有效 final 节点 |
| `WRONG_MEMBERSHIP` | O1 是有效对象，但明显吸收了不属于它的 observations |
| `GEOMETRY_CORRUPTION` | O1 的最终位置、尺度、形状或点云结构明显损坏 |
| `SEMANTIC_IDENTITY_ERROR` | 几何对象存在，但稳定语义身份明显错误；同义词或近义类别不算 |
| `OTHER` | 以上都不合适；必须在备注写一句可见错误 |
| `NOT_APPLICABLE` | 最终状态不是 `WRONG` 时固定使用 |

如果同时看见多种错误，选择**最直接决定 O1 为什么错误的主类型**，在备注补充其他现象。R1 不再询问 checker 是否正确、最早根因、危害置信度、修复动作、修复范围或修复置信度。

## 34.6 当前服务器入口

正式验证根目录：

```text
/home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1
```

R1 已在 2026-08-21 完成并冻结：

```text
progress = 97 / 97
labels SHA-256 = f7db781367e6343fe01fc81a1fbcf48cc92917847dae4fb329eae19e4ff0861a
frozen copy = labels/labels_r1_frozen_20260821.jsonl
```

为避免冻结标签被误改，R1 的 `8765` 页面与 R2 的 `8766` 页面均已在完成后关闭。R2 完成状态为：

```text
protocol = final_endpoint_r2_v2_1
progress = 24 / 24
labels SHA-256 = 83de2b09a8d3022a555465e81dbb61e6d1ed4360915bfe745af43f020de9671b
frozen copy = labels/labels_r2_frozen_20260821.jsonl
```

若以后确实需要重新开放页面，可按原命令重启 loopback 服务后再建立隧道；当前评估不再需要打开网页。历史隧道命令是：

```bash
ssh -N -L 8766:127.0.0.1:8766 -p 64906 chenkejun@frp-van.com
```

服务运行时才打开：

```text
http://127.0.0.1:8766/
```

R1 与 R2 标签分别保存到：

```text
/home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1/labels/labels_r1.jsonl
/home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1/labels/labels_r2.jsonl
```

不要编辑 `r1_worklist.jsonl`、`r2_worklist.jsonl` 或冻结标签。两个历史目录完整保留但不要继续标：

```text
/home/chenkejun/beauty/conceptgraphs/validation_gate              # finding v1
/home/chenkejun/beauty/conceptgraphs/validation_gate_incident_v2  # trigger-incident 中间版
```

## 34.7 R1 完成后机器已经做了什么

R1 满 97 条后，机器已重新验证全部标签键、字段约束、证据哈希、final-pickle 绑定和系统门，并直接报告：

```text
evidence coverage
confirmed endpoint-error count
证据充分案例中的 endpoint-error rate
全 97 例 confirmed yield
把 UNCLEAR 纳入后的保守 lower / upper bounds
room0 / office0 分场景结果
```

不会把当前 headline 写成不必要的 calibration 加权估计。正式实测结果与解释见第 35 节。

随后只有：

```text
evidence_sufficient = YES
+ final_state = WRONG
```

进入专家队列。当前恰有 40 例满足条件，已经生成 `expert/confirmed_endpoint_error_queue.jsonl`。专家读取完整 linked findings、checkers、stages、3205 条 trigger 集合与事件链，建立最早因果阶段和候选干预。修复仍必须实际执行 `intervention → replay → final graph comparison`，只有重跑改善才能写 `repair_verified=true`。

旧的 32 例十字段复杂 R2 仍然退役。为检查简化终点标签本身是否稳定，现在另设 24 例 R2：页面与 R1 使用完全相同的三个字段，隐藏 R1 答案，覆盖两个场景、三种 R1 最终状态和本轮出现的五种终点错误类型。若仍由同一人完成，它只能报告 `intra-rater / test-retest` 稳定性，不能写成独立评审者 `inter-rater reliability`；两轮间隔短还可能因记忆使一致率偏高。

## 34.8 代码、测试与资源边界

当前入口：

```text
conceptgraph/audit/layered_audit.py
conceptgraph/audit/configs/v2_validation.yaml
scripts/assemble_validation_gate_incidents.py
scripts/build_validation_gate_review_packets.py
scripts/serve_validation_gate_incident_r1.py
scripts/compute_validation_gate_incident_metrics.py
scripts/generate_validation_gate_expert_queue.py
scripts/generate_validation_gate_endpoint_r2.py
scripts/compute_validation_gate_r2_agreement.py
docs/validation_gate_incident_labels_README.md
```

证据、审计、新旧协议兼容及新增 R2 回归共 `63 passed`。额外纳入 `test_general_utils.py` 时，当前基础环境会因既有的 `supervision` 缺失在收集阶段停止；这不是本次改动造成的失败，也不影响上述 63 项相关测试。

本轮 final-endpoint 审计、案例投影、测试和服务全部显式 `CUDA_VISIBLE_DEVICES=""`；没有使用 GPU3，也没有占用其他人的 GPU。旧运行、旧验证根、权重链接和他人的缓存均未删除。

## 34.9 这次简化为什么更适合论文

最终方法边界现在很清楚：

```text
screeners：高召回地产生结构化风险信号
endpoint builder：把重复阶段信号归并到唯一 final object
R1：只验证人真正看得到的最终状态
expert trace：只研究已经确认的 endpoint error 根因
replay：验证修复，而不是让人猜修复
```

这不是削弱方法，而是消除伪独立样本、阶段泄漏和无效人工推断。论文可以诚实报告“5587 条风险信号 → 97 个不同 flagged final objects → 人工确认错误 → replay 验证修复”，而不能把规则触发次数包装成错误数量。

---

# 35. 2026-08-21 R1 实测评估与精简 R2

这一节记录真实标签完成后的正式结果。若只想知道“现在得到什么结论、下一步为什么这样做”，直接读本节即可。

## 35.1 一句话结论

R1 证明这套旁路审计不是只会产生大量无效报警：在 97 个不同、可复核的 flagged final objects 中，人工确认 40 个最终状态仍然错误。但当前复合 `review_score` 没有把这些错误排到前面，因此正确动作是：

```text
保留 evidence ledger、endpoint 去重和证据页面
        ↓
废弃“review_score 可以直接代表错误概率”的解释
        ↓
40 个确认错误进入 expert causal trace
        ↓
只选因果集中、可局部干预的错误族做 intervention / replay
        ↓
重跑后 final graph 真正改善，才能声称修复有效
```

正式状态是：

```text
PROCEED_TO_EXPERT_TRACE
repair gate = PENDING_EXPERT_TRACE_AND_REPLAY
```

它既不是“整套方法失败”，也不是“自动修复已经成功”。它说明证据化诊断找到了大量真实终点错误，但排序分数和修复环节仍需要下一阶段验证。R2 进一步说明：R1 的 `WRONG` 判断很稳定，但 `CORRECT/UNCLEAR` 与 `WRONG` 的边界仍有复核者内波动，最终标签不能被描述成完全无主观性。

## 35.2 先确认这 97 个标签可以被统计

正式计算前没有直接相信网页上的 `97/97`，而是重新做了以下检查：

| 检查 | 结果 |
|---|---|
| R1 标签行数 / worklist 行数 | 97 / 97 |
| endpoint 键缺失、额外或重复 | 0 / 0 / 0 |
| 字段枚举与条件逻辑 | PASS |
| R1 标签与 97 个 endpoint 一一对应 | PASS |
| 系统 Evidence Gate、Audit Gate、parity | 全部 PASS |
| 人类页面与系统 source case、final pickle、资产哈希 | PASS |
| 重新核对的页面资产 | 5069 个 |
| R1 冻结 SHA-256 | `f7db781367e6343fe01fc81a1fbcf48cc92917847dae4fb329eae19e4ff0861a` |

正式冻结文件为：

```text
labels/labels_r1_frozen_20260821.jsonl
```

冻结副本与原 `labels_r1.jsonl` 哈希完全一致。R1 服务随后关闭，避免做 R2 时误改首轮答案。

## 35.3 R1 headline：40 个确认错误，不是 5587 个

| 场景 | 可复核 endpoint | 证据充分 | CORRECT | WRONG | UNCLEAR | 证据充分条件下错误率 |
|---|---:|---:|---:|---:|---:|---:|
| room0 | 69 | 67 | 40 | 27 | 2 | 27 / 67 = 40.30% |
| office0 | 28 | 28 | 15 | 13 | 0 | 13 / 28 = 46.43% |
| 合计 | 97 | 95 | 55 | 40 | 2 | 40 / 95 = 42.11% |

证据充分率是：

```text
95 / 97 = 97.94%
```

因为两例人工选择 `UNCLEAR`，对 97 个可复核 endpoints 的错误率不能只写一个假装精确的点估计。保守范围是：

```text
lower = 40 / 97 = 41.24%
upper = (40 + 2) / 97 = 43.30%
```

若把机器侧完全证据阻断的 1 个 endpoint 也放回最初 98 个 flagged endpoints，范围是：

```text
lower = 40 / 98 = 40.82%
upper = (40 + 2 + 1) / 98 = 43.88%
```

这两个范围分别回答“在 97 个可人工复核 endpoints 中怎样界定不确定性”和“把 1 个机器阻断也按最保守情况计入时怎样界定”。它们不是从样本外推总体的置信区间，因为这次对本轮全部 flagged endpoints 做的是普查。

room0 与 office0 的条件错误率相差约 6.13 个百分点，但只有两个具体场景，且 2×2 Fisher 检验 `p=0.651`。因此目前只能描述这两个 run 的差异，不能据此声称 office0 在一般意义上显著更差。

## 35.4 错误组成：先看什么最值得追踪

| 最终错误类型 | 数量 | 占 40 个确认错误 | room0 | office0 |
|---|---:|---:|---:|---:|
| `SEMANTIC_IDENTITY_ERROR` | 17 | 42.5% | 8 | 9 |
| `GEOMETRY_CORRUPTION` | 11 | 27.5% | 10 | 1 |
| `SPURIOUS_OBJECT` | 6 | 15.0% | 4 | 2 |
| `FALSE_SPLIT` | 3 | 7.5% | 2 | 1 |
| `FALSE_MERGE` | 3 | 7.5% | 3 | 0 |

语义身份错与几何损坏合计 28 / 40，即 70%。这使它们成为专家追踪的首要“错误族”，但频数本身还不能决定修哪个：还要看最早因果阶段是否集中、能否局部干预、replay 后是否改善。尤其是 room0 的几何损坏 10 例与 office0 的语义身份错 9 例，值得分别检查是否共享同一因果链，不能先假定它们来自同一个 checker。

## 35.5 人工标签本身有没有明显阶段性或提交异常

| 复核行为 | 结果 |
|---|---:|
| 总复核时间 | 4765.7 秒，约 79.4 分钟 |
| 每例中位时间 | 28.0 秒 |
| 四分位范围 | 15.2～50.3 秒 |
| 少于 5 秒 | 0 例 |
| 少于 10 秒 | 6 例，全部为 `CORRECT` |
| `WRONG` 中位 / 最短 | 34.95 秒 / 13.7 秒 |
| 有文字备注 | 35 例 |

这些行为数据没有显示批量秒点、空标签或把证据不足硬猜成对错的异常。两例 `UNCLEAR` 都在 room0，备注分别说明“视角不够，只看到一点点”和“证据不够不太确定”，与 `evidence_sufficient=NO` 的选择一致。

这不等于单复核者标签天然可靠。它只说明 R1 文件在覆盖、逻辑和基本复核行为上可以进入统计；主观重复稳定性由下面的精简 R2 检查。

## 35.6 对 screeners 的真正评估：能找到错，但当前排序分数方向不对

首先要区分两个问题：

1. 这些规则筛出的 endpoint 中有没有真实错误？有，40 / 97 已确认。
2. 当前 `review_score` 能不能把最可能错误的排在前面？本轮不能。

对 95 个可判 endpoints，以“分数越高越应优先”为方向：

| 排序诊断 | 结果 |
|---|---:|
| 真实错误基线 | 40 / 95 = 42.11% |
| ROC AUC | 0.420 |
| Average Precision | 0.366 |
| Top 5 | 1 / 5 = 20% |
| Top 10 | 2 / 10 = 20% |
| Top 20 | 6 / 20 = 30% |
| Top 40 | 13 / 40 = 32.5% |

AUC 低于 0.5，所有 top-k precision 又低于 42.11% 基线；旧分区也呈现同一现象：`calibration_random` 是 31 / 68 = 45.59%，`diagnostic_priority` 反而只有 9 / 29 = 31.03%。因此 `review_score` 不能在论文中写成错误概率或有效优先级，下一版必须重新学习/校准组合方向，至少先按 R1 结果做 leave-one-scene-out 验证，避免在这两个场景上直接调到过拟合。

linked checker 的数字可以用于安排专家追踪，但不能当成独立规则的因果精度，因为同一 endpoint 可同时属于多个 checker：

| linked checker | 涉及 endpoints | 确认错误 | 证据充分条件 precision | 覆盖 40 个错误 |
|---|---:|---:|---:|---:|
| `ASSOC-002` | 59 | 31 | 53.45% | 77.5% |
| `DET-002` | 26 | 15 | 60.00% | 37.5% |
| `OBJ-003` | 9 | 4 | 57.14% | 10.0% |
| `OBJ-005` | 24 | 11 | 45.83% | 27.5% |
| `FUSE-007` | 26 | 11 | 42.31% | 27.5% |

`ASSOC-002` 最适合优先做“高覆盖候选入口”，`DET-002` 适合做“较高纯度候选入口”。但 association 阶段一共触及 91 / 97 endpoints 和 39 / 40 errors，这也说明“触及很多错误”部分来自覆盖极广，不等于 association 就是 39 个错误的根因。根因必须回到完整事件链逐例确认。

还有一个不能从这次数据回答的问题：未被任何 screener 标记的普通 final objects 没有人工真值，因此当前只能报告 flagged-endpoint yield / precision，不能报告全地图层面的 recall、specificity 或 false-positive rate。若论文需要这些量，必须另抽一组未报警 final objects 做盲审对照。

## 35.7 精简 R2：检查同一终点判断能否重复，不恢复复杂问卷

R2 固定为 24 例，占 R1 的 24.7%，设计如下：

| 分层 | R2 数量 |
|---|---:|
| room0 / office0 | 17 / 7 |
| R1 `CORRECT / WRONG / UNCLEAR` | 12 / 10 / 2 |
| R1 出现的错误类型 | 五类全部至少一例 |

R1 状态只用于机器端分层抽样，不写入 `r2_worklist.jsonl`，也不由 API 返回。实测检查结果：24 行 worklist 中没有 `reviewer_id`、`evidence_sufficient`、`final_state`、`final_error_type`、`review_seconds` 或 `notes`；首例 API 的 `label` 为 `null`。R2 页面仍只显示同一份哈希锁定 endpoint evidence，并填写与 R1 相同的三个字段。

这 24 例是为了让少数状态和五类错误都有机会被重复检验，属于有意分层的稳定性样本。因此：

- 可以报告证据充分性一致率、最终状态一致率、三字段完全一致率、错误类型一致率、confusion matrix 和 Cohen's kappa。
- 不能用 24 例重新估计 97 例的错误率。
- 若由同一人完成，只能写 `intra-rater test-retest`。
- 若 R1 后马上做 R2，要把“短间隔记忆可能抬高一致率”写入限制。
- 真正的 `inter-rater reliability` 仍需要另一位不知道 R1 答案的人独立复核。

R2 完成后运行：

```bash
CUDA_VISIBLE_DEVICES="" python scripts/compute_validation_gate_r2_agreement.py \
  --validation-root /home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1 \
  --r1-labels /home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1/labels/labels_r1_frozen_20260821.jsonl \
  --relationship same-reviewer
```

未满 24 例时固定返回 `NOT_READY`，不会用部分标签提前计算一个看似完整的一致率。

本轮已经完成 24/24，R2 冻结 SHA-256 为：

```text
83de2b09a8d3022a555465e81dbb61e6d1ed4360915bfe745af43f020de9671b
```

实际一致性结果：

| 比较项 | 一致数 | 一致率 | Cohen's kappa |
|---|---:|---:|---:|
| 证据充分性 | 23 / 24 | 95.83% | 0.647 |
| 最终状态 | 20 / 24 | 83.33% | 0.706 |
| 三字段完全一致 | 19 / 24 | 79.17% | 0.722 |
| 两轮都判 `WRONG` 时的错误类型 | 9 / 10 | 90.00% | 0.861 |

24 例仍是小样本：最终状态一致率的 Wilson 95% 区间为 64.15%～93.32%，三字段完全一致率为 59.53%～90.76%。因此这里应报告具体分子/分母、区间和混淆矩阵，不把单个 κ 值包装成“可靠性已经证明”。

状态迁移比单个一致率更能说明问题：

| R1 → R2 | 数量 |
|---|---:|
| `WRONG → WRONG` | 10 |
| `CORRECT → CORRECT` | 9 |
| `CORRECT → WRONG` | 3 |
| `UNCLEAR → UNCLEAR` | 1 |
| `UNCLEAR → WRONG` | 1 |

没有任何 `WRONG → CORRECT/UNCLEAR`。也就是说，R1 已确认的 10 个错误在第二轮全部保留；一旦两轮都认为它是错的，主错误类型也有 9/10 一致。主要不稳定性不是“错误后来被推翻”，而是第二轮把 3 个首轮正确和 1 个首轮不清楚改判成错误；另有 1 例保持 `WRONG` 但从 `SPURIOUS_OBJECT` 改为 `SEMANTIC_IDENTITY_ERROR`。

这支持两个同时成立的结论：

1. 已确认 `WRONG` 的正例相当稳定，可以继续进入因果追踪。
2. `CORRECT/UNCLEAR` 与 `WRONG` 的判断阈值还不够稳定，R1 可能偏保守，也可能是第二轮因熟悉页面而提高了报错敏感度；同一人、短间隔 R2 无法区分这两种解释。

因此没有用 R2 自动覆盖 R1，也没有把 4 个新增 `WRONG` 偷加进 40 例确认错误。5 个存在任一字段分歧的 endpoint 已单独写入 `expert/r2_disagreement_queue.jsonl`，状态固定为 `PENDING_HUMAN_ADJUDICATION`。若后续需要一个最终统一真值，应让另一位独立复核者只裁决这 5 例；若暂不裁决，则论文同时报告 R1 主分析和这里的 repeatability sensitivity analysis。

R2 总用时 507.2 秒，中位每例 13.65 秒；第二轮明显快于 R1，符合重复熟悉效应，也进一步要求把短间隔记忆偏差写进限制。

## 35.8 当前正式产物在哪里

| 产物 | 路径 | 作用 |
|---|---|---|
| 冻结 R1 | `labels/labels_r1_frozen_20260821.jsonl` | 正式首轮人工真值 |
| R1 完整指标 | `metrics/incident_endpoint_metrics.json` | 总体、场景、错误类型、排序、checker、时长 |
| 分场景 / linked checker / linked stage CSV | `metrics/metrics_by_*.csv` | 便于表格检查 |
| 正式决策 | `decision.md` | `PROCEED_TO_EXPERT_TRACE` |
| 40 例专家队列 | `expert/confirmed_endpoint_error_queue.jsonl` | 后续因果追踪与 replay 入口 |
| R2 分层设计 | `r2_selection_manifest.json` | 记录种子、数量、哈希与不泄漏约束 |
| R2 空白工作清单 | `labels/r2_worklist.jsonl` | 24 个稳定性复核 endpoint |
| R2 页面证据清单 | `r2_review_evidence_manifest.json` | 绑定 R2 worklist 与原始冻结证据 |
| 冻结 R2 | `labels/labels_r2_frozen_20260821.jsonl` | 正式第二轮人工选择 |
| R2 一致性结果 | `metrics/r2_repeatability.json`、`r2_repeatability.md` | intra-rater 统计、混淆矩阵与分歧 |
| R2 分歧队列 | `expert/r2_disagreement_queue.jsonl` | 5 例待独立裁决，不混入 40 例确认错误 |

所有评估、R2 生成、测试和服务继续显式禁用 CUDA；没有占用 GPU3，也没有占用任何其他 GPU。

## 35.9 现在能写进论文与不能写进论文的话

可以写：

> 在两个冻结的 Replica 200 帧运行中，5587 个多阶段风险 findings 经 final-object endpoint 去重后对应 98 个不同 flagged endpoints。1 个因证据不足被机器阻断，其余 97 个进行完整人工普查；95 个可判，其中 40 个最终错误得到确认。确认错误以语义身份错误和几何损坏为主。当前启发式复合排序分数未表现出有效优先级，因此后续工作转向确认错误的因果追踪与干预重放，而不把规则分数解释为错误概率。24 例同一复核者重复复核的最终状态一致率为 83.33%（κ=0.706）；R1 的 10 个 `WRONG` 全部在 R2 保持 `WRONG`，但 4 个非错误/不清楚案例改判为错误，说明正例稳定而决策边界仍需独立裁决验证。

不能写：

- “发现了 5587 个真实错误”；
- “40 个错误都由 association 引起”；
- “R1 已证明修复有效”；
- “同一人立刻做的 R2 是独立评审者一致性”；
- “只看 flagged endpoints 就测得了全地图 recall / specificity”。

这组边界让结果看起来更克制，但也更可信：每个数字都对应一个明确分母，每个阶段只回答自己有证据回答的问题。

---

# 附录 A：已经退役的 trigger-incident v2.0 设计记录

下面保留的是促成二次纠偏的中间方案。它没有产生人工标签，路径和数字只用于追溯，**不要按其中的 160 例入口或统计口径操作**。

## A.1 为什么旧版 R1 在 0/160 时停止

旧版把每条 checker finding 当作一个人工案例，并要求一次填写：

```text
证据是否充分
finding 是否正确
根因阶段是否正确
物理解释
最终危害
危害置信度
修复动作
修复范围
修复置信度
替代解释 / 备注
```

深入检查后发现，这种设计同时混淆了四件不同的事：

```text
规则是否触发
事件发生时是否异常
异常是否仍保留在最终地图
什么修复经过重跑才真正有效
```

更重要的是，同一物理事件会跨阶段重复报警。例如同一对重复 proposal 可能同时成为：

```text
DET-001
SEG-004
ASSOC-004
```

而 `SEG-002` 与 `GEO-003` 又读取同一个 `pre_dbscan.second_cluster_ratio`。如果直接按 finding 标注，同一个事件会被算两到三次；统计得到的是“规则命中精度”，不是“独立真实错误精度”。

旧 160 例中还存在 26 例关键历史快照缺口。典型情况是 checker 判断 DBSCAN 前的多簇状态，网页却只能显示 DBSCAN 后点云。让人面对后状态去猜前状态，结论不具备有效性。

因此旧版在尚未产生任何标签时停止，这是好事：没有错误标签需要迁移，也没有为了保住旧设计而继续投入人工成本。

## A.2 中间版单位：trigger-level Incident

新版不再把 checker finding 当作人工单位，而是先构造唯一物理 incident：

```text
同一 scene
+ 完全相同的触发 observation UID 集合
+ 完全相同的 active final-object lineage
= 一个稳定 incident_uid
```

没有触发 observation 的 object-level finding，则按最终对象谱系归并。

这带来三个直接变化：

1. 多个 checker 可以成为同一个 incident 的机器证据标签，但不会让人重复标注。
2. 页面可以明确告诉人：这些触发 observations 最终进入了一个、多个还是零个 active final objects。
3. “多个 observation 后来收敛到同一 final object”只是机器结构事实，不直接替人判定正确；它用于避免把暂态 false split 自动说成最终错误。

## A.3 中间版如何阻断缺口 incident

当前 2026-08-20 正式 runs 没有保存以下历史 3D 坐标快照：

| Checker | 缺失的阶段忠实状态 |
|---|---|
| `SEG-002` | DBSCAN 前点坐标 |
| `GEO-003` | DBSCAN 前点坐标 |
| `GEO-005` | denoise 前后完整对象点云 |
| `FUSE-007` | fusion 前后对象版本点云 |

这些 checker 仍保留在机器审计和 blocked report 中，不会被删除或伪装成别的图。

规则是：

```text
如果一个 incident 只有上述缺口 checker 支持
→ 不进入 R1

如果同一 incident 还有其他具备忠实证据的 checker
→ 可由可复核 finding 代表该 incident
→ 缺口 checker 只作为内部谱系信息，不引导 R1
```

因此新版 160 个正式 incident 的关键视觉缺口为 0，而不是让人自己在缺口案例中艰难选择 `PARTIAL`。

## A.4 中间版 trigger-level 去重结果

“eligible finding”只包括需要有效性复核的 `LIKELY_MAPPING_CONFLICT` 和 `AMBIGUOUS_MAPPING_RISK`；两场景另有共 10 条 `INSUFFICIENT_EVIDENCE` finding，不进入抽样母体。

| 场景 | 原始 findings | eligible findings | 证据阻断前 unique incidents | 被合并的重复 findings | 跨 checker incidents | 因阶段证据缺失阻断 | 可复核 incidents | 正式抽样 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| room0 | 3720 | 3715 | 2049 | 1666 | 1018 | 175 | 1874 | 80 |
| office0 | 1867 | 1862 | 949 | 913 | 542 | 46 | 903 | 80 |
| 合计 | 5587 | 5577 | 2998 | 2579 | 1560 | 221 | 2777 | 160 |

如何理解这张表：

- 5577 条可复核 finding 先变成 2998 个独立 incident，说明 2579 条只是同一事件的重复阶段描述。
- 其中 1560 个 incident 至少被两个 checker 同时报出；checker 重合是证据，不再是多个样本。
- 221 个 incident 因当前 runs 没有保存它们真正需要的历史状态而被阻断。
- 剩余 2777 个 reviewable incidents 才是当前人评母体。
- 每场景仍抽 40 个 `calibration_random` 和 40 个 `diagnostic_priority`，总计 160。保持 160 是为了保留场景/队列统计能力，不代表人工复杂度没下降。

旧表每例至少约 10 个必填记录，160 例最低约 1600 项；新表每例最多 3 个选择，合计最多约 480 项，必填负担下降约 70%，而且不再重复标同一物理事件。

## A.5 中间版已开始简化的 R1 字段

### 问题 1：证据足以判断最终状态吗

```text
YES
```

表示最终点云、最终成员、代表视图和对象上下文足以让你选择 `CORRECT` 或 `WRONG`。

```text
NO
```

表示当前材料不能可靠区分对错。此时最终状态固定选 `UNCLEAR`，不能把“看不清”当成误报。

### 问题 2：最终地图对象状态

```text
CORRECT
```

最终身份、节点数量、成员和几何没有可见错误。即使上游出现过重复 proposal、低 margin 或暂态 CREATE，只要最终状态已正确，就选它。

```text
WRONG
```

错误仍真实存在于最终地图。

```text
UNCLEAR
```

证据不能可靠判断。它不是 `CORRECT`，也不是 checker false positive。

### 问题 3：仅在 WRONG 时选择最终错误类型

| 选项 | 含义 |
|---|---|
| `FALSE_MERGE` | 多个真实物体被保留在同一最终节点 |
| `FALSE_SPLIT` | 同一真实物体仍保留成多个最终节点 |
| `SPURIOUS_OBJECT` | 最终节点只是噪声、背景或残片 |
| `MISSING_OBJECT` | 应存在的真实对象没有有效最终节点 |
| `WRONG_MEMBERSHIP` | 有效 observation 进入了错误最终对象 |
| `GEOMETRY_CORRUPTION` | 最终点云、位置、尺度或形状明显错误 |
| `SEMANTIC_IDENTITY_ERROR` | 几何节点存在，但稳定语义身份明显错误；同义词不算 |
| `OTHER` | 以上均不合适，备注一句可见错误 |
| `NOT_APPLICABLE` | 最终状态不是 WRONG |

R1 明确不再询问：

```text
checker 对不对
最早根因在哪一阶段
危害置信度
应该采取什么修复
修复范围和修复置信度
```

## A.6 页面为何开始改成 final object first

新版固定顺序是：

```text
1. exact final-map objects
2. 完整成员统计与代表视图覆盖
3. 必要时展开同源 trigger observations
4. 必要时展开当时 association 记录
```

而不是先展示 checker 名称和错误假设。API 返回给 R1 页面的数据也移除了：

```text
checker_id
linked checker_ids
stage / stages
subtype / subtypes
review score
sampling cohort / weight
```

这些信息继续保留在冻结 worklist 和 `review_evidence.json` 中，供后续专家追踪，但不会锚定第一轮最终状态判断。

页面中的 final object 来自 manifest 哈希锁定的最终 map pickle，并核对：

```text
object UID
完整 member observation 集合
点数
bbox
完整 point coordinates
```

当前正式 manifest 结果：

```text
160 / 160 incident = TRACEABLE
critical visual gaps = 0
artifact hashes match = true
final-object linkage exact = true
displayed assets checked = 10761
```

## A.7 R1 后阶段边界

只有：

```text
evidence_sufficient = YES
+ final_state = WRONG
```

才进入专家因果追踪队列。

专家阶段可以查看隐藏于 R1 的全部 linked findings、checker、stage、object version 与事件链，然后填写：

```text
earliest causal stage
causal chain
root evidence refs
repair hypothesis
intervention plan
```

但这仍然只是修复假设。最后必须实际执行：

```text
intervention
→ replay
→ 对比 final object graph
→ repair_verified = true / false
```

因此论文中应分别报告：

```text
incident endpoint precision
evidence coverage
confirmed endpoint error type distribution
expert causal attribution（若完成）
verified repair success（若完成 replay）
```

不能再把 root stage accuracy、人工猜测 repair action 和真实修复收益混成一个指标。

## A.8 已退役的中间版服务器入口

正式新版验证根目录：

```text
/home/chenkejun/beauty/conceptgraphs/validation_gate_incident_v2
```

旧版 finding 级验证目录继续保留：

```text
/home/chenkejun/beauty/conceptgraphs/validation_gate
```

正式服务只绑定服务器 loopback：

```text
127.0.0.1:8765
```

保持原 SSH 隧道后，本地仍打开：

```text
http://127.0.0.1:8765/
```

当前状态：

```text
protocol  incident_endpoint_r1_v2
progress  0 / 160
labels    validation_gate_incident_v2/labels/labels_r1.jsonl
```

你只需要完成这 160 个最终对象复核。不要编辑 `r1_worklist.jsonl`，也不要回到旧页面继续填旧字段。

## A.9 中间版计划的后续机器步骤

首先运行 incident endpoint metrics：

```bash
/opt/anaconda3/bin/python scripts/compute_validation_gate_incident_metrics.py \
  --validation-root /home/chenkejun/beauty/conceptgraphs/validation_gate_incident_v2
```

它会报告：

```text
evidence coverage
calibration weighted endpoint-error precision
full-sample lower / upper bounds
priority conditional precision
priority confirmed error yield
by scene / representative checker / representative stage diagnostics
```

当前没有人工标签，所以工具正确返回 `NOT_READY`，不会制造提前结论。

随后只为确认错误生成专家队列：

```bash
/opt/anaconda3/bin/python scripts/generate_validation_gate_expert_queue.py \
  --validation-root /home/chenkejun/beauty/conceptgraphs/validation_gate_incident_v2
```

R1 未完成时同样返回 `NOT_READY`。专家队列不会把 `CORRECT` 或 `UNCLEAR` incident 重新拉回复杂因果表。

## A.10 中间版代码入口

```text
conceptgraph/audit/layered_audit.py
conceptgraph/audit/configs/v2_validation.yaml
scripts/assemble_validation_gate_incidents.py
scripts/build_validation_gate_review_packets.py
scripts/serve_validation_gate_incident_r1.py
scripts/compute_validation_gate_incident_metrics.py
scripts/generate_validation_gate_expert_queue.py
docs/validation_gate_incident_labels_README.md
```

定向及证据链回归：

```text
54 passed
```

无筛选运行仓库全部 pytest 时，仍有三个与本任务无关的原有收集问题：示例脚本引用不存在的作者本机图片绝对路径、环境缺少 `transformers`、环境缺少 `supervision`。这些问题没有被伪装成新版失败，也没有为了让数字好看而修改旧示例。

本次 incident 审计、证据投影、API 检查和服务切换全部显式禁用 CUDA；没有使用 GPU3，也没有占用其他人的 GPU。

## A.11 中间版带来的第一轮纠偏

方向并不是从“证据审计”退回普通人工看图。真正的变化是：

```text
保留：可追溯 ledger、分阶段 screeners、哈希锁定证据、最终谱系

删除：以 rule hit 冒充独立错误、重复标注、让 R1 猜根因和修复

新增：incident 去重、阶段证据 blocker、final endpoint first、真实 replay 验证边界
```

这使人工任务更简单，却让方法论更严格：人只回答自己真正看得到的问题，机器负责去重与谱系，专家只处理已经确认的最终错误，修复必须接受干预重跑检验。
