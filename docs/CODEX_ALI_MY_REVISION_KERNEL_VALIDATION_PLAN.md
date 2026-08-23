# Codex 执行方案：在 `ali-my` 上先验证 Revision Kernel（暂不加入队友新关系方法）

> **用途**：直接给 Codex 执行。
>
> **本阶段目标**：不等待队友的新 relation backend；以 `ali-my` 为开发基线，在不改变 `ali-dev` 原有建图与关系生成逻辑的前提下，先证明“证据化历史追踪 → 局部 counterfactual replay → baseline edge closure → 安全 derived commit”是可实现、可重复、可评价的。
>
> **工作基线**
>
> - 上游算法语义：`ali-dev`
> - 当前开发基线：`ali-my @ 66c109dc042fdf21936e933aa80f4e8307ddd4d6`
> - `ali-my-VLM` 只作为设计/代码参考，不直接在其上继续堆最终方法。
>
> **重要原则**：先把 replay engine 与 controlled experiment 做对，再把 `ali-my-VLM` 的 VLM diagnosis 改造成 repair-constraint generator。不要同时调 VLM 和 replay。

---

# 0. 绝对约束

Codex 必须遵守：

1. **不得修改 `ali-dev`。**
2. **不得覆盖 `ali-my-VLM` 的冻结验证结果、R1/R2 labels、97-case 结果。**
3. 原 `ali-my` evidence sidecar 的核心性质必须保持：
   - `revision.enabled=false` 时，建图算法输出与当前 `ali-my/ali-dev` 保持 parity；
   - evidence/revision logging 失败不能偷偷改变 mapping decision。
4. 本阶段**不得实现新的 relation algorithm**。
5. relation 只使用 `ali-dev` 现有逻辑：
   - frame-level `gobs['edges']`
   - `process_edges(...)`
   - `MapEdgeMapping`
   - 原有 edge cleanup / merge index update。
6. 不允许用 R1/R2 人工答案参与算法 inference。
7. controlled corruption 的 oracle repair constraint 只能用于**验证 replay engine**，必须与最终 VLM method 分开报告。
8. 所有 repair 首先写到 shadow/derived state，禁止原地覆盖 baseline map。
9. 不把“最终 pickle patch”当作 counterfactual replay。
10. 所有新行为必须有 feature flag，默认关闭。

---

# 1. 建议新分支

从 `ali-my` 建：

```bash
git checkout ali-my
git pull
git checkout -b exp/ali-my-revision-kernel-v0
```

不要直接把 `ali-my-VLM` merge 进来。

后续只按需要手工移植：

- VLM typed schema；
- audit/diagnosis prompt 思想；
- VLM client。

不要移植 direct overlay 作为核心 executor。

---

# 2. 第一阶段需要新增的目录

建议：

```text
conceptgraph/
  revision/
    __init__.py
    schemas.py
    provenance.py
    lineage.py
    maturity.py
    incidents.py
    tickets.py
    constraints.py
    trace.py
    dependency.py
    materialize.py
    replay.py
    baseline_relations.py
    conflict.py
    transactions.py
    verify.py
    corruptions.py
    metrics.py

conceptgraph/revision/configs/
    v0.yaml

scripts/
    run_revision_controlled_experiment.py
    evaluate_revision_controlled.py
    inspect_revision_case.py
    build_revision_report.py

tests/
    test_revision_provenance.py
    test_revision_lineage.py
    test_revision_corruption.py
    test_revision_replay.py
    test_revision_baseline_relations.py
    test_revision_conflicts.py
    test_revision_transaction.py
    test_revision_parity.py
```

不要为了“架构优雅”过度拆文件；如果部分文件代码很短可以合并，但职责必须保持清楚。

---

# 3. 配置

新增 Hydra/普通配置：

```yaml
revision:
  enabled: false

  mode: controlled_validation

  log_dir_name: revision

  maturity:
    enabled: true
    min_unique_frames_for_structural_repair: 3
    min_members_per_split_group: 2
    require_non_degenerate_geometry: true

  replay:
    strategy: dependency_local
    rebuild_features: true
    rebuild_geometry: true
    replay_postprocess_events: true

  relation_backend:
    type: ali_dev_baseline
    rebuild_scope: full_temporal_baseline   # v0先保证正确；后续改local

  transaction:
    shadow_only: true
    allow_commit_to_derived_map: true
    max_rebase_count: 3

  debug:
    save_snapshots: true
    save_event_trace: true
```

这些阈值是 MVP safety gate，不得写成论文最终固定理论阈值；后续需要 ablation。

---

# 4. P0-1：先做 revision-disabled parity

这是第一项任务，不是最后补测试。

## 目标

`revision.enabled=false` 时：

- object 数；
- object IDs/membership；
- class；
- pcd hash 或数值等价；
- edge list；
- relation types

必须与原 `ali-my` 对照一致。

如果 byte-level pickle 因非决定性序列化不同，则做 semantic parity：

```text
object membership identical
object feature allclose
point cloud deterministic hash/allclose
edge tuple set identical
```

如果现有 `scripts/compare_mapping_parity.py` 适合，复用/移植，不重复造轮子。

## Acceptance

```text
PASS_REVISION_DISABLED_PARITY
```

不通过前禁止继续改 mapping core。

---

# 5. P0-2：实现 `ProvenanceIndex`

输入：

```text
evidence/
  observations.jsonl
  associations.jsonl
  mapping_events.jsonl
  object_versions.jsonl
  object_pair_decisions.jsonl
  final_membership.json
```

至少提供：

```python
get_observation(obs_uid)
get_association_for_obs(obs_uid)
get_event(event_uid)
get_object_version(version_uid)
get_versions_for_object(object_uid)
get_parent_versions(version_uid)
get_child_versions(version_uid)
get_current_version(object_uid)
get_member_observations(version_uid)
get_incident_edge_events(object_uid)
events_after(sequence)
```

建立内存索引，避免每次扫描 JSONL。

## 注意

现有 evidence 已有 `event_sequence`、parent events、input/output versions，不要重新定义第二套相互冲突的 event ID。

---

# 6. P0-3：实现真正的 `LineageIndex`

当前 evidence 的 origin-based lineage 不能完整表达 future split/merge revision。

本阶段保持原日志只读，在 revision 层构建 DAG：

```text
object_version_uid
   ↓ parent/child
object_version_uid
   ↓ merge
new active version
```

接口：

```python
resolve_descendants(version_uid)
resolve_ancestors(version_uid)
resolve_current_entities(version_uid_or_lineage)
is_descendant(a, b)
```

新增 revision commit 后：

```text
lineage_redirect
lineage_split
```

记录到 `revision_events.jsonl`。

---

# 7. P0-4：Controlled Corruption Harness

**不要 patch final map。**

必须在真实 mapping decision boundary 注入。

建议定义：

```python
class CorruptionPlan:
    case_uid
    frame_idx
    obs_uid
    corruption_type
    source_object_uid
    target_object_uid
```

P0 类型：

### `FORCE_CREATE`

用于 FALSE_SPLIT：

```text
clean match_index = k
corrupt match_index = None
```

### `FORCE_ASSOCIATE`

用于 wrong association / false merge：

```text
clean match = Oa
corrupt match = Ob
```

### `FORCE_POSTPROCESS_MERGE`

如果后续需要验证 merge rollback。

实现 hook：

```python
match_indices = maybe_apply_controlled_corruption(
    frame_idx,
    detection_list,
    objects,
    original_match_indices,
    provenance_context,
    corruption_plan,
)
```

只在：

```text
revision.mode == controlled_validation
```

且 manifest 明确指定 case 时生效。

必须记录：

```json
{
  "case_uid": "...",
  "event_type": "CORRUPTION_INJECTED",
  "original_decision": {},
  "corrupted_decision": {},
  "frame_idx": 0,
  "obs_uid": "...",
  "seed": 0
}
```

---

# 8. Controlled case 选择规则

自动从 clean run 找 candidate，不随便手填。

### FALSE_SPLIT candidate

选择：

- clean 明确 associate；
- similarity matrix valid；
- top1 > threshold；
- 最好 margin 不太小；
- 后续同 object 还有多个 observations。

### FALSE_MERGE candidate

选择两个现实/clean identity 不同的 active objects，强制将 observation 送入 wrong target。

初始 case 可以人工指定一个用于 smoke，但正式 benchmark 需自动化、可重复。

---

# 9. P0-5：实现 `ObservationMaterializer`

这是 replay 能否“真的重算”而不是“拼 final pickle”的关键。

给定 `obs_uid`，恢复与原 mapping 尽可能一致的 detection-like object：

```python
materialize(obs_uid) -> dict
```

需要恢复：

- observation PCD；
- bbox；
- clip feature；
- class_id/class_name；
- image/frame id；
- mask/crop refs；
- confidence；
- caption（如果 baseline 有）；
- obs UID；
- num_detections=1。

优先从已有 evidence refs + 原 saved detection artifact 恢复，不要重新跑 detector/SAM/VLM。

### Fidelity Test

对任意未污染 observation：

```text
materialized detection
vs
original detection before fusion
```

比较：

- pcd point count/hash；
- bbox；
- clip_ft allclose；
- class；
- obs_uid。

如果无法精确恢复，要在报告中标明字段差异，并补充 evidence schema 最小缺口。

---

# 10. P0-6：实现 observation-level `rebuild_object_from_members`

接口：

```python
rebuild_object_from_members(
    obs_uids,
    cfg,
    preferred_uid=None
) -> object
```

必须尽量复用原 ConceptGraphs 函数：

- PCD union；
- `process_pcd`
- `get_bounding_box`
- feature weighted fusion/normalization；
- detection attributes merge。

不要单独写一个和 `merge_obj2_into_obj1` 语义不同的融合器。

推荐实现方式：

```text
first materialized observation
     ↓
for obs in remaining:
    merge_obj2_into_obj1(...)
```

这样最接近原算法。

---

# 11. P0-7：先用 Oracle Constraint 验证 replay engine

第一轮**不要接 VLM**。

对于 controlled corruption，clean branch 已经知道正确 identity/membership。

构造：

```text
OracleRepairConstraint
```

例如：

### FALSE_MERGE

```json
{
  "type": "SEPARATE_MEMBER_GROUPS",
  "groups": {
    "A": ["obs..."],
    "B": ["obs..."]
  }
}
```

### FALSE_SPLIT

```json
{
  "type": "SAME_INSTANCE",
  "entities": ["Oa", "Ob"]
}
```

### WRONG_MEMBERSHIP

```json
{
  "type": "MOVE_OBSERVATION",
  "obs_uid": "...",
  "from": "Oa",
  "to": "Ob"
}
```

目的：

> 先证明“知道怎么修时，系统能否正确重算”。

如果 oracle constraint 都修不好，不允许把问题甩给 VLM。

---

# 12. P0-8：实现 `CausalTracer`

输入：

```text
corrupted final entity / incident
```

输出：

```json
{
  "causal_anchor_event_uid": "...",
  "anchor_frame_uid": "...",
  "affected_versions": [],
  "affected_observations": [],
  "trace": []
}
```

### FALSE_MERGE

优先找：

1. member 首次进入错误 lineage 的 association；
2. 或 postprocess `OBJECT_MERGE`。

### FALSE_SPLIT

找：

1. 被 FORCE_CREATE 的 event；
2. 两条 descendant lineage。

对于 controlled corruption，必须能找回注入 event；否则 tracer 不合格。

---

# 13. P0-9：实现 `DependencyClosure`

第一版以 correctness 优先，可适度保守。

从 anchor 向后收集：

- target lineage 的 subsequent object versions；
- subsequent associations；
- dependent merge events；
- descendant lineages；
- incident edge events。

输出：

```python
DependencyClosure(
    event_uids,
    version_uids,
    entity_uids,
    obs_uids,
    edge_uids,
    start_sequence,
    end_sequence
)
```

必须支持：

```python
hash_outside_closure_before()
hash_outside_closure_after()
```

---

# 14. P0-10：真正的 local replay

不要只用 final member set 重新 fuse 后直接结束。

需要至少模拟：

```text
state immediately before causal anchor
        ↓
apply corrected constraint
        ↓
replay subsequent relevant observations/events
        ↓
current repaired local state
```

## 最小可行实现

### FALSE_MERGE

1. 读取 anchor 前 target object version；
2. 在 local state 中处理该 observation 的正确决策；
3. 遍历 closure 中后续 association events；
4. materialize 每个 obs；
5. 使用原 ConceptGraphs similarity/merge 逻辑重新决策；
6. 对 constraint 强制固定的成员遵守 repair constraint；
7. 处理 dependent postprocess merge；
8. 输出 repaired descendants。

### FALSE_SPLIT

1. 找两个 lineage 的共同时间窗口；
2. 在 anchor 纠正 CREATE；
3. 重放两条 lineage 的后续 observations。

### WRONG_MEMBERSHIP

对两个 lineage 联合 replay。

---

# 15. 必须保留一个 `final_member_refusion` baseline

为了证明“真正 replay 有价值”，额外实现便宜 baseline：

```text
VLM/oracle 最终分组
→ 只从 final members 重新 fuse
→ 不重放历史
```

命名：

```text
FINAL_MEMBER_REFUSION_BASELINE
```

和：

```text
COUNTERFACTUAL_LOCAL_REPLAY
```

比较。

如果二者一样好，论文就不应夸大历史 replay；如果 replay 明显更好，这会成为很强的消融。

---

# 16. P0-11：边部分只使用 `ali-dev` baseline

本阶段**禁止**新 relation inference。

`ali-dev` 当前逻辑：

```text
gobs['edges']
    ↓
match_indices
    ↓
process_edges(...)
    ↓
MapEdgeMapping.add_or_update_edge(...)
```

同时：

- 弱 edge 会按 age/detection count 删除；
- object filter/merge 会更新 edge index。

## 结构 repair 后的问题

旧边的 endpoint 可能失效，因此必须做 closure。

### v0 最稳实现

创建：

```python
class AliDevBaselineRelationBackend
```

只包装现有逻辑，不改语义。

接口：

```python
invalidate(changed_entity_uids)
rebuild(...)
validate(...)
```

### 推荐 v0 rebuild 策略

为了先保证正确：

> **node replay 是 local；edge 允许先做 `GLOBAL_BASELINE_EDGE_REPLAY`。**

即：

1. repaired node mapping 得到完整 frame-level repaired match mapping；
2. 新建空 `MapEdgeMapping`；
3. 按时间顺序读取 saved per-frame `gobs['edges']`；
4. 使用 repaired match mapping 调原 `process_edges`；
5. 应用原 edge cleanup/index update 语义；
6. 得到 baseline-consistent edge graph。

理由：

- 不引入新 relation idea；
- 避免 direct overlay 中“非空 edges 无法安全 structural repair”的问题；
- 先验证 node revision 主线；
- 后续队友 RelationBackend 再替换。

报告必须明确：

```text
Node replay: dependency-local
Edge replay v0: global baseline reconstruction using unchanged ali-dev logic
```

不能冒充“全系统都局部”。

### 后续 v1

再把 edge rebuild 缩到 affected frames/incident edges。

---

# 17. P0-12：StructuralVerifier

至少实现：

```text
V1 membership ownership valid
V2 all obs_uids resolvable
V3 all active object pcd finite/non-empty
V4 bbox finite/non-degenerate
V5 current version references valid
V6 edge endpoints active
V7 no unexpected self-loop
V8 outside closure unchanged
V9 source baseline artifacts unchanged
```

输出：

```json
{
  "pass": true,
  "checks": [...]
}
```

任何硬约束失败：

```text
ABORT
```

---

# 18. P0-13：Shadow Transaction

定义：

```python
RevisionTransaction
```

至少字段：

```text
tx_id
case_uid
causal_anchor_event_uid
base_event_watermark
base_entity_versions
read_set
write_set
dependency_closure
repair_constraint
shadow_output_refs
verification
commit_status
```

v0 只：

```text
baseline → shadow → derived output
```

不需要第一天真正并发修改 live Python objects。

输出目录：

```text
experiments/revision/<run_id>/
  baseline/
  corrupted/
  repaired/
  traces/
  transactions/
  metrics/
```

---

# 19. P1-1：加入 MaturityGate

先只做：

```text
TENTATIVE
MATURE
```

保存原始信号，不生成一个手工总分。

P0 structural repair gate：

- target unique frames；
- members；
- non-degenerate geometry；
- split 时每个 group 最少独立帧数。

接口：

```python
assess_maturity(entity_or_group, action) -> {
    eligible: bool,
    reasons: [],
    raw_signals: {}
}
```

不满足：

```text
DEFER / WAIT_EVIDENCE
```

---

# 20. P1-2：把 `ali-my-VLM` 的 VLM-only 改造成 Constraint Generator

Replay engine 通过 oracle cases 后再做。

从 `ali-my-VLM` 手工移植/适配：

- typed error taxonomy；
- audit schema；
- diagnosis schema；
- VLM client；
- hash-bound evidence idea。

不要直接搬：

- `overlay.py` 作为 final structural executor。

新增 action：

```text
DEFER
```

输出约束而不是 map patch。

---

# 21. VLM Member Pass

当前 `ali-my-VLM` 对：

- SPLIT_OBJECT
- REASSIGN_MEMBERS
- TRIM_GEOMETRY

会返回 `NEEDS_FULL_MEMBER_PASS`，这是正确边界。

本阶段需要实现 full member pass：

1. target 所有 member observations；
2. 先由程序按 frame/view 做代表采样；
3. VLM 先判断 group hypothesis；
4. 如果代表图不足，增量请求更多 member views；
5. 最终输出 `obs_uid` groups，而不是只输出 `V1/V2` 代表图 alias。

必须保留：

```text
ABSTAIN / DEFER
```

---

# 22. P1-3：Online Conflict / Rebase 模拟

即使最初实验是离线 replay，也要把 online protocol 先做成可单测模块。

定义 conflict types：

```text
DISJOINT
APPEND_ONLY_REBASEABLE
LINEAGE_REBASEABLE
HYPOTHESIS_INVALIDATED
UNRESOLVED_CONFLICT
```

## 测试 A：不相关对象更新

repair target O1，delta 只动 O9：

```text
→ DISJOINT
→ 不重启
```

## 测试 B：target 新增 observations

```text
O1@v20 → O1@v23
```

三个新增 obs：

```text
→ APPEND_ONLY_REBASEABLE
→ tail replay
→ 不重新 VLM
```

## 测试 C：target merge 到 descendant

```text
O1 → O7
```

能通过 lineage map：

```text
→ LINEAGE_REBASEABLE
```

## 测试 D：核心 evidence 被替换/删除

旧 repair hypothesis 不再成立：

```text
→ HYPOTHESIS_INVALIDATED
```

---

# 23. P1-4：RepairTicket 状态机

实现：

```text
OPEN
WAIT_EVIDENCE
READY
DIAGNOSING
TRACING
REPLAYING
REBASING
READY_TO_COMMIT
COMMITTED
SUPERSEDED
ABORTED
WAIT_STABILITY
```

### 防无限重启

```text
max_rebase_count
debounce_same_lineage
cooldown
```

达到 max：

```text
WAIT_STABILITY
```

不是 while true 重跑 VLM。

---

# 24. Controlled Evaluation：必须同时跑三条 branch

每个 case：

```text
clean branch
corrupted branch
repaired branch
```

所有分支：

- 同 config；
- 同 random seed；
- 同 saved detections；
- 唯一差异是明确的 corruption/repair。

---

# 25. 评价指标实现

输出一个统一：

```text
revision_metrics.json
revision_report.md
```

至少包括：

## A. Membership

```text
member_precision
member_recall
member_f1
```

相对 clean mapper reference。

## B. Node identity

```text
over_merge_count
over_split_count
duplicate_count
```

## C. Geometry

至少：

```text
bbox_iou_to_clean
center_error_to_clean
extent_error_to_clean
point_support
```

如果实现 point cloud Chamfer 成本高，先不阻塞 P0。

## D. Relation

本阶段：

```text
edge_set_precision_to_clean
edge_set_recall_to_clean
edge_relation_match
dangling_edge_count
```

这测的是 baseline relation recovery，不是队友新算法准确率。

## E. Safety

```text
outside_closure_changed_entities
collateral_damage_count
hard_invariant_failures
```

## F. Cost

```text
local_replay_ms
global_replay_ms
num_replayed_events
total_events
replay_fraction
```

---

# 26. Global Replay Reference

必须实现一个 reference：

> 从 corruption 前的 checkpoint 重新跑余下所有 frames，但使用正确 constraint。

它可能慢，但只用于实验。

比较：

```text
local replay vs global replay
```

关键输出：

```text
state_similarity
runtime_ratio
event_fraction
```

---

# 27. 核心实验矩阵

第一版只跑：

| Method | Purpose |
|---|---|
| Clean | counterfactual mapper reference |
| Corrupted | injected failure |
| Final Member Refusion | cheap structural patch baseline |
| Global Replay | expensive reference |
| Local Replay | proposed kernel |
| Oracle Local Replay | upper bound constraint |

VLM 接入后再加：

| Method | Purpose |
|---|---|
| VLM-only overlay | current baseline |
| VLM constraint + local replay | intended method |

---

# 28. 第一批 smoke cases

不追求很多。

先跑通：

1. 1 个 false split；
2. 1 个 wrong membership；
3. 1 个 false merge。

每个 case 必须生成：

```text
trace.json
dependency.json
corruption.json
constraint.json
transaction.json
before_after_summary.json
edge_rebuild_summary.json
metrics.json
```

最好自动输出 overview 图，但图不是 P0 blocker。

---

# 29. Tests：必须先写的单测

## `test_revision_parity.py`

revision off 不改变 baseline。

## `test_revision_provenance.py`

obs/event/version 索引正确。

## `test_revision_lineage.py`

merge parent/descendant 可解析。

## `test_revision_corruption.py`

指定 obs 的决策只被修改一次且可复现。

## `test_revision_replay.py`

同一 member list rebuild deterministic；oracle case 能恢复。

## `test_revision_baseline_relations.py`

rebuild 后：

- edge endpoint 全存在；
- relation tuple 格式合法；
- 未引入新 relation 类型。

## `test_revision_transaction.py`

source artifacts 不变；shadow output 独立。

## `test_revision_conflicts.py`

四类 conflict 分类正确。

---

# 30. 建议 CLI

### 生成 clean + corruption case

```bash
python scripts/run_revision_controlled_experiment.py \
  --scene room0 \
  --base-run <ALI_MY_FORMAL_RUN> \
  --case-config <CASE_JSON> \
  --output-root experiments/revision_v0
```

### oracle replay

```bash
python scripts/run_revision_controlled_experiment.py \
  --case <CASE_UID> \
  --repair-source oracle \
  --replay local
```

### global reference

```bash
python scripts/run_revision_controlled_experiment.py \
  --case <CASE_UID> \
  --repair-source oracle \
  --replay global
```

### evaluate

```bash
python scripts/evaluate_revision_controlled.py \
  --run-root experiments/revision_v0/<CASE_UID>
```

### tests

```bash
pytest -q tests/test_revision_*.py
```

具体参数可以根据仓库现有 Hydra 风格调整，但功能边界不要改变。

---

# 31. Codex 开发顺序：严格执行

## Step 1

revision-disabled parity。

## Step 2

ProvenanceIndex + LineageIndex。

## Step 3

Controlled corruption hook。

## Step 4

ObservationMaterializer。

## Step 5

Object rebuild from members。

## Step 6

Oracle FALSE_MERGE local replay。

## Step 7

DependencyClosure + outside-closure hash。

## Step 8

Global replay reference。

## Step 9

`AliDevBaselineRelationBackend` + full baseline edge replay。

## Step 10

FALSE_SPLIT / WRONG_MEMBERSHIP。

## Step 11

Maturity / DEFER。

## Step 12

RepairTicket + conflict/rebase simulation。

## Step 13

VLM constraint generator。

**在 Step 6 成功前，不要实现复杂 async worker。**

---

# 32. 每个 Step 的完成标准

## Step 1 PASS

```text
revision off parity = PASS
```

## Step 2 PASS

任意 final object 可：

```text
final object
→ versions
→ member obs
→ association events
→ parents
```

## Step 3 PASS

同 seed/case 可重复制造同一个错误。

## Step 4 PASS

materialized obs 与原 observation payload 对齐。

## Step 6 PASS

至少一个 false merge：

```text
corrupted member F1 < repaired member F1
repaired geometry closer to clean
```

## Step 7 PASS

```text
outside_closure_changed = 0
```

## Step 9 PASS

结构 repair 后：

```text
dangling edges = 0
```

且 edge relation type 全来自 ali-dev baseline。

## Step 12 PASS

target 版本变化不会默认 restart。

---

# 33. 失败时怎么处理

### Materializer 无法恢复原 detection

不要偷偷近似。

输出字段差异，回到 evidence recorder 补“最小 replay payload”。

### Local replay 与 global replay 差异很大

优先查：

- 漏 dependency；
- postprocess merge；
- feature fusion；
- DBSCAN/downsample；
- edge indexing；
- random seed。

不要先调 VLM。

### FALSE_MERGE 修不好

暂时停 false split / online / generality，把单 case 做透。

### edge rebuild 太复杂

保持 `GLOBAL_BASELINE_EDGE_REPLAY`，不要为了局部 edge 优化阻塞 node replay；队友后续再替换。

---

# 34. 不能做的事情

Codex 不要：

- 新造 relation taxonomy；
- 改 `ali-dev process_edges` 的语义；
- 把队友尚未完成的 voxel relation 预先写进本分支；
- 把 R1 frozen labels 作为 repair oracle；
- 用 full-97 数据调 prompt 后再在同一 97 宣称泛化；
- 直接在 final pickle 上删/合并后称为 replay；
- 因 graph global version 变化就整次 repair restart；
- 用一个 handcrafted maturity scalar 直接自动 commit；
- 一开始就适配 VLMaps/FROSS/PUF。

---

# 35. 本阶段最终要交出的证明

本阶段结束时应能给出一个清晰的最小实验：

```text
Clean ConceptGraphs run
        ↓
Inject one false merge at a real association event
        ↓
Continue mapping
        ↓
Observe propagated final error
        ↓
Trace from final node back to corrupted event
        ↓
Apply known/oracle separation constraint
        ↓
Fork shadow local state
        ↓
Replay affected observations/events
        ↓
Rebuild ali-dev baseline edges
        ↓
Verify
        ↓
Derived repaired graph
```

并得到：

```text
member recovery ↑
geometry distance to clean ↓
relation state closer to clean
outside-closure changes = 0
local replay cost < global replay
```

这个闭环成功后，才进行：

```text
oracle constraint
→ VLM constraint
```

替换。

---

# 36. 本阶段结束后的下一步

当且仅当下面四项同时完成：

```text
FALSE_MERGE replay PASS
FALSE_SPLIT / WRONG_MEMBERSHIP 至少一类 PASS
baseline edge closure PASS
controlled recovery metrics PASS
```

再进入正式论文系统：

1. 接入 `ali-my-VLM` 的 VLM constraint generator；
2. 做 maturity-aware repair；
3. 做 snapshot/delta rebase；
4. 接队友 RelationBackend；
5. 多场景 benchmark；
6. natural error 评测。

---

# 37. 给 Codex 的最终一句任务定义

> **不要继续做一个“更聪明的 final-map patcher”。在 `ali-my` 已有 append-only evidence/version ledger 上，实现一个可验证的 revision kernel：能够从 final error 追踪到历史 mapping decision，在 shadow state 中改变该 decision 的约束，用原 ConceptGraphs 计算逻辑局部 replay 受影响 observations/events，并用未修改的 `ali-dev` edge logic 恢复一个一致的 graph；所有源地图保持只读，所有结果通过 controlled clean/corrupt/repaired 三分支进行量化验证。**
