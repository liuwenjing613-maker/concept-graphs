# ICRA 2027 最终方案：基于 ConceptGraphs 的在线可修订 3D Scene Graph

> **方案状态：主线冻结，算法细节继续迭代**
>
> **目标会议：ICRA 2027**
>
> **主实现底座：ConceptGraphs `ali-dev` → `ali-my` → `ali-my-VLM`**
>
> **当前代码基准（2026-08-22）**
>
> - `ali-dev`：当前在线/增量 ConceptGraphs 基线。
> - `ali-my @ 66c109dc042fdf21936e933aa80f4e8307ddd4d6`：在 `ali-dev` 上增加统一证据留存、对象版本、关联/融合/边事件、分层查错器与 final-endpoint 审计。
> - `ali-my-VLM @ 4c6e376462e71f7065222134aab925c8412969e7`：在 `ali-my` 上增加冻结证据包、VLM-only 审计/诊断/验证、derived-map 修复与 repair-aware 评测。
>
> **本文档用途**：作为两人共同投入 ICRA 的完整研究与工程蓝图。论文不是“ConceptGraphs + VLM prompt + voxel edge”的模块拼接，而是围绕一个问题构建一个统一系统：**在线 3D Scene Graph 如何在后来获得的新证据表明过去的增量建图决策错误时，安全地回溯、局部重算并恢复节点及其依赖关系，而不重建全图、不阻塞在线建图，也不因修复器自身的不确定性破坏正确地图。**

---

# 0. 一句话定位

推荐论文工作名：

**Revisable 3D Scene Graphs: Evidence-Grounded Retrospective Revision for Online Open-Vocabulary Mapping**

备选：

**Trace, Replay, Repair: Safe Retrospective Revision for Online 3D Scene Graphs**

核心问题：

> Incremental 3D mapping makes irreversible local decisions under partial observations. When later evidence reveals that a historical association/fusion decision was wrong, how can the system revise only the affected node–relation state online, without rebuilding the entire map or corrupting the live graph?

中文：

> **增量 3D 建图在信息不充分时不断做关联、融合和图更新；一旦早期决策出错，错误会被后续状态继承。本文研究如何利用后来累积的视觉、几何、关联和关系证据，定位历史错误，在局部 shadow branch 中改变历史约束并重新执行受影响的建图过程，经过结构/关系/语义验证后再安全提交。**

这不是以下任何一个问题：

- 不是单纯“把 ConceptGraphs 的 mIoU 做高”；
- 不是“VLM 判断哪里错然后直接改 pickle”；
- 不是“异步 VLM 合并 duplicate”；
- 不是“给 graph 加版本控制”；
- 不是“做一个更好的 voxel relation predictor”；
- 不是一般意义上的 dynamic scene update。

真正的核心是：

**history-aware + evidence-grounded + counterfactual local replay + safe online commit。**

---

# 1. 为什么必须仍以 ConceptGraphs 为主要切入点

ConceptGraphs 很适合作为第一实例化，原因不是它最新，而是它的问题结构和当前代码积累正好匹配本研究。

ConceptGraphs 的增量对象建图包含：

1. 2D 检测/分割；
2. RGB-D 投影得到 observation point cloud；
3. spatial similarity + visual similarity；
4. greedy association；
5. 将 observation 融合到对象；
6. 周期性 denoise/filter/merge；
7. 建立并维护关系边。

这意味着一个早期 association 错误不是孤立错误：

```text
wrong observation association
        ↓
wrong member set
        ↓
wrong fused geometry / CLIP feature
        ↓
later association decisions change
        ↓
postprocess merge may change
        ↓
incident relations change
        ↓
downstream query/segmentation/grounding changes
```

ConceptGraphs 原论文也明确承认 node caption 错误、小/薄物体漏检、duplicate 等失败模式；原始图边由空间候选和 LLM 关系推理产生，原论文报告的 edge precision 不能等价为关系覆盖充分。

更重要的是，当前 `ali-dev` 已经是逐帧在线状态演化程序，而 `ali-my` 已经把上述过程做成了可追溯 event/version ledger。因此继续以 ConceptGraphs 为主，不是在旧方法上做小修补，而是在一个已经掌握且具有真实错误传播链的系统上验证新的 **revision paradigm**。

---

# 2. 当前已有实现：哪些保留，哪些降级为 baseline

## 2.1 `ali-my`：保留为 Provenance Substrate

`ali-my` 当前最大的价值不是 checker，而是统一证据系统。

已有关键证据：

- `manifest.json`
- `frames.jsonl`
- `observations.jsonl`
- `similarities/frame_*.npz`
- `associations.jsonl`
- `mapping_events.jsonl`
- `filter_trace.jsonl`
- `object_versions.jsonl`
- `object_pair_decisions.jsonl`
- `vlm_events.jsonl`
- `final_membership.json`
- observation point clouds / processed masks / feature refs

尤其已有：

- association top-k；
- top1/top2/margin；
- spatial/visual/aggregate similarities；
- target object version before/after；
- parent event；
- transaction UID；
- object member observations；
- object version parents；
- merge before/after member sets；
- edge add/delete/update 及输入 object versions。

因此最终系统**不要重写 evidence recorder**。只应在保证 mapping parity 的前提下补足 revision 所需的最小字段和索引。

### 必须升级的两个点

1. 当前 `lineage_uid` 偏向 origin-based，需要在 revision 层额外建立**真正的 lineage DAG**，能够表达 merge/split/redirect，不必破坏已有 ledger。
2. 增加 transaction/revision 自身的独立日志，例如：
   - `revision_tickets.jsonl`
   - `revision_events.jsonl`
   - `revision_transactions.jsonl`

不要把 revision 运行事件混入原 baseline evidence 语义。

---

## 2.2 分层 checker：从“判错器”降级为 Trigger / Hypothesis Generator

当前两场景 formal validation：

- 5587 条 raw checker findings 最终聚合成 distinct final endpoints；
- 97 个可复核 endpoint 中 55 CORRECT、40 WRONG、2 UNCLEAR；
- `review_score` 对人工错误状态 ROC-AUC 仅约 0.420。

这说明：

> checker 有发现“值得看区域”的能力，但 checker score 不是可靠错误概率。

最终架构：

```text
checker finding
   ↓
suspicion
   ↓
incident aggregation
   ↓
evidence readiness
   ↓
VLM / structured diagnosis
```

禁止：

```text
checker finding → 自动修复
```

Checker 的论文角色是：

**cheap trigger that reduces expensive reasoning scope。**

---

## 2.3 `ali-my-VLM`：保留为 VLM-only baseline + adjudicator prototype

当前 VLM-only 已有一个严谨优点：

- frozen/hash-bound evidence；
- inference 看不到人工 R1；
- 强制反证 audit；
- typed diagnosis；
- independent VLM verification；
- 只改 derived map。

当前结果同时暴露了它的上限：

- saved-label downstream 明显改善；
- 但 native CLIP 表征不变；
- geometry coverage 不变；
- fragmentation 不变；
- structural actions 仍未真正实现；
- 非空 graph edges 下 structural overlay 有安全阻断。

所以它非常适合写成：

**Baseline: post-hoc VLM endpoint patching**

并用于动机：

> Semantic patching can improve labels, but does not recover the historical mapping state.

最终系统保留它的三个思想：

1. counter-evidence audit；
2. typed repair hypothesis；
3. explicit abstention / insufficient evidence。

但 VLM 不再拥有 mutation authority。

---

# 3. 与最新工作的边界：论文必须避开的 headline

截至 2026-08 的直接相关工作至少包括：

- **Think While You Map / ThinkGraphs (2026)**：轻量 online mapper + 异步后台 VLM agents；duplicate track semantic loop closure；spatial relations；informative frame scheduler。
- **PUF (ECCV 2026)**：online 3DSG uncertainty-aware probabilistic association；node/relation Dirichlet evidence accumulation；Gaussian/voxel backends。
- **FROSS (ICCV 2025)**：faster-than-real-time online 3D SSG；ReplicaSSG。
- **CoPa-SG (ICCVW 2025)**：更完整、精确的 relation annotations；parametric/proto-relations。
- **KNA-SG (2026)**：keyframe-node association；可检索视觉证据；query-time relation verification。
- **Graph Rectification (ACL 2026)**：graph version control、edit history、rollback、conflict tracing、repair。
- **DovSG / Interaction-Driven Updates**：动态/长期 scene graph maintenance。

因此必须避免：

- “异步 VLM 修 duplicate”作为主贡献；
- “version control + rollback”作为主贡献；
- “dynamic scene update”作为主贡献；
- “voxel relationship”本身作为主贡献；
- “保存 keyframe 给 VLM 验证”作为主贡献。

我们的差异中心必须始终是：

> **在 RGB-D object-centric metric-semantic mapping 中，对历史 perception decision 做 observation-level counterfactual local replay，并在 live map 继续演化的情况下对 repaired branch 做 dependency-aware rebase 与安全提交。**

---

# 4. 总体系统架构

```text
                         FAST ONLINE PATH
RGB-D_t
   │
   ▼
ConceptGraphs Mapper
(det → obs3D → assoc → fusion → baseline/strong relations)
   │
   ├───────────────────────────────→ Live Graph Head G_t
   │
   ▼
Append-only Provenance / Version Store
   │
   ├──────────→ Per-entity Maturity Manager
   │
   └──────────→ Trigger / Incident Manager
                       │
                  WAIT_EVIDENCE?
                   │       │
                  yes      no
                   │       ▼
                   │   Evidence Retriever
                   │       ▼
                   │   VLM Adjudicator
                   │   (constraint only)
                   │       ▼
                   └── RepairTicket
                           ▼
                      Causal Trace
                           ▼
                   Dependency Closure
                           ▼
                   Snapshot / Shadow Fork
                           ▼
                Counterfactual Local Replay
                    │                 │
                    │                 ▼
                    │          Relation Backend
                    │       invalidate + rebuild
                    │                 │
                    └──────────┬──────┘
                               ▼
                          Verification
                   structural / relation / semantic
                               ▼
                         Delta Rebase
                               ▼
                      Local Commit Barrier
                         ↙             ↘
                     COMMIT          ABORT
```

系统本质是**双速系统**：

- fast path：不能被 VLM 阻塞；
- slow path：只处理少量风险对象/事件。

这与在线机器人系统的实际约束一致。

---

# 5. 模块 A：ConceptGraphsAdapter —— 不重写 mapper，建立稳定接口

## 目标

把当前方法从“散落在 ConceptGraphs 代码中的 repair 逻辑”抽象为可替换 mapper 的接口，同时 ICRA 前只实现 ConceptGraphs。

推荐接口：

```python
class MapAdapter:
    def snapshot(self, entity_uids, event_watermark): ...
    def materialize_observation(self, obs_uid): ...
    def current_version(self, entity_uid): ...
    def resolve_lineage(self, version_uid): ...
    def trace_dependencies(self, seed_events): ...
    def fork_local_state(self, closure): ...
    def apply_constraint(self, local_state, constraint): ...
    def replay_events(self, local_state, events): ...
    def validate_state(self, local_state): ...
    def commit(self, transaction): ...
```

ConceptGraphsAdapter 内部直接调用现有：

- `compute_spatial_similarities`
- `compute_visual_similarities`
- `aggregate_similarities`
- `match_detections_to_objects`
- `merge_obj_matches`
- `merge_obj2_into_obj1`
- denoise/filter/merge
- relation backend

而不是复制一套“近似 ConceptGraphs”。

## 论文 claim

如果最终只有 ConceptGraphs：

> “We design a mapper-adapter interface and instantiate it on ConceptGraphs.”

不能写：

> “Our method is empirically general across mapping architectures.”

---

# 6. 模块 B：ProvenanceIndex —— 把已有 ledger 变成可查询因果索引

不要继续只把 JSONL 当日志文件。加载后建立索引：

```text
obs_uid → observation record
obs_uid → association event
event_uid → mapping event
object_version_uid → version record
object_uid → ordered versions
version_uid → parent versions
version_uid → child versions
object_uid → current active version
version_uid → member obs
object_uid → incident edge events
merge source/target → merge transaction
frame_uid → observations / edge evidence
```

增加：

```text
resolve_descendants(version_uid)
resolve_current_entities(lineage)
trace_obs_to_current(obs_uid)
events_after(event_sequence)
```

## 真正 lineage

不要只依赖当前 `origin_*` 字段。

在 revision runtime 里根据：

- `parent_version_uids`
- `OBJECT_MERGE`
- future `SPLIT`
- redirect/commit records

构建 lineage DAG。

目标是：

> 修复票据绑定“物理实体谱系/历史约束”，而不是绑定某个容易过期的 object index。

---

# 7. 模块 C：MaturityManager —— 解决在线早期视野不足

全局“前 N 帧 warm-up”是错误抽象，因为机器人每进入一个新区域，新对象都重新进入低证据阶段。

应该维护 **per-entity maturity**。

MVP 状态：

```text
TENTATIVE → MATURE
```

最终可扩展：

```text
TENTATIVE → PROVISIONAL → CONFIRMED → STABLE
```

## 不做一个手工总分

不要再做类似 `maturity=0.2*a+...` 的复合分数作为唯一真值。当前 `review_score` 的失败已经说明手工标量可能没有排序能力。

保存原始信号：

- `unique_frame_count`
- camera baseline / viewpoint diversity
- point count growth
- bbox center/extent stability
- class/semantic stability
- association top1-top2 margin history
- candidate switching
- member consistency
- relation support（有 relation backend 后）

然后对不同 action 使用不同 gate。

### RELABEL

允许较早，但要求多视角语义一致。

### MERGE / FALSE_SPLIT 修复

需要两个对象均有足够独立证据，避免因局部视角过早合并。

### SPLIT / FALSE_MERGE 修复

至少要求目标成员能形成两个有独立多视角支持的 group。

### DELETE

最保守。低视野阶段原则上 DEFER。

## 系统偏置

> **证据不足时宁可 temporary fragmentation，不要过早 contamination。**

原因是 false split 通常比 false merge 容易后续恢复。

---

# 8. 模块 D：Trigger / Incident Manager

输入来源：

- detection checker；
- segmentation checker；
- projection/geometry checker；
- association checker；
- fusion checker；
- object identity checker；
- relation inconsistency（队友模块加入后）。

输出不是“错误”，而是：

```json
{
  "incident_uid": "...",
  "target_lineage": "...",
  "trigger_events": [],
  "hypotheses": [],
  "evidence_readiness": "...",
  "priority": "...",
  "status": "WAIT_EVIDENCE|READY"
}
```

## 聚合原则

沿用你已经验证过的 endpoint/lineage 聚合思想：

- 同一 active object / same lineage 不创建大量重复 heavy jobs；
- debounce；
- 新 finding 追加到已有 ticket；
- cooldown；
- 同一谱系不允许无限并发 VLM 修复。

---

# 9. 模块 E：EvidenceRetriever —— 从 final endpoint evidence 升级成 historical repair evidence

保留 `ali-my-VLM` 的好原则：

- hash verification；
- 不向 VLM 泄漏人工标签；
- checker 名称/score 不作为视觉答案；
- exact target state；
- context；
- representative views；
- 证据不足允许 abstain。

但 revision evidence 需要增加**时间结构**。

一个 repair packet 至少包括：

### 当前状态

- current object geometry；
- class/semantic summary；
- current members；
- neighboring entities/relations。

### 历史轨迹

- object version timeline；
- member admission timeline；
- suspicious association candidates；
- relevant merge event；
- before/after state；
- causal-anchor 前后的差异。

### observation 级证据

- raw/processed mask；
- context crop；
- observation PCD；
- view pose；
- original visual feature refs；
- original candidate scores。

### 关系证据

- affected incident edges；
- edge support/confidence；
- edge history。

### evidence gaps

必须显式告诉 VLM：

- 缺多少视角；
- 是否只有相近 viewpoint；
- 是否缺某个 member 的 RGB；
- relation 是否 tentative。

---

# 10. 模块 F：VLM Adjudicator —— VLM 只输出“修复约束”

当前 `ali-my-VLM` 的 error taxonomy 可继续使用：

- FALSE_MERGE
- FALSE_SPLIT
- WRONG_MEMBERSHIP
- GEOMETRY_CORRUPTION
- SEMANTIC_IDENTITY_ERROR
- SPURIOUS_OBJECT
- OTHER

新增系统状态：

- UNDEROBSERVED / DEFER

最终 VLM 输出不再是“直接修改 final map”，而是约束：

```text
LABEL(entity, "chair")

SAME_INSTANCE(O1, O2)

SEPARATE_MEMBER_GROUPS(
  A=[obs_1, obs_4, obs_7],
  B=[obs_2, obs_5, obs_9]
)

MOVE_OBSERVATION(obs_17, from=O1, to=O2)

INVALID_OBJECT(Ox)

DEFER(reason=INSUFFICIENT_VIEW_DIVERSITY)
```

## 保留 counter-evidence 机制

建议：

```text
Hypothesis generation
        ↓
Forced counter-evidence audit
        ↓
Typed repair constraint
```

第二 VLM verifier 只在高风险/高不确定场景使用；最终安全性不能依赖“多个同源 VLM 都同意”。

---

# 11. 模块 G：RepairTicket —— 修复不是一次 API 调用，而是持续存在的 revision intent

建议 schema：

```json
{
  "ticket_id": "...",
  "target_lineage": "...",
  "causal_anchor_event_uid": "...",
  "base_event_watermark": 1234,
  "base_entity_versions": [],
  "evidence_refs": [],
  "diagnosis": {},
  "repair_constraint": {},
  "read_set": [],
  "write_set": [],
  "dependency_closure": [],
  "maturity_at_diagnosis": {},
  "latest_seen_event": 1300,
  "rebase_count": 0,
  "vlm_requery_count": 0,
  "status": "OPEN"
}
```

状态机：

```text
OPEN
 ↓
WAIT_EVIDENCE
 ↓
READY
 ↓
DIAGNOSING
 ↓
TRACING
 ↓
REPLAYING
 ↓
REBASING
 ↓
READY_TO_COMMIT
 ↓
COMMITTED
```

旁支：

- SUPERSEDED
- ABORTED
- STALE_CONFLICT
- WAIT_STABILITY

---

# 12. 模块 H：CausalTracer —— 从错误终态追到历史决策

这是论文主方法之一。

## FALSE_MERGE / WRONG_MEMBERSHIP

目标不是“发现当前 object 看起来怪”，而是找：

> 哪个 observation 或 merge transaction 首次让原本不同 physical groups 进入同一 lineage？

算法：

1. 从 final member set 开始；
2. 沿 object version parents 回溯；
3. 找 member-set 首次发生异常联合的 event；
4. 查看该 event 的 association top-k、margin、spatial/visual evidence；
5. 将其标为 candidate causal anchor；
6. 必要时回溯到 postprocess merge。

## FALSE_SPLIT

1. 两个当前 nodes 形成同一实体假设；
2. 分别追其 origin observations；
3. 找历史上第一次“本应 associate 但 CREATE”的时刻；
4. 对比候选得分与后续共视/几何证据。

## 输出

```text
anchor_event_uid
anchor_frame_uid
pre_anchor_versions
affected_lineages
candidate_cause_type
trace_confidence / trace_evidence
```

注意：

> CausalTracer 负责定位“可操作的历史节点”，VLM 不需要直接浏览全量 200 帧日志。

---

# 13. 模块 I：DependencyClosure —— 决定局部 replay 到底重算多少

这是 local replay 能成立的关键。

Seed：

- causal anchor event；
- affected object versions；
- affected member observations。

向后扩张依赖：

1. 该 object 后续 `OBS_ASSOCIATE`；
2. 依赖其 feature/geometry 的后续 candidate decisions；
3. 与其相关的 postprocess merge；
4. descendants；
5. incident edge events；
6. 若结构变化导致 edge endpoint 变化，则加入相关 edge closure。

第一版宁可稍微保守，不要漏依赖。

目标：

```text
D(r) << entire mapping history
```

同时要求：

```text
Outside(D(r)) state hash unchanged
```

这是非常重要的安全指标。

---

# 14. 模块 J：Snapshot + Shadow Fork

修复绝不能原地操作 live map。

在 `event_watermark = s` 建立：

```text
Snapshot S_s
```

只复制 dependency closure 所需状态：

- affected objects；
-必要邻居只读快照；
- local edges；
- needed observations；
- mapping config。

之后 live mapper 继续运行。

所有新事件形成：

```text
Delta = events(s+1 ... now)
```

不需要单独复制一份完整日志：已有 append-only `mapping_events` 可以作为基础，revision 层记录 watermark 和索引即可。

---

# 15. 模块 K：Counterfactual Local Replay —— 整篇论文最核心算法

普通 patch：

```text
G'_T = Patch(G_T)
```

普通 rollback：

```text
G'_T = G_{τ-1}
```

我们的目标：

```text
G'_T = Replay_D(
    G_{τ-1},
    historical observations/events
    | corrected constraint
)
```

关键是：

> 改变历史 decision/constraint，再重新执行受影响计算，得到历史中从未存在过的新状态。

## P0 三类结构修复

### A. FALSE_MERGE

输入：

```text
O = {obs1, obs2, obs3, obs4, ...}
```

constraint：

```text
A={obs1,obs3,...}
B={obs2,obs4,...}
```

执行：

1. 从原 observation sidecars materialize 原始 detection；
2. 对 A、B 分别用与 ConceptGraphs 一致的 fusion/process PCD 逻辑重建；
3. 重算 CLIP aggregate feature；
4. 重算 bbox；
5. 重放 causal anchor 之后对该 lineage 的新 observations；
6. 得到两个新的 object versions；
7. 生成 redirect/lineage records。

### B. FALSE_SPLIT

constraint：

```text
SAME_INSTANCE(Oa, Ob)
```

执行：

1. 找共同 replay anchor；
2. 将两条 lineage 在 shadow branch 中合并；
3. 用原 ConceptGraphs fusion 重新构造；
4. 重放后续 relevant observations。

### C. WRONG_MEMBERSHIP

constraint：

```text
MOVE(obs_k, Oa → Ob)
```

执行：

1. 从 Oa 重新 fusion 去掉 obs_k；
2. 将 obs_k 加入 Ob；
3. 对两个 affected lineages 分别重放后续事件。

### RELABEL

保留为低成本 metadata repair，但不作为结构恢复 headline。

---

# 16. 模块 L：Relation Layer —— 队友主攻

## 16.1 论文中的角色

不是“第二个独立问题”，而是：

> **Coupled node–relation state restoration。**

节点被 replay 后，原 incident relations 不能继续默认有效：

```text
Node mutation
   ↓
invalidate affected edges
   ↓
local relation reconstruction
   ↓
relation consistency validation
```

反过来，关系也可作为 node repair trigger / verifier：

```text
relation contradictions
   ↓
node/identity suspicion
```

---

## 16.2 队友的第一阶段目标：先真正证明 ConceptGraphs 边的瓶颈

第一件事不是写新算法，而是建立 evaluator：

- Candidate Relation Recall
- Edge existence Precision / Recall / F1
- Predicate R@1 / mR@1
- Relationship/Triplet R@1 / mR@1
- edges per node
- latency/frame

优先 ReplicaSSG；有余力再 3DSSG。

必须把“关系标签准确”与“候选覆盖完整”分开。

---

## 16.3 推荐 RelationBackend 接口

```python
class RelationBackend:
    def build_candidates(self, nodes, local_region): ...
    def infer_relations(self, candidate_pairs): ...
    def invalidate(self, changed_entity_uids): ...
    def rebuild(self, changed_entity_uids, local_neighbors): ...
    def score_consistency(self, local_subgraph): ...
```

---

## 16.4 推荐 P0 physical relations

只先做几何可验证的：

- support/on
- inside/contain
- contact
- near
- above/below

可以保存连续 evidence：

- surface distance
- vertical gap
- XY support overlap
- containment ratio
- contact area

但“voxel-based relation”本身不能作为论文 headline；近期已有 CoPa-SG、PUF 等重合方向。

真正需要的是：

> **high-coverage、local-recomputable、confidence-aware physical relation layer，用于 revision closure。**

---

## 16.5 Relation maturity

edge 也必须区分：

```text
TENTATIVE_EDGE
CONFIRMED_EDGE
```

relation confidence 至少应考虑：

- endpoint node maturity；
- independent observation support count；
- geometric evidence stability。

tentative relation 只能作 soft evidence，不能 veto 一个高质量 node repair。

---

# 17. 模块 M：Verification —— 三种独立信号，而不是三个 VLM

最终 commit 前至少检查：

## 17.1 Structural verifier

硬约束：

- observation membership 合法；
- 不应出现重复 ownership；
- entity point cloud 非空且 finite；
- bbox 合法；
- version/lineage 引用完整；
- edge endpoint 均存在；
- 无意外 self-loop；
- outside dependency closure 哈希不变。

## 17.2 Relation verifier

队友 backend：

- repair 后物理关系是否更自洽；
- 是否产生明显不可能的 support/contain/contact；
- affected relations 是否成功 rebuild；
- relation confidence 是否降低到不可接受程度。

注意 relation backend 不是“真值”。

## 17.3 Semantic/VLM verifier

只做语义/视觉判断：

- member grouping 是否仍能由图像支持；
- repaired identity 是否视觉合理。

最终应避免：

```text
VLM-A → VLM-B → VLM-C 全都说对，所以安全
```

---

# 18. 在线关键设计：不允许“版本不一致就重来”

## 18.1 检查 dependency conflict，不检查 global graph version

RepairTicket 记录：

```text
read_set
write_set
dependency_closure
base versions
event watermark
```

VLM 推理期间 live map 继续更新。

commit 时对 delta 分类：

### DISJOINT

图别处更新：

> 直接忽略，repair 不 stale。

### APPEND_ONLY_REBASEABLE

目标 object 只是新增 observations：

> 把新 observations 在 repaired branch 上 tail replay，不重新调用 VLM。

### LINEAGE_REBASEABLE

目标发生可追踪 merge/redirect：

> 用 lineage DAG remap 到 current descendants，然后 replay relevant delta。

### HYPOTHESIS_INVALIDATED

新证据直接推翻原诊断：

> supersede/cancel，必要时重新诊断。

只有最后一类或无法自动映射的结构冲突才允许 re-query/restart。

---

## 18.2 防 starvation

- debounce：同 lineage 多个 finding 合并到一个 ticket；
- cooldown：高速变化对象进入 WAIT_STABILITY；
- max_rebase：超过次数后不无限循环；
- priority：高影响/高错误概率/低成本优先；
- **local commit barrier**：最终提交只对 dependency closure 短暂加 lease/lock，不冻结整个 mapper。

最终 commit：

```text
acquire local lease
        ↓
read last delta
        ↓
tail replay
        ↓
verify
        ↓
atomic swap/redirect
        ↓
release
```

---

# 19. 通用性：现在设计，后面证明

## 19.1 当前就要通用化的 schema

避免：

```text
conceptgraph_object_id
```

优先：

```text
entity_uid
entity_type
observation_uid
event_uid
version_uid
parent_version_uids
geometry_ref
semantic_ref
dependency_refs
```

Repair operator 使用抽象动作：

```text
MERGE_ENTITY
SPLIT_MEMBERSHIP
REASSIGN_OBSERVATION
REFUSE_GEOMETRY
UPDATE_SEMANTIC
INVALIDATE_RELATIONS
RECOMPUTE_RELATIONS
```

由 `ConceptGraphsAdapter` 实现。

---

## 19.2 deadline 前的 claim 边界

### 只有 ConceptGraphs

可以写：

> extensible / mapper-adapter-based architecture instantiated on ConceptGraphs.

不能写：

> proven general framework across map representations.

### 第二 mapper 有小规模实验

才能写：

> cross-backend applicability.

---

## 19.3 第二方法优先级

核心闭环成功后，若时间充足：

1. 优先接 FROSS/PUF 一类 online node-edge representation，因为 abstraction 最接近；
2. VLMaps 属于 dense spatial feature map，适合后续证明更强 representation generality，但不是 ICRA deadline 前第一优先。VLMaps 把多视角特征融合到空间网格，其特征平均本身会产生噪声，是未来“非 object entity revision”的典型 stress case。

---

# 20. 评价体系：必须把“检测、修复、安全、在线、下游”分开

---

## 20.1 Controlled Causal Corruption —— 主科学实验

这是最重要的一组。

先跑一个 uncorrupted ConceptGraphs trajectory：

```text
G_clean
```

在真实 mapping decision boundary 注入一个可控错误，然后继续跑：

```text
G_corrupt
```

系统修复：

```text
G_repair
```

### 注入类型

#### FALSE_SPLIT

clean 本应 associate：

```text
obs → O
```

强制：

```text
obs → CREATE
```

#### FALSE_MERGE / WRONG ASSOCIATION

clean：

```text
obs → Oa
```

强制：

```text
obs → Ob
```

#### WRONG_MEMBERSHIP

类似，但选择单个 member 污染。

#### 可选 postprocess FALSE_MERGE

强制错误 merge transaction。

重要：

> `G_clean` 是“未注入错误的 mapper reference”，不是现实世界 ground truth。它用于测“能否逆转我们注入的因果扰动”；真实 correctness 仍由 ReplicaSSG/Replica GT/人工自然错误实验补充。

---

## 20.2 Controlled Recovery Metrics

### Membership Recovery

对 clean/reference membership：

- Precision
- Recall
- F1

### Node Recovery

通过 GT/clean matching 后测：

- object identity recovery；
- over-merge / over-split；
- duplicate rate。

### Geometry Recovery

可选组合：

- 3D IoU / bbox IoU；
- point support；
- Chamfer-like distance；
- GT coverage。

### Relation Recovery

- repaired relation vs clean relation；
- ReplicaSSG GT relation；
- graph edit distance / edge F1。

### Recovery Ratio

推荐作为归一化指标：

```text
Recovery = 1 - d(G_repair, G_clean) / d(G_corrupt, G_clean)
```

其中 `d` 必须分 node/member/geometry/relation 分项报告，不要只给一个黑盒分数。

### Collateral Damage

必须测：

```text
outside dependency closure changed entities = 0
```

或极低。

---

# 21. Natural Error Evaluation —— 证明现实价值

现有 97 endpoint / 40 confirmed wrong 非常有价值，但要严格说明：

> 这是 screener-triggered final endpoint population，不是 full-map error recall census。

下一步对 40 confirmed wrong 做 expert causal trace，重点挑：

- FALSE_MERGE
- FALSE_SPLIT
- WRONG_MEMBERSHIP
- semantic error

结构 repair 要增加：

- beneficial
- neutral
- harmful

人工/GT 裁决。

当前 `ali-my-VLM` 的 +12.55 pp mIoU 等结果保留为 baseline evidence，但必须继续明确：

> saved label utility improvement ≠ underlying map-state recovery.

---

# 22. Relation Layer Evaluation —— 队友独立负责

比较：

1. `ali-dev/ConceptGraphs relation`
2. 队友 backend
3. ablations

指标：

- Candidate Recall ↑
- Edge P/R/F1 ↑
- Predicate R@1/mR@1 ↑
- Relationship R@1/mR@1 ↑
- Edge density ↓/合理
- latency/frame ↓
- incremental update cost ↓

队友的方法如果自身结果非常强，可作为 Contribution 3 的主要组成；如果只是稳健改进，它仍然是 revision closure 的关键系统组件，不影响主论文成立。

---

# 23. Online Evaluation —— 必须体现“在线”不是一句话

## 23.1 VLM delay stress test

人为增加：

```text
0 / 1 / 3 / 5 / 10 s
```

或：

```text
0 / 10 / 30 / 50 / 100 frame delay
```

比较：

### Naive strict-version

```text
version mismatch → restart
```

### Ours

```text
dependency conflict + delta rebase
```

指标：

- successful commit rate；
- stale/conflict rate；
- automatic rebase rate；
- VLM re-query count；
- time-to-correct；
- replay cost；
- mapping FPS overhead。

---

## 23.2 Maturity ablation

比较：

- immediate repair；
- maturity-aware repair。

随：

- observation count；
- view diversity；
- entity lifetime

报告：

- repair precision；
- harm rate；
- time-to-correct。

---

# 24. 必须有的 Baselines / Ablations

推荐主表至少：

```text
ConceptGraphs / ali-dev
Corrupted ConceptGraphs
Checker-only (no repair)
VLM-only final-map patch
Global Replay
Ours Local Replay + baseline relation
Ours Local Replay + improved relation
Ours w/o maturity
Ours w/o rebase
Ours w/o relation verification
Oracle repair constraint
```

### `Global Replay`

非常重要。

如果：

```text
Local replay quality ≈ Global replay
Local replay cost << Global replay
```

论文的“局部”价值会非常直观。

---

# 25. 关键指标总表

| 层级 | 指标 |
|---|---|
| Trigger | endpoint precision/recall（只在有 full census 时）/ workload reduction |
| Diagnosis | error-type accuracy / abstention calibration |
| Repair | Repair Precision / Recovery Rate / Member F1 |
| Safety | Harm Rate / False Mutation Rate / Collateral Damage |
| Identity | over-merge / over-split / duplicate |
| Geometry | IoU / support / coverage / fragmentation |
| Relation | candidate recall / edge F1 / R@1 / mR@1 |
| Online | TTC / rebase rate / re-query / commit success |
| Efficiency | map FPS / replay ms / VLM cost / memory |
| Downstream | semseg / ReplicaSSG / grounding/retrieval |

---

# 26. 两人明确分工

## 26.1 你：Revision Kernel Owner

你必须拥有论文主线：

- Provenance contract / index
- MaturityManager
- Trigger aggregation
- RepairTicket
- VLM adjudicator adaptation
- CausalTracer
- DependencyClosure
- Snapshot/shadow
- observation materialization
- local counterfactual replay
- false merge/split/wrong membership operators
- delta rebase
- lineage handling
- transaction
- verification/commit
- controlled corruption benchmark
- natural error repair evaluation
- online latency experiments

你的核心交付不是“VLM accuracy”，而是：

> **一个历史错误真的能被追到过去、局部重新执行、回到更正确状态。**

---

## 26.2 队友：Relation Layer Owner

队友负责：

- ReplicaSSG relation evaluator
- ConceptGraphs/ali-dev edge baseline
- candidate coverage diagnosis
- high-coverage physical relation candidate graph
- voxel/geometric relation evidence
- relation confidence / maturity
- local edge invalidation
- relation rebuild
- relation consistency API
- relation baseline/ablation
- repair 前后 relation recovery
- runtime optimization

---

## 26.3 共同负责

- node↔edge 接口冻结；
- controlled repair scene selection；
- joint verifier；
- main table；
- system figure；
- paper writing；
- final demo。

---

# 27. 推荐执行顺序

## Phase 0：现在立即冻结接口

共同确定：

```text
EntityRef
ObjectVersionRef
RepairConstraint
RepairTicket
RevisionTransaction
RelationBackend
```

不要边写代码边改数据契约。

---

## Phase 1：先证明 replay engine，不先证明 VLM

你的第一阶段：

> 用 oracle-controlled constraint 测 replay 是否正确。

原因：

如果 VLM diagnosis 和 replay 同时调试，失败时不知道是谁的问题。

必须先证明：

```text
known wrong event
+ known correct repair constraint
→ local replay
→ state recovery
```

---

## Phase 2：你做 FALSE_MERGE，队友做 relation evaluator

并行：

### 你

第一个完整结构闭环：

```text
inject false merge
→ trace
→ split member groups
→ refusion
→ replay later events
```

### 队友

```text
ali-dev relation
→ ReplicaSSG evaluator
→ candidate recall
→ edge/predicate/relationship metrics
```

---

## Phase 3：第一次 node→edge closure

即使队友新 relation backend 尚未完成，也先用 baseline relation wrapper 跑通：

```text
node repair
→ invalidate edge
→ baseline edge replay/rebuild
→ no dangling edge
```

然后新 relation backend 再替换。

---

## Phase 4：VLM 从“patcher”改成“constraint proposer”

复用 `ali-my-VLM`：

```text
audit
→ diagnosis
→ typed constraint
```

去掉：

```text
direct final-map structural overlay
```

---

## Phase 5：加入 maturity + online rebase

先做：

- TENTATIVE/MATURE；
- WAIT_EVIDENCE；
- disjoint/append-only delta；
- local rebase。

再做复杂 lineage structural rebase。

---

## Phase 6：扩大 controlled corruption

至少三类：

- false merge
- false split
- wrong membership

多场景、多随机种子。

---

## Phase 7：natural failures + final relation backend

最后才投入更多真实错误人工分析和第二 backend。

---

# 28. 内部 GO / NO-GO 门

这些不是 ICRA 官方标准，而是项目内部风险控制。

## Gate A：Replay 可行

必须至少有一个真实传播后的 FALSE_MERGE：

```text
corruption → propagation → local replay → measurable recovery
```

失败：

> 不再扩关系/通用性，先解决 replay。

## Gate B：安全性

建议内部目标：

- Repair Precision > 85%
- Harm / False Mutation < 5%

未达到时：

> 降低自动 commit 范围，增加 DEFER，不硬宣称 autonomous repair。

## Gate C：结构恢复

建议 controlled structural recovery > 70% 作为较强目标。

若只有 label 改善、geometry/membership 无恢复：

> 不能以 retrospective structural repair 为 headline。

## Gate D：局部性

必须证明：

```text
Local Replay ≈ Global Replay quality
Local Replay << Global Replay cost
```

否则“local replay”价值不足。

## Gate E：online

VLM delay 增加后：

- commit 不应因 global version 变化大量饿死；
- re-query 不应线性爆炸；
- mapping fast path 应基本不被阻塞。

---

# 29. 最终论文三个贡献点建议

## Contribution 1

**Evidence-grounded revisable online 3DSG**

不是“我们有日志”，而是：

> mapping decisions, object versions, observations, and graph dependencies are exposed as a replayable revision substrate.

## Contribution 2 —— 最核心

**Dependency-bounded counterfactual local replay**

> locate a causal historical mapping decision, modify its constraint, and replay only affected observations/events to recover the current state.

## Contribution 3

**Safe coupled node–relation restoration under asynchronous map evolution**

包括：

- node mutation → relation invalidation/rebuild；
- relation consistency → repair verification；
- maturity-aware defer；
- snapshot/delta rebase；
- local atomic commit。

---

# 30. 绝对禁止的过度 claim

论文中必须避免：

- “first graph version control”；
- “first asynchronous VLM 3DSG repair”；
- “first dynamic scene graph update”；
- “first voxel relation graph”；
- “VLM 修复提升 mIoU，所以 geometry 修好了”；
- “97 endpoint = 全地图 recall”；
- “只在 ConceptGraphs 实验但声称 universal across all maps”；
- “controlled clean branch = real-world ground truth”；
- “关系模块自己验证自己，所以证明 repair 正确”。

---

# 31. 最终论文成败判断

最理想的结果不是“所有指标都 SOTA”，而是形成一条很难被审稿人否定的证据链：

```text
(1) ConceptGraphs 的增量 hard decisions 确实产生可传播错误
                         ↓
(2) 当前 checker/VLM-only 能发现部分问题，但 direct patch 不能恢复底层结构
                         ↓
(3) provenance 能定位造成终态错误的历史决策
                         ↓
(4) counterfactual local replay 能恢复 member/geometry
                         ↓
(5) node change 后 relation state 被一致恢复
                         ↓
(6) 与 global replay 相近，但成本明显更低
                         ↓
(7) VLM 延迟下通过 rebase 仍能在线提交，不反复重启
                         ↓
(8) maturity-aware policy 显著降低早期误修
                         ↓
(9) controlled errors + natural errors 都成立
```

如果这条链跑通，这篇论文就不再是“ConceptGraphs 的若干增量改进”，而是一个明确的新研究对象：

> **revisable online 3D scene graph mapping。**

---

# 32. 文献定位清单（执行阶段必须持续对照）

建议持续对照以下工作，避免实现过程中把已有内容重新包装成创新：

1. ConceptGraphs — object-centric open-vocabulary 3DSG 基础。
2. Think While You Map / ThinkGraphs (2026) — async VLM online graph enrichment/correction。
3. PUF (ECCV 2026) — uncertainty-aware online node/relation fusion。
4. FROSS (ICCV 2025) — fast online 3D SSG + ReplicaSSG。
5. CoPa-SG (ICCVW 2025) — dense/precise parametric relations。
6. KNA-SG (2026) — keyframe-node evidence and query-time relation verification。
7. Graph Rectification (ACL 2026) — version control / rollback / conflict tracing。
8. DovSG — dynamic graph local update。
9. Khronos — fast active-window + slow long-term reasoning 的多速在线系统思想。
10. Open3DSG / RelationField 等 — open-vocabulary relation prediction 已有充分研究。

---

# 33. 最后的项目决策

从现在开始，所有开发都应服务于下面这一条主路径：

```text
Evidence
→ Suspicion
→ Wait until mature
→ Diagnose
→ Trace historical cause
→ Fork local branch
→ Counterfactual replay
→ Rebuild dependent relations
→ Rebase live delta
→ Verify
→ Commit or Abort
```

**你的第一优先是把“历史节点修复”变成真实的 replay，而不是继续优化 final-map VLM overlay。**

**队友的第一优先是把关系评价做严谨，然后让 relation backend 成为 node revision 后可局部重建的状态层，而不是独立堆在论文旁边。**

只要先完成一个完整的 `FALSE_MERGE → trace → member regroup → replay → relation rebuild → verify → commit` 闭环，整篇 ICRA 的方法骨架就真正成立。
