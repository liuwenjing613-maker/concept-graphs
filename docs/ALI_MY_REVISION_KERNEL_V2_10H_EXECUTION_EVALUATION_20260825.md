# ALI-MY Revision Kernel V2 远程执行与验证总结（2026-08-25）

## 0. 执行信息

- 执行范围：仅远程服务器 /home/chenkejun/beauty/，未在本地工作区运行实验或修改代码。
- 基线提交：ba55d7ce9b67d7808f15944c390baee18743a5f6
- 工作分支：publish/ali-my-v2-10h-20260824
- 主实验目录：/home/chenkejun/beauty/conceptgraphs/experiments/revision_kernel_v2_10h_20260824
- 冻结协议：docs/revision_v2_10h_audits/EXECUTION_PROTOCOL_20260824.json
- 代码目录：/home/chenkejun/beauty/conceptgraphs/code/official/ali-my-revision-human6-publish-20260824
- 最终 revision 回归：123 passed, 2 warnings
- 生产提交：全部实验均为 production_commit_permitted=false；没有把实验约束写回数据集或正式场景图。

## 1. 总结结论

本轮工作的准确结论是：

> Revision Kernel 的可审计执行层已经从“只处理身份约束”推进到“身份、几何恢复、点级分区、关系影响和自动候选门控”的统一原型；核心 replay/invariant/local-global parity 较扎实，但自动约束生成仍不具备可用的修复准确率。整体状态应判定为 PARTIAL SUCCESS, FAIL-CLOSED，不能判定为完整方法成功。

| 层面 | 结果 | 可以得出的结论 |
|---|---:|---|
| 6 个人工确认 FM/FS | 3 例 association-root 可重放，3 例 non-association-root 延迟 | 不能把全部 6 例当成同一类身份问题 |
| 可重放 3 例 | Native 错 3/3，Natural 仍错 3/3，Sparse 修正 3/3 | 因果对照成立，条件修复率 100% |
| 所有 6 例总体修复 yield | 3/6 = 50% | 另外 3 例不是成功修复，而是当前原语覆盖不到 |
| 身份负例 | 2 个 exact no-op + 1 个合法 merge 全通过 | identity boundary 没有无条件阻断正常行为 |
| 组件消融 | FS 缺 redirect 失败；FM 缺 postprocess boundary 失败 | 两类机制必要性得到样例级支持 |
| 身份 local/global parity | 1 FS + 2 FM 均通过（含 table-part FM） | 同一约束的局部与全局实现一致 |
| 几何开发例 | 44 → 2,148 点，48.82×，local/global PASS | RESTORE_OBSERVATION_GEOMETRY 执行链打通 |
| 新 office0 几何 holdout | 633 → 15,364 点，24.27×，local/global PASS | 几何机制跨场景复现 |
| PARTITION 开发 oracle | 6,287 pre-voxel 点，2,506 emitted，3,781 excluded | 点级纯执行器成立，但尚未接入一对多 association |
| 小规模 relation gold | 44 labels，accuracy/balanced accuracy 0.9545，F1 0.9524 | 冻结基准得到复现，不代表 relation 提升 |
| 新鲜自动生成 | 5 cases / 15 votes，严格动作准确率 20% | 直接 blind action generation 不可用 |
| 新 fresh capability | 1/2 动作正确；1 geometry 修复，1 spurious 安全延迟 | 安全门控有效，但修复覆盖率仅 50% |
| 错误自动提交 | 0 | fail-closed 达成 |
| 生产提交 | 0 | 尚未达到 production promotion 条件 |

## 2. Identity / provenance 正式化

保留并扩展了 immutable observation provenance、stable entity/lineage identity 和稀疏约束语义：

- CREATE_INSTANCE 明确绑定 created identity，并支持 separate_from_identity_uids。
- association 阶段与 postprocess merge 阶段分别执行持久 identity boundary。
- 约束解析、runtime verifier、local/global replay 使用同一 identity/provenance 语义。
- evidence bundle 使用公开 alias，私有 binding 单独保存，模型请求不暴露最终 owner 或人工答案。
- identity targeted shadow search 在 3 个开发例中实现：
  - candidate target recall：3/3
  - selected matches gold：3/3
  - unsafe selection：0
  - wrong hypotheses rejected：3/3

这里必须区分两件事：

1. “生成少量假设，再用独立因果 shadow evaluator 选择”在 3 个开发例中有效；
2. “让模型直接输出最终动作”在新的冻结 5 例中身份准确率为 0/3。

因此当前可靠部分是 shadow search，不是直接自动动作生成。

### 2.1 身份负例、消融和全局一致性

已有实验根目录：

/home/chenkejun/beauty/conceptgraphs/experiments/revision_identity_provenance_v2_20260824

结果：

- exact no-op：
  - native CREATE no-op：PASS
  - native ASSIGN no-op：PASS
  - membership partition、bbox、center、extent、point support 均精确一致。
- 合法 merge：
  - 两个受保护边界存在时，未触碰边界的真实 merge 仍被接受；
  - runtime invariants 与 source hashes 均通过。
- 三例消融：
  - office0 FS：关闭 redirect 后 endpoint 失败。
  - 两个 room0 FM：关闭 association boundary 仍可由 postprocess boundary 挽救；关闭 postprocess boundary 后失败；两者都关闭也失败。
- local/global：
  - office0 FS、room0 FM 06525、room0 table-part FM 9727 均为 9/9 checks PASS；
  - membership partition exact、geometry exact、identity signature exact、endpoint exact。

这说明 identity kernel 不是靠一个全局 “cannot merge” 开关工作，而是由 association boundary、postprocess boundary 和 lineage redirect 分工。

## 3. RESTORE_OBSERVATION_GEOMETRY

### 3.1 实现语义

新增 hash-bound ObservationGeometryContract：

- 精确绑定 observation row hash；
- 精确绑定 raw mask、processed mask、RGB、depth、原 observation PCD；
- replacement PCD/mask 使用独立文件哈希和数组哈希；
- 几何 overlay 在 association similarity 计算之前应用；
- similarity 和严格阈值语义重新计算，不强制指定 identity target；
- overlay、similarity recompute、trace 必须各命中一次；
- source evidence 保持只读。

通用成功条件被修正为：

- replacement points/mask 精确；
- 点支持确实增加；
- 恢复几何非退化；
- observation 只有一个 owner；
- overlay/recompute/trace 各一次；
- 继续采用重新计算后的 Natural 决策；
- no-op 精确、invariants 通过、source immutable。

“点数增长 10×”只保留为诊断项，不再作为跨场景通用定义。开发例恰好增长 48.82×，fresh holdout 恰好增长 24.27×，但这不是方法语义。

### 3.2 开发例 room0

- raw mask：14,825 px
- processed mask：120 px
- 丢失率：99.19%
- original points：44
- restored points：2,148
- 支持增长：48.82×
- local Natural / no-op / Sparse：全部 invariants PASS
- local/global：membership exact，geometry IoU 1.0，center/extent error 0，owner、trace、payload binding exact
- source hashes 未变化。

### 3.3 fresh holdout office0

冻结 incident：

incident_fe3d3de7f35ffa7cd495

observation：

office0_20260820T090828Z_fbd2d994_f000191_r0002

人工答案在 freeze 和模型响应之后才读取，标签为 GEOMETRY_CORRUPTION，说明 raw mask 准确、processed mask 只留下碎片。

结果：

- raw mask：40,356 px
- processed mask：2,755 px
- 丢失率：93.17%
- original points：633
- restored points：15,364
- 支持增长：24.27×
- comparator：STRICT_GREATER_THAN
- threshold：1.2
- restored top-1：1.8217，margin：+0.6217
- local validation：PASS
- local/global parity：全部 16 checks PASS
- membership 1,560 observations exact
- geometry 29 objects exact
- source artifacts/provenance immutable。

恢复后 owner 从一个 2-detection 的 potted-plant fragment 变为一个 92-detection 的 potted-plant main entity。物理上这是合理合并，但人工标签只裁定几何错误，没有裁定 identity merge。因此正式结论是：

GEOMETRY_MECHANISM_PASS_WITH_IDENTITY_SIDE_EFFECT_UNADJUDICATED

不能把它记为已确认的 identity improvement，也不能把它直接记为 identity regression。

## 4. PARTITION_OBSERVATION

### 4.1 为什么没有沿用 stored observation points

原设计尝试直接对 stored 1,990 points 分区。深入核对后发现：

- raw valid-depth points：18,861
- pre-voxel sampled points：6,287
- voxel points：2,705
- stored post-DBSCAN points：1,990
- stored payload 中 mixed semantic voxels：76，占 3.82%
- majority semantic 与 centroid surface 判定不一致：22 voxels。

所以 stored-point partition 已经丢失点级实例信息，继续在其上分区会把 preprocessing artifact 当成 gold。本轮将该路线标记为 REJECT_AS_LOSSY。

### 4.2 当前最优实现

使用 Replica 官方 semantic mesh：

/data/chenkejun/ReplicaSSG/Replica/data/room_0/habitat/mesh_semantic.ply

它与 mapping mesh 顶点/面精确对齐。对 incident f74cb... 的 observation 进行 exact preprocessing reconstruction 后：

- source points：6,287
- emitted table #27：2,328
- emitted table #7：178
- emitted total：2,506
- excluded floor #25：3,705
- excluded rug #60：48
- excluded sofa #9：28
- excluded total：3,781
- exhaustive：true
- disjoint：true
- 双次运行 assignment hash 一致：edbbd2c451bfcf0b06fb356ff410d10317301ef6502c39f9095d4ce620cc6820

当前实现是 development oracle + pure executor，不是生产算法。它已解决 point assignment contract、emitted/excluded disposition、assignment hash binding、exhaustive/disjoint 验证、source preprocessing count trace 和 source immutability。

尚未解决：

- 一个 source observation 在 association 之前变为多个 child observations；
- child UID、事件序列和 lineage 的正式定义；
- emitted child 的独立 feature/PCD/mask；
- relation evidence 如何继承或拆分。

因此 association_stage=DEFER 是正确的 fail-closed 行为。

## 5. Relation gold 与 identity repair impact

room0 小规模 direction-aware gold：

- label count：44
- TP=20，FN=2，FP=0，TN=22
- accuracy：0.954545
- balanced accuracy：0.954545
- precision：1.0
- recall：0.909091
- F1：0.952381
- frozen prediction count：49。

使用 200 frames、2,425 条 edge observations，重新运行 human6 pilot 和 relation rebuild：

- case 06525：Native/Natural/Sparse 均 49 edges，relation set 和 support 完全相同。
- case 9727：relation set 仍为 49；只有受 identity localization 触碰的一条 “on top of” support 从 13 变为 12。
- 受影响范围外 relation set/support 完全一致。
- 所有 branch：dangling edge=0、unexpected self-loop=0、malformed relation=0、novel relation type=0。
- 两个 identity-changed formal-map index 都没有 frozen gold label，因此 44 个 stable labels 的指标保持完全一致。

可证明的是 relation backend 结构安全、未影响稳定 gold 范围。不可证明的是 relation accuracy 得到提升；本轮明确不提出该 claim。

## 6. 两个 fresh holdout 与自动生成

### 6.1 冻结方式

从 40 个 confirmed incidents 中排除 8 个已消费 incident，再按场景分层，以 minimum SHA256(protocol_salt|incident_uid) 每个场景取一个。选择过程只读取 UID、scene、case_dir、finding 和 trigger observation，不读取错误类型、人工备注、最终 owner 或 generator output。

- 冻结清单 hash：929380dc849a0a68431728729635228bb790f88ac370825ad01fceb01aa4a5241
- blind input hash：310e835c6eba78953f4e7a4c51bb18e8d3bea48d63c856b4aedf6e6494898edd
- freeze before answer access
- freeze before API responses
- no outcome-based replacement
- 每例只正式运行一次
- 失败后不改 prompt、不换样本。

### 6.2 自动生成协议与结果

- 5 cases：3 identity development controls + 2 fresh capabilities
- 每例 3 votes，共 15 calls
- 5 个 credential slots 并行轮换
- 同一模型，不把不同 key 声称为独立模型
- credential 仅存在进程内，实验目录和仓库扫描未发现 key 持久化
- inference protocol hash：02b800aebe396534105963e6035110f4e99efb2752ca1eb73fba726ca72bfedb
- generation result hash：ff17c1912e3915aeefcec360dbf6b2ff6cbfe3496a53b51c3ce68e18816529e9

| Case | Posthoc 期望 | 3 votes / strict | 结果 |
|---|---|---|---|
| identity office0 FS | SAME_INSTANCE | DEFER×3 | 错 |
| identity room0 FM | SEPARATE_MEMBER_GROUPS | DEFER×2 + SAME_INSTANCE×1，strict DEFER | 错 |
| identity room0 FM | SEPARATE_MEMBER_GROUPS | DEFER×3 | 错 |
| fresh room0 spurious | DEFER | RESTORE×3 | 错 |
| fresh office0 geometry | RESTORE | RESTORE×3 | 对 |

指标：

- vote action accuracy：3/15 = 20%
- strict aggregate action accuracy：1/5 = 20%
- identity strict accuracy：0/3
- fresh capability strict accuracy：1/2
- correct fresh payload：1/2
- automatic commit eligible：0
- unsafe automatic commit：0。

模型把 “raw mask 面积显著大于 processed mask” 当成恢复的充分证据。对于真实 geometry corruption 这是对的；对于 raw mask 本身就是墙面误检的 spurious object 则完全错误。面积损失只说明 preprocessing 改变很大，不能判断 raw mask 是否可信。

### 6.3 spurious object 的 fail-closed 处理

fresh room0 incident：

incident_b714504baedfbd92b49c

posthoc 标签为 SPURIOUS_OBJECT，人工说明应为 wall fragment。

当前 ConstraintType 没有安全的 DELETE / SUPPRESS_OBJECT 原语。直接执行模型给出的 RESTORE 会放大错误 raw mask。因此新增 capability registry：

- GEOMETRY_CORRUPTION → RESTORE_OBSERVATION_GEOMETRY，可执行候选；
- SPURIOUS_OBJECT → SUPPRESS_SPURIOUS_OBJECT，当前无 executor；
- 无 executor 时自动动作必须为 DEFER。

本例未 materialize constraint、未 replay，source hashes before/after 精确一致，错误 RESTORE 候选被 compiled/shadow gate 阻断。安全检查 PASS，但 endpoint 仍未修复。这是安全成功、效用失败，不能记为 repair success。

### 6.4 posthoc 总结

审计文件：

docs/revision_v2_10h_audits/FRESH_HOLDOUT_POSTHOC_EVALUATION_20260825.json

- conclusion：PARTIAL_SUCCESS_FAIL_CLOSED
- audit integrity and safety：PASS
- method complete：false
- fresh repaired：1/2
- unresolved：1/2
- production commits：0
- 双次 evaluator 输出 hash 一致：126d399593775911c854c7d53b2844ac92924622d860e6972eab79113e5edbb8

## 7. Timing 与资源使用

### 7.1 replay timing

| 场景 | cached local Sparse suffix | full global replay | 表面加速 |
|---|---:|---:|---:|
| room0 geometry dev | 65.62 s | 221.59 s | 3.38× |
| office0 fresh geometry | 6.49 s | 98.84 s | 15.23× |

但不能只报表面加速：

- room0 dev snapshot cold upper bound：2.59 s；
- office0 fresh snapshot cold upper bound：86.79 s；
- office0 cold local 总计约 93.28 s，与 full global 98.84 s 接近；
- office0 amortized snapshot 为 0.83 s，缓存后 local 才体现约 13.5× 的实际收益；
- room0 affected observation count：3,779；
- office0 affected observation count：1,560。

当前 local sparse replay 的 intervention 是稀疏的，但 dependency closure 仍很宽。下一步优化重点应是 closure，而不是只优化 overlay 的亚毫秒开销。

office0 三支 cached suffix：

- Natural：6.308 s
- exact no-op：6.297 s
- Sparse：6.489 s。

Sparse 相对 no-op 约增加 3%，geometry contract 本身不是主要瓶颈。

### 7.2 GPU

服务器可见 8 × NVIDIA RTX 5880 Ada Generation，每卡约 49 GB，driver 580.126.09。

但当前 cg-ali：

- torch 2.0.1
- CUDA build 11.8
- torch.cuda.is_available() == False
- driver initialization failed。

因此本轮 replay、Open3D、semantic partition 实际使用 CPU fallback；没有虚报 GPU 加速。GPU 硬件存在，但 Python/CUDA runtime 组合需单独修复。继续扩大实验前，应先建立一个最小 CUDA allocation smoke test，而不是重复跑大场景。

### 7.3 API

五个 credential slot 被用于同一轮 15-call blind generation。API key 未写入命令行、JSON 或仓库；结束后内存列表清空，安全扫描通过。不同 credential slot 只代表并发和配额隔离，不代表模型独立性。

## 8. 过程中做出的方向调整

1. **拒绝 stored-point partition**：发现 76 个 mixed semantic voxels 后，改用 pre-voxel point assignment + official semantic mesh。
2. **限制 relation claim**：relation metrics 稳定不等于提升；affected endpoints 没有 gold。
3. **修正通用 geometry 判据**：开发例的 10× point gain 不是方法定义。
4. **不执行 spurious 的错误 RESTORE**：模型 3/3 高置信选错；缺 suppression primitive 时正式动作 DEFER。
5. **不把 owner 变化计作正确**：fresh geometry 的同类合并尚无 identity gold。
6. **未扩跑全部场景**：只在足以改变设计的位置跑正式 global parity，避免方法未稳定时浪费算力。

## 9. 与完整框架的对齐程度

| 完整框架层 | 当前实现 | 主要缺口 |
|---|---|---|
| Immutable observation/event provenance | 已实现并由 hash/invariant 保护 | partition child observation provenance 未完成 |
| Stable entity/lineage identity | 已实现 | 自动 evidence 到 identity constraint 的准确映射不足 |
| Sparse typed constraints | ASSIGN/CREATE/MUST/CANNOT/PARTITION/RESTORE/RELABEL 等已具备 | 缺 observation/entity suppression |
| Natural vs Sparse causal contrast | 3 个 replayable human cases 成立 | 3 个 non-association-root 仍缺对应机制 |
| Local/global same-constraint parity | identity 和 geometry 均有真实例通过 | closure 太宽；冷启动优势不足 |
| Exact no-op / legal-action controls | 已通过 | 新 suppression/partition 接入后必须重跑 |
| Point-level correction | pure PARTITION executor 已实现 | 未接一对多 pre-association |
| Relation rebuild and gold | 小规模 gold 与结构安全已实现 | affected relation endpoints 缺 gold |
| Automatic constraint generation | 冻结、并行、无泄漏、shadow fail-closed | 直接动作准确率仅 20%，utility 不合格 |
| Promotion gate | unsafe commit=0 | 尚无任何 commit-eligible fresh case |
| Production deployment | 未启用 | 证据远不足以启用 |

## 10. 当前最急需的下一步设计

### P0-1：先实现 audit-safe suppression，而不是物理删除

建议新增 SUPPRESS_OBSERVATION，暂不新增不可逆 DELETE_ENTITY：

- 绑定 obs UID、事件 UID、active sequence、reason、evidence hashes；
- 在 association 输入前将 observation 标记为 inactive；
- observation 仍保留在 provenance，不删除原始证据；
- entity 因 suppression 变成空集时输出 inactive/tombstoned entity；
- relation rebuild 必须移除 inactive endpoint；
- verifier 检查 suppressed observation 无 active owner、其他 observation unique owner、source immutable、relation 无 inactive dangling endpoint；
- 补 exact suppression no-op、合法 merge、local/global parity 和当前 spurious fresh formal replay。

这种设计比 DELETE_ENTITY 更符合完整框架的可逆、可审计原则。

### P0-2：把自动生成拆成“证据有效性 → 能力路由 → 参数绑定”

不要再让模型一步输出最终 action。建议三段：

1. Evidence validity：RAW_MASK_TRUSTWORTHY / RAW_MASK_SPURIOUS / MULTI_INSTANCE_CONTAMINATION / AMBIGUOUS
2. Capability route：restore / suppress / partition / relabel / identity / defer
3. Deterministic binding：obs UID、entity alias、contract hashes、event sequence 由代码绑定。

mask validity 优先加入非语言证据：

- semantic-mesh wall/floor overlap；
- depth continuity；
- 3D compactness 和 thin-planar ratio；
- 多帧复现率；
- raw/processed support ratio；
- 与当前 object geometry 的一致性。

raw/processed area ratio 只能是一个特征，不能单独触发 RESTORE。

### P0-3：裁定 fresh geometry 的 identity side effect

只需做一个小型人工复核，不需要再跑全场景：

- 原 2-detection potted-plant fragment；
- 目标 92-detection potted-plant entity；
- 恢复 observation 的 raw-mask 3D overlay；
- 两者是否同一物理盆栽。

这一步决定 geometry restore 后的 Natural reassociation 是否可以进入 endpoint success 定义。

### P0-4：完成 PARTITION 的 pre-association one-to-many integration

建议 child UID 为：

{source_obs_uid}::part::{part_uid}

并新增显式事件 OBSERVATION_PARTITIONED：

- source observation 保留但 inactive for association；
- emitted child 独立 materialize；
- excluded child 有 contamination disposition；
- 每个 source point 恰属于一个 child；
- child features 在 partition 后重新计算；
- local/global parity；
- relation evidence 只从 emitted child 继承。

在完成这些之前，不能把当前 semantic-mesh oracle 写成生产 PARTITION。

### P1：自动 identity 改用 hypothesis search

直接 blind action selector 0/3，继续堆 prompt 价值很低。下一版应：

- deterministic candidate retrieval 保证 target recall；
- API 只提出 2–3 个可绑定 hypothesis；
- 每个 hypothesis 编译为 shadow-only constraint；
- 独立 causal replay 选择；
- 没有唯一通过者就 DEFER。

本轮 targeted shadow search 3/3，而直接 selector 0/3，已经给出明确设计证据。

### P1：缩窄 causal closure

记录每次 scope expansion 的触发 version/entity，并尝试 version-level dependency cut、relation rebuild 延迟、prefix cache 按 frame + source hash 复用，以及 local/global mismatch 时自动扩大 scope。目标必须同时报告 cold 和 warm timing，不能只报告 cached suffix。

## 11. Claim boundary

本轮可以声明：

- 3 个 replayable human-confirmed identity cases 中 Natural 仍错而 Sparse 修正；
- identity/geometry local-global 实现一致；
- geometry restoration 在 room0 开发例和 office0 fresh 例均通过；
- point-level PARTITION oracle/executor 是 exhaustive/disjoint；
- relation stable-gold 指标和结构不受局部身份修复破坏；
- 自动门控没有放出错误提交。

本轮不能声明：

- 对全部 40 个 confirmed errors 有 50% 或 100% 修复率；
- 自动生成已经可用；
- relation accuracy 得到提升；
- PARTITION 已生产可用；
- fresh geometry 引起的 identity merge 已被人工确认；
- GPU 已被有效利用；
- 当前结果可以直接写回正式场景图。

## 12. 关键审计文件

- docs/revision_v2_10h_audits/EXECUTION_PROTOCOL_20260824.json
- docs/revision_v2_10h_audits/IDENTITY_EVIDENCE_BUNDLE_MANIFEST_20260824.json
- docs/revision_v2_10h_audits/IDENTITY_SHADOW_SEARCH_AUDIT_20260824.json
- docs/revision_v2_10h_audits/GEOMETRY_RESTORATION_BUILD_AUDIT_20260824.json
- docs/revision_v2_10h_audits/GEOMETRY_RESTORATION_LOCAL_AUDIT_20260824.json
- docs/revision_v2_10h_audits/GEOMETRY_RESTORATION_GLOBAL_PARITY_AUDIT_20260824.json
- docs/revision_v2_10h_audits/PARTITION_PREVOXEL_ORACLE_AUDIT_20260825.json
- docs/revision_v2_10h_audits/RELATION_CAUSAL_IMPACT_AUDIT_20260825.json
- docs/revision_v2_10h_audits/FRESH_BLIND_HOLDOUT_MANIFEST_20260825.json
- docs/revision_v2_10h_audits/FRESH_HOLDOUT_BLIND_INPUT_MANIFEST_20260825.json
- docs/revision_v2_10h_audits/FRESH_HOLDOUT_OFFICE0_GEOMETRY_BUILD_AUDIT_20260825.json
- docs/revision_v2_10h_audits/FRESH_HOLDOUT_OFFICE0_GEOMETRY_LOCAL_AUDIT_20260825.json
- docs/revision_v2_10h_audits/FRESH_HOLDOUT_OFFICE0_GEOMETRY_GLOBAL_PARITY_AUDIT_20260825.json
- docs/revision_v2_10h_audits/FRESH_HOLDOUT_POSTHOC_EVALUATION_20260825.json

## 13. 最终判断

当前最优决策不是继续批量跑更多场景，也不是放宽自动门控以换取表面成功率。应该先：

1. 实现可审计 SUPPRESS_OBSERVATION；
2. 把 mask validity 与 repair capability 分开；
3. 裁定 fresh geometry 的 identity side effect；
4. 完成 PARTITION pre-association integration；
5. 再冻结少量新 holdout。

这样下一轮增加的每个样例都会验证新的机制，而不是重复证明现有 prompt 在相同缺口上失败。
