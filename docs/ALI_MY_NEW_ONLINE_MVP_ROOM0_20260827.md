# ali-my-new 在线对象级修复 MVP：room0 完整实验总结

日期：2026-08-27

分支：`ali-my-new`

最终代码提交：`b0e0abf`
工作树：`/home/chenkejun/beauty/conceptgraphs/code/official/ali-my-new`

## 1. 结论

本轮已经跑通一个不使用 GT、人工标注或旧错误标签的完整在线闭环：

`stride=10 在线建图 → provenance/identity 证据流 → 广泛弱错误扫描 → object 级 ticket → 三级优先队列 → VLM 精选证据诊断 → 固定身份约束 → 同水位 NOOP/候选影子重放 → 场景末组合重放 → 实验内版本切换`。

最终干净实验完成 200/200 个处理帧，生成 79 个 object ticket，调用 20 次 VLM，启动 5 个影子候选，其中 2 个通过、3 个安全拒绝。两条互不冲突的通过约束组合后，最终结构验证和 8 个无标注安全门全部通过，实验内活动指针切到 `v1_repaired`。没有修改全局生产地图，保留了 `v0_baseline` 回滚版本。

这证明当前思路在一个完整场景上具备工程可执行性和安全闭环，但不能据此声称真实世界准确率提升，因为运行和评价都没有使用独立 GT。

## 2. 最终实验协议

- 场景：Replica `room0`
- 输入范围：`start=0, end=2000, stride=10`
- 实际处理帧：200，最终处理帧编号 199
- 最终事件序列：9874
- 建图 GPU：0
- 影子重放 GPU 环境：1；重放主体仍主要是 CPU/Open3D
- 关系边：关闭，先只验证身份闭环
- 检测：复用 `room0_detections_stride10`，没有读取人工标注或 GT
- VLM：只使用当前可验证的一个有效凭据；凭据仅保存在进程内存，不写入磁盘
- VLM 证据成熟期：ticket 至少等待 10 个处理帧
- VLM 上限：20 个 ticket
- 影子重放上限：5 个候选，单队列非抢占执行
- 总墙钟时间：2944.8 秒，约 49 分 05 秒；包含在线建图、20 次 VLM、5 个双分支影子重放和最终基线/组合重放

最终实验目录：

`/home/chenkejun/beauty/conceptgraphs/data/Replica/room0/exps/ali_my_new_room0_online_valid1_stride10_20260827T012740`

## 3. 实现内容

核心文件：

- `conceptgraph/revision/online_mvp.py`
- `scripts/run_ali_my_new_online.py`
- `tests/test_online_mvp.py`

MVP 保留的最小模块如下：

1. 增量证据账本：只读取换行完整的 JSONL 记录；每次轮询冻结文件 EOF，避免快速 writer 使 reader 永远追不上。
2. 一帧延迟提交：看到下一帧后提交上一帧；映射结束时提交最后一帧。
3. 广泛弱扫描：覆盖近阈值创建/关联、关联歧义、语义冲突/漂移、几何跳变、后处理合并冲突等；所有结果仅是 hypothesis，不是真值。
4. object ticket：按稳定 lineage/object scope 合并同一对象问题，避免把每个帧级异常都交给 VLM。
5. 严格字典序优先级：任务阻塞优先；否则按受影响对象数、事件数；仍相同时按 ticket age。
6. 精选 VLM 证据：触发 RGB+机器 mask overlay、触发 crop、当前实体和至多两个候选的清晰且时间多样视图，总数最多 6 张，并保存选择理由和哈希。
7. 固定身份动作：`SAME_INSTANCE`、`MOVE_OBSERVATION`、`SEPARATE_MEMBER_GROUPS`、`DEFER`。几何、语义、动态性或证据不足一律 `DEFER`。
8. 稀疏约束编译：只允许 `ASSIGN_OBSERVATION` 或 `CREATE_INSTANCE`，并把图片别名绑定到不可变 observation/version provenance。
9. 同水位门控：NOOP 必须精确复现；候选必须结构合法、约束命中且产生持久分区变化、目标身份满足。
10. 场景末组合门：结构验证、状态分区改变、观测数守恒、对象数至少保留 NOOP 的 90%、语义纯度/单例率/低纯度率不明显退化、无重复归属、无无效几何。

## 4. 问题扫描和 VLM 分布

在线扫描共生成 129 个弱问题，合并为 79 个 object ticket：

| 弱问题类型 | 数量 |
|---|---:|
| AMBIGUOUS_ASSOCIATION | 44 |
| SEMANTIC_ASSOCIATION_CONFLICT | 35 |
| NEAR_THRESHOLD_ASSOCIATION | 22 |
| POSTPROCESS_MERGE_CONFLICT | 14 |
| GEOMETRY_JUMP | 6 |
| NEAR_THRESHOLD_CREATE | 5 |
| SEMANTIC_DRIFT | 3 |

20 次 VLM 的动作分布：

| 动作 | 数量 |
|---|---:|
| DEFER | 14 |
| SAME_INSTANCE | 4 |
| MOVE_OBSERVATION | 1 |
| SEPARATE_MEMBER_GROUPS | 1 |

编译器生成 5 个 `ASSIGN_OBSERVATION` 和 1 个 `CREATE_INSTANCE` 候选；由于影子上限为 5，只验证优先级最高的 5 个。5 个影子结果为 2 个 `WOULD_COMMIT`、3 个 `DEFER`。

## 5. 两条实际采用约束

| 票 | 根源帧/观测 | 弱触发 | VLM | 冻结帧 | 在线目标门 | 影响排序 |
|---|---|---|---|---:|---|---|
| `ticket_572f97c327cbbf36` | 帧 71，armchair，`r0006` | NEAR_THRESHOLD_CREATE | SAME_INSTANCE，0.86 | 91 | owner 改变，22 个支持帧，pass | 5 个对象谱系，1089 个事件 |
| `ticket_2a70e38dfc839b52` | 帧 75，lamp，`r0016` | NEAR_THRESHOLD_CREATE | SAME_INSTANCE，0.99 | 100 | owner 改变，28 个支持帧，pass | 5 个对象谱系，321 个事件 |

两条约束均编译为 `ASSIGN_OBSERVATION`。第一条在冻结水位的因果闭包包含 1550 个观测、861 个事件、136 个实体；第二条包含 1782 个观测、1229 个事件、138 个实体。这说明一个早期身份决定会产生很大的后续传播范围，也解释了为什么必须从根源约束重放，而不能直接编辑最终成员表。

三个拒绝案例证明安全门有效：

- 冻结帧 84：候选分区改变，但目标身份未满足，拒绝。
- 冻结帧 199：目标形式门通过，但最终分区与 NOOP 相同，拒绝。
- 另一个冻结帧 199 候选也没有持久分区变化，拒绝；其中间重放曾出现明显对象数波动，但没有进入最终组合。

## 6. 最终无标注指标

| 指标 | NOOP 基线 | 修复候选 | 变化 |
|---|---:|---:|---:|
| object 数 | 72 | 71 | -1 |
| observation 数 | 3779 | 3779 | 0 |
| 加权语义纯度 | 0.728764 | 0.728764 | 0 |
| 低纯度 object 数 | 19 | 19 | 0 |
| 低纯度 object 率 | 0.263889 | 0.267606 | +0.003717 |
| singleton 数 | 20 | 20 | 0 |
| singleton 率 | 0.277778 | 0.281690 | +0.003912 |
| 平均每 object 唯一帧数 | 39.5694 | 40.0986 | +0.5291 |
| 重复 observation 归属 | 0 | 0 | 0 |
| 无效几何 object | 0 | 0 | 0 |

低纯度率和 singleton 率的小幅上升来自分母从 72 变为 71；对应的低纯度对象数和 singleton 数都没有增加。加权语义纯度没有改善，因此不能宣称宏观语义指标变好；本轮收益是两处有图像证据支持的身份创建错误被改为已有实例关联，同时保持全部观测、结构和全场健康指标基本稳定。

最终 8 项门全部通过：

- `partition_changed=true`
- `observation_count_conserved=true`
- `object_count_not_collapsed=true`
- `semantic_purity_not_degraded=true`
- `singleton_rate_not_degraded=true`
- `low_purity_rate_not_degraded=true`
- `no_duplicate_ownership=true`
- `no_invalid_geometry=true`

## 7. 实现中发现并最小修复的问题

1. 原证据中个别帧的 observation `filtered_det_idx` 超出相似度矩阵行数。没有补矩阵或伪造证据，而是在 VLM 前验证整帧矩阵契约；不兼容票结构化 `DEFER`，影子异常也不会拖垮主循环。
2. 五个给定凭据中四个在当前服务返回 HTTP 401。首轮在线实验因此只有一个槽完成少量调用。最终实验只使用已验证有效的一个槽；密钥没有持久化。
3. 复用完整账本时，每个历史帧都重复刷新全局因果排序，造成无意义的平方级开销。复用模式改为读完整批次后只刷新一次；在线模式仍逐水位刷新。
4. 快速热缓存建图时，JSONL writer 的写入速度高于 reader，原 tail 循环会追逐移动 EOF，在线水位被饿死。修复为每次 poll 先冻结文件长度，只读取本轮快照，下一轮再读取新增记录，并增加并发追加回归测试。
5. 最终门最初缺少显式观测守恒和对象防塌缩检查。新增两个简单硬门，并用同一两条约束重新验证通过。

## 8. 当前限制

1. 没有 GT 或人工标签。本报告的 purity 是机器语义自一致性，不是独立真实身份准确率。
2. 本场景没有实时机器人任务上下文，因此所有 ticket 的 `task_blocking=false`。三级排序的第 2、3 级已实际运行，第 1 级只有接口和单元测试，尚需接入真实 planner/executor 的依赖对象集合。
3. 最终切换发生在场景结束的实验内版本指针，不是正在执行机器人任务的生产地图原子热切换。
4. 影子重放仍从历史根部重复运行，主体主要消耗 CPU/Open3D；GPU 不能显著缩短当前实现。后排票在单 API 槽下直到帧 199 才启动，成本较高。
5. 当前用“一帧延迟”近似 frame closed。更严格实现应由证据 writer 写入显式 `FRAME_CLOSED` 水位，避免迟到的后处理事件被错过。
6. 只实现身份类约束；语义、几何和动态性问题被保守延期，没有试图用一个 MVP 统一修复所有错误。

## 9. 下一步最小优化顺序

1. API 启动健康检查：启动前验证凭据，只为健康槽分配 ticket，避免失败槽消耗调用上限。
2. 接入真实任务上下文：从 planner/executor 提供当前必须依赖的 object/lineage/relation UID，实际验证阻塞任务优先级。
3. 显式 `FRAME_CLOSED`：由 mapper 记录每帧全部身份/后处理证据已完成的水位。
4. 周期快照与增量追赶：例如每 20 个处理帧保存轻量 checkpoint；候选从最近 `s-1` 快照启动并持续追到移动的 `c`，减少场景末完整重放。
5. 明显塌缩早停：候选中间对象数长期低于 NOOP 合理范围时提前停止，保留证据并 `DEFER`，节省 CPU。

在完成上述 1–3 项前，不建议增加多票联合 VLM、复杂学习式影响分数或关系边重建；当前最重要的是把已验证的简单闭环接到真实任务和明确水位上。

## 10. 可审计产物

- 运行摘要：`online_mvp/run_summary.json`
- 在线事件：`online_mvp/online_events.jsonl`
- object ticket：`online_mvp/tickets.json`
- VLM 证据/响应/编译：`online_mvp/vlm/`
- 冻结账本：`online_mvp/frozen/`
- 影子结果：`online_mvp/shadow/`
- 最终比较：`online_mvp/final/final_comparison.json`
- 活动指针：`online_mvp/final/active_version.json`
- 修复状态：`online_mvp/final/versions/v1_repaired/state.json`
- 回滚状态：`online_mvp/final/versions/v0_baseline/state.json`

提交记录：

- `a88b8a6`：最小在线 object-ticket 主循环
- `7e6b8c5`：复用实验输出隔离
- `213f8f6`：复用账本只刷新一次影响排序
- `a13d039`：观测守恒和对象防塌缩安全门
- `b0e0abf`：冻结 JSONL 轮询 EOF，修复快速 writer 饿死

验证：cg 环境语法编译通过；独立测试环境 `7 passed`。所有最终实验协议文件均记录 `annotations_loaded=false`、`ground_truth_loaded=false`、`api_keys_persisted=false`。
