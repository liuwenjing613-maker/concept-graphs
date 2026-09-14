# Human 门控新增触发：实现与 smoke（2026-09-04）

## 范围与结论

已在服务器 `ali-my-VLM0903` 工作树实现两项新增触发。保留此前尚未提交的 human DISCARD 修改，不改提示词、人工动作语义、证据卡、原两个分数触发和后处理。

服务器工作树：`/home/chenkejun/beauty/conceptgraphs/code/experiments/ali-dev-blocking-gate-v1-20260903`

本次只做 smoke：12 项原有回归 + 11 项新增测试通过；Hydra 配置组合、py_compile、git diff --check 通过。未运行完整场景、未调用 VLM/API、未进行 GPU 实验或精度评测。没有执行 commit/push。

## 触发规则

| 情况 | 实现 |
| --- | --- |
| 原分差触发、原接近阈值 NEW 触发 | 保留原逻辑及原 IoU 候选筛选规则 |
| 原决定 ATTACH，3D 支持度明显下降 | 同一对象最近5次未触发关联中，至少3次；历史支持度中位数≥0.75，当前下降≥0.20，触发 `mask_change` |
| 原决定 NEW，原有规则未触发 | 无论分数是否接近阈值，补充触发 `all_new`；包含空地图首帧和没有有限分数候选的情况 |
| 一个观测同时满足多个条件 | 只生成一个事件、只询问一次；日志记录多个 reasons |

“所有 NEW”指经过现有 2D/3D 预处理后进入关联器、原决定为 NEW 的有效 detection，不是恢复原先过滤掉的原始 masks。人工将 ATTACH 改成 NEW 后不会再重复询问。

`mask_change` 沿用 association 提示词/候选数量，`all_new` 沿用 create 提示词/候选数量。几何触发一旦满足，不受 IoU 去重取消：去重只剩 Candidate A 仍送审。空地图没有伪造候选，人工可选 NEW / UNCERTAIN / DISCARD。

## 空间分数与在线历史

直接传入 mapper 已算好的、融合前的 `spatial_sim`。已核对 `compute_overlap_matrix_general`：对当前 detection 的每个点查询历史 object 最近点，距离严格小于 0.025m 的比例；包围盒不重叠时该值被快速置0。未重新计算分数、未修改距离参数。

历史规则：

- 以稳定 object `id` 为键，不以会变化的列表下标为键。
- 每帧先冻结所有历史检查，再逐观测审查，最后更新窗口。同帧较早观测不会进入同帧较晚观测的参考值。
- 仅最终未触发门控的 baseline ATTACH 进入历史；原始分数触发被 IoU 取消、且几何规则不触发的关联仍属于未触发关联。
- 触发后无论选 A/B/NEW/UNCERTAIN/DISCARD，均不用于训练历史窗口。因 max_events 上限跳过的风险事件也不用于训练历史。
- 对象列表重排后历史跟随 UID；消失的 UID 被清除。对象合并后只保留存活 UID 自己已有的窗口，不把另一个对象窗口移植过来。
- 支持度必须有限且处于[0,1]、矩阵维度必须一致、帧号必须递增；非 overlap 度量启用此规则会明确报错，避免误用 IoU/GIoU 为支持度。

支持度更新发生于本帧 route 结束，在正常 mapper 中紧接着执行本帧融合；只有进入下一帧才会使用这些记录。不读取旧场景最终地图、GT 或未来 observation。

## 动作保持不变

Candidate → 关联相应实例；NEW → 新建；UNCERTAIN → baseline；DISCARD → 舍弃当前观测，不创建也不更新对象。

首帧原来直接 `objects.extend` 绕过门控，现已接入同一空地图快照审查。只有保留的观测才进入地图，计数及对象版本日志也排除 DISCARD。

已将修改前后所有 PROMPT/POLICY 常量、`_prompts` 方法以及 `route_choice` 方法逐项 AST 比对，完全一致。零候选 VLM 请求仅修正了空 enum 的结构有效性，正式动作不变；本次没有发送该请求。

## 配置与用户后续运行

Hydra `association_gate` 默认新增：

```yaml
review_all_new: true
mask_change_enabled: true
support_window: 5
support_min_history: 3
support_reference_min: 0.75
support_drop_threshold: 0.20
```

原 `--mode human` 启动命令会默认开启两项新增规则；请沿用上次其他参数、选择可用 GPU，并使用新的 `--exp-suffix`。正式完整实验保持 `--max-events 0`（默认），正数仍仅用于 smoke，会限制实际审查数量。

消融开关：`--no-mask-change`、`--no-review-all-new`。可用 `--support-drop-threshold 0.25` 修改下降阈值；关闭两项即恢复原触发策略。`off` 模式仍不改 baseline、不积累支持度窗口；`audit` 只留证、不改决策。

主要代码：

- `conceptgraph/slam/association_gate.py`
- `conceptgraph/slam/rerun_realtime_mapping.py`
- `conceptgraph/hydra_configs/rerun_realtime_mapping.yaml`
- `scripts/run_blocking_association_gate.py`
- `tests/test_blocking_gate_triggers.py`

结果子目录 `blocking_association_gate/` 中：

- `human_review.html`：原来的统一实时审查页面，仍只更新这一页。
- `events.jsonl` / `events/*/decision.json`：单次审查、触发 reasons 和冻结支持度信息。
- `spatial_support.jsonl`：所有 baseline ATTACH 的当前支持度、历史帧/值、中位数、降幅、是否用于更新窗口；不显示给人工。
- `iou_prefilter.jsonl`：分数触发被去重取消时，区分整事件取消或被几何触发保留。
- `summary.json`：各触发原因计数、处理/改判/失败/上限跳过等统计。

## Smoke 内容与原始产物

新增11项：支持度边界、去重剩一个候选仍审查与 UID 重排、双条件只审一次、同帧历史隔离及 UID 消失、所有 NEW 不漏不重、实际 mapper 空图初始化分支遵守 DISCARD、关闭开关恢复旧行为、无效空间输入拒绝、启动器参数接线、上限跳过不污染窗口、零候选 schema。

测试使用合成 CPU 输入；运行实际 gate/证据渲染及 mapper 初始化分支，外部日志用 mock，启动器的 subprocess 被 mock，因此没有正式启动建图。人工选项为测试预设，不是人工精度结果。既有12项回归也全部通过。

服务器留存：

- 新增 smoke 原始证据/逐项耗时：`/home/chenkejun/beauty/trigger_v2_smoke_artifacts_20260904/`，汇总 `validation.json`（11项总计约0.362秒，不含解释器导入时间）。
- 新增 smoke 日志：`/home/chenkejun/beauty/trigger_v2_smoke_20260904.log`
- 原有回归日志：`/home/chenkejun/beauty/trigger_v2_regression_20260904.log`
- 修改前已有未提交差异备份：`/home/chenkejun/beauty/pre_trigger_v2_20260904.diff`
- 修改前 gate 源码备份：`/home/chenkejun/beauty/association_gate_pre_trigger_v2_20260904.py`

没有测试失败；无精度提升或退化结论，因为本次按要求未跑完整实验。此前 `.hydra/` 未改动、未提交。

## 局限与下一步

0.20是引用建议的首轮探索值，不是数据校准后的最优阈值。本次只实现并验证逻辑，没有进行离线阈值扫描或已知错误链召回评测。

新增几何规则不保证所有混合都能发现：历史不足、历史本就不稳定、混合点仍贴近已有几何、预处理提前移除了异常点都可能漏报；新视角看到尚未建出的真实表面、位姿误差等也可能误报。对象合并后保留存活 UID 的旧窗口也存在参考分布改变的局限。

全量 NEW 审查会显著增加人工工作量，这是此次扩大覆盖的预期成本。若碎片选择 UNCERTAIN，仍会回退 baseline；希望舍弃当前观测应明确选择已有 DISCARD。

后续由用户从空地图完整在线运行，先比较新增触发数、人工改判/舍弃数及错误类型，再看最终指标。此轮同时新增几何和全 NEW 两项，若有收益仍需用两个独立开关区分贡献。
