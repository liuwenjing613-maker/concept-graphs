# 实验 0 在线关联标注：校准阶段实现总结

日期：2026-09-01  
阶段判定：`READY_FOR_HUMAN_CALIBRATION`，不是实验 0 科学结论，也未解封 room2。

## 1. 本阶段目标

在不修改 mapper 和 canonical map 的前提下，建立可审计的关联事件标注流程，先验证人是否能稳定区分：

- 当前 processed mask 是否可作为单一物理实例；
- 候选节点中哪些与当前 observation 是同一实例；
- mapper 所选节点在事件前是否干净；
- 自动推导的 `KEEP/REASSIGN/NEW/DEFER` 是否有足够证据。

## 2. 实现

服务器代码：

`/home/chenkejun/beauty/conceptgraphs/code/experiments/experiment0_manual_annotation_20260901/`

主要文件：

- `PROTOCOL_CN.md`：标注定义、抽样、root/cascade、在线捕获与统计规则；
- `build_event_packets.py`：只读 causal ledger；一帧延迟确认 frame 闭合；生成 2D/3D hash-bound packet；
- `make_calibration_worklist.py`：冻结 room0 的 20+4 私有校准 worklist；
- `serve_event_labels.py`：只绑定 loopback 的两阶段盲标页面；
- `label_logic.py`：校验并自动推导动作；
- `summarize_annotations.py`：证据充分率、暗重复一致性、标签/用时汇总；
- `test_experiment0.py`：核心动作推导单元测试。

标注结果根：

`/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_manual_annotation_20260901/calibration_room0/`

## 3. 输入与校准样本

仅复用已经看过的 room0 严格在线 PCD evidence run：

- run ID：`room0_20260831T111035Z_5c9d86fa`；
- mapper 完成帧：400/400，latest frame 399；
- evidence manifest SHA-256：`89bb5c2420d295521f3b1bbb2e36afd9a3d535d4bf4af86fee8b2d862a25d6397`；
- worklist：20 个基础包 + 4 个暗重复包；
- 私有构成：8 个自动 GT 疑似错误、8 个自动 GT 正确、4 个 mixed/低纯度排除例；这些 strata 不进入 public 页面。

校准 worklist SHA-256：

`7f8a99ba86e422fe64e6983c36459a0eb4cde007a1ace40bde68e87d3c7760de`

## 4. 验证结果

- 单元测试：`6 passed`；
- packet：`24/24` 成功，`0` failure；
- 产物：336 个文件，其中 288 张 JPG；
- public schema 只含 scene/case/event/frame、当前 observation、匿名候选代码、历史数量和展示资产哈希；
- public 文件静态检查未发现自动 GT、私有抽样组、selected target UID、object version UID 或候选分数；
- `/api/case` 在盲标状态没有 `reveal` 字段；
- 4 组暗重复的 `displayed_asset_sha256` 全部逐组一致；
- 服务只监听服务器 `127.0.0.1:8767`；远程 `/api/status` 返回 24 cases、0 labels、mapper complete；
- 未标注汇总正确返回 `ANNOTATION_INCOMPLETE`，没有提前计算自然发生率。

人工抽查了当前 mask 全图、候选历史 contact sheet 和三视图 PCD 对照：颜色、mask 边界和候选匿名代码可辨，历史图明确标出显示条数与总条数。

## 5. 已知限制

1. 当前阶段 A–E 只显示 top-5（强制包含 mapper target）。若正确节点在 top-5 外，页面不会诱导猜 `NEW`；必须选 `UNCHECKED`，事件先 `DEFER`。正式统计前需为确认错误的少量正例补一个完整 `t^-` map action-adjudication pass。
2. 节点 2D 历史最多显示 6 个代表视图，页面明确给出 `显示数/总数`；3D 对照也不是每条历史的完整人工逐帧检查。Replica 自动 instance GT 将作为隐藏的全历史一致性检查，人工仍可因证据不足选择 `UNCERTAIN`。
3. 当前 calibration 是按自动 GT 平衡抽样，不能计算自然发生率。
4. root/cascade 跨 post-process object merge 的 episode compiler 尚未在本阶段运行；完成标注后再只对人工确认正例编译。
5. 只有一名标注者时，4 个暗重复只能评估 intra-rater consistency，不能替代独立第二标注者。

## 6. 下一步判据

用户先完成 24 个校准包。只有同时满足：

- 20 个基础包完整；
- evidence sufficient (`YES`) 至少 80%；
- 4 个暗重复的 eligible 与 action agreement 均至少 90%；

才冻结 annotation schema 并从空图运行 room2。若不满足，先根据 `PARTIAL/NO` 备注补证据页面；不查看 room2 GT、不调实验 0 结论门槛。

room2 正式 pilot 将使用与风险分数无关的 event-UID hash Bernoulli prevalence sample；case-harvest 队列单独报告，绝不用于自然错误发生率。

