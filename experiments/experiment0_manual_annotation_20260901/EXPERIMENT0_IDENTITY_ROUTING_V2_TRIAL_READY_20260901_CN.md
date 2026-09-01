# 实验 0：ATTACH/NEW 身份路由 v2 试标阶段总结

日期：2026-09-01  
阶段状态：`READY_FOR_SCHEMA_TRIAL`，尚未进入正式发生率统计，也未解封 room2。

## 1. 本阶段完成内容

1. 只读审计 room0 的严格在线 B0 证据账本；地图从 frame 0 空状态运行，未修改模型、阈值、mask、stride、检测结果或 canonical map。
2. 将实验单位从 ATTACH-only false attach 修正为统一动作集合：

   ```text
   a_t ∈ {ATTACH(o_1), ..., ATTACH(o_n), NEW}
   ```

3. 实现五类确定性路由标签：`CORRECT_ATTACH`、`WRONG_ATTACH_EXISTING`、`SHOULD_HAVE_BEEN_NEW`、`WRONG_NEW_FALSE_SPLIT`、`CORRECT_NEW`。
4. 实现 mapper 盲标、动作揭示、完整 `t^-` 地图状态、身份证据状态和目标预污染分离。
5. 为旧 24 条 v1 标签生成不可覆盖的迁移/仲裁 overlay；原始标签和哈希保持不变。
6. 生成 12 个基础案例和 3 个暗重复的 room0 v2 schema trial，并启动 loopback 标注服务。

## 2. 在线证据完整性

证据根目录：

`/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/online_label_trigger_v1_room0_dev_pcd/evidence`

- 处理帧：400。
- 保留 observation：7,507。
- `MERGE_TO_OBJECT / ATTACH`：7,409。
- `CREATE_OBJECT / NEW`：98。
- object version：12,142。
- 后处理 object merge：26。
- 缺失引用：0。
- 重复 membership：0。
- logging error：0。
- NEW 事件同样保存事件前 object UID/version、完整 similarity matrix、top candidates、特征和新节点版本。

结论：无需纯日志重跑，可以直接使用当前 B0 账本构造 v2 标注包。

## 3. 修正后的私有路由审计

使用校正 sidecar：当前 depth/pose 投影到 ReplicaSSG 标注网格，3 cm 最近邻门限。旧 Habitat sidecar 未使用。

高精度、可自动选例的事件数为 4,728：

| 私有选例分层 | 事件数 | root candidate 数 |
|---|---:|---:|
| `CORRECT_ATTACH` | 4,671 | 不适用 |
| `WRONG_ATTACH_EXISTING` | 4 | 1 |
| `SHOULD_HAVE_BEEN_NEW` | 5 | 5 |
| `CORRECT_NEW` | 41 | 不适用 |
| `WRONG_NEW_FALSE_SPLIT` | 7 | 6 |

保守未决：

- 当前 observation GT 不可靠：1,942。
- ATTACH 目标历史少于 3 个可靠 observation：103。
- ATTACH 目标历史少于 3 帧：9。
- ATTACH 目标历史身份混杂：691。
- 同实例只出现在短历史或混杂节点，不能确认其为合法干净目标：34。

这些 GT 分层只用于私下选择清晰案例，不能替代人工标签，也不能直接解释为自然发生率。

## 4. 发现并修正的旧评估问题

旧 evaluator 判断 NEW 是否错误时只检查 top-1 candidate：top-1 与 observation 同实例才记为 false split。这会漏掉“正确旧节点存在，但不是 top-1”的事件。

v2 改为检查事件时全部 `object_uids_before` 及其严格绑定 version：

- 旧严格结果只看到 2 个 NEW false split；
- v2 完整候选检查得到 7 个高置信 false split，属于 6 个独立 root candidate group。

同时，旧 merge evaluator 将“没有严格合法候选”直接解释为应该 NEW。v2 发现其中部分事件的同实例证据存在于短历史或混杂节点，因此不能证明节点不存在：

- 原先 25 个 merge-error/no-correct-candidate 中，只保留 5 个高置信 `SHOULD_HAVE_BEEN_NEW`；
- 其余保守标为未决，不把证据不足误写成 NEW 真值。

## 5. v2 试标包

基础案例 12 个：

- 五种路由单元各 2 个；
- `MIXED_MULTIPLE_INSTANCES` 排除例 1 个；
- `BACKGROUND_OR_FRAGMENT` 排除例 1 个。

暗重复 3 个：分别来自 `WRONG_ATTACH_EXISTING`、`SHOULD_HAVE_BEEN_NEW`、`WRONG_NEW_FALSE_SPLIT`。总计 15 次标注。

注意：`WRONG_ATTACH_EXISTING` 的 4 个事件只属于 1 个因果 group，试标中的两条分别用于测试 root/cascade 分离，不能当作两个独立 root 正例。

两条 `CORRECT_NEW` 来自首帧空 `t^-` 地图，候选列表为空是正确且必要的边界情况；页面只允许选择 `NONE_SHOWN/UNCERTAIN`。

## 6. 完整性与盲标隔离

服务器测试：13/13 通过。

试标包完整性：`PASS`，0 错误。

- 15/15 public/private case 哈希绑定通过。
- 所有展示资产 SHA-256 通过。
- 3/3 暗重复的 observation、候选顺序和资产完全一致。
- 所有私有 GT 合法候选均已进入展示证据，但其加入原因和 GT 身份不公开。
- public JSON 未泄露 GT、自动路由分层、原始 ATTACH/NEW 动作、目标 UID 或相似度分数。
- ATTACH 案例 10 个，NEW 案例 5 个。
- ATTACH 和 NEW 的揭示接口均已实测；未写入测试草稿或测试标签。
- 浏览器实际页面检查通过：普通 ATTACH 案例的 12 张当前/候选图片全部加载，前端 console error 为 0；首帧空图 NEW 案例不显示虚假候选，只保留 `NONE_SHOWN/UNCERTAIN`。

构建过程中首次发现两条首帧正确 NEW 因候选为空被旧假设拒绝。已修正为允许空 `t^-` 候选集，重新构建后 15/15 成功。

## 7. v1 数据处置

- 24/24 原始 v1 标签只读保留。
- 新建 24 条 overlay，未自动迁移任何 v2 路由标签。
- `calibration_016/024` 标记为同事件身份分歧，等待单独仲裁。
- `calibration_020` 仅在 overlay 中修正 observation quality 为 `BACKGROUND_OR_FRAGMENT`。
- `calibration_017–019` 只确认 observation quality；路由标签仍需 v2 证据。
- 旧自动分层和旧派生动作明确失效，禁止与 v2 统计混用。

## 8. 服务器产物

- v2 协议与代码：`/home/chenkejun/beauty/conceptgraphs/code/experiments/experiment0_manual_annotation_20260901/`
- 私有路由审计：`/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_manual_annotation_20260901/identity_routing_v2_audit_room0/`
- v2 试标包：`/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_manual_annotation_20260901/v2_schema_trial_room0/`
- v1→v2 overlay：`/home/chenkejun/beauty/conceptgraphs/results/experiments/experiment0_manual_annotation_20260901/v1_to_v2_overlay/`
- 服务：服务器 `127.0.0.1:8768`，本机 SSH tunnel 使用 `127.0.0.1:18768`。

## 9. 当前结论与下一步

当前可以开始 15 条 v2 schema trial 标注，但仍不能进行 Experiment 0 的发生率统计，也不能启动 room2。

试标完成后应按字段分别检查：observation quality、身份证据状态、合法目标集合、五类路由动作和暗重复一致性；任何核心路由分歧先仲裁。若定义和证据通过，再冻结 v2 schema，并从空图在线运行 room2，同时保留概率 prevalence 队列和高召回 case-harvest 队列的严格分离。
