# `ali-my-VLM`：纯 VLM 终点审计与隔离修复 v1

## 1. 方法边界

这是一个与原 `ali-my` 隔离的方法版本。它复用冻结且可追溯的 final-endpoint evidence packet，但不复用原 checker 的判断，也不原地修改原地图。

97 指两个场景中去重后的可复核 endpoint，不是 97 个已确认错误。冻结 R1 标签为 55 `CORRECT`、40 `WRONG`、2 `UNCLEAR`；这些标签在推理阶段不可见，只允许在结果全部落盘后由独立评估脚本读取。

```text
冻结且哈希绑定的 endpoint evidence
        ↓
terra：逐项反证审计（真实性、语义、几何、成员、误合并/误拆分）
        ↓
sol：重新看图并作终点状态、错误类型与最小修复裁决
        ↓（仅拟修改项）
5.5：独立安全复核
        ↓
仅高置信 RELABEL / DELETE / MERGE_WITH 写入 derived map
```

三个阶段全部是 VLM。代码只负责证据哈希核验、描述性几何整理、JSON 类型约束、置信度门和派生副本写入，不用原 checker 规则替 VLM 作错误判断。

## 2. 输入证据与防泄漏

每例最多发送 10 张、且逐张通过 `review_evidence.json` 中 SHA-256 核验的图片：

- 两张 exact final-map geometry；
- 最多两张 trigger observation panel；
- 一张多视角 timeline；
- 其余名额给 endpoint 的 RGB context 与 exact processed mask。

数值摘要只包含 final object 的保存标签、成员/帧/点数、包围盒、观测标签直方图，以及 endpoint 到 context 的距离、AABB 交并和支持比例。它们是描述性证据，不是错误判定器。

推理入口只从工作表投影 `scene_id`、`case_uid`、`incident_uid`、`representative_finding_uid`、`case_dir`。packet 中若出现人工 `final_state`、`final_error_type`、`reviewer_id` 等字段会直接阻断；checker 名、stage、subtype、review score 也不会进入 prompt。

## 3. 为什么采用三种模型

同一盲例的 API 实测表明 `gpt-5.5`、`gpt-5.6-sol`、`gpt-5.6-terra` 都具备多图识别能力。首轮单一 `terra` 终判在 6 个按错误族覆盖的开发例上只得到 1/5 的错误召回，主要问题是把异常解释成合理对象。

加入强制反证审计后，`terra` 能发现风险，但仍会在终判阶段推翻自己的审计。把 `sol` 用作独立终判后，同一 6 例达到 6/6 状态命中，错误检测 precision/recall/F1 均为 1.0，错误类型 3/5 命中。这个小样本只用于选模型与提示，不代表全量性能。

新增的 12 例分层开发集暴露了真实边界：状态准确率 7/12，排除 `UNCLEAR` 后错误检测 precision/recall/F1 均为 0.857；错误类型在双方都判错时为 4/6。随后针对性修复了“完整 endpoint 被旁边小坏碎片连坐”的假阳性和兼容语义子类过度不确定问题。最终锁版仍保留两个已知难点：

- 一个 R1/R2 都标为 `FALSE_SPLIT` 的长柜体，`terra` 与 `sol` 在完整 10 图证据上仍认为 final endpoint 已覆盖完整柜体；
- 一个 R1 认为应是插头的 `speaker` endpoint，锁版降为 `UNCLEAR`，没有猜测性重标。

因此全量 97 的结果必须作为正式测量，不能把小样本结果外推成结论；提示在全量运行前锁定，不根据全量标签反向调参。

## 4. 修复策略

| 动作 | 自动执行条件 | v1 行为 |
|---|---|---|
| `RELABEL` | 语义错误；终判 ≥0.85；复核 ≥0.80 | 只改派生对象的稳定标签 |
| `DELETE` | 明确伪对象；终判 ≥0.97；复核 ≥0.95 | 只从派生对象列表移除 |
| `MERGE_WITH` | 明确误拆分；终判 ≥0.95；复核 ≥0.90；alias 存在 | 合并派生点云、观测成员和 membership |
| `SPLIT_OBJECT` | 误合并 | 记录计划，等待全成员 VLM 分组 |
| `REASSIGN_MEMBERS` | 成员错误 | 记录计划，等待全成员 VLM 分组 |
| `TRIM_GEOMETRY` | 几何损坏 | 记录计划，等待全成员 VLM 分组 |

代表视图不足以安全决定所有 observation 的归属，所以结构性动作不会伪装成已修复。首次 VLM JSON 若不符合类型约束，允许在相同证据上追加一次格式纠正；两次原始响应都会留档，仍需通过同一安全校验。

## 5. 运行

API key 仅放在当前 shell 环境变量，输入不回显：

```bash
read -rsp 'VLM API key: ' VLM_API_KEY
export VLM_API_KEY
```

锁版默认角色已经写入参数默认值，正式跑 97 个 endpoint：

```bash
python scripts/run_vlm_endpoint_repair.py \
  --validation-root /home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1 \
  --output-root /home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1
```

推理结束后才载入冻结标签：

```bash
python scripts/evaluate_vlm_endpoint_repair.py \
  --run-root /home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1 \
  --labels /home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1/labels/labels_r1_frozen_20260821.jsonl \
  --output /home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1/evaluation.json
```

只将双重通过的动作应用到新地图：

```bash
python scripts/apply_vlm_repair_overlay.py \
  --validation-root /home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1 \
  --run-root /home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/full_97_locked_v1 \
  --output-root /home/chenkejun/beauty/conceptgraphs/experiments/ali-my-VLM/derived_maps_locked_v1
```

## 6. 结果解释

- 97 例只覆盖原 screener 纳入的 endpoint，不是全地图 recall census。
- `derived map` 不等于 repair verified；仍需对象级 diff、可视复核或下游 ReplicaSSG 评测。
- 第三方 API 只接收每例必要图片和无人工标签摘要；密钥、人工答案和完整数据目录不会发送。
- 原冻结 pickle、evidence 与原 `ali-my` 工作树始终保持只读。

## 7. 两场景修复感知评测（2026-08-22）

在 `room0 + office0` 的既有 Replica GT 上完成评测，没有引入新增人工标注。主口径
`n_exclude=6` 下，直接读取地图保存标签的 `map_class_name` 轨道表现为：mIoU
`32.49% → 45.04%`（`+12.55 pp`）、mF1 `+14.06 pp`、fwIoU `+19.18 pp`、
点准确率 `+16.10 pp`。ReplicaSSG 对象分类中，标签轨道 R@1 `+2.70 pp`、
mR@1 `+13.87 pp`；74 个 GT 排名改善 5、恶化 1、不变 68。

与之相对，原生 `native_clip_ft` 轨道在主口径上完全不变；GT 几何覆盖仍为
`52.70%`，碎片化 excess 仍为 12。预测对象由 101 减至 99，但闭集几何有效预测数
没有增加。因此 v1 的可靠结论是：**VLM 修复明显改善了保存标签的下游可用性，但没有
改善原生视觉表征、几何覆盖或碎片化**。其中 `pot` IoU 下降 `80.22 pp`，需结合
标签到闭集词表的映射差异单独复核，不能只按总均值判定个别修复正确性。

可追溯产物：

- [精简可读报告](../artifacts/ali_my_vlm_repair_aware_20260821/repair_aware_summary.md)
- [完整机器可读结果](../artifacts/ali_my_vlm_repair_aware_20260821/repair_aware_summary.json)
