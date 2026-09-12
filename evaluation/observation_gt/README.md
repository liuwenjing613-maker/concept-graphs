# Observation–GT 与 VLM 关联决策评测

路径一协议版本：`observation_gt_association_v1_4_staged_merge_review_20260912`。

当前 VLM 评测由两个互补 evaluator 和一个静态审查页生成器组成：

1. `evaluation.observation_gt.evaluate_observation_gt`（路径一，v1.4）：评价被门控事件的 observation 质量、VLM staged intent 与关联动作；
2. `evaluation.gate_quality.evaluate_gate_quality`（路径二，v1）：以全部 association proposals 为总体，评价 gate 是否成功触发错误候选；
3. `evaluation.observation_gt.generate_gate_review`（审查入口）：读取路径一的冻结输出与原始 gate evidence，生成可独立托管的人工复核网页，不重新推理或改写评测结果。

该工具使用已有建图 evidence，将 processed observation mask 通过深度、内参和 camera-to-world 位姿反投影到三维，再匹配固定 GT mesh。它补充 `evaluation/unified` 的最终地图指标，用于回答 VLM 是否修正或引入 observation 级错误融合。

完整设计见 [PLAN.md](PLAN.md)。

## 评测口径

Observation 分为：

- `CLEAN`：稳定对应唯一 GT 实例，用于身份关联指标；
- `MIXED`：覆盖多个 GT 实例，用于评价 `DISCARD`；
- `UNSCORABLE`：有效深度或 GT 支持不足。

主评测以一次运行中的**全部 VLM gate events** 为唯一固定分母。baseline 和
VLM final 必须使用完全相同的事件集合；每个事件互斥落入 `SUCCESS`、
`FAILURE`、`AMBIGUOUS`、`UNSCORABLE` 或 `UNRESOLVED`，不再因为动作变得
可判定或不可判定而进出主分母。`SUCCESS` 同时包含 CLEAN observation 的正确
身份动作和 MIXED observation 的正确 `DISCARD`。

所有 detection class 都进入上述分类流程，不再设置 `IGNORED` 分支。检测类别为
`other` 但几何上落在有效实例 GT 上的 observation 可以被归为 `CLEAN/MIXED`。
所有在 manifest 的 `instance_id_to_semantic_id` 中声明的官方 GT 实例都参与
purity 计算，包括 wall、ceiling、floor 和 other；不再使用 manifest 的
`instance_classes` 白名单，也不假设 instance ID 必须大于等于 1000。Replica
原生 instance ID（包括 0）会被原样保留。

新 reference 直接来自 Replica 官方 `habitat/mesh_semantic.ply` 与
`habitat/info_semantic.json`。后者提供 instance ID 到 class name 的映射，因此
检测类别 `other` 不再触发跳过；官方对象中的 `other-leaf`、`undefined` 等兜底
名称也保留各自的原生 object ID，并按独立实例计分。只有 mesh face 的 object ID
未在 `info_semantic.json.objects` 声明时才记为 unlabeled（官方文件中这类 ID
不一定为负数）；它们参与 purity 分母，纯 unlabeled observation 归为
`UNSCORABLE / unlabeled_without_instance_identity`。

先从官方文件生成 reference（官方目录名带下划线，评测 scene 名保持现有命名）：

```bash
/root/miniconda3/envs/concept_graphs/bin/python \
  -m evaluation.observation_gt.build_replica_official_reference \
  --scene-dir /root/autodl-tmp/data/Replica_official_minimal/office_2 \
  --scene office2 \
  --out /root/autodl-tmp/evaluation_references/replica_official_native/office2
```

预测实例身份根据 `object_uids_before`、`candidate_object_version_uids` 和 `object_versions.jsonl` 中冻结的事件前成员快照，只用其中 `CLEAN` observations 投票，不使用最终地图。关联到同 GT 的非主预测实例记为 `CORRECT_FRAGMENT`：身份正确，但碎片未消除。对 `CLEAN + NEW`，只有本次 gate 展示候选中的 `UNKNOWN/MIXED_PREDICTION` 会触发 `AMBIGUOUS_NEW`；全图其他未知实例仅保留为诊断。若已有可靠同 GT 实例，仍记为 `DUPLICATE_CREATION`，并通过 `new_action_context` 区分同 GT 实例是否已进入展示候选。

V7 的 `staged_decision.kind=MERGE_REVIEW` 表示 VLM 认为 `same_aliases` 中的多个候选与当前 observation 属于同一实例，并已将候选对提交给实例合并投票。若两票证书尚未完成，执行器可能临时 `DISCARD` 当前 observation；该回退只记录在 `execution_result`，不再直接把 VLM 高层判断记为 `FALSE_DISCARD`。对 CLEAN observation，候选均可靠且与当前 GT 一致时记为 `CORRECT_MERGE_REVIEW`，存在可靠 GT 矛盾时记为 `WRONG_MERGE_REVIEW`，候选身份不足时记为 `AMBIGUOUS_MERGE_REVIEW`。MIXED/UNSCORABLE 仍保留 observation 动作结果，同时输出合并意图诊断。

## 运行

从仓库根目录运行：

```bash
/root/miniconda3/envs/concept_graphs/bin/python -m evaluation.observation_gt.evaluate_observation_gt \
  --run-dir /path/to/completed/run \
  --reference-dir /path/to/reference/room0 \
  --config evaluation/observation_gt/config.example.json \
  --out /path/to/fresh/output
```

快速 smoke test：

```bash
/root/miniconda3/envs/concept_graphs/bin/python -m evaluation.observation_gt.evaluate_observation_gt \
  --run-dir /path/to/completed/run \
  --reference-dir /path/to/reference/room0 \
  --config evaluation/observation_gt/config.example.json \
  --max-frames 5 \
  --out /tmp/observation_gt_smoke
```

`--max-frames` 的输出会显式标记为 `incomplete_smoke_run=true`，不得用于论文结果。输出目录已存在时工具拒绝覆盖，避免混用不同输入或阈值。

路径一完成后可运行路径二：

```bash
/root/miniconda3/envs/concept_graphs/bin/python -m evaluation.gate_quality.evaluate_gate_quality \
  --run-dir /path/to/completed/run \
  --observation-gt-dir /path/to/fresh/output \
  --out /path/to/fresh/gate-quality-output
```

也可以把静态网页生成器作为第三个正式入口单独执行：

```bash
/root/miniconda3/envs/concept_graphs/bin/python -m evaluation.observation_gt.generate_gate_review \
  --run-dir /path/to/completed/run \
  --metrics-dir /path/to/fresh/output \
  --out /path/to/fresh/review-page \
  --verify-images
```

第三个入口只负责把冻结证据与评测分类投影为 HTML，不改变路径一/路径二的分母、类别或指标。`--verify-images` 会在发布前核对所有证据图片的存在性与哈希。

## 主要输出

```text
protocol.json                  输入、代码哈希和冻结协议
observation_labels.jsonl       每个 observation 的 GT 分布与质量类别
prediction_identities.jsonl    每个 gate 事件前预测实例的因果 GT 身份
event_decisions.jsonl          baseline/final 动作与分类
metrics.json                   汇总指标
transition_matrix.csv          baseline → VLM final 转换矩阵
diagnostics/                   三类 observation 的 RGB mask overlay
```

`metrics.json` 中主要字段：

- `primary_denominator`：主分母名称和全部 gate event 数；
- `baseline/final.primary_event_metrics`：统一全事件分母上的成功、失败、模糊、不可评分和未解析数量/比例；
- `baseline/final.scorable_event_metrics`：固定 CLEAN+MIXED cohort 的成功率、失败率、ambiguity 和 resolved coverage；
- `baseline/final.clean_identity_metrics`：全部 CLEAN gate events 上的身份成功、错误和 ambiguity；
- `baseline/final.mixed_discard_metrics`：全部 MIXED gate events 上的 discard recall 和漏丢弃率；
- `baseline/final.discard_classifier_metrics`：把 DISCARD 当作 MIXED 分类器时的 TP/FP/FN/TN、precision、recall、F1 和二分类准确率；
- `baseline/final.association_diagnostics`：wrong fusion、fragment attachment 和 duplicate creation 的条件诊断；
- `vlm_transitions.net_vlm_gain_over_all_events`：非成功变成功的事件数减去成功变非成功的事件数，再除以全部 gate events；
- `vlm_transitions.new_failure_count`：原本不是明确失败、VLM 后变为明确失败的数量，包含 AMBIGUOUS → FAILURE。

## 测试

```bash
/root/miniconda3/envs/concept_graphs/bin/python -m unittest -v \
  evaluation.observation_gt.test_observation_labeler \
  evaluation.observation_gt.test_association_metrics
```

## 解释限制

- 本工具评价已有检测与关联事件，不等价于最终地图质量；两组指标需并列报告。
- `AMBIGUOUS_TARGET` 表示被选择目标身份不可靠；`AMBIGUOUS_NEW` 仅表示本次展示候选中存在身份不可靠的对象，无法排除其与当前 observation 同实例。它们进入统一主分母并作为独立 outcome 报告，但不强行计作成功或失败。全图无关的未知实例不会再使所有后续 NEW 退化为 ambiguous。
- 语义类别本身不会导致 observation 被跳过；是否可评分完全由实例 GT 支持、深度和 purity 条件决定。
- VLM gate 的 `decision_source`、规则覆盖和 action 一致性审计保存在逐事件结果中；只有通过审计的完整运行才能用于论文结论。

## 低内存核对 MIXED / UNSCORABLE 掩码

`review_masks` 会在浏览器中加载原始 RGB，以高亮绿色外接框和绿色轮廓标示 processed mask，同时保留 RGB 细节。页面提供四组数据：GT 为 `MIXED` 但 VLM 质量阶段判为 `USABLE`、VLM 错误丢弃的 `CLEAN`、VLM 正确丢弃的 `MIXED`，以及完整的 `MIXED / UNSCORABLE`；每页固定显示 10 张。第一组还显示每个样本的门控触发原因、固定阈值、VLM 质量理由和最终选择。可按状态、判定原因、帧、类别或 observation UID 筛选，点击图像可在局部裁剪和完整 RGB 帧之间切换。

它不会预生成或保存全量可视化图片。浏览器只懒加载当前页，服务器最多同时读取两个 RGB/mask 并在内存中编码 WebP，20 秒断开失效连接；元数据请求不会被单张图片阻塞。因此额外磁盘占用近似为零，峰值内存与总 observation 数无关。

服务器上运行：

```bash
/root/miniconda3/envs/concept_graphs/bin/python -m evaluation.observation_gt.review_masks \
  --run-dir /root/autodl-tmp/data/Replica/room0/exps/v7_CLIP_auto_room0_clip7525_qwen5090_20260910_125612 \
  --bind 127.0.0.1 --port 8765
```

本地建立 SSH 端口转发后，打开 `http://127.0.0.1:8765`：

```powershell
ssh -N -L 8765:127.0.0.1:8765 -F C:\Users\darren\.ssh\config ali-my-new7
```

只做数据与两张样图的完整性检查、不启动网页：

```bash
/root/miniconda3/envs/concept_graphs/bin/python -m evaluation.observation_gt.review_masks \
  --run-dir /path/to/completed/run --check
```
