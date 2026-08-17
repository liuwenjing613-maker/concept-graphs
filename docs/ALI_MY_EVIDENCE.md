# ali-my：第一阶段统一证据留存

`ali-my` 是从官方 `ali-dev@72f5962` 独立出来的个人开发 worktree：

```text
code/official/main      # 严格 main 基线，不修改
code/official/ali-dev   # 官方 ali-dev 参考，不修改
code/official/ali-my    # 本模块的唯一代码工作区
```

当前实现对应设计文档的第一阶段“统一证据留存”。它是旁路 sidecar，不参与
检测、过滤、去噪、关联、融合或边判断；证据写入失败会被记录并自动旁路，不能改变
建图结果。

## 运行入口

在服务器上进入 worktree 根目录后运行：

```bash
cd /home/chenkejun/beauty/conceptgraphs/code/official/ali-my
./scripts/run_replica_evidence_smoke.sh \
  ali_my_evidence_$(date -u +%Y%m%dT%H%M%SZ) true
```

脚本默认使用 `envs/cg-ali` 的依赖层，但通过 `PYTHONPATH` 明确绑定到
`code/official/ali-my`。参数为：`<exp_suffix> <save_evidence>`；建议每次正式运行使用
唯一的 `exp_suffix`，以保留完整历史。重复使用同一个 suffix 时，JSONL 和证据目录中的
NPZ 会被重置，这是为了避免不同 run 的二进制证据混在一起。

手工运行时，入口是：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/chenkejun/beauty/conceptgraphs/code/official/ali-my \
/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python \
conceptgraph/slam/rerun_realtime_mapping.py \
  dataset_root=/home/chenkejun/beauty/conceptgraphs/data/Replica \
  dataset_config=/home/chenkejun/beauty/conceptgraphs/code/official/ali-my/conceptgraph/dataset/dataconfigs/replica/replica.yaml \
  scene_id=room0 start=0 end=40 stride=20 \
  detections_exp_suffix=smoke_detections_stride20 \
  exp_suffix=ali_my_evidence_smoke \
  make_edges=false save_evidence=true
```

## 证据目录

每个 mapping experiment 在 `data/<Dataset>/<scene>/exps/<exp_suffix>/evidence/` 下生成：

- `manifest.json`：run、分支、commit、配置、模型/提示版本和结束状态。
- `frames.jsonl`：每帧 UID、源图像/深度、位姿、内参、处理或跳过原因。
- `observations.jsonl`：原始检测永久 `obs_uid`、过滤状态、原因、3D bbox、点云引用。
- `similarities/frame_*.npz`：空间、视觉、聚合三类完整候选矩阵及 UID 轴。
- `associations.jsonl`：在线关联/创建决策、阈值、Top-K 候选、margin 和矩阵引用。
- `observation_pcd/*.npz`：每个保留观测的独立点云（可配置采样上限）。
- `mapping_events.jsonl`：对象创建、关联、去噪、过滤、合并和边变化。
- `vlm_events.jsonl`：VLM 原始响应、耗时、状态、解析输出和 prompt fingerprint。
- `final_membership.json`：最终对象到观测 UID 的去重成员映射和边 UID。
- `evidence_summary.json`：帧数、检测数、合并数、缺引用数、重复成员数和日志错误数。

建议回滚/排错时至少保留 `manifest.json`、`mapping_events.jsonl`、
`associations.jsonl`、`observations.jsonl` 和对应的 NPZ；它们可以把一个最终对象追溯到
帧、原始检测、过滤结果、候选分数和后处理事件。

## 配置开关

```yaml
save_evidence: true
evidence_top_k: 3
evidence_save_observation_pcd: true
evidence_observation_pcd_max_points: 10000
```

`save_evidence=false` 时不创建证据文件、不初始化 OpenAI 代理，也不改变原有 mapping
输出。`make_edges=false` 时不会初始化客户端或发起 VLM 请求，因此 smoke test 不需要
API key；此时 `vlm_events.jsonl` 为空是预期行为。

## 已验证的服务器结果（2026-08-17）

Replica `room0` 两帧 GPU smoke（`start=0,end=40,stride=20`）得到：

```text
frames=2, raw_detections=70, kept_observations=47, rejected=23
create=25, associate=22, object_merges=7, final_objects=19
missing_reference_count=0, logging_error_count=0
similarities=2 NPZ, observation_pcd=47 NPZ
```

最终成员引用为 47 个且全部唯一；7 次 `OBJECT_MERGE` 均包含实际使用的 overlap、
visual、text 分数和阈值。证据开关对照中，`save_evidence=true/false` 的对象 JSON 与边
JSON SHA-256 完全一致：

```text
object_json = ba5c6c7f2ad69c6ff87b785e9fb00f3934e4a9f286d90a4a6aad25bee3593e9d
edge_json   = 44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a
```

## 边界

本阶段只负责“证据可追溯”和失败旁路，不改变 ali-dev 的算法阈值或语义策略。当前
`make_edges=false` 的 smoke 没有覆盖真实 VLM 返回；开启 `make_edges=true` 的运行应将
`vlm_events.jsonl` 与 `manifest.json` 一并归档，并确保 API key/模型版本已记录。
