# `ali-my-full` 分支内容清单

更新时间：2026-08-23（Asia/Shanghai）

本分支以 `exp/audit-validity-gate-v1@bff233f` 为统一基线，快进合入 `ali-my-VLM@4c6e376`，再加入 2026-08-22 完成验证的 evidence-backed revision kernel、关系边复现脚本、当前研究设计文档和紧凑评估产物。目标是提供一个可克隆、可审计、可继续开发的 `ali-my` 全量代码分支。

## 代码范围

- 统一证据留存、append-only version ledger、分层因果审计与 R1/R2 validity gate；
- VLM-only endpoint repair 与 repair-aware 评测；
- revision kernel：provenance/lineage index、受控 corruption、精确 observation materialization、causal tracing、dependency-local replay、V1–V9 verifier、shadow transaction 和 typed VLM constraints；
- revision-disabled parity、materialization fidelity、controlled repair、determinism、live corruption、relation closure 与 VLM constraint 的运行/评测脚本；
- 8 场景语义/ReplicaSSG 评测脚本，以及只读关系候选、推理、导出、审计、同步和严格 `ali-dev` room0 重映射脚本。

## 评估与文档

- `artifacts/ali_my_evaluation_20260821/`：早期协议核验与双场景指标；
- `artifacts/ali_my_paper_main_aligned_20260821/`：8 场景 paper-aligned 语义和 ReplicaSSG 结果；
- `artifacts/ali_my_vlm_repair_aware_20260821/`：VLM repair-aware 汇总；
- `artifacts/ali_dev_relations_20260822/`：0.99 阈值关系汇总、严格 room0 评估、完整性审计和 map parity；
- `docs/ALI_MY_REVISION_KERNEL_VALIDATION_SUMMARY_20260822.md`：revision kernel 的完整实验结论、指标与边界；
- `docs/ALI_DEV_RELATION_EDGES_EVALUATION_20260822.md`：8 场景关系结果；
- `docs/ALI_DEV_MAIN_REPRODUCTION_VS_PAPER_20260822.md`：论文、`main` 与派生结果的严格口径对比。

## 未进入 Git 的内容

以下内容体量大、含运行时缓存或应由脚本重新生成，因此未上传：Replica/ReplicaSSG 原始数据集、模型权重、最终 PCD/PKL 地图、约 841 MB 关系证据图片、VLM review packet 图片、临时压缩包、日志缓存、`__pycache__`、运行环境和任何密码/API key。

revision kernel 的完整服务器实验产物仍保留在：

```text
/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822
/home/chenkejun/beauty/conceptgraphs/experiments/revision_v0_20260822_office0
```

关系实验的完整候选、证据图、逐对响应和 sidecar 保留在：

```text
/home/chenkejun/beauty/conceptgraphs/experiments/ali-dev-relations/synced_8scene_gpt56sol_v1
```

这些大体量服务器产物不属于源码仓库；分支中的汇总、哈希、审计 JSON 和复现脚本用于确认其身份与重建流程。
