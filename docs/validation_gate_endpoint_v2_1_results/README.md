# Final Endpoint Validation v2.1：冻结结果包

冻结日期：2026-08-21（Asia/Shanghai）

代码基准：`619c98cb4f5ed7665c8a974c34c179c2ea6cfd33`

这是一份可随 GitHub 分支保存的小型结果包。它包含最终标签、工作清单、指标、决策和后续队列，不包含 5069 张页面证据图、原始 ledger、模型权重、服务日志或 PID 文件。

## 结论

- R1：97/97；55 `CORRECT`、40 `WRONG`、2 `UNCLEAR`。
- 证据充分：95/97；可判案例中的错误率为 40/95 = 42.11%。
- 正式决策：`PROCEED_TO_EXPERT_TRACE`；40 个确认错误进入专家队列。
- R2：24/24；同一复核者的最终状态一致率 83.33%，Cohen's kappa 0.706，三字段完全一致率 79.17%。
- R1 的 10 个 `WRONG` 在 R2 全部仍为 `WRONG`；5 个任一字段分歧的案例单独进入待裁决队列。
- R2 是短间隔 intra-rater/test-retest，不是独立评审者 inter-rater reliability。
- `review_score` ROC AUC 为 0.420，当前复合分数不能解释为错误概率或有效排序。

## 目录

```text
labels/
  r1_worklist.jsonl
  labels_r1_frozen_20260821.jsonl
  r2_worklist.jsonl
  labels_r2_frozen_20260821.jsonl
metrics/
  incident_endpoint_metrics.json
  r2_repeatability.json
  metrics_by_*.csv
  endpoint_error_types.csv
expert/
  confirmed_endpoint_error_queue.jsonl
  r2_disagreement_queue.jsonl
manifests/
  incident_worklist_manifest.json
  review_evidence_manifest.json
  r2_selection_manifest.json
  r2_review_evidence_manifest.json
  evaluation_freeze_manifest_20260821.json
decision.md
r2_repeatability.md
```

## 如何理解

`confirmed_endpoint_error_queue.jsonl` 的 40 例已经由 R1 确认最终错误，可进入 expert causal trace。`r2_disagreement_queue.jsonl` 的 5 例只是两轮标签不一致，状态为 `PENDING_HUMAN_ADJUDICATION`，不能偷偷并入 40 个确认错误。

部分 JSONL 保留正式服务器上的绝对 `case_dir`，用于与冻结 evidence packets 对照；仓库本身不复制这些大体积证据资产。完整人类可读说明见上一级 `ALI_MY_EVIDENCE_AUDIT_METHOD_GUIDE.md` 第 35 节。
