# 实验 0 在线标注、分析与最小回放

当前机器可读标注和结果快照位于 `../../annotation_records/experiment0_manual_annotation_20260901/`。核心新增分析入口：

- `analyze_v2_large_annotations.py`：正式标注质量与错误率统计；
- `compile_room0_human_episodes.py`：合并 R1/R2 并生成错误 episode；
- `run_human_oracle_minimal_replay.py`：B0/B1/B2/B3 严格在线回放；
- `audit_oracle_create_partition.py`：用完整成员分区审计 CREATE 修复，避免探针过度乐观；
- `run_mixed_root_quarantine_replay.py`：frame138 混合 mask 隔离反事实（实验中，不是生产修复）。

当前完整阶段判断见 `EXPERIMENT0_ROOM0_LABEL_ORACLE_STAGE1_REVIEW_20260902_CN.md`。

先读 `PROTOCOL_CN.md`。当前只生成 room0 校准队列，不解封未见场景。

## 1. 生成 20+4 校准 worklist

```bash
python make_calibration_worklist.py \
  --event-records /home/chenkejun/beauty/conceptgraphs/results/experiments/voxel_association_final_20260831/dev/event_records.jsonl \
  --observation-gt /home/chenkejun/beauty/conceptgraphs/results/experiments/online_label_trigger_v1_20260831/dev/room0/observation_gt.jsonl \
  --associations <ROOM0_EVIDENCE>/associations.jsonl \
  --output <PACKET_ROOT>/private_calibration_worklist.jsonl
```

## 2. 构造证据包

```bash
/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python build_event_packets.py \
  --evidence-root <ROOM0_EVIDENCE> \
  --output-root <PACKET_ROOT> \
  --scene room0 \
  --worklist <PACKET_ROOT>/private_calibration_worklist.jsonl \
  --top-k 5 \
  --history-views 6
```

正式在线 run 改用 `--follow --sample-probability <冻结值>`，不传带自动 GT strata 的 calibration worklist。sidecar 只读 evidence，并仅解封已经闭合的前一帧。

## 3. 启动标注页

```bash
python serve_event_labels.py --packet-root <PACKET_ROOT> --host 127.0.0.1 --port 8767
```

本地建立隧道：

```bash
ssh -N -L 18767:127.0.0.1:8767 -p 64906 chenkejun@frp-van.com
```

浏览器打开 `http://127.0.0.1:18767/`。服务只绑定服务器 loopback；本地使用 18767 是为了避开已有工具可能占用的 8767。

## 4. 校准汇总

```bash
python summarize_annotations.py --packet-root <PACKET_ROOT> --output <PACKET_ROOT>/annotation_summary.json
```

在 20 个基础包和 4 个暗重复包完成前，汇总状态必须是 `ANNOTATION_INCOMPLETE`，不能据此报告自然错误率。
