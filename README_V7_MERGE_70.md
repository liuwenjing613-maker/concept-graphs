# v7_merge_70（实际阈值90% / 60%）

基于 v7_merge fb05116，不含 v7_merge_2 补充候选入口。
最终 KEEP_SEPARATE 时，仅 max(A→B,B→A)>0.90 AND min(A→B,B→A)>0.60 直接批准合并。未满足保持不合并；普通VLM合并两票及其他规则保持。
方向可交换；最大值恰好90%或最小值恰好60%不触发；90%/100%满足条件。

```bash
cd /home/chenkejun/beauty/conceptgraphs/code/experiments/v7_merge_70
bash run_v7_merge_70.sh auto --scene room0 --gpu 4 \
  --vlm-urls http://127.0.0.1:11437 \
  --model qwen3.6:35b-a3b-mtp-q4_K_M \
  --detections-exp-suffix v7_merge_auto_20260909_110828_335106_detections
```

该命令固定已有基线的逐帧检测缓存，仍从frame0空地图建立新图；不加载旧最终地图。省略缓存参数则生成独立新缓存。API/GPU需按运行时占用指定，脚本不自动挑选。
默认输出：/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/v7_merge_70_auto_<时间>_<PID>。
支持 --dry-run 核对配置，不启动任务。

20个人工复核事件固定重放：挡住6次DIFFERENT，同时挡住7/8次SAME；6次UNCERTAIN不计对错。适合保守消融，尚无证据证明更优。
完整分析及逐例表：/home/chenkejun/beauty/v7_merge_70_analysis_20260909/summary.md 与 cases.md。
仅完成组件 smoke 与只读已有案例分析，未运行完整场景。
