# v7_merge：不合并 + 包含率超过 90% 直接合并

基于 v7_VLMsplit_image fbd00e0，继承服务器现有输入/输出目录分离更新。

## 唯一算法变化

对最终 KEEP_SEPARATE 的对象对，计算当前在线地图的完整点云双向最近邻包含率。
距离沿用原设置（默认 0.01 米）；任一方向严格大于 0.90，直接批准本次合并，不等待两次 MERGE，也不为这条规则请求人工。
恰好 90% 不触发；不是 mask IoU，不使用绘图采样或未来帧。

这条规则沿用原包含率检查的触发范围：身份 DIFFERENT，以及接口失败/节点污染等最终回退 KEEP_SEPARATE。
已两次拒绝锁定的对象对也检查当前包含率；满足时独立覆盖锁定，未满足仍维持原历史更新解锁规则。
普通 VLM MERGE 仍须两次连续确认。多候选组仍须所有两两关系获批，才整体执行，不能靠传递关系合并。

原始 identity_output、fallback_reason、original_choice 与 decision_source=containment_gt90 分别保留。
containment.json 记录双向比例；containment_points.npz 保存触发时完整点云及 SHA256；实际执行后 execution=MERGED。
几何批准不伪造两次 VLM 票数。

## 运行

```bash
cd /home/chenkejun/beauty/conceptgraphs/code/experiments/v7_merge
bash run_v7_merge.sh auto --scene room0 --gpu 1
```

可改 human；其他未决事件的人工策略保持原样，auto 不要求人工。
默认从空地图、frame 0 开始，end=2000、stride=5，使用独立新检测缓存。
YOLO imgsz=1200、三接口并行和超时原请求加三次重试保持一致。
当前默认接口为 11464、11463、11436（11435 于验证时未启动，换成同模型可用接口）。
`--gpu` 选择检测/建图 GPU；VLM GPU 仍由各接口服务自身配置。
`--dataset-root` 指原图目录，`--output-root` 指实验输出目录。
可加 `--dry-run` 只核对命令。

默认结果目录：
`/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/<scene>/exps/v7_merge_auto_<时间>_<PID>`。

## 验证

仅 smoke，不代表准确率提升。包含率高也可能是被污染节点包含另一物体，本版本用于用户要求的直接合并实验，保留原始结论供比较。
服务器验证报告：`/home/chenkejun/beauty/v7_merge_smoke_20260909/summary.md`。
