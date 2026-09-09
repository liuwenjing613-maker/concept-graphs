# v7_merge_2：补充高包含对象对进入 VLM

基于 v7_merge fb05116。按用户引用对话最后建议，只补充候选入口，不叠加类别、CLIP 或复杂评分条件。

## 行为

每次原有周期/最终合并检查先照常执行，再检查仍存活对象中的无序对象对：
1. 本次原生合并规则未选中过这对；原候选无论 VLM 同意或拒绝，都不在同次检查中补送。
2. 当前完整点云的轴对齐空间范围相交。
3. 完整点云任一方向的最近邻包含率严格 >0.90，距离沿用 0.01m 默认值。

满足后记录 supplemental_trigger=true，进入已有节点质量 + 合并身份 VLM。
**补充候选绝不因包含率直接合并。** 模型不同、质量失败、接口未决回退都保持分开；两次连续 MERGE 才执行；两次拒绝锁定及选中历史更新解锁、同帧去重保留。
**原候选继续保持 v7_merge 的“不合并 + 包含率 >90% 直接批准”策略**，这是引用对话明确要求保留的对照边界。

最近邻仅查询原距离阈值内的点，最终仍按原阈值比较，保留等号边界；无抽样、无近似。
单次检查内缓存空间范围和 KD-tree；实际合并后按对象代次更新，跨检查不复用。地图更新仍同步执行，冻结证据和严格证据记录保持一致。

## 显式指定两个 API 与模型

```bash
cd /home/chenkejun/beauty/conceptgraphs/code/experiments/v7_merge_2
bash run_v7_merge_2.sh auto --scene room0 --gpu 4 \
  --vlm-urls http://127.0.0.1:11464 http://127.0.0.1:11463 \
  --vlm-models qwen3.6:35b-a3b-mtp-q4_K_M qwen3.6:35b-a3b-mtp-q4_K_M
```

两个模型名按顺序与 API 一一绑定，可分别改成不同模型，模型必须支持原有图片输入和结构化输出。
这些是 Ollama base URL（无需加 /api/chat）；它们作为并行池分担所有阶段，不按质量/身份分工。
仍最多三个 API；超时保持同一证据并切换可用接口，初次 + 三次重试，四次超时后按原策略回退。
每次 attempt 保存实际 API、实际请求模型名、返回结果和 H 绑定，不把切换后的调用记成初始模型。
--gpu 只选择检测/建图 GPU，服务 GPU 由各 API 自身配置。
不指定 --vlm-models 时，所有接口使用 --model；可以加 --dry-run 先检查路径和完整绑定。

默认 auto 不请求人工，human 保留原有未决事件人工策略。默认 end=2000、stride=5，从 frame 0 空地图开始；YOLO imgsz=1200、原图 mask、新检测缓存保持一致。

## 输出与验证

默认结果：
/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/<scene>/exps/v7_merge_2_auto_<时间>_<PID>

补充扫描及对象对：blocking_association_gate/vlm_instance_merge/supplemental_scans.jsonl、supplemental_triggers.jsonl。
每个补充审核 decision.json 有 supplemental_trigger、trigger_containment、原模型输出、票数及 execution。
扫描 pass_seconds_including_reviews 包含其间 VLM 耗时；原始日志和 attempt 另有请求耗时。
结果页 blocking_association_gate/index.html，复核页 blocking_association_gate/review/index.html。

服务器报告：/home/chenkejun/beauty/v7_merge_2_smoke_20260909/summary.md。
只读已有地图入口诊断 + 小型控制测试 + 两帧在线 smoke；不是准确率结论。旧地图只用于诊断，不用于在线初始化。
