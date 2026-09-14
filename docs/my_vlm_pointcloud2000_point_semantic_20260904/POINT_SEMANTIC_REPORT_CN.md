# my_vlm_pointcloud2000 点级语义评估（ali-dev 口径）

## 结论

`my_vlm_pointcloud2000` 已按 ali-dev 原生点级语义协议完成 room0 单结果评估。主结果为：mIoU 19.519%，mAcc 37.105%，fwIoU 48.176%，点准确率 60.694%。这里的 mAcc 与评估代码中的 mRecall 相同，表示各现存类别 recall 的宏平均。

本次只读取并评估这一张已完成地图，没有重跑建图、没有评估其他实验、没有调用 VLM/API。

## 主指标（%）

| 指标 | 结果 |
|---|---:|
| mIoU | 19.518813 |
| mAcc / mRecall | 37.104698 |
| mPrecision | 22.346352 |
| mF1 | 23.840992 |
| fwIoU / F-mIoU | 48.175808 |
| Point Accuracy | 60.694252 |
| IoU > 0.15 的类别数 | 7 / 23 |
| IoU > 0.25 的类别数 | 6 / 23 |
| IoU > 0.50 的类别数 | 5 / 23 |
| IoU > 0.75 的类别数 | 2 / 23 |

## 评估口径

- 场景：Replica `room0`。
- 地图配置：`start=0, end=2000, stride=5`；在线证据包含 400 个处理帧，源帧严格覆盖 `0, 5, ..., 1995`。这是完整 2000 原始帧范围的 stride-5 在线运行，不是连续处理 2000 张图。
- 语义读出：地图中保存的原生 1024 维 CLIP 特征；模型 `ViT-H-14`，预训练权重 `laion2b_s32b_b79k`，提示为 `an image of {class}`。
- 类别：Replica 52 类中的场景现存类；主口径 `n_exclude=6`，排除 `other/floor/wall/ceiling/door/window`，最终评估 23 类。
- 投影：对象点云精确最近邻投影到共享 ali-dev stride-5 SLAM 点云，再精确最近邻投影到 Replica Semantic GT；CPU `scipy.spatial.cKDTree, k=1`。
- 距离策略：与 ali-dev 一致，不增加最近邻距离阈值。
- 指标公式：保持 `conceptgraph/scripts/eval_replica_semseg.py` 与 `conceptgraph.utils.eval.compute_metrics` 的现有兼容实现。
- 有效评测点：4,085,377；与既有 ali-dev room0 正式结果的固定评测分母一致。

## 输入与运行诊断

- 最终预测对象：57。
- 对象点云总点数：721,802；无非有限坐标。
- 对象 CLIP 特征：57/57 均为 1024 维；无非有限值。
- 共享 SLAM 点数：7,754,935。
- GT 点数：1,556,890。
- SLAM→预测对象点最近邻距离：均值 0.100593 m，P95 0.557892 m。
- SLAM→GT 最近邻距离：均值 0.005623 m，P95 0.011629 m。
- 评估器内部运行时间：38.724 s；外层 wall time 39.74 s；峰值 RSS 12,629,220 KB。

## 类别结果与失败分析

表现最好的类别为 `blinds` 98.111%、`picture` 77.201%、`sofa` 68.343%、`lamp` 55.308%、`stool` 51.496% IoU。`cushion` 为 49.006%，`book` 为 22.906%。

23 个类别中有 13 个 IoU 为 0，包括 `basket/blanket/cabinet/candle/chair/pillar/plant-stand/plate/pot/table/vase/wall-plug/rug`。主要点级混淆为：

- `rug → indoor-plant`：257,094 点，占 rug GT 的 34.32%。
- `rug → sofa`：185,006 点，占 rug GT 的 24.69%。
- `table → indoor-plant`：178,610 点，占 table GT 的 68.74%。
- `cabinet → plant-stand`：87,092 点，占 cabinet GT 的 99.55%。
- `blanket → cushion`：80,341 点，占 blanket GT 的 91.90%。
- `chair → sofa`：65,756 点，占 chair GT 的 53.27%。

点准确率与 fwIoU 明显高于 mIoU，是因为大类别贡献占主导，而许多小类或困难类完全没有被正确读出；因此不能只看 60.69% 的点准确率判断语义质量。当前主要失败同时包含类别读出错误和对象覆盖/最近邻外推影响，单凭这一项评估不能把误差全部归因于 VLM 关联。

## 严谨性检查

- 地图清单状态为 `MAP_COMPLETED_EVIDENCE_VALID`，严格证据审计通过。
- 400 个在线处理帧无缺失、无乱序，源帧序列精确等于 `0:5:2000`。
- 输入地图哈希与评估结果记录完全一致。
- `semseg_results.json`、`semseg_results.csv` 和 `semseg_conf_matrices.npz` 相互一致。
- 从保存的 52×52 原始混淆矩阵独立裁出 23 类后，重新计算 mIoU、mAcc、mPrecision、mF1、fwIoU、Point Accuracy 和四个 IoU 阈值计数，全部与正式 JSON 逐项一致（容差小于 `1e-12`）。
- 第一次启动因环境没有暴露 `gradslam` 源码路径、第二次因离线 CLIP cache 根目录给错而在读取地图前失败；两次均未产生指标。失败日志已保留，第三次使用现有本地模型权重成功完成，无网络下载。

## 局限

- 这是 room0 单场景结果，不能代表 Replica 多场景总体表现，也不提供场景间方差或置信区间。
- ali-dev 原协议对 SLAM→预测对象点不设置距离阈值；本次 P95 为 0.558 m，远离任一预测对象的点仍会继承最近对象标签。这保证与 ali-dev 可比，但会混合“未覆盖区域”和“语义分类”两类误差。
- 本评估使用地图中已有 CLIP 特征，只反映最终冻结地图的原生语义读出；不等同于对象实例级 AP，也不能单独定位每个 VLM 决策的因果贡献。

## 产物

- 正式指标：`my_vlm_pointcloud2000/point_semantic_ali_dev_nexclude6/semseg_results.json`
- 表格：`my_vlm_pointcloud2000/point_semantic_ali_dev_nexclude6/semseg_results.csv`
- 原始混淆矩阵：`my_vlm_pointcloud2000/point_semantic_ali_dev_nexclude6/semseg_conf_matrices.npz`
- 成功运行日志：`my_vlm_pointcloud2000/point_semantic_ali_dev_nexclude6/evaluation.log`
- 两次启动失败日志：同目录下 `evaluation_attempt1_import_failure.log`、`evaluation_attempt2_cache_path_failure.log`
- 机器可读复核清单：同目录下 `validation.json`
