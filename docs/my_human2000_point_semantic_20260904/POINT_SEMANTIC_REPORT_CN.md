# my_human2000 点级语义评估

## 结果

仅对 `my_human2000` 的 Replica room0 完整在线地图进行评估，沿用 ali-dev 原生点语义协议（`n_exclude=6`）。

| 指标 | 结果 |
|---|---:|
| mIoU | 19.894672% |
| mAcc / mRecall | 37.533832% |
| mPrecision | 22.475420% |
| mF1 | 24.167004% |
| fwIoU | 49.189219% |
| Point Accuracy | 61.623444% |
| IoU > 0.15 / 0.25 / 0.50 / 0.75 | 7 / 6 / 6 / 2 类 |

这里的 mAcc 等于评估器输出的 mRecall，即23个有效类别 recall 的宏平均。

## 口径与完整性

- `start=0, end=2000, stride=5`，400个处理帧严格对应源帧 `0,5,...,1995`。
- 人工门控完成85个事件：A 49、B 8、NEW 6、UNCERTAIN 22；25个事件改变了 ali-dev 原决定。
- 最终64个对象、729,935个对象点；全部对象特征为有效的1024维 ViT-H-14 CLIP 特征。
- 排除 `other/floor/wall/ceiling/door/window`，有效评测类别23个、点4,085,377个。
- 使用共享 ali-dev stride-5 SLAM 点云、Replica Semantic GT，以及 CPU `cKDTree k=1` 精确最近邻；不增加距离阈值。
- SLAM→预测点距离均值0.100589 m、P95 0.557884 m；SLAM→GT均值0.005623 m、P95 0.011629 m。

## 复核与局限

- 正式评估退出码0；内部运行35.305秒，外层wall time 36.29秒。
- 52×52混淆矩阵独立重算的全部指标与JSON一致，容差小于 `1e-12`；JSON/CSV/NPZ、地图哈希和帧序列检查通过。
- 23类中13类IoU为0。较好类别为 blinds 98.116%、picture 76.401%、sofa 71.385%、cushion 56.414%、lamp 53.883%、stool 51.496%。因此Point Accuracy较高不能替代mIoU/mAcc判断。
- 结果目录没有生成独立 `audit/` 报告；本次确认了manifest完成、门控summary完成、帧序列和地图数值有效，但不能声称额外的evidence auditor已经通过。
- 这是room0单场景结果，不代表Replica多场景总体表现；原协议没有预测点距离阈值，未覆盖区域也会继承最近对象标签。

## 产物

正式结果位于 `my_human2000/point_semantic_ali_dev_nexclude6/`：`semseg_results.json`、`semseg_results.csv`、`semseg_conf_matrices.npz`、`evaluation.log` 和 `validation.json`。
