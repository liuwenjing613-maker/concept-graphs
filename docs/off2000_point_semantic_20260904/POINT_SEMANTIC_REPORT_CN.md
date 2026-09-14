# off2000：ali-dev 点级语义评估

## 正式结果

仅评估 room0 的 off2000 最终地图，没有重跑建图或调用 API。

| 指标 | % |
|---|---:|
| mIoU | 19.929851843087345 |
| mAcc / mRecall | 38.73306699705193 |
| mPrecision | 23.764315156047445 |
| mF1（ali-dev 原公式） | 24.747732742187715 |
| fwIoU | 40.68411421813301 |
| Point Accuracy | 51.704432663129985 |

IoU > 0.15 / 0.25 / 0.50 / 0.75 的类别数分别为 8 / 6 / 5 / 2。

## 输入、口径与验证

- 门控模式 off；manifest 与 gate summary 均为 completed。
- start=0, end=2000, stride=5；帧序列严格为源帧 0,5,...,1995，共400个处理帧。
- 地图72个对象、740,267个对象点；全部 CLIP 特征为1024维，坐标和特征无非有限值。
- 地图 SHA256：3527b6ade5fdb54024d719d51a112ce0e249883a6fa1c7cba1781bbafc82d9ee。
- 原生 ViT-H-14/laion2b_s32b_b79k；文本提示 an image of {class}；native_clip，无 Oracle 标签。
- n_exclude=6：排除 other/floor/wall/ceiling/door/window，保留23类、4,085,377个评测点。
- 共享 SLAM 点云7,754,935点，官方语义GT 1,556,890点；精确 CPU cKDTree k=1，无额外距离阈值。
- 评估器未修改，SHA256：4a7d49ff2c12ebf5a2a44159e91b3d340075173e004520d63da877598a429ab7。
- 成功退出码0；内部耗时36.670秒，外层wall time 37.66秒，峰值RSS 12,626,376 KB。
- 从52×52原始混淆矩阵独立重算所有主指标和阈值计数，与JSON误差小于1e-12。
- CSV所有行所有字段与JSON一致；单场景汇总矩阵与room0矩阵相同；输入地图哈希复核通过。
- 本次正式评估无失败重试。

## 失败类别与局限

12类IoU为0：basket、blanket、cabinet、candle、chair、pillar、plant-stand、plate、table、vase、wall-plug、rug。该结果仅代表room0，不能推断整个Replica或VLM/人工关联的因果效果。

本运行没有独立audit报告，不能声称evidence auditor通过。与原ali-dev相同，评估不设预测点距离阈值；SLAM到预测点的距离均值0.100577 m、P95 0.557884 m，远处未覆盖点也会继承最近对象标签。

## 通用评估步骤

以下命令均在服务器执行。无需额外激活conda，无需GPU或API Key。

1. 确认建图完成、最终 `pcd_<实验名>.pkl.gz` 存在；不要对仍在写入的地图执行正式评估。
2. 运行通用入口（已固定与本次相同的完整评估命令）：

```bash
ROOT=/home/chenkejun/beauty/conceptgraphs
EXP=off2000
TARGET="$ROOT/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/$EXP"
bash "$ROOT/beauty/evaluation_tools/eval_replica_point_semantic_ali_dev.sh" "$TARGET" room0 room_0
```

对同一数据目录下的其他room0实验只改EXP。off2000已评估过，因此默认命令会拒绝覆盖；直接查看已有结果即可。若明确需要复跑，最后加一个新的输出子目录名，例如 `point_semantic_recheck_01`。

3. 查看结果：

```bash
cat "$TARGET/point_semantic_ali_dev_nexclude6/semseg_results.csv"
```

CSV中mrecall就是mAcc，fmiou就是fwIoU，数值单位为百分比。两行room0/xxx_room0_only是同一个场景的明细与汇总，不能当成两次实验。

结果目录保留JSON（含逐类结果）、CSV、原始混淆矩阵NPZ和evaluation.log（含完整命令、退出状态与耗时）。

其他Replica场景需同时提供正确实验目录、场景名与GT场景名，例如office0对应office_0，并确保共享SLAM/GT存在。该脚本不自动生成SLAM或GT，也不适用于其他数据集协议。
