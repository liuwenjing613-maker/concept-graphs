# 实例 PQ 与几何覆盖率：独立离线评估

实现日期：2026-09-05。参考 `/home/chenkejun/beauty/conceptgraphs/scripts/evaluate.sh`；原文件未修改。

## 直接使用

```bash
bash /home/chenkejun/beauty/conceptgraphs/scripts/evaluate_instance_pq.sh 实验文件夹名
bash /home/chenkejun/beauty/conceptgraphs/scripts/evaluate_geometry_coverage.sh 实验文件夹名
```

例如把“实验文件夹名”替换为 `v4_human2000`、`my_human2000` 或新实验名称。不传参数时，两脚本与参考 evaluate.sh 一样默认 `off2000_10`。也可修改各脚本的 EXP 默认值。

输入地图自动定位到：

`/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/<EXP>/pcd_<EXP>.pkl.gz`

每次在该实验目录下创建新的带时间戳输出目录，不覆盖已有地图或指标。无需重建地图、重新人工标注、GPU、VLM/API 或 CLIP。

## 代码路径

Python 实现和同版本 shell 入口位于：

`/home/chenkejun/beauty/conceptgraphs/code/experiments/ali-dev-blocking-gate-v1-20260903/scripts/`

- `eval_replica_instance_pq.py`：实例 PQ/RQ/SQ。
- `eval_replica_geometry_coverage.py`：几何覆盖率。
- `replica_eval_common.py`：共同点集、原语义 GT 变换、地图读取和精确最近邻。
- `evaluate_instance_pq.sh`、`evaluate_geometry_coverage.sh`：可直接运行的入口。

根目录 scripts/ 下也放置两个 shell 入口，便于与原 evaluate.sh 并列使用。

## 固定评估口径

两个指标使用同一份 ali-dev 原生 SLAM 点集，不根据预测地图覆盖情况删减分母。语义 GT 采用原评估器的 first-pose 世界坐标变换、精确 k=1 最近邻及 n_exclude=6（other/floor/wall/ceiling/door/window）。room0 共 7,754,935 个参考点、4,085,377 个前景评估点。

默认参考点集对应 0–2000、stride=5 的固定 SLAM 重建。地图必须记录场景及完整起止范围；部分场景/错误场景会报错。地图 stride 可以不同，但必须如实记录，不能把不同采样预算直接归因于新模块。本次参考脚本默认 `off2000_10` 实际 stride=10，这与固定评估点集 stride=5 的差异已写入 results.json。

### 1. 几何覆盖率

对全部固定前景点计算到预测地图任意对象点的最近距离，统计 `distance <= d`，默认 d=0.025/0.05/0.10 米。未覆盖点保留在分母，空地图覆盖率为 0。

这衡量可见参考表面的覆盖，不是表面积均匀采样的 mesh recall，也不衡量几何精度或实例身份。错误对象只要足够接近也可能贡献覆盖，需结合 PQ 判断。

输出：`coverage.csv`、`coverage_by_class.csv`、`point_distances.npz`、`results.json`、`status.json`、`evaluation.log`。CSV/JSON 的 coverage 为 [0,1] 小数，终端显示百分比。逐类结果保留小物体/特定类别的退化。

### 2. 类别无关实例 PQ

GT 来自服务器已有的真实实例标注：

`/data/chenkejun/ReplicaSSG/Replica/data/room_0/labels.instances.annotated.v2.ply`

读取 `objectId` 并核对 `files/objects.json`，不是用语义类别或观测的多数 GT ID 替代实例。实例坐标直接与 SLAM 点集对齐，3 cm 内最近邻转移实例 ID；unknown/无有效 GT 点标为 void。有效实例 GT 低于前景点的 95% 时停止并提示检查对齐，不能在少量容易点上偷偷报告高 PQ。

预测实例：在全部固定 SLAM 点上，以最近地图点的对象索引作为 owner，最大距离 5 cm；超出标为未覆盖，但这些 GT 点仍保留在 GT 面积与 FN 计算中。

- 所有前景实例视为一个类别，不比较 CLIP 类别。
- `IoU > 0.5` 严格匹配；恰好 0.5 不匹配。互斥点分区在该阈值下自然一对一，不使用“先最大总 IoU 匹配再过滤”的歧义流程。
- `RQ = TP/(TP+0.5FP+0.5FN)`；`SQ = sum(IoU)/TP`，无 TP 时为 0；`PQ = RQ × SQ`。
- GT void 不进入 IoU union；未匹配预测中，超过 50% 已分配参考点落在 void 的对象忽略。这是原 PQ 的 void 处理思路，数量单独报告。
- **本实现另作保守的地图扩展：完全没有获得参考点的地图对象仍计 FP，包括重叠重复对象和空对象。** 防止这些对象在最近邻投影后消失、不受任何惩罚。该扩展和类别无关汇总使它属于明确声明的固定点集地图诊断协议，不可冒充官方 Replica/ScanNet PQ。
- 不按对象大小删除难例，最小 GT 实例只要求有一个有效参考点。

公式和 void 规则参考：[Panoptic Segmentation](https://arxiv.org/abs/1801.00868)、[官方 panopticapi](https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py)。本项目的类别无关汇总、3D 最近邻投影和零支持对象惩罚是显式适配。

输出：`instance_pq.csv`、`matches.csv`、`gt_instances.csv`、`overlap_tables.npz`、`point_assignments.npz`、`results.json`、`status.json`、`evaluation.log`。PQ/RQ/SQ 均为 [0,1]，终端为百分比；原始交并计数与逐点对应均可复算。

## 验证

21 项 CPU 单元测试通过：完美分区、同类误合并、过分裂、空预测、空 GT 拒绝、漏实例、重复对象、void 规则、严格 0.5 边界、标签重编号不变性、PQ 分解、距离边界、空地图覆盖、分块最近邻与暴力计算一致、非有限数据拒绝、禁止覆盖结果。

测试文件：`tests/test_replica_instance_coverage.py`。Python 编译和 bash -n 检查通过。

只在 `off2000_10` 的已有地图上分别执行了一次完整离线指标读取验证，未重跑建图，也未运行其他实验：

| 指标 | 结果 |
|---|---:|
| CA-PQ | 40.812502% |
| RQ | 46.616541% |
| SQ | 87.549400% |
| TP / FP / FN | 31 / 22 / 49 |
| GT 实例 / 原地图对象 | 80 / 72 |
| void 规则忽略的预测对象 | 19 |
| 零支持计 FP 对象 | 0 |
| Coverage@2.5 cm | 81.340914% |
| Coverage@5 cm | 82.339916% |
| Coverage@10 cm | 84.157447% |

此场景实例 GT 成功覆盖全部 4,085,377 个前景点；SLAM→实例 GT 距离 P95=8.12 mm，最大=18.08 mm，未触及 30 mm 截断。两个指标的有效前景分母相同。

外层 wall time：覆盖率 26.74 s、实例 PQ 35.04 s；峰值内存约 6.0/6.3 GiB。保留 `/usr/bin/time -v` 日志。结果是脚本验证，不是新门控的提升结论，也不是多场景论文结果。

结果目录（位于 off2000_10/ 下）：

- `geometry_coverage_fixed_support_20260905_171406_320380481/`
- `instance_pq_fixed_support_20260905_171448_777402372/`

独立复核与说明目录：`/home/chenkejun/beauty/instance_coverage_evaluation_20260905/`。

复核程序从保存的原始对应关系重建交集，用整数不等式 `3*intersection > GT_area+pred_area` 复核严格 IoU>0.5，核对逐类覆盖求和、共同点集/地图哈希以及两个脚本逐点距离。见 `validation.json`；该文件仅在所有断言通过后写入。

## 局限与比较要求

实例 GT 采用有距离约束的采样标注表面最近邻，而非精确 mesh 面相交；边界附近仍有转移误差。固定 SLAM 采样密度会影响 IoU，不等同于均匀表面积。一个物体的难例不能因未覆盖而移除；GT void 和全部预测对象数量始终公开。

比较新旧方法时，保持参考点集及哈希、n_exclude、GT/预测距离、场景范围和输入帧预算一致。PQ 提升不能代替语义提升结论，覆盖率提升也不能代替实例正确性。原 mIoU/mAcc 评估继续使用原 evaluate.sh。

本次新增文件未提交/推送 GitHub；未修改现有建图逻辑、历史结果、原 evaluate.sh 或无关工作区文件。
