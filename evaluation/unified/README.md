# 统一指标评估器

选择已完成在线建图的结果文件夹，使用固定完整 GT 网格评估。支持 Replica room0–2、office0–4。此工具只评估最终地图，不继续建图，也不加载未来帧修正地图。

## 使用

```bash
cd evaluation/unified
python3 -m pip install -r requirements.txt
cp config.example.json config.local.json
```
将 `config.local.json` 中的 `reference_root` 和 `clip_text` 改成实际数据路径。相对路径按配置文件所在目录解析。`config.local.json` 不提交 Git。

```bash
# 交互输入结果文件夹，多个地图时选择编号
bash evaluate_folder.sh
# 直接指定文件夹
bash evaluate_folder.sh "/path/to/result_folder" --method merge_2
# 手动指定场景、地图或保存位置
bash evaluate_folder.sh "/path/to/result_folder" --scene room0 --map pcd_run.pkl.gz --out "/path/to/output"
```

`--config /path/to/config.json` 指定参考配置，也可设置 `UNIFIED_EVAL_CONFIG`。`PYTHON=/path/to/python bash evaluate_folder.sh` 指定 Python 环境。`--dry-run` 只检查选择并写清单。

默认输出为所选文件夹的 `unified_metrics/`：`selection_manifest.json`、`comparison.json`、`<method>/<scene>/result.json`、投影、混淆矩阵、实例匹配和逐类指标。未提供 `--method` 时名称是 `selected_result`。输出已有不同输入时会报错，请指定新 `--out`；相同地图、参考和评估代码会复用已有结果。成本元数据不参与缓存指纹，若更新成本清单应使用新输出目录。

## 输入与外部参考数据

- ConceptGraphs：文件夹直接包含 `pcd_*.pkl.gz`；反序列化后包含 `objects`，各物体具有 `pcd_np` 和 `clip_ft`。仅使用可信地图文件，pickle 可执行代码。
- OVI-MAP：已转换的 `.npz`，包含 `xyz`、`instance`、`classes`，不是任意原生 npz。`instance` 是零起始物体索引，-1 表示无归属；`classes` 按物体排列，使用下述共同类别编号。
- `reference_root/<scene>/reference.npz`：同一 GT 网格上的 `xyz`、`semantic`、`instance` 数组。语义编号 1–51，0 为忽略；实例编号遵循 `class_id * 1000 + instance_index`。
- 对应 `manifest.json` 至少包含 `reference_sha256`（reference.npz 哈希）、按编号排列的 `semantic_classes`（51类）、`instance_classes`（排除 wall、ceiling、floor 后的48类）。应保留网格、标签来源和版本等追溯信息。
- `clip_text`：与地图 CLIP 特征匹配的 51 行类别文本特征 `.npy`，顺序与语义类别一致。使用已经验证的固定参考，不应悄然替换网格或文本特征。

当前实验使用完整 NICE-SLAM Replica 网格及 OVI-MAP 提供的对应顶点标签。数据、模型特征和实验地图不随代码上传；其他数据集需要另行准备并验证参考，不能直接套用 Replica 配置。

## 指标和排除规则

所有方法在共同完整 GT 网格上比较。每个 GT 顶点取最近预测点，距离严格小于 5 cm 才接受预测。未覆盖 GT 保留为漏检。

| 指标 | 计算与用途 |
|---|---|
| Instance mIoU | 不看类别，一对一最大总 IoU 匹配；全部合格 GT 实例平均，未匹配为0。衡量物体分割重合程度。 |
| Instance AP25/50/75 | 不看类别，IoU 严格大于对应阈值才匹配；按固定引擎计算精确率-召回率面积，兼顾漏检、误检和重复预测。 |
| Semantic mIoU | 每类交并比再平均；含墙、天花板、地板；GT和预测都缺席的类忽略，只有误报的类保留为0。 |
| Semantic AP50 | 类别正确且 IoU >0.5；在有合格 GT 的类别间平均。 |
| Coverage@5cm | 48 个物体类别的 GT 顶点中，被有效预测覆盖的比例；不要求类别或实例正确。 |
| mAcc | 每类正确预测点数/该类 GT 点数，再对有GT的类别平均；含墙、天花板、地板。 |

实例指标与 Semantic AP 排除墙、天花板、地板、无效 GT 实例及小于100个GT顶点的实例；预测投影不足100点不作有效候选。类别无关指标不会仅按预测类别删除预测。Coverage 不使用100点过滤。语义点指标忽略 GT 未标注点，不按实例大小过滤。

AP 对未匹配预测沿用原引擎忽略规则：忽略区域占预测比例大于当前阈值时不计误检。AP 排序分数是投影点数/最大投影点数，保留6位小数；类别相关 AP 在预测类别内归一化，不是 VLM 置信度。源引擎与许可见 NOTICE.md。

联合结果：实例 mIoU 按 GT 实例数加权；语义指标合并混淆矩阵；AP 跨场景联合计算；Coverage 按顶点数加权。`scene_mean` 单独报告场景简单平均。不同场景集合不能直接比较联合分数。

目录入口不推测未读取的运行成本，缺失不是0；生成式 VLM HTTP 请求和 OVI 的 SigLIP 查询不等价。各方法检测缓存不同的结果不能解释为同输入消融。

## 批量评估与验证

配置文件增加 `entries`，每项包含 `method`、`scene`、`kind`（cg 或 npz）、`map`，可选 `cost`：

```bash
python3 evaluate_unified.py --manifest batch.json --out /path/to/batch_output
python3 -m unittest -v test_evaluator.py
```

测试覆盖完美预测、漏检、空预测、错类别、错合并、错拆分、5cm及IoU边界、小实例和仅误报类别。固定指标代码与当前实验版本保持一致。
