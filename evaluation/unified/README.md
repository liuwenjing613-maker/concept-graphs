# 统一指标评估器

**协议版本：`unified_fullmesh_v2_20260910`。旧 v1 分数不能混入本协议；旧地图必须使用新输出目录重评估。**

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

所有方法在共同完整 GT 网格上比较。投影前排除 owner=-1 的无归属点，每个 GT 顶点取最近有效物体预测点，距离严格小于 5 cm 才接受预测。未覆盖 GT 保留为漏检。

| 指标 | 计算与用途 |
|---|---|
| Instance mIoU | 不看类别，每个 GT 独立取最大 IoU，可重复使用预测；GT 至少10点，预测不作大小过滤，漏检为0。匹配与平均方式参照 OVI 原脚本，评估对象仍使用共同48类。 |
| Instance AP25/50/75 | 不看类别，IoU 严格大于对应阈值才匹配；按固定引擎计算精确率-召回率面积，兼顾漏检、误检和重复预测。 |
| Semantic mIoU | 每类交并比再平均；含墙、天花板、地板；只平均 GT 实际出现的类别。union-present 版本另存诊断。 |
| Semantic AP25 / AP50 | 类别正确且 IoU >0.25 / >0.5；在有合格 GT 的类别间平均。 |
| Object Surface Coverage@5cm | 48 个物体类别的 GT 顶点中，被有效预测覆盖的比例；不要求类别或实例正确。 |
| mAcc | 每类正确预测点数/该类 GT 点数，再对有GT的类别平均；含墙、天花板、地板。 |

主 Instance mIoU 的 GT 最少10点，预测不做100点过滤。原 Hungarian 一对一版本保存为 `instance_mIoU_one_to_one`，继续采用 GT/预测各100点门槛。AP 同样采用100点门槛。三者都对 GT 排除墙、天花板、地板及无效实例；类别无关指标不按预测语义标签过滤候选。Object Surface Coverage 不过滤小实例，字段为 `object_surface_coverage_5cm`，不是实例召回率。

内部诊断保存 TP/FP/FN、Precision/Recall/F1@50：GT/预测至少100点，IoU >=0.5，一对一最大匹配数量优先、总IoU其次。诊断中所有合格但未匹配的预测计FP，不沿用AP的void忽略规则。没有GT时 Recall 为null；GT和预测都为空时F1为null。

CLIP 物体与文本特征均显式 L2 归一化后计算余弦相似度。零向量和非有限值报错。固定51类，不使用 exclude6；door/window 仍参与评估。

AP 对未匹配预测沿用原引擎忽略规则：忽略区域占预测比例大于当前阈值时不计误检。AP 排序分数是投影点数/最大投影点数，保留6位小数；类别相关 AP 在预测类别内归一化，不是 VLM 置信度。源引擎与许可见 NOTICE.md。

联合结果：实例 mIoU 按 GT 实例数加权；语义指标合并混淆矩阵；AP 跨场景联合计算；Coverage 按顶点数加权。`scene_mean` 单独报告场景简单平均。不同场景集合不能直接比较联合分数。

目录入口不推测未读取的运行成本，缺失不是0；生成式 VLM HTTP 请求和 OVI 的 SigLIP 查询不等价。各方法检测缓存不同的结果不能解释为同输入消融。

## 批量评估与验证

配置文件增加 `entries`，每项包含 `method`、`scene`、`kind`（cg 或 npz）、`map`，可选 `cost`：

```bash
python3 evaluate_unified.py --manifest batch.json --out /path/to/batch_output
python3 -m unittest -v test_evaluator test_protocol_v2
```

测试覆盖完美预测、漏检、空预测、错类别、错合并、错拆分、5cm及IoU边界、小实例和仅误报类别。新增格式一致性、无归属最近邻、文本缩放不变性、10点边界与参考 manifest 缓存失效测试。

## 从 v1 迁移

- `instance_mIoU` 现为 GT-best、GT>=10；旧定义保存在 `instance_mIoU_one_to_one`，对应 `gt_instances_one_to_one`。主结果 `gt_instances` 是10点门槛计数。
- `semantic_mIoU` 改为GT-present；旧定义为 `semantic_mIoU_union_present_diagnostic`。
- `coverage_instances` 更名为 `object_surface_coverage_5cm`，更新表格读取字段；原始 covered_instances/vertices_instances 保留作加权分子分母。
- 新增 semantic_AP25 与诊断 TP/FP/FN/P/R/F1。联合 F1 先加总TP/FP/FN再计算，禁止直接平均场景F1。
- 指纹增加 reference_manifest_sha256。旧 projection_cache 字段不再读取，首次计算一律从地图重新投影；同一v2完整结果仍可通过指纹复用。
- 固定 reference.npz、manifest.json 和 clip_text51.npy 无须修改。已有参考包的 config.json 可用于目录入口，入口自动写入新版本。
- 这是共享完整网格上的统一重评估，不能宣称复现 OVI 论文全部数字；前端、语义特征、过滤、聚合与原论文实验条件仍须分别说明。
