# v7_CLIP_TwoRoad

基于同级 v7_CLIP 的独立代码副本。关联/合并继续使用原来的 bbox+softmask 融合 CLIP；语义读出使用单独的 bbox CLIP。原目录未修改。

## 数据流

- 编码：保留已算出的归一化 bbox 向量；融合计算不变，不增加 encode_image 次数。
- 检测缓存：image_feats=融合，bbox_feats=bbox。缓存 schema 为 v7-image-detection-two-road-v1；旧缓存/缺字段缓存拒绝读取。
- 在线对象：clip_ft=关联特征，clip_semantic_ft=语义特征。
- 更新：两路使用完全相同的已接受观测和合并事件，以及原来的检测次数加权、逐次归一化。被丢弃观测不进入对象。
- 匹配、候选排序、合并相似度和阈值继续读取 clip_ft。没有修改阈值、VLM 提示或投票规则。
- 保存：完整地图和逐帧可视化地图保留两路字段。最终输出 clip_readouts.npz（association_fused / semantic_bbox）及 clip_readouts.json（对象顺序、观测 UID、计数）。

## 启动

在本目录执行：

```bash
bash run_v7_CLIP_TwoRoad.sh auto --scene room0 --gpu 1
```

默认生成独立的 v7_CLIP_TwoRoad 前缀输出与检测缓存。该命令仍使用原来的本地 Ollama VLM 端口；当前未启动场景。不要把 codelink URL 直接填入 --vlm-urls：该运行时仍是 Ollama 协议，API 通过实际视觉预检并确定模型后才应接入对应传输协议。

## 语义评测

原始 pcd 文件保留 clip_ft 的关联含义，旧评测器直接读取它会得到融合读出。需要 bbox 语义结果时，先导出专用评测副本：

```bash
/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python scripts/export_two_road_semantic_map.py /absolute/source/pcd_RUN.pkl.gz /absolute/eval_bbox/pcd_RUN.pkl.gz
```

导出仅将副本的 clip_ft 指向 clip_semantic_ft，保持几何、成员、对象顺序和边不变，原文件不动。副本标明 evaluation_only，禁止当作在线初始化地图。对原地图和副本使用相同评测配置，可得到同一张图的两套读出。最终场景仍必须从空地图开始。

## 验证与当前状态

2026-09-10：56 项测试通过，另有 5 项子测试通过；改动 Python 文件语法检查和启动 dry-run 通过。测试涵盖原融合向量精确一致、bbox 独立返回、编码次数、空检测、合并几何/特征一致、混合版本拒绝、过滤对齐、缓存、序列化及语义评测导出。未运行完整或短序列在线场景，尚无真实运行收益与开销结论。

API：5 个 Key 的模型列表请求均 HTTP 200；gpt-5.5 的图片 Chat Completions 请求全部 HTTP 502，提示 Upstream access forbidden, please contact administrator。一个 Key 的 Responses 图片请求同样失败。此结果不代表已验证其他模型。按用户指令，场景启动停止。API Key 仅经隐藏输入进入进程内存，未保存。

验证、原始失败和时间记录：/home/chenkejun/beauty/v7_CLIP_TwoRoad_validation_20260910/
