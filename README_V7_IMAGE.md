> 当前目录为 v7_merge_70，实际阈值90%/60%，当前规则和命令见 [README_V7_MERGE_70.md](README_V7_MERGE_70.md)。下文为继承版本说明。

> 本目录为 v7_merge，当前行为与运行入口见 [README_V7_MERGE.md](README_V7_MERGE.md)。下文为继承版本说明，90% 包含率策略以新说明为准。

# v7_VLMsplit_image

基于 ali-my-v7-VLMsplit / df4d51d，只改变 YOLO 推理输入、事件内 HTTP 并行调度和超时重试。保留 SAM、CLIP、VLM 模型/请求提示词、候选数、选图、包含率、投票和 human/auto 策略。

## 运行

在服务器此目录执行：

```bash
bash run_v7_image.sh auto
bash run_v7_image.sh human
```

默认 room0，start=0/end=2000/stride=5（400帧），GPU1，从空图开始，输出名自动唯一。默认三个接口：11464、11463、11435。最多三个不同接口；每个接口客户端最多一个在途请求。设置两个接口：

```bash
bash run_v7_image.sh auto --vlm-urls http://127.0.0.1:11464 http://127.0.0.1:11463
```

覆盖 GPU 或输出名称：

```bash
bash run_v7_image.sh auto --gpu 1 --exp-suffix MY_UNIQUE_NAME
```

YOLO imgsz=1200、rect=True；Ultralytics 按32倍数调整，1200×680源图实测网络张量为1216×704。SAM仍默认1024，mask保持原图1200×680。不是把RGB/depth/内参都缩放到YOLO尺寸。

## 检测缓存

默认每次生成独立的 `<实验名>_detections`，并保存检测框、mask、CLIP特征、模型权重/类别顺序/软件版本、原图哈希和实际输入尺寸。建图只在处理当前帧时生成和消费该帧检测，不读取未来对象或已有地图。首次生成检测时间包括在本次运行内。

显式复用本版本生成且覆盖全部请求帧的检测缓存：

```bash
bash run_v7_image.sh auto --detections-exp-suffix PREVIOUS_NAME_detections
```

复用检测仍从空地图开始。旧640缓存、缺少本版本manifest的缓存、配置/类别/权重不一致或缺少帧的缓存直接拒绝；不会静默使用旧检测。复用时原图哈希和mask坐标也必须匹配。不要把两帧smoke缓存传给400帧正式运行。

## 并行与超时

质量审核通过后并行比较A/B/C；合并时两个节点的质量审核并行。所有结果返回后按原候选顺序提交，帧间仍阻塞；多SAME仍审核全部对象对，主线程唯一修改地图和投票。

每次HTTP读/连接等超时或HTTP408/504，保持同一输入、候选和H快照，换另一个可用接口重试。初次调用加最多3次重试：累计第4次超时才失败。默认单次超时300秒，可用 `--vlm-timeout 60` 改为60秒。计数属于一个逻辑stage，跨接口共享，不是每个接口再试4次。不会同时重复投递同一逻辑请求；客户端停止等待并不保证后端立即取消旧推理。

超时用尽后：auto丢弃当前观测/不合并；human交给人工选择。超时重试不增加合并票。非超时格式错误、输出截断及视觉不确定仍按原策略处理，未增加新的重试或修改token上限。

每个stage下 `attempts/01..04/attempt.json` 记录接口、耗时、HTTP状态和超时标记；成功响应保留原 `response.json`，最终只生成一个stage结果。并行stage的seconds会重叠，不能简单求和当作墙钟耗时。端口可用性及GPU独占情况需按启动时状态确认。

## 保持的规则

- 2次累计不合并锁定，选中历史视角/mask更新后解锁；2次连续合并获批；同一对象对同帧不重复投票。
- 不合并时双向点云包含率>90%：human复核；auto记录冲突并保持分开，不等待人工。
- 不缩减历史、不缓存VLM答案、不修改提示词、不跳过候选或质量判断。

smoke记录见 docs/V7_IMAGE_SMOKE_20260908.md。完整场景由用户运行。
