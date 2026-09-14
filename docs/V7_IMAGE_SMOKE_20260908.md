# v7_VLMsplit_image smoke 验证（2026-09-08）

## 变更范围

基于 df4d51d，新目录和分支均为 v7_VLMsplit_image。YOLO请求imgsz=1200、rect=True；三个端口11464/11463/11435共享同一模型，事件内候选和双节点质量请求并行，主线程按固定顺序提交。超时原请求加3次重试，第四次超时才按原human/auto策略兜底；HTTP408/504也计超时；其他格式失败/截断仍按原规则处理。重试不多计票。模型、提示词、输出上限、SAM、CLIP、选图、候选、合并与包含率策略未改。

新检测缓存独立保存并记录权重、类别、软件版本、RGB哈希和实际YOLO尺寸；旧640和不完整缓存拒绝。运行目录、最新点云入口及依赖目录与旧v7隔离。

## 验证结果

- 原有 v7 单元检查35项通过，新增并行/超时/缓存检查11项通过，真实对象合并与严格证据回归1项通过。
- 并行模拟验证6个任务最多3个在途HTTP请求、同一接口不重复占用、响应乱序仍按A/B/C归位。
- 超时模拟验证3次超时后第4次成功正常继续，4次超时无第5次调用，跨接口轮换且输入不变，只提交1个stage结果；auto和human按原策略兜底。
- 真实在线2帧 smoke（v7_image_smoke_20260908_a），从空图和新检测开始，退出0、证据audit PASS。两帧实际YOLO张量均为[704,1216]，原图/mask为[680,1200]；原始检测35/37个。
- 该在线smoke包含节点质量6次、合并身份3次，9次真实调用全部成功，三个端口各3次。
- 独立真实三候选传输smoke：使用已完成在线运行f000003_d016_create的冻结输入，验证哈希及H绑定；11464/11463/11435各1次成功，墙钟12.565秒，三次请求耗时合计21.164秒。它仅验证并行链路，不是精度评测或全场景提速测量。
- 1帧缓存复用smoke（v7_image_cache_smoke_20260908_b），off模式、从空图重新建图，退出0、audit PASS，无VLM调用，验证缓存可复用和新目录路径隔离。

## 发现与处理

首次2帧smoke发现继承base_paths.yaml仍指向旧v7目录；已修改新目录默认值并在launcher中显式绑定repo_root。旧v7 latest_pcd_save只在仍指向本次smoke时恢复到其已完成的全程auto地图。随后缓存smoke证明新版本仅更新自己的latest_pcd_save。运行依赖从已验证版本复制，不下载模型或改动旧依赖。

## 产物

日志与真实并行明细：/home/chenkejun/beauty/v7_image_smoke_20260908/
- legacy_unit.log
- parallel_unit.log
- merge_evidence.log
- online_a.log
- cache_b.log
- real_parallel/validation.json 与各stage attempts/

在线产物：
/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/v7_image_smoke_20260908_a
/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/room0/exps/v7_image_smoke_20260908_a_detections

## 局限

只做smoke，未运行400帧、未评估新检测精度或正式全程速度。2帧在线smoke没有触发观测关联阶段，三候选真实调用由独立冻结事件传输smoke验证。超时切换由受控异常模拟验证，未故意占用/干扰真实服务制造300秒超时。停止等待不能保证服务端立即停止超时的旧推理。多端口可能存在共享后端资源，实际加速以完整运行日志为准。

## 用户运行

```bash
cd /home/chenkejun/beauty/conceptgraphs/code/experiments/v7_VLMsplit_image
bash run_v7_image.sh auto
# 或人工兜底
bash run_v7_image.sh human
```

默认400帧、GPU1、三个接口、单次超时300秒、独立新检测缓存。完整实验由用户运行。
