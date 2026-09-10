# v7_CLIP

基于 v7_merge fb05116，接入 DarrenYeh05 feat/clip-bbox-softmask-fusion / 0ae881f。
保留 v7_merge 的合并规则。CLIP 默认 bbox 与抑制背景视图等权融合；缓存绑定参数。

```bash
cd /home/chenkejun/beauty/conceptgraphs/code/experiments/v7_CLIP
bash run_v7_CLIP.sh auto --scene room0 --gpu 1
```

`human` 支持人工兜底；`--clip-masked-weight 0` 恢复 bbox-only。
默认实验名称使用 v7_CLIP 前缀，结果和检测缓存独立。
VLM 默认仍为 11464、11463；与旧任务同时运行会共享这些服务，--gpu 不改变 VLM GPU。

CLIP 版本已完成两帧真实在线 smoke。为避免再次争用 VLM，迁移仅做离线检查，不重复调用 VLM。
历史 smoke 结果及原始证据保留在原路径，不修改其中的来源记录。
报告：/home/chenkejun/beauty/v7_merge_clip_smoke_20260909/summary.md
