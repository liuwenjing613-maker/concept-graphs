# v7_CLIP_must 实现与启动记录

基准：v7_CLIP 235e16c995fefb905fd3b21c11ad4fd1e8d3bbe4。独立分支与目录 /root/autodl-tmp/beauty/v7_CLIP_must 。

修改：
1. 删除 containment >90% 强制合并及覆盖拒绝锁的执行路径。
2. 多 SAME 确定唯一 anchor：baseline 目标属于 SAME 时优先保留，否则选择原 mapper 分数最高的 SAME。仅审核 anchor 与其他 SAME 的独立对象对；决策期间不修改对象。批准且快照仍有效的合并先执行，观测后接纳。先前合并使另一个对象对版本失效时，延期其旧批准，绝不推断传递合并。合并失败不丢弃观测。
3. 观测质量不足、不确定、接口或解析失败回退 baseline；合法 CORRUPTED 才自动 DISCARD。合并质量不足、身份不确定或失败为 DEFER，不执行合并、不累计拒绝票，但打断连续 MERGE 票。明确 DIFFERENT/CONTAMINATED 才否决。

保留：连续两次 MERGE、累计两次拒绝锁定、同 pair 同 frame 一次、历史变化解锁。提示词、CLIP、检测、候选阈值、历史选择未改变。额外沿用 5090 现有 MapObjectList() 初始化兼容修复。

验证：53 项 v7 测试通过；包含真实 Open3D 合并、观测接纳、严格 evidence 版本、回退矩阵、旧快照延期和索引重映射。静态编译、git diff --check 通过。

room2：5090 建图；H800 Qwen3.6-35B-A3B-FP8 推理。start=0/end=2000/stride=5，从空图开始。复用 frozen_clip_frontend_20260912 的逐帧检测与 CLIP，400 帧 RGB 哈希全部核对。未加载旧地图或历史 VLM 结论。模型适配器与已有 H800 运行相同。

状态：已启动，首批接口 HTTP 200，尚未完成，无正式指标。
输出：/root/autodl-tmp/results/Replica/room2/exps/v7_CLIP_must_room2_20260914/
日志、配置哈希、测试：/root/autodl-tmp/beauty/v7_CLIP_must_run_20260914/

局限：失效快照延期可能增加合并延迟；VLM 明确误判仍可能发生。本次 H800 FP8/输入缓存与历史原 v7_CLIP 不能视为严格单因素数值比较。需保留成功、退化、失败及耗时，不能预先宣称能力提升。
