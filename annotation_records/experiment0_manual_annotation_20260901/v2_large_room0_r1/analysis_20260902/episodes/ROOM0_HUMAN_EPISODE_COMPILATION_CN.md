# Experiment 0 room0：人工标签驱动的 episode 与未来证据编译

## 结论

标注已从页面答案转换为事件时节点 UID、root/cascade 角色、候选排名和未来独立视角。后续 oracle replay 只读取这份事件表。

## 人工确认错误

- 合并后的独立人工事件：174
- 人工确认错误事件：14
- root：7；cascade：6；待定：1
- 30 个 mapper update 内至少 2 个独立未来视角：8
- `top-5 + NEW` 覆盖率：0.8571428571428571

## 自然概率队列中的错误

- 错误事件：5；root：1；cascade：3；待定：1
- 30 updates 内两视角覆盖：4/4
- 首批可进入 B0/B1/B2/B3 oracle replay：['room0_large_r1_0143']

## 下一步约束

1. 先跑最少的人工确认 root，不能把 cascade 当独立样本。
2. B0/B1/B2/B3 使用同一错误前 snapshot、同一未来 observation 顺序和相同特征。
3. proposal 与 validation 视角不重叠；不足三视角时只做 evidence ceiling，不声称安全提交。
4. room0 只用于开发；并行准备 room2 从 frame 0 空图的同配置日志。
