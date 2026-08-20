# 已退役：trigger-incident v2.0

这个目录保留用于追溯，不再进行人工标注。它在 `0/160` 时停止，没有标签需要迁移。

停止原因：虽然它合并了同一 trigger 上的跨-checker findings，但 160 个案例中仍有 147 个落在重复 final-owner set 上，同一个 final object 最多会被判断 11 次。对于“最终地图是否正确”的 R1 问题，这种重复没有意义。

当前正式入口：

```text
/home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1
protocol = final_endpoint_r1_v2_1
cases = 97 distinct final objects
URL（经 SSH 隧道）= http://127.0.0.1:8765/
```

不要在本目录创建或填写 `labels_r1.jsonl`，不要启动本目录对应的旧服务。
