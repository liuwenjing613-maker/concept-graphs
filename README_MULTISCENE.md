# 多场景建图与评测

建图示例（结果名必须未使用）：

```bash
bash run_v7_image.sh auto --scene office2 --gpu 0 --exp-suffix v7_auto_office2_v2
```

human 模式将 auto 替换为 human。`--dataset-root` 现在指定原始输入目录，默认 `/home/chenkejun/beauty/conceptgraphs/data/Replica`。
`--output-root` 默认 `/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica`。
输出为 `<output-root>/<scene>/exps/<exp-suffix>`，检测缓存也在该场景的 exps 下。
只链接 results 和 traj.txt 原始输入，不导入旧 exps 或已建地图。已有结果及启动记录仍禁止覆盖。

评测示例：

```bash
bash /home/chenkejun/beauty/conceptgraphs/scripts/evaluate.sh v7_auto_office2_v2 office2
```

支持 room0–room2、office0–office4。评测自动映射 office2→office_2、room0→room_0。
优先读取指定结果根目录，也兼容历史 data/Replica 下的已完成输出。评测目录新建于对应地图旁边，不覆盖已有评测。
脚本不带参数时默认 v7_auto_office2 / office2；建议显式提供结果名和场景。
两个命令都可在末尾加 `--dry-run`，仅检查路径与参数，不调用 GPU/VLM、不建图、不评测。
现有运行不会改变输出位置，此调整仅适用于后续新启动。
