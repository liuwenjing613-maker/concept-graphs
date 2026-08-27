# ali-my-new-VLM：统一三视图 VLM 最小验证（v1.1）

这个验证器只检查两件事：VLM 实际看到的证据是否合适，以及它从有限候选中返回的判断是否清晰、合规。它不会 replay、生成可提交修复或修改场景图。

核心约束：

- 每例只调用一次 VLM，固定三张不同 frame 的图片；
- I1 是 incident 精确帧，I2 是质量最高视图，I3 是互补位姿视图；
- overlay 只读取冻结在线快照内 accepted observation 的 `processed_mask_ref`；
- 不读取 `final_membership.json`、旧 VLM response 或旧 compilation；
- A/E0/E1/E2 固定为黄/青/洋红/绿，但这些颜色只用于指示 mask，不是物体真实颜色；system、incident 和每张 `IMAGE_SPEC` 都明确禁止用蒙版色差判断 identity、material 或 semantic；
- API 请求按 `IMAGE_SPEC -> IMAGE` 逐图交错；
- VLM 只能选有限 candidate ID，输出经过严格 schema 和轴一致性校验；
- 输入文字同时包含 A/E 的机器 label、accepted 支持数、历史 label 计数、top candidate 分数和有限动作含义；
- 对只有在线 `OBJECT_MERGE` 事件、没有旧 VLM packet 的合并冲突，可从合并前不可变版本生成只读冻结案例；仍不读取最终成员关系；
- 当前 API 的 structured-output 子集不接受 `uniqueItems`；请求 schema 保留其余严格约束，数组唯一性由本地 fail-closed 校验补齐；
- 每个 case 保存三张图片、图片 hash、完整提示词、脱敏后的实际请求、API 原始响应、解析结果和 cutoff 审计。

运行示例：

```bash
python scripts/validate_unified_vlm_v1.py \
  --experiment-root /path/to/online/experiment \
  --output-root /path/to/new/output \
  --ticket ticket_x --ticket ticket_y
```

API Key 只从运行进程的 `ALI_VLM_API_KEY_1..N` 环境变量读取，不写入仓库或结果文件。先用 `--prepare-only` 可以在不调用 API 的情况下检查图片和证据审计。

人工查看从结果目录的 `index.html` 开始；每个 case 的 `review.html` 先展开显示文字/数值输入和 alias→label，再显示三图和精简输出解读。原始 JSON、完整请求与 cutoff 审计折叠保留。

可选 `--review-metadata` 只用于在 HTML 中写事后人工对照；程序会在所有 API 调用结束后才加载它，并记录 `request_inclusion=false`，不能影响请求或模型判断。
