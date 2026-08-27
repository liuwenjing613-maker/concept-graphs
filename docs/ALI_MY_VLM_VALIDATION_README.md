# ali-my-new-VLM：统一三视图 VLM 最小验证

这个验证器只检查两件事：VLM 实际看到的证据是否合适，以及它从有限候选中返回的判断是否清晰、合规。它不会 replay、生成可提交修复或修改场景图。

核心约束：

- 每例只调用一次 VLM，固定三张不同 frame 的图片；
- I1 是 incident 精确帧，I2 是质量最高视图，I3 是互补位姿视图；
- overlay 只读取冻结在线快照内 accepted observation 的 `processed_mask_ref`；
- 不读取 `final_membership.json`、旧 VLM response 或旧 compilation；
- A/E0/E1/E2 固定为黄/青/洋红/绿；
- API 请求按 `IMAGE_SPEC -> IMAGE` 逐图交错；
- VLM 只能选有限 candidate ID，输出经过严格 schema 和轴一致性校验；
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

人工查看从结果目录的 `index.html` 开始；每个 case 的 `review.html` 同时展示实际三图输入、候选、输出和完整请求。
