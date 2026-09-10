# v7_merge_re

基于 v7_merge fb05116（父版本继承 v7_VLMsplit_image fbd00e0）。本版只收紧 >0.9 包含率覆盖合并的入口；不合入 v7_CLIP、merge_2 或双向阈值改动。

## 行为
- 仅当前事件身份 VLM 的有效 DIFFERENT 回答可进入阈值检查；阶段输出与身份结果一致，并绑定同一个 H/C snapshot。
- 污染导致未调用身份模型、身份 UNCERTAIN、接口/输入失败、人工/自动兜底 KEEP_SEPARATE、历史锁定均不能触发阈值合并。
- 几何检查自身失败后不经异常兜底重试覆盖合并。
- 合格 DIFFERENT 事件仍沿用任一完整点云方向包含率严格 >0.9 的直接批准；恰好0.9不批准。
- SAME 仍按原两票流程；原拒绝计票、历史更新解锁、提示词、前端、模型及距离半径均不变。
- 质量 INSUFFICIENT 后是否进入身份判断沿用原版；若后续身份有效 DIFFERENT，仍可进入阈值检查。
- human 模式原有人工明确 MERGE 权限保留；本次只修复非明确身份拒绝也进入几何覆盖的问题。

## 运行
```bash
cd /home/chenkejun/beauty/conceptgraphs/code/experiments/v7_merge_re
bash run_v7_merge_re.sh auto --scene office2 --gpu 1
```
默认 start=0/end=2000/stride=5，从空图在线运行400帧，自动生成独立实验名 v7_merge_re_auto_<timestamp>_<pid>。默认服务和模型与 v7_merge 相同。可加 --dry-run 检查路径及命令，不启动建图。
输出根目录：/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/<scene>/exps/<exp-suffix>/。

## 验证与边界
57项测试通过，包括真实Open3D合并与证据、污染/未决/接口失败/锁定禁用几何、快照不匹配、严格0.9、正常SAME两票、同帧去重及新历史解锁。
33次历史强制合并冻结几何检查：15次明确DIFFERENT保留，18次污染兜底拦截；原文件哈希未变。仅固定事件分支验证，不是新在线地图成绩。
未启动全场景建图或新的模型请求。尚不能声称指标提升；此改动仍允许几何覆盖明确DIFFERENT，因而不能保证消除所有错误合并。
详细日志：/home/chenkejun/beauty/v7_merge_re_validation_20260910/。
