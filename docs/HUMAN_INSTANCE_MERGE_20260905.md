# 实例合并人工审查：实现与 smoke

日期：2026-09-05。服务器工作分支：ali-my-VLMv，修改基于 7170943；本次未 commit/push。

服务器代码目录：

`/home/chenkejun/beauty/conceptgraphs/code/experiments/ali-dev-blocking-gate-v1-20260903`

## 本次改动

仅在 human 模式增加后处理实例合并审查。原有四个观测触发条件、原有观测动作（包括 UNCERTAIN / DISCARD）、VLM 提示词和其他模式不变。

原算法满足 overlap / visual / text 阈值、两个对象仍有效且已有 merge_guard 允许之后，在真正融合前暂停。周期合并与最终合并使用同一入口，均覆盖；原算法本来不合并的对象不额外询问。

- 合并：继续执行原有融合、特征更新和去噪代码。
- 不合并：这一对对象保持分离，不执行此次融合。
- 没有新增 VLM 调用，没有自动替人工决定。
- 输入中断、证据失败、快照发生变化时终止本次执行，不默认放行。

同一对对象状态完全不变时复用“不合并”，包括 A/B 反方向重复提议。点云、成员、计数或特征改变后重新审查。这不是永久 cannot-link，也不修改后续 observation 的关联规则。

## 人工页面和证据

沿用当前实验的同一个 `blocking_association_gate/human_review.html`，观测题与合并题轮流更新，不需要每题打开新 HTML。

合并题只有两个按钮：合并 / 不合并。点击按钮复制绑定本题快照的答案，再到原终端粘贴并回车；复制失败时可从输入框手动复制。不要只输入 Y/N：系统故意拒绝无题号或旧题答案，防止旧页面误投下一题。

两个对象各一张 1024×1024 JPEG：上半最多 3 张代表历史红色 mask，下半共享同一坐标尺度的 XY/XZ/YZ 点云对比。洋红色为 OBJECT A，青色为 OBJECT B。历史选择沿用现有 best-mask / recent / diverse 逻辑。历史不足 3 张时不伪造观测。点云直接取本次融合前两个实时对象的 pcd；保留完整有限点集，绘图最多各 5000 点，显示范围沿用已有稳健范围算法。

页面不显示原始相似度，不要求人工做类别匹配。同类、相邻、接触不足以证明同一个物理实例；没有足够同实例证据时保留分离。

## 如何运行

继续用原来的启动命令和 `--mode human` 即可，新审查默认开启。必须使用新的实验输出目录，从帧 0 重新在线建图；不要覆盖已有结果。

对照旧版人工模式时，启动脚本增加 `--no-human-merge-review`。直接使用 Hydra 时对应 `association_gate.human_merge_review=false`。该开关只控制新增实例合并审查，不关闭原有四种观测触发。原有 `max_events` 仍只限制观测事件，不使后处理合并绕过人工审查。

每次实验在 `blocking_association_gate/human_instance_merge/` 留存：

- `events/<事件>/candidate_A.jpg`、`candidate_B.jpg`：人工实际看的卡片。
- `live_pair.npz`：审查时两个对象的完整有限点云。
- `input_manifest.json`、`decision.json`：快照绑定、历史来源、选择、耗时、帧号与阶段。
- `events.jsonl`、`summary.json`、`index.html`：统一审查记录与计数。

合并 proposal 的时间线 s=d=h，阻塞完成的 c 仍绑定同一快照；这些帧表示合并提议时点，不声称是 GT 错误最早开始帧。原自动合并重叠分数可能来自本轮 merge pass 开头，证据始终取本次真正执行前的最新对象，并在记录中明确区分。实际合并执行仍查看原 `evidence/mapping_events.jsonl`，不能把“人工批准数”当成“成功融合数”。

## 验证结果与局限

只做服务器 CPU smoke，未运行完整场景，未调用 API。

- 新增 14 项通过：允许/拒绝路由、拒绝不改对象、反向重复缓存、对象变化失效、连续融合使用最新状态、旧答案拒绝、EOF 不放行、未来历史拒绝、等待中状态变化拒绝、证据卡尺寸/历史/尺度/点云留存、模式开关、merge_objects 转发、周期/最终入口接线。
- 原有 association gate 12 项、四触发回归 11 项通过，共 37 项。
- 修改的 Python 文件编译通过，git diff --check 通过。
- 与修改前服务器备份逐项比较，所有 VLM PROMPT / POLICY 和 `_prompts` 构造方法 AST 完全不变。

测试目录：`/home/chenkejun/beauty/human_instance_merge_validation_20260905/`

最终新增 smoke 日志：`merge_smoke_verified.log`。最终合成证据产物：`human_instance_merge_smoke_2sgicue1/`。

最初测试脚手架存在 mapper 的 measure_time 包装识别错误，以及 oracle / vlm 测试缺少模拟配置，已修正，早期失败日志保留。修订测试没有修改实际 VLM 调用逻辑。

这些 smoke 在真实合并路由中用测试融合函数验证是否调用；不是完整在线建图、不是人工判断质量测试，也没有验证浏览器在用户终端环境中的剪贴板行为。完整 CPU 融合与真实 room0 证据检查未继续执行，遵循用户“只需要 smoke，自己跑”的要求。

本次不会修复已经融合进同一个对象的历史污染，也不改变观测关联、候选池、DBSCAN 或语义读出。能否提升点级 mIoU / mAcc 需要用户的新完整在线实验验证，不能由 smoke 保证。

## 核心文件

- `conceptgraph/slam/human_instance_merge.py`：新增审查、证据、输入与记录。
- `conceptgraph/slam/utils.py`：执行前审查钩子。
- `conceptgraph/slam/association_gate.py`：human 模式开关、复用证据渲染。
- `conceptgraph/slam/rerun_realtime_mapping.py`：周期与最终合并接线。
- `conceptgraph/hydra_configs/rerun_realtime_mapping.yaml`、`scripts/run_blocking_association_gate.py`：配置及消融开关。
- `tests/test_human_instance_merge.py`：新增 smoke。

原有未跟踪的 .hydra/ 和 CloudCompare 导出脚本/测试未改动。
