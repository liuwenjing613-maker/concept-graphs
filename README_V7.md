# ali-my-v7-VLMsplit

本版本用于从第 0 帧开始的完整在线建图。已接入保留的四个 VLM 模块；此次只做 smoke，没有完整场景评测。

2026-09-08 已修复重复白名单扫描与锁定事件完整渲染导致的性能问题，见 [性能修复与逐像素验证](docs/V7_PERFORMANCE_FIX_20260908.md)。

## 服务器运行

代码目录：`/home/chenkejun/beauty/conceptgraphs/code/experiments/ali-my-v7-VLMsplit`。
当前服务器已准备好运行环境。新检出代码首次运行时执行 `bash setup_v7.sh`，只把 Pillow 11.0.0（RAQM）和 OpenCV 4.12.0.88 安装到本目录的 `.runtime-deps/`，不改共享建图环境。固定这些版本是为了与保留结果的 VLM 输入图逐像素一致。

```bash
# 未决事件由人工选择
bash /home/chenkejun/beauty/conceptgraphs/code/experiments/ali-my-v7-VLMsplit/run_v7.sh human --gpu 1 --scene room0 --end 2000 --stride 5

# 未决的新建/关联丢弃当前观测；未决合并选择不合并
bash /home/chenkejun/beauty/conceptgraphs/code/experiments/ali-my-v7-VLMsplit/run_v7.sh auto --gpu 1 --scene room0 --end 2000 --stride 5
```

`human` 和 `auto` 都先运行相同的 VLM；只决定未决事件的处理方式。auto 全程不读取人工输入，包含率冲突也只记录并保持不合并；human 应在交互终端运行。页面点击选项会生成答案，将其复制到当前运行终端并回车，只有当前题及冻结快照的编号有效。人工复核页保留当前题和浏览器草稿，不自动跳题。

默认数据目录为 `/home/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica`，使用 room0 已有逐帧检测缓存。`--end 2000 --stride 5` 是原始帧 0–1999、间隔 5，共 400 个处理帧。每次自动生成新的 `v7_MODE_时间_进程号` 输出目录；也可传入未使用过的 `--exp-suffix NAME`。已有目录默认拒绝覆盖；只有本版本生成的同一运行 checkpoint 可以通过 `--resume latest` 续跑。保留原框架第 0 帧初始化及事件触发规则；初始化旁路计入 `frame_zero_new_bypass`。

输出：`数据目录/room0/exps/运行名/`。终端打印地图、结果页及人工复核页的完整路径；页面服务沿用端口 8895，地址为 `http://127.0.0.1:8895/v7VLM_运行名/` 和其 `review/` 子目录。外部访问需沿用现有端口转发。

默认模型 `qwen3.6:35b-a3b-mtp-q4_K_M`，所有阶段 `think=false`、temperature 0、context 32768、output 1024。默认观察和合并接口均为当前可用的 `http://127.0.0.1:11464`；可以用 `--base-url` 与 `--merge-base-url` 分别指定同模型服务。原 11465 在本次 smoke 时离线。默认建图 GPU 为 1，可用 `--gpu` 修改。

## 决策规则

- 观测质量：保留 focused full 的质量提示词和 CURRENT 审核图。通过后，对冻结候选逐个运行 focused 五面板身份判断；候选字母随当前图绑定。
- 多个候选均判 SAME：为每个无序候选对建立合并事件，走节点质量和合并身份 VLM。整组所有对象对均获批准，才在融合当前观测之前执行整组合并并更新全部关联索引；不通过传递性推断补齐缺失判断。否则按 human/auto 处理当前观测。
- 节点质量：独立检查两节点各最多三个历史视角。CONTAMINATED 或接口失败转未决；INSUFFICIENT 仍进入身份判断，保持保留链路。
- 合并身份：用户指定的 RGB 投影＋放大图＋mask 背景说明版本（原保留 26 例为 22/26）。提示词、请求模板与来源哈希在 `conceptgraph/slam/prompts/v7/`。
- 同一无序对象对每处理帧最多记一票。连续两次 MERGE 才能合并；累计两次 KEEP_SEPARATE 锁定不合并。锁定后，选中历史的观测 UID 或 mask 更新才解锁，单纯点云变化不解锁。A/B 对调不改变历史签名。真正合并后的对象身份代次更新，防止旧票用于新节点。
- 仅在最终选择 KEEP_SEPARATE 时计算完整点云的双向包含率：一个点到另一个完整点云最近点距离不超过当前建图体素边长（默认 0.01 m）即命中。任意方向严格大于 90% 时，human 暂停人工复核，auto 仅记录冲突并保持不合并；不会据此直接合并。正好 90% 不触发。
- quality 不通过、身份不确定、格式/接口/普通输入失败以及尚未满足合并条件的多 SAME 事件：human 请求选择；auto 对新建/关联只丢弃当前观测，对合并选择 KEEP_SEPARATE。历史或快照身份损坏属于一致性错误，会停止，不允许使用错误证据继续建图。
- I1 来自当前事件 S；历史和完整点云冻结于 H，C 绑定同一 H；保留 `s<=d<=h<=c` 及主图当前帧、阶段请求/原始输出/格式诊断/耗时/实际合并回调。

## Smoke 与局限

服务器记录：`/home/chenkejun/beauty/v7_split_smoke_20260907/`；阶段总结：`/home/chenkejun/beauty/v7_split_summary_20260907.md`。

- 29 项 v7 检查：两票规则、锁定/历史更新、双向包含率、人工编号校验、故障兜底、多 SAME 全配对、整组合并及关联索引更新、重叠组拦截、候选请求字母绑定。
- 12 项原门控检查、14 项人工合并检查；脚本语法、Python 编译和页面 HTTP/JavaScript 检查。
- 渲染 smoke 只读一条观测及保留清单中的一条合并，不调用 API；四类输入图与保留版本逐像素一致。
- 以下真实 smoke 为最初版本的历史验证；后续按用户最新要求，auto 已取消包含率冲突的人工暂停，新增无交互回归检查。
- 真实在线 smoke 都从空图第 0 帧开始，仅处理 2 帧。为覆盖事件临时放宽关联触发，最多处理 2 条观察事件；这些参数不进入正式运行默认值。
- 合并 smoke 完成节点质量/身份真实调用，并在不合并包含率 92.75%/99.41% 时正确暂停人工。该次检查由开发者停止，未伪造人工答案。观察链路另外以 `--no-merge-review` 隔离 smoke 检查真实质量与候选身份请求，并验证正常保存地图；该短图不用于建图效果评测。
- 没有重跑 26/103/268 例评测，没有运行完整场景；两票合并执行、多候选及 human 回答分支由小型合成检查覆盖，不代表已验证完整场景质量或真实人工判断。

复现逻辑 smoke：
```bash
cd /home/chenkejun/beauty/conceptgraphs/code/experiments/ali-my-v7-VLMsplit
export PYTHONPATH="$PWD/.runtime-deps:$PWD"
/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python -m unittest discover -s tests -p test_v7_split.py
```


## 分阶段人工复核与 latest checkpoint（2026-09-08）

- 当前观测质量不通过或接口失败：human 仅选择 `USABLE`（可用，继续候选身份 VLM）或 `UNUSABLE`（不可用，丢弃）。
- 历史节点质量不通过或接口失败：human 分别判断 A/B 的 `CLEAN`、`CONTAMINATED`、`INSUFFICIENT`；全部通过才进入合并身份 VLM。污染或证据不足时不合并，若同时命中 >90% 包含率冲突，仍单独请求冲突复核。
- 身份与几何冲突复核保留原来的候选／新建／丢弃和合并／不合并选项。网页显示对应问题和证据，选择后复制答案到运行终端回车。草稿不会作为正式答案提交。

新建运行（默认逐帧保存 checkpoint）：
```bash
bash run_v7.sh human --gpu 0 --exp-suffix v7_human
```
中断后续跑（同一输出名称，其他场景、stride、end、门控参数必须与首次一致）：
```bash
bash run_v7.sh human --gpu 0 --exp-suffix v7_human --resume latest
```
将 human 替换为 auto 可用于 auto 运行。`--gpu` 可换物理卡；模型、模式、输入范围与运行代码必须保持一致。不要同时启动同一输出目录的两个进程。

断点位于 `<实验输出目录>/checkpoints/latest.json`，原子指向最近写完的状态文件，保留最近两份状态。恢复地图、精确点云和包围盒、对象/证据版本、合并票数与锁定、支持历史、运行统计和 Python/NumPy/Torch 随机状态。继续使用原 run_id。

断点粒度为完整帧：若在某帧等待人工或执行 VLM 时中断，恢复时重新处理该帧；本帧尚未提交的人工选择可能需要重答，VLM 请求可能重做。未完成帧的产物移到 `checkpoints/interrupted/`，日志回退到断点长度，避免重复融合或重复计票。不是恢复 Python 调用栈，也不会跳到未来帧。

当前支持冻结检测缓存、make_edges=false、revision=false、vis_render=false（run_v7.sh 默认流程）；在线重新检测的另一工作树不在此次支持范围。旧版本未写 checkpoint 的运行不能恢复。已完成运行拒绝再次续跑。

验证：41 项小型回归通过；2 帧真实在线 CPU smoke 在第二帧合并期间主动中断，恢复后从 frame 1 继续，最终 return_code=0。小图 checkpoint 写入约 0.12–0.19 秒；未验证全场景写盘开销。最初新进程遇到服务器 CUDA 初始化失败；重连后 CUDA 恢复可用，补做 1 帧 GPU 建图及 CUDA tensor checkpoint 加载成功。中断恢复的完整验证使用 CPU。记录：`/home/chenkejun/beauty/v7_checkpoint_smoke_20260908/`。
