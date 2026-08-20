# R1 最终 endpoint 复核：每组最终对象只判断一次

这不是旧版的“逐条 checker finding 标注”。新版以**完全相同的 active final-object 集合**作为一个 endpoint review unit：即使不同 observation、不同阶段产生多条报警，只要最后指向同一组对象，人就只判断一次。没有 active final owner 的孤立事件才按触发 observation 单独保留。

当前 room0 69 + office0 28 = 97 个 endpoints，正好对应 97 个不同 final objects；全部纳入 R1，不是抽样。页面图中的 `O1 [ENDPOINT]` 是待判对象，`[context]` 只帮助比较。

## 你只回答三个问题

1. **证据足以判断最终状态吗？**
   - `YES`：最终点云、对象视图和成员信息足以选“正确”或“错误”。
   - `NO`：关键对象、视角或几何证据不足，不能可靠区分对错。

2. **最终地图对象状态是什么？**
   - `CORRECT`：最终身份、节点数量、成员和几何没有可见错误。即使上游出现过重复 proposal、低 margin 或暂态 CREATE，只要最终状态正确，就选它。
   - `WRONG`：错误仍真实保留在最终地图中。
   - `UNCLEAR`：现有证据不能可靠判断。证据为 `NO` 时固定选它。

3. **如果最终仍然错误，可见错误属于哪一类？**
   - `FALSE_MERGE`：多个真实物体错融成一个最终节点。
   - `FALSE_SPLIT`：同一真实物体仍保留成多个最终节点。
   - `SPURIOUS_OBJECT`：最终节点只是噪声、背景或碎片。
   - `MISSING_OBJECT`：应存在的对象没有有效最终节点。
   - `WRONG_MEMBERSHIP`：有效 observation 被放进错误对象。
   - `GEOMETRY_CORRUPTION`：最终点云、位置、尺度或形状明显损坏。
   - `SEMANTIC_IDENTITY_ERROR`：对象几何存在，但稳定语义身份明显错误；同义词不算。
   - `OTHER`：以上都不合适，必须在备注中写一句说明。
   - 最终不是 `WRONG` 时固定选 `NOT_APPLICABLE`。
   - 多种错误同时存在时，选择最直接决定 O1 为什么错误的主类型，并在备注补充其他现象。

## 固定查看顺序

1. 先看“最终地图对象”的统一坐标图和逐对象放大图。
2. 再看对象成员数、帧跨度、类别统计与代表视图。
3. 仍有疑问时，才展开触发 observation 和系统当时的关联记录。
4. 不判断 checker 是否正确，不猜最早根因，不猜修复动作。

## 为什么不再问根因和修复

阶段根因必须面对该阶段真正使用的历史状态；缺少阶段快照时不能靠最终图片反推。修复是否有效必须通过真实 intervention / replay 验证，也不应由 R1 人工猜测。

R1 完成后，系统只把 `evidence_sufficient=YES + final_state=WRONG` 的 endpoint 送入专家因果追踪队列。最终修复只有在重跑后确实改善对象图时才算验证通过。

## 页面地址

服务只绑定服务器 `127.0.0.1:8765`。保持 SSH 隧道后打开：

`http://127.0.0.1:8765/`

当前协议是 `final_endpoint_r1_v2_1`，验证根目录为 `/home/chenkejun/beauty/conceptgraphs/validation_gate_endpoint_v2_1`。

页面自动保存到 `labels/labels_r1.jsonl`。不要手工改 `r1_worklist.jsonl`，它是冻结的 97 个 final-endpoint 普查清单。
