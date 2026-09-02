# 实验 0：room0 冻结 baseline 双跑确定性审计

## 结论

最终判定：**PASS**。同一冻结配置从空图重新在线运行，经过 UID 规范化后，完整证据轨迹与最终数值状态均与原 run 一致。

- 处理帧 trace：399 / 399，完全一致；
- association 事件：7507，规范化后完全一致；
- 最终对象：72；observation：7507；点：740267；
- 最终 observation 分区、72 个对象点云、bbox 和类别：全部精确一致；
- 相似度矩阵文件：400，数组逐值完全一致；
- 两次 strict evidence 状态：MAP_COMPLETED_EVIDENCE_VALID / MAP_COMPLETED_EVIDENCE_VALID。

## 比较口径

两次运行会生成不同的 run ID、event/transaction UID 和 object UUID，因此整个压缩 pickle 的字节哈希不要求相同。审计按每个对象的 CREATE observation 建立一一映射，再比较 frames、filter trace、observations、associations、mapping events、object versions、object-pair decisions、final membership、逐帧 parity trace、相似度数组和最终数值状态。

## 对实验 0 的意义

room0 的 baseline 在当前冻结环境下具有足够的重复确定性；当前发现的标签与 root/cascade 差异不是随机重跑漂移造成的。这个结论只验证 room0 当前配置，后续新场景仍需保留 run manifest 和 strict evidence audit。
