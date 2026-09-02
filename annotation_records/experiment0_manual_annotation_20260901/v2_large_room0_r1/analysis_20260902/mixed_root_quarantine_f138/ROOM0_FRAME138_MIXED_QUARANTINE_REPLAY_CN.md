# Room0 frame138 混合 mask 隔离反事实

- 执行状态：**PASS**
- 科学结论：**QUARANTINE_INSUFFICIENT_FOR_NATURAL_RECOVERY**
- anchor：`room0_20260831T111035Z_5c9d86fa_f000138_r0016`
- 干预：只隔离 anchor，不给后续 observation 注入 GT 或人工路线。
- 评测：回放完成后才读取校正 GT；MIXED observation 不参与完整实例指标。

| 分支 | GT15 best P/R/F1 | GT19 best P/R/F1 | 两实例最佳实体不同 |
|---|---|---|---:|
| B0 原始 | 0.469/1.000/0.638 | 0.531/1.000/0.694 | False |
| Q1 隔离 | 0.469/1.000/0.638 | 0.531/1.000/0.694 | False |

## 完整性边界

- anchor 最终 owner 数：0（期望 0）。
- 其余 observation 不变量：True。
- 受影响集合之外改动：0。
- 源证据结束校验未变：True。
- 这是人工/离线发现混合根因后的 oracle 隔离上限，不代表自动混合检测器已经完成。
