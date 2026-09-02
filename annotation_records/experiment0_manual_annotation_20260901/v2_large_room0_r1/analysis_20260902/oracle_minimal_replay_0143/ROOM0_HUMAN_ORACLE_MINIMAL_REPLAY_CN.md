# Room0 人工标注驱动的最小 Oracle 回放

- 状态：**PASS**
- 人工根错误：1 例
- 源证据仅在结束时校验一次：True
- B1 只是成员归属对照，不是几何有效的地图。B2/B3 才运行真实 mapper。

| 病例 | 人工类型 | B0 endpoint | B1 endpoint | B2 endpoint | B3 endpoint | B2/B3 invariant |
|---|---|---:|---:|---:|---:|---|
| room0_large_r1_0143 | SHOULD_HAVE_BEEN_NEW | False | False | False | False | True/True |

## 解读边界

- 约束由人工路由标签编译；未用私有 GT 替代人工目标。
- 未来视角是标注后的 oracle 证据上限，不代表已有自动发现器。
- B2 使用类型化前向依赖闭包；B3 使用锚点后全后缀，两者不混称。
- 如果 B2 与 B3 同时失败，说明当前约束传播机制不足，不能归因为闭包裁剪。
