# 实验0：office0 自动候选的精确事件前历史审计

## 结论

私有 GT 初筛得到 2 个主范围名义候选。加入精确 `t^-` 目标版本后，严格干净候选 0 个；事件前已含当前身份的级联候选 2 个；其余边界候选 0 个。

自动 GT 仅用于挑选值得看的事件，不能替代正式人工标签。这里回答的是候选是否可能是独立 root，不是评估整个检测/分割系统的所有错误。

## 候选明细

### frame 87 · office0_20260831T111046Z_87b8f9c0_e00001366

- 当前实例：GT44 `desk-organizer`，纯度 0.975；
- 原目标精确版本：`2caf3b80-cda2-4b43-9f01-ba382b65a0f4@v000012`；
- 事件前成员：9 条（GT28=8, GT44=1）；
- 事件前当前身份证据：1 条；混合 mask：1 条；
- 判定：`AUTO_OUT_PRECONTAMINATED_TARGET_CURRENT_IDENTITY_ALREADY_PRESENT`；
- 原因：TARGET_MEMBERSHIP_PURITY_LT_0_90, TARGET_PROJECTED_PIXEL_PURITY_LT_0_90, CURRENT_IDENTITY_ALREADY_PRESENT_BEFORE_EVENT, TARGET_CAUSAL_CLEANLINESS_UNCERTAIN。

### frame 258 · office0_20260831T111046Z_87b8f9c0_e00005389

- 当前实例：GT15 `blinds`，纯度 0.996；
- 原目标精确版本：`4df16046-1eb0-4804-9953-6cbde664ecf7@v000107`；
- 事件前成员：102 条（GT16=97, GT15=5）；
- 事件前当前身份证据：29 条；混合 mask：30 条；
- 判定：`AUTO_OUT_PRECONTAMINATED_TARGET_CURRENT_IDENTITY_ALREADY_PRESENT`；
- 原因：TARGET_PROJECTED_PIXEL_PURITY_LT_0_90, CURRENT_IDENTITY_ALREADY_PRESENT_BEFORE_EVENT, TARGET_HAS_PRE_EVENT_CONTAMINATION。

## 对下一步的含义

若严格干净候选为 0，则本场景不能给主论文贡献独立自然 root 正例；它仍可作为混合 mask 导致级联污染的机制证据。下一步应筛查新的严格在线场景，而不是把这些级联事件重复计作 root。
