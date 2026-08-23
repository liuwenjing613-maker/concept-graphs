# Revision Kernel V1 branch manifest

- V1 branch: `exp/ali-my-revision-kernel-v1`
- V1 base: `900f117557b9fea2e0924165b5e98917bc88afd9`
- Base branch: `exp/ali-my-revision-kernel-v0`
- Integration reference only: `ali-my-full @ 77fbd907c2775ade5db824b74eef2cf961613e7e`
- Selected cherry-picks: none
- Development scenes: `room0`, `office0`
- Frozen holdout scenes: `office1`, `office2`, `office3`, `office4`, `room1`, `room2`
- Primary selection seed: `20260823`
- Primary sets must be frozen before observing replay outcomes.
- Relation-sensitive cases are `STRESS_SET` only and must not enter the primary population.
- This V1 decision run is intentionally limited to the two development scenes;
  the six holdout scenes remain untouched until the method passes staged gates.
- Global-reference stage: one outcome-blind case per failure type per development
  scene (`3 x 2 = 6`), expanded only after a mismatch.
- Live-fidelity stage: frozen room0 counts `FALSE_MERGE=4`, `FALSE_SPLIT=1`,
  `WRONG_MEMBERSHIP=1`; the earlier 10-case manifest is retained but not aggregated.
- Final decision rule: do not expand global/live/holdout scope after exact staged
  gates unless a method change or a genuine mismatch creates new information gain.

## Frozen experiment roots

- room0 ledger: `/home/chenkejun/beauty/conceptgraphs/data/Replica/room0/exps/ali_my_validity_room0_full_200f_e6b0f17_20260820`
- office0 ledger: `/home/chenkejun/beauty/conceptgraphs/data/Replica/office0/exps/ali_my_validity_office0_full_200f_20260820`
- room0 non-empty relation stream: `/home/chenkejun/beauty/conceptgraphs/experiments/revision_edges_make_true_room0_20260823`
- V1 result root: `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823`
- Final server report: `/home/chenkejun/beauty/ALI_MY_REVISION_KERNEL_V1_IMPLEMENTATION_EVALUATION_20260823.md`

## Base source hashes

| File | SHA-256 at V1 base |
|---|---|
| `conceptgraph/revision/replay.py` | `03e5041bc9d185aa1b69c8661fa36a52ed9d012434b0a88452b5e52fec8ee692` |
| `conceptgraph/revision/cases.py` | `c47cd490f7e3b149209df53b196f970a95f90b800774ac9f0dc22a41b29f2410` |
| `conceptgraph/revision/experiment.py` | `1f67efb232b6cc627278f1a6cd9041da5d948485ae8dc0cc195321fb2cc5fbe5` |
| `conceptgraph/revision/relations.py` | `de407f033f93899d9da030efeb7dbf11d8dc087d600d54fd86eabe2e4c6a9721` |

## Scope rule

V1 production replay modules may consume immutable evidence, sparse constraints, a
snapshot watermark, and the current corrupted head. They may not consume frozen
clean final ownership, affected final groups, GT identity, or expected final
membership. Benchmark-only compilers and evaluators remain physically separate.
