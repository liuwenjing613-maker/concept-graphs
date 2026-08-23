# Revision Kernel V1 manual review

Date: 2026-08-23
Seed: `20260823`
Rule: sample five reported successes and five failed/deferred outcomes per scene
from the frozen primary manifest. Outcome is used only to form the review strata;
it never changes the evaluation population or headline aggregate.

## room0

Manifest:
`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/room0_primary/random_review_manifest.json`

The manifest contained 24 available successes and 6 available failed/deferred
outcomes, and selected 5 + 5. For every selected case I opened the incident,
constraint, snapshot, typed dependency, corruption trace, replay trace, relation
rebuild, runtime verification and benchmark metrics. The checks below are therefore
artifact-level manual review, not a second automated pass/fail classifier.

| Case | Review stratum | Corrupted member F1 | Sparse member F1 | Corrupted bbox IoU | Sparse bbox IoU | Expansion | Manual finding |
|---|---:|---:|---:|---:|---:|---:|---|
| `false_merge_f000144_r0019` | success | 0.688096 | 1.000000 | 0.608158 | 0.999999 | 0 | One anchor intervention; membership and informative 49-edge relation state recovered. Residual bbox roundoff is below the declared `1e-6` outcome tolerance. |
| `wrong_membership_f000183_r0008` | success | 0.992234 | 1.000000 | 0.996571 | 1.000000 | 0 | Low-severity but real membership/geometry damage; sparse target restored the clean partition. |
| `false_merge_f000075_r0018` | success | 0.965354 | 1.000000 | 1.000000 | 1.000000 | 0 | Structural membership damage with no geometry loss; repair did not regress geometry or relation state. |
| `false_split_f000181_r0014` | success | 1.000000 | 1.000000 | 0.695291 | 1.000000 | 29 obs | Membership alone would have hidden the damage. Geometry and relation differed; current-head atomicity correctly expanded the closure by 29 observations before commit verification. |
| `false_merge_f000005_r0014` | success | 0.744703 | 1.000000 | 0.995653 | 1.000000 | 0 | Strong partition damage recovered with exactly one sparse primitive and no outside-scope overlap. |
| `false_split_f000103_r0013` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | Not a repair failure: the injected split self-healed in the native suffix. Earlier `1.15e-8` bbox roundoff was correctly reclassified as numerical noise. |
| `false_split_f000179_r0012` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | No final corrupted-state damage; excluded from the damaging-case success denominator. |
| `false_split_f000003_r0001` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | Early-anchor one-event split self-healed; no causal repair credit assigned. |
| `false_split_f000057_r0013` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | No final effect after natural suffix evolution; correctly retained in the frozen population. |
| `false_split_f000145_r0000` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | Same self-healing pattern; snapshot and runtime gates still passed, but outcome remains non-success. |

Cross-case observations:

- All ten traces contained exactly one external historical intervention. Primitive
  counts matched the compiler semantics (one anchor action for split/merge cases;
  positive assignment plus wrong-target exclusion for wrong-membership cases); none
  encoded a later or final membership trajectory.
- Pre-anchor snapshot validation and all runtime invariants passed for all ten.
- All sparse results matched the reference member partition and aligned relation
  state. Raw runtime entity UUIDs were not used as the relation correctness key.
- Natural recomputation matched sparse repair in all ten, as it did in all 30 room0
  cases. This is a limitation of the external-override benchmark, not evidence that
  the constraint was necessary.
- The five non-successes are all `CORRUPTION_SELF_HEALED_NO_FINAL_EFFECT`, not
  executor crashes, invariant failures, or silently dropped cases.
- Outside the random 5 + 5 review, first divergence
  `false_merge_f000015_r0011` was manually traced from native create, through injected
  target 7, to the erroneous pre-fix target 12. After F33 it applies native create,
  reaches exact member/geometry/informative-relation state, and matches a separate
  same-constraint global diagnostic. The pre-fix artifacts remain retained.

## office0

Manifest:
`/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823/office0_primary/random_review_manifest.json`

The manifest contained 25 available successes and 5 failed/deferred outcomes and
therefore selected 5 + 5. I inspected the constraint, snapshot resolution, single
intervention trace, replay decisions, relation rebuild, runtime failures and final
metrics for every selected case.

| Case | Review stratum | Corrupted member F1 | Sparse member F1 | Corrupted bbox IoU | Sparse bbox IoU | Expansion | Manual finding |
|---|---:|---:|---:|---:|---:|---:|---|
| `false_split_f000129_r0010` | success | 1.000000 | 1.000000 | 0.456374 | 1.000000 | 131 obs | Membership alone hides severe geometry damage. One anchor assignment restored geometry; atomic current-entity handling expanded one entity (131 observations), while the effective scope remained below the full scene. |
| `wrong_membership_f000184_r0003` | success | 0.985278 | 1.000000 | 0.998843 | 1.000000 | 0 | The positive original target and negative wrong target were both explicit anchor primitives. Exactly one historical decision was overridden and the global partition stayed exact. |
| `false_merge_f000020_r0006` | success | 0.976628 | 1.000000 | 1.000000 | 1.000000 | 0 | Structural damage was real; the residual corrupted bbox difference was below the declared geometry tolerance. A single `CANNOT_LINK` restored the create action. |
| `false_split_f000074_r0000` | success | 1.000000 | 1.000000 | 0.998407 | 1.000000 | 0 | Geometry-only final damage was recovered. Snapshot, union-partition collateral gate and runtime invariants all passed. |
| `false_merge_f000032_r0001` | success | 0.976429 | 1.000000 | 0.996636 | 1.000000 | 0 | Membership and geometry recovered with one historical override and no outside-scope mutation. |
| `false_split_f000019_r0000` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | The injected event occurred exactly once but native suffix evolution removed all final damage; no repair credit was assigned. |
| `false_split_f000096_r0001` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | Self-healed temporal split; snapshot and runtime checks passed and the case remained in the frozen denominator. |
| `false_split_f000060_r0001` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | No final membership, geometry or relation damage after the native suffix. |
| `false_split_f000111_r0006` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | Correctly classified as `CORRUPTION_SELF_HEALED_NO_FINAL_EFFECT`, not an executor failure. |
| `false_split_f000117_r0015` | failed/deferred | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0 | Same self-healing pattern; no result was dropped or promoted to success. |

Cross-case observations:

- All ten snapshots resolved every requested seed with no skipped version, and all
  runtime hard-invariant failure lists were empty.
- Every corruption trace contained exactly one live-style historical intervention.
  Constraints contained only the anchor observation, immutable lineage/origin
  reference and original event evidence; no later/final member trajectory appeared.
- All ten sparse outputs passed the symmetric full-observation collateral gate.
- office0 has no informative relation edges, so its relation exactness is a structural
  empty-set check only; room0 is the relation-informative development scene.
- Natural recomputation equaled sparse repair in all 30 office0 cases. The sparse
  primitive is sufficient relative to retaining the injected wrong action, but this
  controlled external-override benchmark does not demonstrate necessity.
- The five non-successes were all self-healing `FALSE_SPLIT` cases. There were no
  crashes, deferrals, snapshot mismatches, invariant failures or collateral failures.
