# Phase 3 postmortem — pre-anchor snapshot and suffix-local replay

## What changed

- Added exact pre-anchor reconstruction, including earlier detections in the anchor
  frame, latest same-lineage version resolution strictly before the anchor, and a
  strict event-sequence watermark.
- Added typed dependency traversal, bounded current-head atomic expansion, suffix
  replay from the snapshot and overlay verification.
- Added natural global clean parity, same-constraint global references, symmetric
  local/global partition comparison, first-divergence tooling and explicit suffix,
  snapshot and cold runtime fields.

## What passed

- Snapshot validation passed 60/60 primary cases with every requested seed resolved,
  zero skipped versions and no approximation fallback.
- The final same-frame audit covered 43 room0 and 46 office0 decisions. It found zero
  watermark crossings and zero recorded decision/origin mismatches.
- Natural full replay reproduced both frozen maps: room0 3,779 observations/72 objects
  and office0 1,560 observations/29 objects, including raw object payload and
  postprocess schedule. room0's 49 directed relation edges were also exact.
- The six outcome-blind global references (one case per failure type per scene) all
  passed runtime invariants, exact union member partition, bbox IoU at least 0.999 and
  relation equality. All three room0 references were relation-informative; office0's
  three empty relation comparisons were correctly marked noninformative.
- The outcome-selected F33 diagnostic was excluded from the headline six and also
  passed over all 3,779 observations, exact geometry and informative relation state.

## What failed

- Early snapshot implementations crossed same-frame boundaries or validated the
  matrix-time rather than anchor-time version (F2, F3 and F17). Old roots are retained;
  the final audit has zero such errors.
- One room0 and one office0 pattern required current-head expansion. Final aggregates
  report three expanded cases in total rather than hiding them in the mean.
- Cold runtime did not establish an end-to-end speedup. room0 formal references had
  cold local/global ratio p50 `1.186`, p95 `1.237`, max `1.243`; office0 had p50
  `0.962`, p95 `1.041`, max `1.050`.
- The exact cache-amortized cost cannot be recovered from current artifacts because
  they store cumulative prefix time, not per-case incremental cache and same-frame
  components.

## Unexpected observations

- Suffix-only replay was consistently cheaper than global replay on the six formal
  references: room0 ratio p50 `0.522`, p95 `0.533`, max `0.534`; office0 p50 `0.386`,
  p95 `0.479`, max `0.489`. Snapshot construction, not suffix replay, is now the
  dominant systems problem.
- Locality remained bounded but scene-dependent. room0 effective observation fraction
  was p50 `3.96%`, p95 `8.46%`, max `10.19%`, with two expanded cases/31 observations.
  office0 was p50 `8.69%`, p95 `16.03%`, max `16.15%`, with one expanded case/131
  observations.
- The post-F33 same-frame audit initially reported one mismatch only because it used a
  top-10 diagnostic candidate list to look up recorded index 25. Direct immutable
  origin comparison proved the state was correct and removed this audit tautology.

## Possible leakage

Snapshot construction reads only immutable observations and object versions strictly
before the anchor watermark. Typed closure and dynamic expansion read historical
dependencies and the current corrupted head, not clean ownership. Clean states and
global references are evaluator-only. The source guard enforces this separation.

## Possible selection bias

- Primary cases are the frozen outcome-blind 60. Formal global references use the
  first frozen case per type and were selected before global outcomes.
- With only one reference per type, the selected false-split cases self-healed; the
  damaging wrong-membership and false-merge cases still exercise repaired quality.
  This limitation is reported rather than repaired by outcome-based replacement.
- The F33 global diagnostic is explicitly outcome-selected and excluded from all
  formal counts and runtime summaries.

## Known limitations

- Only two development scenes were evaluated; six named holdouts remain untouched.
- Relation state is rebuilt with the correct baseline interface but still scans the
  full frame stream. Phase 6 local relation replay is not implemented or claimed.
- The current cold field is a conservative non-amortized upper bound. Exact shared
  cache cost requires new incremental timers before any paper speedup claim.
- No clean-negative harm, blind anchor, online rebase or frame-boundary live commit
  result is present.

## First divergent cases

- F17 first exposed a same-frame watermark error; corrected roots pass 89/89 audited
  same-frame decisions.
- F33 exposed native/historical decision conflation in
  `room0/false_merge_f000015_r0011`; the targeted correction and global diagnostic are
  exact.
- There is no remaining divergence in the formal six local/global references.

## Decision: GO

The two-scene evidence supports dependency-bounded replay quality and a real
suffix-compute reduction. End-to-end/runtime and local-relation claims remain FIX:
instrument incremental snapshot caching and implement a local relation backend before
expanding to holdouts.
