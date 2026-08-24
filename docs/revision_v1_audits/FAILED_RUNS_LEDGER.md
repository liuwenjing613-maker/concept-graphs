# Revision Kernel V1 failed/interrupted runs ledger

Date: 2026-08-23

This file intentionally retains failed or superseded executions. Their artifacts
remain under `/home/chenkejun/beauty/conceptgraphs/experiments/revision_v1_20260823`;
they are excluded from final aggregates by completing the frozen manifests again
with the final implementation semantics.

## F1 — Dependency traversal accidentally approached global scope

- Symptom: the first explicit traversal included 9,696 of 9,874 events.
- Cause: candidate-list appearances were treated as forward causal writes.
- Decision: candidate appearances are read-only neighbor evidence; only explicit
  association, mapping, version-parent/child, merge and relation writes propagate.
- Result after correction: an early representative closure used 224/9,874 events.

## F2 — Pre-anchor state omitted earlier events in the anchor frame

- Symptom: recorded versions triggered earlier in the same frame failed exact
  membership/geometry/feature validation.
- Cause: the prefix ended at frame `tau-1`, while the mapper had already applied
  earlier detections from frame `tau` before the anchor event.
- Decision: reconstruct those earlier decisions from the immutable event ledger,
  while keeping suffix decisions natural.
- Verification: the frame-17 diagnostic became exact for members, bbox, point
  count, CLIP feature and class histogram.

## F3 — Matrix-time candidate version was mistaken for anchor-time active version

- Symptom: `false_merge_f000005_r0014` compared candidate `v000004` against an
  object that had correctly advanced to `v000006` before its anchor.
- Cause: ConceptGraphs computes the frame candidate matrix once, then earlier
  same-frame detections advance object versions.
- Decision: preserve the requested version as evidence, but validate the latest
  same-lineage version strictly before the anchor. The resolution is emitted in
  `pre_anchor_snapshot.json`.
- Verification: the resolved `v000006` passed all six snapshot checks exactly or
  within the declared tolerance.

## F4 — Redundant per-case prefixes and interrupted CPU-heavy batches

- Symptom: early batches repeatedly rebuilt every scene prefix and several cases
  ended in `KeyboardInterrupt` during Open3D DBSCAN.
- Decision: sort cases by anchor time, maintain one incremental prefix cache per
  persistent shard, and terminate child shards when the parent is interrupted.
- Status: interrupted artifacts and logs remain; they are not final metrics.

## F5 — Typed closure was safe historically but not atomic at the current head

- Case: `wrong_membership_f000051_r0000`.
- Symptom: membership and bbox recovered to 1.0, but entity
  `73a395d3-4755-4744-8127-13504a0923ea` partially overlapped the replay write
  set. R11 rejected the transaction and relation state remained different.
- Cause: the historical closure omitted two observations now owned by the same
  current entity.
- Decision: perform bounded dynamic expansion using only the current corrupted
  head (never clean/GT ownership), then rerun and recheck R11.
- Verification: one entity and two observations were added (262 to 264); member
  F1 0.71994 to 1.0, bbox IoU 0.51657 to 1.0, relation false to true, and all
  runtime invariants passed.

## F6 — The first aggregate hid the effective locality tail

- Symptom: per-case artifacts recorded event, observation and dynamically expanded
  closure fractions, but the aggregate exposed only the mean event fraction.
- Risk: a small mean could hide a high p95/max observation scope or rare head-state
  expansion, making the locality claim look stronger than the evidence.
- Decision: retain the compatibility mean and add mean/p50/p95/max distributions
  for event, initial-observation and effective-observation fractions, plus expanded
  case and observation totals.
- Effect: reporting only; replay states and pass/fail outcomes are unchanged. Final
  aggregates are regenerated after every frozen batch with this schema.

## F7 — Bbox roundoff was misclassified as real corruption damage

- Case: `false_split_f000103_r0013`.
- Symptom: all methods had exact membership and relation state; bbox IoU was
  `0.9999999884982824`, only about `1.15e-8` below one. The original `1e-12`
  comparison labeled this as geometry damage and then `CONSTRAINT_INSUFFICIENT`.
- Cause: the damage classifier used an exactness tolerance appropriate for a
  discrete member partition on a floating-point Open3D geometry metric.
- Decision: declare separate tolerances (`member_f1_atol=1e-12`,
  `bbox_iou_atol=1e-6`) and use them consistently for damage, improvement,
  non-regression, method equivalence and aggregate recovery eligibility.
- Auditability: stored branch states are immutable. A dedicated reclassifier reads
  their method metrics, records before/after labels, and regenerates the aggregate;
  it never reruns or edits a replay state.

## F8 — Directory globbing could contaminate a frozen aggregate

- Symptom: the initial aggregator consumed every `*/benchmark_metrics.json`
  below an output root.
- Risk: the required retention of failed/diagnostic runs conflicts with that
  behavior; an extra diagnostic directory could silently enter the headline count.
- Decision: when `manifests/cases.json` exists, aggregation, reclassification,
  same-frame auditing and manual-review sampling now follow its ordered case IDs
  exclusively. Missing frozen cases and ignored extra metric directories are
  emitted as `selection_integrity` fields.
- Verification: a regression test creates one missing frozen case and one extra
  diagnostic case and checks both are reported without contaminating selection.

## F9 — The same frozen-selection risk existed in secondary gates

- Symptom: the primary aggregate, same-frame audit and review sampler were fixed,
  but the global-reference and live-fidelity aggregators still globbed every result
  directory below their output roots.
- Risk: retained diagnostic or superseded comparisons could inflate a secondary
  gate even though its `cases.json` had been frozen before outcomes.
- Decision: both secondary aggregators now follow their ordered frozen case IDs,
  emit missing and ignored-extra IDs, and reject duplicate IDs. A regression test
  exercises primary, global-reference and live-fidelity selection independently.
- Effect: no replay state is changed. Existing clean output roots with exactly the
  frozen cases retain the same metrics; contaminated roots become explicit.

## F10 — Majority-vote entity alignment could hide a relation error

- Symptom: UUID-independent relation evaluation mapped each candidate entity to
  whichever clean entity owned most of its observations. Two split candidates could
  therefore collapse onto one clean endpoint, and an unchanged runtime UUID could
  mask a changed member partition.
- Risk: relation-state recovery could be reported as exact even when its endpoint
  partition was structurally wrong.
- Decision: evaluator alignment now requires an exact observation-member partition.
  Exact partitions remain robust to independent runtime UUIDs; every non-exact
  candidate is placed in an evaluator-only unaligned namespace so incident edges
  cannot match accidentally. Raw-ID metrics remain separate diagnostics.
- Verification: a regression test proves that UUID-only changes pass while a split
  partition retaining a clean-looking UUID fails. Stored branch states will be
  re-evaluated before the final aggregates are reported. The transparent
  reclassifier now refreshes all relation metrics from immutable stored branch
  states and recomputes affected ablation labels; it still never edits replay state.

## F11 — Runtime R6 initially proved presence, not execution semantics

- Symptom: the first runtime verifier checked that each constraint UID appeared in
  a decision trace, but did not independently check that `FORCE_TARGET`/`FORCE_CREATE`
  was the action actually applied or that a forbidden target was avoided. It also
  did not compare the summarized objects with the state membership map.
- Risk: a broken executor could emit the right constraint UID while committing the
  wrong action and still pass the production-only gate.
- Decision: R2 now checks object/member consistency and known observation refs. R6
  checks applied target/create semantics, forbidden targets, unknown UIDs, deferral,
  and rejects primitives not executable at the association boundary. No clean owner
  or expected final grouping is used.
- Verification: an adversarial unit test mutates both an object member list and a
  forced target and requires R2/R6 to fail. The posthoc auditor recomputes runtime
  verification from stored branch states before regenerating final metrics.

## F12 — `math.isclose` had an undeclared relative tolerance

- Symptom: method-equivalence checks supplied the declared absolute tolerances but
  left Python's default `rel_tol=1e-9` enabled.
- Risk: near one, the effective membership equivalence threshold was about `1e-9`
  rather than the reported `1e-12`, potentially overstating ablation equivalence.
- Decision: both membership and bbox equivalence now set `rel_tol=0`; only the
  published dimension-specific absolute tolerances apply.
- Verification: a regression test requires a `5e-10` member-F1 difference to be
  non-equivalent while retaining a `5e-7` bbox-IoU difference within `1e-6`.

## F13 — Global clean parity compared AABB and OBB extent representations

- Symptom: the first room0 clean global replay reproduced all 3,779 memberships,
  72 objects, every raw point digest, CLIP feature, class histogram, bbox center,
  200-frame decision kind, postprocess count, source hash and 49-edge relation state,
  yet failed all 72 object payload rows on `bbox_extent`.
- Root cause: frozen `bbox_np` stores eight oriented-box corners. The first gate took
  their axis-aligned min/max extent and compared it with Open3D's local oriented-box
  `extent`; these are different coordinate representations of the same box. This is
  confirmed by exact point sets and centers, not assumed from a passing headline.
- Decision: compare the two eight-corner sets directly with a permutation-invariant
  symmetric Hausdorff distance at the existing `2e-3` tolerance. This validates
  size and orientation without conflating OBB-local and world-axis extents.
- Verification: a unit test permutes identical corners (must pass) and translates
  them by `0.01` (must fail). The full clean replay gate is rerun; the failed first
  artifact remains retained as evidence.

## F14 — Exact empty relation states needed an explicit denominator

- Symptom: method summaries reported relation exact rate but not whether a relation
  stream was informative. office0's empty/absent stream can be exactly equal without
  testing relation reconstruction at all.
- Risk: an exact rate of 1.0 could be read as evidence equivalent to room0's 2,425
  observations and 49 final directed edges.
- Decision: every method aggregate now reports informative relation count/rate next
  to exact rate. Final tables label office0 relation evidence noninformative and use
  room0 only for relation-recovery claims.

## F15 — Geometry diagnostics mixed representations even when IoU did not

- Symptom: `bbox_iou_to_clean` correctly used point-cloud AABBs, but the accompanying
  center/extent errors used serialized-corner AABB extents on the clean side and
  Open3D OBB-local extents on replay branches. Exact clean replay therefore showed
  IoU 1.0 alongside a meaningless nonzero extent error.
- Decision: IoU, center error and extent error now consistently use `aabb_min/max`.
  The transparent posthoc auditor refreshes every method's membership, geometry,
  relation and cost metrics from immutable stored branch states before classification.
- Verification: a regression case with identical AABBs but deliberately different
  `bbox_extent` representations must report IoU 1 and zero center/extent error.

## F16 — Secondary aggregate fixture lacked the new informativeness field

- Symptom: after adding relation-informative counts, the frozen-selection regression
  fixture raised `KeyError` because its deliberately minimal synthetic global row
  omitted `local_vs_global.relation`.
- Decision: production aggregation treats absent legacy informativeness as false,
  while the current fixture explicitly provides the field. This preserves backward
  readability without converting missing evidence into informative evidence.
- Verification: the targeted aggregate/global/runtime test group is rerun before
  the full suite.

## F17 — A mid-frame snapshot crossed its own event watermark

- Trigger: the independent office0 same-frame audit found three apparent decision
  differences in `false_merge_f000015_r0008`. Inspection showed that two were
  observations already materialized in the pre-anchor snapshot and the third was
  the anchor itself being rematched against objects created earlier in frame 15.
- Root cause: suffix selection used `frame >= anchor_frame`, not the immutable
  association-event watermark. It therefore replayed some earlier same-frame
  observations twice. At the anchor it also recomputed a new similarity matrix,
  although ConceptGraphs freezes the whole frame's matrix before applying any of
  that frame's detections.
- Blast radius: the stricter retained pre-fix audits found 20 duplicate-prefix
  observations across 16/30 room0 cases and 14 across 9/30 office0 cases. The
  original recorded-vs-recomputed decision differed at one office0 anchor. The
  source-marker mismatches in those retained audits are diagnostic schema changes,
  not additional decision divergences.
- Decision: suffix rows must have association sequence strictly greater than the
  snapshot watermark. The anchor frame now rehydrates the exact frozen aggregate
  similarity rows and recorded frame-start decisions from immutable evidence;
  objects created earlier in the frame are deliberately ineligible because they
  did not exist in that matrix. Missing/invalid matrix evidence hard-fails instead
  of falling back to a fresh matcher result.
- Adversarial verification: the affected office0 case changed from a recomputed
  merge into the earlier same-frame object to the recorded native `CREATE_OBJECT`;
  the injected historical action remained the specified wrong target, and the
  sparse `CANNOT_LINK` compiled to `FORCE_CREATE`. Its diagnostic rerun passed with
  zero pre-watermark rows and zero frozen-decision mismatches.
- Retention and rerun: the complete pre-fix primary roots and room0 global-reference
  root were renamed with `_pre_f17_same_frame_boundary`; no evidence was deleted.
  All 60 frozen primary cases and all 12 global references are regenerated before
  final reporting, even though only 25 primary cases exposed duplicate rows.

## F18 — Live membership fidelity ignored simulator-only observations

- Symptom: the first live-versus-simulator comparator evaluated member assignments
  only over observations present in the live mapping. A simulator-only observation
  therefore contributed neither a mismatch nor a coverage failure.
- Risk: an incomplete live run could appear partition-equivalent even when the
  simulator retained extra observations, or the reverse discrepancy was otherwise
  hidden by an asymmetric evaluation scope.
- Failed intermediate fix: the comparator first passed the union as
  `observation_scope` to the existing benchmark metric. Independent review caught
  that the metric deliberately intersects any supplied scope with its clean/reference
  owner set, so the extra simulator observation was still discarded. The first test
  covered only the scope helper and would have missed that second contraction.
- Decision: live fidelity now uses a dedicated symmetric partition scorer over the
  union, because neither independently generated run is the complete reference
  universe. A missing assignment or duplicate ownership on either side is an
  explicit mismatch; entity UUIDs remain irrelevant.
- Verification: the adversarial test exercises the final scorer, supplies an extra
  simulator observation, and requires a non-exact score plus an explicit
  `missing_in_live` row. A second test proves UUID-independent equality and duplicate
  rejection. All live comparison artifacts are generated only after this correction.

## F19 — Unscoped repository pytest collected executable legacy scripts

- Symptom: an unqualified `pytest -q` stopped during collection before running the
  suite. `conceptgraph/scripts/gpt4v_test.py` reads a developer-specific image at
  import time, `lava_15_test.py` imports an unavailable `transformers` package, and
  `tests/test_general_utils.py` requires unavailable `supervision` in the lightweight
  audit interpreter.
- Interpretation: this is not a passing V1 run and is retained as a failed command.
  The two executable scripts are unrelated legacy utilities whose names match
  pytest's default pattern; installing their heavy/model dependencies would not test
  the revision change.
- Verification scope: the complete `tests/` tree excluding only the independently
  identified environment-dependent `test_general_utils.py` passed 114 tests with one
  skip. The stricter V1-named subset passed 51 tests with one skip. The final report
  records both the scoped pass and the unscoped collection limitation.

## F20 — Snapshot validation did not require every requested seed to resolve

- Symptom: the validation gate required at least one validated version and all
  returned validation rows to pass, but did not reject a nonempty `skipped` list.
  With two requested dependency seeds, one valid seed could therefore conceal one
  missing, post-anchor, or otherwise unresolved seed.
- Risk: an incomplete causal snapshot could enter suffix replay despite the declared
  hard-fail policy for missing historical evidence.
- Decision: every requested seed must produce a resolution row, `skipped` must be
  empty, and every unique resolved version must validate. Multiple requests may
  legitimately resolve to one shared active version after lineage convergence.
- Blast-radius audit: all 60 retained pre-F17 cases and every corrected case already
  produced at audit time had zero skipped seeds and equal requested/resolved counts;
  the stricter predicate is therefore outcome-equivalent for the frozen population.
- Verification: one regression test requires a single skipped seed to fail despite
  another passing row; a second permits two complete resolutions sharing one valid
  active version.

## F21 — Local/global partition parity used a one-sided observation universe

- Symptom: the global-reference evaluator called the benchmark membership metric
  with global observations as its scope. A local-only observation was therefore
  outside the reference owner set and could not lower the reported F1.
- Risk: local/global parity could be reported exact despite an asymmetric final
  observation set. The same metric property caused F18, but this second call site
  affected a different scientific gate and required an independent fix.
- Decision: a reusable symmetric, UUID-independent partition scorer now evaluates
  the union of both observation sets, detects missing and multiply owned observations
  on either side, and exposes an explicit `partition_exact` result. Both live fidelity
  and local/global reference gates use it; geometry scope also uses the union.
- Retention: the six pre-F17 room0 global artifacts remain under their labeled
  pre-fix root. No final global reference had started when this was found, so all 12
  final references are generated only with the symmetric gate.
- Verification: the F18 adversarial tests exercise the shared implementation and
  require simulator-only observations and duplicate ownership to fail while allowing
  UUID-renamed but identical partitions.

## F22 — Optional live precheck was stopped after server oversubscription

- Trigger: while both two-worker primary batches and the independent live mapper
  were active, an extra room0 live/simulator precheck raised server load to about
  371 on 224 logical CPUs and began slowing the higher-priority frozen runs.
- Decision: the optional precheck was interrupted rather than treating maximum
  concurrency as an objective. It had not reached comparison output writing, the
  diagnostic directory remained empty, and no partial row can enter an aggregate.
- Follow-up: room0/office0 primary and live mapping keep priority. Final global
  references are reduced to one outcome-blind case per failure type per scene
  (six total) and are expanded only if those staged checks disagree.

## F23 — First reduced-live manifest overstated its freeze timing

- Symptom: the first six-case staged manifest reused the generic field
  `frozen_before_new_live_outcomes=true`, although several independent live maps had
  already completed. No live/simulator comparison score had been generated, and all
  six IDs were already present in the original pre-live 10-case manifest, but the
  literal timing claim was still too strong.
- Decision: that reduced manifest is retained as superseded. The final staged
  manifest explicitly records that it was derived after some live maps existed but
  before any fidelity comparison outcome; it must validate that every selected ID
  was predeclared by an outcome-blind manifest genuinely frozen before live mapping.
- Verification: tests reject a staged case absent from the source manifest and accept
  only a unique, unscreened, pre-live source selection.

## F24 — The original 10-case live orchestration was intentionally curtailed

- Trigger: after the evaluation scope was reduced to two scenes and staged expensive
  checks, continuing all remaining live maps would add cost before the method-level
  decision. The original manifest had four completed/reused false-merge maps and one
  active false-split map.
- Procedure: the parent scheduler was paused so it could not launch another case;
  the already active false-split child was allowed to reach
  `MAP_COMPLETED_EVIDENCE_VALID`, then the parent was interrupted cleanly. No next
  child was launched and no incomplete result entered an orchestration artifact.
- Final scope: the formal staged-six manifest reuses those five predeclared maps and
  runs one predeclared wrong-membership map. The superseded 10-case manifest and logs
  remain retained, but only the staged-six root is compared and aggregated.

## F25 — Runtime headline omitted snapshot reconstruction cost

- Symptom: primary and local/global aggregates labeled `runtime_ms` without making
  clear that the field covered suffix replay only. Each branch also stored a
  cumulative pre-anchor snapshot reconstruction time, but it was absent from the
  headline distribution and speed ratio.
- Risk: a suffix-only local/global ratio could be misread as end-to-end cold-start
  acceleration. Conversely, summing cumulative prefix times across cases would
  double-count the shared incremental prefix cache.
- Decision: artifacts retain the compatibility field but explicitly label it
  suffix-only, add snapshot and non-amortized cold `snapshot+suffix` distributions,
  and report both suffix-only and cold local/global ratios. The current ledger does
  not separate incremental cache time from per-case same-frame prefix time, so no
  fabricated amortized ratio is reported.
- Verification: a regression test with 20 ms suffix and 80 ms snapshot requires a
  100 ms cold total. Final prose reports both available bounds and identifies exact
  cache-amortized runtime as future instrumentation, not a measured result.

## F26 — Frozen global selection silently ignored a reduced `per_type`

- Symptom: preparing office0 with `--per-type 1` printed `prepared: 6` because an
  earlier outcome-free six-case (`per_type=2`) manifest already existed. The reuse
  path checked only that both manifest files existed, not that the frozen parameters
  matched the current request.
- Risk: a staged scope reduction could silently run twice the declared global cases,
  and a future caller could unknowingly associate results with the wrong primary
  manifest.
- Decision: reuse now hard-fails unless `per_type`, the resolved primary manifest
  path, and the ordered case IDs all match. The earlier office six-case
  preparation is retained without outcomes; the final staged office selection uses
  a distinct three-case root.
- Verification: a unit test accepts the identical frozen request and rejects a
  change from two to one case per type.

## F27 — Primary collateral gate used a one-sided observation universe

- Symptom: the affected-object score intentionally evaluated only clean affected
  observations, while the old global membership diagnostic also inherited the clean
  observation universe. A candidate-only observation or duplicate assignment could
  therefore be absent from the repair PASS decision.
- Risk: a locally repaired case could be counted as successful despite creating
  collateral membership outside the affected set.
- Decision: the global diagnostic now compares the union of both observation
  universes, is invariant to entity UUIDs, and fails on either-side missing
  observations, duplicates, or any partition mismatch. `collateral_safe` is now an
  explicit PASS gate and `COLLATERAL_DAMAGE` is part of the failure taxonomy.
- Verification: an adversarial regression keeps affected-only F1 at 1.0 while adding
  a candidate-only observation; the symmetric global partition and final PASS must
  both fail. All stored cases are reclassified from their branch states before final
  aggregation.

## F28 — Frozen primary selection reuse lacked parameter/ID validation

- Symptom: when both primary manifest files already existed, the batch runner reused
  them without checking the requested scene, seed, base ledger, per-type count, global
  flags, or whether `cases.json` still matched the selected IDs and order.
- Risk: a resume command with changed scope could be labeled as the new request while
  silently aggregating the old frozen population, or a partially edited cases file
  could invalidate reproducibility.
- Decision: frozen primary reuse now hard-fails on any request mismatch, duplicate
  case UID, case-count mismatch, or ordered case-ID mismatch.
- Verification: regression tests accept an identical request, reject a changed
  per-type count, and reject a reordered case file.

## F29 — No-oracle AST audit missed string-key access

- Symptom: the guard rejected forbidden variable and attribute identifiers, but a
  runtime read such as `state["clean_owner"]` represented the same leakage as an AST
  string constant and would not be reported.
- Risk: a refactor from attribute access to a mapping payload could silently bypass
  Gate A without changing the leaked information.
- Decision: forbidden exact identifiers are now checked in names, attributes,
  function arguments, keyword arguments and string constants. Runtime source remains
  separately isolated from benchmark/evaluator imports.
- Verification: an adversarial subscript-key regression must fail on
  `clean_membership`, while the five declared runtime modules retain zero violations.

## F30 — Heterogeneous positive conflicts could raise the wrong exception

- Symptom: formatting two conflicting target tuples that mixed `None` and string
  fields used Python's default tuple sort. The diagnostic path could raise a
  `TypeError` before the declared `ConstraintConflictError` was emitted.
- Risk: the constraint still did not execute, but callers would lose the stable hard
  conflict contract and receive a misleading infrastructure error.
- Decision: conflict diagnostics use an explicit normalized sort key; execution
  semantics are unchanged.
- Verification: an adversarial lineage-target plus origin-target pair must raise
  `ConstraintConflictError` with the multiple-positive-target diagnosis.

## F31 — Duplicate observations inside one entity were normalized away

- Symptom: the partition evaluator converted each entity's member list to a set
  before counting duplicates. It caught one observation assigned to two entities,
  but not the same observation repeated twice within one entity payload.
- Risk: the strict live/global parity gate claimed to reject every duplicate yet had
  one malformed-payload blind spot; a generator-valued membership could also be
  consumed by the old emptiness check.
- Decision: indexing now records counts from the original iterable while retaining a
  set only for partition comparison. Iterables are materialized exactly once.
- Verification: a same-entity repeated observation must set the duplicate list and
  make `partition_exact=false`.

## F32 — Duplicate relation rows were collapsed by dictionary indexing

- Symptom: relation metrics keyed edges by `(source, relation, target)` and therefore
  silently retained only the last support value when a backend emitted the same edge
  row more than once.
- Risk: a malformed relation payload could pass exact set/support comparison even
  though the structural backend output was not canonical.
- Decision: both reference and candidate edge multiplicity are counted before map
  comparison; any duplicate makes relation-state exactness false and is emitted in
  the diagnostics.
- Verification: an otherwise exact UUID-independent relation state with one repeated
  candidate edge must fail and report one duplicate edge.

## F33 — `KEEP_NATURAL` accidentally kept the injected historical target

- Case: `room0/false_merge_f000015_r0011`.
- Symptom: native association at the anchor was `CREATE_OBJECT` (`None`), the injected
  false merge targeted object 7, and `CANNOT_LINK` correctly excluded object 7. Because
  another eligible candidate existed, the old resolver selected object 12; final
  member F1 stopped at `0.993216` and relation state was not exact.
- Cause: the executor passed the injected historical match as the constraint engine's
  `natural_match`, then interpreted `KEEP_NATURAL` as “keep historical.” This conflated
  the native counterfactual decision with the deliberately corrupted branch.
- Decision: constraint resolution now receives the actual native matcher result.
  `KEEP_NATURAL` applies that result, while `NO_CONSTRAINT` alone preserves the
  historical branch. Positive force-target/create semantics are unchanged.
- Verification: pure adversarial tests distinguish native create from an injected
  historical target; only the affected frozen case is rerun, followed by full stored-
  state reclassification, same-frame audit and aggregate regeneration. All other 19
  false-merge traces already used `FORCE_CREATE` and are state-hash identical to their
  scene references, so rerunning them would add cost without new information.
- Retention and diagnostic: the pre-fix case is preserved at
  `room0_primary_diagnostics_pre_f33/false_merge_f000015_r0011`. The corrected local
  state also matched an outcome-selected, headline-excluded same-constraint global
  diagnostic over all 3,779 observations, exact geometry and informative relations;
  its root is `room0_global_reference_diagnostic_f33`.

## F34 — Same-frame audit depended on a top-10 diagnostic list

- Case: corrected `room0/false_merge_f000015_r0011`, later observation
  `f000015_r0015`.
- Symptom: the replay trace used recorded target index 25 and the exact immutable
  origin `f000011_r0011`, but `natural_candidates` intentionally retained only the
  ten highest-score diagnostics. The audit failed to find index 25 and reported a
  null target mismatch.
- Risk: a correct frame-start recorded decision could be rejected solely because its
  candidate was outside a presentation/debugging truncation.
- Decision: same-frame fidelity now compares the recorded target version's immutable
  origin against `natural_target_origin_obs_uid`, which is stored directly in every
  decision row. The top-10 list remains diagnostic only.
- Verification: an adversarial trace with natural index 25 and an empty candidate
  list passes when origins match and fails when they differ. The final 60-case
  same-frame audit is regenerated from stored traces without rerunning maps.

## F35 — A positive anchor repair did not persist across a duplicate lineage

- Case: `human6_office0_false_split_51aaf9ba`.
- Symptom: the first human-error pilot forced the frame-32 anchor into the
  existing sofa, but frame 35 naturally scored that sofa only `0.803134 < 1.2`,
  created a new entity, and rebuilt the false split. Native, Natural and Sparse
  were therefore all endpoint-wrong even though the anchor trace said
  `FORCE_TARGET`.
- Cause: `ASSIGN_OBSERVATION` was implemented as an anchor action only. The
  persistent mode did not carry the created duplicate lineage into later
  association decisions.
- Decision: when a positive anchor's recorded action is `CREATE_OBJECT`, derive
  one immutable source-lineage to target-lineage redirect. Later observations
  carrying that source lineage resolve to the unique active target regardless
  of score. This is lineage-level causal closure, not an enumerated final-member
  list. `ANCHOR_ONLY_REPAIR` remains unchanged.
- Pre-run audit finding: the frozen primitive's active interval ends at the
  anchor event. That interval scopes the direct action; it is not the lifetime
  of a persistent derived redirect. This distinction was corrected before the
  V2 execution.
- Verification: the retained V1 root fails. V2 records 130 lineage matches but
  only one changed default decision, precisely the frame-35 create; the two
  reviewed groups become one atomic owner with zero outside-observation changes.

## F36 — `CREATE_INSTANCE` guarded postprocess merges but not associations

- Case: `human6_room0_false_merge_06525b4b`.
- Symptom: V1 rejected the first cross-instance postprocess merge at frame 139.
  Subsequent observations from the protected lineage naturally associated into
  the other outlet, contaminating its inferred lineage. A later merge then
  appeared same-side and was accepted, so the endpoint remained falsely merged.
- Cause: instance persistence was enforced only in object-object postprocessing,
  not at the detection-object association boundary.
- Decision: classify each future observation and active candidate by immutable
  provenance lineage. A candidate is forbidden exactly when one side, but not
  the other, contains a protected `CREATE_INSTANCE` lineage. If the default is
  forbidden, choose the highest-scoring strictly-threshold-eligible same-side
  candidate; if none exists, create a deterministic same-lineage object.
  Unknown observation provenance is not treated as negative evidence.
- Verification: V2 records six association overrides and 24 postprocess vetoes
  for the outlet case. The independent table-part case needs no association
  override and retains six postprocess vetoes. Both are endpoint-correct,
  collateral-safe and runtime-invariant-clean; no similarity threshold was
  relaxed.

## Methodological finding retained, not "fixed" away

In the controlled external-override benchmark, natural recomputation from the
pre-anchor evidence often equals sparse repair. This is expected because the raw
evidence and native matcher were not themselves wrong. It limits the causal claim:
the oracle-sparse constraint is sufficient relative to replaying the recorded
wrong action, but is often not necessary once the correct anchor is known and the
native matcher is rerun. Final reporting must keep this ablation visible.
