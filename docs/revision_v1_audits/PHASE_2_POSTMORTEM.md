# Phase 2 postmortem — one-event temporal corruption fidelity

## What changed

- Replaced static final-partition corruption as the primary benchmark with one native
  association-event override followed by unconstrained suffix evolution.
- Added independent live reruns, a global temporal simulator, decision/target-origin
  comparison, raw map-payload parity and frozen-manifest aggregation.
- Reduced the formal fidelity set from the earlier ten-case plan to a predeclared
  staged six: four false merges, one false split and one wrong membership. This was a
  deliberate method-iteration decision, not outcome filtering.

## What passed

- All 6/6 independent live maps completed with exactly one recorded injection.
- All 6/6 comparisons passed all five gates: injection count/identity, downstream
  decision and target-origin trace, UUID-independent final member partition, raw
  object payload, and denoise/filter/merge schedule.
- Every case had member F1 `1.0`, zero decision-kind mismatch and zero target-origin
  mismatch between live and simulator. Raw object parity included points, bbox,
  features, class histogram and member partitions rather than only final F1.
- A separate raw-map uniqueness audit found zero within-object and zero cross-object
  duplicate observations in every live result. Combined with exact raw
  live/simulator payload parity, this closes the stricter duplicate-evaluator gap
  without rerunning six expensive maps.

## What failed

- No formal staged-six fidelity case failed.
- An optional extra precheck was stopped during server oversubscription and the
  original ten-case scheduler was intentionally curtailed after its active case
  completed. Both decisions and retained roots are documented as F22–F24; neither
  partial run entered the formal aggregate.

## Unexpected observations

- Fidelity was exact even for `false_merge_f000015_r0011`, whose initial sparse
  repair was not exact. This cleanly separated a repair-executor bug from the live
  hook/simulator semantics and enabled the targeted F33 fix.
- Exact point payload parity held across independent live executions, which is much
  stronger than the minimum partition-only gate.

## Possible leakage

The intervention manifest identifies the injected event and requested wrong action;
that is the controlled experiment. Neither live suffix mapping nor simulator suffix
mapping reads final clean ownership. Clean artifacts are used only after both runs to
compare outcomes.

## Possible selection bias

- All six IDs were present in a genuine pre-live, outcome-blind ten-case manifest.
  The final staged-six manifest was frozen before fidelity comparison outcomes and
  records its exact freeze timing.
- The set is room0-only and intentionally overweights false merge (4/1/1). It tests
  method semantics efficiently but is not an incident-frequency estimate.
- One map was reused from an earlier independent live run; its case ID was
  predeclared and its artifact path is explicit in `live_orchestration.json`.

## Known limitations

- Six cases are below the original ten-case guide target and cover one scene.
- Live fidelity runs used `make_edges=false`; relation fidelity is therefore
  noninformative here and is tested separately with the existing room0 make-edges
  stream.
- The gate covers deterministic one-event external overrides, not sensor corruption,
  multi-event faults or delayed online commits.

## First divergent cases

None in the formal staged-six comparison. All comparison artifacts report no first
decision or target-origin mismatch.

## Decision: GO

Temporal simulator semantics are sufficiently faithful for Phase 1–3 controlled
executor conclusions on this staged set. Expansion to ten or more live cases should
wait until another method change can plausibly affect fidelity.
