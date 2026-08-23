# Phase 1 postmortem — no-oracle sparse replay

## What changed

- Added explicit sparse primitives and replay modes. The proposed path consumes an
  anchor-scoped action (`ASSIGN_OBSERVATION` or `CANNOT_LINK`) plus immutable
  lineage/origin references; it never consumes a final member trajectory.
- Kept native and injected-history decisions separate: no active constraint preserves
  the historical branch, while `KEEP_NATURAL` now applies the actual native matcher
  result. This distinction was forced by first divergence F33.
- Split runtime invariant checks from clean-reference benchmark evaluation and added
  an AST leakage guard covering identifiers, attributes, arguments, keyword arguments
  and mapping string keys.

## What passed

- Both frozen primary manifests completed exactly 30/30 cases: ten per failure type
  in room0 and office0, with no missing or unexpected case IDs.
- Of 60 injected cases, 49 produced measurable final damage. All 49 damaging cases
  finished as verified beneficial repairs and reached exact reference membership and
  geometry; the 24 room0 damaging cases also reached exact informative relation state.
- Every damaging case recorded a real override of the injected historical decision.
  The 11 no-damage cases were retained but received no repair credit.
- The five declared production runtime files have zero forbidden clean/GT/oracle
  identifiers or benchmark/evaluator imports, including the adversarial string-key
  form that the first guard missed.
- Full-observation symmetric collateral checks passed in all 60 final states; no
  candidate-only, missing, duplicate or repartitioned observation remained.

## What failed

- Before F33, `room0/false_merge_f000015_r0011` reached only member F1 `0.993216`
  and non-exact relation state. It was correctly reclassified as collateral damage.
  The old artifacts are retained under
  `room0_primary_diagnostics_pre_f33/false_merge_f000015_r0011`.
- The cause was semantic, not numerical: `KEEP_NATURAL` kept the injected historical
  target and chose another eligible object instead of the native create decision.
  The corrected targeted rerun reached exact member, geometry and relation state.
- Eleven temporal corruptions (six room0, five office0) self-healed before the final
  state. These are non-successes by construction, not executor failures.

## Unexpected observations

- Natural recomputation without retaining the injected bad action equals sparse
  repair in all 60 final cases. The sparse constraint is sufficient relative to
  replaying the known wrong historical action, but this controlled benchmark does
  not show that the constraint is necessary once the anchor is known.
- The F33 case was especially useful: live/simulator corruption fidelity was exact,
  isolating the problem to constraint execution rather than the corruption runner.

## Possible leakage

The benchmark compiler knows the original action at the injected event and uses it
to emit a sparse primitive. This is intentionally an oracle-compiled constraint and
must not be presented as blind diagnosis or VLM performance. The executor receives
only that primitive, immutable history and the current corrupted head. Static source
audit and method ablations found no final-owner trajectory access in runtime code.

## Possible selection bias

- Selection was frozen before outcomes with seed `20260823`, ten cases per type and
  round-robin temporal/size/margin strata.
- Eligibility still requires source/target support and anchors in frames 3–185, and
  only the two development scenes were used. These restrictions are explicit and
  prevent interpreting the result as a holdout or all-incident estimate.
- The one F33 rerun was outcome-selected debugging of an already frozen case; it did
  not add, remove or replace any headline case.

## Known limitations

- Anchor localization is injected, not blind.
- Constraints are compiled from known original actions, not produced by a VLM.
- The clean branch is an uncorrupted mapper counterfactual, not physical-world GT.
- Clean-negative harm, online rebase and live commit were not evaluated in this phase.

## First divergent cases

`room0/false_merge_f000015_r0011` was the first and only final primary divergence
before correction. Native action was create (`None`), injected history targeted object
7, and the old constrained result targeted object 12. After separating native from
historical defaults, the applied action is create and all final state dimensions are
exact.

## Decision: GO

The sparse executor passes the intended no-final-trajectory claim on the two-scene
controlled benchmark. A stronger claim that sparse evidence is necessary, inferred
blindly, or safe on clean negatives remains FIX/not-yet-evaluated work.
