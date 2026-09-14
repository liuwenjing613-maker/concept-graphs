# Phase 0 postmortem — branch convergence and regression

## What changed

- Created `exp/ali-my-revision-kernel-v1` from exactly
  `900f117557b9fea2e0924165b5e98917bc88afd9`.
- Recorded the V0 base, `ali-my-full` reference, source hashes, frozen ledgers,
  development scenes and holdouts in `docs/REVISION_V1_BRANCH_MANIFEST.md`.
- Kept `ali-my-full` as a reference only; no whole-branch merge or cherry-pick was
  used, avoiding replacement of the newer revision/relation files.

## What passed

- Existing and newly added revision tests passed in the server environment before
  large execution. The final scoped repository count is `128 passed, 1 skipped`;
  the Revision-specific count is `65 passed, 1 skipped`.
- Exact evidence materialization passed for room0 `3779/3779` and office0
  `1560/1560`, with zero approximation or missing-payload fallback.
- The non-empty room0 relation regression covered 200/200 frames, 2,425 relation
  observations and 49 final directed edges. ali-dev and ali-my edge identities and
  supports were exact; detection, process-edge and map-edge sources agreed.
- Frozen source hashes remained stable throughout the Phase 1–3 executions.

## What failed

- No V0 regression gate failed. Failed/superseded implementation attempts F1–F4
  are retained in `FAILED_RUNS_LEDGER.md`; they did not enter frozen aggregates.

## Unexpected observations

- room0 provided informative relation evidence, whereas office0's current relation
  stream was empty/noninformative. Exact office0 empty-edge equality is therefore
  not relation-reconstruction evidence.
- Materialization remained exact even under parallel server load, so detection/SAM
  reruns were not required for executor experiments.

## Possible leakage

Phase 0 regression tools may read frozen clean artifacts to reproduce V0 behavior;
they do not generate the proposed sparse-replay result. Runtime leakage is audited
separately in Phase 1.

## Possible selection bias

There is no outcome-based case selection in Phase 0. Development scenes are
explicitly room0/office0; six other Replica scenes remain holdouts and are not
silently represented as evaluated.

## Known limitations

- This phase preserves rather than re-proves every historical V0 artifact.
- It does not certify physical-world GT correctness; the frozen clean map remains
  an uncorrupted mapper counterfactual reference.

## First divergent cases

None for branch base, evidence materialization, or relation-backend parity.

## Decision: GO

The branch and frozen sources are suitable for Phase 1–3 executor validation.
