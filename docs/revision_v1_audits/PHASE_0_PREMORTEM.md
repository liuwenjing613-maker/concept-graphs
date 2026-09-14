# Phase 0 premortem — branch convergence and regression

1. **Scientific claim.** V1 starts from the published V0 executor and relation
   supplement without silently changing their established behavior.
2. **Metric that should degrade if this phase has no effect.** Existing revision
   tests, evidence integrity/materialization, disabled parity, or the non-empty
   relation regression would fail relative to the frozen V0 artifacts.
3. **Oracle access.** Regression tools may read benchmark clean state because they
   only reproduce V0 claims. No new proposed runtime result is produced in Phase 0.
4. **False PASS route.** Running only newly added unit tests, omitting native
   Open3D/backend checks, or comparing against regenerated rather than frozen
   artifacts could hide a regression.
5. **Minimal adversarial counterexample.** A branch manifest with the wrong base
   commit or a mutated source ledger must fail hash/base validation.
6. **Selection audit.** No case selection occurs in this phase.
7. **If everything is perfect.** First check that the exact frozen roots and
   non-empty relation stream were used and that tests were not silently skipped.

Decision before implementation: proceed only from `900f117`; do not merge the
diverged `ali-my-full` snapshot.
