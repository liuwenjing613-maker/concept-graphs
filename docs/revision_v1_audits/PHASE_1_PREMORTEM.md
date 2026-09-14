# Phase 1 premortem — no-oracle sparse replay

1. **Scientific claim.** A replay executor can apply an explicit sparse historical
   constraint while every unconstrained observation follows the native ConceptGraphs
   matcher, without reading the clean final ownership trajectory.
2. **Metric that should degrade if this phase has no effect.** Constraint-hit and
   constraint-satisfaction counts should be zero; on adversarial decisions the
   repaired membership/recovery should be worse or the replay should defer.
3. **Oracle access.** The benchmark compiler may use the known injected intervention
   to emit the correct sparse action. Runtime constraint resolution/replay may not
   read clean final membership, affected final groups, GT, or expected owners.
4. **False PASS routes.** (a) forcing every future observation from final ownership;
   (b) relabeling output membership only; (c) evaluator and executor sharing the
   same final grouping; (d) simply removing an artificial override and letting the
   deterministic mapper reproduce its original action. Route (d) is audited with
   a mandatory `NO_CONSTRAINT_REPLAY` ablation plus parsed/hit/override counts.
5. **Minimal adversarial counterexample.** `MUST_LINK(A,B)` and
   `CANNOT_LINK(A,B)` in the same active scope must hard-fail; an unresolved target
   must `DEFER`, never fall back to a clean owner.
6. **Selection audit.** Freeze stratified case IDs before running any replay outcome.
   Rank-1 and relation-impact screening are excluded from the primary population.
7. **If every F1 is 1.0.** First compare against no-constraint replay, inspect source
   for forbidden identifiers, verify only the anchor was constrained, and check
   whether output entity assignment used benchmark clean membership.

Additional falsification: proposed runtime modules are source-audited and imported
without constructing any benchmark clean-state adapter.
