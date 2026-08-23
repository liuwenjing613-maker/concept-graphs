# Phase 2 premortem — one-event temporal corruption fidelity

1. **Scientific claim.** The offline temporal simulator applies exactly the same
   single association intervention as the live mapping hook and then permits native
   suffix evolution.
2. **Metric that should degrade if this phase has no effect.** Injection count,
   downstream decision-trace agreement, final membership agreement, object geometry,
   postprocess counts, or relation state will disagree with live reruns.
3. **Oracle access.** The intervention manifest identifies the injected event; the
   simulator does not read final clean ownership to make suffix decisions.
4. **False PASS routes.** Comparing a static final-membership edit, comparing only
   the anchor record, limiting scope to members chosen from clean final groups, or
   accepting F1 similarity while downstream decisions differ.
5. **Minimal adversarial counterexample.** A plan whose target is not uniquely active
   must hard-fail and an intervention applied zero or two times must fail fidelity.
6. **Selection audit.** Ten fidelity cases are selected from a frozen manifest by
   failure type and temporal stratum, not by whether live/simulator agreement passes.
7. **If all ten match exactly.** Verify the live artifacts are independent executions,
   source hashes/run IDs differ as expected, and the comparator checks decision and
   postprocess traces rather than only final membership.

Methodological warning: because the corruption is an external override, a natural
replay without that override may recover the baseline. This is not evidence that the
sparse constraint caused recovery; Phase 1's no-constraint ablation remains required.
