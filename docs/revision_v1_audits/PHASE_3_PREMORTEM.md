# Phase 3 premortem — pre-anchor snapshot and suffix-local replay

1. **Scientific claim.** Starting from a validated state immediately before the
   anchor, dependency-bounded suffix replay can reach the same result as a fair global
   replay using the identical sparse constraint.
2. **Metric that should degrade if this phase has no effect.** Pre-anchor member/
   geometry fidelity, local-vs-global partition and geometry agreement, recovery,
   closure fraction, or first-divergence location will expose failure.
3. **Oracle access.** Snapshot reconstruction uses only historical versions and exact
   observation payloads before the watermark. Benchmark clean state is evaluation-only.
4. **False PASS routes.** Rebuilding affected identities from empty using their final
   member lists; overlaying clean final objects; comparing local sparse replay with a
   global full-membership oracle; or excluding cases after observing mismatch.
5. **Minimal adversarial counterexample.** A missing observation payload or a
   pre-anchor version whose reconstructed bbox exceeds tolerance must stop replay.
   A dependency crossing to an outside candidate must expand or defer, never ignore it.
6. **Selection audit.** Primary, stress, and global-reference subsets are frozen and
   disjoint before replay outcomes. Failures remain in aggregates.
7. **If local and global are all exact.** Check whether closure accidentally equals
   the full event stream, whether original suffix memberships were copied, whether
   snapshot content extends past the anchor, and whether the evaluator canonicalizes
   both sides independently.

Known risk before implementation: postprocess every five frames can propagate beyond
the initial object set. The first implementation must report closure expansion and
`POSTPROCESS_DIVERGENCE`; it may not hide this with a permissive tolerance.
