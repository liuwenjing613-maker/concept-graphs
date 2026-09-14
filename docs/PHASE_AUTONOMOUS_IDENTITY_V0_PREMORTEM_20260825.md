# Autonomous Identity Closure V0 — Premortem

Date: 2026-08-25
Execution scope: remote server only
Baseline: `publish/ali-my-v2-10h-20260824 @ 79a3d5f`
Working branch: `exp/ali-my-autonomous-repair-v0-20260825`

## Paper claim supported

This phase tests the narrowest autonomous loop that is scientifically useful:

```text
existing machine incident ticket
→ finite identity hypotheses including NO-OP
→ counterfactual replay
→ held-out evidence scoring
→ calibrated selective commit or DEFER
```

It does not claim blind scene-wide discovery, online live-map commit, geometry
repair, or universal autonomous repair.

## Production information boundary

The production path may read only immutable provenance, machine checker output,
candidate bindings, proposal-view hashes, held-out observation evidence, replay
states, runtime validity results, and a frozen calibration artifact. It must not
read `human_label`, `final_state`, `final_error_type`, endpoint gold, desired
owner, expected action, gold constraints, or benchmark parity outcomes.

Human labels are permitted only in a separate offline evaluator and calibration
fit command. The runtime decision artifact must pass a recursive forbidden-field
audit.

## Shortcut that could pass without solving the problem

1. Selecting the repair whose final membership matches the human endpoint.
2. Reusing the gold constraint fingerprint or local/global parity result.
3. Treating mapper agreement as independent evidence.
4. Counting constraint hits or boundary vetoes as semantic correctness.
5. Achieving zero harm by always returning DEFER.
6. Choosing thresholds after inspecting holdout outcomes.

## Adversarial test added before implementation

The office0 false-split case is the first counterexample. A read-only replay
diagnostic showed that the native NO-OP has better mapper self-likelihood than
the human-correct SAME_INSTANCE repair:

```text
NO-OP mean applied log-likelihood:  -0.0093
SAME mean applied log-likelihood:   -0.0220
```

Therefore a verifier that simply reuses mapper association likelihood must not
be allowed to auto-commit this case. The primary-only path is expected to DEFER
or fail selection until genuinely independent held-out visual evidence is added.

## Metric expected to fail

`primary_score_only` is not expected to select all three native identity repairs.
If it reports 3/3, oracle leakage or outcome-screened feature design is the first
hypothesis. Direct VLM action generation is also expected to remain a weak
baseline.

## Selection-bias controls

- Keep all three previously frozen association-root identity cases.
- Add clean negatives by a deterministic checker/stage rule, not by score outcome.
- Freeze evidence split and hashes before critic responses.
- Report candidate recall separately from candidate selection.
- Do not tune on room1/office1/office2 holdouts in this phase.
