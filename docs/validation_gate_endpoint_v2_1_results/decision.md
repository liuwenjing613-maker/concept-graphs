# Incident-level validation gate

Status: `PROCEED_TO_EXPERT_TRACE`

## R1 endpoint results

- Selection mode: endpoint_census
- Unique incidents: 97
- Evidence sufficiency: 0.979381
- Confirmed endpoint errors: 40
- Confirmed endpoint correct: 55
- Human-unclear endpoints: 2
- Endpoint-error rate among sufficient cases: 0.421053
- Full endpoint bounds: [0.412371, 0.43299]
- Bounds including the machine-blocked endpoint: [0.408163, 0.438776]
- Total / median review time: 4765.7 s / 28.0 s

## Screener ranking diagnostic

- review_score ROC AUC: 0.420455
- review_score average precision: 0.365647
- adjudicable endpoint-error prevalence: 0.421053

Higher review_score was intended to rank risk. AUC and top-k lift below the prevalence baseline mean the score should be revised rather than used as a probability.

## Interpretation

R1 answers only whether an error remains visible in the exact final map. Checker stage and repair are not human-label fields.
Even after an R1 pass, repair remains pending until confirmed endpoint errors receive expert causal traces and actual intervention/replay verification.

## Known limitation

This run has one human R1 reviewer; it does not claim inter-rater reliability.
