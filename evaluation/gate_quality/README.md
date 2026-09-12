# Semantic Gate Quality Evaluation

Protocol: `semantic_gate_quality_v1_20260912`.

This evaluator treats the semantic gate as an error detector. Unlike the existing
gate-event-only evaluation, its population is every association proposal. A gate
event is positive when its `current_observation_uid` occurs in
`blocking_association_gate/events.jsonl`; absence is treated as not triggered only
after `blocking_association_gate/summary.json` proves the gate run completed and
`stats.processed` equals the event-log length.

The primary task is wrong-fusion detection over determinate CLEAN pre-gate ATTACH
proposals:

- positive oracle label: pre-gate ATTACH is `WRONG_FUSION`;
- negative oracle label: pre-gate ATTACH is `CORRECT_CANONICAL` or
  `CORRECT_FRAGMENT`;
- TP/FN/FP/TN are defined by crossing that oracle label with gate triggered/not
  triggered.

Prediction identity is inferred from each candidate's frozen object version before
the event. Only CLEAN historical members vote, and the current observation is
explicitly excluded. Final map best-IoU is never used.

The evaluator also reports a broader CLEAN-action task that includes duplicate NEW
and false DISCARD, plus post-gate correction/harm. Gate detection metrics and VLM
correction metrics remain separate.

## Inputs

First produce a complete observation-GT result:

```bash
/root/miniconda3/envs/concept_graphs/bin/python \
  -m evaluation.observation_gt.evaluate_observation_gt \
  --run-dir /path/to/run \
  --reference-dir /path/to/reference/office2 \
  --config evaluation/observation_gt/config.example.json \
  --out /path/to/run/observation_gt_metrics
```

Then evaluate gate quality:

```bash
/root/miniconda3/envs/concept_graphs/bin/python \
  -m evaluation.gate_quality.evaluate_gate_quality \
  --run-dir /path/to/run \
  --observation-gt-dir /path/to/run/observation_gt_metrics \
  --out /path/to/run/gate_quality_metrics
```

## Outputs

- `metrics.json`: confusion matrices, miss rate, recall, precision, residual error,
  false intervention, correction, and harm;
- `gate_quality_events.jsonl`: every association proposal and its causal labels;
- `missed_wrong_fusions.jsonl`: false negatives for direct inspection;
- `caught_wrong_fusions.jsonl`: true positives;
- `false_interventions.jsonl`: correct ATTACH proposals that triggered the gate;
- `confusion_matrix.csv`: compact table for analysis;
- `protocol.json`: definitions and input/code hashes.

Run tests from the repository root:

```bash
/root/miniconda3/envs/concept_graphs/bin/python -m unittest -v \
  evaluation.gate_quality.test_gate_quality
```
