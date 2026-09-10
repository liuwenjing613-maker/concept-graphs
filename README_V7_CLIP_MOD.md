# v7_CLIP_mod: mask weight experiment

Based on v7_CLIP commit 235e16c. Source v7_CLIP remains unchanged.

## Run on server

```bash
cd /home/chenkejun/beauty/conceptgraphs/code/experiments/v7_CLIP_mod
bash run_v7_CLIP_mod.sh auto --scene room0 --mask-weight 0.25 --gpu 1 --vlm-ports 11464
```

The example port must pass a current vision inference check before a formal run.
`--mask-weight` (alias `--clip-masked-weight`) defaults to 0.25; bbox weight is 1-alpha.
Use 0 / 0.25 / 0.5 / 0.75 / 1. The fusion remains normalize((1-alpha)*normalize(bbox)+alpha*normalize(mask)).
At alpha=0 only bbox is encoded; at alpha=1 only the softmask view is encoded.
Padding 20, background factor 0.1 and blur radius 3 remain unchanged. Softmask-only retains the same crop and suppressed background.

Available scenes: room0 room1 room2 office0 office1 office2 office3 office4.
`--vlm-ports 11464 11463` selects at most three distinct localhost endpoints.
Alternatively use `--vlm-urls http://127.0.0.1:11464`; do not combine the two options.
Actual HTTP calls and timeout failover use only the selected endpoints, overriding inherited V7_VLM_URLS.
`--gpu` controls YOLO/SAM/CLIP/mapping, not the server GPU used by VLM.
`human` retains interactive fallback; `auto` retains the source fallback policy.
Add `--dry-run` to print the resolved configuration without starting mapping.

Each run starts from an empty map at frame 0 (end=2000, stride=5). Existing map directories are rejected.
Names include mode, scene, mask weight, timestamp and PID. Detection caches are isolated and their contract binds mask weight; incompatible cache reuse is rejected.
Do not pass a previous map or a different-alpha detection cache.

Default outputs:
`/data/chenkejun/beauty/conceptgraphs/results/experiments/oracle_three_error_20260828/pilot/b0_dataset/Replica/<scene>/exps/v7_CLIP_mod_<mode>_<scene>_mask<alpha>_<timestamp>_<PID>`

Launch metadata (weights, actual endpoints, GPU, command, timing):
`/home/chenkejun/beauty/conceptgraphs/results/blocking_association_gate_v1/launches/<experiment>.json`
Result and review pages retain the parent launcher's behavior and are printed at startup.

## Current VLM check

```bash
PYTHONPATH="$PWD/.runtime-deps:$PWD" /home/chenkejun/beauty/conceptgraphs/envs/cg-ali/bin/python scripts/check_vlm_ports.py
```

This issues a small image request to each selected port. It does not restart services.
2026-09-10 corrected check: 11437, 11464 and 11463 successfully run the experiment model with num_ctx=32768. All three also passed replay of an existing real merge request (original image hash, prompt, think/format/options preserved), HTTP 200, about 3.3 seconds each.
Earlier probes omitted num_ctx and used the server default 262144, causing memory failures; they did not establish failure of the actual experiment configuration. The checker now defaults to 32768, matching v7.
11435 is not listening; 11436 still fails to load the vision component. 11437 also passed the original real merge request and parser check.
No server restart or mapping-code parameter change was needed. Details: /home/chenkejun/beauty/v7_CLIP_mod_20260910/repair_summary.md. Use --vlm-ports 11464 11463.
Report: `/home/chenkejun/beauty/v7_CLIP_mod_20260910/`.

## Verification and limits

52 inherited unit tests passed; two regression tests passed (all five alpha values match the source numerically using a deterministic encoder, endpoint environment reaches the mapping subprocess).
12 dry runs cover all eight scenes and five alpha values; six invalid input cases were rejected.
No complete scene or actual CLIP GPU mapping smoke was run. No accuracy metrics were produced; existing result spreadsheets were not changed.



Results migration requested by user: destination /data/chenkejun/beauty/conceptgraphs/results/. Completed run directories retain links at their old /home paths. Active runs and active annotation services stay on the original disk until separately migrated. New v7_CLIP_mod runs default to the data-disk Replica output root above. Migration status and inventory: /home/chenkejun/beauty/results_migration_20260910/.
