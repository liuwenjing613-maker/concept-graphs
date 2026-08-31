# Corrected mask-first re-audit (2026-09-01)

This directory replaces the earlier mask-first absolute numbers that were built
from a Habitat semantic sidecar with a mismatched visible viewport.  The old
numbers remain in Git history for auditability but must not be cited.

The corrected run uses `room0` and `office0`, 400 online frames per scene
(frames 0..1995, stride 5), and starts every map from empty state.  Semantic
sidecars are reconstructed from the current depth and pose by exact nearest
assignment to the ReplicaSSG annotated semantic mesh with a 3 cm gate.

Contents:

- `final_summary/`: Chinese report and machine-readable consolidated record.
- `point_semantic_corrected/`: ali-dev-compatible exact CPU cKDTree semantic
  metrics for B0, OP, OM_pure, OM_all, OA, and the OA GT-label isolation.
- `structure_corrected/`: class-agnostic AP/F1 at 2.5, 5, and 10 cm.
- `sidecars/`: sidecar construction requests, hashes, and 400-frame alignment
  manifests.  Per-frame `.npz` files are intentionally omitted.
- `formal/` and `og/`: online-run manifests and READY sentinels.  Large map
  pickles are intentionally omitted; their hashes are preserved in the final
  consolidated JSON and evaluation inputs.

Main interpretation:

1. B0 semantic metrics reproduce the frozen ali-dev result exactly.
2. OM_pure gives a clear structure gain, while maximal OM_all is not a stable
   improvement over OM_pure across the two scenes.
3. OA gives the largest structure increment.
4. Native semantic metrics are limited because changed masks retain the source
   proposal CLIP feature.  On identical OA geometry, strict GT labels raise the
   two-scene mIoU from 28.08% to 90.30%; this is an isolation upper bound, not a
   deployable result.

No credentials, API keys, model weights, datasets, or large map files are stored
in this directory.
