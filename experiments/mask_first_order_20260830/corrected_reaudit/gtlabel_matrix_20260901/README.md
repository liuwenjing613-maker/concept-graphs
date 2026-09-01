# Uniform GT-label matrix (2026-09-01)

This directory contains the symmetric object-level GT-label oracle for
`B0`, `MF_OP`, `MF_OM_pure`, `MF_OM_all`, and `MF_OM_all_OA` on `room0` and
`office0`.

Each predicted object receives exactly one majority GT class using the same
retained SLAM points and exact nearest-neighbour correspondences as the ali-dev
point-semantic metric (`n_exclude=6`). It is not a pointwise GT replacement.
Geometry, object partition, point ownership, input map, GT denominator, and
metric formulas are unchanged between the native and GT-label runs.

Read `summary/GT_LABEL_MATRIX_CN.md` first. The per-stage directories retain
the full JSON result and compressed confusion matrices. The evaluator and
compiler are under `scripts/repairability_audit/`.

These scores are oracle upper bounds for causal diagnosis, not deployable
method performance.
