# v7_CLIP_must_v2 — room2 online validation

This version follows the user's 2026-09-14 approval of selective quality confirmation, quality-dependent merge voting and P0-4 exact-evidence quality caching. P0-3 changes are not included.

## Source and scope
The parent commit in this isolated repository freezes the exact 5090 must working tree, including its original uncommitted P0 fixes. Its upstream ancestor is v7_CLIP 235e16c995fefb905fd3b21c11ad4fd1e8d3bbe4. No existing worktree was modified. Association candidate generation, mapper thresholds, CLIP features, identity prompts and identity evidence selection remain unchanged. Multiple-SAME selects one observation anchor and proposes separately reviewed persistent pairs; snapshot certificates remain mandatory and stale certificates defer. Geometry has no merge override.

## Quality evidence and prompts
Quality images use lossless PNG: full RGB at left with red contour and 12% fill; same-frame crop at right with 25% fill and outside background scaled to 25%. Preserve mask holes and all mask extent. Observation images contain no text; node images use only H1/H2/H3. Node quality selects the highest-quality representative, then visual and geometric outliers from histories with positive quality at least 25% of the representative. This selection uses only current online history; it does not modify identity evidence selection.

Actual frozen prompts are in conceptgraph/slam/prompts/v7_must_v2. Observation primary is boundary, negative confirmation concise. Node primary is concise, negative confirmation boundary. Confirmation uses the same image without the primary answer. Both negatives must agree to return CORRUPTED/CONTAMINATED; disagreement or uncertainty returns INSUFFICIENT. Interface and format failures are separately marked. Positive primaries require no second call. Quality confirmation is never a merge vote. Strict JSON accepts only an optional enclosing whole JSON fence; no free-text answer extraction for quality. Prompt text requests 90/120 Chinese characters; the tested validator limit remains 120.

## Merge decision table
- Any confirmed CONTAMINATED node: KEEP_SEPARATE, no identity inference.
- Otherwise continue identity inference, including quality INSUFFICIENT or interface failure.
- Both CLEAN plus identity SAME: require one positive merge vote.
- At least one quality INSUFFICIENT, without interface failure, plus identity SAME: require two consecutive votes on distinct frames.
- Quality interface/format failure: the current event DEFERs even if identity says SAME; never add a reject vote solely for failure.
- Identity UNCERTAIN/interface failure: DEFER. Identity DIFFERENT: KEEP_SEPARATE.
- Existing two-rejection lock, history-change unlock, generation separation, same-pair/frame deduplication and DEFER streak-reset are retained.

## P0-4 cache
Run-local cache keys include exact rendered image bytes, ordered history identities/mask references/RGB hashes, observation identity or node UID+merge-generation, both prompts, complete inference parameters and verified model digest. Cache only successful visual conclusions, including INSUFFICIENT; never cache interface/format failure. Cache hits explicitly reference their source record and bind the resolved quality result to the current H snapshot. No identity decisions or merge certificates are cached. The map snapshot itself is still checked independently before every merge.

## Environment and preflight
5880 GPU 5 maps; GPU 3 is unusable and is not used. Existing Ollama endpoints 11463/11464 serve qwen3.6:35b-a3b-mtp-q4_K_M, digest c7bd058dd9774cae7dae32ef8cf3822aaacddd21e635e3cb6ec38effca0dc57f. Startup verifies each endpoint's digest. An isolated ultralytics 8.3.0 matches the original frozen cache contract; shared packages are unchanged.

Original 400-frame detector/CLIP cache: all 6401 source files verified before relocation. Only absolute source paths in copied cache metadata were relocated after matching the raw RGB hashes. All 802 RGB/depth/pose/camera files match 5090. No front-end recomputation and no loaded prior map.

## Validation
78 tests pass (64 v7 tests plus 14 association/human-merge regression tests). 100 quality layouts and 200 prompt/parameter payloads were checked; eight real quality cases passed interface/routing/cache checks. The current renderer matches the original preview function pixel-for-pixel, but historical saved PNGs have minor pixel differences (recorded per case); historical replay quality scores are not claimed as current exact reproductions. New model outputs still contain visual errors, so this validates integration, not perfect visual accuracy.

A four-frame fresh online smoke run completed with strict evidence audit PASS, zero logging/reference/duplicate-membership errors, 33 physical HTTP attempts, one executed CLEAN/one-vote merge and two deferred two-vote merge events. The initial smoke exposed a JPEG-only blind-annotation validator; the validator now supports actual PNG/JPEG signatures and has a regression test. The failed run is not included in results. Semantic audit findings are not visual ground truth or evidence that all mapping decisions are correct.

Formal run: start=0, end=2000, stride=5 (400 frames), scene room2, empty map, own empty quality cache. Run root: /home/chenkejun/beauty/v7_CLIP_must_v2_run_20260914. Formal metrics are pending. Historical must used H800 FP8, so comparison to it must disclose model differences and cannot be called a strict single-factor ablation.
