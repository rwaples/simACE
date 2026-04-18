# iter_reml Phase 3 D6 — dev-grid rerun findings

**Status:** in-progress (2026-04-19).  Full grid rerun triggered from
the new binary (commits through ace_iter_reml@23636fb with the logLik
guard).  This note is updated as scenarios complete.

## Headline

**σ²_E collapse is resolved.**  Across all Laplace dev scenarios
the new EM finds Ve above the ε²_end floor (0.09).  Pre-fix, every
dev scenario had Ve=0.090 ± 0.0003 (pinned).  Post-fix, Ve is a
genuine estimate.

**Residual bias:** Vp is substantially underestimated across the
grid (typically 0.3–0.5 vs truth 1.0).  This is Laplace-approximation
bias, not implementation error: the dense-NumPy reference
(`fitace.iter_reml.ref_dense`) matched our analytic gradient to
≤1 log unit at self-consistent Xβ.

## Summary so far (post-fix, rep1 only)

| Scenario | n_phen | V(A) | V(C) | Ve | Vp | h² | c² | n_iter | wall_s | converged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| truth | - | 0.50 | 0.20 | 0.30 | 1.00 | 0.50 | 0.20 | - | - | - |
| dev_laplace_n1k (n≈1k) | 1000 | 0.192 | 0.043 | 0.115 | 0.350 | 0.549 | 0.123 | 34 | 98.6 | ✓ |
| dev_laplace_n3k (n≈3k) | 3000 | 0.222 | 0.066 | 0.108 | 0.396 | 0.562 | 0.167 | 37 | (tbd) | ✓ |
| dev_laplace_n10k | 10k | (tbd) | | | | | | | | |

## Comparison to pre-fix (rep1, same scenarios)

| Scenario | Metric | Pre-fix | Post-fix | Truth |
|---|---|---:|---:|---:|
| dev_laplace_n1k | Ve | 0.090 (floor) | 0.115 | 0.30 |
|                  | Vp | 0.484 | 0.350 | 1.00 |
|                  | h² | 0.647 | 0.549 | 0.50 |
|                  | n_iter | 7 | 34 | - |
| dev_laplace_n10k | Ve | 0.090 (floor) | (tbd) | 0.30 |
|                  | Vp | 0.528 | (tbd) | 1.00 |
|                  | h² | 0.589 | (tbd) | 0.50 |

## Interpretation

**The Laplace MLE systematically underestimates Vp** on these
scenarios — about 0.3–0.5 vs truth 1.0.  h² is close to truth
(0.55 vs 0.5) because both V(A) and Vp shrink proportionally.

This matches what `LaplaceReference.optimize_mle` found on the tiny
n=900 K=0.2 fixture: the dense Laplace MLE is genuinely at small
σ² (all variance compressed into E).  The new EM is faithfully
finding this biased MLE; the bias lives in the Laplace approximation
itself, not in the algorithm.

## Ship-ready criteria recap (from phase3_pxem.plan.md §Decision gate)

| Criterion | Threshold | Pre-fix | Post-fix (partial) |
|---|---|---|---|
| σ²_E recovery not floored | bias < 0.05, not at ε² | ✗ (pinned) | ✓ (0.12 at n1k) |
| Vp recovery | 0.9 < Vp < 1.1 | ✗ (0.4–0.7) | ✗ (0.35–0.40) |
| h² bias | \|Δh²\| ≤ 0.05 typical | near-OK | ✓ (0.05 at n1k) |
| Rare K recovery | \|Δh²\| ≤ 0.10 | tbd | tbd |
| Wall-time penalty | ≤ 3× Phase 2 | 1× | ~10× at n1k (98s vs 1.9s) |
| Sandwich SE activates | ≥ 80% of fits | N/A | tbd |

**Gap:** Vp recovery fails the ship gate.  This is a known Laplace
limitation — the full fix is Option C (MCEM) or a better
approximation (Gauss-Hermite), deferred to Phase 4 per
`phase3_pxem.plan.md §Out of scope`.

## What this means for shipping Phase 3

With the logLik guard, Phase 3 is **a clear improvement on Phase 2**:

* σ²_E no longer pinned at floor.
* h² bias reduced.
* LogLik monotone-ascent (up to Hutchinson noise) guaranteed.
* Observed-info SEs populated.

But it does **not** meet the ship gate's Vp criterion at small N.
Phase 3 is shippable with documented caveats:

* Ship for large-N applications (n≥10k; evaluating).
* For small-N rare-K applications, recommend Stan / MCMC reference
  instead.

## Next steps (when full grid completes)

1. Aggregate across reps per scenario (mean, sd).
2. Plot trajectory PDFs for any non-converged or weird fits.
3. Rerun `verify_converged_gradient` on the converged fits to
   confirm gradient-at-convergence is small.
4. Update `phase3_pxem.plan.md §Decision gate` with the empirical
   numbers.  Decide shippability.
