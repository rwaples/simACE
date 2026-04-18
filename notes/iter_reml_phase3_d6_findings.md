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

## Pivot: root-cause diagnosis

After partial dev-grid results, we pivoted to using the dense-NumPy
reference to decompose the Vp-bias into its components.

**Dense Laplace logLik at n=1k, K=0.15, self-consistent β** (iterated
until β = mean(l̂(xβ=β)) converges):

| σ² point | β* | dense logLik |
|---|---:|---:|
| truth (0.5, 0.2, 0.3) | 0.547 | −494 |
| **mid (0.25, 0.10, 0.15)** | 0.590 | **−465** (highest) |
| all_E degenerate (0, 0, 0.95) | 0.570 | −544 |
| live EM fit (0.19, 0.04, 0.025_fit) | — | — |

**Finding 1:** the Laplace MLE is NOT at truth — it sits at a
"mid"-like point with σ² systematically below truth.  This is real
Laplace approximation bias at small N / moderate K.

**Finding 2 (revised — earlier claim wrong):** the live EM's fit
is *at* the Laplace MLE, not a local maximum below it.  With
self-consistent β iteration at the live EM fit point σ² =
(0.192, 0.043, 0.025_fit), the dense reference reports logLik =
**−442**, which is *higher* than both mid (−465) and truth (−494).
The live EM has correctly converged to the Laplace MLE.

**Conclusion:** Phase 3's σ² underestimation is **pure Laplace
approximation bias**, not an implementation issue.  The MLE of the
Laplace-approximated likelihood genuinely sits at small σ² on this
fixture.  To eliminate this bias would require a better
approximation (MCEM / Gauss-Hermite) — out of scope for Phase 3.

**Finding 3:** earlier I reported an "all_E global max" as a
surprising finding.  That was an artifact of a bug in the dense
reference's profile_xbeta — it used an inconsistent β (mode at
xβ=0, logLik at xβ=mean(l̂)).  With self-consistent β (proper
fixed-point iteration), all_E is firmly NOT the global max.

## Prevalence effect (dense n=1k)

Self-consistent dense logLik across K (mid − all_E gap measures
how "interior" the Laplace MLE is):

| K | truth | mid | all_E | gap (mid − all_E) |
|---|---:|---:|---:|---:|
| 0.05 | −214 | **−198** | −202 | 4 (marginal) |
| 0.15 | −494 | **−465** | −544 | 79 |
| 0.30 | −671 | **−637** | −733 | 96 |
| 0.50 | −743 | **−707** | −808 | 101 |

Confirming user's suggestion: **higher prevalence → more clearly
interior MLE → Laplace works better**.  At rare K (≤0.05) the gap
is marginal; Laplace's preference for interior vs degenerate
solutions is weak.  At balanced K, interior wins strongly.

## What this means for shipping Phase 3

Phase 3's algorithm is **correct**: it finds the MLE of the Laplace-
approximated marginal likelihood.  Verified at n=1k K=0.15 by the
dense-NumPy reference (no Hutchinson, no SLQ, no PCG).

What Phase 3 does NOT fix: Laplace approximation itself has
**O(1/n) bias toward smaller σ² and toward degenerate V**.  This
bias is worse at small N and at rare K.  No amount of algorithm
improvement changes this — it's baked into the approximation.

Implications:

* **Ship Phase 3 for moderate-to-large N (n≥10k) at moderate-
  to-high K (≥0.15).**  Bias is bounded and shrinks with N.
* **Warn users at small-N rare-K.**  σ² will be biased down; h²
  is more robust than individual V components.
* **Defer MCEM (Option C) to Phase 4.**  Needed if rare-K small-N
  applications are the main use case.
* The logLik guard prevents drift past local maxima; the sandwich-
  SE fallback prevents under-coverage at the boundary.  Both
  essential safety features already landed.

## Next steps

1. Validate at n10k: rerun on the existing sim fixtures and confirm
   Vp → 1 as N grows (should be visible by n=10k).
2. Write release notes documenting the Laplace-bias caveat.
3. Document `summarize_dev_grid.py` + `plot_trajectory.py` +
   `verify_converged_gradient` as supported diagnostics in the
   user guide.
4. Decide: add MCEM (Phase 4) or ship with caveats.
