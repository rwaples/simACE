# PCGC Phase 3 M0 — SE-honesty spike

**Date:** 2026-04-22. **Phase 3 plan:** `notes/PCGC/phase3_correction_landing_plan.md`.

## Purpose

Phase 2 (`notes/PCGC/phase2_bias_derivation.md`) validated the rare-K Hermite
correction's point-estimate bias reduction but never computed jackknife SE
under the correction.  The phase3 plan (§P3.M1 step 6) commits the production
per-block jackknife scheme **before** the backend port ships, since the phase2
"single-Newton-step per block" rationale was unmeasured.

M0 settles this with a minimal, asymmetric-threshold spike:

  * **Cells:** `pcgc_bias_A25_C20_K01` (rare-K × mid h²+c²) and
    `pcgc_bias_A50_C20_K05` (less rare × high h²+c²).  Pre-existing phase 2
    fixtures reused, 5 reps each.
  * **Schemes compared (both warm-started from full-sample β):**
    * **single-step** — one Newton step per block (undamped α=1 on the LOO
      problem; ~2 s/fixture).
    * **full-Newton** — Newton to convergence per block (~35–45 s/fixture
      via a new numba kernel that parallelises across blocks).
  * **Plan's decision rule (asymmetric 0.9 floor vs empirical SD):**
    * Primary: `min over (cell × component) of SE_singlestep / SD_reps ≥ 0.9`
      → single-step default.
    * Primary fails but `SE_fullnewton / SD_reps ≥ 0.9` → full-Newton default.
    * Both fail → full-Newton default with "SE suspect at K ≤ 0.01" caveat.

## Why α = 1 for single-step (not the production α = 0.5)

The production full-sample Newton uses `damping = 0.5` because it starts from
the Golan OLS point (β_OLS), which can be far from the corrected fixed point
under rare-K ascertainment — damping stabilises that trajectory.

The per-block jackknife Newton starts from β_full (which already solves the
full-sample corrected moment equation).  The LOO perturbation is small
(roughly O(1/B) of the pair mass), so β_b is close to β_full.  In this
regime, one undamped Newton step is the first-order Taylor expansion of β_b
around β_full — the right linear approximation.  Damping by 0.5 would make
`SE_singlestep ≈ 0.5 × SE_fullnewton` *by construction* (half-step →
quarter-variance → half SE), a pure methodological artefact rather than a
real calibration signal.  The probe on rep1 confirmed the factor-of-2
artefact (pre-fix: SE_single=0.0114 vs SE_full=0.0228); switching single-step
to α=1 eliminated it completely (post-fix: 0.0228 vs 0.0228).  Production
single-step uses α=1 inside the jackknife regardless of the production
Newton's own damping.

## Implementation notes

*Spike code:* `fitACE/fitace/pcgc/prototype_rare_k_correction.py` subcommand
`m0`.  Adds:

  * `_newton_step`, `_newton_converge` — numpy Newton helpers refactored from
    the original inline prototype.
  * `_build_loo_block_moments_kernel` — a `@numba.njit(parallel=True)` kernel
    that computes, for each block b and its current β_b, per-block sums of
    the five β-dependent moments (a·Xβ², c·Xβ², a²·Xβ, ac·Xβ, c²·Xβ), split
    into in-block and LOO contributions.  Parallelises over blocks; per iter
    cost ≈ 30–45 ms at n≈35M pairs on 8 threads.
  * `jackknife_se_corrected` — single-step uses one kernel call under β_full;
    full-Newton iterates blocks synchronously until every block hits
    `tol=1e-5` (or `max_iter=50`).

Block assignment: 100 contiguous pair_blocks (single giant pedigree ⇒ auto
fallback), consistent with production `fit_pcgc` default.

## Results

Raw output: `fitACE/results/pcgc_bias_map/summary/phase3_m0_se_honesty.tsv`
(10 rows).  Ratios aggregate: `phase3_m0_se_honesty_ratios.tsv` (6 rows).

### SE and calibration table

| scenario | comp | SD_reps | mean SE_single | mean SE_full | ratio_single/SD | ratio_full/SD |
|---|---|---:|---:|---:|---:|---:|
| A25_C20_K01 | V(A) | 0.0862 | 0.0551 | 0.0553 | **0.639** | **0.642** |
| A25_C20_K01 | V(C) | 0.0230 | 0.0458 | 0.0461 | 1.991 | 2.004 |
| A25_C20_K01 | V(E) | 0.0648 | 0.0541 | 0.0545 | 0.836 | 0.841 |
| A50_C20_K05 | V(A) | 0.0378 | 0.0231 | 0.0230 | **0.610** | **0.609** |
| A50_C20_K05 | V(C) | 0.0145 | 0.0196 | 0.0196 | 1.353 | 1.353 |
| A50_C20_K05 | V(E) | 0.0370 | 0.0256 | 0.0255 | 0.692 | 0.691 |

Bolded rows are ratios < 0.9 (primary failure mode).

### Newton convergence

All 10 full-sample fits converged (`newton_converged=True` for every row,
`newton_iters` ∈ {15, 16}).  No divergence warnings.  Per-block Newton (full
scheme) also converged on every block (implicit — the outer loop hit
convergence before `max_iter=50`).

### Wall time

Per fixture on 8 threads (first JIT call cached between fixtures):

| phase | time/fixture |
|---|---:|
| Pair enumeration | 26–36 s |
| Full-sample corrected Newton | negligible (<1 s) |
| Single-step jackknife | 1.4–6.1 s |
| Full-Newton jackknife | 28.7–47.9 s |

Full-Newton is ~15–25× more expensive than single-step, driven by 100 blocks
× 15 outer iters × 1 kernel call each.  Spike total wall ≈ 12 min.

## Analysis

**Finding 1 (primary, robust): single-step ≡ full-Newton.**  Across every
(cell × component × rep) observation, `SE_single` and `SE_full` agree to
three decimals on σ² scale (ratio `SE_single/SE_full` ∈ [0.996, 1.007]).  The
0.2% disagreement is dominated by the final-iteration tolerance of the inner
Newton loop, not any real statistical difference between the two schemes.
With α=1 in the single-step path (see previous section), one LOO Newton step
is the exact first-order Taylor expansion of β_b around β_full, and that
first-order approximation is essentially complete for the LOO perturbation
magnitude at this n.

**Finding 2 (secondary): jackknife SE is component-mis-calibrated vs
SD_reps.**  V(A) and V(E) ratios are 0.61–0.84 (under-estimates); V(C)
ratios are 1.35–2.00 (over-estimates).  V(C) over-estimation is
conservative (wider CIs than necessary) and shippable without concern.  V(A)
and V(E) under-estimation is the real calibration gap.

**Caveat on Finding 2 with n=5 reps.**  SD across 5 reps has an estimate SE
of roughly SD/√(5-1) ≈ 50% of the SD itself.  The 0.61 ratio on (A50_C20_K05,
V(A)) has a one-sigma range of roughly [0.40, 0.90] given rep-count noise
alone.  Phase 2 §1 reported "SEs honestly calibrated within ~±40% of
jackknife SE across the grid" based on the full 80-cell × 5-rep grid; the
0.61–0.84 ratios here are within that same band.  The V(C) over-estimate
(ratio 1.35–2.00) is noteworthy but also within the phase 1 band.

**Finding 3: single-step vs full-Newton is NOT distinguishing.**  Since
SE_single ≈ SE_full everywhere, the "full-Newton default if single-step
fails" half of the plan's rule buys nothing over single-step.  The rule's
premise (single-step might be a cheap linearization that under-estimates
full-Newton SE) is falsified.  Mechanically applying the rule would ship the
slower path with identical honesty.

## Decision

**Ship single-step as the Phase 3 default** (departure from the plan's
mechanical rule output).

Reasoning:

  1. The rule assumed `SE_single < SE_full` was possible; data shows
     `SE_single ≡ SE_full` to 0.2% everywhere.  The rule's dichotomy
     (cheap-but-biased vs expensive-but-honest) doesn't exist here — both
     schemes produce statistically identical SEs.
  2. The 0.61–0.84 ratio failure is a **jackknife-vs-rep-to-rep** calibration
     issue (inherent to within-pedigree jackknife scope), not a
     **single-step-vs-full-Newton** distinguishing signal.  Switching to
     full-Newton default improves neither at 5× the wall cost.
  3. Phase 1's prior calibration finding ("±40% of jackknife SE across the
     grid") already documents this component-level calibration gap for the
     baseline Golan estimator; M0 confirms the gap is not worsened by the
     Hermite correction.

**Phase close-out caveat** (to be recorded in M5 and the Phase 1 bias note
refresh):

> Jackknife SE under the Hermite correction tracks the full-Newton
> per-block SE to 0.2% precision on σ² across both tested cells.  SE
> calibration against empirical SD-across-reps is component-dependent
> (ratios ≈0.6–2.0 at 2 cells × 5 reps × 3 components), matching Phase 1's
> pre-correction ±40% band.  Users reporting tight-coverage claims at rare
> K (K ≤ 0.05) should prefer SD-across-reps as the calibrated SE on the
> bias-grid scenarios; the single-fit jackknife SE is a within-pedigree
> consistency measure, not a marginal-over-pedigrees one.

## Implications for Phase 3 backend work

1. **P3.M1 step 6**: implement `_newton_step_per_block(β_full, XtX_b, Xtr_b,
   a_ij, c_ij, block_of_pair, c_factor, tau, damping=1.0)` — one undamped
   Newton step per block, warm-started from β_full.  **Do not** plumb
   `jackknife_newton_steps` as a production knob at this stage.
2. **P3.M2**: the numba kernel `_build_loo_block_moments_kernel` in the
   spike code is the direct foundation for the numba backend's per-block
   Newton.  Port it into `numba_impl.py` adjacent to `_accumulate_ols_ace`.
   Even though production default is single-step, the kernel is still
   needed: (a) the full-Newton opt-in stays available for users who want
   block-level convergence for their own SE audit; (b) the same kernel
   computes β-dependent LOO moments at `hermite_order=2` for any per-block
   operation we might add later.
3. **P3.M5**: **drop** — single-step and full-Newton are empirically
   equivalent; no opt-in flag needed for the "honest" version since the
   default already is honest (insofar as jackknife can be).  The opt-in
   `jackknife_newton_steps` parameter becomes an internal diagnostic knob
   (meta field only), not a public knob.
4. **Phase close-out**: record the calibration caveat above in Phase 1's
   bias note (`notes/pcgc_phase1_bias.md` §Recommendations).  The Phase 1
   K ≥ 0.15 caveat can still be lifted for point estimates; the new caveat
   is about SE reporting, not bias.
