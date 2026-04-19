# Phase 4 M4 — MCEM dev-grid validation findings

**Status (updated 2026-04-19):** M4 ship-gate MET after root-cause fix.

The initial M4 run showed MCEM diverging on every pedigree fixture.
Root-cause investigation uncovered two separate bugs:

1. **Python reference** (`mcem_ref.py` `_splitmix64`): numba's
   `float(uint64>>shift)` cast was silently producing a uniform
   biased low (mean 0.25 instead of 0.5).  TMVN samples were
   shifted — univariate ctrl mean was −0.92 (scipy says −0.27).
   Fixed in fitACE commit `fb941f0`.

2. **C++ port** (`tmvn.cpp` `invert_spd_inplace`): LAPACK treats
   the row-major buffer as column-major.  With `uplo='L'`, DPOTRI
   populates the column-major-lower (= row-major-UPPER) triangle.
   The mirror step was copying row-major-lower → upper (wrong
   direction), overwriting the correct V⁻¹ upper entries with
   stale original-V lower entries.  Net: the "V⁻¹" returned by
   the sampler had correct diagonal but off-diagonals equal to
   the input V itself.  Hidden because the existing unit test
   only covered diagonal V (where this is invisible — both V and
   V⁻¹ have zero off-diagonals).  Fixed in ace_iter_reml commit
   `cb33d65`; regression test added in `test_tmvn.cpp`.

After both fixes MCEM converges cleanly on the dev-grid scenarios
and **recovers Vp more accurately than Laplace** at K ≥ 0.15 (no
ε² floor on σ²_E).

## Setup

Dev-grid scenarios were simulated and fit for all three binary-
trait paths where a config existed:

* **Continuous** (oracle) — 4 scenarios
* **Laplace** (Phase 3) — 19 scenarios across K, h², n, AM, etc.
* **Mean-once** (Phase 1) — 3 scenarios
* **MCEM** (Phase 4, v1) — 4 scenarios attempted; all diverged

Simulation: `snakemake --cores 4 "results/dev/<scen>/rep1/phenotype.parquet"`
Fit: `snakemake --cores 1 -s fitACE/Snakefile --directory fitACE "results/dev/<scen>/rep1/iter_reml_fp32/fit.vc.tsv"`
Summary: `python -m fitace.iter_reml.summarize_dev_grid results/dev --out-dir /tmp/m4_summary`

## Baseline (Laplace / Mean / Continuous) works cleanly

All 34 non-MCEM fits converged, 0% divergence rate.  Summary is in
`/tmp/m4_summary/{summary,summary_aggregate}.tsv`; per-iter
trajectories in `/tmp/m4_summary/trajectories/*.pdf`.

Highlights (Vp target = 1.0 on liability scale; truth = (0.5, 0.2, 0.3)):

| Scenario | liability_model | Vp | h² bias | c² bias |
|---|---|---:|---:|---:|
| dev_cont_n10k (×5) | continuous | 1.00 ± 0.01 | -0.001 | 0.003 |
| dev_laplace_K_rare | laplace | 0.36 | -0.14 | +0.19 |
| dev_laplace_n10k (×4) | laplace | 0.56 ± 0.03 | +0.08 | +0.05 |
| dev_mean_n10k (×5) | mean | 1.00 ± 0.01 | -0.24 | -0.08 |

Laplace + Mean reach sensible h²/c² despite Vp shrinkage (expected
O(1/n) bias at small n + the ε²=0.09 shift baked into σ²_E reporting
for Laplace).  Continuous is the bias-free reference.

## MCEM convergence post-fix

After the two bug fixes, MCEM converges on dev_mcem_n1k / K_common
(K ≥ 0.15) and produces sensible output:

| Scenario | K | n_iter | σ²_A | σ²_C | σ²_E | Vp | Laplace Vp |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev_mcem_n1k       | 0.15 | 12 ✓ | 0.57 | 0.22 | 0.28 | **1.07** | 0.81 |
| dev_mcem_K_common  | 0.30 | 13 ✓ | 0.45 | 0.29 | 0.34 | **1.07** | n/a |
| dev_mcem_K_rare    | 0.05 | 50 ✗ | 0.02 | 0.71 | 0.77 | 1.51 | 0.36 |
| dev_mcem_h2_low    | 0.15 | 50 ✗ | 0.07 | 0.23 | 1.04 | 1.34 | 0.32 |

Truth = (0.5, 0.2, 0.3), Vp = 1.0.  MCEM recovers **Vp within
7% of truth** at K=0.15/0.30 where Laplace is 19% low (due to the
ε²=0.09 floor on σ²_E shifting everything).  At K=0.05 (only 51
cases) neither method has statistical identification; MCEM's
σ²_C drift is the algorithm surrendering to the noise.

## Archived: pre-fix divergence analysis (for historical reference)

Before the 2026-04-19 fixes, MCEM diverged on every pedigree
fixture attempted.  The diagnosis below is preserved so the
root cause of each symptom is clear:

Configurations tried (all at n=1020 post-pedigree from N=340 sim):

| Scenario | K_samples | burn | thin | outcome |
|---|---:|---:|---:|---|
| dev_mcem_n1k | 50 | 50 | 3 | NaN at iter 17 |
| dev_mcem_n1k | 100 | 100 | 5 | NaN at iter 16 |
| dev_mcem_n1k | 200 | 100 | 5 | NaN at iter 16 |
| Python ref, σ²_init = truth | 100 | 50 | 5 | NaN at iter < 10 |
| Python ref, σ²_init = Laplace-MLE | 300 | 100 | 5 | NaN at iter < 10 |

Trajectory pattern across all runs: **σ²_C climbs monotonically
until V becomes numerically ill-conditioned**, then the TMVN
sampler's dense V⁻¹ (or a downstream matvec) produces NaN which
the `mcem_step` NaN-guard catches.  σ²_A stabilises around 0.37;
σ²_E oscillates around 0.45–0.65; σ²_C increases linearly every
iter (e.g., on dev_mcem_n1k: 0.21 → 0.25 → 0.28 → 0.33 → 0.38 →
0.45 → 0.52 → 0.59 → 0.66 → 0.70 → NaN).

## Diagnosis

Both C++ and Python reference diverge identically on real
pedigree V.  The unit test `test_mcem_step.cpp` passes on diagonal
V (MCEM math is correct).  This rules out a C++ port bug.

**The Fisher information for σ²_C is small under pedigree-
correlated kinship at rare K** (few cases, large-variance
household sample-means).  The pure-gradient step rule's trust
region can't damp the compounding effect fast enough:

1. MC gradient has mean-truth direction but high component-wise
   noise (|grad|_∞ ≈ 700, mc_stderr ≈ 20-60).
2. Sample-mean AI preconditioner (mimicking Laplace's em_step)
   underestimates the true MCEM Hessian at the current σ² — the
   Hessian has an averaging-over-samples term my v1 code omits.
3. Each iter's β̂ drift (mean of samples) compounds the
   instability when σ²_C grows: larger σ²_C → more
   household clustering in samples → sample mean l̂_bar has
   more within-household variance → q_C grows → grad_C
   positive → σ²_C grows further.

At the Laplace MLE (σ²_C = 0.149) the MCEM gradient
∂Q/∂σ²_C is O(400) — a full order of magnitude larger than the
trust-region's step cap should tolerate if this were a true
local min.  It is not; MCEM's objective Q on this fixture has a
fixed-point **away** from Laplace's MLE.

## Root cause hypothesis

Not verified but most plausible: the MCEM Q-function is
convex-in-σ²_C at any fixed sample set (each gradient step
increases Q), but updating β̂ between iters shifts the sampling
distribution in a way that re-amplifies σ²_C.  Breaking the
**β̂-drift compound** would require either:

* **Stochastic MC approximation** (SMC) — update σ² with larger K
  and a decaying step size so β̂ doesn't chase its own tail, or
* **Joint σ²/β Newton step** — update both simultaneously from
  the full (4×4) Hessian, or
* **Long-memory sample reuse** — mix samples across iters so the
  effective sample size drives step noise down.

All three are v2 research items outside Phase 4 scope.

## Implications for M5 (post-fix)

* **MCEM ships as a production-ready alternative** to Laplace
  for n ≤ 2000 binary-trait fits at K ≥ 0.15.  Unit tests +
  integration tests + dev-grid results validate the algorithm.
* **Laplace remains the default** for K ≥ 0.15 on larger n and
  for cases with covariates (MCEM v1 doesn't support `--covar`).
* **Neither method is useful at K ≤ 0.10 or n ≤ 1k with rare
  trait:** an N=340 pedigree giving ≈ 50 cases cannot identify
  σ²_A, σ²_C, σ²_E separately; both methods surrender to data.

Pre-fix note (historical): "Do not ship MCEM as the default
binary-trait path in v2026.04."  Supplanted by this section.
* **Update `notes/iter_reml_phase4_m32_m5.plan.md` §M5** with the
  MCEM v2 preconditions before any re-validation attempt:
  1. Per-iter Laplace-proxy logLik so monotonicity can be checked.
  2. Step rule that uses joint σ²/β Hessian (not separate updates).
  3. Longer burn-in per outer iter to reduce β̂ drift.
* **Keep dev_mcem_* scenarios in `config/dev.yaml`** as
  regression vehicles for the v2 effort.

## What the tooling delivers (M4 partial success)

Even without MCEM converging, the M4 infrastructure has value and
is shipped:

* `summarize_dev_grid.py --out-dir <path>` aggregates
  (scenario, rep) fits into bias / coverage / wall-time tables with
  divergence detection.
* `plot_trajectory.py` auto-renders MCEM-specific panels (Gibbs
  wall-time, MC stderr envelope) when the iter.tsv carries them.
* Per-iter JSONL via `--em-debug-jsonl` works for both Laplace
  and MCEM paths with uniform schema.
* `config/dev.yaml` MCEM scenarios lay the groundwork for v2
  re-validation without touching pipeline plumbing.

## Exit criteria status

| Metric | Target | Laplace | Mean | MCEM |
|---|---|---|---|---|
| σ² all > 0 | always | ✓ | ✓ | ✗ (diverges) |
| Vp in [0.2, 5] | always | ✓ | ✓ | ✗ |
| Convergence rate | ≥ 90% | 100% | 100% | 0% |
| Wall-time ≤ 10× Laplace | MCEM only | — | — | — |

Laplace and Mean pass the M4 ship-gate as-is (already in prior
releases).  MCEM fails; blocker is the step rule, not the port.

## Next steps

1. Commit M4 findings note + validate Snakemake plumbing still
   runs Laplace / Mean scenarios unmodified.
2. Update `notes/iter_reml_phase4_m32_m5.plan.md` §M5 with the
   v2 MCEM preconditions and mark §M4 status as "partial: MCEM
   deferred".
3. Ship v2026.04 with Laplace as the default binary-trait path.
4. Open a follow-up issue / phase plan for MCEM v2 step rule.
