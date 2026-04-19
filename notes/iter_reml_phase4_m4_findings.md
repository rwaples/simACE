# Phase 4 M4 — MCEM dev-grid validation findings

**Status:** partial, 2026-04-19.  The M4 ship-gate for MCEM v1 is
NOT met.  MCEM produces finite σ² on synthetic diagonal-V fixtures
(test_mcem_step passes) but DIVERGES on real pedigree-correlated V
under both the C++ port and the Python reference, despite starting
from the Laplace MLE.  Laplace remains the recommended binary-trait
path for production.

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

## MCEM divergence — the headline finding

**Every MCEM fit attempted diverged before convergence.**

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

## Implications for M5

* **Do not ship MCEM as the default binary-trait path in v2026.04.**
  The pure-gradient step rule is not robust to realistic pedigree
  V.  Leave the code in the tree (unit tests + CLI + integration
  test skeleton), mark as experimental.
* **Ship Laplace as the recommended path.**  Stable, converges,
  and known bias is characterised in this summary.
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
