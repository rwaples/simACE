# iter_reml Phase 5 roadmap — post-v2026.04 hardening

**Status:** planning, 2026-04-19.  Post-ships of Phase 4 MCEM
(ace_iter_reml `cb33d65`, fitACE `fb941f0`, simACE v2026.04).

Phase 4 delivered working MCEM for n ≤ 2000, K ≥ 0.15 binary-trait
fits.  Four follow-up streams are scoped as Phase 5, ordered by
value-per-effort:

| Milestone | Value | Effort | Ships in |
|---|---|---|---|
| M1: Multi-rep validation + C++↔Python cross-check | **high** | 1-3 days | v2026.05 |
| M2: Continuous/MeanOnce regression harness (P3.D7) | medium | 1 day | v2026.05 |
| M3: Louis observed-info SE + bridge-sampled MCEM logLik | **high** | 4-7 days | v2026.06 |
| M4: Sparse-precision Gibbs (lift n ≤ 2000 cap) | **high** | 1-2 weeks | v2026.07+ |

M1 + M2 can land in a single minor release (`v2026.05`); they don't
touch the algorithm.  M3 is a material statistical upgrade that
unlocks proper SEs and AIC/BIC/LRT — its own release.  M4 is the
scalability story; likely its own phase.

## Individual plans

* [`iter_reml_phase5_m1_validation.plan.md`](iter_reml_phase5_m1_validation.plan.md)
  — multi-rep dev-grid + Python↔C++ cross-check
* [`iter_reml_phase5_m2_regression.plan.md`](iter_reml_phase5_m2_regression.plan.md)
  — Continuous / MeanOnce regression harness (retires P3.D7)
* [`iter_reml_phase5_m3_se_loglik.plan.md`](iter_reml_phase5_m3_se_loglik.plan.md)
  — Louis observed-info SE + bridge-sampled MCEM logLik
* [`iter_reml_phase5_m4_sparse_gibbs.plan.md`](iter_reml_phase5_m4_sparse_gibbs.plan.md)
  — Henderson A⁻¹ sparse-precision Gibbs for n > 2000

## Non-goals (not in Phase 5)

* **Replacing Laplace** — Laplace stays as the default for K ≥ 0.15
  with covariates or n > 2000.  MCEM is an alternative, not a
  successor.
* **New variance components** (D, AxC interaction, etc.) — scope
  would balloon; v1 is A+C+E only.
* **Parallelisation across outer iters** — MCEM is inherently
  sequential.  Parallel chains (K independent streams) is a cheap
  optimisation left for a separate perf pass.
* **GPU offload** — dense V⁻¹ at n=2000 fits in 32 MB; not
  GPU-relevant at v1 scale.

## Phase 5 ship-gate

* M1 + M2 together pass without introducing regressions against
  the Phase 3 baseline (70 iter_reml tests unchanged).
* M3's SEs produce 95% coverage ≥ 85% on the h² dev-grid (matching
  the Laplace Louis-sandwich threshold).
* M4 fits n=10k in <10 minutes at K=0.15 without accuracy loss
  vs v2026.04 at n=2000.

## Risk log

Carried over from Phase 4:

* **Rare-K identifiability** (K ≤ 0.05, n ≤ 1k): not fixable by
  any algorithm — data is underpowered.  Document, don't promise.
* **TMVN mixing on highly-correlated V**: with tight sibships the
  Gibbs chain may require long burn-in.  Currently mitigated by
  warm-starting from previous iter's sample mean; M4's sparse-
  precision Gibbs may change this.
* **β̂ drift**: MCEM's β update is `mean(sample mean)`.  Under
  strong inter-household correlation this could still drift;
  watch in M1 replicate sweeps.

## Decision checkpoints

* **After M1**: do MCEM replicates converge to the same σ² within
  MC noise across seeds?  If high between-seed variance, revisit
  the step rule before M3.
* **After M3**: do Louis SEs give ≥ 85% coverage?  If not, audit
  the derivation (perhaps implement both the outer-product form
  and Oakes form and compare).
* **After M4**: does sparse-precision Gibbs match dense-V⁻¹ MCEM
  on n ≤ 2000 within MC noise?  If not, sparse approach has a bug.
