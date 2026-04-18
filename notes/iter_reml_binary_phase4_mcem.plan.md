# iter_reml — Phase 4: MCEM for rare-K binary traits

**Status:** plan, 2026-04-19.
**Predecessor:** `iter_reml_phase3_ship_ready.md` — Phase 3 is
algorithmically correct but the Laplace approximation has bias at
small N / rare K that Phase 3 cannot fix.
**Motivation:** rare-K applications (K ≤ 0.05) are a priority and
the dense-reference investigation showed Laplace is nearly
degenerate there (interior-MLE vs all_E gap ≈ 4 log units at K=0.05
on n=1k).

## Why MCEM

Laplace approximates `p(l | y, σ², β) ≈ N(l̂, H⁻¹)` — a local
Gaussian around the posterior mode.  For the threshold model this
is a bad approximation when:

* `y` is highly imbalanced (rare K) — posterior is heavily skewed
  at the truncation boundary.
* N is small — posterior has wider tails, Gaussian approximation
  misses mass.
* σ² is near zero — posterior is concentrated, H is nearly
  singular.

**MCEM replaces the Laplace approximation with Monte Carlo samples**:

```
E-step (Monte Carlo):
    draw K samples l^(k) ~ p(l | y, σ²^(t), β^(t))
    these are samples from a truncated multivariate normal (TMVN)
        l | y ~ TMVN(Xβ, V, bounds = {l_i > τ if y_i=1, l_i < τ else})
    compute Monte Carlo estimates of E_q[(l−Xβ)'V⁻¹·M_k·V⁻¹(l−Xβ)],
    E_q[tr(V⁻¹·Σ·M_k)] etc., replacing the Laplace (μ, Σ=H⁻¹) with
    (sample mean, sample covariance).

M-step:
    same as Phase 3 EM — update (σ², β) by ascending Q.
    Gradient is now the Monte Carlo estimate; AI matrix optional.
```

MCEM converges to the EXACT EM fixed point (no Laplace bias)
provided K is large enough.  The trade-off is wall-time: K
samples × outer iters of TMVN sampling ≈ 5k–50k total TMVN draws.

## What changes vs Phase 3

| Piece | Phase 3 (Laplace) | Phase 4 (MCEM) |
|---|---|---|
| E-step | `find_laplace_mode` — 6–8 Newton-PCG iters | TMVN sampler — K samples at ~O(n²) each |
| Posterior moments | `μ = l̂`, `Σ = H⁻¹` (Woodbury-applied) | sample mean `μ̂`, sample covariance `Σ̂` |
| q_k in gradient | `(μ−Xβ)'V⁻¹·M_k·V⁻¹(μ−Xβ)` analytic | Monte Carlo mean over samples |
| tr(V⁻¹·Σ·M_k) | Hutchinson + Woodbury via S | Sample-covariance Monte Carlo |
| logLik | via SLQ on S + ½log|S| | bridge sampling / path sampling |
| Observed info | finite-diff of gradient | louis-style sandwich over samples |
| Wall-time at n=10k | 200–500 s | **~2000–10000 s** (estimate) |

## The hard problem: TMVN sampling

Sampling `l | y ~ TMVN(Xβ, V, half-space truncations)` at
n = 10k with sparse V (union(A, C) + diag) is non-trivial.

### Candidate samplers

**Option A — Gibbs with structured block updates.**  Update
`l_i` from its conditional `l_i | l_{−i}, y_i`, which is
univariate-truncated-normal.  Full-conditional mean/var computed
from V⁻¹ via sparse solves.  Requires O(n) sweeps per sample;
each sweep is O(nnz(V)) work.  Slow to mix for correlated
pedigrees (ρ close to 1 between kin).  Reference: Albert & Chib
(1993).

* **Pro**: simple, no preprocessing, works out of the box.
* **Con**: mixing is bad for strongly correlated pedigrees; may
  need O(1000) sweeps per sample.

**Option B — Botev (2017) minimax exponential tilt.**  Exact
rejection sampler using a tilted exponential proposal.  Requires
Cholesky of V (dense — O(n³) preprocessing at n=10k is ~100 s).
Each sample is O(n²) after preprocessing.  State-of-the-art for
TMVN at moderate n.

* **Pro**: exact (rejection), fast samples once Cholesky is built.
* **Con**: dense Cholesky at n=10k is 800 MB and 100+ s; at n=50k
  it's 20 GB and 10⁴ s.  Not feasible at production scale without
  extension to sparse Cholesky.

**Option C — Sparse Cholesky + Botev.**  Replace the dense
Cholesky with a sparse one (e.g., `scipy.sparse.linalg.splu` or
CHOLMOD).  Keeps preprocessing tractable at n=10k.  Botev's
tilting then works on the sparse factor.  Unclear whether the
tilting efficiency degrades.  Research-level.

**Option D — Hamiltonian Monte Carlo (HMC) on TMVN.**  HMC
leverages the gradient of log-TMVN and reflects at truncation
boundaries.  Good mixing, implementable in Stan natively.  But
requires gradient of the log-prior `−½(l−Xβ)'V⁻¹(l−Xβ)` which
means a V⁻¹ apply per leapfrog step — same PCG cost we already
have per Laplace mode-find step, × 100+ leapfrog steps × K
samples × outer iters.  Probably too slow.

**Option E — Exact Gibbs with sparse precision.**  V⁻¹ is sparse
for pedigree models (Henderson decomposition — each individual's
prior depends only on parents).  Use V⁻¹ directly (not V), express
conditionals in precision form: `l_i | l_{−i}, y_i ∝
exp(−½·τ_ii·(l_i − m_i)²) · 1[y_i = 1[l_i > τ]]` with precision
`τ_ii = (V⁻¹)_ii` and mean `m_i` computed from neighbours via
V⁻¹.  Avoids O(n³) preprocessing.  Per-sweep cost: O(nnz(V⁻¹)).
Reference: Fan & Pantula (1997) for the sparse precision form.

* **Pro**: O(nnz) per sweep, direct extension of Phase 3's PCG
  infrastructure (already have V⁻¹·x).
* **Con**: still Gibbs so mixing may be slow.

### Recommended path

**Start with Option E (sparse-precision Gibbs).**  Reuses the
existing V⁻¹·x infrastructure (via PCG), avoids dense Cholesky,
simple to implement.  Mix it with **Option A** fallback for blocks
where the sparse precision form has trouble.  Escalate to
**Option C** (sparse Cholesky + Botev) only if mixing is
prohibitively slow on real-data pedigrees.

Pilot study on the n=1k K=0.15 fixture first — see if 50 sweeps
per sample give good mixing (effective sample size ≥ 25%).

## Implementation outline

### M1: TMVN prototype in Python (2-3 days)

* `fitace/iter_reml/mcem/tmvn.py`: TMVN sampler using Option E
  (sparse-precision Gibbs) on top of the existing dense reference's
  V⁻¹ inverse.  Sample count K configurable.
* `fitace/iter_reml/mcem/ref_mcem.py`: MCEM reference optimiser —
  E-step via TMVN, M-step via analytic gradient (same formula as
  Phase 3 but without the log|H|-via-l̂ correction, since moments
  come directly from samples).
* Validation: run on the n=1k K=0.15 fixture, compare MCEM σ² to
  truth and to dense Laplace MLE.  Expected: MCEM σ² should be
  closer to truth than Laplace MLE.

### M2: Mixing / variance analysis (2 days)

* Burn-in / thinning tuning for Gibbs on pedigree-correlated
  data.
* Effective sample size (ESS) diagnostics.  Compare to IID samples
  from the dense-reference mode-Laplace as a sanity check (MCEM
  samples should have mean ≈ Laplace mode).
* Monte Carlo variance of σ² gradient vs K — how many samples
  needed for 10% SE on σ² estimate?

### M3: C++ port (5-7 days)

* `fitace/ace_iter_reml/src/tmvn.{h,cpp}`: sparse-precision Gibbs.
  Reuse existing PCG solver for the mean computation and existing
  sparse V⁻¹·x infrastructure.  OMP-parallelise the K samples
  (different RNG streams per thread).
* Extend `EmStepInputs` with `liability_model = "mcem"` branch.
* M-step reuses the Phase 3 gradient machinery (replace
  analytic-Σ with sample-Σ).
* CLI: `--liability-model mcem --mcem-samples 100 --mcem-burnin 50 --mcem-thin 10`.

### M4: Integration + validation (3-5 days)

* Run on the full dev grid, compare to Phase 3 Laplace.
* Coverage at small-N rare-K: verify bias < 0.05 and SE coverage
  ≥ 90% on known-truth replicates.
* Wall-time: measure / tune K to hit a reasonable budget (target
  ≤ 10× Phase 3 Laplace).

### M5: Release + docs (1-2 days)

* Release notes documenting when to use Laplace vs MCEM.
* Update `iter_reml_phase3_ship_ready.md` with MCEM as the rare-K
  escape hatch.

**Total estimate: 13-19 days.**

## Risks

| Risk | Mitigation |
|---|---|
| Gibbs mixing slow on strongly correlated pedigrees (ρ > 0.5 between sibs) | Burn-in longer, thin, or switch to sparse-Cholesky Botev (Option C) |
| Monte Carlo noise on σ² gradient too large at feasible K | Variance reduction: antithetic pairs, control variates, Rao-Blackwellisation |
| Wall-time prohibitive at n ≥ 50k | Profile / parallelise samples; at scale, consider importance sampling from the Laplace approximation |
| Post-convergence logLik (needed for LRT / AIC / BIC) harder than in Phase 3 | Bridge sampling or path sampling — established methods; ~1-2 days extra |
| Dense reference no longer tractable as ground truth at n > 2k | Use Stan for ground truth on moderate fixtures |

## Open questions before starting

1. **Is matching truth exactly the goal, or is matching Stan
   posterior-mean enough?** MCEM's MLE will differ from Stan's
   posterior mean by the prior-induced shrinkage.  If we want
   truth-alignment, we may need a prior-adjusted score (restricted
   ML = REML in the Gaussian world).

2. **Do we need per-iter uncertainty, or only final SE?** MCEM can
   produce posterior-draws of σ² (via the complete-data score's
   covariance × K outer iters).  Cheap bonus.

3. **Does Phase 3 Laplace stay as a fast default?** I'd say yes —
   Laplace is fine at n ≥ 10k K ≥ 0.15 and 10× faster.  MCEM only
   for rare-K applications.  Two `liability_model` options.

4. **What's the minimum-viable scope for Phase 4?** Question for
   user: ship MCEM python reference first (M1-M2 only) for
   research use, defer C++ port to Phase 5?  Or C++ port is
   required for shipping?

## Cold-start reading order

1. `iter_reml_phase3_ship_ready.md` — what Phase 3 does + its
   Laplace limitation.
2. `iter_reml_phase3_d6_findings.md` — empirical evidence of
   Laplace bias from the dense reference.
3. This file — Phase 4 plan.
4. `fitace/iter_reml/ref_dense.py` — dense Laplace reference as a
   working example for the MCEM reference.
5. Albert & Chib (1993) "Bayesian Analysis of Binary and
   Polychotomous Response Data" for the Gibbs sampler pattern.
6. Wei & Tanner (1990) "A Monte Carlo Implementation of the EM
   Algorithm" for MCEM convergence theory.
