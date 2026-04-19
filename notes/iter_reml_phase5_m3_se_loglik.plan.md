# Phase 5 M3 — Louis observed-info SE + bridge-sampled MCEM logLik

**Status:** plan, 2026-04-19.  Ships in v2026.06.  Blocked on
M1 replicate-variance baseline.

## Goal

Close the two Phase 4 §M5.4 deferrals:

1. **Proper SEs for MCEM.**  Currently `se_sigma2 = NaN` for
   MCEM fits; the AI-derived SE formula from Phase 3 Laplace
   doesn't apply because MCEM's AI is not the Fisher information
   (it's a Hutchinson estimate of `tr(V⁻¹·M_k·V⁻¹·M_l)` under the
   posterior, not the prior).  Louis (1982) gives the standard
   correction.

2. **MCEM log-likelihood.**  Currently `logLik_em = NaN` for MCEM
   iters; the Laplace logLik formula doesn't apply.  Bridge
   sampling (Meng & Wong 1996) gives a consistent estimate of
   `log p(y | σ², β)` using the TMVN samples we already have.
   Enables AIC / BIC / LRT + per-iter monotonicity checks.

## Ship criteria

1. **SE coverage** (Louis, v2 MCEM):
   * `notes/iter_reml_phase5_m3_findings.md` reports 95% CI
     coverage on dev-grid (n=1020, K=0.15, 50 reps per scenario).
   * Coverage ≥ 85% on h² matches the Laplace Louis-sandwich
     benchmark from Phase 3.

2. **MCEM logLik monotonicity**:
   * Per-iter `logLik_em` increasing on convergent runs (modulo
     MC noise bounded by bridge-sampling stderr).
   * AIC / BIC comparable against Laplace on overlapping scenarios
     (MCEM's AIC should not be > Laplace's AIC by more than 2 log
     units on K ≥ 0.15 fixtures — otherwise MCEM's extra noise is
     hurting model comparison).

3. **Release note update**: `README.md` path-selection table adds
   a "has SE?" column; currently MCEM column reads "NaN in v1".

## Out of scope

* Frequentist confidence intervals beyond ±1.96·SE (e.g., profile
  likelihood CIs) — possible extension but not in this milestone.
* Bayesian credible intervals from the TMVN posterior — different
  question, not Phase 5.

## Background: Louis observed info

EM observed information (Louis 1982):

    I_obs(σ²) = -∂²Q/∂σ²∂σ²ᵀ
             = I_Fisher(σ²) - Cov_posterior(score)

where the Fisher info is the negative expected Hessian of the
complete-data log-likelihood (which MCEM's per-sample AI already
estimates), and `Cov_posterior(score)` is the posterior covariance
of the score function.

For our model:
    score_k(l) = -½·tr(V⁻¹·M_k) + ½·(l-Xβ)'·V⁻¹·M_k·V⁻¹·(l-Xβ)
              = -½·tr(V⁻¹·M_k) + ½·u'·M_k·u   with u = V⁻¹·(l-Xβ)

So `Cov_posterior(score)[k,l]` is estimated by the MC covariance
of the per-sample score, which is just
`Cov_K(u'_s·M_k·u_s, u'_s·M_l·u_s)` we already compute for the MC
stderr on the gradient.

Full Louis info:
    I_obs[k,l] = AI[k,l] - ¼·Cov_K(u'·M_k·u, u'·M_l·u)

where AI is the per-sample-averaged Fisher info (current M3.2.v2
AI matrix in `em_mstep.cpp`), and the covariance term is computed
across the K samples.

## Background: bridge sampling for logLik

We want `log p(y | σ², β)` where
    p(y | σ², β) = ∫ p(y | l) · p(l | σ², β) dl

For binary-threshold:
    p(y | l) = ∏_i 1_{y_i = 1 ⇒ l_i > τ, y_i = 0 ⇒ l_i ≤ τ}

So `p(y | σ², β) = P(l ∈ R_y | σ², β)` where R_y is the half-space
orthant corresponding to y.  For rare K / large n this is
astronomically small — direct MC is hopeless.

Bridge sampling (Meng & Wong 1996) estimates ratios:

    p(y) = [E_{q(l)} α(l)·p(y, l) / q(l)]
         / [E_{p(l|y)} α(l)·p(y, l) / p(y, l)]

with the bridge function α optimally chosen.  Simplest practical
form (iterative bridge sampling):

    r_hat = [ (1/K) Σ_{l_s ~ p(l|y)} q(l_s) / (s₁·p(l_s|σ²,β) + s₂·q(l_s)/r_hat) ]
          / [ (1/K') Σ_{l'_t ~ q(l)} p(l'_t|σ²,β) / (s₁·p(l'_t|σ²,β) + s₂·q(l'_t)/r_hat) ]

where
    r = p(y) itself (the marginal we want)
    q(l) is a proposal (we'll use Laplace approximation)
    s₁ = K / (K + K')
    s₂ = K' / (K + K')
    samples from p(l|y) come from the MCEM TMVN chain (already
                have these for the M-step)
    samples from q(l) are cheap Gaussian draws from the Laplace
                approximation N(l̂, H⁻¹)

One iteration: set r_hat = 1, compute the formula, update r_hat,
iterate until convergence (usually 3-5 iters sufficient).

Cost: K' = K extra Gaussian draws (trivial) + K_bridge iterations
on the bridge formula.  Per-iter cost: 2·K·n ops.  Negligible
vs the MCEM Gibbs work.

## Milestones

### M3.1 — Louis SE for MCEM (~2 days)

Extend `em_mstep.cpp`:

```cpp
// In mcem_step per-sample loop, also accumulate:
//   cov_score[k][l] += (q_k^s - q_bar_k) * (q_l^s - q_bar_l)

// Post-loop:
//   cov_K[k][l] = cov_score[k][l] / (K - 1)
//   I_obs[k][l] = AI[k][l] - 0.25 * cov_K[k][l]

// Emit in MCEmStepResult as observed_info matrix.
```

In `ai_reml.cpp` MCEM post-loop: invert `I_obs` instead of leaving
`result.cov` as zero.  SPD check + fallback to AI⁻¹ if not SPD
(can happen with small K or strong collinearity).

Update `em_mstep.h` + result-row schema accordingly.

Test: `tests/iter_reml/test_mcem_se.py`:
* SE finite, positive, monotone in K (more samples → smaller SE).
* Coverage simulation: n=1020, K=0.15, 50 reps; count how often
  the 95% CI for h² contains the true h².

### M3.2 — Bridge-sampled MCEM logLik (~3-4 days)

1. New function in `em_mstep.cpp`:

   ```cpp
   double mcem_bridge_loglik(
       const PhenotypeLayer& layer,
       Mat V, Vec l_hat, const std::vector<double>& W_diag,
       const std::vector<double>& tmvn_samples,  // from mcem_step
       int K_bridge,        // extra Gaussian proposal draws
       std::uint64_t seed);
   ```

2. Proposal `q(l)` = Laplace approximation `N(l̂, H⁻¹)` where l̂ is
   the Laplace mode at the current σ² (cheap: one Newton-PCG
   mode-find, already wired for `LiabilityLaplace`) and H⁻¹ is
   approximated by the Woodbury-form posterior from Phase 3.

3. Iterate bridge formula to convergence (3-5 iters, τ ≤ 1e-4).

4. Return `log p(y | σ², β)` estimate + bridge-sampling stderr.

Per-iter call inside the MCEM branch of `run_ai_reml` (at iter
end, similar to the Laplace branch's `em_logLik` call).  Gated
by `opts.em_per_iter_loglik` (existing flag).

Post-convergence: `AiRemlResult.logLik` populated from the last
iter's bridge-sampled logLik.

Test: `tests/iter_reml/test_mcem_bridge_loglik.py`:
* On a Gaussian fixture (not binary — where log p(y) has a
  closed form via dense V's log-det), MCEM-via-bridge should
  recover log p(y) within ±0.5 log units at K=500.
* On a binary fixture, logLik should be monotonically increasing
  across outer iters (up to bridge MC noise).

### M3.3 — Dev-grid re-validation (~1 day)

Re-run dev_mcem_* with v2026.06 (Louis SE + bridge logLik).
`summary_aggregate.tsv` gets new columns:
* `coverage_h2` (fraction of 95% CIs containing truth)
* `bridge_logLik_mean` ± SD
* `aic_mcem_minus_laplace` (negative = MCEM fits better)

Write `notes/iter_reml_phase5_m3_findings.md`.

### M3.4 — Docs + CLI (~half day)

* Update `fitace/iter_reml/README.md` — MCEM SE no longer NaN;
  logLik usable for AIC / BIC.
* CLI flag: `--mcem-bridge-K` (default 100) to tune bridge
  sampling cost.  Document in `ace_iter_reml --help`.
* Update `fit.py` wrapper to expose `mcem_bridge_K`.

## Testing

* New C++ tests: `test_louis_info` (synthetic), `test_bridge_loglik`
  (Gaussian fixture with known answer).
* New Python tests: `test_mcem_se.py`, `test_mcem_bridge_loglik.py`.
* Cumulative test count: ~80 → 90.

## Risks

* **Louis info non-SPD at low K**: with K=50, per-sample Cov
  estimate is noisy; I_obs may be non-SPD.  Mitigation: SPD
  fallback to AI⁻¹ (documented).  Increase K for production use.

* **Bridge sampling variance**: if q(l) = Laplace(σ²) is far
  from p(l | y, σ², β) (eg at rare K where Laplace biases), the
  bridge estimator has high variance.  Mitigation: use the
  v2026.04 MCEM's l̂_bar as the proposal mean instead of Laplace
  mode.  Investigate during M3.2.

* **AIC comparison fragility**: AIC = 2k - 2·logLik.  If
  MCEM's logLik has bridge stderr of 3-5 log units, AIC
  differences below ~6 are noise.  Document and don't over-
  interpret.

## Effort rollup

| Sub-milestone | Effort | Cumulative |
|---|---:|---:|
| M3.1 Louis SE | 2 days | 2 |
| M3.2 Bridge logLik | 3-4 days | 5-6 |
| M3.3 Dev-grid re-validation | 1 day | 6-7 |
| M3.4 Docs + CLI | 0.5 day | 6.5-7.5 |

Total: 4-7 working days.  Order-of-magnitude one week wall-clock
for a focused dev, with M3.2's bridge sampling as the dominant
cost (research-grade code + validation).

## Decision: Oakes vs Louis vs sandwich

Three candidate SE formulas for MCEM:

1. **Louis (as above)**: `I_obs = AI - ¼·Cov_K(score)`.  Classical,
   well-behaved when K is adequate.  First choice.

2. **Oakes identity**: `I_obs = -dQ/dσ² · (I + ∂²E[log p]/∂σ²∂σ²)⁻¹ ...`
   Avoids the score covariance but requires a second MC estimate.
   Backup if Louis gives poor coverage.

3. **Bootstrap sandwich**: resample the TMVN chain in blocks,
   recompute σ². Non-parametric but expensive (need K ≥ 200).
   Plan B if both 1 + 2 fail.

Plan defaults to Louis (option 1).  If M3.1's coverage is poor,
implement option 2 (Oakes) as a CLI-toggled alternative and
compare on the dev-grid.
