# iter_reml — Phase 3: full PX-EM outer loop for binary-trait MLE

**Status:** plan, post Phase 2 landing (2026-04-18).
**Predecessor:** `iter_reml_binary_phase2.plan.md` (Phase 2 Laplace + ε-annealing landed but σ²_E collapses on threshold data).
**Motivation:** engineering findings from the Phase 2 dev-scenario sweep.

## Why Phase 3 exists

Phase 2 landed the Laplace posterior-mode refresh inside the existing
AI-REML outer loop and surfaced a **real, quantitative bias** that the
retrofit cannot address:

On the dev-scenario grid (24 scenarios × 1–5 reps, fp32):

| Path | h² mean (truth 0.500) | σ²_E mean (truth 0.300) |
|---|---:|---:|
| Continuous oracle | 0.499 ± 0.029 | 0.298 ± 0.025 |
| Phase 1 Mean | 0.262 ± 0.043 | 0.623 ± 0.041 |
| **Phase 2 Laplace** | **0.584 ± 0.049** | **0.090 ± 0.000** (pinned) |

σ²_E hits the ε²_end floor (0.09) in *every* Laplace fit, *every*
replicate — SD across reps = 0.000. Vp < 1 across all Laplace
scenarios. The collapse mechanism:

- The Gaussian-REML gradient treats the mode `l̂` as if it were a
  fixed observation: `grad_k = ½·(q_k − tr(V⁻¹·M_k))`.
- The mode `l̂` actually *depends* on σ² — specifically, `l̂` shrinks
  toward τ for marginal cases/controls as σ² drops. So `var(l̂)` is
  less than a true random sample from `N(Xβ, V)` would have.
- REML, seeing an `l̂` with too-low variance, finds a σ² that
  matches. σ²_E is the "residual" component; it takes the full hit.
- Feedback: smaller σ²_E → tighter `l̂` (posterior collapses) →
  smaller σ²_E → floor.

**What wouldn't fix it** (tried in Phase 2 as P2.7a):
- Adding log|H|-gradient correction via Woodbury (`grad_k += ½·tr(H⁻¹·V⁻¹·M_k·V⁻¹) = ½·(tr_k − tr(W^½·M_k·W^½·S⁻¹))`). Math is right, gradient becomes bounded, but the AI matrix paired with it is still Gaussian-REML curvature. Newton steps mis-size; σ²_E still hits the floor.

**What would fix it** (this plan):
- Separate E- and M-steps with their own objective, gradient, curvature, and convergence criterion. Use the fact that the M-step *Q-function* has a known closed-form derivative and Hessian (given the E-step's second moments), not AI-REML's Gaussian profile.

## The math we need

### Model (unchanged)

```
l ~ N(Xβ, V(σ²))     V = σ²_A·A + σ²_C·C + σ²_E·I
y_i = 1[l_i > τ]      τ = Φ⁻¹(1 − K)
```

### EM framework

Treat `l` as missing data. At iteration `t`:

**E-step** — given `σ²^(t)`, `β^(t)`, compute:
- `μ^(t) = E[l | y, σ²^(t), β^(t)]`  — conditional mean
- `Σ^(t) = Cov[l | y, σ²^(t), β^(t)]` — conditional covariance

Exact posterior is a truncated MVN. Laplace approximation:
- `μ^(t) ≈ l̂^(t)` (posterior mode — already computed by Phase 2's `find_laplace_mode`)
- `Σ^(t) ≈ H⁻¹(l̂^(t), σ²^(t))` where `H = V⁻¹ + W(l̂)`

**M-step** — maximise the `Q`-function:
```
Q(σ², β ; σ²^(t), β^(t)) = E_{l|y,σ²^(t),β^(t)} [log p(y, l | σ², β)]
                          = const + log p(y|l̂) (does not depend on σ²)
                                  + log p(l̂|σ², β)_adjusted
```
where the adjusted log-prior is:
```
E_q[log p(l|σ²,β)] = -½ log|V(σ²)|
                    - ½ (μ^(t) − Xβ)'·V⁻¹·(μ^(t) − Xβ)
                    - ½ tr(V⁻¹·Σ^(t))
```

**Key difference from Phase 2:** the last term `−½·tr(V⁻¹·Σ)` is the
posterior-variance correction that prevents σ²_E from collapsing.

### M-step gradient

```
∂Q/∂σ²_k = -½ tr(V⁻¹·M_k)
          + ½ (μ − Xβ)'·V⁻¹·M_k·V⁻¹·(μ − Xβ)
          + ½ tr(V⁻¹·M_k·V⁻¹·Σ)
```

With `Σ = H⁻¹` and via Woodbury (see Phase 2 P2.7a notes), the third
term simplifies:
```
tr(V⁻¹·M_k·V⁻¹·H⁻¹) = tr(V⁻¹·M_k) − tr(W^½·M_k·W^½·S⁻¹)
                              └────────── computed identically
                                          across VCs via Hutchinson
```
So:
```
∂Q/∂σ²_k = ½ (q_k^μ − tr(W^½·M_k·W^½·S⁻¹))
```
where `q_k^μ = (μ − Xβ)'·V⁻¹·M_k·V⁻¹·(μ − Xβ)`.

This is the gradient Phase 2 P2.7a tried to use. **The failure in
P2.7a was not the gradient** — it was pairing it with the wrong
curvature.

### M-step Hessian (this is what P2.7a missed)

```
∂²Q/∂σ²_k ∂σ²_l = -½ tr(V⁻¹·M_k·V⁻¹·M_l)        [standard]
                + tr(V⁻¹·M_k·V⁻¹·M_l·V⁻¹·Σ)     [new]
                - (μ−Xβ)'·V⁻¹·M_k·V⁻¹·M_l·V⁻¹·(μ−Xβ)  [Gaussian q-term]
                - ½ tr(V⁻¹·M_k·V⁻¹·M_l·V⁻¹·Σ)   [posterior-Σ curvature]
```

Several of these are awkward but tractable. The simpler AI-style
approximation (using outer products of score terms):

```
AI_kl^EM = ½ [e_k · w_l + e_l · w_k] + second-moment terms via Σ
```

**Practical simplification** (widely used in REML-for-GLMMs): just use
the AI-REML curvature on the *adjusted* quadratic form, i.e., replace
`u = V⁻¹·(l̂ − Xβ)` with `u_adjusted` derived from both `μ` and `Σ`.
One principled choice:

```
AI_kl^EM ≈ ½ · w_k' · (V⁻¹·[μμ' + Σ]·V⁻¹ − V⁻¹) · w_l · ... (TODO)
```

**This needs careful derivation before coding.** An alternative that
avoids deriving a new AI matrix: use a **damped gradient step** in the
M-step (Newton with line search), not Newton–Raphson with the AI
curvature. Slower but robust.

### Convergence

Monotone-ascent guarantee: `Q(σ²^(t+1), β^(t+1); σ²^(t), β^(t)) ≥ Q(σ²^(t), β^(t); σ²^(t), β^(t))` holds for any M-step that improves Q
(Generalised EM). Full max is not required — a single damped
gradient step counts. This is what makes GEM robust.

The outer EM loop converges monotonically in the true Laplace-
marginalised log-likelihood:
```
log L_Laplace(σ²^(t+1)) ≥ log L_Laplace(σ²^(t))
```
(up to Laplace approximation error, which is O(1/n)).

## Implementation plan

### Option A — GEM with damped gradient M-step (recommended)

Simpler, more robust. ~4-5 days of focused work.

**Step 1: dedicated M-step function.** New `src/em_mstep.{h,cpp}`:

```cpp
struct MStepResult {
    std::array<double, 3> sigma2;   // updated σ²
    std::vector<double> beta;       // updated β (if X set)
    double Q_value;                  // Q at new parameters
    bool converged;
    int inner_iters;                 // line-search / backtrack count
};

MStepResult em_mstep(
    const std::array<double, 3>& sigma2_current,
    const std::vector<double>& beta_current,
    const PhenotypeLayer& layer,       // has l̂ = mode
    VCModel& model, Mat V,
    PcgSolver& pcg,
    const AiRemlOptions& opts);
```

Internals:
1. Compute `u = V⁻¹·(l̂ − Xβ)` (1 PCG).
2. Compute `e_k = M_k·u`, `w_k = V⁻¹·e_k` (3 matvecs + 3 PCGs) — as in Phase 2 AI-REML.
3. Compute Hutchinson estimator of `tr(W^½·M_k·W^½·S⁻¹)` (m probes × 1 CG-on-S) — already scaffolded in `compute_trace_corrections`.
4. EM gradient: `grad_k = ½·(q_k − tr_WMWS_k)`.
5. Damped gradient step: `Δσ² = α · grad`. Armijo backtracking:
   - Start with α = 1/max(|AI_diag|) (heuristic).
   - Evaluate `Q(σ² + αΔσ²)`: requires recomputing `l̂` under trial σ² (EXPENSIVE).
   - Backtrack: halve α until Q increases.
6. Alternative: *fixed-step GEM* — take one small step per outer iter (α ∝ 1/(1+t)) without line search. No re-mode-find per M-step. Simpler.

**Step 2: outer EM loop.** Replace `run_ai_reml`'s outer loop with an EM-specific driver for LiabilityLaplace layers:

```cpp
for t = 0 .. max_iter:
    // E-step (Phase 2 Laplace mode + W diagonal)
    refresh_V(V, model)
    pcg.set_operator(V)
    if (layer == Laplace) {
        find_laplace_mode(layer, Xβ, σ², V, laplace_opts)
        // layer.l_hat and layer.laplace.W_diag are now at σ²^(t)
    }
    // M-step (either AI-REML for Gaussian/MeanOnce, or em_mstep for Laplace)
    if (layer == Laplace) {
        mstep_result = em_mstep(σ², β, layer, model, V, pcg, opts)
    } else {
        // existing AI-REML step (Phase 1 path preserved)
    }
    // Convergence: ΔQ or Δσ²
```

Keep Phase 2's `run_ai_reml` intact for continuous / MeanOnce layers.

**Step 3: EM convergence.** Monitor `ΔQ / n < tol_em` instead of
Gaussian `dLLpred < tol_g`. Fallback to `max|Δσ²_k|/(|σ²_k|+ε) < tol_step`.

**Step 4: post-convergence Q and info matrix.** After outer loop:
- Report `logLik_Laplace = Q(σ²_final, β_final) - ½ log|H(l̂_final)|`.
- Observed information matrix `I_obs(σ²) = -∂²log L_Laplace/∂σ²²` at σ²_final: use numerical differentiation (3 extra mode-finds + 3 EM gradient evaluations). ~30s extra at n=10k.
- Sandwich SE correction: already landed in Phase 2; will kick in properly once σ²_E isn't boundary-pinned.

**Testing strategy:**
1. **Unit**: `em_mstep` monotonic-ascent on toy n=100, σ² rank-1. Finite-difference verify Q values along gradient.
2. **Integration**: rerun dev scenarios. Expect σ²_E recovery near truth (0.3), Vp near 1, h² recovery within ±0.03.
3. **Regression**: Continuous/MeanOnce paths unchanged (existing `run_ai_reml` reused).

### Option B — Proper Newton M-step with EM AI matrix

Faster per outer iter, but requires deriving the right AI matrix. ~1.5-2 weeks.

Not recommended as first attempt — the GEM approach in Option A will converge and give correct estimates, just slower. Optimise later if wall-time becomes a bottleneck.

### Option C — MCEM (stochastic imputation)

E-step samples K draws from truncated-normal posterior `p(l | y, σ², V)` instead of computing the mode. M-step averages σ² gradient over samples. Naturally handles posterior variance without Laplace approximation.

Pros: avoids Laplace bias entirely.
Cons: truncated-MVN sampling is non-trivial (Botev 2017 minimax tilt, or Gibbs with long chains on correlated pedigrees); Monte Carlo noise in σ² updates; needs many samples for accuracy.

Defer unless Option A's Laplace error shows up as a blocker on real data.

## Effort estimate

| Task | Estimate |
|---|---|
| D1. Derive EM M-step Q / gradient / convergence criterion cleanly | 1 day |
| D2. Implement `em_mstep` + Armijo backtracking (or fixed-step GEM) | 1.5 days |
| D3. Implement EM outer loop driver, preserve existing run_ai_reml | 1 day |
| D4. Post-convergence Q, logLik, observed-info via finite diff | 0.5 days |
| D5. Unit tests on `em_mstep` monotonic ascent | 0.5 days |
| D6. Integration: rerun dev scenarios, verify σ²_E not floored | 0.5 days |
| D7. Regression: non-Laplace paths unchanged (interpretation tests) | 0.5 days |
| D8. Docs + CLI flag reorg (em_max_iter, em_tol, etc.) | 0.5 days |
| Slop / debugging | 1.5 days |
| **Total** | **~7 days (1.5 weeks)** |

## Risk register

1. **Line search cost.** Re-finding the mode at each trial σ² is
   expensive. Mitigation: fixed-step GEM (no line search); or backtrack
   on a cached `l̂` (same mode, different σ²-dependent Q value —
   biased but cheap; accept and retune).

2. **EM plateaus.** GEM can stall at saddle points if gradients go
   to zero prematurely. Mitigation: track Q per iter; if flat for ≥3
   iters, perturb σ² and retry.

3. **Boundary behaviour.** σ²_E may still try to go negative under
   EM if the model is mis-specified (e.g., thresholding a non-Gaussian
   truly-continuous trait). Keep the σ² ≥ 0 floor; if hit, log
   diagnostic rather than silent clamp.

4. **Laplace approximation error.** For very rare K (≤ 0.01), the
   mode is a poor representation of the posterior (heavy skew).
   Option C (MCEM) is the real fix; document as "not recommended for
   K < 0.01".

5. **Covariate interaction.** The EM M-step also updates β via GLS
   on the adjusted second-moment form. Current `run_ai_reml` GLS-β
   step needs to be re-derived for EM — probably just reuses μ in
   place of the naïve observation. Small but careful work.

## Scope boundaries

**In**:
- LiabilityLaplace EM with GEM gradient M-step.
- Observed-information SEs via finite difference.
- Integration with ε-annealing (from Phase 2).
- Dev-scenario regression with σ²_E recovery near truth.

**Out** (future work if needed):
- MCEM (Option C).
- Full proper Newton M-step (Option B).
- Log|H| via SLQ for AIC/BIC (harder to derive for EM; defer to LRT
  based on Q ratios instead).
- Multi-trait / bivariate extension.

## Decision gate for shipping

After Option A lands, re-run the dev-scenario grid + 5 replicates on
key scenarios. Ship-ready criteria:

| Criterion | Threshold |
|---|---|
| σ²_E recovery on threshold-true data | bias < 0.05, not floored |
| Vp recovery | 0.9 < Vp < 1.1 |
| h² bias matching Phase 2 or better | |Δh²| ≤ 0.05 on typical scenarios |
| Rare K (0.05) recovery | |Δh²| ≤ 0.10 (known harder case) |
| Wall-time penalty | ≤ 3× Phase 2 fixed-ε Laplace |
| Sandwich SE activates | sandwich_se_applied=1 in ≥ 80% of fits |
| No regression on continuous / MeanOnce paths | bit-identical where possible |

If all green → promote as the "laplace" liability model; deprecate
the Phase 2 AI-REML-retrofitted Laplace (keep code for history, hide
behind `--laplace-legacy` flag).

## Context to reload before starting

Cold-start reading order:

1. `notes/iter_reml_binary_phase2.plan.md` — original Phase 2 spec.
2. This file — P2.11 findings + P3 plan.
3. `fitace/ace_iter_reml/src/laplace.{h,cpp}` — existing mode-finder
   + `compute_trace_corrections` + `compute_sandwich_correction`.
   The Woodbury infrastructure is already there.
4. `fitace/ace_iter_reml/src/ai_reml.cpp` — existing AI-REML outer
   loop; section around line 745 (outer iter) and line 1442 (SE
   output) are the integration points to study.
5. Dev-scenario aggregator output (last known — re-run after any
   σ² changes): 24 scenarios + 12 replicates, `iter_reml_fp32`
   precision.
