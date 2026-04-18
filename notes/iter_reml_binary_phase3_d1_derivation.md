# iter_reml Phase 3 D1 — EM M-step derivation & design decisions

**Purpose:** lock the math, the algorithm, and the open-question decisions
before implementing `em_mstep.{h,cpp}` (D2) and the EM outer-loop driver
(D3) per `iter_reml_binary_phase3_pxem.plan.md`.

**Scope:** LiabilityLaplace layer only. Continuous / LiabilityMeanOnce
keep the existing Phase 2 `run_ai_reml` path unchanged.

---

## 1. Model & EM setup

Model (unchanged):
```
l ~ N(Xβ, V(σ²))       V = σ²_A·A + σ²_C·C + σ²_E·I
y_i = Φ( sign_i · (l_i − τ)/ε )   (probit-ε smoothing; ε annealed)
```

Treat `l` as the missing data, `y` as observed. At iter t, given
current `θ^(t) = (σ²^(t), β^(t))`:

- **E-step**: compute the Laplace posterior of `l` under `q^(t)`:
  - `μ^(t) ≈ l̂^(t)` — posterior mode (Phase 2 `find_laplace_mode`).
  - `Σ^(t) ≈ H^(t)⁻¹` where `H^(t) = V(σ²^(t))⁻¹ + W(l̂^(t))`.
  - `W` diagonal — already written to `layer.laplace.W_diag`.

- **M-step**: update `(σ², β)` by ascending the Q-function
  ```
  Q(σ², β; θ^(t)) = E_{q^(t)}[log p(y, l | σ², β)]
  ```

EM monotonicity (standard): any M-step improving Q also improves
`log L_Laplace(σ²)` up to O(Laplace error) = O(1/n).

---

## 2. Q-function — closed form

`log p(y,l|σ²,β) = log p(y|l) − ½log|V| − ½(l−Xβ)'V⁻¹(l−Xβ) + const`.
`log p(y|l)` does not depend on `(σ², β)`. Taking `E_{q^(t)}` and
using `E_q[(l−Xβ)'V⁻¹(l−Xβ)] = (μ−Xβ)'V⁻¹(μ−Xβ) + tr(V⁻¹Σ)`:

```
Q(σ²,β; θ^(t)) = C^(t)
                 − ½·log|V(σ²)|
                 − ½·(μ^(t) − Xβ)'·V(σ²)⁻¹·(μ^(t) − Xβ)
                 − ½·tr(V(σ²)⁻¹·Σ^(t))
```

`C^(t)` lumps terms that do not depend on `(σ², β)`.

**The `−½·tr(V⁻¹·Σ)` term is what Phase 2 was missing** — it is the
posterior-variance correction that prevents σ²_E from collapsing.

---

## 3. Gradient w.r.t. σ²_k

Using `∂V⁻¹/∂σ²_k = −V⁻¹·M_k·V⁻¹` (standard matrix calculus), and
holding `(μ^(t), Σ^(t))` fixed across the M-step:

```
∂Q/∂σ²_k = −½·tr(V⁻¹·M_k)
           + ½·(μ−Xβ)'·V⁻¹·M_k·V⁻¹·(μ−Xβ)        ≡ ½·q_k^μ
           + ½·tr(V⁻¹·M_k·V⁻¹·Σ^(t))
```

With `Σ^(t) = H^(t)⁻¹`, Woodbury (`H⁻¹ = V − V·W^½·S⁻¹·W^½·V`,
`S = I + W^½·V·W^½`) gives:
```
V⁻¹·H⁻¹·V⁻¹ = V⁻¹ − W^½·S⁻¹·W^½
⇒ tr(V⁻¹·M_k·V⁻¹·H⁻¹) = tr(V⁻¹·M_k) − tr(W^½·M_k·W^½·S⁻¹)
```
So the `−½tr(V⁻¹·M_k)` term and `+½tr(V⁻¹·M_k)` from the trace
correction **cancel exactly**, leaving:

```
∂Q/∂σ²_k = ½·(q_k^μ − tr(W^½·M_k·W^½·S⁻¹))           [EM-GRAD]
```

This is identical to what P2.7a tried. **The math was right; the
failure was pairing [EM-GRAD] with the Gaussian-REML AI matrix as
Newton curvature.** P2.7a then took Newton steps of the wrong size
and floored σ²_E.

### Implementation mapping

All operators below already exist in the Phase 2 codebase:

| Quantity | Existing code | Cost |
|---|---|---|
| `u = V⁻¹·(μ − Xβ)` (where `μ = l̂`) | existing PCG in ai_reml.cpp step (2) | 1 PCG |
| `e_k = M_k · u` (A, C, I) | existing MatMult in step (3) | 2 MatMults |
| `q_k^μ = u'·e_k` | existing VecDot in step (4) | 3 dots |
| `tr(W^½·M_k·W^½·S⁻¹)` | `laplace.cpp :: compute_trace_corrections` | m CG-on-S + 3m MatMult |

So the full EM gradient cost per iter =
**1 PCG (u) + 2 MatMults + m CG-on-S + 3m MatMults + 3 dots**
= Phase 2 Laplace cost minus the 3 w_k PCG solves and the AI-matrix
construction. In practice, slightly *cheaper* per outer iter than
Phase 2 Laplace AI-REML, because we skip `w_k = V⁻¹·e_k`.

---

## 4. M-step: step rule (open question → decision)

The plan left the M-step curvature as a TODO. I considered three
options and picked one.

### Option 4a — Proper EM Newton (full AI^EM)

Derive `∂²Q/∂σ²_k∂σ²_l`, which contains the new term
`tr(V⁻¹·M_k·V⁻¹·M_l·V⁻¹·Σ)`. Via Woodbury:
```
tr(V⁻¹·M_k·V⁻¹·M_l·V⁻¹·Σ) = tr(V⁻¹·M_k·V⁻¹·M_l)
                           − tr(V⁻¹·M_k·V⁻¹·M_l·W^½·S⁻¹·W^½)
```
The second term is Hutchinson-estimable but requires per-pair (k,l)
probes on a more complex operator (M_k·V⁻¹·M_l·W^½·S⁻¹·W^½). 6
additional trace estimates per iter + care about variance
scaling. **1.5–2 weeks of work**. Fastest per-iter convergence.

### Option 4b — Gaussian AI as preconditioner (damped quasi-Newton)

Compute Phase 2's `AI_kl = ½·(e_k'·w_l + e_l'·w_k)` (requires 3 extra
`w_k = V⁻¹·e_k` PCGs). Use it as a preconditioner:
```
Δσ² = α · AI⁻¹ · grad_EM
```
Damp `α` via trust region on `max|Δσ²_k|/(σ²_k + ε)`. AI is the wrong
curvature for EM, but it still gives a reasonable scale in σ² space,
and damping with a trust region guards against over-stepping. **No
new math, adds 3 PCGs per outer iter.** Per-iter cost ≈ Phase 2.

### Option 4c — Fixed-step GEM (no curvature)

Pure gradient ascent:
```
Δσ² = α_t · (grad_EM / max(‖grad‖, 1))    with α_t = α_0 / (1 + t·decay)
```
Simplest possible. EM monotonicity guarantees convergence as long as
the step is small enough. Slow in the flat tail; may take 50+ outer
iters vs Phase 2's typical 15–25. **No curvature work.**

### Decision: **Option 4b** (AI-preconditioner + trust region)

Rationale:
- 4a is a rathole — we do not know the AI^EM formula is well-
  conditioned, and the new trace operators are expensive. Defer to a
  follow-up if wall-time becomes a constraint.
- 4c is too slow. At n=50k with each outer iter ~5–8 s, 50 iters ~5
  min per fit; dev grid × 5 reps × 24 scenarios makes the nightly run
  painful.
- 4b is the sweet spot: re-use the existing AI code, add 3 PCGs,
  damp aggressively. The AI being "wrong" is mostly a concern at
  boundaries (σ² near 0) and near convergence (where curvature
  matters for SE anyway). Damping handles boundaries; we switch to
  observed-info finite diff for final SE (section 7).

**Trust-region rule** (adapted from Phase 2):
```
δ_t = 0.3 · min(σ²^(t))              initial cap
while max_k |Δσ²_k| / (σ²_k^(t) + σ²_floor) > δ_t:
    α ← α / 2   (up to 6 halvings)
accept step; grow δ by 1.5× if step was unsafeguarded; else halve.
```

No re-evaluation of Q at trial σ². We trust that the EM gradient
points ascent and the trust region keeps us from overshooting.

---

## 5. β handling under EM

GLS-β closed form from ∂Q/∂β = 0:
```
β̂^(t+1) = (X'·V⁻¹·X)⁻¹ · X'·V⁻¹·μ^(t)
```

Identical in form to Phase 2's GLS-β, with `μ^(t) = l̂^(t)` in place
of the observed phenotype `y`. Phase 2 already computes β̂ from `l̂`
(line ~1300–1380 of ai_reml.cpp, `compute_beta_gls_liability`).
**No change needed.**

Post-convergence SE(β̂):
```
Var(β̂) = (X'·V⁻¹·X)⁻¹
```
This is the Gaussian-REML form; it ignores the Laplace posterior
variance contribution from the `μ = l̂(σ², β)` dependence. Phase 3
keeps this for shipping; a Louis-style sandwich on β is a follow-up
(not in scope for D1–D8).

---

## 6. Convergence criterion

Plan suggests `ΔQ/n < tol_em` OR `max|Δσ²_k|/(|σ²_k|+ε) < tol_step`
over a window.

Evaluating `ΔQ` exactly across iters is expensive (mixes `V^(t+1)`
with `Σ^(t)`). Use the cheap proxies instead:

1. **Primary**: `max_k |σ²_k^(t) − σ²_k^(t−1)| / (σ²_k^(t) + 1e-4) < tol_step`
   for K consecutive iters (default K=3, tol_step=1e-3).
2. **Secondary**: `‖grad_EM‖_∞ · max_k(σ²_k) < tol_grad`
   (default 1e-3). Bails early when gradient is numerically zero
   before step rule damping kicks in.
3. **Hard cap**: `em_max_iter = 100` outer iters.

Deliberately NOT used:
- Q-value tracking across iters (mixed `V^(t+1)/Σ^(t)` cost).
- `dLLpred` (Phase 2's Gaussian proxy) — not the right scale for EM.

Post-convergence we evaluate the Laplace marginal log-likelihood
exactly once (section 7).

---

## 7. Post-convergence: logLik and observed info

### Laplace marginal log-likelihood

```
log L_Laplace(σ²) = log p(y, l̂ | σ², β) − ½·log|H(l̂, σ²)/(2π)|
```

Expanding and using `log|H| = log|V⁻¹ + W| = log|I + W^½·V·W^½| − log|V| = log|S| − log|V|`:
```
log L_Laplace(σ²) = log p(y|l̂)
                  − ½·(l̂ − Xβ)'·V⁻¹·(l̂ − Xβ)
                  − ½·log|S|
```
(The `−½·log|V|` and `+½·log|V|` cancel. Nice.)

Components:
- `log p(y|l̂)`: sum of log-probit-ε terms, already computable from
  `l̂` and `y_raw`. O(n).
- `(l̂ − Xβ)'·V⁻¹·(l̂ − Xβ) = u'·(l̂ − Xβ)`: 1 PCG + 1 dot.
- `log|S|`: SLQ on the `S = I + W^½·V·W^½` operator. Same kernel as
  Phase 2's `slq_estimate_logdet` but with operator S instead of V.
  **New code**: small wrapper around the existing CG-on-S apply
  function + Lanczos. Cheap (~30 ms at n=10k).

### Observed information for SE

`I_obs(σ²) = −∂²log L_Laplace/∂σ²_k∂σ²_l` at the MLE. By Fisher's
identity at the optimum,
```
∂log L_Laplace/∂σ² = ∂Q/∂σ²|θ=θ_MLE
```
so finite-differencing the EM gradient gives the observed info:
```
I_obs[k,l] ≈ −( grad_EM(σ² + h·δ_l)[k] − grad_EM(σ²)[k] ) / h
symmetrised: I_obs ← ½·(I_obs + I_obs')
```

Cost: 3 additional E-steps (mode-find) + 3 gradient evaluations.
≈ 3× one normal outer iter. Scales as Phase 2's log|H|-gradient
cost. Acceptable.

Default `h = max(1e-3·σ²_k, 1e-6)` (scaled, as σ²_E can be ~0.3 or
~1.0).

`SE(σ²_k) = sqrt(diag(I_obs⁻¹))`.

If `I_obs` is not SPD (σ² at boundary), fall back to Phase 2's
Gaussian-AI SE with a `sandwich_se_applied=0` diagnostic.

---

## 8. Concrete API (D2 spec)

New header `fitace/ace_iter_reml/src/em_mstep.h`:

```cpp
namespace ace_iter_reml {

struct EmStepInputs {
    const std::array<double, 3>& sigma2_current;
    const std::vector<double>& beta_current;         // empty if no X
    const PhenotypeLayer& layer;                     // has l̂, W
    Mat V;                                           // V(σ²_current), bound to pcg
    PcgSolver& pcg;
    Mat A, C;
    const std::vector<std::vector<double>>* X;       // may be null
    const AiRemlOptions& opts;                       // probes, seed, trust region
    int iter;                                        // outer iter index
};

struct EmStepResult {
    std::array<double, 3> sigma2_next;
    std::array<double, 3> grad;
    std::array<double, 3> tr_WMWS;                   // trace corrections
    std::array<std::array<double, 3>, 3> AI;         // Gaussian AI (preconditioner)
    std::vector<double> beta_next;
    double delta_trust_next;                         // adapted for next iter
    int pcg_iters_avg;
    int pcg_iters_min;
    int pcg_iters_max;
    int n_safeguard_halvings;
    std::array<double, 3> step;                      // Δσ² after damping
};

EmStepResult em_step(const EmStepInputs& in, double delta_trust);

// Laplace marginal log-likelihood at final σ² (post-convergence).
struct EmLogLik {
    double log_p_y_given_l;
    double quad_u_y;             // u'·(l̂ − Xβ)
    double log_det_S;            // SLQ
    double logLik;               // log p(y|l̂) − ½·quad_u_y − ½·log|S|
};
EmLogLik em_logLik(const PhenotypeLayer& layer,
                   const std::vector<double>& Xbeta,
                   Mat V, PcgSolver& pcg,
                   const LaplaceOptions& lopts,
                   int slq_steps, int slq_probes, uint64_t seed);

// Observed-info matrix via finite-difference of EM gradient.
// Drives 3 trial E-steps internally.
struct EmObservedInfo {
    std::array<std::array<double, 3>, 3> I_obs;
    bool spd;                    // false → caller should fall back
};
EmObservedInfo em_observed_info(/* ... */);

}
```

New outer driver in `ai_reml.cpp` (or new file `em_driver.cpp`):
- Branch at the top of `run_ai_reml`: if `opts.layer` is LiabilityLaplace,
  dispatch to `run_em_reml(...)` which follows the EM loop below.
- Non-Laplace layers keep the existing path.

```cpp
for (iter = 0; iter < opts.em_max_iter; ++iter) {
    refresh_V(V, model);   pcg.set_operator(V);

    // E-step — mode + W, unchanged from Phase 2
    LaplaceOptions lopts = anneal(opts.laplace_opts, iter);
    find_laplace_mode(layer, Xbeta_prev, sigma2, V, lopts);

    // β GLS — unchanged from Phase 2 (uses μ = l̂)
    if (X) beta = gls_beta(layer.l_hat, X, V, pcg);

    // M-step — [EM-GRAD] gradient + AI preconditioner + trust region
    EmStepResult step = em_step(inputs, delta_trust);

    // Convergence (section 6)
    if (converged(step, sigma2_history)) break;

    sigma2 = step.sigma2_next;
    delta_trust = step.delta_trust_next;
}

// Post-convergence (section 7)
EmLogLik ll = em_logLik(...);
EmObservedInfo info = em_observed_info(...);
```

---

## 9. What this does NOT do (defer)

- **MCEM (Option C)**: posterior sampling instead of Laplace mode.
  Needed only if Laplace error shows up as a blocker on rare K (<0.01).
- **Full EM Newton (Option 4a)**: deferred until wall-time is a
  constraint.
- **Sandwich SE on β**: β variance stays Gaussian-GLS form. Follow-up.
- **SLQ on S reuse**: first pass builds a new SLQ operator wrapper.
  Can be refactored into the existing `slq.cpp` framework later.

---

## 10. Decision summary (locked for D2)

| Decision | Choice | Where derived |
|---|---|---|
| Gradient | `½·(q_k^μ − tr(W^½·M_k·W^½·S⁻¹))` | §3 |
| β update | GLS with `μ = l̂` (reuse Phase 2) | §5 |
| Step rule | AI⁻¹·grad preconditioned + trust region damping | §4 |
| Trust region | init 0.3·min(σ²), grow 1.5× / halve | §4 |
| Convergence | σ²-range stability over K=3 iters + grad∞ backup | §6 |
| logLik | `log p(y|l̂) − ½·quad − ½·log|S|` via new SLQ on S | §7 |
| Obs info | Finite-difference EM gradient (3 trial E-steps) | §7 |
| β SE | `(X'V⁻¹X)⁻¹` — Gaussian form, ignores μ(β) dependence | §5 |
| Deferred | MCEM, EM-Newton, β sandwich | §9 |

---

## 11. Open items for user review before D2

1. **Trust-region init `0.3·min(σ²)`**: aggressive or conservative?
   Phase 2 uses `1e10` unbounded initial. EM first-step is riskier
   because grad may be large; aggressive cap protects against σ²_E →
   0 in iter 0. Alternative: first iter ungated, subsequent gated.

2. **AI-preconditioner fallback**: if AI is non-SPD (near boundary),
   fall back to identity preconditioner (straight gradient ascent
   with step ‖grad‖⁻¹·cap) or to diagonal AI only?

3. **Safeguard limit**: 6 halvings vs Phase 2's 10. EM monotonicity
   should make halving rarely necessary. Keep 6 as tighter budget to
   surface pathological scenarios faster.

4. **Initialisation**: start EM from Phase 1 Mean estimates (like
   Phase 2) or from `(1/3, 1/3, 1/3)`? Phase 1 Mean gives sensible
   (σ²_A, σ²_C) but overestimates σ²_E on binary data — EM will
   correct quickly but may take extra iters. Phase 2 start preferred.

5. **Reuse vs new file**: write `em_mstep.{h,cpp}` fresh, or graft
   into `ai_reml.cpp`? Plan says fresh file. Prefer fresh file for
   testability.
