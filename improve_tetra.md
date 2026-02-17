# Improving Liability Correlation Estimation Under Weibull Censoring

The tetrachoric correlation estimates the liability correlation from binary affected/unaffected status using a threshold model. Under the Weibull proportional hazards model with censoring, this underestimates the true liability correlation because it treats all unaffected individuals identically — someone censored at age 20 is very different from someone censored at age 79.

Two approaches for better estimation:

## Option A: Pairwise Weibull Frailty Likelihood

Since the data-generating model is known (Weibull PH with liability frailty), write the pairwise likelihood directly and maximize over the liability correlation ρ.

### Model

For a relative pair (i, j) with correlated liabilities:

```
(L_i, L_j) ~ BVN(0, 0, 1, 1, ρ)
```

Individual hazard and survival:

```
h(t | L) = k * rate^k * t^(k-1) * exp(β * L)
S(t | L) = exp(-(rate * t)^k * exp(β * L))
f(t | L) = h(t | L) * S(t | L)
```

### Pairwise likelihood

For pair (i, j) with observed (t_i, δ_i, t_j, δ_j) where δ=1 is event, δ=0 is censored:

```
PL(ρ) = ∫∫ ∏_{k∈{i,j}} [ f(t_k|L_k)^δ_k · S(t_k|L_k)^(1-δ_k) ] · φ₂(L_i, L_j; ρ) dL_i dL_j
```

Expanding the individual contributions:

```
g(t, δ, L) = [k * rate^k * t^(k-1) * exp(β*L)]^δ * exp(-(rate*t)^k * exp(β*L))
```

### Implementation

1. **Gauss-Hermite quadrature** for the 2D integral over (L_i, L_j). Transform from standard normal to correlated bivariate normal using Cholesky. ~20 nodes per dimension is sufficient.

2. **Optimize** ρ on (-1, 1) using `minimize_scalar(bounds=(-0.999, 0.999), method="bounded")` — same pattern as the existing tetrachoric code.

3. **SE** via analytic Fisher information, same approach as the current tetrachoric SE (bivariate normal PDF squared over cell probabilities generalizes to the derivative of the log-pairwise-likelihood).

### Pseudocode

```python
from numpy.polynomial.hermite_e import hermegauss

def pairwise_weibull_corr(t_i, d_i, t_j, d_j, rate, k, beta, n_quad=20):
    """Estimate liability correlation from paired survival data."""
    nodes, weights = hermegauss(n_quad)
    # weights for standard normal: w_k * phi(x_k) already folded in

    # Precompute per-individual log-contributions on the quadrature grid
    # log g(t, delta, L) for each quadrature node
    # ...

    def neg_log_pairwise_lik(rho):
        # Cholesky: L_i = z_i, L_j = rho*z_i + sqrt(1-rho^2)*z_j
        # Sum over all pairs, integrating over (z_i, z_j) via quadrature
        ...

    result = minimize_scalar(neg_log_pairwise_lik, bounds=(-0.999, 0.999), method="bounded")
    return result.x
```

### Considerations

- Requires knowing (or estimating) rate, k, β. In the simulation context these are known from params. For real data, estimate marginally first, then estimate ρ conditional on those estimates.
- Computationally heavier than tetrachoric (n_quad^2 evaluations per pair), but still fast for typical sample sizes. Can vectorize across pairs.
- This is the exact likelihood for the known DGP, so it is the most efficient estimator for validation purposes.


## Option B: Semiparametric Shared Frailty Model

Fit a Cox proportional hazards model with a shared (cluster-level) frailty term. The frailty captures unobserved heterogeneity within relative pairs, and its variance relates to the liability correlation.

### Model

```
h_i(t) = h_0(t) * exp(β * x_i) * w_j
```

where w_j is the shared frailty for cluster (family/pair) j, typically assumed gamma-distributed:

```
w_j ~ Gamma(1/θ, 1/θ)    [mean 1, variance θ]
```

### Mapping frailty variance to liability correlation

Under a gamma frailty with variance θ:
- Kendall's tau for a pair = θ / (θ + 2)
- This gives a concordance measure, but mapping to the *liability* correlation ρ requires assumptions about the frailty distribution

For a log-normal frailty (closer to the BVN liability model):
- The frailty is exp(β * L) where L ~ N(0, 1)
- Pairs share correlated L values
- The frailty variance and liability correlation are linked through β and ρ

### Implementation

Using `lifelines` (already a dependency):

```python
from lifelines import CoxPHFitter

# For each relationship type (MZ twins, DZ twins, siblings, etc.):
# 1. Stack pair members into long format with a cluster ID
# 2. Fit CoxPH with shared frailty on the cluster
# 3. Extract frailty variance estimate
```

Alternatively, using R's `coxph` or `frailtypack` via `rpy2` for more frailty distribution options (log-normal frailty directly estimates the liability variance).

### Considerations

- **Pros:** Semiparametric — does not assume Weibull baseline hazard, only proportional hazards. Robust to baseline hazard misspecification. Standard method with well-understood properties.
- **Cons:** Gamma frailty is the default in most packages but doesn't directly map to a BVN liability model. Log-normal frailty is closer but less commonly implemented. Estimates frailty *variance* rather than pairwise *correlation*, so the mapping back to ρ depends on the frailty distribution chosen and the value of β.
- Less natural for validation (where you know the DGP) than Option A, but more appropriate for a real-data analysis pipeline.
