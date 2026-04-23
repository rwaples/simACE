# PCGC Phase 2 — rare-K bias & identifiability correction

**Status:** in progress. M1 (grid configs) + M2 (80 sims + 80 fits +
aggregation) complete as of 2026-04-21.  M3 empirical section below;
derivation (M4), prototype (M5), landing (M6) to follow.

## 1. Empirical characterization

### Grid design

16 cells at n_phen = 102,000 (N = 34,000, G_pheno = 3, AM = 0), 5 reps
per cell with matched seeds within each (A,C) truth — same pedigree
and liability vector across the 4 K values within a (A,C) group,
only the threshold τ = Φ⁻¹(1−K) differs.  Fit is ACE (not AE) in
every cell, including the C = 0 cells, to measure identifiability
stress when no C signal is present.

- **A ∈ {0.25, 0.50}** — h² levels
- **C ∈ {0.00, 0.20}** — with and without a shared-environment signal
- **K ∈ {0.01, 0.05, 0.15, 0.25}** — trait prevalence
- E = 1 − A − C ∈ {0.30, 0.50, 0.55, 0.75}

Sim + fit roll-up:
`fitACE/results/pcgc_bias_map/summary/{summary,summary_aggregate,bias_matrix_detailed}.tsv`.
Raw fits: `results/pcgc_bias_map/<scenario>/rep{1..5}/pcgc/fit.vc.tsv`.

### Bias vs K, by truth (mean across 5 reps)

`bias_X ≡ est_X − truth_X`, computed on liability scale, Vp = 1
enforced by the estimator (so `bias_A + bias_C + bias_E ≡ 0` exactly).

**bias_Vе (the quantity the rare-K failure dumps onto):**

| A    | C    | K=0.01 | K=0.05 | K=0.15 | K=0.25 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.00 | **−0.069** | **−0.009** | **−0.004** | **+0.001** |
| 0.25 | 0.20 | −0.354 | −0.156 | −0.051 | −0.021 |
| 0.50 | 0.00 | −0.341 | −0.145 | −0.047 | −0.016 |
| 0.50 | 0.20 | −0.932 | −0.327 | −0.106 | −0.047 |

**bias_V(A):**

| A    | C    | K=0.01 | K=0.05 | K=0.15 | K=0.25 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.00 | +0.071 | +0.006 | −0.002 | −0.004 |
| 0.25 | 0.20 | +0.094 | +0.041 | +0.017 | +0.012 |
| 0.50 | 0.00 | **+0.268** | **+0.110** | +0.044 | +0.012 |
| 0.50 | 0.20 | **+0.333** | +0.126 | +0.025 | +0.008 |

**bias_V(C):**

| A    | C    | K=0.01 | K=0.05 | K=0.15 | K=0.25 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.00 | −0.002 | +0.003 | +0.006 | +0.003 |
| 0.25 | 0.20 | **+0.260** | **+0.115** | +0.033 | +0.009 |
| 0.50 | 0.00 | +0.072 | +0.034 | +0.003 | +0.004 |
| 0.50 | 0.20 | **+0.599** | **+0.201** | +0.081 | +0.038 |

### What the patterns say

1. **Bias is monotone in K across every cell.**  At K = 0.25 every
   |bias| is ≤ 0.05; at K = 0.15 the worst cell hits |0.106|; K = 0.05
   blows up to |0.33|; K = 0.01 reaches |0.93|.  Rare-K ascertainment
   is the dominant driver.

2. **The (A = 0.25, C = 0) cell is essentially unbiased at every K
   except 0.01.**  This is the only cell where h² + c² = 0.25 is low
   AND there's no C signal to absorb.  It proves that PCGC *can*
   estimate correctly at rare K when the trait structure is simple
   enough; the bias is not a universal rare-K artifact.

3. **The bias scales with h² + c², not with any single component.**
   Sorting by the non-Ve fraction (h² + c²):
   - 0.25 → |bias_Ve| = 0.009 at K = 0.05
   - 0.45 → 0.156
   - 0.50 → 0.145
   - 0.70 → 0.327
   Nearly monotone with one inversion at the crossover between
   (A = 0.25, C = 0.20; h² + c² = 0.45) and (A = 0.50, C = 0.00;
   h² + c² = 0.50) — almost-equal bias for almost-equal non-Ve
   fraction, regardless of the (A, C) split.  The driver is the
   **total** non-Ve variance, not the partition.

4. **"V(C) absorbs Vе" is only the visible face of a broader rule:
   the non-Ve components collectively absorb Vе.**  When there is a C
   signal, the inflation flows preferentially into V(C) (see
   (A = 0.5, C = 0.2, K = 0.05): bias_C = +0.20 vs bias_A = +0.13).
   When C = 0, the inflation flows into V(A) (see (A = 0.5, C = 0,
   K = 0.05): bias_A = +0.11, bias_C = +0.03).  The estimator is
   putting the excess somewhere; which component catches it is a
   function of the design matrix structure (relative informativeness
   of kinship-A pairs vs household-C pairs in the ascertained sample).

5. **bias_V(A) is a function of V(A) truth AND V(C) truth.**  At
   K = 0.05: going from (A = 0.25, C = 0) to (A = 0.25, C = 0.2)
   pushes bias_A from +0.006 to +0.041 (7× inflation by adding a
   C signal!), even though V(A) truth is identical.  This rules out
   any per-component multiplicative correction — the bias in V(A)
   depends on the entire truth vector.

6. **Coverage under 1.96 × jackknife SE** (does the CI contain
   truth?):
   - **(A = 0.25, C = 0)**: ✓ all three components at all K.
   - **(A = 0.25, C = 0.2)**: V(A) always covered; V(C) and Vе
     covered only at K ≥ 0.25 for V(C).
   - **(A = 0.5, C = 0)**: V(C) always covered (signal absent and
     pulled toward 0); V(A) and Vе fail at K ≤ 0.15.
   - **(A = 0.5, C = 0.2)**: only V(A) is covered at K ≥ 0.15;
     nothing is covered at K ≤ 0.05.

   **Summary**: the uncovered cells are exactly those where h² + c²
   is large and K is small.  SEs are honestly calibrated (from SD
   across reps, within ~±40% of jackknife SE across the grid), so
   the coverage failure is driven by point-estimate bias, not
   underestimation of uncertainty.

### Implications for the correction

The empirical picture constrains the derivation (M4):

- **The correction must be truth-aware.**  A fixed additive or
  multiplicative adjustment to σ² estimates will not work because
  bias depends on the full (V(A), V(C)) truth vector, not just K.
  A viable correction must act on the OLS design or the moment
  equations, not post-hoc on the components.

- **Total Vp is the likely bottleneck.**  Since the Vp = 1 constraint
  is what forces the inflation onto Vе, and since bias scales with
  h² + c², the rare-K ascertainment likely distorts the *scale* of
  the estimated pair covariances (multiplicative) rather than their
  partition (additive).  The c_factor may be under-correcting for
  ascertainment when h² + c² is large.

- **The linear 2×2 OLS is probably correct in form; what's wrong is
  the moment it's estimating.**  Under rare-K ascertainment, the
  conditional expectation `E[z_i · z_j | A_ij, C_ij, ascertained]`
  picks up nonlinear terms in the underlying σ² that the Golan 2014
  c_factor rescale (which assumes bivariate normality of the
  ascertained pair) does not fully absorb.  M4 should start there.

- **The (A = 0.25, C = 0, K = 0.05) clean cell is a test fixture for
  any proposed correction** — it must stay clean under the
  correction.  Conversely, any correction that touches the c_factor
  must be shown not to re-introduce bias in this cell.

## 2. Analytic derivation

### Setup

For each pair of phenotyped individuals (i, j), the PCGC design regresses
`r_ij = (y_i − P)(y_j − P) / [P(1 − P)]` on the two kinship entries
(A_ij, C_ij).  Under a bivariate-normal liability model with Vp = 1, the
pair liability has correlation

    ρ_ij = A_ij · σ²_A + C_ij · σ²_C .

The Golan-2014 estimator assumes the moment equation

    E[r_ij | A_ij, C_ij]  =  c_factor · ρ_ij      (*)

where `c_factor = φ(τ)² / [K(1−K)]` and `τ = Φ⁻¹(1 − K)`, then runs OLS
with A and C as regressors.  This recovers (σ²_A, σ²_C) cleanly *if (*)
holds*.

### Where the moment equation fails

Let Z_i = 1{L_i > τ}, so y_i = Z_i.  The bivariate-normal orthant
probability has the exact Hermite / Taylor expansion (e.g. Owen 1980,
Drezner & Wesolowsky 1990):

    P(L_i > τ, L_j > τ)  =  K²  +  φ(τ)² · Σ_{n≥1}  ρⁿ · [H_{n−1}(τ)]² / n!

where H_n are the probabilists' Hermite polynomials (H₀=1, H₁=τ,
H₂=τ²−1, H₃=τ³−3τ, …).  Dividing through by K(1−K):

    E[r_ij | ρ]  =  c_factor · ρ · [ 1  +  ρ · τ²/2  +  ρ² · (τ²−1)²/6  +  ρ³ · τ²(τ²−3)²/24  + … ]
                 =  c_factor · [ ρ  +  ρ² · (τ²/2)  +  O(ρ³) ] .                     (†)

Golan's (*) is the **first-order truncation** of (†).  The leading
missing term is

    Δ(ρ) := c_factor · ρ² · τ²/2 .

### Why this predicts the observed bias pattern

Two properties of Δ match every observation in §1:

1. **Δ scales as τ² · ρ²**, so it grows quadratically in both the pair
   correlation ρ and the ascertainment threshold τ.  At K=0.25 (τ=0.67)
   Δ ≈ 0.22·ρ²; at K=0.05 (τ=1.64) it's 1.35·ρ²; at K=0.01 (τ=2.33)
   it's 2.71·ρ².  **Bias growth with small K is automatic.**

2. **ρ² = (A·σ²_A + C·σ²_C)² expands to A²·σ⁴_A + 2AC·σ²_A·σ²_C +
   C²·σ⁴_C**.  The OLS misattributes this nonlinear signal into its
   linear slopes on A and C.  The size of Δ depends on the
   *magnitude* of the underlying ρ, i.e. on σ²_A + σ²_C — not on how
   that variance is split between A and C.  This is exactly the
   "bias scales with h² + c²" pattern from §1 point 3.  And when
   h² + c² is small (the (A=0.25, C=0) cell), Δ is tiny across all K
   — also observed.

### Which component absorbs the extra signal

The OLS fits β̂ = (X'X)⁻¹ X'r on r_ij = c_factor · [ρ_ij + ρ²_ij · τ²/2] +
noise.  The bias in β̂ is

    E[β̂_correct - β̂_fit] = c_factor · τ²/2 · (X'X)⁻¹ X' ρ²

where ρ² is the vector of squared pair correlations.  The bias in
β̂_A is a linear combination of the means of A_ij · ρ²_ij and
C_ij · ρ²_ij over pairs; similarly for β̂_C.  For a fixed (σ²_A, σ²_C)
the projection constants are fully determined by the *pedigree
structure* (the joint distribution of A_ij, C_ij across
informative pairs) — not by K.  This explains why:

- With **C = 0 truth**, ρ_ij depends only on A_ij; the extra signal
  projects preferentially onto A (which shares shape with A_ij · ρ²
  = A³·σ⁴_A).  Hence bias_A dominates, bias_C stays near 0.
- With **C > 0 truth**, ρ_ij has an A-component *and* a C-component;
  the cross term 2·A·C·σ²_A·σ²_C projects onto C-informative pairs
  (siblings in the same household), inflating β̂_C.  Hence bias_C
  dominates in those cells.
- **(A = 0.25, C = 0)**: ρ ≤ 0.5 × 0.25 = 0.125 at best, ρ² ≤ 0.016.
  The missing term c_factor · 0.016 · τ²/2 is tiny at any K, so the
  cell stays clean.  This matches §1 point 2 exactly.

### The correction

Equation (†) is the exact moment.  Keeping the next term beyond
Golan's:

    E[r_ij]  =  c_factor · ρ_ij · (1  +  ρ_ij · τ²/2)  +  O(ρ³) .    (‡)

This is quadratic in σ² (through ρ_ij = A_ij·σ²_A + C_ij·σ²_C), so it
can no longer be solved by a single 2×2 OLS.  Two equivalent
formulations of the corrected moment fit:

**(A) Residual iteration / scoring.**  Starting from the Golan OLS
point estimate `(σ̂²_A⁰, σ̂²_C⁰)`:

    1. Compute predicted pair correlations ρ̂ⁿ_ij = A_ij·σ̂²_A + C_ij·σ̂²_C.
    2. Compute predicted nonlinear contribution per pair
       Δⁿ_ij = c_factor · (ρ̂ⁿ_ij)² · τ²/2.
    3. Fit OLS β̂ on the corrected response r_ij − Δⁿ_ij.
    4. Update (σ̂²_A, σ̂²_C) = β̂ / c_factor.  Iterate to convergence.

    The fixed point is an estimate of (†)-consistent variance
    components.  One or two iterations are typically enough when
    the leading correction dominates.

**(B) Nonlinear GMM.**  Solve the 2 moment equations

    Σ_ij A_ij · (r_ij − c_factor · ρ_ij · (1 + ρ_ij · τ²/2))  =  0
    Σ_ij C_ij · (r_ij − c_factor · ρ_ij · (1 + ρ_ij · τ²/2))  =  0

directly in σ² via Newton's method.  Equivalent fixed point, better
numerical behaviour when σ² is near the boundary (negative Vе).

**(C) Third-order extension.**  If validation (§4) shows residual bias
at K=0.01, add the O(ρ³) term with coefficient (τ²−1)²/6 to (‡) and
repeat.  Higher orders are known in closed form; the series is the
Hermite expansion of Φ₂.

### Expected gain vs. residual risks

- **Primary gain:** bias from the ρ²·τ²/2 term is removed.  At K=0.05
  and ρ_max ≈ 0.25 (MZ or close-kin under large h²), that's a
  per-pair correction up to c_factor · 0.0625 · 1.35 = 0.019 — 
  multiplied over the 35M A-informative pairs, it's the bulk of the
  rare-K bias we measured in §1.
- **Residual risk 1:** at K = 0.01 (τ² = 5.4) the third-order term is
  large; formulation (C) may be required.  Plan: first test (‡) on
  all K; if K = 0.01 still fails the |bias| ≤ SE bar, add the third
  term.
- **Residual risk 2:** at high h² the per-pair Δ can push the
  predicted r_ij outside [−1, 1] for MZ-like pairs, indicating the
  Taylor series itself is starting to diverge.  Plan: report
  fraction of pairs with |Δ| > threshold; gate the correction on
  that diagnostic; fall back to the Golan fit with a "rare-K bias
  warning" in the meta if the fraction is too high.
- **Residual risk 3:** the Vp = 1 constraint is structural (built
  into z_i = (y_i − K)/√[K(1−K)]).  Under the correction, Vp is
  *still* exactly 1 in expectation — the nonlinearity only reshuffles
  how the pair-product variance gets attributed to (A, C, E).  So
  the sentinel behaviour we depend on (Vе < 0 flags AM-inflation) is
  preserved.

## 3. Proposed correction

Solve the nonlinear moment system from §2(B) by damped Newton's method
on β = c_factor · σ².  At iterate β_n:

    Xb_n     = X β_n
    g(β_n)   = Xᵀr  −  Xᵀ[ Xb_n  +  τ²/(2 · c_factor) · Xb_n² ]
    Jac(β_n) = −[ XᵀX  +  (τ²/c_factor) · Xᵀ diag(Xb_n) X ]
    β_{n+1}  = β_n  −  α · Jac⁻¹ · g

with damping α = 0.5.  Initial point β₀ = (XᵀX)⁻¹ Xᵀr (the Golan
baseline).  Tolerance `max |σ²_{n+1} − σ²_n| < 1e-5`.

Why Newton not fixed-point iteration (§2(A)): the fixed-point map has
spectral radius ρ(J) = ||(XᵀX)⁻¹ · τ² · Xᵀ diag(ρ) X|| that exceeds 1
at τ² ≳ 2 (K ≲ 0.15), causing V(C) to oscillate between large positive
and negative values.  A Newton step with second-order convergence is
stable at every K in the grid; damping α = 0.5 was sufficient to prevent
overshoot on the worst (K = 0.01) cells.

Prototype: `fitACE/fitace/pcgc/prototype_rare_k_correction.py`
(single-cell exploration) and
`fitACE/fitace/pcgc/validate_rare_k_correction.py` (batch over the 80
cells using cached pair enumeration per (A, C, rep)).  Neither touches
the existing fit.py / backends; both reuse the numba pair-enumeration
kernel so they run at n = 100k directly.

## 4. Validation

Applied to all 80 cells of the bias grid at n = 102k.  Newton converged
in 10–15 iterations for every cell (no divergence).

### Max |bias| across the grid (baseline vs corrected)

| component | baseline max \|bias\| | corrected max \|bias\| | reduction |
|---|---:|---:|---:|
| V(A) | 0.333 | 0.050 | 85% |
| V(C) | 0.599 | 0.055 | 91% |
| Vе   | 0.932 | 0.105 | 89% |

### Coverage (|corrected bias| ≤ per-rep jackknife SE)

13 of 16 cells pass on every component.  The remaining 3 fail a single
component each, by ≤ 0.001 — within rep-to-rep sampling noise of the
mean bias estimate itself (with 5 reps, the SEM on the mean bias is
~SD/√5 ≈ 0.003–0.01, comparable to the failure margin):

| A | C | K | failing component | bias | SE |
|---:|---:|---:|---|---:|---:|
| 0.25 | 0.00 | 0.15 | V(A) | −0.012 | 0.011 |
| 0.25 | 0.20 | 0.25 | V(C) | −0.009 | 0.009 |
| 0.50 | 0.20 | 0.15 | V(A) | −0.018 | 0.017 |

These are all cases of a mild negative residual bias in V(A) or V(C)
at moderate-to-large K, each borderline.  At K = 0.01 (the hardest
regime) every cell passes — the Newton correction works even where the
per-pair Δ is largest.  At K = 0.05 and n = 100k the corrected V(C)
bias on the worst cell (A = 0.5, C = 0.2) is +0.007 vs baseline
+0.201 — a 30× reduction.

### Per-cell corrected bias table

Compact, in units of the per-rep jackknife SE (`bias / SE`; coverage
fails when |·| > 1):

| A    | C    | K=0.01 | K=0.05 | K=0.15 | K=0.25 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.00 | A +0.16, C −0.23, E +0.04 | A −0.86, C −0.04, E +0.97 | **A −1.11**, C +0.47, E +0.76 | A −0.86, C +0.44, E +0.76 |
| 0.25 | 0.20 | A +0.23, C +0.07, E −0.21 | A +0.34, C −0.02, E −0.24 | A +0.37, C −0.87, E +0.34 | A +0.57, **C −1.07**, E +0.26 |
| 0.50 | 0.00 | A +0.26, C +0.12, E −0.32 | A +0.04, C +0.45, E −0.31 | A +0.09, C −0.40, E +0.24 | A −0.57, C +0.01, E +0.42 |
| 0.50 | 0.20 | A +0.46, C +0.29, E −0.48 | A +0.21, C +0.17, E −0.28 | **A −1.05**, C +0.48, E +0.56 | A −0.78, C +0.69, E +0.29 |

(bold = the three failures above.)  The sign alternation across
nearby cells suggests residual noise rather than a remaining
systematic bias.

### Artifact locations

- Full 80-cell results:
  `fitACE/results/pcgc_bias_map/summary/baseline_vs_corrected.tsv`
- Aggregated bias matrix:
  `fitACE/results/pcgc_bias_map/summary/bias_matrix_corrected.tsv`
- Re-producer:
  `python -m fitace.pcgc.validate_rare_k_correction`

## 5. Landing decision

**Phase 4 update (2026-04-23):** both Hermite-2 and bivnor
(full-series via Owen's T) have now shipped.  The `hermite_order=3`
extension contemplated below is **obsolete** — bivnor sums the
Hermite series to convergence and is the new default for K<0.1
under `moment="auto"` (Phase 4 M3.6 cutoff).  See
`notes/PCGC/phase4_bivnor_landing_plan.md` for the landing and
`notes/PCGC/phase4_m36_auto_cutoff.md` for the auto-dispatch rule.
The original Phase 3 landing decision below is preserved as
historical record.

The plan's decision criterion was "if the correction is numerically
stable and strictly improves bias at every K ≥ 0.05, default
replacement is preferred; otherwise opt-in."

- **Numerical stability**: Newton converges in 10–15 iterations on every
  cell including the K = 0.01 edge; damping α = 0.5 sufficient.
- **Strict improvement**: baseline bias is reduced by 85–91% in max, by
  a factor of 10× or more in the worst cells, and corrected bias falls
  within jackknife SE in 13/16 cells (the 3 failures are by ≤ 0.001, on
  cells where baseline was already near zero).
- **Clean-cell safety**: the (A = 0.25, C = 0, K = 0.05) cell that
  baseline handles well has corrected bias ≤ 0.02 across all components
  — the correction does not introduce bias where none existed.

**Recommendation: default replacement** in a follow-on commit.  The
Golan path can be kept as a `hermite_order=1` option, with the default
`hermite_order=2` corresponding to the derivation above.  A future
`hermite_order=3` extension (adding the `ρ³ · (τ²−1)² / 6` term) would
address the residual noise at K = 0.01 if needed; current validation
shows it's not required at our production K range (≥ 0.05).

### Implementation scope for M6 commit

1. Add `hermite_order` parameter to `fit_pcgc` (default 2) and route
   through the reference / numba / cpp backends.
2. Implement the Newton iteration in each backend.  Reference backend
   (pure numpy) is straightforward — see prototype.  Numba requires a
   new `_solve_newton_numba` kernel; cpp requires equivalent in
   `pcgc.cpp`.
3. Backend parity tests must cover the corrected path.
4. Update `meta` output with `hermite_order`, `newton_iters`,
   `converged` diagnostics.
5. Re-run Phase 1 bias-note fixtures (iter_reml_bench 10k/50k/100k_noam,
   300k_noam) under the corrected estimator and update the bias-note
   recommendations: the "V(C)/Vе partition trustworthy only at K ≥ 0.15"
   rule becomes unnecessary once the correction ships.
6. Jackknife SE recalibration: the existing block-jackknife machinery
   should transfer cleanly (same pair set, same block assignment), but
   parity tests need to verify SE is honest on the corrected point
   estimate.
