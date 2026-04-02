# Observed-Scale vs Liability-Scale Heritability

## Heritability on the observed scale

Heritability on the observed scale (h²_obs) is the proportion of phenotypic variance in the *measured* trait attributable to additive genetic variance:

h²_obs = V_A / V_P

where both V_A and V_P are computed on whatever scale the phenotype is actually recorded.

For **continuous traits**, observed-scale and liability-scale heritability are the same thing — there is no transformation.

For **binary (disease) traits**, the observed phenotype is 0/1 (unaffected/affected), so V_P = K(1−K) where K is prevalence. This is distinct from liability-scale heritability (h²_L), which refers to the variance proportion on the latent continuous liability.

## The liability threshold model (shared foundation)

Both the Dempster–Lerner and Falconer results assume:

1. A latent continuous **liability** ℓ that is normally distributed in the population.
2. A **threshold** t such that individuals with ℓ > t are affected (prevalence K = P(ℓ > t)).
3. Liability decomposes as ℓ = A + C + E with Var(ℓ) = 1 (standardized).

The two scales are related by:

h²_obs = h²_L × z² / (K(1−K))

where z = φ(Φ⁻¹(1−K)) is the height of the standard normal density at the liability threshold.

### Key properties

- **h²_obs is always smaller than h²_L** — dichotomizing a continuous liability discards information, compressing the apparent genetic signal.
- **Prevalence-dependent** — h²_obs shrinks as the disease becomes rarer (or more common), even if the underlying genetic architecture is unchanged. This makes it problematic for comparing heritability across diseases with different prevalences.
- The conversion factor z²/K(1−K) has a maximum of 2/π ≈ 0.637 (at K = 0.5) and declines toward 0 as K → 0 or K → 1.

## Dempster–Lerner (1950)

Dempster and Lerner were working in **animal breeding** (the original paper is about poultry). Their contribution was showing that the regression of offspring liability on parent liability is preserved through the threshold, but **attenuated** by a specific factor.

The core result: if you compute the regression of offspring phenotype (0/1) on parent phenotype (0/1), the observed-scale heritability relates to the liability-scale heritability by:

h²_obs = h²_L × z² / (K(1−K))

**Intuition:** The factor z²/K(1−K) captures how much information the binary cut destroys. It asks: "how much does a small shift in mean liability translate into a change in prevalence?" That is governed by the density at the threshold (z), normalized by the variance of the binary outcome (K(1−K)). When K is very small, the threshold sits far out in the tail where the density z is tiny — so binary status becomes a very noisy readout of liability, and h²_obs collapses.

## Falconer (1965)

Falconer was working in **human genetics**, where controlled crosses are not possible. He wanted to estimate h²_L from observable family data — specifically from the **recurrence risk** in relatives (K_R, the prevalence among relatives of affected individuals).

His approach inverts the problem. Given:

- Population prevalence K
- Recurrence risk in relatives K_R
- Known genetic relatedness r (e.g., 0.5 for first-degree relatives)

He showed:

h²_L = (a_R − a) / r

where a = Φ⁻¹(1−K) and a_R = Φ⁻¹(1−K_R) are the liability thresholds corresponding to the population and relative prevalences, respectively. The division by r accounts for the degree of genetic sharing.

**Intuition:** If a relative of an affected person has higher risk (K_R > K), their mean liability must be shifted rightward. Falconer asks: "how far did the liability distribution shift?" — converting prevalence differences back to liability-scale distances via the inverse normal. Then he divides by r to scale up from the shared-genetics effect to the total genetic effect.

### Relationship between the two

They are algebraically equivalent — two faces of the same transformation:

| Method             | Direction          | Input                    | Output   |
|--------------------|--------------------|--------------------------|----------|
| Dempster–Lerner    | liability → observed | h²_L                    | h²_obs   |
| Falconer           | observed → liability | family recurrence risks  | h²_L     |

Falconer's method is the **back-transformation** that Dempster–Lerner's result implies must exist. In practice, "Falconer's method" usually refers to the full estimation procedure (plug in K and K_R, get h²_L), while "Dempster–Lerner" refers to the theoretical conversion factor.

## Estimating observed-scale heritability without the liability threshold model

The liability threshold model is convenient but strong (assumes normally distributed latent liability, sharp threshold). Several classes of methods estimate heritability on the observed scale while avoiding that assumption entirely.

### 1. Linear models applied directly to binary phenotypes

The simplest approach: treat the 0/1 phenotype as a quantitative trait and decompose its variance.

**Variance-component / ANOVA approach:**

h²_obs = V_A / K(1−K)

where V_A is estimated from the covariance between relatives on the binary scale. No latent variable is invoked — you are just asking "what fraction of the variance in this Bernoulli outcome is explained by additive genetics?"

**Haseman–Elston regression** is the cleanest example: regress (Y_i − Y_j)² on IBD sharing π̂_ij across relative pairs. Completely agnostic about the phenotype distribution — it only relates phenotypic similarity to genetic similarity. The slope estimates V_A on whatever scale Y is measured.

**GCTA-GREML on binary traits:** The raw estimate from a linear mixed model on 0/1 outcomes is observed-scale. Lee et al. (2011) showed how to convert it to liability scale, but the unconverted estimate stands on its own as a model-free quantity. It just asks: "how much of the variance in case/control status do the SNPs capture?"

### 2. Regression-based estimators from GWAS summary statistics

**LD Score Regression (LDSC):** Regresses χ² association statistics on LD scores. For binary traits, the raw output is h²_SNP on the observed scale. No threshold model is involved in the estimation — it is a method-of-moments estimator relating test statistics to LD structure. The liability-scale conversion is an optional post-hoc step that *adds* the threshold assumption.

### 3. Concordance/correlation approaches without back-transformation

The classical twin formulae:

h²_obs ≈ 2(r_MZ − r_DZ)

If you use the **Pearson correlation** on the raw 0/1 phenotypes (or concordance rates directly), you stay on the observed scale without any latent variable:

r_Pearson = (P(both affected) − K²) / K(1−K)

Note: using the **tetrachoric correlation** instead assumes bivariate normality (i.e., the liability model), so that does invoke the threshold.

### 4. Survival / frailty models (a different latent model, not the threshold)

For age-at-onset traits, **Cox frailty models** sidestep the threshold entirely:

h_i(t) = h_0(t) × Z_i

where Z_i is a multiplicative frailty with genetic and environmental components. The frailty variance σ²_Z maps to heritability on the **hazard scale**, which is neither the liability scale nor the simple observed 0/1 scale. This is conceptually appealing for diseases because:

- It naturally handles censoring.
- It models *when* you get sick, not just *whether*.
- The "scale" is the instantaneous hazard, which has direct epidemiological meaning.

The downside: results depend on the assumed frailty distribution (gamma, log-normal, etc.), so you have traded the normality assumption of the threshold model for a distributional assumption on the frailty.

### 5. Logistic mixed models (log-odds scale)

logit(P(Y_i = 1)) = X_i β + g_i

where g_i ~ N(0, σ²_g G). This estimates heritability on the **log-odds scale**, which is yet a third scale — neither observed nor liability. Methods like GMRM-logit work this way. The log-odds scale has nicer statistical properties than the observed scale (no floor/ceiling effects), but the estimate is not directly comparable to either h²_obs or h²_L.

### 6. Prediction-based / cross-validation approaches

Estimate h² as the empirical ceiling on out-of-sample prediction accuracy (e.g., Nagelkerke R², AUC-derived measures). These are inherently observed-scale and assumption-free — they just ask "how well can genetic data predict case status?" No latent variable needed.

## Why the threshold model persists despite alternatives

The reason everyone keeps converting back to liability scale is that **observed-scale heritability has a fundamental comparability problem**: the same genetic architecture yields different h²_obs at different prevalences. A disease with K = 0.01 will always look less heritable on the observed scale than one with K = 0.30, even if the underlying genetic contribution is identical. The liability-scale conversion removes this dependence, making heritability comparable across traits — at the cost of assuming the threshold model is correct.

The tradeoff:

- **Observed scale, no threshold model:** Fewer assumptions, but prevalence-dependent and not comparable across diseases.
- **Liability scale, threshold model:** Requires normality + threshold assumptions, but yields a prevalence-invariant quantity.
