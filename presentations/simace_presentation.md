---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 36px;
    color: #333;
  }
  h2 {
    font-size: 30px;
    color: #333;
  }
  table {
    font-size: 20px;
  }
  blockquote {
    border-left: 4px solid #2171B5;
    background: #f0f6fc;
    padding: 0.5em 1em;
    font-style: normal;
  }
  img {
    display: block;
    margin: 0 auto;
  }
---

<!-- _paginate: false -->

# **simACE**
## A Multi-Generational Pedigree Simulator for Validating Heritability Estimators

**Ryan Wiberg**

---

<!-- MOTIVATION (~3 min) -->

# Why Simulate?

- Heritability estimation methods need validation against **known truth**
- Real registry data: true $h^2$ is unknown
  - Cannot measure estimator **bias** or **variance**
  - Cannot distinguish methodological artifacts from biology
- Need: simulated families where we **set** $A$, $C$, $E$ and check what we **recover**

> *If you don't know the answer, you can't grade the test.*

---

# The Gap

Existing simulation approaches lack one or more of:

1. **Time-to-event phenotypes** — not just binary affected/unaffected
2. **Realistic family complexity** — half-siblings, MZ twins, multi-partner mating
3. **Proper censoring** — age-window observation + competing-risk mortality
4. **End-to-end validation** — ground-truth comparison in one pipeline

> **simACE** provides all four in a single, configurable, reproducible framework.

---

<!-- PIPELINE & FEATURES (~5 min) -->

# Pipeline Overview

```
Simulate ──→ Dropout ──→ Phenotype ──→ Censor ──→ Sample ──→ Validate ──→ Statistics
 (A,C,E)                  (TTE/LTM)   (age+death)                          (vs truth)
```

- **Snakemake-orchestrated**; YAML-configured scenarios
- Each stage is an independent CLI tool — composable and testable
- Multiple replicates per scenario for variance estimation

---

# Pedigree Structure

![w:480](../results/beck/beck_adhd/plots/pedigree_counts.png)

- Multi-partner mating — zero-truncated Poisson ($\lambda = 0.5$)
- MZ twins at configurable rate ($p = 0.02$ per eligible mating)
- Pedigree dropout — simulates incomplete observation
- Burn-in generations — avoids founder artifacts
- $N = 1\text{M}$ per generation (3M individuals in beck_adhd)

---

# Configurable Scenarios

**Running example: `beck_adhd`** (parameters from Beck et al. fits)

| Parameter | Value |
|-----------|-------|
| $A$ | 0.50 |
| $C$ | 0.00 |
| $E$ | 0.50 |
| Model | Cure-frailty |
| Baseline | Lognormal |
| $\beta_{\text{sex}}$ | 0.50 |
| Prevalence (F / M) | 4.8% / 8.3% |
| $N$ | 1,000,000 |

**Pluggable phenotype models:** Weibull, exponential, Gompertz, lognormal, log-logistic, gamma, cure-frailty, ADuLT LTM, ADuLT Cox

**Also:** sex-specific prevalence, per-generation censoring windows, case-ascertainment sampling, assortative mating (copula-based)

---

<!-- MODEL SPECIFICATION (~7 min) -->

# Model: Liability Decomposition

**Total liability** for individual $i$, trait $k$:

$$L_{i}^{(k)} = A_{i}^{(k)} + C_{i}^{(k)} + E_{i}^{(k)}, \qquad \text{Var}(A^{(k)}) + \text{Var}(C^{(k)}) + \text{Var}(E^{(k)}) = 1$$

**Founders** — correlated bivariate draws:

$$\begin{pmatrix} A^{(1)} \\ A^{(2)} \end{pmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\; \begin{pmatrix} \sigma_{A_1}^2 & r_A \,\sigma_{A_1}\sigma_{A_2} \\ r_A \,\sigma_{A_1}\sigma_{A_2} & \sigma_{A_2}^2 \end{pmatrix}\right), \qquad C^{(1,2)} \sim \mathcal{N}(\mathbf{0}, \Sigma_C)$$

**Offspring** — Mendelian sampling (infinitesimal model):

$$A_{\text{child}}^{(k)} = \bar{A}_{\text{parents}}^{(k)} + \epsilon^{(k)}, \qquad \epsilon^{(k)} \sim \mathcal{N}\!\left(0,\; \tfrac{\sigma_{A_k}^2}{2}\right)$$

- $C$: shared within household (same mother), drawn fresh each generation
- $E$: independent per individual — no familial correlation

---

# Model: Liability $\to$ Time-to-Event

**Proportional hazards frailty:**

$$z_i = \exp\!\big(\beta \cdot L_i + \beta_{\text{sex}} \cdot \text{sex}_i\big) \qquad \text{(individual frailty)}$$

$$h(t \mid z_i) = h_0(t) \cdot z_i \qquad \text{(hazard function)}$$

$$S(t \mid z_i) = \exp\!\big(-H_0(t) \cdot z_i\big) \qquad \text{(survival function)}$$

$$t_i = H_0^{-1}\!\big(-\log(U_i)\, /\, z_i\big),\quad U_i \sim \text{Unif}(0,1] \qquad \text{(inverse-CDF sampling)}$$

**Cure-frailty** (beck_adhd):
- Susceptible if $L_i > \Phi^{-1}(1 - p)$ where $p$ = prevalence
- Among susceptible: $t_i$ from lognormal frailty ($\mu = 2.89$, $\sigma = 0.66$)
- Non-susceptible: never develop disease ("cured" fraction)

---

# Model: Censoring

**Two independent layers:**

**1. Age-window censoring:**
$$t_{\text{obs}} \in [a_{\text{left}}^{(g)},\; a_{\text{right}}^{(g)}]$$
Per-generation observation interval. Events outside $\to$ censored.

**2. Competing-risk mortality:**
$$T_{\text{death}} \sim \text{Weibull}(\lambda_d, \rho_d)$$
If onset > death age $\to$ death-censored. ($\lambda_d = 164$, $\rho_d = 2.73$ by default)

Overall sensitivity: ~67% in beck_adhd

![w:550](../results/beck/beck_adhd/plots/censoring_cascade.png)

---

<!-- PHENOTYPE OUTPUTS (~5 min) -->

# Output: Cumulative Incidence

![w:780](../results/beck/beck_adhd/plots/cumulative_incidence.phenotype.png)

- True (parametric) vs. observed CIF per generation
- Censoring effects visible: younger generations have less follow-up
- Sex-stratified CIF also available ($\beta_{\text{sex}} = 0.50$, male prevalence 8.3% vs. female 4.8%)

---

# Output: Liability by Affected Status

![w:550](../results/beck/beck_adhd/plots/liability_violin.phenotype.png)

- Affected individuals drawn from upper tail of liability ($p \approx 4.4\%$)
- Clear separation validates cure-frailty threshold mechanism
- Distribution shape stable across replicates

---

# Output: Censoring Confusion

![w:550](../results/beck/beck_adhd/plots/censoring_confusion.png)

- Joint distribution of true and observed event status
- Age-window and death censoring interact differently across generations
- Realistic missingness patterns matching registry data

---

<!-- VALIDATION (~7 min) -->

# Validation: Variance Component Recovery

![w:780](../results/beck/beck_adhd/plots/additive_shared.by_generation.png)

- Realized $(A + C) / \text{Var}(L)$ matches parametric truth across all 6 generations
- Each point is one replicate ($n = 3$); dashed line = configured value
- Stable transmission: no drift, no founder artifacts after burn-in

---

# Validation: Heritability by Generation

![w:780](../results/beck/beck_adhd/plots/heritability.by_generation.png)

- Narrow-sense $h^2 = \text{Var}(A) / \text{Var}(L)$ matches configured $A = 0.50$
- Consistent across generations $\to$ Mendelian sampling variance is correctly calibrated
- Also validated via Falconer's formula: $\hat{h}^2 = 2(r_{\text{MZ}} - r_{\text{DZ}})$

---

# Validation: Tetrachoric Correlations by Relationship

![w:780](../results/beck/beck_adhd/plots/tetrachoric.phenotype.png)

- MLE tetrachoric $r$ by relationship type vs. parametric expectation
- Pair counts shown per group ($n > 1\text{M}$ for most types)
- MZ $\approx A + C$, full sib $\approx \frac{1}{2}A + C$, half sib $\approx \frac{1}{4}A$, cousin $\approx \frac{1}{8}A$

---

# Validation: Parent-Offspring Liability Regression

![w:780](../results/beck/beck_adhd/plots/parent_offspring_liability.by_generation.png)

- Midparent-offspring regression slope $\approx h^2$ under infinitesimal model
- Independent validation of heritability (not relying on twin contrast)

---

# Bivariate Extension

![w:500](../results/cross_trait/high_correlation/plots/cross_trait_tetrachoric.png)

**Configurable cross-trait correlations:**
- Genetic correlation $r_A$
- Shared environment $r_C$
- Recovery validated per relationship type

Shown: `high_correlation` scenario ($r_A = 0.7$, $r_C = 0.7$)

Framework extends naturally to bivariate genetic epidemiology.

---

<!-- SUMMARY (~3 min) -->

# Summary

**simACE**: a configurable, validated multi-generational pedigree simulator

**What it provides:**
- Known ground-truth $A$, $C$, $E$
- Time-to-event phenotypes with proper censoring
- Realistic family structures (half-sibs, MZ twins, dropout)
- 9 pluggable phenotype models
- Integrated validation pipeline

**Technical highlights:**
- Bivariate traits with configurable $r_A$, $r_C$
- Snakemake-orchestrated, YAML-configured
- Publication-ready atlas outputs
- Open source (MIT license)
- Scales to $N = 1\text{M}$+ per generation

> **If you don't know the answer, you can't grade the test.**

---

# Future Directions

- **Disease-specific scenarios** — cardiovascular disease, additional psychiatric disorders
- **Downstream method validation** — EPIMIGHT and other heritability estimators
- **Extended phenotype models** — gene-environment interaction, time-varying exposures
- **Larger-scale validation** — population-level benchmarks, cross-method comparison

<br>

## Thank you

Questions?
