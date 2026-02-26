# ACE Simulation Framework: Methods

*This document describes the ACE simulation framework: how it generates synthetic multi-generational family data with known genetic and environmental parameters, assigns disease phenotypes, and validates statistical recovery of those parameters. The framework provides a controlled testbed for evaluating twin- and family-study methods, where ground-truth values are available for comparison against estimates.*

---

## Simulation Overview

simACE is a multi-generational individual-based simulation framework implementing an ACE variance-component model, in which an individual's quantitative liability is decomposed into additive genetic ($A$), common shared-environment ($C$), and unique-environment ($E$) components. 

The simulation generates a multi-generational pedigree with realistic family structures including full- and half-sibling, and monozygotic (MZ) twins.

Binary phenotypes are related to the liability via either a Weibull proportional-hazards frailty survival model (i.e. time-to-event) or a simple liability-threshold model.

Censoring is included in the Weibull model, and includes random mortality as well as deterministic left- and right-censoring by age, configurable by generation.  

The simulation code is written in Python and the pipeline is orchestrated by Snakemake and supports named scenarios, replicate runs, and automated validation including sanity checks and plots.

**Key Terms**

- **Liability** — an unobserved continuous score representing an individual's underlying risk for a disease; higher liability means higher risk.
- **Founders** — the initial generation of pedigree individuals whose genetic and environmental values are drawn from scratch rather than inherited.
- **Burn-in** — early simulated generations that are discarded so that the recorded pedigree is not influenced by the arbitrary starting conditions of the founder generation.
- **Frailty** — a multiplicative modifier on disease hazard derived from an individual's liability; it controls how quickly risk accumulates with age.
- **Censoring** — the inability to observe a disease event because the individual died, was too young or old,  or was not yet born during the study window.
- **Prevalence** — the proportion of individuals classified as affected in a given generation.

## Conceptual simulation steps

```
+----------------+   +----------------+   +----------------+   +-----------+
| 1. Pedigree    |-->| 2. Phenotype   |-->| 3. Statistical |-->| 4. Valid- |
|    Simulation  |   |    Assignment  |   |    Analysis    |   |    ation  |
|                |   |                |   |                |   |           |
| Build families |   | Convert to     |   | Estimate corr. |   | Compare   |
| with known ACE |   | observable data|   | & heritability |   | to truth  |
+----------------+   +----------------+   +----------------+   +-----------+
```

**Stage 1 — Pedigree Simulation** creates a multi-generational population where each individual has known additive-genetic (A), shared-environment (C), and unique-environment (E) values, linked to parents and siblings through realistic family structures. 

**Stage 2 — Phenotype Assignment** converts continuous liabilities into observable outcomes — either an age-at-onset via a survival model or a binary affected/unaffected status via a threshold model — and applies censoring to mimic real study limitations. *

*Stage 3 — Statistical Analysis** estimates correlations between relatives and computes heritability using only the observable phenotype data, as a real researcher would. *

*Stage 4 — Validation** compares every estimate back to the known ground-truth parameters, confirming both code correctness and estimator performance.

## Pedigree Simulation

### Founder generation

The simulation begins by creating a starting population from scratch. Each founder receives randomly drawn additive-genetic, shared-environment, and unique-environment values — these are the ground-truth components that downstream analyses will attempt to recover.

A founder cohort of $N$ individuals (default $N = 100{,}000$) is initialised at generation $g = 0$. Sex is assigned as $\text{sex}_i \sim \text{Bernoulli}(0.5)$, coded 0 = female, 1 = male.

**Single-trait ACE decomposition.** For a single trait, each individual's liability is the sum of three independent components:

$$
L_i = A_i + C_i + E_i
$$

where $A_i \sim \mathcal{N}(0, \sigma^2_A)$ is the additive genetic component, $C_i \sim \mathcal{N}(0, \sigma^2_C)$ is the common (shared) environment, and $E_i \sim \mathcal{N}(0, \sigma^2_E)$ is the unique environment. Total phenotypic variance is normalised to unity, so $\sigma^2_A + \sigma^2_C + \sigma^2_E = 1$ and the variance components directly represent proportions of total variance. For example, with parameters $A = 0.5$, $C = 0.2$, $E = 0.3$, half the variation in liability is additive genetic, 20% is due to the shared household, and 30% is individual-specific noise.

**Extension to two correlated traits.** The simulation models two traits jointly ($k = 1, 2$), allowing genetic (C) and shared-environment components (C) to be correlated across traits. Each founder receives variance components drawn from bivariate normal distributions:

$$
\begin{pmatrix} A_{i,1} \\ A_{i,2} \end{pmatrix}
\sim \mathcal{N}\!\left(
\mathbf{0},\;
\begin{bmatrix}
\sigma^2_{A_1} & r_A\,\sigma_{A_1}\sigma_{A_2} \\
r_A\,\sigma_{A_1}\sigma_{A_2} & \sigma^2_{A_2}
\end{bmatrix}
\right)
$$

$$
\begin{pmatrix} C_{i,1} \\ C_{i,2} \end{pmatrix}
\sim \mathcal{N}\!\left(
\mathbf{0},\;
\begin{bmatrix}
\sigma^2_{C_1} & r_C\,\sigma_{C_1}\sigma_{C_2} \\
r_C\,\sigma_{C_1}\sigma_{C_2} & \sigma^2_{C_2}
\end{bmatrix}
\right)
$$

$$
E_{i,k} \sim \mathcal{N}(0,\; \sigma^2_{E_k}), \quad
\sigma^2_{E_k} = 1 - \sigma^2_{A_k} - \sigma^2_{C_k}
$$

where $r_A$ and $r_C$ are the cross-trait genetic and shared-environment correlations, respectively. Unique-environment components $E_1$ and $E_2$ are drawn independently (no cross-trait $E$ correlation). Each founder's total liability is $L_{i,k} = A_{i,k} + C_{i,k} + E_{i,k}$.

In plain terms, each founder receives a random genetic hand, a household environment, and individual noise that sum to their overall liability.

### Reproduction - mating and family structure

The simulation assumes a constant population size ($N$) across generations.  Each new generation must include N offspring and for each offspring we must select parents. This is accomplished in few steps.

During reproduction, potential mother and fathers are paired and each pair defines a social household.  The mother of is assigned a family size drawn from a Poisson distribution:

$$
n_{\text{children},f} \sim \text{Poisson}(\lambda), \quad \lambda = 2.3 \text{ (default)}
$$

Households are randomly ordered and offspring are generated for each mother until a total of $N$ offspring is reached. The paternity of each offspring is randomly determined by the $p_{\text{nonsocial\_father}}$ parameter: with chance $1 - p_{\text{nonsocial}}$ the social household male is the father, otherwise a random male is selected. This creates maternal half-siblings (same mother, different biological father) at an expected pairwise proportion of $1 - (1 - p_{\text{nonsocial}})^2$ among all maternal sibling pairs.

All children within a household (i.e. that share a mother) share a household identifier and thus have an identical $C$ term.

### Monozygotic twin generation

Identical (monozygotic) twins share 100% of their genetic material; the simulation inserts them randomly into the birth order at a configurable rate.

At each offspring is generated, a Bernoulli trial with probability $p_{\text{MZ}}$ (default 0.02) determines whether that two MZ twins are generated. MZ twins are assigned identical biological parents and share identical A and C components.

### Offspring inheritance

Each offspring inherits genetic material from both parents but grows up in a new household environment. The key consequence is that siblings share roughly half their genetic variation but fully share their childhood environment.

Offspring variance components are generated according to standard quantitative-genetic assumptions:

**Additive genetic ($A$).** Under the infinitesimal model (Bulmer, 1971), each offspring's breeding value is the midparent value plus Mendelian sampling noise with segregation variance equal to half the additive genetic variance:

$$
A_{\text{offspring},k} = \frac{A_{\text{mother},k} + A_{\text{father},k}}{2} + \epsilon_{k}
$$

$$
\begin{pmatrix} \epsilon_1 \\ \epsilon_2 \end{pmatrix}
\sim \mathcal{N}\!\left(
\mathbf{0},\;
\begin{bmatrix}
\tfrac{1}{2}\sigma^2_{A_1} & r_A \cdot \tfrac{\sigma_{A_1}}{\sqrt{2}} \cdot \tfrac{\sigma_{A_2}}{\sqrt{2}} \\
r_A \cdot \tfrac{\sigma_{A_1}}{\sqrt{2}} \cdot \tfrac{\sigma_{A_2}}{\sqrt{2}} & \tfrac{1}{2}\sigma^2_{A_2}
\end{bmatrix}
\right)
$$

In short, each child's genetic value equals the average of the parents' values plus a Mendelian sampling term — the random half of each parent's genome that gets passed on.

MZ twins share identical $A$ values: the second twin's breeding values and sex are copied from the first.

**Common environment ($C$).** A single bivariate draw $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_C)$ is made per household and assigned to all children within it. Critically, $C$ is *not* transmitted from parent to child — it represents the offspring's own shared rearing environment. This is the standard ACE assumption with no autoregressive $C$ transmission. That is, siblings share C because they grow up in the same household, but their parents' childhood environment does not carry over.

**Unique environment ($E$).** Each child receives independent draws $E_{i,k} \sim \mathcal{N}(0, \sigma_{E_k})$, uncorrelated across siblings, twins, and traits. Even MZ twins differ in E — it represents measurement error, developmental noise, and non-shared environmental exposures.

### Burn-in and recording

Early generations are discarded so that the recorded data is not influenced by the arbitrary starting conditions of the founder generation.

The simulation runs for $G_{\text{sim}}$ total generations (default 6), of which the first $G_{\text{sim}} - G_{\text{ped}}$ generations (default 2) serve as burn-in to reduce founder-specific effects and approach equilibrium allele-frequency distributions. Only the final $G_{\text{ped}}$ generations (default 4) are recorded in the output pedigree. The recorded pedigree contains $N \times G_{\text{ped}}$ individuals with contiguous identifiers, each annotated with generation number, parental identifiers, MZ twin partner (if any), household identifier, all six variance components, and both trait liabilities.

## Phenotype Models

The pedigree simulation produces continuous liabilities, but real studies observe discrete outcomes — a diagnosis age or an affected/unaffected classification. The models below translate liabilities into these observable phenotypes.

### Weibull proportional-hazards frailty model

This model simulates *when* disease occurs: higher liability leads to earlier onset (on average). The age-of-onset of this model allows a natural extension to also include censoring (see below).

Continuous liabilities are mapped to age-at-onset via a Weibull proportional-hazards frailty model. For each trait $k$, assuming liability ($L$) is standardised to mean=0 and sd=1. The conditional hazard function is:

$$
h(t \mid L) = \frac{\rho}{\lambda} \left(\frac{t}{\lambda}\right)^{\rho - 1} \exp(\beta\,L)
$$

where $\lambda > 0$ is the Weibull scale parameter, $\rho > 0$ is the Weibull shape parameter, and $\beta$ scales the effect of indiviudal liabilty (i.e. log-hazard coefficient for liability). The corresponding survival function is:

$$
S(t \mid L) = \exp\!\left[-\left(\frac{t}{\lambda}\right)^{\!\rho} \exp(\beta\,L)\right]
$$

Event times are generated by inverse-CDF sampling. Defining the individual frailty as $z_i = \exp(\beta\,\tilde{L}_i)$ and drawing $U_i \sim \text{Uniform}(0, 1]$:

$$
t_i = \lambda \left(\frac{-\log U_i}{z_i}\right)^{1/\rho}
$$

Intuitively, each person draws a random "event ticket" ($U_i$); their liability-derived frailty ($z_i$) determines how quickly that ticket converts into disease onset — higher frailty means earlier onset.

The two traits use independent Weibull parameters ($\beta_k, \lambda_k, \rho_k$) and independent uniform draws, allowing different baseline hazard shapes: $\rho < 1$ produces a decreasing hazard (early-onset), $\rho = 1$ a constant hazard (exponential), and $\rho > 1$ an increasing hazard (late-onset).

### Censoring

In real studies, not every disease event is observed — people die of other causes, are too young to have developed the disease, or were not yet born when the study began. Censoring models these limitations.

Two independent censoring mechanisms are applied to the raw event times:

**Age-window censoring.** Each phenotyped generation $g$ has a configurable observation window $[a_g^L,\, a_g^R]$ representing the period during which events are ascertainable (e.g., $[40, 80]$ for the oldest cohort, $[0, 45]$ for the youngest). An individual with raw onset $t_i$ is left-censored if $t_i < a_g^L$ (onset before observation began), right-censored if $t_i > a_g^R$ (onset after follow-up ended), and the observed time is clipped to the window:

$$
t_{\text{obs},i} = \text{clip}(t_i,\; a_g^L,\; a_g^R)
$$

For example, with default settings the oldest generations (gen 0–2) are fully censored, while the youngest generation (gen 5) has window $[0, 45]$, so indiviudals in this generation will only be marked as affected if their age-of-onset is before age 45.

**Competing-risk death censoring.** A random age of mortality is drawn per individual from a Weibull distribution with mortality-specific parameters ($\lambda_d, \rho_d$):

$$
t_{\text{death},i} = \lambda_d \left(-\log U_i^{(d)}\right)^{1/\rho_d}, \quad U_i^{(d)} \sim \text{Uniform}(0, 1]
$$

If disease onset occurs after death ($t_{\text{obs},i} > t_{\text{death},i}$), the individual is death-censored and the observed time is set to the death age. An individual is classified as affected ($\delta_i = 1$) only if the disease event is observed within the age window and before death:

$$
\delta_i = \mathbf{1}[\text{not age-censored}] \;\cdot\; \mathbf{1}[\text{not death-censored}]
$$

An individual is classified as affected only if disease onset falls within the observation window *and* before death — both conditions must hold simultaneously.

### Liability-threshold model

This model directly classifies individuals as affected or unaffected by a liability cutoff — appropriate when only case/control status is available, without timing information.

As an alternative to the time-to-event phenotype, a binary affection status can be assigned via the liability-threshold model. Within each generation $g$, the liability ($\tilde{L}$) is standardised to zero mean and unit variance.

A generation-specific prevalence $\pi_g$ defines the threshold as the $(1 - \pi_g)$ quantile of the standardised liability distribution. Individuals whose standardised liability exceeds this threshold are classified as affected:

$$
\delta_{i,k} = \mathbf{1}\!\left[\tilde{L}_{i,k}^{(g)} \geq \Phi^{-1}(1 - \pi_g)\right]
$$

where $\Phi^{-1}$ is the standard normal quantile function. Stated simply, if the prevalence is 10% in a particular generation, the top 10% of the liability distribution is classified as affected. Prevalence may be specified as a scalar (uniform across generations) or per-generation.

## Validation via statistical analysis

### Relationship pair extraction

Before computing correlations, individuals must be grouped into pairs by relationship type — the correlation structure within each group is what identifies genetic and environmental effects.

For downstream correlation analyses, individuals are organised into seven relationship categories: MZ twins, full siblings, maternal half-siblings, paternal half-siblings, mother-offspring pairs, father-offspring pairs, and first cousins. Full and half-siblings are identified via merge-based groupby on shared parental identifiers. First cousins are enumerated by grouping individuals by grandparent and generating all cross-family pair combinations, with deduplication via canonical integer key packing.

### Tetrachoric correlation estimation

Binary phenotypes (affected/unaffected) systematically underestimate the true association between relatives because they discard information about *how far* above or below the threshold each person falls. Tetrachoric correlation corrects for this by assuming an underlying continuous bivariate normal liability distribution.

Tetrachoric correlations between binary affection indicators are estimated for each relationship type under the assumption that the observed dichotomy arises from an underlying bivariate normal liability distribution $(L_1, L_2) \sim \text{BVN}(0, 0, 1, 1, r)$. For thresholds $t_k = \Phi^{-1}(1 - \hat{\pi}_k)$ derived from observed prevalences, the four cell probabilities of the $2 \times 2$ contingency table are:

$$
P_{11}(r) = P(L_1 > t_1,\; L_2 > t_2 \mid r)
$$

and analogously for $P_{10}, P_{01}, P_{00}$. The bivariate normal CDF is evaluated via Owen's $T$ function. The correlation $\hat{r}$ is obtained by maximum likelihood, minimising the negative log-likelihood:

$$
-\ell(r) = -\sum_{(a,b) \in \{0,1\}^2} n_{ab} \log P_{ab}(r)
$$

over $r \in (-0.999, 0.999)$ using bounded scalar optimisation. The standard error is derived from the observed Fisher information $I(\hat{r}) = n \cdot \phi_2(t_1, t_2; \hat{r})^2 / \prod_{(a,b)} P_{ab}(\hat{r})$, where $\phi_2$ is the bivariate normal density. In essence, this answers: given the observed concordance patterns among relatives, what is the most likely underlying liability correlation?

### Pairwise Weibull survival correlation estimation

When phenotypes include censored survival times rather than simple binary outcomes, the tetrachoric approach is biased because it ignores *when* events occur and whether observations are censored. The pairwise Weibull method uses the full survival data — event times and censoring indicators — to estimate liability correlations.

To estimate liability correlations from censored time-to-event data, we employ a pairwise composite likelihood approach. For a pair $(i, j)$ with correlated liabilities $(L_i, L_j) \sim \text{BVN}(0, 0, 1, 1, r)$, the marginal pairwise likelihood is:

$$
\mathcal{L}(r) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty}
g(t_i, \delta_i \mid L_i)\; g(t_j, \delta_j \mid L_j)\;
\phi_2(L_i, L_j;\, r)\; dL_i\, dL_j
$$

where $g(t, \delta \mid L) = h(t \mid L)^\delta \cdot S(t \mid L)$ is the individual Weibull contribution (hazard for events, survival for censored observations). The bivariate integral is evaluated via two-dimensional Gauss-Hermite quadrature (probabilist's convention) with $n_q = 20$ nodes per dimension. Using the Cholesky decomposition $L_i = x_m$ and $L_j = r\,x_m + \sqrt{1 - r^2}\,x_n$, the integral becomes:

$$
\mathcal{L}(r) \approx \sum_{m=1}^{n_q} \sum_{n=1}^{n_q}
w_m\, w_n\;
g(t_i, \delta_i \mid x_m)\;
g(t_j, \delta_j \mid r\,x_m + \sqrt{1-r^2}\,x_n)
$$

Numerical integration is necessary because the liabilities themselves are unobserved — we must integrate over all possible liability values, weighted by the bivariate normal probability that each pair of values would produce the observed survival data.

Log-sum-exp stabilisation is applied per pair to prevent numerical overflow. The total negative log-likelihood across all pairs is minimised over $r \in (-0.999, 0.999)$ via bounded scalar optimisation. Standard errors are computed from the numerical Hessian using a central second-difference approximation with step size $h = 10^{-4}$:

$$
\hat{d}^2 = \frac{-\ell(\hat{r} + h) - 2(-\ell(\hat{r})) + (-\ell(\hat{r} - h))}{h^2}, \quad
\text{SE}(\hat{r}) = \frac{1}{\sqrt{\hat{d}^2}}
$$

The inner likelihood evaluation is compiled with Numba JIT (`@njit`, `parallel=True`) and parallelised over pairs using `prange`, eliminating temporary three-dimensional array allocations and achieving approximately 5-7x speedup over the equivalent NumPy broadcasting implementation. A pure NumPy fallback is retained for environments without Numba.

### Heritability estimation

With correlations estimated for each relationship type, heritability follows from classical twin-study logic.

Narrow-sense heritability is estimated by Falconer's formula from liability correlations of MZ and dizygotic (DZ; full-sibling) pairs:

$$
\hat{h}^2 = 2(\hat{r}_{\text{MZ}} - \hat{r}_{\text{DZ}})
$$

The expected MZ liability correlation is $A + C$ and the expected DZ correlation is $\tfrac{1}{2}A + C$, yielding $\hat{h}^2 \approx A$. The intuition is straightforward: if MZ twins are much more correlated than DZ pairs, the difference must be due to their extra genetic sharing — and that difference directly estimates heritability. Parent-offspring regressions provide a complementary estimate: offspring liability is regressed on midparent liability, where the expected slope equals the heritability $A$ under the infinitesimal model.

## Validation

Because the ground-truth parameters are known for every simulated individual, every output can be checked against expectation. Validation confirms both that the code is functioning correctly and that the statistical estimators perform as intended.

Automated validation checks are performed on each simulated replicate and organised into five categories:

**Structural integrity.** Verification that individual identifiers are contiguous, parent references are valid (or $-1$ for founders), mothers are female, fathers are male, and the sex ratio is approximately balanced (0.45-0.55).

**Twin properties.** Bidirectional consistency of twin pointers; MZ pairs share identical parents, $A$ values (within floating-point tolerance), and sex; the observed twin rate matches the expected rate $2 \cdot p_{\text{MZ}} \cdot f_{\text{elig}}$, where $f_{\text{elig}}$ is the fraction of eligible birth positions.

**Half-sibling statistics.** The observed proportion of maternal half-sibling pairs among all maternal sibling pairs is compared to the expected $1 - (1 - p_{\text{nonsocial}})^2$.

**Variance-component recovery.** Founder-generation variances of $A$, $C$, and $E$ are compared to configured parameters; cross-trait correlations $r_A$ and $r_C$ are verified; $E$ components are confirmed to be uncorrelated across siblings and across traits; and $C$ values are confirmed to be identical within households.

**Population-level checks.** Generation sizes equal $N$; the number of recorded generations equals $G_{\text{ped}}$; the mean family size matches the configured Poisson rate. Tolerances are set as $\max(4 \cdot \text{SE}, \epsilon_{\min})$ where SE is the sampling standard error of the relevant statistic.

## Implementation

The simulation framework is implemented in Python as an installable package (`sim_ace`) with NumPy for vectorised array operations, SciPy for optimisation and special functions, pandas and PyArrow for data management, and Numba for JIT-compiled numerical kernels. The workflow is managed by Snakemake with per-scenario configuration, per-replicate seed offsets for reproducibility (seed incremented by replicate number), and support for SLURM-based HPC execution. All random number generation uses NumPy's PCG64 generator (`numpy.random.default_rng`) with deterministic seed management across pipeline stages (trait-2 phenotyping uses seed $+ 100$; death censoring uses seed $+ 1000$).

## Assumptions and limitations

Every simulation makes simplifying assumptions. The following are the most consequential for interpreting results and should be considered when evaluating the framework:

**No assortative mating.** Spouses are paired randomly with respect to phenotype and liability. Real populations exhibit phenotypic and genetic assortative mating, which inflates sibling correlations and can bias heritability estimates upward. Extensions incorporating spousal correlation on liability would be needed to model this effect.

**No gene-environment interaction.** Liability is strictly additive ($L = A + C + E$) with no multiplicative or interactive terms. If the true data-generating process involves gene-environment interaction ($G \times E$) or gene-environment correlation ($rGE$), the ACE decomposition will absorb these effects into the additive components, potentially biasing variance estimates.

**No cross-trait unique-environment correlation ($r_E = 0$).** Unique-environment components $E_1$ and $E_2$ are drawn independently across traits. This precludes modelling trait pairs where individual-specific environmental exposures (e.g., shared lifestyle factors) induce correlation beyond what is captured by $A$ and $C$.

**No environmental transmission across generations.** The common-environment component $C$ is drawn freshly per household each generation with no autoregressive transmission from parental $C$ values. This is the standard ACE assumption but does not capture intergenerational persistence of socioeconomic status, neighbourhood effects, or cultural transmission that may inflate parent-offspring resemblance in real populations.

**Fixed population size.** Each generation contains exactly $N$ individuals with no population growth, decline, bottlenecks, or migration. Demographic dynamics that alter effective population size or introduce population stratification are not modelled.

**Tetrachoric correlation bias under censoring.** When affection status is derived from the Weibull frailty model with age-window and death censoring, the observed prevalence may differ from the uncensored prevalence. Tetrachoric correlations estimated from censored binary outcomes are attenuated relative to the true underlying liability correlation. The pairwise Weibull survival correlation method, which explicitly accounts for censoring in the likelihood, is preferred for validation of censored phenotypes.
