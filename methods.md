# ACE Simulation Framework: Methods

*Draft for academic paper. Full equations, covering both Weibull frailty and liability threshold models.*

---

## Simulation Overview

We developed a multi-generational population simulation framework implementing the classical ACE variance-component model, in which an individual's quantitative liability is decomposed into additive genetic ($A$), common shared-environment ($C$), and unique-environment ($E$) components. The simulation generates pedigrees with realistic family structures — including monozygotic (MZ) twins, half-siblings, and configurable non-paternity — and maps continuous liabilities to observable phenotypes via either a Weibull proportional-hazards frailty model (time-to-event) or a liability-threshold model (binary affection). The pipeline is orchestrated by Snakemake and supports named scenario configurations, replicate runs, and automated validation.

## Pedigree Simulation

### Founder generation

A founder cohort of $N$ individuals (default $N = 100{,}000$) is initialised at generation $g = 0$. Sex is assigned as $\text{sex}_i \sim \text{Bernoulli}(0.5)$, coded 0 = female, 1 = male.

For two correlated traits ($k = 1, 2$), each founder receives variance components drawn from zero-mean bivariate normal distributions. Total phenotypic variance is normalised to unity for each trait, so the parameters represent variance proportions directly:

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

### Mating and family structure

At each generation, a new offspring cohort of exactly $N$ individuals is produced. Females from the parental generation serve as mothers, each assigned a family size drawn from a Poisson distribution:

$$
n_{\text{children},f} \sim \text{Poisson}(\lambda), \quad \lambda = 2.3 \text{ (default)}
$$

Family sizes are drawn sequentially and the final family is clipped so that total offspring sum to exactly $N$. Males are paired one-to-one with mothers to form households. All children within a household share a household identifier, which determines shared-environment ($C$) values.

To model non-paternity, each child independently has probability $p_{\text{nonsocial}}$ (default 0.05) of having a biological father drawn uniformly at random from all males, rather than the household's social father. This creates maternal half-siblings (same mother, different biological father) at an expected pairwise proportion of $1 - (1 - p_{\text{nonsocial}})^2$ among all maternal sibling pairs.

### Monozygotic twin generation

MZ twin pairs are generated positionally within the birth order. At each offspring position (excluding the last in the cohort), a Bernoulli trial with probability $p_{\text{MZ}}$ (default 0.02) determines whether that child and the next form an MZ pair. When consecutive positions both succeed, the second is suppressed to prevent overlapping pairs. MZ twins are assigned identical biological parents and the same household identifier.

### Offspring inheritance

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

MZ twins share identical $A$ values: the second twin's breeding values and sex are copied from the first.

**Common environment ($C$).** A single bivariate draw $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_C)$ is made per household and assigned to all children within it. Critically, $C$ is *not* transmitted from parent to child — it represents the offspring's own shared rearing environment. This is the standard ACE assumption with no autoregressive $C$ transmission.

**Unique environment ($E$).** Each child receives independent draws $E_{i,k} \sim \mathcal{N}(0, \sigma_{E_k})$, uncorrelated across siblings, twins, and traits.

### Burn-in and recording

The simulation runs for $G_{\text{sim}}$ total generations (default 6), of which the first $G_{\text{sim}} - G_{\text{ped}}$ generations (default 2) serve as burn-in to reduce founder-specific effects and approach equilibrium allele-frequency distributions. Only the final $G_{\text{ped}}$ generations (default 4) are recorded in the output pedigree. The recorded pedigree contains $N \times G_{\text{ped}}$ individuals with contiguous identifiers, each annotated with generation number, parental identifiers, MZ twin partner (if any), household identifier, all six variance components, and both trait liabilities.

## Phenotype Models

### Weibull proportional-hazards frailty model

Continuous liabilities are mapped to age-at-onset via a Weibull proportional-hazards frailty model. For each trait $k$, the individual liability is first standardised within the recorded generations to $\tilde{L}_{i,k} = (L_{i,k} - \bar{L}_k) / s_{L_k}$. The conditional hazard function is:

$$
h(t \mid \tilde{L}) = \frac{\rho}{\lambda} \left(\frac{t}{\lambda}\right)^{\rho - 1} \exp(\beta\,\tilde{L})
$$

where $\lambda > 0$ is the Weibull scale, $\rho > 0$ is the shape, and $\beta$ is the log-hazard coefficient for standardised liability. The corresponding survival function is:

$$
S(t \mid \tilde{L}) = \exp\!\left[-\left(\frac{t}{\lambda}\right)^{\!\rho} \exp(\beta\,\tilde{L})\right]
$$

Event times are generated by inverse-CDF sampling. Defining the individual frailty as $z_i = \exp(\beta\,\tilde{L}_i)$ and drawing $U_i \sim \text{Uniform}(0, 1]$:

$$
t_i = \lambda \left(\frac{-\log U_i}{z_i}\right)^{1/\rho}
$$

The two traits use independent Weibull parameters ($\beta_k, \lambda_k, \rho_k$) and independent uniform draws, allowing different baseline hazard shapes: $\rho < 1$ produces a decreasing hazard (early-life risk), $\rho = 1$ a constant hazard (exponential), and $\rho > 1$ an increasing hazard (late-onset risk).

### Censoring

Two independent censoring mechanisms are applied to the raw event times:

**Age-window censoring.** Each generation $g$ has a configurable observation window $[a_g^L,\, a_g^R]$ representing the period during which events are ascertainable (e.g., $[40, 80]$ for the oldest cohort, $[0, 45]$ for the youngest). An individual with raw onset $t_i$ is left-censored if $t_i < a_g^L$ (onset before observation began), right-censored if $t_i > a_g^R$ (onset after follow-up ended), and the observed time is clipped to the window:

$$
t_{\text{obs},i} = \text{clip}(t_i,\; a_g^L,\; a_g^R)
$$

**Competing-risk death censoring.** A single death age is drawn per individual from a Weibull distribution with mortality-specific parameters ($\lambda_d, \rho_d$), shared across both traits:

$$
t_{\text{death},i} = \lambda_d \left(-\log U_i^{(d)}\right)^{1/\rho_d}, \quad U_i^{(d)} \sim \text{Uniform}(0, 1]
$$

If disease onset occurs after death ($t_{\text{obs},i} > t_{\text{death},i}$), the individual is death-censored and the observed time is set to the death age. An individual is classified as affected ($\delta_i = 1$) only if the disease event is observed within the age window and before death:

$$
\delta_i = \mathbf{1}[\text{not age-censored}] \;\cdot\; \mathbf{1}[\text{not death-censored}]
$$

### Liability-threshold model

As an alternative to the time-to-event phenotype, a binary affection status can be assigned via the liability-threshold model. Within each generation $g$, the liability is standardised to zero mean and unit variance:

$$
\tilde{L}_{i,k}^{(g)} = \frac{L_{i,k} - \bar{L}_k^{(g)}}{s_{L_k}^{(g)}}
$$

A generation-specific prevalence $\pi_g$ defines the threshold as the $(1 - \pi_g)$ quantile of the standardised liability distribution. Individuals whose standardised liability exceeds this threshold are classified as affected:

$$
\delta_{i,k} = \mathbf{1}\!\left[\tilde{L}_{i,k}^{(g)} \geq \Phi^{-1}(1 - \pi_g)\right]
$$

where $\Phi^{-1}$ is the standard normal quantile function. Prevalence may be specified as a scalar (uniform across generations) or per-generation, enabling simulation of secular trends in disease frequency.

## Statistical Analysis

### Relationship pair extraction

For downstream correlation analyses, individuals are organised into seven relationship categories: MZ twins, full siblings, maternal half-siblings, paternal half-siblings, mother-offspring pairs, father-offspring pairs, and first cousins. Full and half-siblings are identified via merge-based groupby on shared parental identifiers. First cousins are enumerated by grouping individuals by grandparent and generating all cross-family pair combinations, with deduplication via canonical integer key packing.

### Tetrachoric correlation estimation

Tetrachoric correlations between binary affection indicators are estimated for each relationship type under the assumption that the observed dichotomy arises from an underlying bivariate normal liability distribution $(L_1, L_2) \sim \text{BVN}(0, 0, 1, 1, r)$. For thresholds $t_k = \Phi^{-1}(1 - \hat{\pi}_k)$ derived from observed prevalences, the four cell probabilities of the $2 \times 2$ contingency table are:

$$
P_{11}(r) = P(L_1 > t_1,\; L_2 > t_2 \mid r)
$$

and analogously for $P_{10}, P_{01}, P_{00}$. The bivariate normal CDF is evaluated via Owen's $T$ function. The correlation $\hat{r}$ is obtained by maximum likelihood, minimising the negative log-likelihood:

$$
-\ell(r) = -\sum_{(a,b) \in \{0,1\}^2} n_{ab} \log P_{ab}(r)
$$

over $r \in (-0.999, 0.999)$ using bounded scalar optimisation. The standard error is derived from the observed Fisher information $I(\hat{r}) = n \cdot \phi_2(t_1, t_2; \hat{r})^2 / \prod_{(a,b)} P_{ab}(\hat{r})$, where $\phi_2$ is the bivariate normal density.

### Pairwise Weibull survival correlation estimation

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

Log-sum-exp stabilisation is applied per pair to prevent numerical overflow. The total negative log-likelihood across all pairs is minimised over $r \in (-0.999, 0.999)$ via bounded scalar optimisation. Standard errors are computed from the numerical Hessian using a central second-difference approximation with step size $h = 10^{-4}$:

$$
\hat{d}^2 = \frac{-\ell(\hat{r} + h) - 2(-\ell(\hat{r})) + (-\ell(\hat{r} - h))}{h^2}, \quad
\text{SE}(\hat{r}) = \frac{1}{\sqrt{\hat{d}^2}}
$$

The inner likelihood evaluation is compiled with Numba JIT (`@njit`, `parallel=True`) and parallelised over pairs using `prange`, eliminating temporary three-dimensional array allocations and achieving approximately 5-7x speedup over the equivalent NumPy broadcasting implementation. A pure NumPy fallback is retained for environments without Numba.

### Heritability estimation

Narrow-sense heritability is estimated by Falconer's formula from liability correlations of MZ and dizygotic (DZ; full-sibling) pairs:

$$
\hat{h}^2 = 2(\hat{r}_{\text{MZ}} - \hat{r}_{\text{DZ}})
$$

The expected MZ liability correlation is $A + C$ and the expected DZ correlation is $\tfrac{1}{2}A + C$, yielding $\hat{h}^2 \approx A$. Parent-offspring regressions provide a complementary estimate: offspring liability is regressed on midparent liability, where the expected slope equals the heritability $A$ under the infinitesimal model.

## Validation

Automated validation checks are performed on each simulated replicate and organised into five categories:

**Structural integrity.** Verification that individual identifiers are contiguous, parent references are valid (or $-1$ for founders), mothers are female, fathers are male, and the sex ratio is approximately balanced (0.45-0.55).

**Twin properties.** Bidirectional consistency of twin pointers; MZ pairs share identical parents, $A$ values (within floating-point tolerance), and sex; the observed twin rate matches the expected rate $2 \cdot p_{\text{MZ}} \cdot f_{\text{elig}}$, where $f_{\text{elig}}$ is the fraction of eligible birth positions.

**Half-sibling statistics.** The observed proportion of maternal half-sibling pairs among all maternal sibling pairs is compared to the expected $1 - (1 - p_{\text{nonsocial}})^2$.

**Variance-component recovery.** Founder-generation variances of $A$, $C$, and $E$ are compared to configured parameters; cross-trait correlations $r_A$ and $r_C$ are verified; $E$ components are confirmed to be uncorrelated across siblings and across traits; and $C$ values are confirmed to be identical within households.

**Population-level checks.** Generation sizes equal $N$; the number of recorded generations equals $G_{\text{ped}}$; the mean family size matches the configured Poisson rate. Tolerances are set as $\max(4 \cdot \text{SE}, \epsilon_{\min})$ where SE is the sampling standard error of the relevant statistic.

## Implementation

The simulation framework is implemented in Python as an installable package (`sim_ace`) with NumPy for vectorised array operations, SciPy for optimisation and special functions, pandas and PyArrow for data management, and Numba for JIT-compiled numerical kernels. The workflow is managed by Snakemake with per-scenario configuration, per-replicate seed offsets for reproducibility (seed incremented by replicate number), and support for SLURM-based HPC execution. All random number generation uses NumPy's PCG64 generator (`numpy.random.default_rng`) with deterministic seed management across pipeline stages (trait-2 phenotyping uses seed $+ 100$; death censoring uses seed $+ 1000$).

## Limitations

The following simplifying assumptions are inherent to the current framework and should be considered when interpreting simulation results:

**No assortative mating.** Spouses are paired randomly with respect to phenotype and liability. Real populations exhibit phenotypic and genetic assortative mating, which inflates sibling correlations and can bias heritability estimates upward. Extensions incorporating spousal correlation on liability would be needed to model this effect.

**No gene-environment interaction.** Liability is strictly additive ($L = A + C + E$) with no multiplicative or interactive terms. If the true data-generating process involves gene-environment interaction ($G \times E$) or gene-environment correlation ($rGE$), the ACE decomposition will absorb these effects into the additive components, potentially biasing variance estimates.

**No cross-trait unique-environment correlation ($r_E = 0$).** Unique-environment components $E_1$ and $E_2$ are drawn independently across traits. This precludes modelling trait pairs where individual-specific environmental exposures (e.g., shared lifestyle factors) induce correlation beyond what is captured by $A$ and $C$.

**No environmental transmission across generations.** The common-environment component $C$ is drawn freshly per household each generation with no autoregressive transmission from parental $C$ values. This is the standard ACE assumption but does not capture intergenerational persistence of socioeconomic status, neighbourhood effects, or cultural transmission that may inflate parent-offspring resemblance in real populations.

**Fixed population size.** Each generation contains exactly $N$ individuals with no population growth, decline, bottlenecks, or migration. Demographic dynamics that alter effective population size or introduce population stratification are not modelled.

**Tetrachoric correlation bias under censoring.** When affection status is derived from the Weibull frailty model with age-window and death censoring, the observed prevalence may differ from the uncensored prevalence. Tetrachoric correlations estimated from censored binary outcomes are attenuated relative to the true underlying liability correlation. The pairwise Weibull survival correlation method, which explicitly accounts for censoring in the likelihood, is preferred for validation of censored phenotypes.
