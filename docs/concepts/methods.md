# ACE Simulation Framework: Methods

*This document describes the ACE simulation framework: how it generates synthetic multi-generational family data with known genetic and environmental parameters, assigns disease phenotypes via pluggable survival and threshold models, and validates statistical recovery of those parameters. The framework supports random or assortative mating, six baseline hazard distributions, mixture cure and ADuLT phenotype models, sex-specific effects, observation-window and competing-risk censoring, pedigree dropout, and case-ascertainment subsampling. It provides a controlled testbed for evaluating twin- and family-study methods, where ground-truth values are available for comparison against estimates.*

---

## Simulation Overview

simACE is a multi-generational individual-based simulation framework implementing an ACE variance-component model, in which an individual's quantitative liability is decomposed into additive genetic ($A$), common shared-environment ($C$), and unique-environment ($E$) components.

The simulation generates a multi-generational pedigree with realistic family structures including full- and half-siblings, and monozygotic (MZ) twins. Mating may be random or assortative on one or both trait liabilities.

Binary phenotypes are related to the liability via a proportional-hazards frailty survival model (with a choice of six baseline hazard distributions), a mixture cure frailty model, an ADuLT liability threshold model, or a simple liability-threshold model.

Censoring is included in the survival models, and includes random mortality as well as deterministic left- and right-censoring by age, configurable by generation.

The simulation code is written in Python and the pipeline is orchestrated by Snakemake and supports named scenarios, replicate runs, and automated validation including sanity checks and plots.

**Key Terms**

- **Liability** — an unobserved continuous score representing an individual's underlying risk for a disease; higher liability means higher risk.
- **Founders** — the initial generation of pedigree individuals whose genetic and environmental values are drawn from scratch rather than inherited.
- **Burn-in** — early simulated generations that are discarded so that the recorded pedigree is not influenced by the arbitrary starting conditions of the founder generation.
- **Frailty** — a multiplicative modifier on disease hazard derived from an individual's liability; it controls how quickly risk accumulates with age.
- **Censoring** — the inability to observe a disease event because the individual died, was too young or old, or was not yet born during the study window.
- **Prevalence** — the proportion of individuals classified as affected in a given generation.
- **Assortative mating** — non-random partner selection in which individuals with similar (positive assortment) or dissimilar (negative assortment) liability values mate preferentially, inflating parent-offspring and sibling resemblance beyond what random mating produces.
- **Cure fraction** — the proportion of the population that will never develop the disease regardless of follow-up time; in the mixture cure model this equals $1 - K$, where $K$ is the prevalence.
- **Cumulative incidence proportion (CIP)** — the probability that an individual has experienced the event by a given age; used in the ADuLT models to map liability rank to age-at-onset.

## Conceptual simulation steps

```
+----------+  +---------+  +-----------+  +--------+  +--------+  +-----------+  +---------+
|1. Pedigree|->|2. Ped.  |->|3. Pheno-  |->|4. Cen- |->|5. Sub- |->|6. Statis- |->|7. Valid-|
|   Simul-  |  |  Drop-  |  |   type    |  |  sor-  |  |  sam-  |  |   tical   |  |  ation  |
|   ation   |  |   out   |  |  Assign.  |  |  ing   |  |  ple   |  |  Analysis |  |         |
|           |  |         |  |           |  |        |  |        |  |           |  |         |
|Build fam- |  |Remove   |  |Convert to |  |Apply   |  |Draw    |  |Estimate   |  |Compare  |
|ilies with |  |random   |  |observable |  |age &   |  |study   |  |corr. &    |  |to ground|
|known ACE  |  |fraction |  |outcomes   |  |death   |  |sample  |  |heritab.   |  |truth    |
+----------+  +---------+  +-----------+  +--------+  +--------+  +-----------+  +---------+
```

**Stage 1 — Pedigree Simulation** creates a multi-generational population where each individual has known additive-genetic (A), shared-environment (C), and unique-environment (E) values, linked to parents and siblings through realistic family structures. Mating may be random or assortative.

**Stage 2 — Pedigree Dropout** (optional) randomly removes a fraction of individuals from the pedigree and severs any parent/twin links that reference removed individuals, simulating incomplete ascertainment of pedigree structure.

**Stage 3 — Phenotype Assignment** converts continuous liabilities into observable outcomes — an age-at-onset via a survival model, a binary affected/unaffected status via a threshold model, or a combination of both in the mixture cure model — and optionally incorporates sex-specific effects on the hazard or threshold.

**Stage 4 — Censoring** applies observation-window constraints and competing-risk mortality to the raw event times, producing the final observed phenotype data.

**Stage 5 — Subsampling** (optional) draws a study sample from the full population, with optional case-ascertainment weighting to enrich for affected individuals.

**Stage 6 — Statistical Analysis** estimates correlations between relatives and computes heritability using only the observable phenotype data, as a real researcher would.

**Stage 7 — Validation** compares every estimate back to the known ground-truth parameters, confirming both code correctness and estimator performance.

## Pedigree Simulation

### Founder generation

The simulation begins by creating a starting population from scratch. Each founder receives randomly drawn additive-genetic, shared-environment, and unique-environment values — these are the ground-truth components that downstream analyses will attempt to recover.

A founder cohort of $N$ individuals (default $N = 100{,}000$) is initialised at generation $g = 0$. Sex is assigned as $\text{sex}_i \sim \text{Bernoulli}(0.5)$, coded 0 = female, 1 = male.

**Single-trait ACE decomposition.** For a single trait, each individual's liability is the sum of three independent components:

$$
L_i = A_i + C_i + E_i
$$

where $A_i \sim \mathcal{N}(0, \sigma^2_A)$ is the additive genetic component, $C_i \sim \mathcal{N}(0, \sigma^2_C)$ is the common (shared) environment, and $E_i \sim \mathcal{N}(0, \sigma^2_E)$ is the unique environment. Total phenotypic variance is normalised to unity, so $\sigma^2_A + \sigma^2_C + \sigma^2_E = 1$ and the variance components directly represent proportions of total variance. For example, with parameters $A = 0.5$, $C = 0.2$, $E = 0.3$, half the variation in liability is additive genetic, 20% is due to the shared household, and 30% is individual-specific noise.

**Extension to two correlated traits.** The simulation models two traits jointly ($k = 1, 2$), allowing genetic (A) and shared-environment (C) components to be correlated across traits. Each founder receives variance components drawn from bivariate normal distributions:

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

### Reproduction — mating and family structure

The simulation maintains a constant population size ($N$) across generations. Each new generation must include $N$ offspring, produced by pairing males and females from the parental generation.

**Mating counts.** Each parent draws a mating count from a zero-truncated Poisson (ZTP) distribution — a Poisson draw conditioned on being at least 1, so every individual participates in at least one mating:

$$
n_{\text{matings},i} \sim \text{ZTP}(\lambda), \quad \lambda = 2.3 \text{ (default)}
$$

The zero-truncated Poisson is implemented by rejection: Poisson draws of zero are redrawn until all counts are $\geq 1$. Males and females draw independently. The total mating-slot counts are then balanced by randomly trimming slots from the sex with more total slots, so that both sexes contribute the same total number of mating slots $T = \min(\sum n_{\text{male}}, \sum n_{\text{female}})$.

**Partner pairing.** Each parent's mating count is expanded into that many "slots" via replication. Under random mating (default), the male slot array is shuffled and paired positionally with the female slot array, producing $M = T$ mating pairs. Under assortative mating, the pairing algorithm is described in the next section. Duplicate $(mother, father)$ pairs are resolved by swapping conflicting entries with nearby partners.

**Offspring allocation.** The $N$ offspring are distributed across the $M$ matings via a multinomial draw with equal probabilities:

$$
(c_1, c_2, \ldots, c_M) \sim \text{Multinomial}(N,\; 1/M, \ldots, 1/M)
$$

so each mating receives a random number of children, with the total summing to exactly $N$.

**Household assignment.** All offspring sharing the same mother are assigned the same household identifier, and share a common-environment ($C$) draw. This means maternal half-siblings (same mother, different father) share $C$, while paternal half-siblings (same father, different mother) do not.

### Assortative mating

When assortative mating is enabled (parameters $\text{assort}_1 \neq 0$ or $\text{assort}_2 \neq 0$), couple formation is modified so that individuals with correlated liability values are paired preferentially. The target mate Pearson correlations are $r_1 = \text{assort}_1$ for trait 1 and $r_2 = \text{assort}_2$ for trait 2.

**Single-trait case.** When only one assortment parameter is nonzero, a bivariate Gaussian copula approach is used. Each parent's liability is converted to a rank-based score. The effective target correlation is $r_{\text{eff}} = \min(\sqrt{r_1^2 + r_2^2},\; 1)$. A bivariate normal sample $(z_f, z_m) \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ with $\Sigma_{12} = r_{\text{eff}}$ is drawn for each mating slot. Females and males, each pre-sorted by their weighted rank score $|r_1| \cdot \text{rank}_{1} + |r_2| \cdot \text{rank}_{2}$, are then paired according to the bivariate normal rank ordering — producing mate pairs whose liability correlation approximates the target. Negative assortment is achieved by reversing the relevant rank ordering before scoring.

**Both-traits case (4-variate Gaussian copula).** When both $r_1$ and $r_2$ are nonzero, the algorithm targets a 4-variate Gaussian copula structure following Border et al. (2022, *Science*, Eq. 2). Let $\mathbf{R}_{mf}$ denote the $2 \times 2$ target mate-correlation matrix:

$$
\mathbf{R}_{mf} = \begin{bmatrix} r_1 & c \\ c & r_2 \end{bmatrix}
$$

where $c = \rho_w \sqrt{|r_1 r_2|} \operatorname{sign}(r_1 r_2)$ is the cross-trait cross-sex mate correlation induced by the within-person liability correlation $\rho_w$. Alternatively, the user may specify the full $\mathbf{R}_{mf}$ matrix directly via the `assort_matrix` parameter, in which case $r_1 = R_{mf,11}$, $r_2 = R_{mf,22}$, and $c = R_{mf,12} = R_{mf,21}$ (symmetry is enforced).

Let $\mathbf{R}_{ff}$ be the within-female cross-trait liability correlation matrix:

$$
\mathbf{R}_{ff} = \begin{bmatrix} 1 & \rho_w \\ \rho_w & 1 \end{bmatrix}
$$

where $\rho_w$ is the within-person cross-trait liability correlation. The full 4-variate matrix $\boldsymbol{\Sigma}_4 = \bigl[\begin{smallmatrix} \mathbf{R}_{ff} & \mathbf{R}_{mf}^\top \\ \mathbf{R}_{mf} & \mathbf{R}_{ff} \end{smallmatrix}\bigr]$ must be positive semi-definite; this is validated at configuration time. The algorithm proceeds in two phases:

*Phase 1 — Conditional-expectation initialization.* Each parent's liability is converted to quantile-normal scores. The conditional-expectation matrix $\mathbf{B} = \mathbf{R}_{mf} \mathbf{R}_{ff}^{-1}$ maps female scores to expected male scores. Female quantile-normal vectors are projected through $\mathbf{B}$ to obtain target male vectors. Both targets and actual male scores are projected onto the dominant right singular vector of $\mathbf{R}_{mf}$ (via SVD), and males are rank-matched to females along this projection — providing a good initial permutation.

*Phase 2 — Metropolis greedy refinement.* Random pairs of male positions $(i, j)$ are proposed for swapping. A swap is accepted if it reduces the total squared error across all four elements of $\mathbf{R}_{mf}$:

$$
\sum_{k \in \{1,\, 2,\, 12,\, 21\}} (S_k + \Delta_k - T_k)^2 < \sum_{k \in \{1,\, 2,\, 12,\, 21\}} (S_k - T_k)^2
$$

where $S_1, S_2$ are the same-trait running cross-product sums, $S_{12} = \sum_m z_{f,1}^{(m)} z_{m,2}^{(m)}$ and $S_{21} = \sum_m z_{f,2}^{(m)} z_{m,1}^{(m)}$ are the cross-trait sums, $T_k = r_k \cdot M$ for same-trait and $T_{12} = T_{21} = c \cdot M$ for cross-trait targets. Refinement continues until the per-element correlation error is below $5 \times 10^{-4}$ or $8M$ proposals have been evaluated.

### Monozygotic twin generation

Identical (monozygotic) twins share 100% of their genetic material; the simulation inserts them into the pedigree at a configurable rate.

After offspring are allocated to matings, each mating with at least 2 offspring is eligible for twin assignment. A Bernoulli trial with probability $p_{\text{MZ}}$ (default 0.02) determines whether the mating produces an MZ twin pair. When assigned, the first two offspring of the mating are designated as MZ twins — they share identical biological parents, identical $A$ values, and the same sex. At most one MZ pair is generated per mating.

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

MZ twins share identical $A$ (and $C$) values: the second twin's breeding values and sex are copied from the first.

**Common environment ($C$).** A single bivariate draw $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_C)$ is made per household and assigned to all children within it. Critically, $C$ is *not* transmitted from parent to child — it represents the offspring's own shared rearing environment. That is, siblings share C because they grow up in the same household, but their parents' childhood environment does not carry over.

**Unique environment ($E$).** Each child receives independent draws $E_{i,k} \sim \mathcal{N}(0, \sigma_{E_k})$, uncorrelated across siblings, twins, and traits. Even MZ twins differ in E — it reflects all residual sources of liability.

### Burn-in and recording

See [Simulation Design § Multi-generational pedigree](simulation-design.md#multi-generational-pedigree) for the burn-in / recording / phenotyping split across $G_{\text{sim}}$, $G_{\text{ped}}$, and $G_{\text{pheno}}$.

The recorded pedigree contains $N \times G_{\text{ped}}$ individuals with contiguous identifiers, each annotated with generation number, parental identifiers, MZ twin partner (if any), household identifier, all six variance components, and both trait liabilities.

## Pedigree Dropout

As an optional post-simulation step, a random fraction of individuals can be removed from the pedigree to simulate incomplete observation of family structure. A dropout rate $d \in [0, 1)$ specifies the proportion to remove; $n_{\text{drop}} = \lfloor N_{\text{total}} \cdot d \rceil$ individuals are deleted uniformly at random without replacement. Deletion is unconditional on family structure, so it degrades pedigree connectivity without biasing which relationship types are affected.

See [Subsampling and Dropout § Pedigree dropout](../user-guide/subsampling-and-dropout.md#pedigree-dropout-pedigree_dropout_rate) for how broken parent/twin links propagate through downstream stages and which pre-configured scenarios use it.

## Phenotype Models

The pedigree simulation produces continuous liabilities, but real studies observe discrete outcomes — a diagnosis age or an affected/unaffected classification. The models below translate liabilities into these observable phenotypes.

### Proportional-hazards frailty model

This model simulates *when* disease occurs: higher liability leads to earlier onset (on average). Continuous liabilities are mapped to age-at-onset via a proportional-hazards frailty model. For each trait $k$, with liability $L$, the conditional hazard function is:

$$
h(t \mid L) = h_0(t) \cdot \exp(\beta\,L)
$$

where $h_0(t)$ is the baseline hazard function and $\beta$ scales the effect of individual liability on the log-hazard. The corresponding cumulative hazard and survival functions are:

$$
H(t \mid L) = H_0(t) \cdot \exp(\beta\,L), \quad S(t \mid L) = \exp\!\left[-H_0(t) \cdot \exp(\beta\,L)\right]
$$

Event times are generated by inverse-CDF sampling. Defining the individual frailty as $z_i = \exp(\beta\,\tilde{L}_i)$ and drawing $U_i \sim \text{Uniform}(0, 1]$, the event time is:

$$
t_i = H_0^{-1}\!\left(\frac{-\log U_i}{z_i}\right)
$$

Intuitively, each person draws a random "event ticket" ($U_i$); their liability-derived frailty ($z_i$) determines how quickly that ticket converts into disease onset — higher frailty means earlier onset. Six baseline hazard distributions are supported, each providing a different shape for how the background risk of disease changes with age:

| Model | Parameters | Baseline hazard $h_0(t)$ | Cumulative hazard $H_0(t)$ | Inverse $t = H_0^{-1}(x)$ |
|---|---|---|---|---|
| **Weibull** | scale $\lambda$, shape $\rho$ | $\frac{\rho}{\lambda}\left(\frac{t}{\lambda}\right)^{\rho-1}$ | $\left(\frac{t}{\lambda}\right)^{\!\rho}$ | $\lambda\, x^{1/\rho}$ |
| **Exponential** | rate $b$ | $b$ | $b\,t$ | $x / b$ |
| **Gompertz** | rate $b$, shape $\gamma$ | $b\,\exp(\gamma\,t)$ | $\frac{b}{\gamma}\bigl(\exp(\gamma\,t) - 1\bigr)$ | $\frac{1}{\gamma}\log\!\left(1 + \frac{x\,\gamma}{b}\right)$ |
| **Lognormal** | $\mu$, $\sigma$ | $\frac{\phi(z_t)}{\sigma\,t\,\bar{\Phi}(z_t)}$ | $-\log\bar{\Phi}(z_t)$ | $\exp\!\bigl(\mu + \sigma\,\Phi^{-1}(1 - e^{-x})\bigr)$ |
| **Loglogistic** | scale $\alpha$, shape $k$ | $\frac{(k/\alpha)(t/\alpha)^{k-1}}{1 + (t/\alpha)^k}$ | $\log\!\bigl(1 + (t/\alpha)^k\bigr)$ | $\alpha\,(e^{x} - 1)^{1/k}$ |
| **Gamma** | shape $k$, scale $\theta$ | $\frac{f_0(t)}{S_0(t)}$ | $-\log S_0(t)$ | $F_{\Gamma}^{-1}(1 - e^{-x};\, k,\, \theta)$ |

where $z_t = (\log t - \mu)/\sigma$, $\phi$ is the standard normal density, $\bar{\Phi} = 1 - \Phi$ is the survival function, $f_0$ and $S_0$ are the Gamma density and survival functions, and $F_{\Gamma}^{-1}$ is the Gamma quantile function.

The Weibull model with $\rho < 1$ produces a decreasing hazard (early-onset diseases), $\rho = 1$ reduces to the exponential (constant hazard), and $\rho > 1$ produces an increasing hazard (late-onset). The Gompertz model gives an exponentially increasing hazard, characteristic of age-related mortality. The lognormal and loglogistic models produce non-monotone hazards that first increase then decrease, suitable for diseases with a peak incidence age. The Gamma model provides a flexible alternative with similar behaviour.

The two traits use independent baseline parameters and independent uniform draws, allowing different baseline hazard shapes for each trait.

### Censoring

In real studies, not every disease event is observed — people die of other causes, are too young to have developed the disease, or were not yet born when the study began. Censoring models these limitations.

Two independent censoring mechanisms are applied to the raw event times:

**Age-window censoring.** Each phenotyped generation $g$ has a configurable observation window $[a_g^L,\, a_g^R]$ representing the period during which events are ascertainable (e.g., $[40, 80]$ for the oldest cohort, $[0, 45]$ for the youngest). An individual with raw onset $t_i$ is left-censored if $t_i < a_g^L$ (onset before observation began), right-censored if $t_i > a_g^R$ (onset after follow-up ended), and the observed time is clipped to the window:

$$
t_{\text{obs},i} = \text{clip}(t_i,\; a_g^L,\; a_g^R)
$$

Generations that should contribute family structure but no observed cases are assigned a zero-width window (e.g. $[80, 80]$), which fully censors every individual because no continuous onset time can equal the boundary exactly. With default settings the oldest generations (gen 0–2) use this convention, while the youngest generation (gen 5) has window $[0, 45]$, so individuals in that generation will only be marked as affected if their age-of-onset is before age 45.

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

### Mixture cure frailty model

Many diseases affect only a fraction of the population — the remainder are "cured" (or never susceptible). The mixture cure frailty model (Berkson & Gage, 1952; Farewell, 1982) separates *who* develops the disease from *when* among those who do:

**Susceptibility (WHO).** A liability threshold determines case status. The liability is standardised to zero mean and unit variance. Given a population prevalence $K$ (scalar, per-generation, or per-sex$\times$generation), the threshold is $\Phi^{-1}(1 - K)$ and individuals with liability exceeding the threshold are classified as susceptible:

$$
\text{susceptible}_i = \mathbf{1}\!\left[\tilde{L}_i > \Phi^{-1}(1 - K)\right]
$$

The cure fraction is $1 - K$. Non-susceptible individuals ("cured") are assigned a sentinel event time of $10^6$, ensuring they are right-censored under any realistic observation window.

**Age-at-onset (WHEN).** Among susceptible individuals only, age-at-onset is generated via the proportional-hazards frailty model described above (any of the six baseline hazards may be used):

$$
t_i = H_0^{-1}\!\left(\frac{-\log U_i}{z_i}\right), \quad z_i = \exp(\beta\,\tilde{L}_i)
$$

The key difference from the standard frailty model is that the frailty mechanism operates only on the susceptible subpopulation, while susceptibility itself is determined by the same underlying liability via the threshold.

### ADuLT phenotype models

The Age-Dependent Liability Threshold (ADuLT) models (Pedersen et al., *Nature Communications*, 2023) combine a liability threshold for case/control classification with a cumulative incidence proportion (CIP) mapping for age-at-onset assignment.

**ADuLT-LTM (liability threshold model).** Case status is determined by the liability threshold as in the simple threshold model. Among cases, onset age is assigned via a logistic CIP function. The effective liability is computed on the probit scale:

$$
L_{\text{eff},i} = \beta \, \tilde{L}_i + \beta_{\text{sex}} \cdot \text{sex}_i
$$

The cumulative incidence rate (CIR) for each case is:

$$
\text{CIR}_i = \Phi(-L_{\text{eff},i})
$$

which is then clipped to $[\epsilon,\; K - \epsilon]$ (where $K$ is the prevalence) and mapped to onset age via the inverse logistic CIP function:

$$
t_i = x_0 + \frac{1}{k} \log\!\left(\frac{\text{CIR}_i}{K - \text{CIR}_i}\right)
$$

where $x_0$ is the midpoint age and $k$ is the growth rate of the logistic CIP curve. Higher liability (more positive $L_{\text{eff}}$) produces smaller $\text{CIR}$ values (since $\Phi(-L)$ decreases) and therefore earlier onset. Controls (below the threshold) are assigned $t = 10^6$.

**ADuLT-Cox (proportional hazards model).** A Weibull(shape=2) proportional hazards model generates raw event times, which are then rank-mapped to onset ages via the CIP function:

$$
\tilde{t}_i = \sqrt{\frac{-\log U_i}{\exp(\beta \, \tilde{L}_i) \cdot \exp(\beta_{\text{sex}} \cdot \text{sex}_i)}}, \quad U_i \sim \text{Uniform}(0, 1]
$$

Individuals are sorted by $\tilde{t}_i$ and assigned a running CIP: $\text{CIP}_i = \text{rank}_i / (n + 1)$. Cases are those with $\text{CIP}_i < K$, and their onset age is:

$$
t_i = x_0 + \frac{1}{k} \log\!\left(\frac{\text{CIP}_i}{K - \text{CIP}_i}\right)
$$

Controls ($\text{CIP}_i \geq K$) receive $t = 10^6$. When prevalence varies by group (sex, generation, or sex$\times$generation), ranking and CIP assignment are performed within each group separately to ensure exact per-group case rates.

### Sex-specific effects

All phenotype models support sex-specific modification of the hazard or threshold via two mechanisms:

**Sex-specific hazard coefficient ($\beta_{\text{sex}}$).** An additional coefficient modifies the hazard or mapping as a function of the binary sex covariate ($\text{sex} \in \{0, 1\}$, where 0 = female and 1 = male).

In the frailty and cure frailty models, $\beta_{\text{sex}}$ enters the frailty multiplicatively:

$$
z_i = \exp\!\bigl(\beta \, \tilde{L}_i + \beta_{\text{sex}} \cdot \text{sex}_i\bigr)
$$

so that males (when $\beta_{\text{sex}} > 0$) experience uniformly higher hazard — and therefore earlier onset on average — than females at the same liability.

In the ADuLT-LTM model, $\beta_{\text{sex}}$ enters the probit-scale effective liability $L_{\text{eff}} = \beta \, \tilde{L} + \beta_{\text{sex}} \cdot \text{sex}$, shifting the CIR mapping. In the ADuLT-Cox model, $\beta_{\text{sex}}$ modifies the raw Weibull event time divisor as $\exp(\beta_{\text{sex}} \cdot \text{sex})$.

**Sex-specific prevalence.** For models that use a prevalence parameter (cure frailty, ADuLT, and the simple threshold model), prevalence can be specified per sex or per sex$\times$generation:

$$
K_i = \begin{cases} K_{\text{female}} & \text{if } \text{sex}_i = 0 \\ K_{\text{male}} & \text{if } \text{sex}_i = 1 \end{cases}
$$

where each sex-specific value may itself be a per-generation dictionary. This produces sex-differentiated thresholds and case rates, reflecting the sex differences in disease prevalence observed in many real conditions.

## Subsampling and Case Ascertainment

After phenotyping and censoring, an optional subsampling step draws a study sample of $N_{\text{sample}}$ individuals from the full population. Two modes are supported:

**Uniform subsampling.** When the case-ascertainment ratio is 1.0 (default), $N_{\text{sample}}$ individuals are drawn uniformly at random without replacement.

**Case-ascertainment bias.** When the ratio $\alpha \neq 1$, cases (individuals with $\delta_1 = 1$) are sampled with weight $\alpha$ relative to controls (weight 1). The sampling probability for individual $i$ is:

$$
p_i = \frac{w_i}{\sum_j w_j}, \quad w_i = \begin{cases} \alpha & \text{if } \delta_{i,1} = 1 \\ 1 & \text{otherwise} \end{cases}
$$

With $\alpha > 1$, cases are overrepresented in the sample (enrichment), mimicking case-control study designs. With $\alpha < 1$, cases are underrepresented. With $\alpha = 0$, only controls are sampled.

## Validation via Statistical Analysis

### Relationship pair extraction

Before computing correlations, individuals must be grouped into pairs by relationship type — the correlation structure within each group is what identifies genetic and environmental effects.

Relationship pairs are extracted using sparse matrix algebra over parent→child adjacency matrices. Let $\mathbf{A}_m$ be the $n \times n$ CSR matrix where entry $(i, j) = 1$ if individual $j$ is the mother of individual $i$, and $\mathbf{A}_f$ the analogous matrix for fathers. The combined parent matrix is $\mathbf{A} = \mathbf{A}_m + \mathbf{A}_f$.

The following matrix products identify relationship categories:

**Siblings.** All pairs sharing at least one parent are found via the shared-parent matrix $\mathbf{S} = \mathbf{A} \mathbf{A}^\top$, where $S_{ij} > 0$ indicates individuals $i$ and $j$ share at least one parent. Pairs are then classified by checking original (pre-remapped) parental identifiers: full siblings share both mother and father; maternal half-siblings share a mother but not a father; paternal half-siblings share a father but not a mother. Twin pairs are excluded from sibling counts.

**Parent-offspring.** Mother-offspring and father-offspring pairs are read directly from the nonzero entries of $\mathbf{A}_m$ and $\mathbf{A}_f$, respectively.

**Grandparent-grandchild.** The two-hop reach matrix $\mathbf{A}^2 = \mathbf{A} \cdot \mathbf{A}$ connects individuals to their grandparents. Nonzero entries in $\mathbf{A}^2$ where the generation difference is 2 identify grandparent-grandchild pairs.

**Avuncular.** Aunt/uncle–niece/nephew pairs are identified by combining the full-sibling matrix $\mathbf{F}$ with the parent matrix: if individual $i$'s parent is a full sibling of individual $j$, then $(i, j)$ is an avuncular pair. This is computed as $\mathbf{A} \cdot \mathbf{F}$.

**First cousins.** The shared-grandparent matrix $\mathbf{A}^2 (\mathbf{A}^2)^\top$ identifies pairs sharing at least one grandparent. First cousins are those who share a grandparent but are not siblings, not parent-offspring, not avuncular, and belong to the same generation.

**Second cousins.** Analogously, $\mathbf{A}^3 (\mathbf{A}^3)^\top$ identifies pairs sharing at least one great-grandparent. Second cousins are those who share a great-grandparent but do not fall into any closer relationship category.

**MZ twins.** Twin pairs are identified directly from the twin-partner column in the pedigree, deduplicated so that each pair appears once.

In total, ten relationship categories are extracted: MZ twins, full siblings, maternal half-siblings, paternal half-siblings, mother-offspring, father-offspring, avuncular, grandparent-grandchild, first cousins, and second cousins. When a sample mask is provided (e.g., after subsampling), only pairs where both individuals are in the sample are returned.

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

**Half-sibling statistics.** The observed proportion of maternal half-sibling pairs among all maternal sibling pairs is compared to the expected proportion given the mating structure.

**Variance-component recovery.** Founder-generation variances of $A$, $C$, and $E$ are compared to configured parameters; cross-trait correlations $r_A$ and $r_C$ are verified; $E$ components are confirmed to be uncorrelated across siblings and across traits; and $C$ values are confirmed to be identical within households.

**Population-level checks.** Generation sizes equal $N$; the number of recorded generations equals $G_{\text{ped}}$; the mean family size matches the configured mating rate. Tolerances are set as $\max(4 \cdot \text{SE}, \epsilon_{\min})$ where SE is the sampling standard error of the relevant statistic.

## Implementation

The simulation framework is implemented in Python as an installable package (`simace`) with NumPy for vectorised array operations, SciPy for optimisation and special functions, pandas and PyArrow for data management, and Numba for JIT-compiled numerical kernels (phenotype inversion, Metropolis sweeps, and pairwise survival likelihood evaluation). Relationship extraction uses SciPy sparse CSR matrices for $O(\text{nnz})$ computation of sibling, cousin, and higher-order relationship pairs via matrix products. The workflow is managed by Snakemake with per-scenario configuration, per-replicate seed offsets for reproducibility (seed incremented by replicate number), and support for SLURM-based HPC execution. All random number generation uses NumPy's PCG64 generator (`numpy.random.default_rng`) with deterministic seed management.

## Assumptions and Limitations

Every simulation makes simplifying assumptions. The following are the most consequential for interpreting results and should be considered when evaluating the framework:

**No gene-environment interaction.** Liability is strictly additive ($L = A + C + E$) with no multiplicative or interactive terms. If the true data-generating process involves gene-environment interaction ($G \times E$) or gene-environment correlation ($rGE$), the ACE decomposition will absorb these effects into the additive components, potentially biasing variance estimates.

**No cross-trait unique-environment correlation ($r_E = 0$).** Unique-environment components $E_1$ and $E_2$ are drawn independently across traits. This precludes modelling trait pairs where individual-specific environmental exposures (e.g., shared lifestyle factors) induce correlation beyond what is captured by $A$ and $C$.

**No environmental transmission across generations.** The common-environment component $C$ is drawn freshly per household each generation with no autoregressive transmission from parental $C$ values. This is the standard ACE assumption but does not capture intergenerational persistence of socioeconomic status, neighbourhood effects, or cultural transmission that may inflate parent-offspring resemblance in real populations.

**Fixed population size.** Each generation contains exactly $N$ individuals with no population growth, decline, bottlenecks, or migration. Demographic dynamics that alter effective population size or introduce population stratification are not modelled.

**Tetrachoric correlation bias under censoring.** When affection status is derived from the survival frailty model with age-window and death censoring, the observed prevalence may differ from the uncensored prevalence. Tetrachoric correlations estimated from censored binary outcomes are attenuated relative to the true underlying liability correlation. The pairwise Weibull survival correlation method, which explicitly accounts for censoring in the likelihood, is preferred for validation of censored phenotypes.

## Data types and memory efficiency

The pedigree uses narrowed data types to reduce memory at large population sizes:

- **int32** for person identifiers (id, mother, father, twin, household_id) — supports up to $2.1 \times 10^9$ individuals per pedigree. An overflow guard validates $N \times G_\text{ped} < 2^{31}$ at simulation start.
- **int32** for generation (consistent with ID columns).
- **int8** for sex (0/1).
- **float32** for variance components ($A_1, C_1, E_1, A_2, C_2, E_2$) — approximately 7 significant digits, sufficient for stochastic draws from unit-variance distributions.
- **float64** for liabilities ($L_1, L_2$) — full 15-digit precision, used by all downstream phenotype models.

Composite key computations (e.g., encoding a $(i, j)$ pair as $i \times \text{max\_id} + j$ for duplicate detection or set subtraction) explicitly cast to int64 before multiplication because $\text{max\_id}^2$ overflows int32.
