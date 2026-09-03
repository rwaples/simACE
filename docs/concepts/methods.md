# Methods

simACE generates multi-generational family data with known genetic and environmental parameters, assigns disease phenotypes through pluggable survival and threshold models, censors and subsamples the result, and checks that the known parameters can be recovered. Mating is random or assortative. Six baseline hazard distributions, a mixture cure model, and the two ADuLT models are available. Censoring covers per-generation observation windows and competing-risk death. A unified ascertainment stage combines random pedigree dropout with case-weighted subsampling. The result is a testbed for twin and family methods in which every estimate has a ground truth to compare against.

This page explains the maths behind each stage. It is written to be read away from the code. Stage-by-stage config keys are in the [user guide](../user-guide/configuration.md).

## Simulation overview

simACE is an individual-based simulation of an ACE variance-component model. Each individual's liability is the sum of an additive genetic component $A$, a common environment component $C$, and a unique environment component $E$.

The simulation builds a multi-generational pedigree containing full siblings, half siblings, and monozygotic (MZ) twins. Mating may be random or assortative on one or both trait liabilities.

A phenotype model turns liability into a binary phenotype. The choices are a proportional-hazards frailty model with six baseline hazards, a mixture cure frailty model, the two ADuLT models, and a simple liability-threshold model.

The censoring stage applies competing-risk death censoring and per-generation age-window censoring.

The code is Python. Snakemake runs the pipeline and supports named scenarios, replicate runs, validation checks, and plots.

**Key terms**

- **Liability**: an unobserved continuous score for an individual's underlying disease risk. Higher liability means higher risk.
- **Founders**: the first generation. Their genetic and environmental values are drawn from scratch rather than inherited.
- **Burn-in**: early generations that are simulated and discarded so that the recorded pedigree does not depend on the founder generation's starting conditions.
- **Frailty**: a multiplicative modifier on disease hazard derived from liability. It sets how fast risk accumulates with age.
- **Censoring**: the loss of a disease event from the record because the individual died first, or was outside the observation window.
- **Prevalence**: the proportion of a generation classified as affected.
- **Assortative mating**: non-random partner choice. Under positive assortment individuals with similar liability mate preferentially. Under negative assortment dissimilar individuals do. Positive assortment raises parent-offspring and sibling resemblance above what random mating produces.
- **Cure fraction**: the proportion of the population that never develops the disease, however long the follow-up. In the mixture cure model this is $1 - K$, where $K$ is the prevalence.
- **Cumulative incidence function (CIF)**: the probability that an individual has had the event by a given age. The ADuLT models use it to map liability rank to age at onset.

## Pipeline stages

```
+----------+  +-----------+  +--------+  +-----------+  +-----------+  +---------+
|1. Simu-  |->|2. Pheno-  |->|3. Cen- |->|4. Ascer-  |->|5. Statis- |->|6. Valid-|
|   late   |  |   type    |  |  sor   |  |  tainment |  |   tics    |  |  ation  |
|          |  |           |  |        |  |           |  |           |  |         |
|Build fam-|  |Convert to |  |Apply   |  |Drop, then |  |Estimate   |  |Compare  |
|ilies with|  |observable |  |age &   |  |draw study |  |corr. &    |  |to ground|
|known ACE |  |outcomes   |  |death   |  |sample     |  |heritab.   |  |truth    |
+----------+  +-----------+  +--------+  +-----------+  +-----------+  +---------+
```

1. **Simulate** builds a multi-generational population. Each individual has known $A$, $C$, and $E$ values and links to parents and siblings. Mating is random or assortative.
2. **Phenotype** turns liability into observable outcomes: an age at onset from a survival model, an affected status from a threshold model, or both from the mixture cure model. Sex-specific effects can enter the hazard or the threshold.
3. **Censor** applies the observation window and competing-risk death to the raw event times. Its output is the observed phenotype.
4. **Ascertainment** removes a random fraction of individuals from the pedigree, severing parent and twin links to them, then draws a study sample, optionally weighted toward cases.
5. **Statistics** estimates correlations between relatives and heritability from the observed phenotype alone.
6. **Validation** compares every estimate to the known parameters. It checks both the code and the estimators. Validation reads the full pedigree from before ascertainment.

## Pedigree simulation

### Founder generation

The simulation begins with a founder cohort of $N$ individuals at generation $g = 0$. The default is $N = 100{,}000$. Each founder receives random $A$, $C$, and $E$ values. Those are the ground-truth components that the analysis stages try to recover. Sex is $\text{sex}_i \sim \text{Bernoulli}(0.5)$, coded 0 for female and 1 for male.

**Single-trait ACE decomposition.** For one trait, each individual's liability is the sum of three independent components:

$$
L_i = A_i + C_i + E_i
$$

where $A_i \sim \mathcal{N}(0, \sigma^2_A)$, $C_i \sim \mathcal{N}(0, \sigma^2_C)$, and $E_i \sim \mathcal{N}(0, \sigma^2_E)$. Total variance is normalised to one, so $\sigma^2_A + \sigma^2_C + \sigma^2_E = 1$ and each component is a share of the total. With $A = 0.5$, $C = 0.2$, and $E = 0.3$, half the variation in liability is additive genetic, a fifth comes from the shared household, and the rest is individual noise.

**Two correlated traits.** The simulation models two traits jointly, $k = 1, 2$. The $A$ and $C$ components may be correlated across traits. Each founder draws them from bivariate normals:

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

where $r_A$ and $r_C$ are the cross-trait genetic and common environment correlations. The unique environment components $E_1$ and $E_2$ have cross-trait correlation $r_E$, which defaults to 0. Each founder's liability is $L_{i,k} = A_{i,k} + C_{i,k} + E_{i,k}$.

### Reproduction: mating and family structure

Population size stays at $N$ across generations. Each new generation has exactly $N$ offspring, produced by pairing males and females from the parent generation.

**Mating counts.** Each parent draws a mating count from a zero-truncated Poisson distribution, so every individual mates at least once:

$$
n_{\text{matings},i} \sim \text{ZTP}(\lambda), \quad \lambda = 0.5 \text{ (default)}
$$

The simulation draws the truncated Poisson by rejection. It redraws any zero until every count is at least 1. Males and females draw independently. The sex with more mating slots in total then loses slots at random until both sexes have $T = \min(\sum n_{\text{male}}, \sum n_{\text{female}})$.

**Partner pairing.** Each parent's mating count becomes that many slots. Under random mating, the default, the simulation shuffles the male slots and pairs them positionally with the female slots, giving $M = T$ couples. Under assortative mating it pairs them as described in the next section. If the same mother and father appear together twice, the simulation swaps one of the conflicting entries with a nearby partner.

**Offspring allocation.** A multinomial draw with equal probabilities spreads the $N$ offspring over the $M$ couples:

$$
(c_1, c_2, \ldots, c_M) \sim \text{Multinomial}(N,\; 1/M, \ldots, 1/M)
$$

Each couple gets a random number of children, and the counts sum to $N$.

**Household assignment.** All offspring of the same mother get the same household identifier and the same $C$ draw. Maternal half-siblings therefore share $C$. Paternal half-siblings do not.

### Assortative mating

When $\text{assort}_1 \neq 0$ or $\text{assort}_2 \neq 0$, the simulation pairs individuals with correlated liabilities preferentially. The target mate correlations are $r_1 = \text{assort}_1$ for trait 1 and $r_2 = \text{assort}_2$ for trait 2.

**Single-trait case.** When only one assortment parameter is nonzero, the simulation uses a bivariate Gaussian copula. It converts each parent's liability to a rank score. The effective target correlation is $r_{\text{eff}} = \min(\sqrt{r_1^2 + r_2^2},\; 1)$. For each mating slot it draws $(z_f, z_m) \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ with $\Sigma_{12} = r_{\text{eff}}$. It sorts females and males by the weighted rank score $|r_1| \cdot \text{rank}_{1} + |r_2| \cdot \text{rank}_{2}$, then pairs them in the rank order of the bivariate normal draws. The mate correlation of the resulting pairs approximates the target. For negative assortment the simulation reverses the relevant rank order before scoring.

**Both-traits case.** When both $r_1$ and $r_2$ are nonzero, the simulation targets the 4-variate Gaussian copula of Border et al. (2022, Science, equation 2). Let $\mathbf{R}_{mf}$ be the $2 \times 2$ target mate-correlation matrix:

$$
\mathbf{R}_{mf} = \begin{bmatrix} r_1 & c \\ c & r_2 \end{bmatrix}
$$

where $c = \rho_w \sqrt{|r_1 r_2|} \operatorname{sign}(r_1 r_2)$ is the cross-trait, cross-sex mate correlation implied by the within-person liability correlation $\rho_w$. You can instead give the full matrix through the `assort_matrix` parameter. Then $r_1 = R_{mf,11}$, $r_2 = R_{mf,22}$, and $c = R_{mf,12} = R_{mf,21}$. The matrix must be symmetric.

Let $\mathbf{R}_{ff}$ be the within-female cross-trait liability correlation matrix:

$$
\mathbf{R}_{ff} = \begin{bmatrix} 1 & \rho_w \\ \rho_w & 1 \end{bmatrix}
$$

The full 4-variate matrix $\boldsymbol{\Sigma}_4 = \bigl[\begin{smallmatrix} \mathbf{R}_{ff} & \mathbf{R}_{mf}^\top \\ \mathbf{R}_{mf} & \mathbf{R}_{ff} \end{smallmatrix}\bigr]$ must be positive semi-definite. Config validation checks that. The pairing then runs in two phases.

*Phase 1: conditional-expectation initialisation.* The simulation converts each parent's liability to a quantile-normal score. The matrix $\mathbf{B} = \mathbf{R}_{mf} \mathbf{R}_{ff}^{-1}$ maps female scores to expected male scores. It projects the female score vectors through $\mathbf{B}$ to get target male vectors. It then projects both the targets and the actual male scores onto the dominant right singular vector of $\mathbf{R}_{mf}$ and rank-matches males to females along that line. That gives the starting permutation.

*Phase 2: greedy Metropolis refinement.* The simulation proposes random pairs of male positions $(i, j)$ for swapping. It accepts a swap if the swap reduces the total squared error over the four elements of $\mathbf{R}_{mf}$:

$$
\sum_{k \in \{1,\, 2,\, 12,\, 21\}} (S_k + \Delta_k - T_k)^2 < \sum_{k \in \{1,\, 2,\, 12,\, 21\}} (S_k - T_k)^2
$$

where $S_1, S_2$ are the same-trait running cross-product sums, $S_{12} = \sum_m z_{f,1}^{(m)} z_{m,2}^{(m)}$ and $S_{21} = \sum_m z_{f,2}^{(m)} z_{m,1}^{(m)}$ are the cross-trait sums, $T_k = r_k \cdot M$ for the same-trait targets, and $T_{12} = T_{21} = c \cdot M$ for the cross-trait targets. Refinement stops when every per-element correlation error is below $5 \times 10^{-4}$ or after $8M$ proposals. Both constants are set in `simace/simulation/simulate.py`.

### Monozygotic twins

After offspring are allocated, every couple with at least two offspring is eligible for a twin pair. A Bernoulli trial with probability $p_{\text{MZ}}$, default 0.02, decides whether the couple gets one. If so, the couple's first two offspring become MZ twins. They share the same parents, the same $A$ and $C$ values, and the same sex. A couple gets at most one MZ pair.

### Offspring inheritance

Each offspring inherits from both parents but grows up in a new household. Full siblings share half their additive genetic variance and all of their household environment.

**Additive genetic ($A$).** Under the infinitesimal model (Bulmer, 1971), each offspring's breeding value is the midparent value plus Mendelian sampling noise. The segregation variance is half the additive genetic variance:

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

The Mendelian sampling term is the random half of each parent's genome that this child received. For MZ twins the simulation copies the first twin's breeding values and sex to the second.

**Common environment ($C$).** Each household draws one bivariate $\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_C)$ value and gives it to every child in the household. Parents do not pass $C$ to children. $C$ is the environment the offspring are reared in. Siblings share it because they grow up together. Their parents' own childhood environment does not carry over.

**Unique environment ($E$).** Each child draws $E_{i,k} \sim \mathcal{N}(0, \sigma^2_{E_k})$ independently of siblings, twins, and the other trait. Even MZ twins differ in $E$. It stands for every residual source of liability.

### Burn-in and recording

See [Simulation design, multi-generational pedigree](simulation-design.md#multi-generational-pedigree) for how $G_{\text{sim}}$, $G_{\text{ped}}$, and $G_{\text{pheno}}$ split the run into burn-in, recorded, and phenotyped generations.

The recorded pedigree has $N \times G_{\text{ped}}$ individuals with contiguous identifiers. Each row carries the generation number, the parent identifiers, the MZ twin partner if any, the household identifier, all six variance components, and both trait liabilities.

## Phenotype models

The pedigree simulation produces continuous liabilities. Real studies observe a diagnosis age or an affected status. The models below turn liabilities into those observables. Every model that applies a threshold first standardises the liability under the `standardize` mode. The default mode z-scores once across the whole phenotyped cohort. See [ACE model, standardisation](ace-model.md#standardisation) for the three modes and the per-trait hazard override.

### Proportional-hazards frailty model

This model, `frailty`, decides when disease occurs. Higher liability brings earlier onset on average. For each trait $k$, with liability $L$, the conditional hazard is:

$$
h(t \mid L) = h_0(t) \cdot \exp(\beta\,L)
$$

where $h_0(t)$ is the baseline hazard and $\beta$ scales the effect of liability on the log hazard. The cumulative hazard and survival functions are:

$$
H(t \mid L) = H_0(t) \cdot \exp(\beta\,L), \quad S(t \mid L) = \exp\!\left[-H_0(t) \cdot \exp(\beta\,L)\right]
$$

Event times come from inverse-CDF sampling. With frailty $z_i = \exp(\beta\,\tilde{L}_i)$ and $U_i \sim \text{Uniform}(0, 1]$, the event time is:

$$
t_i = H_0^{-1}\!\left(\frac{-\log U_i}{z_i}\right)
$$

Each person draws one uniform $U_i$. The frailty $z_i$ sets how fast that draw turns into onset. Higher frailty means earlier onset. Six baseline hazards are available. Each gives a different shape for how background risk changes with age:

| Model | Parameters | Baseline hazard $h_0(t)$ | Cumulative hazard $H_0(t)$ | Inverse $t = H_0^{-1}(x)$ |
|---|---|---|---|---|
| **Weibull** | scale $\lambda$, shape $\rho$ | $\frac{\rho}{\lambda}\left(\frac{t}{\lambda}\right)^{\rho-1}$ | $\left(\frac{t}{\lambda}\right)^{\!\rho}$ | $\lambda\, x^{1/\rho}$ |
| **Exponential** | rate $b$ | $b$ | $b\,t$ | $x / b$ |
| **Gompertz** | rate $b$, shape $\gamma$ | $b\,\exp(\gamma\,t)$ | $\frac{b}{\gamma}\bigl(\exp(\gamma\,t) - 1\bigr)$ | $\frac{1}{\gamma}\log\!\left(1 + \frac{x\,\gamma}{b}\right)$ |
| **Lognormal** | $\mu$, $\sigma$ | $\frac{\phi(z_t)}{\sigma\,t\,\bar{\Phi}(z_t)}$ | $-\log\bar{\Phi}(z_t)$ | $\exp\!\bigl(\mu + \sigma\,\Phi^{-1}(1 - e^{-x})\bigr)$ |
| **Loglogistic** | scale $\alpha$, shape $k$ | $\frac{(k/\alpha)(t/\alpha)^{k-1}}{1 + (t/\alpha)^k}$ | $\log\!\bigl(1 + (t/\alpha)^k\bigr)$ | $\alpha\,(e^{x} - 1)^{1/k}$ |
| **Gamma** | shape $k$, scale $\theta$ | $\frac{f_0(t)}{S_0(t)}$ | $-\log S_0(t)$ | $F_{\Gamma}^{-1}(1 - e^{-x};\, k,\, \theta)$ |

where $z_t = (\log t - \mu)/\sigma$, $\phi$ is the standard normal density, $\bar{\Phi} = 1 - \Phi$ is the normal survival function, $f_0$ and $S_0$ are the Gamma density and survival functions, and $F_{\Gamma}^{-1}$ is the Gamma quantile function.

The Weibull hazard decreases with age when $\rho < 1$, which suits early-onset disease. At $\rho = 1$ it is the exponential, with constant hazard. When $\rho > 1$ it increases, which suits late-onset disease. The Gompertz hazard rises exponentially, like age-related mortality. The lognormal and loglogistic hazards rise and then fall, which suits a disease with a peak incidence age. The Gamma hazard is a flexible alternative with similar behaviour.

The two traits use independent baseline parameters and independent uniform draws, so each trait can have its own hazard shape.

### Censoring

Not every disease event is observed. People die of other causes, are too young to have developed the disease, or were born after the study began. The censor stage models both losses.

**Age-window censoring.** Each phenotyped generation $g$ has an observation window $[a_g^L,\, a_g^R]$ during which events are observable. The oldest cohort might have $[40, 80]$ and the youngest $[0, 45]$. An individual with raw onset $t_i$ is left-censored if $t_i < a_g^L$, because onset came before observation began. The individual is right-censored if $t_i > a_g^R$, because onset came after follow-up ended. The observed time is clipped to the window:

$$
t_{\text{obs},i} = \text{clip}(t_i,\; a_g^L,\; a_g^R)
$$

A generation that should contribute family structure but no observed cases gets a zero-width window such as $[80, 80]$. No continuous onset time equals the boundary exactly, so every individual in that generation is censored. At the defaults, generations 0 to 2 use a zero-width window. Generation 5 has window $[0, 45]$, so an individual there is marked affected only if onset came before age 45.

**Competing-risk death censoring.** Each individual draws an age at death from a Weibull distribution with its own parameters $(\lambda_d, \rho_d)$:

$$
t_{\text{death},i} = \lambda_d \left(-\log U_i^{(d)}\right)^{1/\rho_d}, \quad U_i^{(d)} \sim \text{Uniform}(0, 1]
$$

If onset comes after death, $t_{\text{obs},i} > t_{\text{death},i}$, the individual is death-censored and the observed time becomes the death age. An individual is affected, $\delta_i = 1$, only if the event falls inside the age window and before death:

$$
\delta_i = \mathbf{1}[\text{not age-censored}] \;\cdot\; \mathbf{1}[\text{not death-censored}]
$$

### Simple liability-threshold model

The `simple_ltm` model classifies individuals as affected or unaffected by a liability cutoff. Use it when only case status matters and timing does not.

A prevalence $\pi_g$ for generation $g$ sets the threshold at the $(1 - \pi_g)$ quantile of the standardised liability. An individual whose standardised liability exceeds the threshold is affected:

$$
\delta_{i,k} = \mathbf{1}\!\left[\tilde{L}_{i,k} \geq \Phi^{-1}(1 - \pi_g)\right]
$$

where $\Phi^{-1}$ is the standard normal quantile function. At 10% prevalence the top 10% of the liability distribution is affected. Prevalence may be one scalar for every generation or a per-generation mapping. Cases then get an onset age from a small onset sub-model, either a fixed age or a normal draw, that is independent of liability. That onset passes through the censor stage like any other.

### Mixture cure frailty model

Many diseases affect only part of the population. The rest never become susceptible. The `cure_frailty` model (Berkson and Gage, 1952; Farewell, 1982) separates who develops the disease from when.

**Susceptibility.** A liability threshold decides case status. Given a prevalence $K$, which may be a scalar, per generation, or per sex and generation, the threshold is $\Phi^{-1}(1 - K)$. An individual whose standardised liability exceeds it is susceptible:

$$
\text{susceptible}_i = \mathbf{1}\!\left[\tilde{L}_i > \Phi^{-1}(1 - K)\right]
$$

The cure fraction is $1 - K$. Non-susceptible individuals get a sentinel event time of $10^6$, which every realistic observation window right-censors.

**Age at onset.** Among susceptible individuals only, the proportional-hazards frailty model above generates the onset age. Any of the six baseline hazards may be used:

$$
t_i = H_0^{-1}\!\left(\frac{-\log U_i}{z_i}\right), \quad z_i = \exp(\beta\,\tilde{L}_i)
$$

In the `frailty` model every individual eventually has an event. Here the frailty acts only on the susceptible subpopulation, and the same liability decides susceptibility through the threshold.

### ADuLT models

The Age-Dependent Liability Threshold (ADuLT) family has two models that share a logistic CIF age scale. `adult.ltm` applies a liability threshold for case status and a deterministic CIF mapping for onset age. `adult.cox` draws Weibull proportional-hazards raw times, ranks them, and caps cases at the target prevalence $K$. It then maps the ranks onto the same CIF age scale.

**`adult.ltm`.** The liability threshold decides case status as in `simple_ltm`. Among cases, a logistic CIF assigns the onset age. This mapping is deterministic. It is not a proportional-hazards model. The effective liability on the probit scale is:

$$
L_{\text{eff},i} = \beta \, \tilde{L}_i + \beta_{\text{sex}} \cdot \text{sex}_i
$$

Each case's cumulative incidence value is:

$$
c_i = \Phi(-L_{\text{eff},i})
$$

The model clips $c_i$ to $[\epsilon,\; K - \epsilon]$ and maps it to an onset age through the inverse logistic CIF:

$$
t_i = x_0 + \frac{1}{k} \log\!\left(\frac{c_i}{K - c_i}\right)
$$

where $x_0$ is the midpoint age and $k$ is the growth rate of the logistic curve. Higher liability gives a smaller $c_i$, because $\Phi(-L)$ decreases in $L$, and therefore earlier onset. Controls, below the threshold, get $t = 10^6$.

**`adult.cox`.** A Weibull proportional-hazards model with shape 2 generates raw event times:

$$
\tilde{t}_i = \sqrt{\frac{-\log U_i}{\exp(\beta \, \tilde{L}_i) \cdot \exp(\beta_{\text{sex}} \cdot \text{sex}_i)}}, \quad U_i \sim \text{Uniform}(0, 1]
$$

The model sorts individuals by $\tilde{t}_i$ and assigns a running cumulative incidence $c_i = \text{rank}_i / (n + 1)$. Cases are those with $c_i < K$. Their onset age is:

$$
t_i = x_0 + \frac{1}{k} \log\!\left(\frac{c_i}{K - c_i}\right)
$$

Controls, with $c_i \geq K$, get $t = 10^6$. The proportional-hazards interpretation applies to the raw time ordering. The raw survival law has $H_0(t)=t^2$ and relative hazard $\exp(\beta\,\tilde{L} + \beta_{\text{sex}}\cdot\text{sex})$. When prevalence varies by sex, generation, or both, the model ranks and assigns $c_i$ within each group, so each group hits its case rate exactly.

### Sex-specific effects

Every phenotype model accepts sex-specific effects through two mechanisms.

**Sex-specific hazard coefficient ($\beta_{\text{sex}}$).** A coefficient on the binary sex covariate, where $\text{sex} = 0$ is female and $\text{sex} = 1$ is male.

In the `frailty` and `cure_frailty` models it enters the frailty multiplicatively:

$$
z_i = \exp\!\bigl(\beta \, \tilde{L}_i + \beta_{\text{sex}} \cdot \text{sex}_i\bigr)
$$

When $\beta_{\text{sex}} > 0$, males have a uniformly higher hazard than females at the same liability, and therefore earlier onset on average.

In `adult.ltm` it enters the effective liability $L_{\text{eff}} = \beta \, \tilde{L} + \beta_{\text{sex}} \cdot \text{sex}$, which shifts the cumulative incidence mapping. In `adult.cox` it enters the raw Weibull time's divisor as $\exp(\beta_{\text{sex}} \cdot \text{sex})$.

**Sex-specific prevalence.** The models that take a prevalence, `cure_frailty`, both ADuLT models, and `simple_ltm`, accept it per sex or per sex and generation:

$$
K_i = \begin{cases} K_{\text{female}} & \text{if } \text{sex}_i = 0 \\ K_{\text{male}} & \text{if } \text{sex}_i = 1 \end{cases}
$$

Each sex-specific value may itself be a per-generation mapping. That gives sex-specific thresholds and case rates, as seen in many real conditions.

## Ascertainment

After phenotyping and censoring, the ascertainment stage reduces the full population to the analysis dataset. See [ADR 0001](../adr/0001-unified-ascertainment-stage.md). It applies two steps to IDs rather than to weights. Weights would cancel silently under a fixed-size weighted draw.

**Step 1: uniform dropout.** A dropout rate $d \in [0, 1)$ sets the fraction to remove. The stage deletes $n_{\text{drop}} = \mathrm{round}(N_{\text{total}} \cdot d)$ individuals uniformly at random, without regard to trait, sex, or generation. Any `mother`, `father`, or `twin` reference to a removed individual becomes $-1$.

**Step 2: case-weighted draw.** From the post-dropout pool the stage draws $N_{\text{sample}}$ individuals. When the case-ascertainment ratio $\alpha = 1$ the draw is uniform. Otherwise each individual's sampling probability is:

$$
p_i = \frac{w_i}{\sum_j w_j}, \quad w_i = \begin{cases} \alpha & \text{if } \delta_{i,1} = 1 \\ 1 & \text{otherwise} \end{cases}
$$

With $\alpha > 1$ the sample overrepresents cases, like a case-control design. With $\alpha < 1$ it underrepresents them. With $\alpha = 0$ it contains only controls.

The output pedigree is the ancestor closure of the sampled IDs within the post-dropout pedigree. Every parent reachable through an unbroken chain of links is kept. The stage then rewrites any remaining dangling twin reference to $-1$, so kinship and relationship-pair extraction work on the analysis dataset.

Validation is unaffected. It reads `pedigree.full.parquet`, the pedigree from before ascertainment.

## Validation via statistical analysis

### Relationship pair extraction

Before computing correlations, the stats stage groups individuals into pairs by relationship type. The correlation within each group is what identifies genetic and environmental effects.

Pair extraction lives in the external [`pedigree-graph`](https://github.com/rwaples/pedigree-graph) package. `PedigreeGraph` reads the mother and father columns and classifies sibling pairs directly from them. Full siblings share both a known mother and a known father. Maternal half-siblings share a known mother and are not full siblings. Paternal half-siblings share a known father and are not full siblings. Twin pairs come from the twin column and are excluded from the sibling counts. Parent-offspring pairs are the mother and father links themselves.

For every other category, `PedigreeGraph` builds a sparse parent adjacency matrix $\mathbf{A}$, with $A_{ij} = 1$ when $j$ is a parent of $i$, and takes products of it. Powers of $\mathbf{A}$ reach grandparents and great-grandparents. Products of the form $\mathbf{A}^k (\mathbf{A}^k)^\top$ find pairs that share an ancestor $k$ meioses up, and the multiplicity of each entry separates pairs that share two ancestors, a mated pair, from pairs that share one. Products with the full-sibling matrix give the avuncular categories. Each category is defined by the `(up, down, n_ancestors)` triple in the registry `REL_REGISTRY`, which holds 23 categories. The full table is in [Simulation design](simulation-design.md#pedigree-relationship-types). When the caller passes a sample mask, such as the post-ascertainment subset, extraction returns only pairs with both members in the sample.

### Tetrachoric correlation estimation

Binary phenotypes underestimate the association between relatives because they discard how far each person sits above or below the threshold. The tetrachoric correlation corrects for that by assuming a bivariate normal liability underneath.

For each relationship type the stats stage assumes that the observed dichotomy arises from $(L_1, L_2) \sim \text{BVN}(0, 0, 1, 1, r)$. With thresholds $t_k = \Phi^{-1}(1 - \hat{\pi}_k)$ from the observed prevalences, the four cell probabilities of the $2 \times 2$ table are:

$$
P_{11}(r) = P(L_1 > t_1,\; L_2 > t_2 \mid r)
$$

and likewise for $P_{10}, P_{01}, P_{00}$. The bivariate normal CDF is evaluated through Owen's $T$ function, in `_owens_t` in `simace.core._numba_utils`. The estimate $\hat{r}$ minimises the negative log-likelihood:

$$
-\ell(r) = -\sum_{(a,b) \in \{0,1\}^2} n_{ab} \log P_{ab}(r)
$$

over $r \in (-0.999, 0.999)$ by bounded scalar optimisation. The standard error comes from the observed Fisher information $I(\hat{r}) = n \cdot \phi_2(t_1, t_2; \hat{r})^2 / \prod_{(a,b)} P_{ab}(\hat{r})$, where $\phi_2$ is the bivariate normal density. The estimate answers one question. Given the concordance pattern among relatives, what liability correlation is most likely?

### Pairwise Weibull survival correlation estimation

When the phenotype is a censored onset time rather than a binary outcome, the tetrachoric estimate is biased. It ignores when events occurred and which observations were censored. The pairwise Weibull estimator uses the event times and censoring indicators directly. It lives in fitACE, in `fitace_frailty.weibull_mle`, and is described here because it is the reference estimator for censored phenotypes.

For a pair $(i, j)$ with liabilities $(L_i, L_j) \sim \text{BVN}(0, 0, 1, 1, r)$, the pairwise likelihood is:

$$
\mathcal{L}(r) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty}
g(t_i, \delta_i \mid L_i)\; g(t_j, \delta_j \mid L_j)\;
\phi_2(L_i, L_j;\, r)\; dL_i\, dL_j
$$

where $g(t, \delta \mid L) = h(t \mid L)^\delta \cdot S(t \mid L)$ is the individual Weibull contribution, the hazard for an event and the survival for a censored observation. The estimator evaluates the integral by two-dimensional Gauss-Hermite quadrature in the probabilist's convention with $n_q$ nodes per dimension. With the Cholesky substitution $L_i = x_m$ and $L_j = r\,x_m + \sqrt{1 - r^2}\,x_n$, the integral becomes:

$$
\mathcal{L}(r) \approx \sum_{m=1}^{n_q} \sum_{n=1}^{n_q}
w_m\, w_n\;
g(t_i, \delta_i \mid x_m)\;
g(t_j, \delta_j \mid r\,x_m + \sqrt{1-r^2}\,x_n)
$$

The integral is needed because liabilities are unobserved. The estimator integrates over every liability pair, weighted by the bivariate normal probability that the pair would produce the observed data.

The estimator applies log-sum-exp stabilisation per pair to avoid overflow. It minimises the total negative log-likelihood over all pairs for $r \in (-0.999, 0.999)$ by bounded scalar optimisation. The standard error comes from the numerical Hessian by a central second difference with step $h = 10^{-4}$:

$$
\hat{d}^2 = \frac{-\ell(\hat{r} + h) - 2(-\ell(\hat{r})) + (-\ell(\hat{r} - h))}{h^2}, \quad
\text{SE}(\hat{r}) = \frac{1}{\sqrt{\hat{d}^2}}
$$

Numba compiles the inner likelihood loop and parallelises it over pairs with `prange`. A NumPy fallback runs when Numba is absent.

### Heritability estimation

With a correlation for each relationship type, heritability follows from twin-study logic.

Falconer's formula estimates narrow-sense heritability from the liability correlations of MZ pairs and of full-sibling pairs, which stand in for DZ twins:

$$
\hat{h}^2 = 2(\hat{r}_{\text{MZ}} - \hat{r}_{\text{DZ}})
$$

The expected MZ correlation is $A + C$ and the expected full-sibling correlation is $\tfrac{1}{2}A + C$, so $\hat{h}^2 \approx A$. The extra resemblance of MZ pairs over full siblings comes from their extra genetic sharing, and twice that gap is the heritability. Parent-offspring regression gives a second estimate. The slope of offspring liability on midparent liability equals $A$ under the infinitesimal model.

## Validation checks

Every simulated individual has known parameters, so every output can be checked against expectation. Validation confirms both that the code works and that the estimators do.

The validate stage in `simace.analysis.validate` runs ten check families on each replicate:

- **Structural** (`validate_structural`). Identifiers are contiguous. Parent references are valid IDs or $-1$ for founders. Mothers are female and fathers are male. The sex ratio lies in $[0.45, 0.55]$.
- **Statistical** (`validate_statistical`). Founder variances of $A$, $C$, and $E$ match the configured values. Total variance is near 1. The cross-trait correlations $r_A$, $r_C$, and $r_E$ match. $C$ is identical within households. $E$ is uncorrelated between siblings.
- **Twins** (`validate_twins`). Twin pointers are symmetric. MZ pairs share parents, $A$ values within floating-point tolerance, and sex. Under the standard mating model the observed twin rate matches $p_{\text{MZ}}$.
- **Half-sibs** (`validate_half_sibs`). The half-sibling structure matches the mating-pair model.
- **Heritability** (`validate_heritability`). The heritability estimates for both traits recover the configured $A$.
- **Population** (`validate_population`). Each generation has $N$ individuals. The number of recorded generations is $G_{\text{ped}}$. Mean offspring per mother is near $N / n_{\text{females}}$.
- **Assortative mating** (`validate_assortative_mating`). The Pearson correlation of mother and father liability for each trait matches the `assort` target.
- **AM equilibrium** (`validate_am_equilibrium`). Under assortative mating, the final-generation $\mathrm{Var}(A)$ matches the infinitesimal-recursion prediction.
- **Consanguinity** (`validate_consanguineous_matings`). Consanguineous matings are detected, and the resulting shortfall in distinct grandparents is reconciled.
- **Effective size** (`validate_effective_size`). Each of the eight Ne estimators matches its expected value when the configuration has one.

Correlation checks use a tolerance of four standard errors with a floor of 0.05. See `_corr_tolerance` in `simace.analysis.validate._common`.

## Implementation

`simace` is an installable Python package. NumPy does the vectorised array work, SciPy the optimisation and special functions, Polars the DataFrames at every stage boundary (ADR 0015), and Numba the compiled kernels for phenotype inversion, Metropolis sweeps, and the tetrachoric likelihood. Relationship extraction uses SciPy sparse CSR matrices in the `pedigree-graph` package. Snakemake runs the workflow with per-scenario configuration and per-replicate seed offsets, the seed plus the replicate number. SLURM execution goes through `snakemake-executor-plugin-slurm`, pinned in `pixi.toml`. All random draws use NumPy's PCG64 generator through `numpy.random.default_rng` with explicit seeds.

## Assumptions and limitations

Each assumption below shapes what the results can say.

**No gene-environment interaction.** Liability is additive, $L = A + C + E$, with no interaction terms. If the real data-generating process has gene-environment interaction or gene-environment correlation, an ACE decomposition absorbs them into the additive components and biases the variance estimates.

**Cross-trait unique environment correlation defaults to zero.** $E_1$ and $E_2$ are independent across traits unless `rE` is set. The default scenarios therefore do not model trait pairs whose individual-specific exposures, such as shared lifestyle factors, correlate beyond what $A$ and $C$ capture.

**No environmental transmission across generations.** Each household draws $C$ afresh. Nothing carries over from the parents' $C$. That is the standard ACE assumption. It leaves out the persistence of socioeconomic status, neighbourhood, and culture that can inflate parent-offspring resemblance in real populations.

**Fixed population size.** Every generation has exactly $N$ individuals. There is no growth, decline, bottleneck, or migration, so nothing changes the effective population size or introduces stratification.

**Tetrachoric bias under censoring.** When affected status comes from a frailty model with age-window and death censoring, observed prevalence differs from uncensored prevalence, and tetrachoric correlations from the censored binary outcome are attenuated. For censored phenotypes prefer the pairwise Weibull estimator, whose likelihood accounts for censoring.

## Data types and memory

The pedigree uses narrow dtypes to keep large populations in memory:

- **int32** for person identifiers: `id`, `mother`, `father`, `twin`, and `household_id`. That allows up to $2.1 \times 10^9$ individuals. A guard at simulation start rejects $N \times G_\text{ped}$ above the int32 maximum.
- **int32** for `generation`, matching the ID columns.
- **int8** for `sex`.
- **float32** for the variance components $A_1, C_1, E_1, A_2, C_2, E_2$. About seven significant digits, enough for draws from unit-variance distributions.
- **float64** for the liabilities $L_1$ and $L_2$, which every phenotype model reads.

Code that encodes a pair $(i, j)$ as one key, $i \times \text{max\_id} + j$, for duplicate detection or set subtraction casts to int64 first, because $\text{max\_id}^2$ overflows int32.
