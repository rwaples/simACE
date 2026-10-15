# ACE model

## Liability decomposition

The liability of individual $i$ on trait $k$ is the sum of three components:

$$L_i^{(k)} = A_i^{(k)} + C_i^{(k)} + E_i^{(k)}$$

The three variances sum to one ($A + C + E = 1$), so each is a share of the total.

- **A**, additive genetic. Inherited under the infinitesimal model.
- **C**, common environment. Shared by all offspring of the same mother, and optionally passed in part from parents to offspring.
- **E**, unique environment. Drawn independently for each individual.

## Inheritance of A

Each offspring receives the midparent value plus Mendelian sampling noise:

$$A_{\text{offspring}} = \frac{A_{\text{mother}} + A_{\text{father}}}{2} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_A^2 / 2)$$

Founders draw the two traits' additive values jointly from a bivariate normal with cross-trait genetic correlation $r_A$.

## Common environment (C)

Every offspring of the same mother shares one $C$ draw. The simulation calls that group a household. Each household draws its own $C$.

By default parents do not pass $C$ to their children. With C transmission, each offspring also inherits a share of each parent's $C$:

$$C_{\text{offspring}} = b_m C_{\text{mother}} + b_f C_{\text{father}} + u_{\text{household}}$$

$b_m$ and $b_f$ are the maternal and paternal transmission coefficients, set per trait under `pedigree.trait{t}.C_transmission.{maternal,paternal}`. Both default to 0, each lies in $[0, 1]$, and $b_m^2 + b_f^2 \le 1$. The configured $C$ stays the total variance in expectation. The household draw $u$ supplies whatever transmission does not. Under transmission, parent-offspring pairs and paternal half-siblings share part of $C$, and maternal half-siblings with different fathers share less than all of it when $b_f > 0$. See [ADR 0025](../adr/0025-vertical-c-transmission.md).

## Unique environment (E)

Each individual draws $E$ independently for each trait. $E$ adds no familial correlation.

## Cross-trait correlations

Two traits can be correlated through each component:

| Parameter | Meaning |
|---|---|
| $r_A$ | Cross-trait genetic correlation |
| $r_C$ | Cross-trait common environment correlation |
| $r_E$ | Cross-trait unique environment correlation. Config key `rE`, default 0 |

## Standardisation

The `standardize` config key sets how the phenotype stage normalises liability
before a threshold or hazard step. It accepts three values:

| Mode | Behaviour |
|---|---|
| `none` | Raw liability is used. For threshold models, realised prevalence can drift when the cohort distribution differs from the N(0,1) reference. |
| `global` (default) | The liability is z-scored once across the whole phenotyped cohort: $L_z = (L - \bar L) / \mathrm{sd}(L)$. Per-generation prevalence still drifts when variance changes between generations. |
| `per_generation` | The liability is z-scored within each generation, removing shifts in the generation mean and variance. The observed case fraction can still differ from the target in a finite sample. |

Threshold models use the scaled liability to select cases. Hazard models use
it to determine event-time behavior. Some models use both steps; see
[Phenotype models, Standardization](../user-guide/phenotype-models.md#standardization)
for the setting each model reads.
