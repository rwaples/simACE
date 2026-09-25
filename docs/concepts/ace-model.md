# ACE model

## Liability decomposition

The liability of individual $i$ on trait $k$ is the sum of three components:

$$L_i^{(k)} = A_i^{(k)} + C_i^{(k)} + E_i^{(k)}$$

The three variances sum to one ($A + C + E = 1$), so each is a share of the total.

- **A**, additive genetic. Inherited under the infinitesimal model.
- **C**, common environment. Shared by all offspring of the same mother.
- **E**, unique environment. Drawn independently for each individual.

## Inheritance of A

Each offspring receives the midparent value plus Mendelian sampling noise:

$$A_{\text{offspring}} = \frac{A_{\text{mother}} + A_{\text{father}}}{2} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_A^2 / 2)$$

Founders draw the two traits' additive values jointly from a bivariate normal with cross-trait genetic correlation $r_A$.

## Common environment (C)

Every offspring of the same mother shares one $C$ draw. The simulation calls that group a household. Parents do not pass $C$ to their children. Each household draws its own $C$.

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
