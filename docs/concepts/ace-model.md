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

The `standardize` config key sets how the phenotype stage normalises liability before it applies a threshold. It accepts three values:

| Mode | Behaviour |
|---|---|
| `none` | The raw liability is compared to the N(0,1)-scale threshold. Realised prevalence drifts whenever the cohort variance differs from 1. |
| `global` (default) | The liability is z-scored once across the whole phenotyped cohort: $L_z = (L - \bar L) / \mathrm{sd}(L)$. Per-generation prevalence still drifts when variance changes between generations. |
| `per_generation` | The liability is z-scored within each generation. Each generation hits its target prevalence exactly, however $\mathrm{Var}(C)$ or $\mathrm{Var}(E)$ drifts across cohorts. |

Config loading still accepts the legacy booleans. `true` becomes `global` and `false` becomes `none`, so older scenario files keep working.

### Per-trait hazard override

Models with a hazard step accept a per-trait key, `standardize_hazard`, inside `phenotype.trait{N}.params`. Those models are `frailty`, `cure_frailty`, `first_passage`, and `adult` with `method: cox`.

```yaml
phenotype:
  trait1:
    model: cure_frailty
    params:
      distribution: weibull
      scale: 2160
      rho: 0.8
      prevalence: 0.10
      standardize_hazard: per_generation   # overrides the global key for the hazard step
```

`standardize_hazard` accepts the same three modes and defaults to the value of the global `standardize`. Models with no hazard step reject the key with a trait-prefixed error. Those models are `simple_ltm` and `adult` with `method: ltm`.

`cure_frailty` is the only model that reads both keys. `standardize` sets the threshold step, which decides case status. `standardize_hazard` sets the hazard step, which decides age at onset among cases. Setting `standardize: per_generation` with `standardize_hazard: global` holds per-generation prevalence exact while keeping one hazard slope across generations.

### Per-model routing

| Model | Threshold step reads | Hazard step reads |
|---|---|---|
| `simple_ltm` | `standardize` | none |
| `adult.ltm` | `standardize` | none |
| `adult.cox` | none | `standardize_hazard`, default `standardize` |
| `frailty` | none | `standardize_hazard`, default `standardize` |
| `first_passage` | none | `standardize_hazard`, default `standardize` |
| `cure_frailty` | `standardize` | `standardize_hazard`, default `standardize` |

Switching `params.method` on the `adult` model between `ltm` and `cox` changes which key scales the liability for that trait. `adult.ltm` applies a threshold to the liability, so it reads `standardize`. `adult.cox` applies a hazard to the liability, so it reads `standardize_hazard`. If you set `standardize_hazard` on an `adult.ltm` trait, config validation raises an error that names this rule.
