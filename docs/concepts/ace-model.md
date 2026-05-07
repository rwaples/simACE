# ACE Model

## Liability decomposition

Total liability for individual $i$ on trait $k$ is:

$$L_i^{(k)} = A_i^{(k)} + C_i^{(k)} + E_i^{(k)}$$

The three components sum to unit variance ($A + C + E = 1$):

- **A** (Additive genetic) -- heritable component following the infinitesimal model
- **C** (Common/shared environment) -- shared by offspring of the same mother
- **E** (Unique/personal environment) -- independent per individual

## Inheritance of A

Additive genetic values are transmitted from parents to offspring via midparent averaging
plus Mendelian sampling noise:

$$A_{\text{offspring}} = \frac{A_{\text{mother}} + A_{\text{father}}}{2} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_A^2 / 2)$$

For the founder generation, additive values for both traits are drawn jointly from a
bivariate normal with cross-trait genetic correlation $r_A$.

## Common environment (C)

$C$ is shared by all offspring of the same mother (household effect). It is **not**
inherited -- there is no parent-to-child $C$ transmission. Each mother's household
draws $C$ independently.

## Unique environment (E)

$E$ is drawn independently per individual per trait. By design, it contributes no
familial correlation.

## Cross-trait correlations

Two traits can be correlated through their components:

| Parameter | Meaning |
|---|---|
| $r_A$ | Cross-trait genetic correlation |
| $r_C$ | Cross-trait common environment correlation |
| $r_E$ | Cross-trait unique environment correlation (fixed at 0) |

## Standardisation

The `standardize` config flag controls how liability is normalised before
phenotyping. It accepts three values:

| Mode | Behaviour |
|---|---|
| `none` | Raw liability is compared to the N(0,1)-scale threshold. Realised prevalence drifts whenever the cohort variance differs from 1. |
| `global` (default) | Liability is z-scored once across the whole phenotyped cohort: $L_z = (L - \bar L) / \mathrm{sd}(L)$. Per-generation prevalence still drifts when variance changes generation-to-generation. |
| `per_generation` | Liability is z-scored within each generation independently. Each generation hits its target prevalence exactly, regardless of how $\mathrm{Var}(C)$ or $\mathrm{Var}(E)$ drifts across cohorts. |

Legacy boolean values are accepted at config load (`true → "global"`,
`false → "none"`) so older scenario files continue to work unchanged.

### Per-trait hazard override

The four hazard-bearing model families (`frailty`, `cure_frailty`,
`first_passage`, and `adult` with `method: cox`) accept a per-trait
override `standardize_hazard` inside `phenotype.trait{N}.params`:

```yaml
phenotype:
  trait1:
    model: cure_frailty
    params:
      distribution: weibull
      scale: 2160
      rho: 0.8
      prevalence: 0.10
      standardize_hazard: per_generation   # overrides global flag for the hazard step
```

`standardize_hazard` accepts the same three modes (`none`, `global`,
`per_generation`) and defaults to inheriting whatever was selected for
the global `standardize`. Models that have no hazard step (`threshold`
and `adult` with `method: ltm`) reject the field with a trait-prefixed
error.

`cure_frailty` is the only family that honors **both** knobs
independently: `standardize` sets the threshold step (case status) while
`standardize_hazard` sets the hazard step (case-onset distribution). A
mixed setting like `standardize: per_generation` together with
`standardize_hazard: global` preserves per-generation prevalence while
keeping the hazard slope constant across generations.

### Per-model routing

| Model | Threshold step uses | Hazard step uses |
|---|---|---|
| `threshold` | `standardize` | — |
| `adult.ltm` | `standardize` | — |
| `adult.cox` | — | `standardize_hazard` (default = `standardize`) |
| `frailty` | — | `standardize_hazard` (default = `standardize`) |
| `first_passage` | — | `standardize_hazard` (default = `standardize`) |
| `cure_frailty` | `standardize` | `standardize_hazard` (default = `standardize`) |

Note that toggling `params.method` between `"ltm"` and `"cox"` on the
`adult` family silently changes which knob controls L scaling for that
trait — LTM honors the global `standardize`, Cox honors
`standardize_hazard`. This asymmetry reflects the underlying math (LTM
is a threshold-on-L, Cox is a hazard-on-L). Setting
`standardize_hazard` on an `adult.ltm` model raises a validation error
that points back at this rule.
