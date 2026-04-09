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

When `standardize: true`, liability is normalised before phenotyping.

For frailty and cure-frailty models, standardisation uses the **global** mean and
standard deviation across all generations pooled. This means per-generation prevalence
will vary when variance components change across generations.

The simple liability-threshold model standardises **per-generation**, preserving exact
prevalence within each generation.
