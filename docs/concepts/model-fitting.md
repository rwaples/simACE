# Model Fitting

The [`fitACE`](https://github.com/rwaples/fitACE) package handles
statistical model fitting on simulated data. This page provides
conceptual context; see the fitACE README for usage instructions.

## Phenotype models

Continuous liabilities are mapped to observable affected status and age-of-onset phenotypes via survival, cure, first-passage, and threshold models:

| Model | Description |
|---|---|
| **Frailty** | Proportional hazards with choice of baseline hazard (Weibull, Gompertz, lognormal, etc.). Liability scales hazard via $z = \exp(\beta L)$. Given sufficient time, every individual eventually becomes affected. |
| **Cure-Frailty** | Mixture model separating **who** gets the disease (susceptible vs. non-susceptible) from **when** (age-of-onset among susceptibles). Supports sex-specific prevalence. |
| **ADuLT LTM** | Deterministic liability threshold model with logistic cumulative incidence function (Pedersen et al., 2023); **not** proportional hazards. |
| **ADuLT Cox** | Weibull(shape=2) proportional hazards for raw event-time ordering, followed by rank-based CIF-to-age mapping (Pedersen et al., 2023). |
| **Simple LTM** | Liability threshold for case status at prevalence `K`, with a fixed or normally-distributed age-of-onset. Onset flows through the standard censor stage like the other models. |

## Censoring

Two censoring layers mimic real-world data limitations:

1. **Age-window censoring** -- per-generation observation intervals `[left, right]`.
   Events before `left` are left-truncated; events after `right` are right-censored.
2. **Competing-risk mortality** -- death age drawn from a Weibull distribution,
   independent of disease liability. Individuals who die before onset are death-censored.

The combined effect: only a fraction of true cases are observed as affected.

## Ascertainment

The pipeline can restrict the observed analysis dataset via a unified
**ascertainment** stage (see [ADR 0001](../adr/0001-unified-ascertainment-stage.md))
that combines three knobs in two explicit steps (dropout → case-weighted draw):

- **Dropout** (`dropout_rate`) -- uniform random removal of individuals from
  the pedigree; severs parent/twin references to removed IDs
- **Case ascertainment** (`case_ascertainment_ratio`) -- weights cases (vs
  controls) during the `N_sample` draw
- **Subsample size** (`N_sample`) -- target size of the post-ascertainment
  analysis dataset; pass-through when `0` or `>= post-dropout pool`

## Heritability estimation

The simulation validates several heritability estimation approaches:

- **Falconer's $h^2$** -- from tetrachoric correlations between MZ and DZ twins
- **Tetrachoric correlations** -- by relationship type and generation
- **Weibull frailty MLE** -- pairwise survival-time correlation via Gauss-Hermite quadrature
- **EPIMIGHT** -- external R package for heritability from family data
- **PA-FGRS** -- polygenic risk scores from family history
