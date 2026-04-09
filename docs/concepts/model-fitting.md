# Model Fitting

The `fit_ace` package handles statistical model fitting on simulated data.
This page gives conceptual context; see the
[fit_ace README](https://github.com/ryanwaples/ACE/blob/master/fit_ace/README.md) for usage instructions.

## Phenotype models

Continuous liabilities are mapped to age-of-onset phenotypes via time-to-event models:

| Model | Description |
|---|---|
| **Frailty** | Proportional hazards with choice of baseline hazard (Weibull, Gompertz, lognormal, etc.). Liability scales hazard via $z = \exp(\beta L)$. Given enough time, everyone eventually becomes affected. |
| **Cure-Frailty** | Mixture model separating **who** gets the disease (susceptible vs. non-susceptible) from **when** (age-of-onset among susceptibles). Supports sex-specific prevalence. |
| **ADuLT LTM** | Deterministic liability threshold model with logistic cumulative incidence proportion (Pedersen et al., 2023). |
| **ADuLT Cox** | Proportional hazards with Weibull noise and rank-based CIP-to-age mapping (Pedersen et al., 2023). |
| **Simple LTM** | Binary affected status from liability threshold, without age-of-onset or censoring. |

## Censoring

Two censoring layers mimic real-world data limitations:

1. **Age-window censoring** -- per-generation observation intervals `[left, right]`.
   Events before `left` are left-truncated; events after `right` are right-censored.
2. **Competing-risk mortality** -- death age drawn from a Weibull distribution,
   independent of disease liability. Individuals who die before onset are death-censored.

The combined effect: only a fraction of true cases are observed as affected.

## Subsampling and ascertainment

The pipeline can restrict observed data to study the impact on downstream estimates:

- **Subsampling** (`N_sample`) -- random subset of phenotyped individuals
- **Case ascertainment** (`case_ascertainment_ratio`) -- cases sampled at higher rate
- **Pedigree dropout** (`pedigree_dropout_rate`) -- random removal of individuals,
  breaking pedigree links and downgrading relationship types

## Heritability estimation

The simulation validates several heritability estimation approaches:

- **Falconer's $h^2$** -- from tetrachoric correlations between MZ and DZ twins
- **Tetrachoric correlations** -- by relationship type and generation
- **Weibull frailty MLE** -- pairwise survival-time correlation via Gauss-Hermite quadrature
- **EPIMIGHT** -- external R package for heritability from family data
- **PA-FGRS** -- polygenic risk scores from family history
