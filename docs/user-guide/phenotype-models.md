# Phenotype models

A phenotype model maps each individual's liability `L = A + C + E` to an
affected status and, for the time-to-event models, an age of onset. Each trait
selects its model under `phenotype.trait1` or `phenotype.trait2` in the
scenario. [Configuration](configuration.md#phenotype) shows where the
block sits.

## Model families

| Model | Case status | Age of onset | Proportional hazards | `beta` scales |
|---|---|---|---|---|
| `frailty` | Onset falls inside the observation window | Parametric event time with hazard multiplied by `exp(beta * L)`. Every individual has an onset given enough follow-up | Yes | The log hazard |
| `cure_frailty` | Liability above the threshold set by `prevalence`. Others have no onset | Same parametric event time, among cases only | Among cases | The log hazard |
| `adult`, `method: ltm` | Liability above the threshold set by `prevalence` | Deterministic inverse of the logistic cumulative incidence at the individual's liability rank | No | The probit mapping to age |
| `adult`, `method: cox` | Rank of a Weibull proportional-hazards event time, cut at `prevalence` | Rank mapped through the logistic cumulative incidence | For the event-time ordering | The log hazard of the raw event time |
| `first_passage` | First passage of a Brownian motion with drift inside the observation window | The first-passage time | No | The drift |
| `simple_ltm` | Liability above the threshold set by `prevalence` | Fixed age, or drawn from a normal distribution, independent of liability | No | Nothing. `beta` and `beta_sex` are accepted but unused |

## `frailty` and `cure_frailty` parameters

`params.distribution` names the baseline hazard. Each distribution has its own
parameters.

| Distribution | Parameters |
|---|---|
| `weibull` | `scale`, `rho` |
| `exponential` | `rate` |
| `gompertz` | `rate`, `gamma` |
| `lognormal` | `mu`, `sigma` |
| `loglogistic` | `scale`, `shape` |
| `gamma` | `shape`, `scale` |

`cure_frailty` also takes `params.prevalence`, which sets the susceptible
fraction. The registry is `BASELINE_HAZARDS` in `simace/phenotype/hazards.py`.

## `adult` parameters

| Parameter | Description |
|---|---|
| `method` | `ltm` or `cox` |
| `cip_x0`, `cip_k` | Midpoint and slope of the logistic cumulative incidence curve. Both methods share them |
| `prevalence` | Lifetime case fraction |

The two methods share the age scale but choose cases differently. `ltm` is a
threshold on liability and is not a proportional-hazards model. `cox` ranks
Weibull event times with noise and maps the ranks to ages.

## `first_passage` parameters

A latent process starts at a height set by liability and drifts toward zero.
Onset is the first time it reaches zero.

| Parameter | Description |
|---|---|
| `drift` | Drift of the latent process. Negative drift reaches zero in finite time, so every individual has an onset. Positive drift leaves a fraction with no onset |
| `shape` | Sets the starting height and so the time scale of first passage |

## `simple_ltm` parameters

`simple_ltm` sets case status by a probit threshold at prevalence `K`. It then
gives each case an onset age that does not depend on liability. Onset ages
pass through the censor stage like every other model, so the observed affected
fraction after censoring is below `K`.

| Parameter | Description |
|---|---|
| `prevalence` | Case fraction `K`, in any of the three forms below |
| `onset` | `{kind: fixed, age: A}` gives every case onset at age `A`. `{kind: normal, mean: M, sd: S}` draws each onset from `Normal(M, S)` |

```yaml
phenotype:
  trait1:
    model: simple_ltm
    params:
      prevalence: 0.10
      onset: {kind: fixed, age: 30}
  trait2:
    model: simple_ltm
    params:
      prevalence: 0.20
      onset: {kind: normal, mean: 35, sd: 8}
```

The censor stage marks a control with the onset age `1e6`. A `normal` draw
that reaches `1e6` is clipped onto that age, and the censor stage then counts
the case as a control. Draws stay below `1e6` when `mean` and `sd` are on a
human lifespan scale.

## Prevalence forms

`params.prevalence` is required for `adult`, `cure_frailty`, and `simple_ltm`.
It is rejected for `frailty` and `first_passage`, whose case fraction follows
from the event-time process. The loader also rejects `prevalence` outside
`params`.

`params.prevalence` takes one of three forms.

- A scalar such as `0.10`. Every individual has the same prevalence.
- A per-generation dict such as `{2: 0.03, 3: 0.05, 4: 0.08, 5: 0.12}`. Every
  phenotyped generation needs an entry.
- A sex-specific dict such as `{female: 0.08, male: 0.12}`. Each value may be
  a scalar or a per-generation dict.

## Standardization

The top-level `standardize` setting, one of `none`, `global`, or
`per_generation`, controls how liability is standardized before phenotyping.
Models with a separate hazard step, meaning `frailty`, `cure_frailty`,
`first_passage`, and `adult` with `method: cox`, also accept
`params.standardize_hazard`. It overrides `standardize` for the hazard step
only. When omitted, it inherits `standardize`.

`simple_ltm` and `adult` with `method: ltm` have no hazard step and reject
`standardize_hazard`.

| Model | Threshold step reads | Hazard step reads |
|---|---|---|
| `simple_ltm` | `standardize` | none |
| `adult.ltm` | `standardize` | none |
| `adult.cox` | none | `standardize_hazard`, default `standardize` |
| `frailty` | none | `standardize_hazard`, default `standardize` |
| `first_passage` | none | `standardize_hazard`, default `standardize` |
| `cure_frailty` | `standardize` | `standardize_hazard`, default `standardize` |

For `cure_frailty`, `standardize` scales liability for case selection and
`standardize_hazard` scales it for age at onset among cases. Switching
`adult.params.method` between `ltm` and `cox` changes which step uses the
liability.

[ACE model, Standardisation](../concepts/ace-model.md#standardisation)
explains the effect of each mode on liability and prevalence.

To add a model family, see [Adding a phenotype model](adding-a-phenotype-model.md).
