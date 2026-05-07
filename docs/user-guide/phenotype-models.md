# Phenotype Models

simACE maps each individual's continuous liability $L = A + C + E$ to an
observable affection status (and, for the time-to-event families, an
age of onset) via a configurable **phenotype model**. The model family
and its parameters are set per trait under the `phenotype.trait1` and
`phenotype.trait2` sub-blocks of the scenario configuration; the
overall configuration schema is described in
[Configuration](configuration.md).

## Schema

```yaml
phenotype:
  trait1:
    model: frailty            # frailty | cure_frailty | adult | first_passage
    params:                   # model-specific (see tables below)
      distribution: weibull
      scale: 2160
      rho: 0.8
    beta: 1.0                 # liability coefficient
    beta_sex: 0.0             # additive sex effect (sex == 1 is male)
  trait2:
    ...
```

`prevalence` is a property of the threshold-based families (`adult`,
`cure_frailty`) only and lives **inside** their `params` block:

```yaml
phenotype:
  trait1:
    model: adult
    params:
      method: ltm
      cip_x0: 50
      cip_k: 0.2
      prevalence: 0.10        # required for adult / cure_frailty
    beta: 1.0
```

`frailty` and `first_passage` reject `prevalence` outright — case
fraction emerges from the hazard for those families and a target
prevalence is meaningless.

| Key | Type | Description |
|---|---|---|
| `model` | str | Phenotype model family (see below). |
| `params` | dict | Model-specific (see tables below). For `adult` / `cure_frailty`, includes the required `prevalence`. |
| `beta` | float | Liability coefficient. For `frailty` / `cure_frailty` this multiplies the latent liability into the log-hazard; for `first_passage` it scales the drift; for `adult` it scales the standardized liability used in the threshold mapping. |
| `beta_sex` | float | Additive sex effect, applied in the same units as `beta`. |

## Model families

- `frailty` — proportional-hazards frailty. Liability scales the baseline
  hazard via $z = \exp(\beta\,L)$. Given sufficient time every individual
  eventually onsets. Requires `params.distribution` and the hazard
  parameters of that distribution. Does **not** accept `params.prevalence`.
- `cure_frailty` — mixture of a susceptible fraction (frailty) and an
  immune fraction (never onsets). The susceptible fraction is sized from
  `params.prevalence`. Same `params.distribution` schema as `frailty`,
  with `prevalence` added.
- `adult` — ADuLT liability-threshold variants (Pedersen et al., 2023).
  Requires `params.method` $\in$ {`ltm`, `cox`}, the cumulative-incidence
  parameters `cip_x0`, `cip_k`, and `params.prevalence`.
- `first_passage` — first-passage-time of a Brownian motion with drift.
  Requires `params.drift` and `params.shape`. Does **not** accept
  `params.prevalence`.

## Hazard distributions for `frailty` / `cure_frailty`

Set under `params.distribution`:

| Distribution | Required params |
|---|---|
| `weibull` | `scale`, `rho` |
| `exponential` | `rate` |
| `gompertz` | `rate`, `gamma` |
| `lognormal` | `mu`, `sigma` |
| `loglogistic` | `scale`, `shape` |
| `gamma` | `shape`, `scale` |

The registry is in `simace.phenotyping.hazards.BASELINE_HAZARDS`; each
entry maps a distribution name to a vectorized inverter.

## ADuLT methods for `adult`

Set under `params.method`:

| Method | Required params | Description |
|---|---|---|
| `ltm` | `cip_x0`, `cip_k`, `prevalence` | Deterministic liability-threshold with logistic cumulative-incidence proportion. |
| `cox` | `cip_x0`, `cip_k`, `prevalence` | Proportional hazards with Weibull noise and rank-based CIP-to-age mapping. |

## `first_passage` params

| Param | Description |
|---|---|
| `drift` | Drift rate of the latent random walk. Negative drift caps the susceptible fraction (cure-like behaviour); non-negative drift gives an eventually-affected population. |
| `shape` | Boundary parameter governing the time scale of first passage. |

## Parallel `simple_ltm` output

Independently of the configured family, every scenario produces a
parallel `phenotype.simple_ltm.parquet` output by applying
`simace.phenotyping.threshold.apply_threshold` to `liability1` /
`liability2`. For `adult` / `cure_frailty` traits this path uses the
configured `params.prevalence`; for `frailty` / `first_passage` traits
it falls back to the documented defaults `(0.10, 0.20)`. The cut respects
the global `standardize` flag (so `standardize: per_generation` makes the
benchmark prevalence-preserving per generation; `global` matches the
cohort-wide z-score).

## Standardization

The global `standardize` flag (`none` / `global` / `per_generation`)
controls how liability is normalised before phenotyping. The four
hazard-bearing families additionally accept a per-trait override
`standardize_hazard` inside `params`:

```yaml
phenotype:
  trait1:
    model: cure_frailty
    params:
      distribution: weibull
      scale: 2160
      rho: 0.8
      prevalence: 0.10
      standardize_hazard: per_generation   # overrides global standardize
                                           # for the hazard step only
    beta: 1.0
```

When omitted, `standardize_hazard` inherits the global `standardize`
value. `threshold` and `adult` with `method: ltm` reject the field —
they have no separate hazard step. See [ACE Model §
Standardisation](../concepts/ace-model.md#standardisation) for the
per-model routing table and `cure_frailty`'s two-knob behaviour.

## Prevalence

For `adult` / `cure_frailty`, three forms of `params.prevalence` are
accepted:

- **Scalar** (e.g. `0.10`) — same prevalence for every individual.
- **Per-generation dict** (e.g. `{2: 0.03, 3: 0.05, 4: 0.08, 5: 0.12}`)
  — prevalence varies across generations; every phenotyped generation
  must have an entry.
- **Sex-specific dict** (e.g. `{female: 0.08, male: 0.12}`) — prevalence
  differs by sex; each sex value may itself be a scalar or a
  per-generation dict.

## Adding a new phenotype model

Each model family is a frozen dataclass under `simace/phenotyping/models/`
that subclasses `PhenotypeModel`. To add a fifth family:

1. Write `simace/phenotyping/models/my_model.py` exposing a class
   `MyModel(PhenotypeModel)` with typed parameter fields and the abstract
   methods (`from_config`, `add_cli_args`, `from_cli`, `cli_flag_attrs`,
   `to_params_dict`, `simulate`).
2. Validate parameters in `__post_init__`. `from_config` and `from_cli`
   should wrap any `ValueError` / `TypeError` from construction with
   trait context via the `wrap_trait_error` helper.
3. Import the class in `simace/phenotyping/models/__init__.py` and add
   `"my_model": MyModel` to the `MODELS` dict.

That's the entire registration surface — no decorator, no auto-discovery.
The dispatcher in `simace.phenotyping.phenotype._simulate_one_trait`,
the validator in `simace.config._validate_phenotype_config`, and the
CLI in `simace.phenotyping.phenotype.cli` all read from `MODELS` and
pick up the new family automatically.
