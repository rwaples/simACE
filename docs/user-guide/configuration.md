# Configuration

Simulation parameters live in YAML files under `config/`.
`config/_default.yaml` holds the defaults. Each `config/{folder}.yaml` file is a
scenario file. It defines the scenarios for one output folder, and the folder
takes its name from the file. Files whose name starts with `_` are not
scenario files.

A scenario inherits every default and overrides only the values it lists.
This page shows the sectioned form. The loader also accepts the older flat
keys listed under [Legacy flat keys](#legacy-flat-keys). Mixing the flat and
sectioned form for one parameter is an error.

<!-- scenario-defaults:start -->

## Top-level parameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `seed` | int | `42` | Base random seed. Replicate `r` uses `seed + r - 1` |
| `replicates` | int | `3` | Number of independent replicates per scenario |
| `folder` | str | `base` | Output folder under `results/` |
| `N` | int | `100000` | Population size per generation |
| `G_ped` | int | `6` | Recorded pedigree generations |
| `G_pheno` | int | `3` | Number of most recent generations to phenotype |
| `G_sim` | int | `8` | Total simulated generations. `G_sim - G_ped` are burn-in |
| `standardize` | str | `global` | Liability standardization mode: `none`, `global`, or `per_generation`. The legacy values `true` and `false` map to `global` and `none` |
| `plot_format` | str | `png` | Image extension for plots. Use `png` or `svg`, because the HTML atlas embeds the images. `pdf` works only for the `atlas.pdf` export |
| `drop_from` | str or null | `null` | Name of another scenario whose pedigree and gene-drop outputs this scenario reuses |
| `use_gene_drop` | bool | `false` | Read the tstrait-derived `A1` instead of the parametric one in every downstream stage |
| `blended_diagnosis` | dict or null | `null` | Per-generation blend of the two liabilities that fitACE applies to trait-1 case status before EPIMIGHT. The simulator ignores it |

[ACE model, Standardisation](../concepts/ace-model.md#standardisation)
explains how `standardize` interacts with the threshold and hazard models.
Hazard-bearing models can override it per trait with
`phenotype.trait{N}.params.standardize_hazard`; see
[Phenotype models, Standardization](phenotype-models.md#standardization).

## Pedigree

```yaml
pedigree:
  mating_model: standard
  mating_lambda: 0.5
  p_mztwin: 0.02
  assort1: 0
  assort2: 0
  assort_matrix: null
  trait1:
    A: 0.5
    C: 0.0
    E: 0.5
  trait2:
    A: 0.4
    C: 0.2
    E: 0.4
  rA: 0.0
  rC: 0.0
  rE: 0.0
```

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `pedigree.mating_model` | str | `standard` | `standard` or `wright_fisher`. See [ADR 0002](../adr/0002-wright-fisher-mating-model.md) |
| `pedigree.mating_lambda` | float | `0.5` | Parameter of the zero-truncated Poisson number of mates. The default gives about 23 percent of individuals more than one mate |
| `pedigree.p_mztwin` | float | `0.02` | Probability that a birth is a monozygotic twin pair |
| `pedigree.assort1` | float or dict | `0` | Mate correlation on trait 1 liability. A dict sets it by generation |
| `pedigree.assort2` | float or dict | `0` | Mate correlation on trait 2 liability. A dict sets it by generation |
| `pedigree.assort_matrix` | matrix or null | `null` | Optional 2 by 2 female-by-male mate-correlation matrix. Its diagonal replaces `assort1` and `assort2` |
| `pedigree.trait1.A` | float or dict | `0.5` | Trait 1 additive genetic variance |
| `pedigree.trait1.C` | float or dict | `0.0` | Trait 1 common environment variance |
| `pedigree.trait1.E` | float or dict | `0.5` | Trait 1 unique environment variance |
| `pedigree.trait2.A` | float or dict | `0.4` | Trait 2 additive genetic variance |
| `pedigree.trait2.C` | float or dict | `0.2` | Trait 2 common environment variance |
| `pedigree.trait2.E` | float or dict | `0.4` | Trait 2 unique environment variance |
| `pedigree.rA` | float | `0.0` | Cross-trait correlation of the additive genetic components |
| `pedigree.rC` | float | `0.0` | Cross-trait correlation of the common environment components |
| `pedigree.rE` | float | `0.0` | Cross-trait correlation of the unique environment components |

Under `wright_fisher`, the loader rejects any `mating_lambda` override. It
also rejects nonzero `p_mztwin`, `assort1`, or `assort2` overrides and any
non-null `assort_matrix` override.

## Phenotype

Each trait has its own block under `phenotype.trait1` and `phenotype.trait2`.

```yaml
phenotype:
  trait1:
    model: frailty
    params:
      distribution: weibull
      scale: 2160
      rho: 0.8
    beta: 1.0
    beta_sex: 0.0
  trait2:
    model: frailty
    params:
      distribution: weibull
      scale: 333
      rho: 1.2
    beta: 1.5
    beta_sex: 0.0
```

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `phenotype.trait1.model` | str | `frailty` | Trait 1 phenotype model |
| `phenotype.trait1.params.distribution` | str | `weibull` | Trait 1 baseline event-time distribution |
| `phenotype.trait1.params.scale` | float | `2160` | Trait 1 Weibull scale in age units |
| `phenotype.trait1.params.rho` | float | `0.8` | Trait 1 Weibull shape |
| `phenotype.trait1.beta` | float | `1.0` | Trait 1 liability coefficient. Its meaning depends on the model |
| `phenotype.trait1.beta_sex` | float | `0.0` | Trait 1 additive male effect in the same coefficient units as `beta` |
| `phenotype.trait2.model` | str | `frailty` | Trait 2 phenotype model |
| `phenotype.trait2.params.distribution` | str | `weibull` | Trait 2 baseline event-time distribution |
| `phenotype.trait2.params.scale` | float | `333` | Trait 2 Weibull scale in age units |
| `phenotype.trait2.params.rho` | float | `1.2` | Trait 2 Weibull shape |
| `phenotype.trait2.beta` | float | `1.5` | Trait 2 liability coefficient. Its meaning depends on the model |
| `phenotype.trait2.beta_sex` | float | `0.0` | Trait 2 additive male effect in the same coefficient units as `beta` |

`model` is one of `frailty`, `cure_frailty`, `adult`, `first_passage`, or
`simple_ltm`. The contents of `params` depend on the model. The threshold
models `adult`, `cure_frailty`, and `simple_ltm` require `params.prevalence`.
It accepts a scalar, a per-generation dict, or a sex-specific dict whose
`female` and `male` values are scalars or per-generation dicts. See
[Phenotype models, Prevalence forms](phenotype-models.md#prevalence-forms) for
the three forms and [Phenotype models](phenotype-models.md) for every model's
parameters.

simACE does not convert time units. The shipped configurations treat one age
unit as one year. Event-time `scale` parameters, cumulative-incidence age
parameters, onset ages, and censoring ages must use the same unit.

## Censoring

```yaml
censoring:
  max_age: 80
  gen_censoring:
    0: [80, 80]
    1: [80, 80]
    2: [80, 80]
    3: [40, 80]
    4: [0, 80]
    5: [0, 45]
  death_scale: 164
  death_rho: 2.73
```

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `censoring.max_age` | float | `80` | Maximum follow-up age, in age units |
| `censoring.gen_censoring` | dict | `{0: [80, 80], 1: [80, 80], 2: [80, 80], 3: [40, 80], 4: [0, 80], 5: [0, 45]}` | Observation window `[left, right]` for each generation, in age units |
| `censoring.death_scale` | float | `164` | Weibull scale of the competing-risk death age, in age units |
| `censoring.death_rho` | float | `2.73` | Weibull shape of the competing-risk death age |

For a Weibull death age, the median is
`death_scale * (log(2)) ** (1 / death_rho)`. The defaults give about 143.4 age
units. With the shipped convention, that is 143.4 years.

## Ascertainment and analysis

```yaml
ascertainment:
  N_sample: 0
  case_ascertainment_ratio: 1
  dropout_rate: 0

analysis:
  max_degree: 3
  estimate_inbreeding: false
  skip_ne_coancestry: true
```

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `ascertainment.N_sample` | int | `0` | Sample size after ascertainment. `0` keeps the whole post-dropout population |
| `ascertainment.case_ascertainment_ratio` | float | `1` | Sampling weight of a case relative to a control in the `N_sample` draw |
| `ascertainment.dropout_rate` | float | `0` | Fraction of individuals removed at random from the pedigree before the draw |
| `analysis.max_degree` | int | `3` | Highest relationship degree to extract. `3` includes first cousins. `2` stops at half-siblings, grandparents, and avuncular pairs |
| `analysis.estimate_inbreeding` | bool | `false` | Compute exact inbreeding coefficients and exact pairwise kinship |
| `analysis.skip_ne_coancestry` | bool | `true` | Skip the coancestry-rate estimator of effective population size and report `ne_coancestry` as null. The other seven estimators still run |

The default `analysis.skip_ne_coancestry: true` avoids the high memory cost of
the coancestry-rate estimator. Set it to `false` for a pedigree small enough
to compute the estimator.

[Ascertainment](ascertainment.md) explains the dropout and draw steps. The
`analysis` section configures the analyze stage, which writes each replicate's
`report.yaml` (see [Running the pipeline](running-the-pipeline.md)).

## Gene drop with tstrait

When `use_gene_drop` is true, [gene drop](../concepts/gene-drop.md)
replaces the parametric trait-1 additive component with a genetic value from
tstrait. tstrait computes that value from founder haplotypes dropped through
the pedigree.

```yaml
tstrait:
  num_causal: 1000
  frac_causal: null
  maf_threshold: 0.01
  alpha: -0.5
  effect_mean: 0.0
  effect_var: 1.0
  trait_id: 0
  share_architecture: false
```

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `tstrait.num_causal` | int or null | `1000` | Number of causal sites. Set exactly one of this parameter and `frac_causal` |
| `tstrait.frac_causal` | float or null | `null` | Fraction of sites that pass the MAF filter to use as causal. Set exactly one of this parameter and `num_causal` |
| `tstrait.maf_threshold` | float | `0.01` | Minimum minor-allele frequency. `0` disables the filter |
| `tstrait.alpha` | float | `-0.5` | Exponent of the effect-size dependence on allele frequency |
| `tstrait.effect_mean` | float | `0.0` | Mean of the raw effect sizes before frequency scaling |
| `tstrait.effect_var` | float | `1.0` | Variance of the raw effect sizes before frequency scaling |
| `tstrait.trait_id` | int | `0` | Trait that gets the genetic value. Only trait 1, ID `0`, is supported |
| `tstrait.share_architecture` | bool | `false` | Reuse the same causal sites and effects in every replicate |

Heritability under gene drop is `A1 / (A1 + C1 + E1)` from the pedigree
section. There is no `tstrait.h2` parameter.

<!-- scenario-defaults:end -->

`tskit_preprocess` is a separate top-level block for the one-time step that
canonicalizes the source tree sequences. It is not part of any scenario. The
two directory defaults point at the maintainer's local copy of the SimHumanity
data, so set both for your machine.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tskit_preprocess.source_dir` | path | `/data/Documents/humanity_sim/simhumanity_trees_RO` | Directory of per-chromosome SimHumanity `.trees` files |
| `tskit_preprocess.output_dir` | path | `/data/Documents/humanity_sim/preprocessed_p2` | Directory for the canonicalized chromosomes, the concatenated tree sequence, and the site catalog |
| `tskit_preprocess.pop` | str | `p2` | Founder population to keep |
| `tskit_preprocess.chroms` | list[int] | `1` through `22` | Autosomes to include |

## Legacy flat keys

Older scenario files can use flat keys. New files should use the sectioned
keys because their ownership is visible in the YAML structure. The loader
accepts either form, but it rejects a parameter supplied in both forms.

<details markdown="1">
<summary>Show the legacy flat-key mapping</summary>

<!-- legacy-aliases:start -->

| Sectioned key | Legacy flat key |
|---|---|
| `pedigree.mating_model` | `mating_model` |
| `pedigree.mating_lambda` | `mating_lambda` |
| `pedigree.p_mztwin` | `p_mztwin` |
| `pedigree.assort1` | `assort1` |
| `pedigree.assort2` | `assort2` |
| `pedigree.assort_matrix` | `assort_matrix` |
| `pedigree.trait1.A` | `A1` |
| `pedigree.trait1.C` | `C1` |
| `pedigree.trait1.E` | `E1` |
| `pedigree.trait2.A` | `A2` |
| `pedigree.trait2.C` | `C2` |
| `pedigree.trait2.E` | `E2` |
| `pedigree.rA` | `rA` |
| `pedigree.rC` | `rC` |
| `pedigree.rE` | `rE` |
| `phenotype.trait1.model` | `phenotype_model1` |
| `phenotype.trait1.params` | `phenotype_params1` |
| `phenotype.trait1.beta` | `beta1` |
| `phenotype.trait1.beta_sex` | `beta_sex1` |
| `phenotype.trait2.model` | `phenotype_model2` |
| `phenotype.trait2.params` | `phenotype_params2` |
| `phenotype.trait2.beta` | `beta2` |
| `phenotype.trait2.beta_sex` | `beta_sex2` |
| `censoring.max_age` | `censor_age` |
| `censoring.gen_censoring` | `gen_censoring` |
| `censoring.death_scale` | `death_scale` |
| `censoring.death_rho` | `death_rho` |
| `ascertainment.N_sample` | `N_sample` |
| `ascertainment.case_ascertainment_ratio` | `case_ascertainment_ratio` |
| `ascertainment.dropout_rate` | `dropout_rate` |
| `analysis.max_degree` | `max_degree` |
| `analysis.estimate_inbreeding` | `estimate_inbreeding` |
| `analysis.skip_ne_coancestry` | `skip_ne_coancestry` |
| `tstrait.num_causal` | `tstrait_num_causal` |
| `tstrait.frac_causal` | `tstrait_frac_causal` |
| `tstrait.maf_threshold` | `tstrait_maf_threshold` |
| `tstrait.alpha` | `tstrait_alpha` |
| `tstrait.effect_mean` | `tstrait_effect_mean` |
| `tstrait.effect_var` | `tstrait_effect_var` |
| `tstrait.trait_id` | `tstrait_trait_id` |
| `tstrait.share_architecture` | `tstrait_share_architecture` |

<!-- legacy-aliases:end -->

</details>

## What the loader rejects

The scenario loader rejects these configuration errors before `simace run` starts
simulation jobs:

- An unknown flat key or sectioned key.
- The same parameter in both flat and sectioned form.
- A scenario name that appears in more than one file.
- A configuration filename whose stem contains characters other than letters,
  digits, and underscores.
- `pedigree.trait1.E: null` or `pedigree.trait2.E: null` after defaults and
  scenario overrides are resolved.
- A `phenotype.trait{N}.model` outside the five supported model families.
- A missing or unknown `distribution` for `frailty` and `cure_frailty`.
- A missing or unknown `method` for `adult`.
- A missing `onset` dict or an unknown `onset.kind` for `simple_ltm`.
- Missing `params.prevalence` for `adult`, `cure_frailty`, or `simple_ltm`.
- `params.prevalence` for `frailty` or `first_passage`, or `prevalence` placed
  directly under a trait instead of inside `params`.
- An unknown `pedigree.mating_model` or an incompatible explicit override for
  `wright_fisher`, as described under [Pedigree](#pedigree).

Individual stages also validate their numeric ranges when they run. For
example, ascertainment requires `0 <= dropout_rate < 1` and a nonnegative
`case_ascertainment_ratio`. The tstrait effect-assignment stage requires
exactly one of `num_causal` and `frac_causal`.

[Writing a scenario](writing-a-scenario.md) shows how to add a scenario to a
scenario file.
