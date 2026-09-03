# Configuration

Simulation parameters live in YAML files under `config/`.
`config/_default.yaml` holds the defaults. Each `config/{folder}.yaml` file
defines the scenarios for one output folder, and the folder name is the
filename. Files whose name starts with `_` are not scenario files.

A scenario inherits every default and overrides only the values it lists.
Write new scenarios in the sectioned form shown on this page. The loader also
accepts the older flat keys such as `A1` and `censor_age`, but mixing the flat
and sectioned form for one parameter is an error.

## Top-level parameters

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `seed` | int | 42 | Base random seed. Replicate `r` uses `seed + r - 1` |
| `replicates` | int | 3 | Number of independent replicates per scenario |
| `folder` | str | `base` | Output folder under `results/` |
| `N` | int | 100000 | Population size per generation |
| `G_ped` | int | 6 | Recorded pedigree generations |
| `G_pheno` | int | 3 | Number of most recent generations to phenotype |
| `G_sim` | int | 8 | Total simulated generations. `G_sim - G_ped` are burn-in |
| `standardize` | str | `global` | Liability standardization: `none`, `global`, or `per_generation`. The legacy values `true` and `false` map to `global` and `none` |
| `plot_format` | str | `png` | Image extension for plots. Use `png` or `svg`, because the HTML atlas embeds the images. `pdf` works only for the `atlas.pdf` export |
| `drop_from` | str or null | `null` | Name of another scenario whose pedigree and gene-drop outputs this scenario reuses |
| `use_gene_drop` | bool | `false` | Read the tstrait-derived `A1` instead of the parametric one in every downstream stage |
| `blended_diagnosis` | dict or null | `null` | Per-generation blend of the two liabilities that fitACE applies to trait-1 case status before EPIMIGHT. The simulator ignores it |

[ACE model, Standardisation](../concepts/ace-model.md#standardisation)
explains how `standardize` interacts with the threshold and hazard models.

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

| Parameter | Description |
|---|---|
| `mating_model` | `standard` or `wright_fisher` ([ADR 0002](../adr/0002-wright-fisher-mating-model.md)). Under `wright_fisher`, the loader rejects any override of `mating_lambda`, `p_mztwin`, `assort1`, `assort2`, or `assort_matrix` |
| `mating_lambda` | Parameter of the zero-truncated Poisson number of mates. The default gives about 23 percent of individuals more than one mate |
| `p_mztwin` | Probability that a birth is a monozygotic twin pair |
| `assort1`, `assort2` | Mate correlation on trait 1 and on trait 2 liability |
| `assort_matrix` | Optional 2 by 2 female-by-male mate-correlation matrix. When set, `assort1` and `assort2` are its diagonal |
| `trait1.A`, `trait2.A` | Additive genetic variance |
| `trait1.C`, `trait2.C` | Common environment variance |
| `trait1.E`, `trait2.E` | Unique environment variance |
| `rA`, `rC`, `rE` | Cross-trait correlation of A, of C, and of E |

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

`model` is one of `frailty`, `cure_frailty`, `adult`, `first_passage`, or
`simple_ltm`. The contents of `params` depend on the model. The threshold
models `adult`, `cure_frailty`, and `simple_ltm` require `params.prevalence`.
[Phenotype models](phenotype-models.md) lists every model with its parameters.

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

| Parameter | Description |
|---|---|
| `max_age` | Maximum follow-up age |
| `gen_censoring` | Observation window `[left, right]` for each generation |
| `death_scale`, `death_rho` | Weibull scale and shape of the competing-risk mortality |

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

| Parameter | Description |
|---|---|
| `ascertainment.N_sample` | Sample size after ascertainment. `0` keeps the whole post-dropout population |
| `ascertainment.case_ascertainment_ratio` | Sampling weight of a case relative to a control in the `N_sample` draw |
| `ascertainment.dropout_rate` | Fraction of individuals removed at random from the pedigree before the draw |
| `analysis.max_degree` | Highest relationship degree to extract. `3` includes first cousins. `2` stops at half-siblings, grandparents, and avuncular pairs |
| `analysis.estimate_inbreeding` | Compute exact inbreeding coefficients and exact pairwise kinship |
| `analysis.skip_ne_coancestry` | Skip the coancestry-rate estimator of effective population size and report `ne_coancestry` as null. The other seven estimators still run. The default is `true` because this estimator dominates the memory of the `effective_size` rule. Set it to `false` for scenarios with a small pedigree |

[Ascertainment](ascertainment.md) explains the dropout and draw steps.

## Gene drop with tstrait

When `use_gene_drop` is true, the [gene-drop branch](../concepts/gene-drop.md)
replaces the parametric trait-1 additive component with a genetic value that
tstrait computes from founder haplotypes dropped through the pedigree.

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

| Parameter | Description |
|---|---|
| `tstrait.num_causal` | Number of causal sites. Set either this or `frac_causal`, not both |
| `tstrait.frac_causal` | Fraction of sites that pass the MAF filter to use as causal. Set either this or `num_causal`, not both |
| `tstrait.maf_threshold` | Minimum minor-allele frequency. `0` disables the filter |
| `tstrait.alpha` | Exponent of the effect-size dependence on allele frequency |
| `tstrait.effect_mean`, `tstrait.effect_var` | Mean and variance of the raw effect sizes before frequency scaling |
| `tstrait.trait_id` | Which trait gets the genetic value. Trait 2 stays parametric |
| `tstrait.share_architecture` | Reuse the same causal sites and effects in every replicate |

Heritability on this branch is `A1 / (A1 + C1 + E1)` from the pedigree
section. There is no `tstrait.h2` parameter.

`tskit_preprocess` is a separate top-level block for the one-time step that
canonicalizes the source tree sequences. It is not part of any scenario.

| Parameter | Default | Description |
|---|---|---|
| `tskit_preprocess.source_dir` | `/data/Documents/humanity_sim/simhumanity_trees_RO` | Directory of per-chromosome SimHumanity `.trees` files |
| `tskit_preprocess.output_dir` | `/data/Documents/humanity_sim/preprocessed_p2` | Directory for the canonicalized chromosomes, the concatenated tree sequence, and the site catalog |
| `tskit_preprocess.pop` | `p2` | Founder population to keep |
| `tskit_preprocess.chroms` | `1` to `22` | Autosomes to include |

## Write a scenario

A folder file holds only scenario dictionaries. This is the start of
`config/base.yaml`:

```yaml
baseline10K:
  seed: 1042
  N: 10000
baseline100K:
  seed: 2042
  N: 100000
baseline100K_sample5K:
  seed: 2042
  N: 100000
  ascertainment:
    N_sample: 5000
```

Sections merge over the defaults field by field, so a scenario can change one
value inside a section and inherit the rest. `high_heritability` in
`config/heritability.yaml` changes only the variance components:

```yaml
high_heritability:
  seed: 4042
  pedigree:
    trait1:
      A: 0.8
      C: 0.0
      E: 0.2
    trait2:
      A: 0.8
      C: 0.0
      E: 0.2
```

To run a scenario, target its folder and name as described in
[Running the pipeline](running-the-pipeline.md).
