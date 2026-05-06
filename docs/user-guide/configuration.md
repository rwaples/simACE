# Configuration

## Overview

Simulation parameters are defined in YAML files under `config/`.
`config/_default.yaml` contains the global defaults, while each
`config/{folder}.yaml` file defines scenarios for one output folder. Scenario
files use bare scenario names; the folder name is inferred from the filename.

Each scenario inherits the defaults and overrides only the values that differ.
The preferred authoring style is the hierarchical schema shown below. The
loader still accepts older flat keys such as `A1` or `censor_age` for
compatibility, but new configs should use the sectioned form.

## Defaults

### Top-level globals

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `seed` | int | 42 | Base random seed; replicate seeds are derived from this value |
| `replicates` | int | 3 | Number of independent replicates per scenario |
| `folder` | str | `base` | Output folder under `results/` |
| `N` | int | 100000 | Population size per generation |
| `G_ped` | int | 6 | Recorded pedigree generations |
| `G_pheno` | int | 3 | Last `G_pheno` generations to phenotype |
| `G_sim` | int | 8 | Total simulated generations; `G_sim - G_ped` is burn-in |
| `standardize` | str | `global` | Liability standardization mode: `none`, `global`, or `per_generation` |
| `plot_format` | str | `png` | Plot extension, usually `png` or `pdf` |
| `drop_from` | str / null | `null` | Reuse another scenario's pedigree and gene-drop outputs |
| `use_gene_drop` | bool | `false` | Use tstrait-derived `A1` instead of parametric `A1` downstream |

See [ACE Model § Standardisation](../concepts/ace-model.md#standardisation)
for how `standardize` interacts with threshold and hazard-bearing models.

### Pedigree

```yaml
pedigree:
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
| `mating_lambda` | Zero-truncated Poisson mating-count parameter; default gives about 23% multi-partner individuals |
| `p_mztwin` | Probability of monozygotic twin birth |
| `assort1`, `assort2` | Mate correlation on trait 1 and trait 2 liability |
| `assort_matrix` | Optional full 2x2 female/male mate-correlation matrix |
| `trait{1,2}.A` | Additive genetic variance component |
| `trait{1,2}.C` | Shared/common environment variance component |
| `trait{1,2}.E` | Unique environment variance component |
| `rA`, `rC`, `rE` | Cross-trait correlations for A, C, and E |

### Phenotype

Each trait is configured independently under `phenotype.trait1` and
`phenotype.trait2`:

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

`model` must be one of `frailty`, `cure_frailty`, `adult`, or
`first_passage`. `params` is model-specific; threshold-based families
(`adult` and `cure_frailty`) require `params.prevalence`. See
[Phenotype Models](phenotype-models.md) for the full model catalogue,
required parameters, supported prevalence forms, and `standardize_hazard`
rules.

### Censoring

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
| `gen_censoring` | Per-generation `[left, right]` observation windows |
| `death_scale`, `death_rho` | Weibull competing-risk mortality parameters |

### Sampling and analysis

```yaml
sampling:
  N_sample: 0
  case_ascertainment_ratio: 1
  pedigree_dropout_rate: 0

analysis:
  max_degree: 2
  estimate_inbreeding: false
```

| Parameter | Description |
|---|---|
| `sampling.N_sample` | Subsample size; `0` keeps all phenotyped individuals |
| `sampling.case_ascertainment_ratio` | Case sampling weight relative to controls |
| `sampling.pedigree_dropout_rate` | Fraction of individuals removed from the pedigree before downstream stages |
| `analysis.max_degree` | Maximum relationship degree to extract |
| `analysis.estimate_inbreeding` | Compute exact inbreeding coefficients and exact pairwise kinship |

## tstrait and gene drop

The [gene-drop branch](../concepts/gene-drop.md) replaces the parametric
trait-1 additive component with a tstrait-derived genetic value computed from
founder haplotypes dropped through the simACE pedigree.

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
| `use_gene_drop` | Top-level switch that makes downstream stages read `pedigree.full.tstrait.parquet` |
| `drop_from` | Top-level scenario name to reuse an existing drop/graft |
| `tstrait.num_causal` | Absolute number of causal sites; mutually exclusive with `frac_causal` |
| `tstrait.frac_causal` | Fraction of MAF-eligible sites to use as causal; mutually exclusive with `num_causal` |
| `tstrait.maf_threshold` | Minimum minor-allele frequency filter; `0` disables filtering |
| `tstrait.alpha` | Effect-size frequency-dependence exponent |
| `tstrait.effect_mean`, `tstrait.effect_var` | Raw effect-size distribution parameters before MAF scaling |
| `tstrait.trait_id` | Single-trait selector; trait 2 remains parametric |
| `tstrait.share_architecture` | Share causal sites and effects across replicates |

The gene-drop branch derives heritability from the standard A/C/E components:
$h^2 = A_1 / (A_1 + C_1 + E_1)$. There is no separate `tstrait.h2` parameter.

`tskit_preprocess` is a standalone top-level block for canonicalizing source
tree sequences:

| Parameter | Default | Description |
|---|---|---|
| `tskit_preprocess.source_dir` | `/data/Documents/humanity_sim/simhumanity_trees_RO` | Source directory for per-chromosome SimHumanity `.trees` files |
| `tskit_preprocess.output_dir` | `/data/Documents/humanity_sim/preprocessed_p2` | Output directory for canonicalized chromosomes, concatenated trees, and site catalog |
| `tskit_preprocess.pop` | `p2` | Founder population to filter |
| `tskit_preprocess.chroms` | `1..22` | Autosomes to include |

## Defining scenarios

Per-folder files contain only scenario dictionaries. For example,
`config/base.yaml`:

```yaml
baseline10K:
  seed: 1042
  N: 10000

baseline100K_sample5K:
  seed: 2042
  N: 100000
  sampling:
    N_sample: 5000
```

Nested sections are merged over the defaults, so a scenario can override only
one field inside a section:

```yaml
high_heritability:
  folder: heritability
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

Run a scenario by targeting its resolved folder and scenario name:

```bash
snakemake --cores 4 results/base/baseline10K/scenario.done
```
