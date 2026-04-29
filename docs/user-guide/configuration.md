# Configuration

## Overview

All simulation parameters are defined in YAML config files under `config/`.
The main file `config/_default.yaml` contains a `defaults` section and a `scenarios` section.
Each scenario inherits all defaults and overrides only the parameters it needs.

Additional config files (`config/{folder}.yaml`) are auto-discovered by Snakemake
and their scenarios are merged into the pipeline.

## Parameter reference

### Simulation

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | int | 42 | Random seed for reproducibility |
| `replicates` | int | 3 | Number of independent replicates per scenario |
| `folder` | str | `base` | Output folder name under `results/` |

### Variance components

Trait liability is decomposed as $L = A + C + E$ with unit variance ($A + C + E = 1$).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A1`, `A2` | float | 0.5 | Additive genetic variance (trait 1, 2) |
| `C1`, `C2` | float | 0.2 | Common environment variance |
| `rA` | float | 0.3 | Cross-trait genetic correlation |
| `rC` | float | 0.5 | Cross-trait common environment correlation |
| `standardize` | bool | true | Standardise liability before phenotyping |

$E$ is computed as $1 - A - C$ for each trait. Cross-trait $r_E$ is fixed at 0.

### Pedigree

| Parameter | Type | Default | Description |
|---|---|---|---|
| `N` | int | 100000 | Population size per generation |
| `G_ped` | int | 6 | Generations recorded in the pedigree |
| `G_pheno` | int | 3 | Generations to phenotype (last G_pheno of G_ped) |
| `G_sim` | int | 8 | Total generations simulated (G_sim - G_ped = burn-in) |
| `mating_lambda` | float | 0.5 | ZTP mating count parameter (~23% multi-partner) |
| `p_mztwin` | float | 0.02 | Probability of MZ twin birth |
| `assort1` | float | 0 | Mate correlation on trait 1 liability ([-1, 1]) |
| `assort2` | float | 0 | Mate correlation on trait 2 liability ([-1, 1]) |

### Phenotype model

| Parameter | Type | Default | Description |
|---|---|---|---|
| `phenotype_model1/2` | str | `weibull` | Baseline hazard model |
| `phenotype_params1/2` | dict | -- | Model-specific parameters (e.g., `scale`, `rho`) |
| `beta1/2` | float | 1.0 / 1.5 | Liability effect on log-hazard |
| `beta_sex1/2` | float | 0 | Sex effect on log-hazard |

Supported models: `weibull`, `exponential`, `gompertz`, `lognormal`, `loglogistic`,
`gamma`, `cure_frailty`, `adult_ltm`, `adult_cox`.

### Censoring

| Parameter | Type | Default | Description |
|---|---|---|---|
| `censor_age` | int | 80 | Maximum censoring age |
| `gen_censoring` | dict | per-gen windows | Per-generation `[left, right]` observation windows |
| `death_scale` | float | 164 | Competing-risk mortality Weibull scale |
| `death_rho` | float | 2.73 | Competing-risk mortality Weibull shape |

### Liability threshold model

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prevalence1/2` | float or dict | 0.10 / 0.20 | Proportion affected per generation |

Prevalence can be a scalar (same for all generations) or a per-generation dict
(e.g., `{ 2: 0.03, 3: 0.05, 4: 0.08, 5: 0.12 }`).

### Sampling and ascertainment

| Parameter | Type | Default | Description |
|---|---|---|---|
| `N_sample` | int | 0 | Subsample size (0 = keep all) |
| `case_ascertainment_ratio` | float | 1 | Case sampling weight relative to controls |
| `pedigree_dropout_rate` | float | 0 | Fraction of individuals to drop from pedigree |

### Miscellaneous

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_degree` | int | 2 | Max relationship degree to extract |
| `estimate_inbreeding` | bool | false | Compute exact inbreeding coefficients |
| `plot_format` | str | `png` | Plot file format (`png` or `pdf`) |

### tstrait and gene drop

The [gene-drop branch](../concepts/gene-drop.md) replaces the parametric $A$
with a tstrait-derived genetic value computed from real founder haplotypes
dropped through the simACE pedigree. Three top-level globals control how
that branch wires into the standard pipeline; nine `tstrait.*` keys control
the causal architecture.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_gene_drop` | bool | `false` | When true, `pedigree_dropout` reads `pedigree.full.tstrait.parquet` (rescaled gene-drop $A$) instead of the parametric `pedigree.full.parquet`. Triggers the full tstrait sub-pipeline + augment as upstream dependencies. |
| `drop_from` | str / null | `null` | Name of another scenario whose `pedigree.full.parquet` and `genotypes_chrom_*.trees` this scenario should consume — used for architecture sweeps that share one expensive drop+graft across many tstrait variants. `null` = run own drop. |
| `tstrait.num_causal` | int / null | 1000 | Absolute number of causal sites. Mutually exclusive with `frac_causal`. |
| `tstrait.frac_causal` | float / null | `null` | Fraction of post-MAF eligible sites to mark as causal. Mutually exclusive with `num_causal`. |
| `tstrait.maf_threshold` | float | 0.01 | Drop sites with $\min(\mathrm{AF}, 1-\mathrm{AF}) \le$ this. `0` disables the filter. |
| `tstrait.alpha` | float | -0.5 | Effect-size frequency-dependence exponent: $\beta = \mathcal{N}(\mu, \sigma_\beta^2) \cdot [2p(1-p)]^{\alpha}$. `0` = no MAF dependence; `-0.5` = LDAK-thin / Speed et al. |
| `tstrait.effect_mean` | float | 0.0 | Raw $\beta$ mean. |
| `tstrait.effect_var` | float | 1.0 | Raw $\beta$ variance (before MAF scaling). The augment step rescales the realised genome-wide $A$ to match `A1` regardless of this value. |
| `tstrait.trait_id` | int | 0 | Single-trait only for now; trait 2's $A_2$ stays parametric. |
| `tstrait.share_architecture` | bool | `false` | If true, causal sites + effects are shared across reps within a scenario (only env noise differs). |

Heritability for the gene-drop branch is **derived** from the standard
A/C/E components: $h^2 = A_1 / (A_1 + C_1 + E_1)$. There is no separate
`tstrait.h2` knob.

The `tskit_preprocess` block (also under top-level config) points at the
SimHumanity source and the directory where the canonicalized chromosomes
and shared site catalog are written:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tskit_preprocess.source_dir` | str | `/data/Documents/humanity_sim/simhumanity_trees_RO` | Directory containing per-chromosome SimHumanity `.trees` files. |
| `tskit_preprocess.output_dir` | str | `/data/Documents/humanity_sim/preprocessed_p2` | Where the canonicalized chroms, concat .trees, and `site_catalog.parquet` are written. |
| `tskit_preprocess.pop` | str | `p2` | Population name to filter to (founders for the drop). |
| `tskit_preprocess.chroms` | list[int] | 1..22 | Autosomes to include. |

## Defining scenarios

Add named scenarios under the `scenarios` key. Each scenario only needs to specify
parameters that differ from defaults:

```yaml
scenarios:
  baseline10K:
    seed: 1042
    N: 10000

  high_heritability:
    folder: heritability
    seed: 4042
    A1: 0.8
    C1: 0.0
    A2: 0.8
    C2: 0.0
```

## Per-folder configuration

Additional config files placed in `config/` are auto-discovered by Snakemake.
For example, `config/cross_trait.yaml` defines scenarios that output to `results/cross_trait/`.
Each file follows the same `defaults` + `scenarios` structure.
