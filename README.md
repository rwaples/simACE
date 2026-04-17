# ACE - Simulate registry-scale age-of-onset phenotypes with realistic pedigrees and the ACE liability model. 

simACE simulates millions of indiviudals in multi-generational pedigrees with heritable ACE variance components for two correlated traits.

## Overview
The project contains two installable Python packages:
- **simace** -- Simulation, phenotyping, censoring, sampling, analysis, and plotting
- **[fit_ace](fit_ace/README.md)** -- Statistical model fitting (EPIMIGHT heritability, PA-FGRS genetic risk scores, Weibull frailty correlation, Stan models) — see [fit_ace/README.md](fit_ace/README.md) for setup, usage, and outputs

simACE is conceptually split into four simulation stages:
- Multi-generation pedigree simulation with heritable liability
- Age-of-onset phenotyping
- Censoring
- Sampling and observation

The full pipeline also includes statistical validation, summary statistics, model fitting, and plotting.


### Reproduction and liabilty

- Each generation, many couples are formed from potential parents.  
- An individual can be part of multiple couples. 
- Males and female are paired randomly by default or assortatively on the liability for one or both traits (follwoing Border et al. 2022).
- Offspring are distributed across matings via a multinomial draw.  Population size is kept constant and some couples produce no offspring.
- Monozygotic (MZ) twins are assigned to matings with ≥2 offspring.
- Full pedigree is recorded.

At default settings, ~77% of individuals have one partner and ~23% have two or more, producing a natural mix of full-sibs and maternal and paternal half-sibs. 

Total liability for individual $i$ on trait $k$ is decomposed as $L_i^{(k)} = A_i^{(k)} + C_i^{(k)} + E_i^{(k)}$, where the three components sum to unit variance:

- **A** (Additive genetic) — inherited from parents via midparent averaging plus Mendelian sampling noise under the infinitesimal model ($\epsilon \sim \mathcal{N}(0, \sigma_A^2/2)$). For the founder generation, additive values for both traits are drawn jointly from a bivariate normal with cross-trait genetic correlation $r_A$.
- **C** (Common/shared environment) — shared by all offspring of the same mother (household effect). Not inherited - there is (currently) no parent-to-child C transmission.
- **E** (Unique/personal environment) — drawn independently per individual per trait. No familial correlation by design.


### Age of onset phenotyping
The continuous liabilities for each trait are mapped to age-of-onset phenotypes via time-to-event phenotype models. Phenotype model options currently include: 
- **Frailty** — Proportional hazards frailty (random effects) model with choice of baseline hazard form (Weibull, Gompertz, lognormal, etc.). Liability scales the hazard via `z = exp(beta * L)`.  In pure frailty models, given enough time everyone eventually will become affected.
- **Cure-Frailty** — Mixture model that separates **who** gets the disease (succeceptable vs non-susceptible indiviudals), from **when** (age-of-onset among susceptibles).  Supports sex-specific prevalence (`K_m`, `K_f`) and four baseline hazards models.
- **ADuLT LTM** — Deterministic liability threshold model with logistic cumulative incidence proportion (CIP) age-of-onset mapping (Pedersen et al., Nat Commun 2023)
- **ADuLT Cox** — Proportional hazards model with Weibull noise and rank-based CIP-to-age mapping (Pedersen et al., 2023)
- A separate simple liability-threshold model produces binary affected status without age of onset or censoring for comparison. 
- It is also possible to only phenotype a subset of simulated generations. 
- See [distributions.md](distributions.md) for model details.

### Censoring
After phenotyping, two censoring layers are applied to mimic real-world data limitations:

1. **Age-window censoring** — each generation has a configurable observation interval `[left_age, right_age]`. Events occurring before `left_age` are left-truncated; events after `right_age` are right-censored. This models differential follow-up: e.g., some generations may be fully observed `[0, 80]` while the youngest is only followed to age 45 `[0, 45]`.
2. **Competing-risk mortality** — a death age is drawn independently from a Weibull distribution (`death_scale`, `death_rho`). If onset occurs after death, the individual is death-censored at their death age. This introduces realistic attrition that is correlated with age but independent of disease liability.

The combined effect is that only a fraction of true cases are observed as affected — the rest are censored by follow-up limits or death. 

### Sampling and ascertainment
It it possiblle to further restrict the observed data via sampling to construct the final output. 

- **Subsampling** (`N_sample`) — draws a random subset of phenotyped individuals before generating outputs, reflecting limited registry coverage for phenotype data. Relationship extraction continues on the full pedigree.
- **Case ascertainment bias** (`case_ascertainment_ratio`) — when combined with subsampling, cases are sampled at a different rate than non-affected individuals (e.g., ratio=5 means a case is 5x more likely to be drawn). Reflects the design of many data sets. 
- **Pedigree dropout** (`pedigree_dropout_rate`) — randomly removes a fraction of individuals from the pedigree, reflecting incomplete registration for pedigree building. Dropped individuals' pedigree links are broken, which can downgrade full-sibling pairs to half-siblings and/or break multi-hop relationship paths.




## Pedigree Relationship Types

`PedigreeGraph` (in `simace/core/pedigree_graph.py`) extracts 23 relationship categories from simulated pedigrees using sparse matrix algebra. Each type is parameterised by `(up, down, n_ancestors)` — meioses up from individual A to common ancestor(s), meioses down to individual B, and whether the link is through 1 (half/lineal) or 2 (full, mated-pair) ancestors. Kinship = `n_ancestors × (1/2)^(up + down + 1)`.

| Code | Label | Up | Down | Ancestors | Kinship | Degree |
|------|-------|---:|-----:|----------:|--------:|-------:|
| MZ | MZ twin | — | — | — | 1/2 | 0 |
| MO | Mother-offspring | 1 | 0 | 1 | 1/4 | 1 |
| FO | Father-offspring | 1 | 0 | 1 | 1/4 | 1 |
| FS | Full sib | 1 | 1 | 2 | 1/4 | 1 |
| MHS | Maternal half sib | 1 | 1 | 1 | 1/8 | 2 |
| PHS | Paternal half sib | 1 | 1 | 1 | 1/8 | 2 |
| GP | Grandparent | 2 | 0 | 1 | 1/8 | 2 |
| Av | Avuncular | 1 | 2 | 2 | 1/8 | 2 |
| GGP | Great-grandparent | 3 | 0 | 1 | 1/16 | 3 |
| HAv | Half-avuncular | 1 | 2 | 1 | 1/16 | 3 |
| GAv | Great-avuncular | 1 | 3 | 2 | 1/16 | 3 |
| 1C | 1st cousin | 2 | 2 | 2 | 1/16 | 3 |
| GGGP | Great²-grandparent | 4 | 0 | 1 | 1/32 | 4 |
| HGAv | Half-great-avuncular | 1 | 3 | 1 | 1/32 | 4 |
| GGAv | Great²-avuncular | 1 | 4 | 2 | 1/32 | 4 |
| H1C | Half-1st-cousin | 2 | 2 | 1 | 1/32 | 4 |
| 1C1R | 1st cousin 1R | 2 | 3 | 2 | 1/32 | 4 |
| G3GP | Great³-grandparent | 5 | 0 | 1 | 1/64 | 5 |
| HGGAv | Half-great²-avuncular | 1 | 4 | 1 | 1/64 | 5 |
| G3Av | Great³-avuncular | 1 | 5 | 2 | 1/64 | 5 |
| H1C1R | Half-1st-cousin 1R | 2 | 3 | 1 | 1/64 | 5 |
| 1C2R | 1st cousin 2R | 2 | 4 | 2 | 1/64 | 5 |
| 2C | 2nd cousin | 3 | 3 | 2 | 1/64 | 5 |

The `max_degree` parameter controls extraction depth (default 2, covering through 1st cousins). Degree 3-5 types require deeper matrix products and are only computed when requested. The registry is importable as `REL_REGISTRY` and `PAIR_KINSHIP` from `simace.core.pedigree_graph`.

### Inbreeding and exact kinship

By default, kinship values are computed from the `(up, down, n_ancestors)` formula, which assumes no inbreeding. When `estimate_inbreeding: true` is set in config, `PedigreeGraph` computes exact inbreeding coefficients and pairwise kinship using sparse matrix propagation:

1. **`compute_inbreeding()`** builds the kinship matrix `K` generation by generation using sparse products (`P_g @ K` for cross-generation, `K_cross @ P_g.T` for within-generation). The inbreeding coefficient `F_i = K[mother_i, father_i]` is extracted each generation. For non-consanguineous pedigrees (all `F = 0`), both functions short-circuit instantly.

2. **`compute_pair_kinship(pairs)`** looks up exact kinship for each extracted pair from the cached sparse `K` matrix. When inbreeding is present, kinship values deviate from the nominal formula by a factor of `(1 + F_a)` where `F_a` is the inbreeding coefficient of the common ancestor.

| Pedigree | `compute_inbreeding` | `compute_pair_kinship` | Total |
|----------|---------------------:|-----------------------:|------:|
| N=10K, 6 gens (60K individuals) | 12.9s | 2.3s | 15.2s |
| N=100K, 4 gens (400K individuals) | 11.9s | 0.9s | 12.7s |

Cost is dominated by the sparse `P_g @ K` products, which scale with the number of nonzero kinship entries (i.e., the number of related pairs in the pedigree). Fewer generations means sparser `K` and faster computation, even at larger `N`.

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or Miniforge)
- Python 3.10+
- Linux or macOS (Windows may try [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install))

## Setup

```bash
git clone <repo-url>
cd ACE
conda env create -f environment.yml   # creates environment, installs all dependencies + both packages
conda activate ACE
```

### Verify installation

```bash
pytest tests/           # unit tests, should complete in ~1s
```

## Quick start

Run the smallest scenario to confirm everything works (takes a minute or two):

```bash
snakemake --cores 4 results/test/small_test/scenario.done
```

Check the output:

```bash
ls results/test/small_test/rep1/    # pedigree.parquet, phenotype files, validation, stats
cat logs/test/small_test/rep1/simulate.log
```

## Snakemake usage

Use `--cores N` where N is the number of parallel jobs.

```bash
# Run everything (default target — 16 scenarios, ~60-90 min)
snakemake --cores 4

# Run a single scenario (all stages)
snakemake --cores 4 results/base/baseline10K/scenario.done

# Run a single stage for one scenario
snakemake --cores 4 results/base/baseline10K/simulate.done   # pedigree simulation only
snakemake --cores 4 results/base/baseline10K/phenotype.done  # simulation + phenotyping
snakemake --cores 4 results/base/baseline10K/validate.done   # simulation + validation
snakemake --cores 4 results/base/baseline10K/stats.done      # phenotyping + stats + plots

# EPIMIGHT heritability for one scenario (see fit_ace/README.md)
snakemake --cores 4 results/base/baseline10K/epimight.done

# Dry run to see what will be executed
snakemake -n --cores 4
```

In snakemake, if a run is interrupted or fails, re-running the same command resumes from where it left off — completed steps are skipped automatically.

## Configuration

Define named scenarios in `config/{folder}.yaml`; defaults live in `config/_default.yaml` and are inherited unless overridden:

```yaml
defaults:
  seed: 42
  replicates: 3
  folder: base                              # Name of folder where results are stored

  # Trait 1 variance components (A + C <= 1.0; E = 1 - A - C)
  A1: 0.5                                   # Additive genetic variance
  C1: 0.2                                   # Common environment variance

  # Trait 2 variance components
  A2: 0.5
  C2: 0.2

  # Cross-trait correlations
  rA: 0.3                                   # Genetic correlation
  rC: 0.5                                   # Common environment correlation

  # Population and generation structure
  N: 100000                                 # Population size per generation
  G_ped: 6                                  # Generations to record in pedigree
  G_pheno: 3                                # Generations to phenotype (last G_pheno of G_ped)
  G_sim: 8                                  # Total generations (G_sim - G_ped = burn-in)
  standardize: true                          # Standardize liability before phenotyping

  # Pedigree structure + variance components
  pedigree:
    mating_lambda: 0.5                       # ZTP mating count lambda (~23% multi-partner)
    p_mztwin: 0.02                           # Probability of MZ twin birth
    assort1: 0                               # Mate correlation on trait 1 liability
    assort2: 0                               # Mate correlation on trait 2 liability
    trait1: {A: 0.5, C: 0.0, E: 0.5}        # Trait 1 ACE variance components
    trait2: {A: 0.4, C: 0.2, E: 0.4}        # Trait 2 ACE variance components
    rA: 0.0                                  # Cross-trait additive genetic correlation
    rC: 0.0                                  # Cross-trait shared environment correlation

  # Phenotype model per trait: frailty / cure_frailty / adult / first_passage
  phenotype:
    trait1:
      model: frailty                         # Model family
      params:                                # Model-specific parameters
        distribution: weibull                # Baseline hazard distribution
        scale: 2160                          # Weibull scale
        rho: 0.8                             # Weibull shape (<1 = decreasing hazard)
      beta: 1.0                              # Effect of liability on log-hazard
      prevalence: 0.10                       # Proportion affected per generation
    trait2:
      model: frailty
      params: {distribution: weibull, scale: 333, rho: 1.2}
      beta: 1.5
      prevalence: 0.20

  # Censoring + competing-risk mortality
  censoring:
    max_age: 80                              # Max censoring age
    gen_censoring:                           # Per-generation [left, right] observation windows
      { 0: [80, 80], 1: [80, 80], 2: [80, 80], 3: [40, 80], 4: [0, 80], 5: [0, 45] }
    death_scale: 164                         # Competing-risk mortality Weibull scale
    death_rho: 2.73                          # Competing-risk mortality Weibull shape

  # Sampling
  sampling:
    N_sample: 0                              # Subsample phenotype before stats (0 = keep all)
    case_ascertainment_ratio: 1              # Case sampling weight relative to controls
    pedigree_dropout_rate: 0                 # Fraction of individuals to drop from pedigree

  # Plot output
  plot_format: png                           # Plot file format: png or pdf

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

To add new simulations, simply add a new scenario to the config file. 

Prevalence can also be specified per-generation as a dict (e.g. `prevalence1: { 2: 0.03, 3: 0.05, 4: 0.08, 5: 0.12 }`).

## Outputs

### Simulation Data

Each scenario replicate produces (see [OUTPUTS.md](OUTPUTS.md) for column schemas and YAML structures):

| File | Description |
|------|-------------|
| `results/{folder}/{scenario}/rep{N}/pedigree.full.parquet` | Full pedigree before dropout; **temp** — auto-deleted after dropout and validation |
| `results/{folder}/{scenario}/rep{N}/pedigree.parquet` | Pedigree after dropout (identical to full when `pedigree_dropout_rate=0`) |
| `results/{folder}/{scenario}/rep{N}/phenotype.raw.parquet` | Raw time-to-event phenotypes (before censoring); **temp** — auto-deleted after censoring |
| `results/{folder}/{scenario}/rep{N}/phenotype.parquet` | Censored time-to-event phenotypes (age-at-onset, censoring, affected status) |
| `results/{folder}/{scenario}/rep{N}/phenotype.sampled.parquet` | Subsampled phenotype for stats (N_sample individuals); **temp** — auto-deleted after stats |
| `results/{folder}/{scenario}/rep{N}/phenotype.simple_ltm.parquet` | Liability-threshold binary affected status |
| `results/{folder}/{scenario}/rep{N}/phenotype.simple_ltm.sampled.parquet` | Subsampled threshold phenotype for stats; **temp** — auto-deleted after stats |
| `results/{folder}/{scenario}/rep{N}/params.yaml` | Simulation parameters for this replicate |
| `results/{folder}/{scenario}/rep{N}/phenotype_stats.yaml` | Phenotype statistics (correlations, prevalence, CIF, etc.) |
| `results/{folder}/{scenario}/rep{N}/simple_ltm_stats.yaml` | Threshold phenotype statistics |

### Validation and Logs

| File | Description |
|------|-------------|
| `results/{folder}/{scenario}/rep{N}/validation.yaml` | Structural and statistical validation results |
| `results/{folder}/validation_summary.tsv` | Aggregated validation metrics across scenarios |
| `results/{folder}/plots/` | Cross-scenario validation and phenotype plots |
| `logs/{folder}/{scenario}/` | Log files |
| `benchmarks/{folder}/{scenario}/` | Runtime and memory benchmarks |

### Plot Atlases

Multi-page PDF atlases collect all figures for a scenario or folder into a single document with figure captions:

| File | Description |
|------|-------------|
| `results/{folder}/{scenario}/plots/atlas.pdf` | Per-scenario atlas: liability structure, phenotype, censoring, correlations, heritability, and simple LTM figures |
| `results/{folder}/plots/atlas.pdf` | Per-folder atlas: cross-scenario validation plots (variance components, correlations, heritability, bias, runtime, memory) |
| `results/{folder}/{scenario}/rep{N}/epimight/plots/atlas.pdf` | EPIMIGHT atlas (see [fit_ace/README.md](fit_ace/README.md)) |

### Output Format Reference

See [OUTPUTS.md](OUTPUTS.md) for complete documentation of all output formats, including parquet column schemas, YAML file structures, validation_summary.tsv columns, benchmark format, and plot inventories.

## Subsampling and Dropout

### Subsampling (`N_sample`)

When `N_sample > 0`, the pipeline randomly draws `N_sample` individuals from the phenotype before computing statistics. This reduces runtime and disk usage for large populations while preserving population-level signals. The sampling step (`sample.smk`) writes a temporary `.sampled.parquet` that is auto-deleted after stats complete.

Because sampling breaks pedigree completeness — parents and other relatives may not be in the sample — the relationship extraction code in `PedigreeGraph` uses two strategies to recover as many valid pairs as possible:

| Relationship type | How it works with subsampled data |
|---|---|
| **Siblings** (full, maternal HS, paternal HS) | Classified using **original pedigree parent IDs** stored in the DataFrame columns, not row indices. Two sampled individuals are detected as siblings if their `mother`/`father` columns match, regardless of whether those parents are in the sample. Full sibs share both parent IDs; half-sibs share one and differ on the other. |
| **Parent-offspring** | Detected when a parent is present in the sample (its ID maps to a valid row index). Each parent link is independent — a child with only its mother in the sample still yields a mother-offspring pair. |
| **Grandparent-grandchild, avuncular, cousins, 2nd cousins** | Detected via sparse matrix products on parent→child edges. Each edge is built independently (mother edges and father edges are separate matrices), so a child with only one parent in the sample still contributes edges through that parent. However, these relationships require intermediate ancestors to be in the sample to form multi-hop paths. |
| **MZ twin** | Detected when both twins are in the sample (twin partner ID maps to a valid row index). |

### Case Ascertainment Bias (`case_ascertainment_ratio`)

When `case_ascertainment_ratio != 1` and `N_sample > 0`, sampling uses weighted probabilities instead of uniform random selection. Cases (`affected1 == True`) receive weight = ratio while controls receive weight = 1; weights are normalized to probabilities and passed to `rng.choice(p=..., replace=False)`.

For example, with 10% prevalence and `case_ascertainment_ratio: 5`, a case is 5x more likely to be drawn than a control, yielding ~36% cases in the sample versus the population 10%.

Edge cases:
- **ratio = 0**: only controls are sampled; `N_sample` is clamped to the number of available controls
- **ratio = 1** (default): uniform sampling (fast path, backward compatible)
- **0 cases or all cases**: falls back to uniform sampling with a warning
- **N_sample = 0**: ratio has no effect (all individuals pass through); logs a warning if ratio != 1
- **Extreme ratios**: warns if >90% of total cases would be expected in the sample

The ratio is recorded in per-rep stats YAML when != 1 but no correction is applied to downstream estimates — this is intentional, as the purpose is to study the bias.

### Pedigree Dropout (`pedigree_dropout_rate`)

When `pedigree_dropout_rate > 0`, the pipeline randomly removes that fraction of individuals from the simulated pedigree before phenotyping, modelling incomplete real-world observation. The dropout step (`dropout.smk`) runs between simulation and phenotyping:

```
simulate → pedigree.full.parquet (temp) → dropout → pedigree.parquet
                                      ↘ validate (reads full pedigree)
```

Dropped individuals are deleted entirely. All parent/twin links pointing to a dropped individual are set to -1 (unknown). This means multi-hop relationships through missing individuals (e.g. grandparent-grandchild via a removed parent) become undetectable, and former full-sib pairs whose shared parent was dropped are reclassified as half-sibs. Individuals with only one known parent can still participate in half-sib detection through the surviving parent.

Three pre-configured dropout scenarios are included: `baseline100K_dropout10` (10%), `baseline100K_dropout30` (30%), and `baseline100K_dropout50` (50%).

## Exporting to R

All simulation outputs are stored as parquet files. To convert them to tab-separated text files that R can read with `read.delim()` or `data.table::fread()`:

```bash
# Single file (writes .tsv.gz alongside the .parquet)
simace-parquet-to-tsv results/base/baseline10K/rep1/pedigree.parquet

# Multiple files at once
simace-parquet-to-tsv results/base/baseline10K/rep1/*.parquet

# Uncompressed .tsv instead of .tsv.gz
simace-parquet-to-tsv --no-gzip results/base/baseline10K/rep1/pedigree.parquet

# Control float precision (default: 4 decimal places)
simace-parquet-to-tsv -p 8 results/base/baseline10K/rep1/pedigree.parquet

# Custom output path
simace-parquet-to-tsv results/.../pedigree.parquet -o my_output.tsv.gz
```

Via Snakemake, request any `.tsv.gz` or `.tsv` file and the generic rule will convert the matching `.parquet` automatically:

```bash
snakemake --cores 1 results/base/baseline10K/rep1/pedigree.tsv.gz
snakemake --cores 1 results/base/baseline10K/rep1/phenotype.tsv   # uncompressed
```

Float precision for the Snakemake rule defaults to 4 and can be set globally via `tsv_float_precision` in `config/_default.yaml` defaults.

## Project Structure

```
ACE/
├── Snakefile                          # Root entry point (no -s flag needed)
├── config/
│   ├── _default.yaml                 # Default simulation parameters
│   └── {folder}.yaml                 # Per-folder scenario definitions
│
├── simace/                           # Simulation package (pip install -e .)
│   ├── __init__.py                    # setup_logging(), _snakemake_tag()
│   ├── core/                          # Shared infrastructure
│   │   ├── utils.py                   # save_parquet, safe_corrcoef, yaml_loader, PAIR_TYPES, etc.
│   │   ├── cli_base.py               # Shared CLI boilerplate (add_logging_args, init_logging)
│   │   ├── pedigree_graph.py          # Sparse-matrix pedigree relationship extraction
│   │   ├── compute_hazard_terms.py    # Baseline hazard functions (Weibull, Gompertz, etc.)
│   │   ├── weibull_mle.py             # Gauss-Hermite MLE for Weibull frailty correlation
│   │   └── _numba_utils.py            # Shared Numba kernels
│   ├── simulation/
│   │   └── simulate.py                # Pedigree simulation (mating, reproduce, run_simulation)
│   ├── phenotyping/
│   │   ├── phenotype.py               # Frailty, cure-frailty, ADuLT phenotype models
│   │   └── threshold.py               # Liability-threshold binary phenotype
│   ├── censoring/
│   │   └── censor.py                  # Age-window and competing-risk death censoring
│   ├── sampling/
│   │   ├── dropout.py                 # Pedigree dropout (random individual removal)
│   │   └── sample.py                  # Subsampling with case ascertainment bias
│   ├── analysis/
│   │   ├── stats.py                   # Tetrachoric correlations, heritability, pair counts
│   │   ├── simple_ltm_stats.py        # Simple LTM phenotype statistics
│   │   ├── validate.py                # Structural + statistical validation
│   │   ├── ltm_falconer.py            # Falconer h² from tetrachoric correlations
│   │   ├── survival_stats.py          # Pairwise frailty correlation wrappers
│   │   └── gather.py                  # Gather validation results into TSV
│   └── plotting/
│       ├── plot_utils.py              # Shared plotting helpers (finalize_plot, violin, heatmap)
│       ├── plot_phenotype.py          # Phenotype plot orchestrator + CLI
│       ├── plot_distributions.py      # Mortality, age-at-onset, cumulative incidence
│       ├── plot_liability.py          # Joint liability, violin, affection plots
│       ├── plot_correlations.py       # Tetrachoric + parent-offspring correlations
│       ├── plot_pedigree_counts.py    # Pedigree relationship pair counts diagram
│       ├── plot_simple_ltm.py         # Simple LTM phenotype plots
│       ├── plot_validation.py         # Validation summary plots
│       ├── plot_atlas.py              # Multi-page PDF atlas with figure captions
│       ├── plot_pipeline.py           # Pipeline DAG diagram
│       └── plot_table1.py             # Epidemiological Table 1
│
├── fit_ace/                           # Model fitting package — see fit_ace/README.md
│
├── workflow/
│   ├── common.py                      # Shared helpers (get_param, get_folder, etc.)
│   ├── rules/                         # Modular Snakemake rule files
│   │   ├── targets.smk                # Target rules: all, scenario, per-stage sentinels
│   │   ├── simulate.smk, dropout.smk  # Pedigree simulation and dropout
│   │   ├── phenotype.smk, sample.smk  # Phenotyping and sampling
│   │   ├── validate.smk, stats.smk    # Validation and statistics
│   │   ├── epimight.smk, pafgrs.smk   # fit_ace pipeline rules
│   │   └── epimight_bias.smk          # EPIMIGHT bias analysis
│   └── scripts/                       # Snakemake script wrappers
├── tests/                             # Mirrors simace/fit_ace sub-package structure
├── external/                          # Reference implementations (gitignored)
├── results/{folder}/{scenario}/       # Per-scenario simulation outputs
├── logs/{folder}/{scenario}/          # Log files
└── benchmarks/{folder}/{scenario}/    # Runtime and memory benchmarks
```

## Documentation (under construction)

- **[OUTPUTS.md](OUTPUTS.md)** — Output format reference (parquet schemas, YAML structures, TSV columns, plots)
- **[methods.md](methods.md)** — Methods document (variance decomposition, Weibull frailty, censoring, liability threshold, tetrachoric correlation, heritability estimation, etc)
- **[distributions.md](distributions.md)** — Phenotype model reference (frailty hazard distributions, ADuLT LTM, ADuLT Cox)
- **[fit_ace/README.md](fit_ace/README.md)** — Model fitting: EPIMIGHT heritability, PA-FGRS, Stan models
- **[CLAUDE.md](CLAUDE.md)** — Architecture guide

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'simace'` | Run `conda activate ACE` first — the package is only available inside the conda environment |
| `FileNotFoundError: config/_default.yaml` | Run snakemake from the ACE repo root directory |
| Simulation killed or frozen (large N) | Reduce `--cores` to lower parallel memory usage, or skip N=1M/2M scenarios |
| `IncompleteFilesException` on re-run | Snakemake detected a previously interrupted output; run `snakemake --cores 4 --rerun-incomplete` |

## License

MIT
