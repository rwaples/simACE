# ACE - Simulate registry-scale age-of-onset phenotypes with realistic pedigrees and the ACE liability model. 

simACE simulates millions of indiviudals in multi-generational pedigrees with heritable ACE variance components for two correlated traits.

## Overview
The project contains two installable Python packages:
- **sim_ace** -- Simulation, phenotyping, censoring, sampling, analysis, and plotting
- **fit_ace** -- Statistical model fitting: EPIMIGHT heritability, PA-FGRS genetic risk scores, Weibull frailty correlation, and Stan-based models

Both packages are orchestrated by a single Snakemake workflow. `fit_ace` depends on `sim_ace` for shared infrastructure (pedigree graphs, hazard functions, plotting utilities).

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




## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or Miniforge)
- Python 3.10+
- Linux or macOS (Windows may try [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install))

## Setup

```bash
git clone <repo-url>
cd ACE
conda env create -f environment.yml   # creates environment and installs sim_ace package
conda activate ACE
pip install -e .                       # install sim_ace
pip install -e fit_ace/                # install fit_ace
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

# Run individual stages
snakemake --cores 4 simulate_all     # pedigree simulation only
snakemake --cores 4 phenotype_all    # simulation + phenotyping
snakemake --cores 4 validate_all     # simulation + validation + folder summaries
snakemake --cores 4 stats_all        # phenotyping + stats + plots
snakemake --cores 4 epimight_all     # EPIMIGHT heritability analysis

# Run a single named scenario
snakemake --cores 4 results/base/baseline10K/scenario.done

# Dry run to see what will be executed
snakemake -n --cores 4
```

In snakemake, if a run is interrupted or fails, re-running the same command resumes from where it left off — completed steps are skipped automatically.

## Configuration

Define named scenarios in `config/config.yaml`. Defaults are inherited unless overridden:

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

  # Population and reproduction
  N: 100000                                 # Population size per generation
  G_ped: 6                                  # Generations to record in pedigree
  G_pheno: 3                                # Generations to phenotype (last G_pheno of G_ped)
  G_sim: 8                                  # Total generations (G_sim - G_ped = burn-in)
  mating_lambda: 0.5                         # ZTP mating count lambda (~23% multi-partner)
  p_mztwin: 0.02                            # Probability of MZ twin birth
  assort1: 0                                 # Mate correlation on trait 1 liability ([-1, 1])
  assort2: 0                                 # Mate correlation on trait 2 liability ([-1, 1])

  # Phenotype model per trait: weibull/exponential/gompertz/lognormal/loglogistic/gamma/cure_frailty/adult_ltm/adult_cox
  # Trait 1
  beta1: 1.0                                # Effect of liability on log-hazard
  phenotype_model1: weibull                 # Baseline hazard model
  phenotype_params1:
    scale: 2160                             # Weibull scale
    rho: 0.8                                # Weibull shape (<1 = decreasing hazard)

  # Trait 2
  beta2: 1.5
  phenotype_model2: weibull
  phenotype_params2:
    scale: 333
    rho: 1.2                                # Weibull shape (>1 = increasing hazard)

  standardize: true                          # Standardize liability before phenotyping

  # Censoring
  censor_age: 80                             # Max censoring age
  gen_censoring:                             # Per-generation [left, right] observation windows
    { 0: [80, 80], 1: [80, 80], 2: [80, 80], 3: [40, 80], 4: [0, 80], 5: [0, 45] }
  death_scale: 164                           # Competing-risk mortality Weibull scale
  death_rho: 2.73                            # Competing-risk mortality Weibull shape

  # Liability threshold model
  prevalence1: 0.10                          # Trait 1: proportion affected per generation
  prevalence2: 0.20                          # Trait 2: proportion affected per generation

  # Statistics
  N_sample: 0                                # Subsample phenotype before stats (0 = keep all)
  case_ascertainment_ratio: 1                # Case sampling weight relative to controls (1 = uniform)
  pedigree_dropout_rate: 0                   # Fraction of individuals to drop from pedigree (0 = none)

  # EPIMIGHT relationship kinds to analyze
  epimight_kinds: [PO, FS, HS, mHS, pHS]     # Relationship types for heritability estimation

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
| `results/{folder}/{scenario}/rep{N}/epimight/plots/atlas.pdf` | EPIMIGHT atlas: CIF curves, heritability, genetic correlation across relationship kinds |

### Output Format Reference

See [OUTPUTS.md](OUTPUTS.md) for complete documentation of all output formats, including parquet column schemas, YAML file structures, validation_summary.tsv columns, benchmark format, and plot inventories.

## EPIMIGHT

EPIMIGHT estimates heritability (h²) and genetic correlation from time-to-event data using the [EPIMIGHT](https://github.com/BioPsyk/epimight) R package. It compares cumulative incidence in relatives of affected individuals against the general population, then applies Falconer's formula to derive heritability estimates stratified by birth year with fixed/random effects meta-analysis.

### Setup

EPIMIGHT requires a separate R-based conda environment:

```bash
conda env create -f fit_ace/epimight/environment.yml
conda run -n epimight Rscript -e "install.packages('fit_ace/epimight/EPIMIGHT/epimight', repos=NULL, type='source')"
```

### Running via Snakemake

```bash
# All scenarios and replicates
snakemake --cores 4 epimight_all

# Single scenario (one replicate)
snakemake --cores 4 results/base/baseline100K/rep1/epimight/plots/atlas.pdf

# Single relationship kind
snakemake --cores 4 results/base/baseline100K/rep1/epimight/tsv/h2_d1_FS.tsv
```

Which relationship kinds are analyzed is controlled by the `epimight_kinds` config parameter (default: `[PO, FS, HS, mHS, pHS]`).

### Outputs

Each replicate's `epimight/` directory contains:

| File | Description |
|------|-------------|
| `trait1.epimight_in.parquet`, `trait2.epimight_in.parquet` | Time-to-event input data for traits 1 and 2 |
| `true_parameters.json` | True h² and genetic correlation from variance components |
| `results_{kind}.md` | Summary report per relationship kind (cohort sizes, h² meta-analysis, genetic correlation, true vs observed comparison) |
| `tsv/cif_*.tsv`, `tsv/h2_*.tsv`, `tsv/gc_*.tsv` | CIF curves, heritability estimates, and genetic correlation per kind |
| `plots/atlas.pdf` | Multi-page PDF atlas: CIF curves, h² over time, h² bar charts, genetic correlation, and observed vs true comparison |

See the [Outputs > Plot Atlases](#plot-atlases) table for the full output tree.

### More information

See [fit_ace/epimight/README.md](fit_ace/epimight/README.md) for the full pipeline reference — manual steps, column schemas, cohort definitions, and relationship kinds.

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
sim-ace-parquet-to-tsv results/base/baseline10K/rep1/pedigree.parquet

# Multiple files at once
sim-ace-parquet-to-tsv results/base/baseline10K/rep1/*.parquet

# Uncompressed .tsv instead of .tsv.gz
sim-ace-parquet-to-tsv --no-gzip results/base/baseline10K/rep1/pedigree.parquet

# Control float precision (default: 4 decimal places)
sim-ace-parquet-to-tsv -p 8 results/base/baseline10K/rep1/pedigree.parquet

# Custom output path
sim-ace-parquet-to-tsv results/.../pedigree.parquet -o my_output.tsv.gz
```

Via Snakemake, request any `.tsv.gz` or `.tsv` file and the generic rule will convert the matching `.parquet` automatically:

```bash
snakemake --cores 1 results/base/baseline10K/rep1/pedigree.tsv.gz
snakemake --cores 1 results/base/baseline10K/rep1/phenotype.tsv   # uncompressed
```

Float precision for the Snakemake rule defaults to 4 and can be set globally via `tsv_float_precision` in `config/config.yaml` defaults.

## Project Structure

```
ACE/
├── Snakefile                          # Root entry point (no -s flag needed)
├── config/
│   └── config.yaml                    # Simulation parameters (named scenarios)
│
├── sim_ace/                           # Simulation package (pip install -e .)
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
├── fit_ace/                           # Model fitting package (pip install -e fit_ace/)
│   ├── pafgrs/                        # PA-FGRS genetic risk scores
│   │   ├── pafgrs.py                  # Pearson-Aitken scoring (Bayesian posterior mean/variance)
│   │   └── pafgrs_metrics.py          # Validation metrics (r, R², AUC, calibration)
│   ├── epimight/                      # EPIMIGHT heritability analysis (separate conda env)
│   │   ├── create_parquet.py          # Convert phenotype to EPIMIGHT TTE format
│   │   ├── guide-yob.R               # CIF, h², genetic correlation (R)
│   │   ├── plot_epimight.py           # EPIMIGHT diagnostic atlas
│   │   ├── epimight_bias_analysis.py  # Bias quantification
│   │   ├── EPIMIGHT/                  # Vendored BioPsyk EPIMIGHT R package
│   │   └── environment.yml            # Conda env for R dependencies
│   ├── stan/                          # Stan-based model fitting
│   │   ├── fit_ace.py, fit_reml.py    # Python wrappers
│   │   └── *.stan                     # Stan model files
│   └── plotting/
│       ├── plot_pafgrs.py             # PA-FGRS diagnostic atlas
│       └── plot_epimight_bias.py      # EPIMIGHT bias analysis atlas
│
├── workflow/
│   ├── common.py                      # Shared helpers (get_param, get_folder, etc.)
│   ├── rules/                         # Modular Snakemake rule files
│   │   ├── targets.smk                # Target rules: all, simulate_all, phenotype_all, etc.
│   │   ├── simulate.smk, dropout.smk  # Pedigree simulation and dropout
│   │   ├── phenotype.smk, sample.smk  # Phenotyping and sampling
│   │   ├── validate.smk, stats.smk    # Validation and statistics
│   │   ├── epimight.smk               # EPIMIGHT heritability pipeline
│   │   ├── epimight_bias.smk          # EPIMIGHT bias analysis
│   │   └── pafgrs.smk                 # PA-FGRS scoring pipeline
│   └── scripts/                       # Snakemake script wrappers
├── tests/                             # Mirrors sim_ace/fit_ace sub-package structure
├── external/                          # Reference implementations (gitignored)
├── results/{folder}/{scenario}/       # Per-scenario simulation outputs
├── logs/{folder}/{scenario}/          # Log files
└── benchmarks/{folder}/{scenario}/    # Runtime and memory benchmarks
```

## Documentation (under construction)

- **[OUTPUTS.md](OUTPUTS.md)** — Output format reference (parquet schemas, YAML structures, TSV columns, plots)
- **[methods.md](methods.md)** — Methods document (variance decomposition, Weibull frailty, censoring, liability threshold, tetrachoric correlation, heritability estimation, etc)
- **[distributions.md](distributions.md)** — Phenotype model reference (frailty hazard distributions, ADuLT LTM, ADuLT Cox)
- **[fit_ace/epimight/README.md](fit_ace/epimight/README.md)** — EPIMIGHT heritability pipeline
- **[CLAUDE.md](CLAUDE.md)** — Architecture guide

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'sim_ace'` | Run `conda activate ACE` first — the package is only available inside the conda environment |
| `FileNotFoundError: config/config.yaml` | Run snakemake from the ACE repo root directory |
| Simulation killed or frozen (large N) | Reduce `--cores` to lower parallel memory usage, or skip N=1M/2M scenarios |
| `IncompleteFilesException` on re-run | Snakemake detected a previously interrupted output; run `snakemake --cores 4 --rerun-incomplete` |

## License

MIT
