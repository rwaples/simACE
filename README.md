# ACE - Population Genetics Simulation

Simulates multi-generational pedigrees with ACE variance components for two correlated traits:
- **A** - Additive genetic
- **C** - Common/shared environment
- **E** - Unique environment

Continuous liabilities are mapped to observable phenotypes via one of three time-to-event models:
- **Frailty** — Proportional hazards frailty model with pluggable baseline hazard (Weibull, Gompertz, lognormal, etc.)
- **ADuLT LTM** — Deterministic liability threshold model with logistic CIP age-of-onset mapping (Pedersen et al., Nat Commun 2023)
- **ADuLT Cox** — Proportional hazards model with Weibull noise and rank-based CIP-to-age mapping (Pedersen et al., 2023)

A separate liability-threshold model produces binary affection status. The pipeline includes automated structural and statistical validation, phenotype statistics, and plotting. See [distributions.md](distributions.md) for model details.

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
```

### EPIMIGHT (optional)

To run heritability analysis via EPIMIGHT, create its separate R-based conda environment:

```bash
conda env create -f epimight/environment.yml
conda run -n epimight Rscript -e "install.packages('epimight/EPIMIGHT/epimight', repos=NULL, type='source')"
```

See [epimight/README.md](epimight/README.md) for full pipeline details.

### Verify installation

```bash
pytest tests/           # unit tests, should complete in ~1s
```

## Quick start

Run the smallest scenario to confirm everything works (takes a few seconds):

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
  fam_size: 2.3                             # Mean family size (Poisson)
  p_mztwin: 0.02                            # Probability of MZ twin birth
  p_nonsocial_father: 0.05                  # Probability of non-social paternity

  # Phenotype model per trait: weibull/exponential/gompertz/lognormal/loglogistic/gamma/adult_ltm/adult_cox
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
  extra_tetrachoric: false                   # Estimate additional tetrachoric correlations (slow; set true to enable) [UNDER DEVELOPEMENT]

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

Each scenario replicate produces:

| File | Description |
|------|-------------|
| `results/{folder}/{scenario}/rep{N}/pedigree.parquet` | Pedigree with id, sex, parents, twin, household, A/C/E values, liabilities |
| `results/{folder}/{scenario}/rep{N}/phenotype.parquet` | Time-to-event phenotypes (age-at-onset, censoring, affected status) |
| `results/{folder}/{scenario}/rep{N}/phenotype.liability_threshold.parquet` | Liability-threshold binary affected status |
| `results/{folder}/{scenario}/rep{N}/params.yaml` | Parameters used for this replicate |

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
| `results/{folder}/{scenario}/plots/atlas.pdf` | Per-scenario atlas: liability structure, Weibull phenotype, censoring, correlations, heritability, and threshold model figures |
| `results/{folder}/plots/atlas.pdf` | Per-folder atlas: cross-scenario validation plots (variance components, correlations, heritability, bias, runtime, memory) |
| `results/{folder}/{scenario}/rep{N}/epimight/plots/atlas.pdf` | EPIMIGHT atlas: CIF curves, heritability, genetic correlation across relationship kinds |

### Parquet Column Reference

#### pedigree.parquet

Core pedigree structure with latent variance components for two correlated traits.

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Unique individual identifier |
| `sex` | int8 | 0 = female, 1 = male |
| `mother` | int64 | Mother's id (-1 for founders) |
| `father` | int64 | Father's id (-1 for founders) |
| `twin` | int64 | MZ twin partner's id (-1 if not a twin) |
| `generation` | int8 | Generation number (0 = oldest recorded) |
| `household_id` | int64 | Shared-environment household group |
| `A1`, `A2` | float32 | Additive genetic component (traits 1 and 2) |
| `C1`, `C2` | float32 | Common/shared environment component |
| `E1`, `E2` | float32 | Unique environment component |
| `liability1`, `liability2` | float32 | Total liability (A + C + E) |

#### phenotype.parquet

Extends the pedigree with time-to-event phenotypes and censoring. Includes all pedigree columns above, plus:

| Column | Type | Description |
|--------|------|-------------|
| `t1`, `t2` | float32 | Raw (uncensored) age-at-onset from the phenotype model |
| `death_age` | float32 | Age at death from competing-risk mortality |
| `t_observed1`, `t_observed2` | float32 | Observed age-at-onset after age and death censoring |
| `age_censored1`, `age_censored2` | bool | True if onset falls outside the generation's observation window |
| `death_censored1`, `death_censored2` | bool | True if onset occurs after death |
| `affected1`, `affected2` | bool | True if the individual is observed as affected (not age- or death-censored) |

#### phenotype.liability_threshold.parquet

Binary affected status from a liability-threshold model. Each generation has an independent prevalence-based threshold.

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Individual identifier |
| `generation` | int8 | Generation number |
| `mother`, `father`, `twin` | int64 | Family links (same as pedigree) |
| `A1`, `C1`, `E1`, `liability1` | float32 | Trait 1 variance components and liability |
| `A2`, `C2`, `E2`, `liability2` | float32 | Trait 2 variance components and liability |
| `affected1`, `affected2` | bool | True if liability exceeds the generation-specific threshold |

## Project Structure

```
ACE/
├── Snakefile                          # Root entry point (no -s flag needed)
├── config/
│   └── config.yaml                    # Simulation parameters (named scenarios)
├── sim_ace/                           # Installable package (pip install -e .)
│   ├── __init__.py                    # setup_logging() + public API re-exports
│   ├── cli_base.py                    # Shared CLI boilerplate (add_logging_args, init_logging)
│   ├── utils.py                       # Shared helpers (save_parquet, optimize_dtypes, safe_corrcoef, etc.)
│   ├── simulate.py                    # Pedigree simulation (mating, reproduce, run_simulation)
│   ├── phenotype.py                   # Phenotype models (frailty, adult_ltm, adult_cox)
│   ├── censor.py                      # Age-window and competing-risk death censoring
│   ├── threshold.py                   # Liability-threshold model
│   ├── validate.py                    # Structural + statistical validation
│   ├── stats.py                       # Tetrachoric correlations, relationship pairs
│   ├── pedigree_graph.py             # Sparse-matrix pedigree relationship extraction
│   ├── threshold_stats.py             # Threshold phenotype statistics
│   ├── survival_corr.py               # Pairwise Weibull survival correlation estimation
│   ├── gather.py                      # Gather validation results into TSV
│   ├── plot_phenotype.py              # Phenotype plot orchestrator + CLI
│   ├── plot_distributions.py          # Mortality, age-at-onset, cumulative incidence plots
│   ├── plot_liability.py              # Joint liability, violin, affection plots
│   ├── plot_pedigree_counts.py        # Pedigree relationship pair counts diagram
│   ├── plot_correlations.py           # Tetrachoric + parent-offspring correlation plots
│   ├── plot_threshold.py              # Threshold phenotype plots
│   ├── plot_atlas.py                  # Multi-page PDF atlas with figure captions
│   └── plot_validation.py             # Validation summary plots
├── epimight/                         # EPIMIGHT heritability analysis (separate conda env)
│   ├── create_parquet.py             # Convert phenotype → TTE format
│   ├── guide-yob.R                   # CIF, h², genetic correlation (R)
│   ├── plot_epimight.py              # Plot atlas generation
│   ├── environment.yml               # Conda env for R dependencies
│   └── README.md                     # EPIMIGHT pipeline documentation
├── workflow/
│   ├── common.py                      # Shared helpers (get_param, get_folder, etc.)
│   ├── rules/                         # Modular Snakemake rule files
│   │   ├── targets.smk                # Target rules: all, simulate_all, phenotype_all, etc.
│   │   ├── simulate.smk               # Pedigree simulation rule
│   │   ├── phenotype.smk              # Phenotyping rules (Weibull + threshold)
│   │   ├── validate.smk               # Validation, gathering, and validation plots
│   │   ├── stats.smk                  # Statistics and phenotype plots
│   │   └── epimight.smk               # EPIMIGHT heritability pipeline
│   └── scripts/                       # Snakemake script wrappers
├── tests/                             # Unit, statistical, and edge-case tests
├── results/{folder}/{scenario}/rep{N}/ # Per-replicate simulation outputs
├── logs/{folder}/{scenario}/          # Log files
└── benchmarks/{folder}/{scenario}/    # Runtime and memory usage benchmarks
```

## Documentation (under construction)

- **[methods.md](methods.md)** — Methods document (variance decomposition, Weibull frailty, censoring, liability threshold, tetrachoric correlation, heritability estimation, etc)
- **[distributions.md](distributions.md)** — Phenotype model reference (frailty hazard distributions, ADuLT LTM, ADuLT Cox)
- **[epimight/README.md](epimight/README.md)** — EPIMIGHT heritability pipeline
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
