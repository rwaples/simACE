# ACE - Population Genetics Simulation

Simulates multi-generational pedigrees with ACE variance components for two correlated traits:
- **A** - Additive genetic
- **C** - Common/shared environment
- **E** - Unique environment

Continuous liabilities are mapped to observable phenotypes via a Weibull proportional-hazards frailty model (time-to-event) or a liability-threshold model (binary affection). The pipeline includes automated structural and statistical validation, phenotype statistics, and publication-quality plots.

## Setup

```bash
conda env create -f environment.yml
conda activate ACE
pip install -e .
```

## Usage

```bash
# Run everything (default target)
snakemake --cores 6

# Run individual stages
snakemake --cores 6 simulate_all     # pedigree simulation only
snakemake --cores 6 phenotype_all    # simulation + phenotyping
snakemake --cores 6 validate_all     # simulation + validation + folder summaries
snakemake --cores 6 stats_all        # phenotyping + stats + plots

# Run a single scenario
snakemake --cores 6 results/base/baseline10K/scenario.done

# Dry run to see what will be executed
snakemake -n --cores 6
```

## Configuration

Define named scenarios in `config/config.yaml`. Defaults are inherited unless overridden:

```yaml
defaults:
  seed: 42
  replicates: 3
  folder: base
  A1: 0.5                     # Trait 1 additive genetic variance
  C1: 0.2                     # Trait 1 common environment variance
  A2: 0.5                     # Trait 2 additive genetic variance
  C2: 0.2                     # Trait 2 common environment variance
  rA: 0.3                     # Cross-trait genetic correlation
  rC: 0.5                     # Cross-trait common environment correlation
  N: 100000                   # Population size per generation
  G_ped: 4                    # Generations to record in pedigree
  G_sim: 6                    # Total generations (G_sim - G_ped = burn-in)
  fam_size: 2.3               # Mean family size (Poisson)
  p_mztwin: 0.02              # Probability of MZ twin birth
  p_nonsocial_father: 0.05    # Probability of non-social paternity

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

To add new simulations, add a named scenario to the config file.

## Outputs

Each scenario replicate produces:

| File | Description |
|------|-------------|
| `results/{folder}/{scenario}/rep{N}/pedigree.parquet` | Pedigree with id, sex, parents, twin, household, A/C/E values, liabilities |
| `results/{folder}/{scenario}/rep{N}/phenotype.weibull.parquet` | Weibull frailty phenotypes (age-at-onset, censoring, affection) |
| `results/{folder}/{scenario}/rep{N}/phenotype.liability_threshold.parquet` | Liability-threshold binary affection status |
| `results/{folder}/{scenario}/rep{N}/params.yaml` | Parameters used for this replicate |
| `results/{folder}/{scenario}/rep{N}/validation.yaml` | Structural and statistical validation results |
| `results/{folder}/validation_summary.tsv` | Aggregated validation metrics across scenarios |
| `results/{folder}/plots/` | Cross-scenario validation and phenotype plots |
| `logs/{folder}/{scenario}/` | Log files |
| `benchmarks/{folder}/{scenario}/` | Runtime and memory benchmarks |

## Project Structure

```
ACE/
├── Snakefile                          # Root entry point (no -s flag needed)
├── config/
│   └── config.yaml                    # Simulation parameters (named scenarios)
├── sim_ace/                           # Installable package (pip install -e .)
│   ├── simulate.py                    # Pedigree simulation (mating, reproduce, run_simulation)
│   ├── phenotype.py                   # Weibull frailty phenotype model
│   ├── censor.py                      # Age-window and competing-risk death censoring
│   ├── threshold.py                   # Liability-threshold model
│   ├── validate.py                    # Structural + statistical validation
│   ├── stats.py                       # Tetrachoric correlations, relationship pairs
│   ├── threshold_stats.py             # Threshold phenotype statistics
│   ├── survival_corr.py               # Pairwise Weibull survival correlation estimation
│   ├── gather.py                      # Gather validation results into TSV
│   ├── plot_phenotype.py              # Phenotype plot orchestrator
│   ├── plot_distributions.py          # Mortality, age-at-onset, cumulative incidence plots
│   ├── plot_liability.py              # Joint liability, violin, affection plots
│   ├── plot_correlations.py           # Tetrachoric + parent-offspring correlation plots
│   ├── plot_threshold.py              # Threshold phenotype plots
│   └── plot_validation.py             # Validation summary plots
├── workflow/
│   ├── common.py                      # Shared helpers (get_param, get_folder, etc.)
│   ├── rules/                         # Modular Snakemake rule files
│   │   ├── targets.smk                # Target rules: all, simulate_all, phenotype_all, etc.
│   │   ├── simulate.smk               # Pedigree simulation rule
│   │   ├── phenotype.smk              # Phenotyping rules (Weibull + threshold)
│   │   ├── validate.smk               # Validation, gathering, and validation plots
│   │   └── stats.smk                  # Statistics and phenotype plots
│   └── scripts/                       # Snakemake script wrappers
├── tests/                             # Unit, statistical, and edge-case tests
├── results/{folder}/{scenario}/rep{N}/ # Per-replicate outputs
├── logs/{folder}/{scenario}/          # Log files
└── benchmarks/{folder}/{scenario}/    # Runtime benchmarks
```

## Documentation

- **[methods.md](methods.md)** — Full mathematical description of all models (variance decomposition, Weibull frailty, censoring, liability threshold, tetrachoric correlation, pairwise survival correlation, heritability estimation)
- **[CLAUDE.md](CLAUDE.md)** — Developer reference and architecture guide

## License

MIT
