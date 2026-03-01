# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACE is a computational population genetics research project for simulating multi-generational pedigrees with variance components:
- **A** (Additive genetic)
- **C** (Common/shared environment)
- **E** (unique Environment)

The simulation models inheritance patterns, family structures, and twin relationships across generations.

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ACE
```

## Running Simulations

```bash
# Run everything (default target)
snakemake --cores 4

# Run individual stages
snakemake --cores 4 simulate_all     # pedigree simulation only
snakemake --cores 4 phenotype_all    # simulation + phenotyping
snakemake --cores 4 validate_all     # simulation + validation + folder summaries
snakemake --cores 4 stats_all        # phenotyping + stats + plots

# Run a single scenario
snakemake --cores 4 results/base/baseline10K/scenario.done

# Dry run to see what will be executed
snakemake -n --cores 4
```

## Architecture

```
ACE/
├── Snakefile                          # Root entry point (no -s flag needed)
├── config/
│   └── config.yaml                    # Simulation parameters (named scenarios)
├── sim_ace/                           # Installable package (pip install -e .)
│   ├── __init__.py                    # setup_logging() + public API re-exports
│   ├── cli_base.py                    # Shared CLI boilerplate (add_logging_args, init_logging)
│   ├── utils.py                       # Shared helpers (safe_corrcoef, to_native, validation_result, etc.)
│   ├── simulate.py                    # Pedigree simulation (mating, reproduce, run_simulation)
│   ├── phenotype.py                   # Weibull frailty phenotype simulation
│   ├── censor.py                      # Age/death censoring
│   ├── threshold.py                   # Liability threshold model
│   ├── validate.py                    # Structural + statistical validation
│   ├── stats.py                       # Phenotype statistics (tetrachoric, correlations)
│   ├── pedigree_graph.py             # Sparse-matrix pedigree relationship extraction
│   ├── threshold_stats.py             # Threshold phenotype statistics
│   ├── survival_corr.py               # Pairwise Weibull survival correlations
│   ├── gather.py                      # Gather validation results into TSV
│   ├── plot_phenotype.py              # Phenotype plot orchestrator + CLI
│   ├── plot_distributions.py          # Mortality, age-at-onset, cumulative incidence plots
│   ├── plot_liability.py              # Joint liability, violin, affection plots
│   ├── plot_pedigree_counts.py        # Pedigree relationship pair counts diagram
│   ├── plot_correlations.py           # Tetrachoric + parent-offspring correlation plots
│   ├── plot_threshold.py              # Threshold phenotype plots
│   ├── plot_atlas.py                  # Multi-page PDF atlas with figure captions
│   └── plot_validation.py             # Validation summary plots
├── workflow/
│   ├── common.py                      # Shared helpers (get_param, get_folder, etc.)
│   ├── rules/                         # Modular rule files
│   │   ├── targets.smk                # Target rules: all, simulate_all, phenotype_all, etc.
│   │   ├── simulate.smk               # rule simulate_pedigree_liability
│   │   ├── phenotype.smk              # rules phenotype_weibull, phenotype_threshold
│   │   ├── validate.smk               # rules validate_pedigree_liability, gather_validation, plot_validation
│   │   └── stats.smk                  # rules stats_weibull, stats_threshold, plot_*
│   └── scripts/                       # Snakemake script wrappers
├── results/{folder}/{scenario}/rep{N}/  # Simulation output per scenario
│   ├── pedigree.parquet               # Pedigree data
│   └── params.yaml                    # Parameters used
├── results/{folder}/validation_summary.tsv  # Per-folder validation summary
├── results/{folder}/plots/            # Cross-scenario validation plots
├── logs/{folder}/{scenario}/          # Log files
└── benchmarks/{folder}/{scenario}/    # Runtime benchmarks
```

**Core Simulation Functions** (in `sim_ace/simulate.py`):
- `mating()` - Generates parent-offspring pairings with configurable family size, MZ twin rate, and non-social father proportion
- `reproduce()` - Simulates genetic and environmental inheritance; MZ twins share genotypes and sex
- `add_to_pedigree()` - Builds pedigree DataFrames tracking IDs, sex, parents, twin status, and A/C/E values
- `run_simulation()` - Orchestrates the full simulation loop

**Weibull frailty model fitting** has been extracted to a separate project at `~/Documents/fitACE/`.

**Configuration** (`config/config.yaml`):
Define named scenarios with parameters: seed, folder, A, C, N, G_ped, G_sim, fam_size, p_mztwin, p_nonsocial_father

## Generating methods.pdf

```bash
# Requires pandoc and tectonic (install via conda if missing)
conda install -y -c conda-forge pandoc tectonic

# Regenerate PDF from methods.md
pandoc methods.md -o methods.pdf --pdf-engine=tectonic -V geometry:margin=1in -V fontsize=11pt --standalone
```

## Dependencies

Key libraries: numpy, scipy, pandas, pyarrow, matplotlib, seaborn, snakemake (with SLURM executor for HPC)
