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
conda env create -f pack.yml

# Activate environment
conda activate ACE
```

## Running Simulations

```bash
# Run everything (default target)
snakemake --cores 6

# Run individual stages
snakemake --cores 6 simulate_all     # pedigree simulation only
snakemake --cores 6 phenotype_all    # simulation + phenotyping
snakemake --cores 6 validate_all     # simulation + validation + folder summaries
snakemake --cores 6 stats_all        # phenotyping + stats + plots
snakemake --cores 6 analyze_all      # Weibull frailty model fitting

# Run a single scenario
snakemake --cores 6 results/base/baseline10K/scenario.done

# Dry run to see what will be executed
snakemake -n --cores 6
```

## Architecture

```
ACE/
├── Snakefile                          # Root entry point (no -s flag needed)
├── config/
│   └── config.yaml                    # Simulation parameters (named scenarios)
├── workflow/
│   ├── common.py                      # Shared helpers (get_param, get_folder, etc.)
│   ├── rules/                         # Modular rule files
│   │   ├── targets.smk                # Target rules: all, simulate_all, phenotype_all, etc.
│   │   ├── simulate.smk              # rule simulate
│   │   ├── phenotype.smk             # rules phenotype_weibull, phenotype_threshold
│   │   ├── validate.smk              # rules validate, gather_validation, plot_validation
│   │   ├── stats.smk                 # rules phenotype_stats, threshold_stats, plot_*
│   │   └── analyze.smk              # rules prepare_weibull, run_weibull
│   └── scripts/                       # Snakemake script wrappers
├── results/{folder}/{scenario}/rep{N}/  # Simulation output per scenario
│   ├── pedigree.parquet               # Pedigree data
│   └── params.yaml                    # Parameters used
├── results/{folder}/validation_summary.tsv  # Per-folder validation summary
├── results/{folder}/plots/            # Cross-scenario validation plots
├── results/analysis/{folder}/{scenario}/  # Weibull analysis output
├── logs/{folder}/{scenario}/          # Log files
└── benchmarks/{folder}/{scenario}/    # Runtime benchmarks
```

**Core Simulation Functions** (in `workflow/scripts/simulate.py`):
- `mating()` - Generates parent-offspring pairings with configurable family size, MZ twin rate, and non-social father proportion
- `reproduce()` - Simulates genetic and environmental inheritance; MZ twins share genotypes and sex
- `add_to_pedigree()` - Builds pedigree DataFrames tracking IDs, sex, parents, twin status, and A/C/E values
- `run_simulation()` - Orchestrates the full simulation loop

**Configuration** (`config/config.yaml`):
Define named scenarios with parameters: seed, folder, A, C, N, G_ped, G_sim, fam_size, p_mztwin, p_nonsocial_father

## Dependencies

Key libraries: numpy, scipy, pandas, tskit, msprime, snakemake (with SLURM executor for HPC)
