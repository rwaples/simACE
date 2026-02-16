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
# Run simulation pipeline (all scenarios)
snakemake --cores 1 -s workflow/simulate.smk

# Run analysis pipeline (all scenarios, requires simulation outputs)
snakemake --cores 1 -s workflow/analyze.smk

# Simulate a single scenario
snakemake --cores 1 -s workflow/simulate.smk results/baseline10K/scenario.done

# Analyze a single scenario
snakemake --cores 1 -s workflow/analyze.smk results/analysis/baseline10K/scenario.analyzed

# Dry run to see what will be executed
snakemake -n -s workflow/simulate.smk
snakemake -n -s workflow/analyze.smk
```

## Architecture

```
ACE/
├── config/
│   └── config.yaml              # Simulation parameters (named scenarios)
├── workflow/
│   ├── simulate.smk             # Simulation, validation & visualization pipeline
│   ├── analyze.smk              # Model fitting pipeline (Weibull)
│   └── scripts/
│       └── simulate.py          # Simulation functions
├── results/{scenario}/          # Simulation output per scenario
│   ├── pedigree.parquet         # Pedigree data
│   └── params.yaml              # Parameters used
├── results/analysis/{scenario}/ # Analysis output per scenario
├── logs/{scenario}/             # Log files
└── benchmarks/{scenario}/       # Runtime benchmarks
```

**Core Simulation Functions** (in `workflow/scripts/simulate.py`):
- `mating()` - Generates parent-offspring pairings with configurable family size, MZ twin rate, and non-social father proportion
- `reproduce()` - Simulates genetic and environmental inheritance; MZ twins share genotypes and sex
- `add_to_pedigree()` - Builds pedigree DataFrames tracking IDs, sex, parents, twin status, and A/C/E values
- `run_simulation()` - Orchestrates the full simulation loop

**Configuration** (`config/config.yaml`):
Define named scenarios with parameters: seed, A, C, N, G_ped, G_sim, fam_size, p_mztwin, p_nonsocial_father

## Dependencies

Key libraries: numpy, scipy, pandas, tskit, msprime, snakemake (with SLURM executor for HPC)
