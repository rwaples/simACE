# ACE - Population Genetics Simulation

Simulates multi-generational pedigrees with ACE variance components:
- **A** - Additive genetic
- **C** - Common/shared environment
- **E** - Unique environment

## Setup

```bash
conda env create -f environment.yml
conda activate ACE
```

## Usage

```bash
# Run all scenarios
snakemake --cores 1 -s workflow/Snakefile

# Run specific scenario
snakemake --cores 1 -s workflow/Snakefile results/baseline/pedigree.parquet

# Dry run
snakemake -n -s workflow/Snakefile
```

## Configuration

Define named scenarios in `config/config.yaml`:

```yaml
scenarios:
  baseline:
    seed: 42
    A: 0.5                    # Additive genetic variance
    C: 0.2                    # Common environment variance
    N: 1000000                # Population size per generation
    ngen: 3                   # Number of generations
    fam_size: 2.7             # Mean family size
    p_mztwin: 0.02            # Proportion of MZ twins
    p_nonsocial_father: 0.15  # Proportion of non-social fathers
```

To add new simulations, add a new named scenario to the config file.

## Outputs

Each scenario produces:

| File | Description |
|------|-------------|
| `results/{scenario}/pedigree.parquet` | Pedigree data (id, sex, mother, father, twin, A, C, E) |
| `results/{scenario}/params.yaml` | Parameters used for this run |
| `logs/{scenario}/simulate.log` | Log file |
| `benchmarks/{scenario}/simulate.tsv` | Runtime and memory usage |

## Project Structure

```
ACE/
├── config/
│   └── config.yaml              # Simulation parameters
├── workflow/
│   ├── Snakefile                # Main workflow
│   └── scripts/
│       └── simulate.py          # Simulation functions
├── results/{scenario}/          # Outputs
├── logs/{scenario}/             # Logs
└── benchmarks/{scenario}/       # Benchmarks
```
