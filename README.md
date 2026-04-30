# simACE — Simulate registry-scale age-of-onset phenotypes with realistic pedigrees and the ACE liability model

simACE simulates millions of individuals in multi-generational pedigrees with
heritable ACE variance components for two correlated traits. It is designed
for evaluating and benchmarking statistical methods that estimate
heritability and familial correlations from population health registries.

📖 **Full documentation**: see the [`docs/`](docs/) directory (built with mkdocs)
or the rendered site. Model fitting (EPIMIGHT, PA-FGRS, sparseREML, iter_reml,
Stan, PCGC) lives in the sister repo [`fitACE`](https://github.com/rwaples/fitACE),
which depends on simACE.

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (Miniconda or Miniforge)
- Python 3.10+
- Linux or macOS (Windows users may try [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install))

## Setup

```bash
git clone <repo-url>
cd simACE
conda env create -f envs/environment.yml   # creates env, installs dependencies + simace
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

Use `--cores N` where N is the number of parallel jobs. Always run from the
repo root — the root `Snakefile` is the entry point (no `-s` flag).

```bash
# Run everything (default target — all scenarios, all stages)
snakemake --cores 4

# Run a single scenario
snakemake --cores 4 results/base/baseline10K/scenario.done

# Dry run to see what will be executed
snakemake -n --cores 4
```

If a run is interrupted or fails, re-running the same command resumes from
where it left off — completed steps are skipped automatically.

For per-stage targets, force-rebuilding, and resuming interrupted runs, see
[Running the Pipeline](docs/user-guide/running-the-pipeline.md).

## Configuration

Define named scenarios in `config/{folder}.yaml`; defaults live in
`config/_default.yaml` and are inherited unless overridden:

```yaml
defaults:
  seed: 42
  replicates: 3
  folder: base                              # Output folder under results/

  # Trait 1 variance components (A + C <= 1.0; E = 1 - A - C)
  A1: 0.5
  C1: 0.2

  # Trait 2 variance components
  A2: 0.5
  C2: 0.2

  # Cross-trait correlations
  rA: 0.3                                   # Genetic correlation
  rC: 0.5                                   # Common environment correlation

  # Population and generation structure
  N: 100000                                 # Population size per generation
  G_ped: 6                                  # Generations recorded in pedigree
  G_pheno: 3                                # Generations to phenotype
  G_sim: 8                                  # Total generations (G_sim - G_ped = burn-in)

scenarios:
  baseline10K:
    seed: 1042
    N: 10000

  high_heritability:
    folder: heritability
    A1: 0.8
    C1: 0.0
    A2: 0.8
    C2: 0.0
```

To add new simulations, simply add a new scenario to a config file. For the
full parameter reference (phenotype models, censoring, sampling, etc.), see
[Configuration](docs/user-guide/configuration.md).

## Outputs

Each scenario replicate produces a pedigree parquet, censored time-to-event
phenotypes, a liability-threshold binary phenotype, per-replicate stats and
validation YAMLs, and a multi-page PDF plot atlas. See
[Output Structure](docs/user-guide/output-structure.md) for the complete file
inventory, parquet column schemas, YAML structures, and plot listings.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'simace'` | Run `conda activate ACE` first — the package is only available inside the conda environment |
| `FileNotFoundError: config/_default.yaml` | Run snakemake from the simACE repo root directory |
| Simulation killed or frozen (large N) | Reduce `--cores` to lower parallel memory usage, or skip large-N scenarios |
| `IncompleteFilesException` on re-run | Snakemake detected a previously interrupted output; run `snakemake --cores 4 --rerun-incomplete` |

## License

MIT
