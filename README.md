# simACE

Simulate registry-scale age-of-onset phenotypes in multi-generational pedigrees under the ACE liability model.

simACE simulates millions of individuals in multi-generational pedigrees with
heritable ACE variance components for two correlated traits. It is designed
for evaluating and benchmarking statistical methods that estimate
heritability and familial correlations from population health registries.

Full documentation is in the [`docs/`](docs/) directory (built with mkdocs)
and on the [rendered site](https://rwaples.github.io/simACE/). Model fitting
(EPIMIGHT, PCGC, iterative/sparse REML, LDAK TetraHer, PA-FGRS, Stan, frailty) lives in
the private companion repo [`fitACE`](https://github.com/rwaples/fitACE),
which depends on simACE.

## Prerequisites

- Linux. Windows users may use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install). The pipeline is not supported on macOS, but the library install below works anywhere.
- `git` and `curl`

## Setup

simACE runs in a locked [pixi](https://pixi.sh) environment. One committed
lockfile means every install materializes the same environment:

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # single user-space binary
exec $SHELL                                     # pick up ~/.pixi/bin on PATH

git clone https://github.com/rwaples/simACE.git
cd simACE
pixi install --locked
```

To use simace as a library from your own environment instead, a plain
`pip install` works without pixi and on any OS. See
[installation](docs/getting-started/installation.md).

### Verify installation

```bash
pixi run pytest tests/
```

## Quick start

Run the smallest scenario to confirm everything works (takes a minute or two):

```bash
pixi run snakemake --cores 4 results/test/small_test/scenario.done
```

Check the output:

```bash
ls results/test/small_test/rep1/    # pedigree.parquet, trait files, report.yaml, params.yaml
cat logs/test/small_test/rep1/simulate.log
```

## Snakemake usage

Use `--cores N` where N is the number of parallel jobs. Always run from the
repo root. The root `Snakefile` is the entry point, so no `-s` flag is needed.

```bash
# Run everything (default target: all scenarios, all stages)
pixi run snakemake --cores 4

# Run a single scenario
pixi run snakemake --cores 4 results/base/baseline10K/scenario.done

# Dry run to see what will be executed
pixi run snakemake -n --cores 4
```

If a run is interrupted or fails, re-running the same command resumes from
where it left off. Snakemake skips completed steps.

For per-stage targets, force-rebuilding, and resuming interrupted runs, see
[Running the pipeline](docs/user-guide/running-the-pipeline.md).

## Configuration

Global defaults live in `config/_default.yaml` under a `defaults:` key
(seed, replicates, variance components, population structure, phenotype
models, censoring, ascertainment, …). Scenarios live in per-folder files
`config/{folder}.yaml`. Each file holds bare scenario dicts, one top-level
key per scenario, each overriding only the defaults it changes. The results
folder name comes from the filename:

```yaml
# config/heritability.yaml → outputs under results/heritability/{scenario}/
high_heritability:
  seed: 4042
  pedigree:
    trait1: {A: 0.8, C: 0.0, E: 0.2}        # A + C + E = 1 per trait
    trait2: {A: 0.8, C: 0.0, E: 0.2}

baseline_small:
  seed: 1042
  N: 10000                                  # Population size per generation
```

Key defaults you will most often override: `pedigree.trait{1,2}.{A,C,E}`
(variance components), `pedigree.rA`/`pedigree.rC` (cross-trait
correlations), `N`, `G_ped`/`G_pheno`/`G_sim` (generations recorded /
phenotyped / simulated), `seed`, `replicates`. The loader still accepts the
older flat keys (`A1`, `C1`, …) for compatibility.

To add new simulations, add a scenario to an existing folder file or create
a new `config/{folder}.yaml`. Files are auto-discovered, and names starting
with `_` are skipped. For the
full parameter reference (phenotype models, censoring, ascertainment, etc.), see
[Configuration](docs/user-guide/configuration.md).

## Outputs

Each scenario replicate produces the full and post-ascertainment pedigree
parquets, outcomes-only censored time-to-event trait parquets, a curated
`report.yaml` with its `plot_payload.yaml` companion, and a browsable HTML
plot atlas (PDF export on demand). See
[Output structure](docs/user-guide/output-structure.md) for the complete file
inventory, parquet column schemas, YAML structures, and plot listings.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'simace'` | Run commands through `pixi run …` from the repo root |
| `FileNotFoundError: config/_default.yaml` | Run snakemake from the simACE repo root directory |
| Simulation killed or frozen (large N) | Reduce `--cores` to lower parallel memory usage, or skip large-N scenarios |
| `IncompleteFilesException` on re-run | Snakemake detected a previously interrupted output; run `pixi run snakemake --cores 4 --rerun-incomplete` |

## License

MIT
