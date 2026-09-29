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

## Setup

Install [pixi](https://pixi.sh) at the version `requires-pixi` in `pixi.toml`
allows (the [Installation](docs/getting-started/installation.md) page has the
pinned command), then install the locked simACE environment:

```bash
git clone https://github.com/rwaples/simACE.git
cd simACE
pixi install --locked
```

See [Installation](docs/getting-started/installation.md) for pixi setup,
supported platforms, development checks, and library installation.

## Quick start

Run the smallest scenario to confirm everything works:

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

Scenarios inherit defaults from `config/_default.yaml`. Follow
[Writing a scenario](docs/user-guide/writing-a-scenario.md) to add one, and use
[Configuration](docs/user-guide/configuration.md) to look up parameters and
their defaults.

## Outputs

Each scenario replicate produces the full and post-ascertainment pedigree
parquets, outcomes-only censored time-to-event trait parquets, a curated
`report.yaml` with its `plot_payload.yaml` companion, and a browsable HTML
plot atlas (PDF export on demand). See
[Output structure](docs/user-guide/output-structure.md) for the complete file
inventory, parquet column schemas, YAML structures, and plot listings.

## Troubleshooting

See [Running the pipeline: troubleshooting](docs/user-guide/running-the-pipeline.md#troubleshooting)
for fixes to common Snakemake and environment errors.

## License

MIT
