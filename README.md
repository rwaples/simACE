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

Install [pixi](https://pixi.sh), then install the locked simACE environment:

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
pixi run simace run small_test
```

Check the output:

```bash
ls results/test/small_test/rep1/    # pedigree.parquet, cohort.parquet, report.yaml, params.yaml, run.yaml, timing.tsv
cat logs/test/small_test/rep1/simulate.log
```

## Running scenarios

`simace run <scenario|folder>...` runs every replicate of each named
scenario, or of every scenario in a folder, through simulate, phenotype,
censor, ascertain, and analyze, then draws each scenario's plots and HTML
atlas. Run it from anywhere inside the repo.

```bash
# See the exact commands without running anything
pixi run simace run baseline10K --dry-run

# Run one scenario, three replicates at a time
pixi run simace run baseline10K --jobs 3

# Run every scenario in config/base.yaml through one pool of six workers
pixi run simace run base --jobs 6

# Summarize the folder's configured scenarios and draw the validation atlas
pixi run simace gather base

# List scenarios and which replicates are complete
pixi run simace ls base
```

A replicate is complete once its `run.yaml` exists and every output still
matches the size and time it recorded. Rerunning a scenario skips complete
replicates, recomputes interrupted ones (or ones with an output rewritten by
hand) from scratch, and refuses replicates whose config has changed since
they ran (`--force` recomputes them). Plots and the atlas are rebuilt on every run, and
`simace ls` says whether they are current. A summary per scenario names every failed or
refused replicate.

Each stage is also a subcommand that takes explicit file paths
(`pixi run simace simulate --help`). See
[Running the pipeline](docs/user-guide/running-the-pipeline.md).

## Configuration

Scenarios inherit defaults from `config/_default.yaml`. Follow
[Writing a scenario](docs/user-guide/writing-a-scenario.md) to add one, and use
[Configuration](docs/user-guide/configuration.md) to look up parameters and
their defaults.

## Outputs

Each scenario replicate produces two parquets, the recorded pedigree and the
cohort (the analysis pedigree's members with censored time-to-event outcomes
for the analysis sample), a curated
`report.yaml` with its `plot_payload.yaml` companion, and a browsable HTML
plot atlas (PDF export on demand). See
[Output structure](docs/user-guide/output-structure.md) for the complete file
inventory, parquet column schemas, YAML structures, and plot listings.

## Troubleshooting

See [Running the pipeline: troubleshooting](docs/user-guide/running-the-pipeline.md#troubleshooting)
for fixes to common errors.

## License

MIT
