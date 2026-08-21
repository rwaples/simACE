# Running the Pipeline

## Snakemake basics

All commands are issued from the repository root directory. The root
`Snakefile` is the entry point; the `-s` flag should not be used.

```bash
pixi run snakemake --cores 4    # 4 parallel jobs (use --cores 1 for debugging)
```

## Dry run

To preview what will be executed without running anything:

```bash
pixi run snakemake -n --cores 4
```

## Pipeline targets

| Target | What it runs |
|---|---|
| `snakemake --cores 4` | Everything (default -- all scenarios, all stages) |
| `results/{folder}/{scenario}/epimight.done` | EPIMIGHT heritability estimation |
| `results/{folder}/{scenario}/scenario.done` | All stages for one scenario |
| `results/{folder}/{scenario}/simulate.done` | Pedigree simulation only |
| `results/{folder}/{scenario}/phenotype.done` | Simulation + phenotyping + ascertainment (produces `trait.parquet`) |
| `results/{folder}/{scenario}/validate.done` | Simulation + validation + folder summaries |
| `results/{folder}/{scenario}/stats.done` | Phenotyping + stats + plots |

## Running a single scenario

```bash
pixi run snakemake --cores 4 results/base/baseline10K/scenario.done
```

The `scenario.done` sentinel file signals that all stages are complete for that scenario.

## Force rebuilding

Use `-f` to force-rebuild a specific output:

```bash
# Regenerate a scenario's atlas (HTML is the default artifact)
pixi run snakemake --cores 4 -f results/base/baseline10K/plots/atlas.html

# The PDF atlas is an on-demand export
pixi run snakemake --cores 4 -f results/base/baseline10K/plots/atlas.pdf
```

## Pipeline stages

The pipeline runs stages in order, with each stage depending on the previous:

```
simulate -> phenotype -> censor -> ascertainment -> stats/validate -> plots
```

Snakemake tracks file dependencies automatically; re-running the same
command after an interrupted run resumes from where it stopped.

## Resuming interrupted runs

When Snakemake detects incomplete files from a previously interrupted
run:

```bash
pixi run snakemake --cores 4 --rerun-incomplete
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'simace'` | Run commands through `pixi run …` from the repo root |
| `FileNotFoundError: config/_default.yaml` | Run snakemake from the simACE repo root directory |
| Simulation killed or frozen (large N) | Reduce `--cores` to lower parallel memory usage |
| `IncompleteFilesException` on re-run | Run with `--rerun-incomplete` |
