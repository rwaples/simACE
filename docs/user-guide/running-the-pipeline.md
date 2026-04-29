# Running the Pipeline

## Snakemake basics

All commands are issued from the repository root directory. The root
`Snakefile` is the entry point; the `-s` flag should not be used.

```bash
snakemake --cores 4    # 4 parallel jobs (use --cores 1 for debugging)
```

## Dry run

To preview what will be executed without running anything:

```bash
snakemake -n --cores 4
```

## Pipeline targets

| Target | What it runs |
|---|---|
| `snakemake --cores 4` | Everything (default -- all scenarios, all stages) |
| `results/{folder}/{scenario}/epimight.done` | EPIMIGHT heritability estimation |
| `results/{folder}/{scenario}/scenario.done` | All stages for one scenario |
| `results/{folder}/{scenario}/simulate.done` | Pedigree simulation only |
| `results/{folder}/{scenario}/phenotype.done` | Simulation + phenotyping |
| `results/{folder}/{scenario}/validate.done` | Simulation + validation + folder summaries |
| `results/{folder}/{scenario}/stats.done` | Phenotyping + stats + plots |

## Running a single scenario

```bash
snakemake --cores 4 results/base/baseline10K/scenario.done
```

The `scenario.done` sentinel file signals that all stages are complete for that scenario.

## Force rebuilding

Use `-f` to force-rebuild a specific output:

```bash
# Regenerate plots for a scenario
snakemake --cores 4 -f results/base/baseline10K/plots/atlas.pdf
```

## Pipeline stages

The pipeline runs stages in order, with each stage depending on the previous:

```
simulate -> dropout -> phenotype -> censor -> sample -> stats/validate -> plots
```

Snakemake tracks file dependencies automatically; re-running the same
command after an interrupted run resumes from where it stopped.

## Resuming interrupted runs

When Snakemake detects incomplete files from a previously interrupted
run:

```bash
snakemake --cores 4 --rerun-incomplete
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'simace'` | Run `conda activate ACE` first |
| `FileNotFoundError: config/_default.yaml` | Run snakemake from the ACE repo root directory |
| Simulation killed or frozen (large N) | Reduce `--cores` to lower parallel memory usage |
| `IncompleteFilesException` on re-run | Run with `--rerun-incomplete` |
