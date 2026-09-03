# Running the pipeline

Run every command from the repository root. The root `Snakefile` is the
entry point, so do not pass `-s`.

## Choose a target

Every target is a file. Snakemake builds whatever that file depends on.

| Target | What it builds |
|---|---|
| no target | Every scenario in every `config/*.yaml` file, all stages |
| `results/{folder}/{scenario}/simulate.done` | The pedigree |
| `results/{folder}/{scenario}/phenotype.done` | The pedigree, phenotypes, censoring, and ascertainment. Writes `trait.parquet` |
| `results/{folder}/{scenario}/stats.done` | Everything above, plus the analyze stage's `report.yaml` and the scenario plots |
| `results/{folder}/{scenario}/validate.done` | Everything above, plus the folder's `report_summary.tsv` and validation atlas |
| `results/{folder}/{scenario}/scenario.done` | All stages for one scenario. Also builds the folder-level summary, which needs the sibling scenarios |
| `results/{folder}/folder.done` | Every scenario in one folder |

The stages run in this order: simulate, phenotype, censor, ascertainment,
analyze, plots. Each stage reads the files the previous one wrote.

The `epimight.done` target exists only when the fitACE_epimight repository
is checked out inside `fitACE/`. See the fitACE documentation.

## Preview the run

To see which jobs Snakemake would run without running them, add `-n`:

```bash
pixi run snakemake -n --cores 4 results/base/baseline10K/scenario.done
```

Before a run that takes more than a few minutes, preview it with `-n`.

## Run one scenario

```bash
pixi run snakemake --cores 4 results/base/baseline10K/scenario.done
```

To run several scenarios, use `--cores 8`. To debug a failing rule, use
`--cores 1`.

## Rebuild one output

To rebuild a file that already exists, pass `-f` with the file:

```bash
pixi run snakemake --cores 4 -f results/base/baseline10K/plots/atlas.html
```

The same command with `atlas.pdf` builds the PDF export.

## Resume an interrupted run

Snakemake tracks which outputs are complete. To continue after an
interruption, rerun the same command. If Snakemake reports
`IncompleteFilesException`, add `--rerun-incomplete`:

```bash
pixi run snakemake --cores 4 --rerun-incomplete results/base/baseline10K/scenario.done
```

## Convert parquet to TSV

To read a parquet file in R or a spreadsheet, convert it with
`simace-parquet-to-tsv`. It writes a `.tsv.gz` file next to each parquet file.

```bash
simace-parquet-to-tsv results/base/baseline10K/rep1/pedigree.parquet
simace-parquet-to-tsv results/base/baseline10K/rep1/*.parquet
```

For an uncompressed `.tsv`, pass `--no-gzip`. To write eight decimal places
instead of the default four, pass `-p 8`. Snakemake can also produce the
file:

```bash
pixi run snakemake --cores 1 results/base/baseline10K/rep1/pedigree.tsv.gz
```

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'simace'` | Run the command through `pixi run` from the repository root |
| `FileNotFoundError: config/_default.yaml` | Run the command from the repository root |
| `IncompleteFilesException` | Add `--rerun-incomplete` |
| A large-N simulation is killed or hangs | Lower `--cores` so fewer jobs share memory |
