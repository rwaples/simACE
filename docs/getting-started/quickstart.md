# Quick start

In this tutorial we run the smallest scenario end to end, look at the files it
writes, and open its plot atlas. It assumes you have finished the
[Installation](installation.md).

## Run the smoke test

From the repository root, run:

```bash
pixi run snakemake --cores 4 results/test/small_test/scenario.done
```

Snakemake prints one block per rule as it runs. The run takes under a
minute on a laptop. When it finishes, the log ends with the `scenario` rule
and a step count at 100%:

```
Finished jobid: 0 (Rule: scenario)
29 of 29 steps (100%) done
```

## Check the output

Now list the first replicate:

```bash
ls results/test/small_test/rep1/
```

You see `pedigree.parquet`, `trait.parquet`, `report.yaml`, and `params.yaml`,
alongside other files and subdirectories. The
[Output structure](../user-guide/output-structure.md) page describes each one.

To read the simulation log, run:

```bash
cat logs/test/small_test/rep1/simulate.log
```

## Open the atlas

Snakemake compiles the scenario's plots into one HTML file. Open it in a
browser:

```
results/test/small_test/plots/atlas.html
```

To get a PDF instead, run:

```bash
pixi run snakemake --cores 4 results/test/small_test/plots/atlas.pdf
```

[Interpreting results](../user-guide/interpreting-results.md) describes each
plot.

## Run a full scenario

The `scenario.done` target also builds the folder-wide `report_summary.tsv`,
which needs every scenario in that folder. To build one scenario on its own,
target its `stats.done` and its atlas. Dry-run first with `-n`:

```bash
pixi run snakemake -n --cores 4 results/base/baseline100K/stats.done results/base/baseline100K/plots/atlas.html
pixi run snakemake --cores 4 results/base/baseline100K/stats.done results/base/baseline100K/plots/atlas.html
```

Scenario parameters live in `config/base.yaml`. Defaults live in
`config/_default.yaml`. The scenario runs as shipped, with no edits.

## Next steps

- [Writing a scenario](../user-guide/writing-a-scenario.md) shows how to add
  a scenario. [Configuration](../user-guide/configuration.md) lists every
  parameter.
- [Running the pipeline](../user-guide/running-the-pipeline.md) lists every
  target.
- [Output structure](../user-guide/output-structure.md) lists every file.
