# Quick start

In this tutorial we run the smallest scenario end to end, look at the files it
writes, and open its plot atlas. It assumes you have finished the
[Installation](installation.md).

## Run the smoke test

From the repository root, run:

```bash
pixi run simace run small_test
```

`simace run` prints one line as each stage starts and finishes, tagged with
the replicate. The run takes under a minute on a laptop. It ends with the
scenario's plots and atlas:

```
[small_test/rep3] analyze finished in 1.9s, peak 292 MB
[small_test] plot started
[small_test] plot finished in 21.5s
[small_test] atlas started
[small_test] atlas finished in 1.1s
```

Run the same command again. Every replicate is reported as
`skip (run.yaml matches)`, and only the plots are redrawn.

## Check the output

Now list the first replicate:

```bash
ls results/test/small_test/rep1/
```

You see `pedigree.parquet`, `trait.parquet`, `report.yaml`, and `params.yaml`,
alongside the other stage outputs, `run.yaml`, and `timing.tsv`. The
[Output structure](../user-guide/output-structure.md) page describes each one.

To read the simulation log, run:

```bash
cat logs/test/small_test/rep1/simulate.log
```

## Open the atlas

`simace run` compiles the scenario's plots into one HTML file. Open it in a
browser:

```
results/test/small_test/plots/atlas.html
```

To get a PDF instead, run:

```bash
pixi run simace run small_test --format pdf
```

[Interpreting results](../user-guide/interpreting-results.md) describes each
plot.

## Summarize the folder

`small_test` lives in the `test` folder with one other scenario. To compare
every scenario in a folder, run both, then gather the folder:

```bash
pixi run simace run coverage_scenario
pixi run simace gather test
```

`simace gather` writes `results/test/report_summary.tsv` and the validation
atlas at `results/test/plots/atlas.html`.

## Run a full scenario

Preview a larger scenario with `--dry-run`, then run it:

```bash
pixi run simace run baseline100K --dry-run
pixi run simace run baseline100K
```

Scenario parameters live in `config/base.yaml`. Defaults live in
`config/_default.yaml`. The scenario runs as shipped, with no edits.

## Next steps

- [Writing a scenario](../user-guide/writing-a-scenario.md) shows how to add
  a scenario. [Configuration](../user-guide/configuration.md) lists every
  parameter.
- [Running the pipeline](../user-guide/running-the-pipeline.md) covers
  reruns, parallel replicates, and running one stage by hand.
- [Output structure](../user-guide/output-structure.md) lists every file.
