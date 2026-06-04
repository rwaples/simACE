# Quick Start

## Run the smoke test

Run the smallest scenario to confirm the pipeline is functional (this
takes a minute or two):

```bash
snakemake --cores 4 results/test/small_test/scenario.done
```

## Check the output

```bash
ls results/test/small_test/rep1/    # pedigree.parquet, trait files, reports, params
cat logs/test/small_test/rep1/simulate.log
```

A successful run produces these key files per replicate:

| File | What it contains |
|---|---|
| `pedigree.parquet` | Analysis pedigree: parent links, generation, sex, household IDs, ACE components, liabilities |
| `trait.parquet` | Outcomes-only censored time-to-event traits (`id`, onset/censoring/affected columns) |
| `trait.simple_ltm.parquet` | Outcomes-only liability-threshold binary affected status |
| `report.yaml` | Curated v2 scientific report: `scopes`, `quality_checks`, `truth`, `observed`, `estimators` (dense plot arrays go to `plot_payload.yaml`) |
| `params.yaml` | The resolved parameters for this replicate |

## Explore the atlas

Per-scenario plots are compiled into a self-contained HTML atlas (open it in any
browser):

```
results/test/small_test/plots/atlas.html
```

A multi-page PDF atlas is available on demand — build it with
`snakemake --cores 4 results/test/small_test/plots/atlas.pdf`.

See [Interpreting Results](../user-guide/interpreting-results.md) for descriptions of each plot.

## Next steps

- [Configuration](../user-guide/configuration.md) -- customise scenarios and parameters
- [Running the Pipeline](../user-guide/running-the-pipeline.md) -- full pipeline usage
- [Output Structure](../user-guide/output-structure.md) -- complete file layout
