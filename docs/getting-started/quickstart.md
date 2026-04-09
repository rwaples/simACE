# Quick Start

## Run the smoke test

Run the smallest scenario to confirm everything works (takes a minute or two):

```bash
snakemake --cores 4 results/test/small_test/scenario.done
```

## Check the output

```bash
ls results/test/small_test/rep1/    # pedigree.parquet, phenotype files, validation, stats
cat logs/test/small_test/rep1/simulate.log
```

A successful run produces these key files per replicate:

| File | What it contains |
|---|---|
| `pedigree.parquet` | Full pedigree with parent links, generation, sex, liability components |
| `phenotype.parquet` | Censored time-to-event phenotypes (age-at-onset, affected status) |
| `phenotype.simple_ltm.parquet` | Liability-threshold binary affected status |
| `validation.yaml` | Structural and statistical validation results |
| `phenotype_stats.yaml` | Phenotype statistics (correlations, prevalence, CIF) |
| `params.yaml` | The resolved parameters for this replicate |

## Explore the atlas

Per-scenario plots are compiled into a multi-page PDF atlas:

```
results/test/small_test/plots/atlas.pdf
```

See [Interpreting Results](../user-guide/interpreting-results.md) for what each plot means.

## Next steps

- [Configuration](../user-guide/configuration.md) -- customise scenarios and parameters
- [Running the Pipeline](../user-guide/running-the-pipeline.md) -- full pipeline usage
- [Output Structure](../user-guide/output-structure.md) -- complete file layout
