# Output Structure

## Directory layout

```
results/{folder}/{scenario}/
├── rep1/
│   ├── params.yaml                        # Resolved parameters for this replicate
│   ├── pedigree.parquet                   # Pedigree after dropout
│   ├── phenotype.parquet                  # Censored time-to-event phenotypes
│   ├── phenotype.simple_ltm.parquet       # Liability-threshold binary phenotype
│   ├── phenotype_stats.yaml               # Phenotype statistics
│   ├── simple_ltm_stats.yaml              # Threshold phenotype statistics
│   └── validation.yaml                    # Structural + statistical validation
├── rep2/
├── rep3/
└── plots/
    ├── *.png                              # Per-scenario diagnostic plots
    └── atlas.pdf                          # Multi-page PDF atlas
```

## Simulation data files

| File | Description | Temp? |
|---|---|---|
| `pedigree.full.parquet` | Full pedigree before dropout | Yes |
| `pedigree.parquet` | Pedigree after dropout (identical to full when `dropout_rate=0`) | No |
| `phenotype.raw.parquet` | Raw time-to-event phenotypes before censoring | Yes |
| `phenotype.parquet` | Censored time-to-event phenotypes | No |
| `phenotype.sampled.parquet` | Subsampled phenotype for stats | Yes |
| `phenotype.simple_ltm.parquet` | Liability-threshold binary affected status | No |
| `phenotype.simple_ltm.sampled.parquet` | Subsampled threshold phenotype | Yes |
| `params.yaml` | Simulation parameters for this replicate | No |
| `phenotype_stats.yaml` | Per-replicate phenotype statistics | No |
| `simple_ltm_stats.yaml` | Per-replicate threshold statistics | No |

Temp files are auto-deleted by Snakemake after downstream rules complete.

## Validation and logs

| File | Description |
|---|---|
| `results/{folder}/{scenario}/rep{N}/validation.yaml` | Per-replicate validation results |
| `results/{folder}/validation_summary.tsv` | Aggregated metrics across scenarios |
| `results/{folder}/plots/` | Cross-scenario validation and phenotype plots |
| `logs/{folder}/{scenario}/` | Log files |
| `benchmarks/{folder}/{scenario}/` | Runtime and memory benchmarks |

## Plot atlases

| File | Description |
|---|---|
| `results/{folder}/{scenario}/plots/atlas.pdf` | Per-scenario atlas |
| `results/{folder}/plots/atlas.pdf` | Per-folder cross-scenario validation atlas |
| `results/{folder}/{scenario}/rep{N}/epimight/plots/atlas.pdf` | EPIMIGHT atlas |

## Exporting to R

All outputs are parquet files. Convert to TSV for R:

```bash
# Single file (writes .tsv.gz alongside the .parquet)
sim-ace-parquet-to-tsv results/base/baseline10K/rep1/pedigree.parquet

# Multiple files
sim-ace-parquet-to-tsv results/base/baseline10K/rep1/*.parquet

# Uncompressed .tsv
sim-ace-parquet-to-tsv --no-gzip results/base/baseline10K/rep1/pedigree.parquet

# Custom float precision (default: 4 decimal places)
sim-ace-parquet-to-tsv -p 8 results/base/baseline10K/rep1/pedigree.parquet
```

Or via Snakemake (auto-converts matching `.parquet`):

```bash
snakemake --cores 1 results/base/baseline10K/rep1/pedigree.tsv.gz
```
