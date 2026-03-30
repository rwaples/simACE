# fit_ace — Statistical Model Fitting for ACE

Statistical model fitting package for the ACE project: EPIMIGHT heritability, PA-FGRS genetic risk scores, Weibull frailty correlation, and Stan-based models.

`fit_ace` depends on `sim_ace` for shared infrastructure (pedigree graphs, hazard functions, plotting utilities). Both packages are orchestrated by the root Snakemake workflow.

## Installation

```bash
conda activate ACE
pip install -e fit_ace/
```

## EPIMIGHT

EPIMIGHT estimates heritability (h²) and genetic correlation from time-to-event data using the [EPIMIGHT](https://github.com/BioPsyk/epimight) R package. It compares cumulative incidence in relatives of affected individuals against the general population, then applies Falconer's formula to derive heritability estimates stratified by birth year with fixed/random effects meta-analysis.

### Setup

EPIMIGHT requires a separate R-based conda environment:

```bash
conda env create -f fit_ace/epimight/environment.yml
conda run -n epimight Rscript -e "install.packages('fit_ace/epimight/EPIMIGHT/epimight', repos=NULL, type='source')"
```

### Running via Snakemake

```bash
# All scenarios and replicates
snakemake --cores 4 epimight_all

# Single scenario (one replicate)
snakemake --cores 4 results/base/baseline100K/rep1/epimight/plots/atlas.pdf

# Single relationship kind
snakemake --cores 4 results/base/baseline100K/rep1/epimight/tsv/h2_d1_FS.tsv
```

Which relationship kinds are analyzed is controlled by the `epimight_kinds` config parameter (default: `[PO, FS, HS, mHS, pHS]`).

### Outputs

Each replicate's `epimight/` directory contains:

| File | Description |
|------|-------------|
| `trait1.epimight_in.parquet`, `trait2.epimight_in.parquet` | Time-to-event input data for traits 1 and 2 |
| `true_parameters.json` | True h² and genetic correlation from variance components |
| `results_{kind}.md` | Summary report per relationship kind (cohort sizes, h² meta-analysis, genetic correlation, true vs observed comparison) |
| `tsv/cif_*.tsv`, `tsv/h2_*.tsv`, `tsv/gc_*.tsv` | CIF curves, heritability estimates, and genetic correlation per kind |
| `plots/atlas.pdf` | Multi-page PDF atlas: CIF curves, h² over time, h² bar charts, genetic correlation, and observed vs true comparison |

See [epimight/README.md](epimight/README.md) for the full pipeline reference — manual steps, column schemas, cohort definitions, and relationship kinds.

## PA-FGRS

Pearson-Aitken Family-based Genetic Risk Scores. Computes Bayesian posterior mean and variance of genetic liability given observed family history.

### Running via Snakemake

```bash
snakemake --cores 4 pafgrs_all
```

## Stan Models

Stan-based ACE model fitting (REML and full Bayesian). See `fit_ace/stan/` for model files and Python wrappers.

## Package Structure

```
fit_ace/
├── pafgrs/                        # PA-FGRS genetic risk scores
│   ├── pafgrs.py                  # Pearson-Aitken scoring (Bayesian posterior mean/variance)
│   └── pafgrs_metrics.py          # Validation metrics (r, R², AUC, calibration)
├── epimight/                      # EPIMIGHT heritability analysis (separate conda env)
│   ├── create_parquet.py          # Convert phenotype to EPIMIGHT TTE format
│   ├── guide-yob.R               # CIF, h², genetic correlation (R)
│   ├── plot_epimight.py           # EPIMIGHT diagnostic atlas
│   ├── epimight_bias_analysis.py  # Bias quantification
│   ├── EPIMIGHT/                  # Vendored BioPsyk EPIMIGHT R package
│   └── environment.yml            # Conda env for R dependencies
├── stan/                          # Stan-based model fitting
│   ├── fit_ace.py, fit_reml.py    # Python wrappers
│   └── *.stan                     # Stan model files
└── plotting/
    ├── plot_pafgrs.py             # PA-FGRS diagnostic atlas
    └── plot_epimight_bias.py      # EPIMIGHT bias analysis atlas
```

### Workflow Rules

| Rule file | Description |
|-----------|-------------|
| `workflow/rules/epimight.smk` | EPIMIGHT heritability pipeline |
| `workflow/rules/epimight_bias.smk` | EPIMIGHT bias analysis |
| `workflow/rules/pafgrs.smk` | PA-FGRS scoring pipeline |
