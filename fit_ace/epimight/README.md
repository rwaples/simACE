# EPIMIGHT Analysis Pipeline

Estimates heritability (h2) and genetic correlation from ACE-simulated time-to-event data using the [EPIMIGHT](https://github.com/BioPsyk/epimight) R package.

## Setup

Create the conda environment (one-time):

```bash
conda env create -f epimight/environment.yml
```

The EPIMIGHT R package is installed automatically on first Snakemake run. For manual use outside Snakemake:

```bash
conda run -n epimight Rscript -e "install.packages('epimight/EPIMIGHT/epimight', repos=NULL, type='source')"
```

## Pipeline

### Step 1: Create TTE input

Converts ACE `phenotype.parquet` into EPIMIGHT's time-to-event format: one parquet per trait plus a JSON file with true simulation parameters.

Uses `PedigreeGraph` from `sim_ace/pedigree_graph.py` to extract all relationship pairs and compute per-kind diagnosed relative counts.

```bash
python epimight/create_parquet.py \
  --phenotype results/base/baseline100K/rep1/phenotype.parquet \
  --output-dir results/base/baseline100K/rep1/epimight/
```

**Input columns used:** `id`, `mother`, `father`, `twin`, `sex`, `generation`, `death_age`, `affected1`, `affected2`, `t_observed1`, `t_observed2`, `A1`, `C1`, `E1`, `A2`, `C2`, `E2`

**Outputs:**

| File | Description |
|------|-------------|
| `trait1.epimight_in.parquet` | Time-to-event data for trait 1 (disorder 1) |
| `trait2.epimight_in.parquet` | Time-to-event data for trait 2 (disorder 2) |
| `true_parameters.json` | True h2 and genetic correlation from variance components |

**TTE columns:**

| Column | Description |
|--------|-------------|
| `person_id` | Individual identifier |
| `born_at` | Generation number |
| `born_at_year` | Calendar year (1960 + generation) |
| `dead_at_year` | Death year (born_at_year + death_age) |
| `failure_status` | 0 = censored, 1 = affected |
| `failure_time` | Observed age at event or censoring (integer years) |
| `diagnosed_relatives_{kind}` | Count of affected relatives for each relationship kind |
| `n_relatives_{kind}` | Total count of relatives for each relationship kind (diagnostics) |

**Relationship kinds:**

| Kind | ACE pair types | Falconer coefficient |
|------|---------------|---------------------|
| PO | Mother-offspring + Father-offspring | 0.5 |
| FS | Full sib + MZ twin | 0.5 |
| HS | Maternal half sib + Paternal half sib | 0.25 |
| mHS | Maternal half sib | 0.25 |
| pHS | Paternal half sib | 0.25 |
| 1C | 1st cousin | 0.125 |
| Av | Avuncular | 0.25 |
| 1G | Grandparent-grandchild | 0.25 |

### Step 2: Run EPIMIGHT analysis

Runs cumulative incidence, heritability, and genetic correlation analyses stratified by birth year. The `relationship_kind` argument selects which `diagnosed_relatives_{kind}` column to use for defining the exposed cohort, and sets the Falconer coefficient accordingly.

```bash
conda run -n epimight Rscript epimight/guide-yob.R \
  results/base/baseline100K/rep1/epimight/ \
  FS
```

**Arguments:**

| Position | Description | Default |
|----------|-------------|---------|
| 1 | Directory containing trait1/trait2 parquets | `.` |
| 2 | Relationship kind (PO, FS, HS, mHS, pHS, 1C, Av, 1G) | `FS` |

**Analyses performed:**

1. **Cumulative Incidence Functions (CIF)** -- risk curves for each trait in each cohort, stratified by birth year
2. **Heritability (h2)** -- Falconer's formula comparing general population risk vs relatives-of-affected risk, stratified by birth year, with fixed/random effects meta-analysis
3. **Genetic Correlation (GC)** -- cross-trait correlation (rhh and rhog) on both a full time x year grid and at maximum follow-up per year, with meta-analysis

**Cohorts:**

| Cohort | Definition |
|--------|------------|
| c1 | All individuals (base population) |
| c2 | Individuals with at least one relative (of the chosen kind) diagnosed with trait 1 |
| c3 | Individuals with at least one relative (of the chosen kind) diagnosed with trait 2 |

## Output

Results go to the same directory as the input parquets:

```
results/{folder}/{scenario}/rep{N}/epimight/
├── trait1.epimight_in.parquet    # TTE trait 1
├── trait2.epimight_in.parquet   # TTE trait 2
├── true_parameters.json         # True h2 and genetic correlation
├── results_{kind}.md            # Summary report
├── tsv/
│   ├── cif_d1_c1_{kind}.tsv     # CIF: trait 1 in base cohort
│   ├── cif_d1_c2_{kind}.tsv     # CIF: trait 1 in relatives of d1-affected
│   ├── cif_d1_c3_{kind}.tsv     # CIF: trait 1 in relatives of d2-affected
│   ├── cif_d2_c1_{kind}.tsv     # CIF: trait 2 in base cohort
│   ├── cif_d2_c3_{kind}.tsv     # CIF: trait 2 in relatives of d2-affected
│   ├── h2_d1_{kind}.tsv         # h2 trait 1 (all time points)
│   ├── h2_d2_{kind}.tsv         # h2 trait 2 (all time points)
│   └── gc_full_{kind}.tsv       # Genetic correlation full grid
└── plots/
    └── atlas.pdf                # Multi-page PDF atlas of all EPIMIGHT figures
```

The summary report (`results_{kind}.md`) contains:

- Cohort sizes
- Heritability at maximum follow-up per birth year
- Heritability meta-analysis (fixed and random effects)
- Genetic correlation at maximum follow-up per birth year (rhh and rhog)
- Genetic correlation meta-analysis
- True vs observed comparison with relative error

Running with different relationship kinds produces separate output files (e.g., `results_FS.md`, `results_PO.md`) that coexist in the same directory.

### Step 3: Plot atlas

Generates a multi-page PDF atlas with all EPIMIGHT figures across relationship kinds.

```bash
python epimight/plot_epimight.py results/base/baseline100K/rep1/epimight/
```

**Figures included:**

- CIF curves (base population and exposed cohorts)
- Heritability over time (h2 vs follow-up age)
- Heritability bar charts with fixed-effect meta-analysis estimates
- Genetic correlation bar charts (rhh and rhog)
- Summary table comparing observed vs true parameters

**Output:** `plots/atlas.pdf` within the epimight directory.

## Snakemake integration

Instead of running the steps manually, use Snakemake to manage the full pipeline:

```bash
# Single scenario (one replicate)
snakemake --cores 4 results/base/baseline100K/rep1/epimight/plots/atlas.pdf

# All replicates for one scenario
snakemake --cores 4 results/base/baseline100K/epimight.done
```

Snakemake uses `conda run -n epimight` to invoke R in the epimight env, and auto-installs the epimight R package on first run.

## Example

```bash
# Full pipeline from ACE phenotype to EPIMIGHT results
python epimight/create_parquet.py \
  --phenotype results/base/baseline100K/rep1/phenotype.parquet \
  --output-dir results/base/baseline100K/rep1/epimight/

# Run with different relationship kinds
conda run -n epimight Rscript epimight/guide-yob.R \
  results/base/baseline100K/rep1/epimight/ PO

conda run -n epimight Rscript epimight/guide-yob.R \
  results/base/baseline100K/rep1/epimight/ FS

# Generate plot atlas
python epimight/plot_epimight.py results/base/baseline100K/rep1/epimight/
```
