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
simace-parquet-to-tsv results/base/baseline10K/rep1/pedigree.parquet

# Multiple files
simace-parquet-to-tsv results/base/baseline10K/rep1/*.parquet

# Uncompressed .tsv
simace-parquet-to-tsv --no-gzip results/base/baseline10K/rep1/pedigree.parquet

# Custom float precision (default: 4 decimal places)
simace-parquet-to-tsv -p 8 results/base/baseline10K/rep1/pedigree.parquet
```

Or via Snakemake (auto-converts matching `.parquet`):

```bash
snakemake --cores 1 results/base/baseline10K/rep1/pedigree.tsv.gz
```

---

## Detailed schemas

The remainder of this page documents column-level parquet schemas, YAML file
structures, validation summary columns, benchmark fields, plot inventories,
and EPIMIGHT outputs. Path patterns use `{folder}`, `{scenario}`, and `{rep}`
as placeholders matching values from `config/_default.yaml`.

### Per-replicate file writers

| File | Format | Description | Writer |
|------|--------|-------------|--------|
| `pedigree.parquet` | Parquet | Pedigree structure with ACE variance components | `simace/simulation/simulate.py` |
| `phenotype.raw.parquet` | Parquet | Raw time-to-event phenotypes (before censoring) | `simace/phenotyping/phenotype.py` |
| `phenotype.parquet` | Parquet | Censored time-to-event phenotypes | `simace/censoring/censor.py` |
| `phenotype.sampled.parquet` | Parquet | Downsampled phenotype for plotting and stats | `workflow/scripts/simace/sample.py` |
| `phenotype.simple_ltm.parquet` | Parquet | Binary affected status from liability-threshold model | `simace/phenotyping/threshold.py` |
| `phenotype.simple_ltm.sampled.parquet` | Parquet | Downsampled threshold phenotype | `workflow/scripts/simace/sample.py` |
| `params.yaml` | YAML | Simulation parameters for this replicate | `simace/simulation/simulate.py` |
| `phenotype_stats.yaml` | YAML | Phenotype statistics (correlations, prevalence, CIF, etc.) | `workflow/scripts/simace/compute_phenotype_stats.py` → `simace/analysis/stats/runner.py` |
| `phenotype_samples.parquet` | Parquet | Further downsampled phenotype rows for stats scatter plots | `workflow/scripts/simace/compute_phenotype_stats.py` → `simace/analysis/stats/runner.py` |
| `validation.yaml` | YAML | Structural and statistical validation results | `simace/analysis/validate.py` |

### Per-scenario, per-folder, and sentinel files

| File | Format | Description |
|------|--------|-------------|
| `results/{folder}/{scenario}/plots/*.png` | PNG (or PDF) | Phenotype and simple LTM figures (see [Plots](#plots)) |
| `results/{folder}/{scenario}/plots/atlas.pdf` | PDF | Multi-page atlas combining all scenario figures |
| `results/{folder}/{scenario}/scenario.done` | Sentinel | Empty file indicating scenario completion |
| `results/{folder}/validation_summary.tsv` | TSV | Aggregated validation metrics across scenarios and replicates |
| `results/{folder}/plots/*.png` | PNG | Cross-scenario validation plots |
| `results/{folder}/plots/atlas.pdf` | PDF | Multi-page atlas of cross-scenario validation figures |
| `results/{folder}/folder.done` | Sentinel | Empty file indicating folder completion |
| `results/{folder}/epimight.done` | Sentinel | Empty file indicating EPIMIGHT completion |

### Logs and benchmarks

| File | Format | Description |
|------|--------|-------------|
| `logs/{folder}/{scenario}/rep{rep}/*.log` | Text | Per-rule log files |
| `benchmarks/{folder}/{scenario}/rep{rep}/*.tsv` | TSV | Per-rule Snakemake benchmark files |
| `benchmarks/{folder}/{scenario}/*.tsv` | TSV | Per-scenario benchmark files (plotting, atlas assembly) |
| `benchmarks/{folder}/*.tsv` | TSV | Per-folder benchmark files (gather, validation plots) |

---

## Parquet column reference

### pedigree.parquet

Core pedigree structure with latent variance components for two correlated traits.

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Unique individual identifier |
| `sex` | int8 | 0 = female, 1 = male |
| `mother` | int64 | Mother's id (-1 for founders) |
| `father` | int64 | Father's id (-1 for founders) |
| `twin` | int64 | MZ twin partner's id (-1 if not a twin) |
| `generation` | int8 | Generation number (0 = oldest recorded) |
| `household_id` | int64 | Shared-environment household group |
| `A1`, `A2` | float32 | Additive genetic component (traits 1 and 2) |
| `C1`, `C2` | float32 | Common/shared environment component |
| `E1`, `E2` | float32 | Unique environment component |
| `liability1`, `liability2` | float32 | Total liability (A + C + E) |

### phenotype.raw.parquet

Raw time-to-event phenotypes before age-window and competing-risk censoring. Subset of generations defined by `G_pheno`. Includes pedigree columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `t1`, `t2` | float32 | Raw (uncensored) age-at-onset from the phenotype model |

### phenotype.parquet

Extends phenotype.raw with censoring applied via age windows and competing-risk death. Contains all pedigree and raw phenotype columns, plus:

| Column | Type | Description |
|--------|------|-------------|
| `death_age` | float32 | Age at death from competing-risk mortality |
| `t_observed1`, `t_observed2` | float32 | Observed age-at-onset after age and death censoring |
| `age_censored1`, `age_censored2` | bool | True if onset falls outside the generation's observation window |
| `death_censored1`, `death_censored2` | bool | True if onset occurs after death |
| `affected1`, `affected2` | bool | True if the individual is observed as affected (not age- or death-censored) |

### phenotype.simple_ltm.parquet

Binary affected status from a liability-threshold model. Each generation has an independent prevalence-based threshold.

| Column | Type | Description |
|--------|------|-------------|
| `id` | int64 | Individual identifier |
| `generation` | int8 | Generation number |
| `sex` | int8 | 0 = female, 1 = male |
| `mother`, `father`, `twin` | int64 | Family links (same as pedigree) |
| `household_id` | int64 | Shared-environment household group |
| `A1`, `C1`, `E1`, `liability1` | float32 | Trait 1 variance components and liability |
| `A2`, `C2`, `E2`, `liability2` | float32 | Trait 2 variance components and liability |
| `affected1`, `affected2` | bool | True if liability exceeds the generation-specific threshold |

### Sampled parquets

The pipeline produces several downsampled parquet files to keep plotting and stats computation tractable for large populations:

| File | Source | Purpose |
|------|--------|---------|
| `phenotype.sampled.parquet` | `phenotype.parquet` | Downsampled rows for phenotype stats input; preserves parents of sampled individuals |
| `phenotype.simple_ltm.sampled.parquet` | `phenotype.simple_ltm.parquet` | Downsampled threshold rows retained for explicit downstream targets |
| `phenotype_samples.parquet` | `phenotype.sampled.parquet` | Further downsampled during stats computation for scatter/histogram plots |

All sampled parquets share the same column schema as their source files.

---

## YAML file schemas

### params.yaml

Flat key-value file recording the simulation parameters used for a replicate. Written by `simace/simulation/simulate.py`.

| Key | Type | Description |
|-----|------|-------------|
| `seed` | int | Random seed for this replicate |
| `rep` | int | Replicate number |
| `A1`, `C1`, `E1` | float | Trait 1 variance components (E1 = 1 - A1 - C1) |
| `A2`, `C2`, `E2` | float | Trait 2 variance components |
| `rA` | float | Cross-trait additive genetic correlation |
| `rC` | float | Cross-trait common environment correlation |
| `rE` | float | Cross-trait unique environment correlation |
| `N` | int | Population size per generation |
| `G_ped` | int | Number of generations in output pedigree |
| `G_sim` | int | Total generations simulated (including burn-in) |
| `mating_lambda` | float | ZTP mating count lambda |
| `p_mztwin` | float | Probability of MZ twin birth |
| `assort1` | float | Mate correlation on trait 1 liability (0 = random) |
| `assort2` | float | Mate correlation on trait 2 liability (0 = random) |

### phenotype_stats.yaml

Phenotype statistics computed from the censored phenotype. Written by
`workflow/scripts/simace/compute_phenotype_stats.py`, which calls
`simace.analysis.stats.runner`. Top-level sections:

| Section | Description |
|---------|-------------|
| `n_individuals` | Total individual count |
| `n_generations` | Number of phenotyped generations |
| `case_ascertainment_ratio` | Ascertainment ratio used for sampling (conditional; only when ratio != 1.0) |
| `prevalence` | Per-trait observed prevalence (float per trait) |
| `mortality` | Decade-binned mortality rates (`decade_labels`, `rates`) |
| `person_years` | Total and per-trait person-years at risk |
| `family_size` | Family size statistics: `mean`, `median`, `q1`, `q3`, `n_families`, and size distribution |
| `regression` | Liability-vs-age-at-onset regression per trait (`slope`, `intercept`, `r`, `r2`, `n`) |
| `cumulative_incidence` | Cumulative incidence curves per trait (`ages`, `observed_values`, `true_values`, `half_target_age`) |
| `joint_affection` | Cross-trait joint affection counts and proportions |
| `cumulative_incidence_by_sex` | CIF curves stratified by trait and sex |
| `cumulative_incidence_by_sex_generation` | CIF curves stratified by trait, generation, and sex |
| `censoring` | Generation observation windows and censor age (present when `gen_censoring` is configured) |
| `censoring_confusion` | Per-trait confusion matrix for censoring vs true affection (conditional) |
| `censoring_cascade` | Per-trait, per-generation censoring cascade counts (conditional) |
| `mate_correlation` | 2×2 Pearson correlation matrix between mated pairs' liabilities (`matrix`, `n_pairs`); conditional on pedigree |
| `pair_counts` | Count of extracted relationship pairs by type (MZ, FS, HS, PO, etc.) |
| `pair_counts_ped` | Pair counts from full pedigree (when pedigree file is provided; conditional) |
| `n_individuals_ped` | Individual count in full pedigree (conditional) |
| `n_generations_ped` | Generation count in full pedigree (conditional) |
| `liability_correlations` | Pearson correlations of liability by trait and pair type |
| `parent_offspring_corr` | Parent-offspring liability correlations by trait and generation |
| `parent_offspring_corr_by_sex` | Parent-offspring liability correlations stratified by parent sex |
| `parent_status` | Parent affection status breakdown (affected/unaffected parent counts and offspring prevalence) |
| `tetrachoric` | Tetrachoric correlations of affection status by trait and pair type (`r`, `se`, `n_pairs`) |
| `tetrachoric_by_generation` | Tetrachoric correlations stratified by generation |
| `tetrachoric_by_sex` | Tetrachoric correlations stratified by sex |
| `cross_trait_tetrachoric` | Cross-trait tetrachoric correlations (`same_person`, `same_person_by_generation`, `cross_person`) |

Sections marked "conditional" are only present when the corresponding data or config options are available.

### validation.yaml

Structural and statistical validation results. Written by `simace/analysis/validate.py`. Top-level sections:

| Section | Description |
|---------|-------------|
| `structural` | Pedigree integrity checks: `id_integrity`, `parent_references`, `sex_parent_consistency`, `sex_distribution` |
| `twins` | Twin validation: `twin_bidirectional`, `twin_same_parents`, `twin_same_A1`, `twin_same_A2`, `twin_same_sex`, `twin_rate` |
| `half_sibs` | Half-sibling checks: `half_sib_pair_proportion`, `offspring_with_half_sib` |
| `statistical` | Variance component checks: `variance_A1`..`E2`, `total_variance_trait1/2`, `cross_trait_rA/rC/rE`, `c1/2_inheritance`, `e1_independence` |
| `heritability` | Heritability estimates: MZ/DZ twin correlations, Falconer estimates, parent-offspring regressions (per trait) |
| `population` | Population structure: `generation_sizes`, `generation_count`, `family_size` |
| `per_generation` | Per-generation stats: `n`, liability mean/variance/sd, component A/C/E mean/variance |
| `summary` | Overall result: `passed` (bool), `checks_passed`, `checks_failed`, `checks_total` |
| `assortative_mating` | Mate liability correlation checks: observed vs expected mate correlations per trait |
| `consanguineous_matings` | Consanguineous mating detection: shared grandparent counts, grandparent-link discrepancy reconciliation |
| `family_size_distribution` | Family size by parent sex: `mother`/`father` each with `mean`, `median`, `std`, `n_parents` |
| `parameters` | Full scenario parameters (copy of config values used) |

Each individual check within `structural`, `twins`, `half_sibs`, `statistical`, `heritability`, `population`, `assortative_mating`, and `consanguineous_matings` is a dict containing at minimum `passed` (bool) and `details` (str), plus check-specific fields like `observed`, `expected`, and `tolerance`.

---

## validation_summary.tsv

Aggregated metrics across all scenarios and replicates within a folder. Written by `simace/analysis/gather.py`. One row per replicate.

| Column | Source |
|--------|--------|
| `scenario`, `rep` | Parsed from file path |
| `N`, `G_ped`, `G_sim` | `parameters` |
| `A1`, `C1`, `E1`, `A2`, `C2`, `E2` | `parameters` |
| `rA`, `rC` | `parameters` |
| `p_mztwin`, `mating_lambda`, `seed` | `parameters` |
| `checks_failed` | `summary.checks_failed` |
| `observed_twin_rate` | `twins.twin_rate.observed_rate` |
| `variance_A1`..`E2` | `statistical.variance_*.observed` |
| `observed_rA`, `observed_rC`, `observed_rE` | `statistical.cross_trait_r*.observed` |
| `mz_twin_A1_corr`, `mz_twin_liability1_corr` | `heritability.mz_twin_*_correlation.observed` |
| `mz_twin_A2_corr`, `mz_twin_liability2_corr` | `heritability.mz_twin_*_correlation.observed` |
| `dz_sibling_A1_corr`, `dz_sibling_liability1_corr` | `heritability.dz_sibling_*_correlation.observed` |
| `dz_sibling_A2_corr`, `dz_sibling_liability2_corr` | `heritability.dz_sibling_*_correlation.observed` |
| `half_sib_prop_expected`, `half_sib_prop_observed` | `half_sibs.half_sib_pair_proportion.*` |
| `offspring_with_half_sib_expected`, `offspring_with_half_sib_observed` | `half_sibs.offspring_with_half_sib.*` |
| `half_sib_A1_corr`, `half_sib_liability1_corr`, `half_sib_shared_C1` | `half_sibs.half_sib_*.*` |
| `simulate_seconds`, `simulate_max_rss_mb` | Parsed from benchmark TSV |
| `mother_mean_offspring`, `father_mean_offspring` | `family_size_distribution.*.mean` |
| `falconer_h2_trait1`, `falconer_h2_trait2` | `heritability.falconer_estimate_*.observed` |
| `parent_offspring_A1_slope`, `parent_offspring_A1_r2` | `heritability.parent_offspring_A1_regression.*` |
| `parent_offspring_liability1_slope`, `parent_offspring_liability1_r2` | `heritability.parent_offspring_liability1_regression.*` |
| `parent_offspring_A2_slope`, `parent_offspring_A2_r2` | `heritability.parent_offspring_A2_regression.*` |
| `parent_offspring_liability2_slope`, `parent_offspring_liability2_r2` | `heritability.parent_offspring_liability2_regression.*` |

---

## Benchmark TSVs

Snakemake automatically writes benchmark files in TSV format with a standard header.

| Column | Description |
|--------|-------------|
| `s` | Wall-clock seconds |
| `h:m:s` | Wall-clock time in h:m:s format |
| `max_rss` | Maximum resident set size in MB (Linux/macOS; 1 on Windows) |
| `max_vms` | Maximum virtual memory size in MB |
| `max_uss` | Maximum unique set size in MB |
| `max_pss` | Maximum proportional set size in MB |
| `io_in` | I/O read in MB |
| `io_out` | I/O write in MB |
| `mean_load` | Mean CPU load |
| `cpu_time` | CPU time in seconds |

Benchmark files are written for each pipeline rule. Per-replicate benchmarks:

- `benchmarks/{folder}/{scenario}/rep{rep}/simulate.tsv`
- `benchmarks/{folder}/{scenario}/rep{rep}/dropout.tsv`
- `benchmarks/{folder}/{scenario}/rep{rep}/phenotype.tsv`
- `benchmarks/{folder}/{scenario}/rep{rep}/censor_weibull.tsv`
- `benchmarks/{folder}/{scenario}/rep{rep}/phenotype_simple_ltm.tsv`
- `benchmarks/{folder}/{scenario}/rep{rep}/sample_phenotype.tsv`
- `benchmarks/{folder}/{scenario}/rep{rep}/sample_simple_ltm.tsv`
- `benchmarks/{folder}/{scenario}/rep{rep}/phenotype_stats.tsv`
- `benchmarks/{folder}/{scenario}/rep{rep}/validate.tsv`

Per-scenario benchmarks:

- `benchmarks/{folder}/{scenario}/plot_phenotype.tsv`
- `benchmarks/{folder}/{scenario}/assemble_atlas.tsv`

Per-folder benchmarks:

- `benchmarks/{folder}/gather_validation.tsv`
- `benchmarks/{folder}/plot_validation.tsv`

---

## Plots

Plot files are written as PNG by default (configurable via `plot_format` in `_default.yaml`). All scenario plots live under `results/{folder}/{scenario}/plots/`.

### Phenotype plots

Ordered by narrative flow: pedigree structure, liability, phenotype, censoring, correlations.

| File | Description |
|------|-------------|
| `pedigree_counts.ped.{ext}` | Relationship pair counts from full pedigree |
| `pedigree_counts.{ext}` | Relationship pair counts from phenotyped subset |
| `family_structure.{ext}` | Family structure breakdown (sibship sizes, half-sibling fractions) |
| `mate_correlation.{ext}` | Mate liability correlations (assortative mating) |
| `cross_trait.{ext}` | Cross-trait liability scatter |
| `parent_offspring_liability.by_generation.{ext}` | Parent-offspring liability correlations by generation |
| `heritability.by_generation.{ext}` | Liability-scale heritability by generation |
| `heritability.by_sex.by_generation.{ext}` | Liability-scale heritability by sex and generation |
| `additive_shared.by_generation.{ext}` | Additive and shared environment by generation |
| `liability_violin.phenotype.{ext}` | Liability violin plots by affection status |
| `liability_violin.phenotype.by_generation.{ext}` | Liability violins by generation and affection status |
| `liability_violin.phenotype.by_sex.by_generation.{ext}` | Liability violins by sex, generation, and affection status |
| `age_at_onset_death.{ext}` | Age-at-onset and death age distributions |
| `mortality.{ext}` | Mortality rates by decade |
| `cumulative_incidence.phenotype.{ext}` | Cumulative incidence curves by trait |
| `cumulative_incidence.by_sex.{ext}` | Cumulative incidence by sex |
| `cumulative_incidence.by_sex.by_generation.{ext}` | Cumulative incidence by sex and generation |
| `censoring.{ext}` | Censoring window visualization |
| `censoring_confusion.{ext}` | Censoring confusion matrix |
| `censoring_cascade.{ext}` | Censoring cascade by generation |
| `liability_vs_aoo.{ext}` | Liability vs age-at-onset scatter |
| `joint_affected.phenotype.{ext}` | Cross-trait joint affection proportions |
| `tetrachoric.phenotype.{ext}` | Tetrachoric correlation heatmap |
| `tetrachoric.phenotype.by_sex.{ext}` | Tetrachoric correlations stratified by sex |
| `tetrachoric.phenotype.by_generation.{ext}` | Tetrachoric correlations by generation |
| `cross_trait.phenotype.{ext}` | Cross-trait phenotype correlations |
| `cross_trait.phenotype.t2.{ext}` | Cross-trait phenotype correlations (trait 2 focus) |
| `cross_trait_tetrachoric.{ext}` | Cross-trait tetrachoric correlations |

### Validation plots (`results/{folder}/plots/`)

| File | Description |
|------|-------------|
| `family_size.{ext}` | Family size distributions |
| `twin_rate.{ext}` | Observed vs expected twin rates |
| `half_sib_proportions.{ext}` | Half-sibling proportion comparisons |
| `variance_components.{ext}` | Actual vs expected variance components |
| `correlations_A.{ext}` | Additive genetic correlations |
| `correlations_phenotype.{ext}` | Phenotypic correlations |
| `heritability_estimates.{ext}` | Heritability estimates vs true values |
| `cross_trait_correlations.{ext}` | Cross-trait correlation comparisons |
| `summary_bias.{ext}` | Summary bias across checks |
| `runtime.{ext}` | Execution time by scenario |
| `memory.{ext}` | Memory usage by scenario |

### Atlases

Multi-page PDF atlases combine all plots for a scope into a single document with figure captions:

| File | Contents |
|------|----------|
| `results/{folder}/{scenario}/plots/atlas.pdf` | All phenotype figures for one scenario |
| `results/{folder}/plots/atlas.pdf` | All cross-scenario validation figures for one folder |
| `results/{folder}/{scenario}/rep{rep}/epimight/plots/atlas.pdf` | EPIMIGHT CIF, heritability, and genetic correlation figures |

---

## EPIMIGHT outputs

EPIMIGHT heritability analysis outputs are written under `results/{folder}/{scenario}/rep{rep}/epimight/`. See [epimight/README.md](https://github.com/rwaples/fitACE/blob/master/epimight/README.md) for full pipeline documentation.

Key output files:

| File | Format | Description |
|------|--------|-------------|
| `trait1.epimight_in.parquet` | Parquet | Time-to-event data for trait 1 |
| `trait2.epimight_in.parquet` | Parquet | Time-to-event data for trait 2 |
| `true_parameters.json` | JSON | True heritability and genetic correlation from variance components |
| `results_{kind}.md` | Markdown | Summary report per relationship kind (FS, PO, HS, mHS, pHS, etc.) |
| `tsv/cif_d1_c1_{kind}.tsv` | TSV | CIF: trait 1 in base cohort |
| `tsv/cif_d1_c2_{kind}.tsv` | TSV | CIF: trait 1 in relatives of trait-1-affected |
| `tsv/cif_d1_c3_{kind}.tsv` | TSV | CIF: trait 1 in relatives of trait-2-affected |
| `tsv/cif_d2_c1_{kind}.tsv` | TSV | CIF: trait 2 in base cohort |
| `tsv/cif_d2_c3_{kind}.tsv` | TSV | CIF: trait 2 in relatives of trait-2-affected |
| `tsv/h2_d1_{kind}.tsv` | TSV | Heritability estimates for trait 1 |
| `tsv/h2_d2_{kind}.tsv` | TSV | Heritability estimates for trait 2 |
| `tsv/gc_full_{kind}.tsv` | TSV | Genetic correlation full grid |
| `plots/atlas.pdf` | PDF | Multi-page PDF of all EPIMIGHT figures |
