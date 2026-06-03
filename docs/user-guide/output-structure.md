# Output Structure

## Directory layout

```
results/{folder}/{scenario}/
├── rep1/
│   ├── params.yaml                        # Resolved parameters for this replicate
│   ├── pedigree.full.parquet              # Recorded pedigree (full pre-ascertainment)
│   ├── pedigree.parquet                   # Analysis pedigree (ancestor closure of sampled IDs)
│   ├── trait.full.parquet                 # Phenotyped population (full pre-ascertainment, durable)
│   ├── trait.parquet                      # Analysis sample (post-ascertainment censored phenotypes)
│   ├── trait.simple_ltm.parquet           # Post-ascertainment liability-threshold benchmark
│   ├── report.yaml                        # Curated v2 scientific report (scopes/quality_checks/truth/observed/estimators)
│   └── plot_payload.yaml                  # Dense incidence/censoring arrays for plotting
├── rep2/
├── rep3/
└── plots/
    ├── *.png                              # Per-scenario diagnostic plots
    ├── atlas.html                         # Self-contained HTML atlas (default)
    └── atlas.pdf                          # Multi-page PDF atlas (on-demand export)
```

## Simulation data files

| File | Description | Temp? |
|---|---|---|
| `pedigree.full.parquet` | Full simulated pedigree (post-burn-in, pre-ascertainment) — consumed by validation, phenotype, phenotype_simple_ltm, and ascertainment | No |
| `pedigree.parquet` | Post-ascertainment pedigree (ancestor closure of sampled IDs; identical row count to `.full` when `dropout_rate=0` and `N_sample=0`) | No |
| `trait.raw.parquet` | Raw time-to-event phenotypes before censoring | Yes |
| `trait.full.parquet` | Phenotyped population — post-censor phenotypes for the full pre-ascertainment population; durable so Analyze can quantify ascertainment distortion (ADR 0008) | No |
| `trait.simple_ltm.full.parquet` | Pre-ascertainment liability-threshold benchmark | Yes |
| `trait.parquet` | Post-ascertainment censored time-to-event phenotypes (canonical output) | No |
| `trait.simple_ltm.parquet` | Post-ascertainment liability-threshold benchmark (canonical output) | No |
| `params.yaml` | Simulation parameters for this replicate | No |
| `report.yaml` | Curated v2 per-replicate scientific report (`scopes`, `quality_checks`, `truth`, `observed`, `estimators`) | No |
| `plot_payload.yaml` | Dense incidence/censoring arrays for reproducible plotting (companion to `report.yaml`) | No |

Temp files are auto-deleted by Snakemake after downstream rules complete.

## Validation and logs

| File | Description |
|---|---|
| `results/{folder}/{scenario}/rep{N}/report.yaml` | Curated v2 per-replicate scientific report |
| `results/{folder}/report_summary.tsv` | Aggregated metrics across scenarios (gathered from each report's `truth`/`estimators`/`scopes`) |
| `results/{folder}/plots/` | Cross-scenario validation and phenotype plots |
| `logs/{folder}/{scenario}/` | Log files |
| `benchmarks/{folder}/{scenario}/` | Runtime and memory benchmarks |

## Plot atlases

| File | Description |
|---|---|
| `results/{folder}/{scenario}/plots/atlas.html` | Per-scenario atlas (default, self-contained HTML) |
| `results/{folder}/{scenario}/plots/atlas.pdf` | Per-scenario atlas (on-demand PDF export) |
| `results/{folder}/plots/atlas.html` | Per-folder cross-scenario validation atlas (default) |
| `results/{folder}/plots/atlas.pdf` | Per-folder validation atlas (on-demand PDF export) |
| `results/{folder}/{scenario}/rep{N}/epimight/plots/atlas.html` | EPIMIGHT atlas (default; `.pdf` on demand) |

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
| `pedigree.full.parquet` | Parquet | Full simulated pedigree (post-burn-in, pre-ascertainment); persistent for validation | `simace/simulation/simulate.py` |
| `pedigree.parquet` | Parquet | Post-ascertainment pedigree (ancestor closure of sampled IDs, dangling refs severed) | `simace/ascertainment/__init__.py` |
| `trait.parquet` | Parquet | Post-ascertainment per-individual trait outcomes (censored time-to-event + affected status) | `simace/ascertainment/__init__.py` |
| `trait.simple_ltm.parquet` | Parquet | Post-ascertainment per-individual simple-LTM benchmark (parallel LTM trait) | `simace/ascertainment/__init__.py` |
| `params.yaml` | YAML | Simulation parameters for this replicate | `simace/simulation/simulate.py` |
| `report.yaml` | YAML | Curated v2 per-replicate scientific report (`scopes`, `quality_checks`, `truth`, `observed`, `estimators`) | `workflow/scripts/simace/analyze.py` → `simace/analysis/analyze.py` |
| `plot_payload.yaml` | YAML | Dense incidence/censoring arrays for plotting (companion to `report.yaml`) | `workflow/scripts/simace/analyze.py` → `simace/analysis/analyze.py` |
| `plotting_sample.parquet` | Parquet | Further downsampled trait rows for stats scatter plots | `workflow/scripts/simace/analyze.py` → `simace/analysis/analyze.py` |

### Per-scenario, per-folder, and sentinel files

| File | Format | Description |
|------|--------|-------------|
| `results/{folder}/{scenario}/plots/*.png` | PNG (or PDF) | Phenotype and simple LTM figures (see [Plots](#plots)) |
| `results/{folder}/{scenario}/plots/atlas.html` | HTML | Self-contained scenario atlas (default build) |
| `results/{folder}/{scenario}/plots/atlas.pdf` | PDF | Multi-page scenario atlas (on-demand export) |
| `results/{folder}/{scenario}/scenario.done` | Sentinel | Empty file indicating scenario completion |
| `results/{folder}/report_summary.tsv` | TSV | Aggregated validation metrics across scenarios and replicates |
| `results/{folder}/plots/*.png` | PNG | Cross-scenario validation plots |
| `results/{folder}/plots/atlas.html` | HTML | Self-contained cross-scenario validation atlas (default build) |
| `results/{folder}/plots/atlas.pdf` | PDF | Cross-scenario validation atlas (on-demand export) |
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

### trait.parquet

Extends phenotype.raw with censoring applied via age windows and competing-risk death. Contains all pedigree and raw phenotype columns, plus:

| Column | Type | Description |
|--------|------|-------------|
| `death_age` | float32 | Age at death from competing-risk mortality |
| `t_observed1`, `t_observed2` | float32 | Observed age-at-onset after age and death censoring |
| `age_censored1`, `age_censored2` | bool | True if onset falls outside the generation's observation window |
| `death_censored1`, `death_censored2` | bool | True if onset occurs after death |
| `affected1`, `affected2` | bool | True if the individual is observed as affected (not age- or death-censored) |

### trait.simple_ltm.parquet

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

### Ascertained outputs

`pedigree.parquet`, `trait.parquet`, and `trait.simple_ltm.parquet` are the canonical post-ascertainment outputs that both simACE-stats and fitACE consume. Under non-trivial ascertainment (`dropout_rate > 0` or `N_sample > 0`), these files contain a subset of the full simulated population:

- `pedigree.parquet` is the **ancestor closure** of the sampled IDs within the post-dropout pedigree, with dangling `mother` / `father` / `twin` references rewritten to −1.
- `trait.parquet` and `trait.simple_ltm.parquet` share an identical `id` column (the sampled set), restricted to the trailing `G_pheno` generations.

The pre-ascertainment outputs (`trait.raw.parquet`, `trait.full.parquet`, `trait.simple_ltm.full.parquet`) are Snakemake `temp()` files — auto-deleted once ascertainment has consumed them.

`plotting_sample.parquet` is a *further* downsampled parquet produced inside the Stats stage for scatter/histogram plots; it shares the `trait.parquet` schema.

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

### report.yaml

Curated per-replicate **scientific report** written by
`workflow/scripts/simace/analyze.py`, which calls
`simace.analysis.analyze.run_analysis` (ADR 0008, `schema.version: 2`). It holds
scalars, small categorical tables, and by-generation summaries only — dense plot
arrays live in the companion `plot_payload.yaml`. Values are organized by what
they *mean*, and tagged with one of four population **scopes**: `recorded_pedigree`
(full pre-ascertainment pedigree), `phenotyped_population` (full pre-ascertainment
phenotyped rows), `analysis_sample` (ascertained `trait.parquet`), and
`analysis_pedigree` (ancestor closure supporting the sample).

| Group | Description |
|---------|-------------|
| `schema` | `{name: simace_report, version: 2}` |
| `replicate` | `folder`, `scenario`, `rep`, `seed` |
| `inputs` | Full resolved `parameters`, plus curated `trait_model` and `ascertainment` summaries |
| `scopes` | Per-scope source file, `n_individuals`, `n_generations` (+ `ancestor_closure_ratio` for the analysis pedigree) |
| `quality_checks` | Normalized pass/fail rows `{id, scope, severity, status, observed, expected, tolerance, message}` plus a `summary` |
| `truth` | Generated/realized ground truth on `recorded_pedigree`: realized A/C/E variances + liability h² (per trait, with `realized_by_generation`), `cross_trait` correlations, `family_structure` (twin rate, half-sibs, consanguinity, offspring distribution), `assortative_mating` |
| `observed` | Descriptive summaries bucketed by scope: a first-class `ascertainment` block (per-trait before/after affected fractions + enrichment, retained fraction; raw scope sizes and closure live once in `scopes`), `phenotyped_population` prevalence, and the re-bucketed `analysis_sample` / `analysis_pedigree` stats |
| `estimators` | Heritability split into `observed_scale` (binary-affected) and `liability_scale` (twin/sibling/parent-offspring) |

### plot_payload.yaml

Durable companion (`schema.version: 1`) holding the dense incidence and
censoring-window arrays (`ages`, `observed_values`, `aj_values`, …) needed to
render plots reproducibly. Organized by scope to mirror `observed`. It is part of
plot regeneration, not the scientific report; where a scalar is duplicated,
`report.yaml` is canonical.

---

## report_summary.tsv

Aggregated metrics across all scenarios and replicates within a folder. Written by `simace/analysis/gather.py`. One row per replicate.

Columns are emitted from `REPORT_SUMMARY_REGISTRY` (`simace/analysis/report_schema.py`); paths below are relative to the `report.yaml` root.

| Column | Source |
|--------|--------|
| `folder`, `scenario`, `rep` | Parsed from file path |
| `N`, `G_ped`, `G_sim`, `A1`..`E2`, `rA`, `rC`, `p_mztwin`, `mating_lambda`, `seed` | `inputs.parameters` |
| `quality_passed`, `checks_failed`, `quality_n_warn` | `quality_checks.summary` (`passed` / `n_failed` / `n_warn`) |
| `recorded_pedigree_n`, `phenotyped_population_n`, `analysis_sample_n`, `analysis_pedigree_n`, `ancestor_closure_ratio` | `scopes.*` |
| `retained_fraction`, `trait{1,2}_affected_before`, `trait{1,2}_affected_after` | `observed.ascertainment.*` |
| `observed_twin_rate`, `expected_twin_rate` | `truth.recorded_pedigree.family_structure.twin_rate.*` |
| `variance_A1`..`E2` | `truth.recorded_pedigree.traits.trait{1,2}.realized.var_*` |
| `observed_rA`, `observed_rC`, `observed_rE` | `truth.recorded_pedigree.cross_trait.*` |
| `mz_twin_*_corr`, `dz_sibling_*_corr`, `falconer_h2_trait{1,2}`, `parent_offspring_*` | `estimators.heritability.liability_scale.trait{1,2}.*` |
| `half_sib_prop_observed`, `offspring_with_half_sib_observed`, `half_sib_*_corr`, `half_sib_shared_C{1,2}` | `truth.recorded_pedigree.family_structure.half_sibs.*` |
| `mate_corr_liability{1,2}` | `truth.recorded_pedigree.assortative_mating.*` |
| `mother_mean_offspring`, `father_mean_offspring` | `truth.recorded_pedigree.family_structure.offspring_distribution.*.mean` |
| `n_half_sib_matings`, `n_full_sib_matings`, `missing_gp_links`, `gp_reconciled` | `truth.recorded_pedigree.family_structure.consanguineous_matings.*` |
| `simulate_seconds`, `simulate_max_rss_mb` | Parsed from benchmark TSV |

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
- `benchmarks/{folder}/{scenario}/rep{rep}/analyze.tsv`

Per-scenario benchmarks:

- `benchmarks/{folder}/{scenario}/plot_phenotype.tsv`
- `benchmarks/{folder}/{scenario}/assemble_atlas.tsv`

Per-folder benchmarks:

- `benchmarks/{folder}/gather_report_summary.tsv`
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
| `heritability.by_generation.{ext}` | Additive genetic and common environment variance proportions by generation |
| `heritability.by_sex.by_generation.{ext}` | Liability-scale heritability by sex and generation |
| `liability_violin.phenotype.{ext}` | Liability violin plots by affection status |
| `liability_violin.phenotype.by_generation.{ext}` | Liability violins by generation and affection status |
| `liability_violin.phenotype.by_sex.by_generation.{ext}` | Liability violins by sex, generation, and affection status |
| `mortality.{ext}` | Mortality rates and death-age distributions |
| `age_at_onset_death.{ext}` | Age-at-onset distributions |
| `cumulative_incidence.phenotype.{ext}` | Cumulative incidence curves by trait |
| `censoring.{ext}` | Censoring window visualization |
| `cumulative_incidence.by_sex.{ext}` | Cumulative incidence by sex |
| `cumulative_incidence.by_sex.by_generation.{ext}` | Cumulative incidence by sex and generation |
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

An atlas combines all plots for a scope into a single document with figure
captions. The default is a self-contained HTML atlas (embedded plots, native
overview + Table 1, inline-SVG equations); a multi-page PDF atlas is an
on-demand export (ADR 0010 — build with `snakemake .../atlas.pdf`):

| File | Contents |
|------|----------|
| `results/{folder}/{scenario}/plots/atlas.html` | All phenotype figures for one scenario (default) |
| `results/{folder}/{scenario}/plots/atlas.pdf` | Same scenario figures, on-demand PDF export |
| `results/{folder}/plots/atlas.html` | All cross-scenario validation figures for one folder (default) |
| `results/{folder}/plots/atlas.pdf` | Same validation figures, on-demand PDF export |
| `results/{folder}/{scenario}/rep{rep}/epimight/plots/atlas.html` | EPIMIGHT CIF, heritability, and genetic correlation figures (default; `.pdf` on demand) |

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
| `plots/atlas.html` | HTML | Self-contained atlas of all EPIMIGHT figures (default; `.pdf` on demand) |
