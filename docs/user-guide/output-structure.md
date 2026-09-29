# Output structure

Every scenario writes under `results/{folder}/{scenario}/`. The placeholders
`{folder}`, `{scenario}`, and `{rep}` take the folder name, the scenario
name, and the replicate number, starting at 1.

```
results/{folder}/{scenario}/
├── rep1/
│   ├── params.yaml
│   ├── pedigree.parquet
│   ├── cohort.parquet
│   ├── phenotyped_population.yaml
│   ├── report.yaml
│   ├── plot_payload.yaml
│   ├── plotting_sample.parquet
│   ├── timing.tsv
│   └── run.yaml
├── rep2/
├── rep3/
└── plots/
    ├── *.png
    ├── atlas.html
    ├── atlas.pdf
    ├── plots.yaml
    └── timing.tsv
results/{folder}/
├── report_summary.tsv
└── plots/
    ├── *.png
    ├── atlas.html
    └── atlas.pdf
```

A replicate directory also holds per-method subdirectories and files that
fitACE writes, such as `epimight/`. This page lists the simACE outputs only.

## Per-replicate files

| File | Written by | Description |
|---|---|---|
| `params.yaml` | `simace run` (`simace/simulation/emit_params.py`) | The resolved simulation parameters for this replicate |
| `pedigree.parquet` | `simace/simulation/simulate.py` | The recorded pedigree after burn-in, with every recorded link. Ascertainment never rewrites it |
| `cohort.parquet` | `simace/cli/cohort_stage.py` | One row per member of the analysis pedigree, with censored outcomes set for the analysis sample and null for the other members. This and `pedigree.parquet` are what fitACE reads. See [Ascertainment, Who is in `cohort.parquet`](ascertainment.md#who-is-in-cohortparquet) |
| `phenotyped_population.yaml` | `simace/cli/cohort_stage.py` | Size and prevalence of the whole phenotyped population before ascertainment. See [phenotyped_population.yaml](#phenotyped_populationyaml) |
| `report.yaml` | `simace/analysis/analyze.py` | The per-replicate report. See [report.yaml](#reportyaml) |
| `plot_payload.yaml` | `simace/analysis/analyze.py` | Dense arrays for the incidence and censoring plots |
| `plotting_sample.parquet` | `simace/analysis/analyze.py` | A downsampled join of traits and pedigree for scatter plots |
| `timing.tsv` | `simace run` | One row per stage: `stage`, `wall_s`, `max_rss_mb` (the stage process's peak resident memory), `exit_code` |
| `run.yaml` | `simace run` | Written after every stage succeeds. Records the scenario, replicate, seed, parameters, stages, and results `layout` (2) the replicate was computed with, plus the simace version and git ref (`source`) that built it. A rerun skips the replicate only when this matches and every other file above exists. See [Running the pipeline](running-the-pipeline.md#rerun-and-resume) |

Every stage writes each output to a temporary `<name>.<random>.tmp` beside it
and renames it into place when it finishes, so a file under its final name is
always complete, even when two commands write the same output at once.

`results/{folder}/{scenario}/.run.lock` holds the pid of the last
`simace run` of the scenario. A running `simace run` keeps it locked so a
second run of the same scenario refuses to start.

This file set is results layout 2
([ADR 0021](../adr/0021-two-canonical-replicate-parquets.md)). A replicate
written before it holds `pedigree.full.parquet`, a selected `pedigree.parquet`,
and `trait*.parquet` files instead, reads as stale, and is recomputed only
with `--force` ([Running the pipeline](running-the-pipeline.md#rerun-and-resume)).

`cohort.parquet` holds outcomes only. To get generation, sex, family links,
variance components, or liabilities, rebuild the analysis frames from the two
files and join on `id`:

```python
from simace.core.cohort import read_cohort, read_pedigree, selected_views
from simace.core.trait_schema import hydrate_trait

rep = "results/test/small_test/rep1"
views = selected_views(read_pedigree(f"{rep}/pedigree.parquet"), read_cohort(f"{rep}/cohort.parquet"))
sample = hydrate_trait(views.trait, views.pedigree, kind="censored")
```

`views.trait` is the analysis sample and `views.pedigree` the analysis
pedigree. `read_pedigree` and `read_cohort` refuse a file without the
Parquet key-value metadata `simace_layout=2`, so a selected
`pedigree.parquet` from an older replicate is never read as the recorded
pedigree.

## Per-scenario and per-folder files

| File | Description |
|---|---|
| `results/{folder}/{scenario}/plots/*.png` | Scenario plots. [Interpreting results](interpreting-results.md) lists them |
| `results/{folder}/{scenario}/plots/atlas.html` | All scenario plots in one HTML file, with captions, a parameter page, and Table 1 |
| `results/{folder}/{scenario}/plots/atlas.pdf` | The same atlas as a PDF. Built on demand ([ADR 0010](../adr/0010-html-primary-atlas-rendering.md)) |
| `results/{folder}/{scenario}/plots/plots.yaml` | Which replicates' `run.yaml` files the plots and atlas were built from, and the atlas files written. `simace ls` reads it to report the plots as current, stale, or absent |
| `results/{folder}/{scenario}/plots/timing.tsv` | Wall time and peak memory of the `plot`, `atlas`, and (with `--format pdf`) `atlas-pdf` stages |
| `results/{folder}/report_summary.tsv` | One row per replicate across every scenario in the folder, written by `simace gather`. See [report_summary.tsv](#report_summarytsv) |
| `results/{folder}/plots/*.png` | Validation plots comparing scenarios |
| `results/{folder}/plots/atlas.html`, `atlas.pdf` | The validation plots as an atlas |
| `logs/{folder}/{scenario}/rep{rep}/{stage}.log` | One log per stage: `simulate`, `cohort`, `analyze`. `cohort.log` gives the wall time and peak memory of each of its phases: phenotype, censor, ascertain |
| `logs/{folder}/{scenario}/{plot,atlas,atlas-pdf}.log` | The scenario plot and atlas logs |

Image files use the extension set by `plot_format`, `png` by default.

## Parquet columns

### pedigree.parquet

Column types below are what `results/test/small_test/rep1/pedigree.parquet`
holds at this commit. This command prints the schema of any parquet file in
the tree:

```bash
pixi run python -c "import pyarrow.parquet as pq, sys; print(pq.read_schema(sys.argv[1]))" results/test/small_test/rep1/pedigree.parquet
```

| Column | Type | Description |
|---|---|---|
| `id` | int32 | Individual identifier |
| `sex` | int8 | 0 is female, 1 is male |
| `mother`, `father` | int32 | Parent identifiers. -1 when the parent is outside the recorded pedigree |
| `twin` | int32 | Identifier of the monozygotic twin. -1 when there is none |
| `generation` | int32 | 0 is the oldest recorded generation |
| `household_id` | int32 | Group that shares the common environment. Assigned by mother |
| `A1`, `C1`, `E1`, `A2`, `C2`, `E2` | float32 | Variance components for trait 1 and trait 2 |
| `liability1`, `liability2` | float64 | `A + C + E` for each trait |

### cohort.parquet

One row per member of the analysis pedigree, in `pedigree.parquet` row
order. On rows of the analysis sample, `affected1` is not null and so is
every column below except `t1` and `t2`. On the other rows, the ancestors the
sample needs, every column except `id` is null.

| Column | Type | Description |
|---|---|---|
| `id` | int32 | Individual identifier |
| `t1`, `t2` | float32 | Onset age before censoring. May be null on a sample row, meaning no onset ([ADR 0019](../adr/0019-null-raw-onset-censoring-semantics.md)) |
| `death_age` | float32 | Age at death from the competing-risk mortality |
| `t_observed1`, `t_observed2` | float32 | Onset age after age-window and death censoring |
| `age_censored1`, `age_censored2` | bool | True when onset falls outside the generation's observation window |
| `death_censored1`, `death_censored2` | bool | True when death precedes onset |
| `affected1`, `affected2` | bool | True when the individual is neither age-censored nor death-censored |

## YAML files

### params.yaml

A flat mapping of the parameters this replicate ran with. Keys at this commit:
`seed`, `rep`, `N`, `G_ped`, `G_sim`, `A1`, `C1`, `E1`, `A2`, `C2`, `E2`,
`rA`, `rC`, `rE`, `mating_model`, `mating_lambda`, `p_mztwin`, `assort1`,
`assort2`, `max_degree`, `skip_ne_coancestry`, and `simace_version`.
`max_degree` and `skip_ne_coancestry` record the analysis controls that apply
to the replicate. `seed` is the base seed plus `rep - 1`. To list the keys,
run:

```bash
grep -o '^[a-zA-Z_0-9]*' results/test/small_test/rep1/params.yaml
```

### report.yaml

`simace/analysis/analyze.py` writes the report through `run_analysis`
([ADR 0008](../adr/0008-curated-analyze-report.md)). `schema.version` is 2.
The report holds scalars, small tables, and per-generation summaries. Dense
arrays go to `plot_payload.yaml`.

Every value is tagged with one of four population scopes.

| Scope | Population |
|---|---|
| `recorded_pedigree` | Every individual in `pedigree.parquet` |
| `phenotyped_population` | Every individual in the trailing `G_pheno` generations, after censoring and before ascertainment. Summarized in `phenotyped_population.yaml` |
| `analysis_sample` | Every row of `cohort.parquet` with `affected1` not null |
| `analysis_pedigree` | Every row of `cohort.parquet`, with its pedigree columns from `pedigree.parquet` |

| Top-level key | Contents |
|---|---|
| `schema` | `name: simace_report` and `version: 2` |
| `replicate` | `folder`, `scenario`, `rep`, `seed` |
| `inputs` | The resolved `parameters`, plus `trait_model` and `ascertainment` summaries |
| `scopes` | For each scope, the source file, `n_individuals`, and `n_generations`. The analysis pedigree adds `ancestor_closure_ratio` |
| `quality_checks` | One row per check with `id`, `scope`, `severity`, `status`, `observed`, `expected`, `tolerance`, `message`, plus a `summary` |
| `truth` | Realized values on `recorded_pedigree`: variance components and liability heritability per trait, with `realized_by_generation`, plus `cross_trait`, `family_structure`, and `assortative_mating` |
| `observed` | Descriptive statistics per scope. `ascertainment` holds affected fractions before and after sampling, enrichment, and the retained fraction |
| `estimators` | Heritability estimates, split into `observed_scale` from affected status and `liability_scale` from twin, sibling, and parent-offspring pairs |

### phenotyped_population.yaml

The `cohort` stage writes this summary while it holds the censored outcomes
of the whole phenotyped population, which are not stored per individual. The
analyze stage reads it for the `phenotyped_population` scope and the
before-and-after comparison in `observed.ascertainment`.

| Key | Contents |
|---|---|
| `n_individuals` | Number of phenotyped individuals |
| `n_generations` | Number of phenotyped generations, `G_pheno` |
| `prevalence` | Affected fraction per trait (`trait1`, `trait2`), and the same per generation under `by_generation`, keyed by `generation` |

### plot_payload.yaml

`schema.version` is 1. The file holds the incidence and censoring arrays such as
`ages`, `observed_values`, and `aj_values`, grouped by scope in the same
layout as `observed`. Where a scalar appears in both files, `report.yaml` is
canonical.

## report_summary.tsv

`simace/analysis/gather.py` writes one row per replicate for every scenario in
the folder. The columns come from `REPORT_SUMMARY_REGISTRY` in
`simace/analysis/report_schema.py`. Each entry names a column and the path
inside `report.yaml` that fills it. `folder`, `scenario`, and `rep` come from
the file path. `simulate_seconds` and `simulate_max_rss_mb` come only from the
`simulate` row of the replicate's `timing.tsv`; they do not describe the whole
pipeline. Read the registry for the full list.

## Stage timing

`simace run` runs each stage as its own process and appends one row to the
replicate's `timing.tsv` when the stage exits: `stage`, `wall_s` (elapsed
seconds), `max_rss_mb` (the process's peak resident memory in MiB, from
`wait4`), and `exit_code`. The scenario's `plots/timing.tsv` holds the same
columns for the `plot` and `atlas` stages. A recomputed replicate starts a
fresh `timing.tsv`.

The reproducible benchmark driver copies these files into an immutable run
directory and adds process-tree memory measurements. See
[Benchmark pipeline performance](benchmarking.md).

## TSV exports

`simace parquet-to-tsv` writes a `.tsv.gz` file next to a parquet file, with
four decimal places by default. [Running the pipeline, Convert parquet to
TSV](running-the-pipeline.md#convert-parquet-to-tsv) has the commands.

## EPIMIGHT outputs

fitACE_epimight writes under `results/{folder}/{scenario}/rep{rep}/epimight/`.
Its [README](https://github.com/rwaples/fitACE_epimight/blob/master/README.md)
documents the files.
