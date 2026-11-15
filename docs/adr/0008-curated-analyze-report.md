# ADR 0008: Curated Analyze Report (`report.yaml` v2) + `plot_payload.yaml`

## Status

Accepted. This ADR is the surviving record of the Analyze stage's report
lineage; the three earlier decisions it built on have been retired and their
durable parts are restated here:

- **Grouped stats report.** The Stats stage replaced a flat
  `phenotype_stats.yaml` with a grouped report plus `plotting_sample.parquet`
  (the plotting-only downsample, unchanged since). Report grouping follows
  domain meaning rather than module location.
- **One Analyze job.** Validate and Stats run sequentially in a single process
  and one Snakemake rule (`analyze`, `analyze.smk`), with each phase's large
  frame explicitly freed before the next so peak memory is the max of the
  phases, not their sum. Validate (full recorded pedigree) and Stats
  (post-ascertainment sample) build their own relationship graphs over their
  own scopes; cross-stage graph sharing was deferred and remains so. The
  `validate.done` / `stats.done` targets both resolve to this rule. The debug
  CLIs `simace-validate` and `simace-phenotype-stats` are retained and write
  their own-named files; they are not pipeline artifacts.
- **One report artifact, migrated by hard cut.** `validation.yaml` and
  `stats_report.yaml` were merged into a single `report.yaml`. fitACE never
  reads fields from these files, only declares them as Snakemake target paths,
  so each interface change is a hard cut: no dual emit, no old-schema reader,
  simACE merges first, fitACE repoints its paths, and a brief CI gap between
  the two merges is expected. Existing result directories are regenerated.

**Status note (2026-09-29).** [ADR 0021](0021-two-canonical-replicate-parquets.md)
removed `trait.full.parquet` from finished output. The Analyze stage's second
input is now `phenotyped_population.yaml`, the phenotyped-population summary
the `cohort` stage writes while it holds the censored phenotyped rows. The
three memory phases and the report contract below are unchanged; phase 2
reads that file instead of loading and hydrating the trait rows.

**Status note (2026-10-17).** [ADR 0028](0028-analyze-validate-boundary.md)
reuses the name `validation.yaml`, retired here, for `simace validate`'s
per-rep on-demand results, with a required `schema` block that tells it apart
from the old files. `quality_checks` keeps only quality checks: checks whose
failure means the rep is wrong. Analyze's check families move to
`simace/analysis/quality/`.

**Status note (2026-11-15).** #89 types both ends of the report. The
`report.yaml` and `plot_payload.yaml` contract below is unchanged, byte for
byte; what changes is how code reaches it:

- **Producer.** The Stats phase returns a `SampleStats`
  (`simace/analysis/sample_stats.py`) already in scope shape: an
  `AnalysisSampleStats` for `observed.analysis_sample`, an
  `AnalysisPedigreeStats` for `observed.analysis_pedigree`, the two scope
  sizes, and the observed-scale heritability estimators. Its fields are the
  report keys in report order, so the six intermediate stats groups and the
  re-bucketing between them are gone. Each quality-check family returns a
  `FamilyResult`: the checks it asserts, as `QualityCheck` records, and what
  it measures, as a typed measurements record. `quality_checks` rows come
  from the checks; `truth` and `estimators` from the measurements. A value
  with no closed-form verdict is a measurement, never an informational check.
- **No `df_ped is None` mode.** The Stats phase requires the analysis
  pedigree. The `simace stats` debug command, the one caller that omitted
  it, is removed, and with it the retained debug CLIs this ADR's first
  paragraph lists (the validate one went with ADR 0028).
- **Plotting boundary.** The plotting code parses `report.yaml` and
  `plot_payload.yaml` into a typed `ReportView`
  (`simace/plotting/stats_report.py`) that carries the producer's
  `SampleStats` records, replacing the flat-dict adapter that kept plot
  helpers unchanged. Absence is declared in the types: a field is optional
  only when a correct run can omit it (the censoring tables without
  per-generation censoring windows, and dense arrays when no payload is
  read), and its figure draws a placeholder. Every other field is required.
  `partition_dense` drops empty tables, so an always-computed table that is
  absent reads back empty, not missing.

## Context

The merged `report.yaml` had its shape dictated by the producer seams — a `validation` group beside the six stats
groups (`metadata`, `incidence`, …). That shape mixed three concerns: pass/fail
quality checks, generated ground-truth quantities, observed sample summaries —
and carried dense plot-only arrays (200-point incidence curves, 300-point
censoring-window curves) that dwarf the scientific content and are never read by
a human. It also could not express *which population* a number describes.

`GLOSSARY.md` locks the vocabulary: a **per-replicate scientific report** (not a
plot cache, not a cross-replicate aggregate), a durable **plot payload** for
dense arrays, and four explicit **scopes** — `recorded_pedigree`,
`phenotyped_population`, `analysis_sample`, `analysis_pedigree`.

## Decision

`report.yaml` becomes a curated scientific report with top-level groups
`schema`, `replicate`, `inputs`, `scopes`, `quality_checks`, `truth`,
`observed`, `estimators` (`schema.version: 2`). Dense plot arrays move to a
durable companion **`plot_payload.yaml`** (`schema.version: 1`), organized by
scope to mirror `observed`. `report.yaml` holds only scalars, small categorical
tables, and by-generation summaries; a contract check
(`report_schema.assert_report_contract`) rejects any dense-array key.

- **`quality_checks`** — every validation check normalized to a uniform row
  `{id, scope, severity, status, observed, expected, tolerance, message}` plus a
  summary. All current checks run on the recorded pedigree.
- **`truth`** — generated/realized ground truth on `recorded_pedigree`: realized
  A/C/E variances and liability h² (per trait, plus by-generation), cross-trait
  correlations, family structure (twin rate, half-sibs, consanguinity, offspring
  distribution), assortative mating.
- **`observed`** — descriptive summaries bucketed by scope. A first-class
  `observed.ascertainment` block compares `phenotyped_population` to
  `analysis_sample` (affected fractions before/after, retained fraction,
  ancestor-closure ratio). The remaining six-stats-group content is re-bucketed
  faithfully under `analysis_sample` / `analysis_pedigree`.
- **`estimators`** — heritability split into `observed_scale` (binary-affected
  estimators) and `liability_scale` (twin/sibling/parent-offspring,
  relationship-derived).

To quantify ascertainment distortion, the pre-ascertainment phenotyped rows
(`trait.full.parquet`) become **durable** (previously `temp()`) and a third
Analyze input. `run_analysis` runs three memory phases — recorded pedigree,
phenotyped population, analysis sample — each freeing its frame before the next,
so peak memory stays `max` of the phases.

The folder-level aggregate `validation_summary.tsv` is renamed
**`report_summary.tsv`**; its registry (`REPORT_SUMMARY_REGISTRY` in
`report_schema.py`) maps wide columns into the `truth` / `estimators` / `scopes`
/ `observed.ascertainment` paths, and gains scope-size and ascertainment columns.

The plotting code consumes the report through an adapter
(`plotting_report_view`) that rebuilds the flat plotting view from the
scope-organized report plus the plot payload, so plot helpers are unchanged.

## Consequences

- `report.yaml` is human-readable and scope-explicit; `plot_payload.yaml` is the
  durable, reproducible plot-array companion.
- The summary registry resolves into `truth` / `estimators` / `scopes` /
  `observed`; `gather` reads the report root (no `validation` descent).
- fitACE lists `trait.full.parquet` + `plot_payload.yaml` as durable sim outputs;
  it still references all of these as Snakemake paths only, never fields.
- Existing result directories must be regenerated; this is a hard cut.
- Debug CLIs (`simace-validate`, `simace-phenotype-stats`) keep their raw
  own-named outputs and are not redesigned in this pass.

## Non-goals

- **No metric pruning.** v2 re-homes current values rather than dropping content.
- **No benchmarks in `report.yaml`.** Timing/memory stay in benchmark TSVs and
  may be joined by summary/gather code after the fact.
- **No debug-CLI redesign**, no cross-stage relationship-graph reuse, no change
  to fitACE estimator outputs.
