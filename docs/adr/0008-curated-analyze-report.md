# ADR 0008: Curated Analyze Report (`report.yaml` v2) + `plot_payload.yaml`

## Status

Accepted. Supersedes the report *shape* established in ADR 0007 (the `validation`
group + six stats groups). The single-artifact, hard-cut migration discipline of
ADR 0003 / 0007 is retained.

## Context

ADR 0007 merged the Analyze outputs into one `report.yaml`, but its shape was
dictated by the producer seams — a `validation` group beside the six stats
groups (`metadata`, `incidence`, …). That shape mixed three concerns: pass/fail
quality checks, generated ground-truth quantities, observed sample summaries —
and carried dense plot-only arrays (200-point incidence curves, 300-point
censoring-window curves) that dwarf the scientific content and are never read by
a human. It also could not express *which population* a number describes.

`CONTEXT.md` locks the vocabulary: a **per-replicate scientific report** (not a
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
