# ADR 0007: Merged Analyze Report (`report.yaml`)

## Status

Accepted. Supersedes the "No Validate-stage change" non-goal of ADR 0003 and the
"No schema or artifact-path change" non-goal of ADR 0006. **Superseded in part by
ADR 0008**, which replaces the `validation` + six-stats-group shape with the
curated `quality_checks` / `truth` / `observed` / `estimators` groups and splits
dense plot arrays into `plot_payload.yaml`.

## Context

The Analyze stage (ADR 0006) runs Validate and Stats in one process but wrote
two per-replicate artifacts: `validation.yaml` (ground-truth checks on the full
pre-ascertainment pedigree) and `stats_report.yaml` (the six-group descriptive
report on the post-ascertainment subsample), plus `plotting_sample.parquet`.
Keeping two artifacts for one stage is the last seam preventing Analyze from
being a single cohesive unit.

Mapping every consumer across both repos showed the cross-repo cost is small:
**fitACE never reads fields** from either YAML — `fitACE/workflow/common.py`
only declares them as Snakemake target *paths*. Every field-level consumer is
inside the simACE repo (plotting, gather, one docs example script). ADR 0003
already established the migration pattern for this exact kind of interface
change: a hard cut (no dual-emit), simACE merges first, then fitACE updates its
path, with a brief expected CI gap.

## Decision

The Analyze stage writes a single per-replicate **`report.yaml`** that replaces
both `validation.yaml` and `stats_report.yaml`. `plotting_sample.parquet` is
unchanged.

`report.yaml` keeps the six stats groups at the top level — `metadata`,
`incidence`, `censoring`, `pedigree`, `correlations`, `heritability` (ADR 0003) —
and folds the validation report in under a new top-level **`validation`** group,
holding its existing sub-structure verbatim (`structural`, `twins`, `half_sibs`,
`statistical`, `heritability`, `population`, `per_generation`,
`assortative_mating`, `consanguineous_matings`, `summary`,
`family_size_distribution`, `parameters`). Nesting under `validation` avoids the
`heritability` key collision between the two halves and makes consumer migration
a mechanical one-level descent.

`run_analysis` assembles `report = {**stats_report, "validation":
validation_report}` and writes it once. The Phase-1 `del df_full; gc.collect()`
is retained: the validation report is a small summary dict, so holding it across
Phase 2 to merge does not change peak memory (`max(validate, stats)`).

This is a hard cut: simACE stops writing both old files. The debug-only CLIs
`simace-validate` and `simace-phenotype-stats` still write their own-named files
for ad-hoc use; those are not pipeline artifacts.

## Consequences

- One artifact per replicate for the whole Analyze stage. `validate.done` and
  `stats.done` both resolve to the single `analyze` rule producing `report.yaml`.
- simACE consumers are repointed: `gather.extract_metrics` descends into
  `report["validation"]`; plotting loaders read `report.yaml` (the stats view
  ignores the extra `validation` key; per-generation / validation params come
  from `report["validation"]`).
- fitACE updates `workflow/common.py` to target `report.yaml`. A brief CI gap
  between the simACE and fitACE merges is expected (per ADR 0003).
- Existing result directories with `validation.yaml` / `stats_report.yaml` must
  be regenerated before plotting or gathering with the current code.

## Non-goals

- **No cross-stage graph/pair sharing.** The two halves still build their own
  graphs over their own (disjoint) scopes; ADR 0006's deferral stands.
- **No stats-group schema change.** The six groups keep their existing shapes
  and contents; validation keeps its sub-structure. Only the file boundary moves.
- **`effective_size.yaml` is untouched** (separate opt-in rule).
